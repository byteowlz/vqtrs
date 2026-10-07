//! Audio front end, reproducing the reference bit for bit: WAV decoding as
//! libsndfile (`soundfile`) does it, and transformers'
//! `Gemma4AudioFeatureExtractor` (NumPy): padding, semicausal framing, the
//! periodic Hann window, `np.fft.rfft`, `NumPy`'s SIMD complex `abs`, the HTK
//! mel filter bank applied by a `float64` GEMM, `log` and the frame mask.

use candle_core::{Device, Tensor};

use super::pocketfft::{Rfft, cabs};

/// Feature-extractor settings (`processor_config.json`, `feature_extractor`).
#[derive(Debug, Clone, Copy)]
pub struct MelSettings {
    pub sampling_rate: usize,
    pub frame_length: usize,
    pub hop_length: usize,
    pub fft_length: usize,
    pub feature_size: usize,
    pub min_frequency: f64,
    pub max_frequency: f64,
    pub mel_floor: f64,
    /// `pad_to_multiple_of` (samples).
    pub pad_multiple: usize,
    /// `max_length` (samples); longer audio is truncated.
    pub max_samples: usize,
}

impl Default for MelSettings {
    fn default() -> Self {
        Self {
            sampling_rate: 16_000,
            frame_length: 320,
            hop_length: 160,
            fft_length: 512,
            feature_size: 128,
            min_frequency: 0.0,
            max_frequency: 8000.0,
            mel_floor: 1e-3,
            pad_multiple: 128,
            max_samples: 480_000,
        }
    }
}

/// Decode a WAV file to mono `f32` samples at its native rate, converting
/// like libsndfile (`soundfile.read(dtype="float32")`): `int16` and `int32`
/// samples scale by `2^-15` / `2^-31`, 24-bit samples are left-justified to
/// 32 bits first, unsigned 8-bit samples are centred. Stereo audio is
/// averaged (`np.mean(axis=1)` in `float32`).
pub fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, usize), String> {
    let le16 = |at: usize| {
        bytes
            .get(at..at + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
    };
    let le32 = |at: usize| {
        bytes
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    if bytes.get(..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
        return Err("not a RIFF/WAVE file".into());
    }
    let riff_end = usize::try_from(le32(4).ok_or("short RIFF header")?)
        .map_err(|e| e.to_string())?
        .checked_add(8)
        .ok_or("RIFF size overflow")?;
    if riff_end > bytes.len() || riff_end < 12 {
        return Err("truncated RIFF/WAVE file".into());
    }
    let (mut format, mut channels, mut rate, mut bits, mut align, mut data) =
        (0_u16, 0_usize, 0_usize, 0_usize, 0_usize, None);
    let mut pos: usize = 12;
    while pos < riff_end {
        let body = pos.checked_add(8).ok_or("chunk size overflow")?;
        if body > riff_end {
            return Err("truncated WAV chunk header".into());
        }
        let id = bytes.get(pos..pos + 4).ok_or("short chunk header")?;
        let len = le32(pos + 4).ok_or("short chunk header")? as usize;
        let end = body.checked_add(len).ok_or("chunk size overflow")?;
        if end > riff_end {
            return Err("truncated WAV chunk".into());
        }
        match id {
            b"fmt " => {
                if len < 16 {
                    return Err("short fmt chunk".into());
                }
                format = le16(body).ok_or("short fmt chunk")?;
                channels = usize::from(le16(body + 2).ok_or("short fmt chunk")?);
                rate = le32(body + 4).ok_or("short fmt chunk")? as usize;
                align = usize::from(le16(body + 12).ok_or("short fmt chunk")?);
                bits = usize::from(le16(body + 14).ok_or("short fmt chunk")?);
                if format == 0xFFFE {
                    if len < 40 {
                        return Err("short extensible fmt chunk".into());
                    }
                    let valid_bits = usize::from(le16(body + 18).ok_or("short fmt chunk")?);
                    if valid_bits != bits {
                        return Err("WAV valid bits differ from container width".into());
                    }
                    // PCM and IEEE-float GUIDs differ only in the first word.
                    if bytes.get(body + 26..body + 40)
                        != Some(&[0, 0, 0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113])
                    {
                        return Err("unsupported WAV subformat GUID".into());
                    }
                    format = le16(body + 24).ok_or("short extensible fmt chunk")?;
                }
            }
            b"data" => data = bytes.get(body..end),
            _ => {}
        }
        pos = end.checked_add(len & 1).ok_or("chunk size overflow")?;
        if pos > riff_end {
            return Err("missing WAV chunk padding".into());
        }
    }
    let data = data.ok_or("WAV without a data chunk")?;
    if channels == 0 || rate == 0 || bits == 0 || !bits.is_multiple_of(8) {
        return Err("invalid WAV fmt chunk".into());
    }
    if channels > 2 {
        return Err("only mono and stereo WAV are supported".into());
    }
    let expected_align = channels
        .checked_mul(bits / 8)
        .ok_or("WAV block size overflow")?;
    if align != expected_align || !data.len().is_multiple_of(align) {
        return Err("invalid WAV block alignment or incomplete sample frame".into());
    }
    let samples = decode_samples(data, format, bits)?;
    let mono = if channels == 1 {
        samples
    } else {
        samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().fold(0.0_f32, |acc, &v| acc + v) / channels as f32)
            .collect()
    };
    if mono.iter().any(|v| !v.is_finite()) {
        return Err("WAV contains non-finite samples".into());
    }
    Ok((mono, rate))
}

fn decode_samples(data: &[u8], format: u16, bits: usize) -> Result<Vec<f32>, String> {
    Ok(match (format, bits) {
        (1, 8) => data
            .iter()
            .map(|&b| (i32::from(b) - 128) as f32 * (1.0 / 128.0))
            .collect(),
        (1, 16) => data
            .chunks_exact(2)
            .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) * (1.0 / 32_768.0))
            .collect(),
        (1, 24) => data
            .chunks_exact(3)
            .map(|b| i32::from_le_bytes([0, b[0], b[1], b[2]]) as f32 * (1.0 / 2_147_483_648.0))
            .collect(),
        (1, 32) => data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 * (1.0 / 2_147_483_648.0))
            .collect(),
        (3, 32) => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        (3, 64) => data
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
            .collect(),
        other => return Err(format!("unsupported WAV sample format {other:?}")),
    })
}

/// `np.hanning(n + 1)[:-1]` (periodic Hann) narrowed to `f32`.
#[expect(
    clippy::suboptimal_flops,
    reason = "NumPy rounds each window calculation separately"
)]
pub fn hann_window(n: usize) -> Vec<f32> {
    let m = (n + 1) as f64;
    (0..n)
        .map(|i| {
            let k = (1.0 - m) + 2.0 * i as f64; // arange(1 - M, M, 2)
            (0.5 + 0.5 * (std::f64::consts::PI * k / (m - 1.0)).cos()) as f32
        })
        .collect()
}

/// `np.linspace(start, stop, num)` (`endpoint=True`).
#[expect(
    clippy::suboptimal_flops,
    reason = "NumPy linspace multiplies then adds with separate rounding"
)]
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let div = (num - 1) as f64;
    let increment = (stop - start) / div;
    let mut y: Vec<f64> = (0..num).map(|i| i as f64 * increment + start).collect();
    if let Some(last) = y.last_mut() {
        *last = stop;
    }
    y
}

/// transformers' `mel_filter_bank(..., norm=None, mel_scale="htk")`,
/// row-major `(num_frequency_bins, num_mel_filters)`.
pub fn mel_filters(s: &MelSettings) -> Vec<f64> {
    let to_mel = |f: f64| 2595.0 * (1.0 + (f / 700.0)).log10();
    // LLVM otherwise specializes constant-base pow(10, x), differing from
    // NumPy's libm pow at centre 115. Keep the reference operation.
    let to_hz = |m: f64| 700.0 * (std::hint::black_box(10.0_f64).powf(m / 2595.0) - 1.0);
    let bins = s.fft_length / 2 + 1;
    let n_mel = s.feature_size;
    let mel_freqs = linspace(to_mel(s.min_frequency), to_mel(s.max_frequency), n_mel + 2);
    let filter_freqs: Vec<f64> = mel_freqs.iter().map(|&m| to_hz(m)).collect();
    let fft_freqs = linspace(0.0, (s.sampling_rate / 2) as f64, bins);
    let diff: Vec<f64> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();
    let mut out = vec![0.0_f64; bins * n_mel];
    for (b, &ff) in fft_freqs.iter().enumerate() {
        for m in 0..n_mel {
            let down = -(filter_freqs[m] - ff) / diff[m];
            let up = (filter_freqs[m + 2] - ff) / diff[m + 1];
            // np.maximum(np.zeros(1), np.minimum(down, up))
            let v = down.min(up);
            out[b * n_mel + m] = if v > 0.0 { v } else { 0.0 };
        }
    }
    out
}

/// Output of the feature extractor for one clip.
#[derive(Debug, Clone)]
pub struct MelFeatures {
    /// `(frames, feature_size)` log-mel features, masked frames zeroed.
    pub features: Vec<f32>,
    /// One flag per frame: every sample of its window is real audio.
    pub mask: Vec<bool>,
}

/// The front-end tables, computed once.
pub struct MelFrontEnd {
    s: MelSettings,
    window: Vec<f32>,
    filters: Tensor,
    fft: Rfft,
}

impl MelFrontEnd {
    pub fn new(s: MelSettings, dev: &Device) -> candle_core::Result<Self> {
        if s.fft_length != 512
            || s.frame_length == 0
            || s.frame_length > s.fft_length
            || s.hop_length == 0
            || s.feature_size == 0
            || s.pad_multiple == 0
            || s.max_samples == 0
        {
            return Err(candle_core::Error::Msg(
                "invalid audio feature-extractor settings".into(),
            ));
        }
        let filters =
            Tensor::from_vec(mel_filters(&s), (s.fft_length / 2 + 1, s.feature_size), dev)?;
        Ok(Self {
            window: hann_window(s.frame_length),
            filters,
            fft: Rfft::new(s.fft_length),
            s,
        })
    }

    pub const fn settings(&self) -> &MelSettings {
        &self.s
    }

    /// `Gemma4AudioFeatureExtractor.__call__` for one waveform.
    pub fn features(&self, samples: &[f32]) -> candle_core::Result<MelFeatures> {
        let s = &self.s;
        // pad(padding="longest", truncation, pad_to_multiple_of): zeros after.
        let real = samples.len().min(s.max_samples);
        let padded = real.div_ceil(s.pad_multiple) * s.pad_multiple;
        // Semicausal framing: frame_length / 2 zeros in front.
        let pad_left = s.frame_length / 2;
        let mut wave = vec![0.0_f32; pad_left + padded];
        wave[pad_left..pad_left + real].copy_from_slice(&samples[..real]);
        let unfold = s.frame_length + 1;
        let frames = if wave.len() >= unfold {
            (wave.len() - unfold) / s.hop_length + 1
        } else {
            0
        };
        let bins = s.fft_length / 2 + 1;

        let mut magnitude = Vec::with_capacity(frames * bins);
        for f in 0..frames {
            let start = f * s.hop_length;
            let frame: Vec<f32> = wave[start..start + s.frame_length]
                .iter()
                .zip(&self.window)
                .map(|(&x, &w)| x * w)
                .collect();
            magnitude.extend(
                self.fft
                    .rfft(&frame)
                    .into_iter()
                    .map(|(re, im)| f64::from(cabs(re, im))),
            );
        }
        // np.matmul(float32 magnitudes, float64 filters): float64 GEMM.
        let mel = Tensor::from_vec(magnitude, (frames, bins), self.filters.device())?
            .matmul(&self.filters)?
            .flatten_all()?
            .to_vec1::<f64>()?;
        let mask: Vec<bool> = (0..frames)
            .map(|f| {
                let end = f * s.hop_length + unfold - 1; // in the left-padded signal
                end >= pad_left && end - pad_left < real
            })
            .collect();
        let features = mel
            .chunks_exact(s.feature_size)
            .zip(&mask)
            .flat_map(|(row, &valid)| {
                // log(mel + floor).astype(float32) * mask (a multiply: -0.0 for
                // masked negative features, as NumPy).
                let keep = if valid { 1.0_f32 } else { 0.0 };
                row.iter()
                    .map(move |&v| ((v + s.mel_floor).ln() as f32) * keep)
            })
            .collect();
        Ok(MelFeatures { features, mask })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read<T, const N: usize>(dir: &std::path::Path, name: &str, f: fn([u8; N]) -> T) -> Vec<T> {
        std::fs::read(dir.join(name))
            .unwrap()
            .chunks_exact(N)
            .map(|c| f(c.try_into().unwrap()))
            .collect()
    }

    fn wav(format: u16, channels: u16, bits: u16, data: &[u8]) -> Vec<u8> {
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(36 + data.len() as u32 + (data.len() as u32 & 1)).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16_u32.to_le_bytes());
        out.extend_from_slice(&format.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&16_000_u32.to_le_bytes());
        let align = channels * (bits / 8);
        out.extend_from_slice(&(16_000 * u32::from(align)).to_le_bytes());
        out.extend_from_slice(&align.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        if !data.len().is_multiple_of(2) {
            out.push(0);
        }
        out
    }

    #[test]
    fn wav_normalizes_pcm_and_stereo() {
        let data: Vec<u8> = [i16::MIN, 0, i16::MAX, 16_384]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();
        let (mono, rate) = decode_wav(&wav(1, 1, 16, &data)).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(mono, [-1.0, 0.0, 32_767.0 / 32_768.0, 0.5]);
        assert_eq!(
            decode_wav(&wav(1, 2, 16, &data)).unwrap().0,
            [-0.5, f32::midpoint(32_767.0 / 32_768.0, 0.5)]
        );
        assert_eq!(
            decode_wav(&wav(1, 1, 8, &[0, 128, 255])).unwrap().0,
            [-1.0, 0.0, 127.0 / 128.0]
        );
        assert_eq!(
            decode_wav(&wav(1, 1, 24, &[0, 0, 128, 0, 0, 64]))
                .unwrap()
                .0,
            [-1.0, 0.5]
        );
        let data: Vec<u8> = [i32::MIN, 1_073_741_824]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect();
        assert_eq!(decode_wav(&wav(1, 1, 32, &data)).unwrap().0, [-1.0, 0.5]);
        let data: Vec<u8> = [-0.25_f32, 0.75]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert_eq!(decode_wav(&wav(3, 1, 32, &data)).unwrap().0, [-0.25, 0.75]);
    }

    #[test]
    fn wav_rejects_malformed_chunks_and_nonfinite_samples() {
        let valid = wav(1, 1, 16, &[0, 0]);
        for len in 0..valid.len() {
            assert!(decode_wav(&valid[..len]).is_err(), "prefix {len}");
        }
        let mut short_fmt = valid;
        short_fmt[16..20].copy_from_slice(&4_u32.to_le_bytes());
        assert!(decode_wav(&short_fmt).is_err());
        assert!(decode_wav(&wav(1, 1, 16, &[0])).is_err());
        assert!(decode_wav(&wav(1, 0, 16, &[])).is_err());
        assert!(decode_wav(&wav(1, 3, 16, &[0; 6])).is_err());
        assert!(decode_wav(&wav(3, 1, 32, &f32::NAN.to_le_bytes())).is_err());
    }

    #[test]
    fn empty_and_short_audio_have_no_valid_frames() {
        let frontend = MelFrontEnd::new(MelSettings::default(), &Device::Cpu).unwrap();
        assert!(frontend.features(&[]).unwrap().features.is_empty());
        let short = frontend.features(&[0.0; 100]).unwrap();
        assert!(short.mask.iter().all(|&v| !v));
        assert!(
            MelFrontEnd::new(
                MelSettings {
                    hop_length: 0,
                    ..MelSettings::default()
                },
                &Device::Cpu
            )
            .is_err()
        );
    }

    /// Compare WAV decoding, features and masks for every audio parity case.
    #[test]
    #[ignore = "needs generated audio fixtures and Python reference dumps"]
    fn features_match_reference() {
        let dir = std::path::PathBuf::from(
            std::env::var("EG2_AUDIO_REF_DIR").expect("EG2_AUDIO_REF_DIR"),
        );
        let cases_file =
            std::path::PathBuf::from(std::env::var("EG2_AUDIO_CASES").expect("EG2_AUDIO_CASES"));
        let cases: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&cases_file).unwrap()).unwrap();
        let base = cases_file.parent().unwrap();
        let frontend = MelFrontEnd::new(MelSettings::default(), &Device::Cpu).unwrap();
        for case in cases.as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let wav = std::fs::read(base.join(case["path"].as_str().unwrap())).unwrap();
            let (wave, rate) = decode_wav(&wav).unwrap();
            assert_eq!(rate, frontend.settings().sampling_rate);
            let features = frontend.features(&wave).unwrap();
            let reference: Vec<f32> = read(
                &dir,
                &format!("{name}.input.input_features.bin"),
                f32::from_le_bytes,
            );
            let mask: Vec<u8> =
                std::fs::read(dir.join(format!("{name}.input.input_features_mask.bin"))).unwrap();
            assert_eq!(features.features.len(), reference.len());
            assert_eq!(
                features.mask,
                mask.iter().map(|&v| v != 0).collect::<Vec<_>>()
            );
            let same = features
                .features
                .iter()
                .zip(&reference)
                .filter(|(a, b)| a.to_bits() == b.to_bits())
                .count();
            let max_abs = features
                .features
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            println!(
                "{name}: features identical {same}/{}, max_abs={max_abs:e}",
                reference.len()
            );
            assert_eq!(same, reference.len(), "{name}");
        }
    }

    /// Window, filter bank and the float64 mel GEMM against `NumPy` dumps.
    #[test]
    #[ignore = "needs NumPy reference dumps"]
    fn tables_match_numpy() {
        let dir = std::path::PathBuf::from(std::env::var("EG2_MEL_DIR").expect("EG2_MEL_DIR"));
        let s = MelSettings::default();
        let window: Vec<f32> = read(&dir, "window_f32.bin", f32::from_le_bytes);
        let ours = hann_window(s.frame_length);
        let w_same = ours
            .iter()
            .zip(&window)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        let filters: Vec<f64> = read(&dir, "filters_f64.bin", f64::from_le_bytes);
        let ours_f = mel_filters(&s);
        let f_same = ours_f
            .iter()
            .zip(&filters)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        println!(
            "window identical {w_same}/{}, filters identical {f_same}/{}",
            window.len(),
            filters.len()
        );

        let mag: Vec<f32> = read(&dir, "fft_abs_f32.bin", f32::from_le_bytes);
        let reference: Vec<f64> = read(&dir, "mel_f64.bin", f64::from_le_bytes);
        let dev = Device::Cpu;
        let mag = Tensor::from_vec(
            mag.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            (50, 257),
            &dev,
        )
        .unwrap();
        let fb = Tensor::from_vec(filters, (257, 128), &dev).unwrap();
        let mel = mag
            .matmul(&fb)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let m_same = mel
            .iter()
            .zip(&reference)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        println!("mel GEMM identical {m_same}/{}", reference.len());
        assert_eq!(w_same, window.len());
        assert_eq!(f_same, ours_f.len());
        assert_eq!(m_same, reference.len());
    }
}
