//! Default EmbeddingGemma 2 video preprocessing: 1 fps, at most 32 uniformly
//! sampled frames, checkpoint-configured soft-token budget (140), and no timestamps.
//!
//! Containers are an optional `ffmpeg`/`ffprobe` command boundary (no shell).
//! Only MOV/MP4 and Matroska/WebM demuxers and the local-file input protocol
//! are enabled. A private temporary file is removed on every return path.
//! Limits: 16 mebibytes encoded, 2,097,152 pixels/frame (accepts 1920x1080),
//! 3,600 decoded source frames,
//! 120 fps, 120 seconds, and 30 seconds wall time across all subprocesses.
//! Stdout, decoder single allocations (64 mebibytes), and decoder pixels are bounded.
//! Selected raw RGB stdout is at most 192 mebibytes (32 full-resolution frames);
//! prepared pixel/position buffers are below 119 mebibytes with the checkpoint's
//! 140-token budget, independent of resolution (up to 950 at the supported 1120).
//! This is NOT an OS sandbox or a hard total-RSS limit; use an up-to-date trusted
//! `FFmpeg` installation. Frame counting decodes the entire accepted source;
//! selecting 32 frames does not make decoding a long video cheap.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::image::{self, ImageSettings};

/// Interleaved RGB8 frame; dimensions and buffer length are validated before use.
#[derive(Debug, Clone)]
pub struct RgbFrame {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Exactly `height * width * 3` bytes, RGB channel order.
    pub data: Vec<u8>,
}

/// One frame's padded patches and `(x, y)` positions.
#[derive(Debug, Clone)]
pub struct FramePatches {
    /// Row-major `(max_patches, patch_dim)` floats, zero-padded (1260×768 at 140 tokens).
    pub pixel_values: Vec<f32>,
    /// Row-major `(max_patches, 2)` positions, `[-1, -1]` for padding.
    pub positions: Vec<[i64; 2]>,
}

/// Maximum encoded container size, independent of the HTTP body limit.
pub const MAX_CONTAINER_BYTES: usize = 16 * 1024 * 1024;
const MAX_PIXELS: usize = 2_097_152;
const MAX_SOURCE_FRAMES: usize = 3600;
const MAX_FPS: f64 = 120.0;
const MAX_SECONDS: f64 = 120.0;
const MAX_SAMPLED_FRAMES: usize = 32;
const PROCESS_BUDGET: Duration = Duration::from_secs(30);
const FORMATS: &str = "matroska,webm,mov,mp4,m4a,3gp,3g2,mj2";

/// Optional timing for already-decoded frames. Missing rate or duration skips
/// rate-based sampling, exactly as the HF processor does for decoded arrays.
#[derive(Debug, Clone, Copy)]
pub struct VideoMetadata {
    /// Number of source frames (must match a supplied decoded frame slice).
    pub total_num_frames: usize,
    /// Native frame rate, if known; finite, positive, and at most 120.
    pub fps: Option<f64>,
    /// Duration in seconds, if known; finite, positive, and at most 120.
    pub duration: Option<f64>,
}

impl VideoMetadata {
    fn validate(self) -> Result<(), String> {
        if self.total_num_frames == 0 || self.total_num_frames > MAX_SOURCE_FRAMES {
            return Err("video must contain between 1 and 3,600 source frames".into());
        }
        for (value, max, name) in [
            (self.fps, MAX_FPS, "fps"),
            (self.duration, MAX_SECONDS, "duration"),
        ] {
            if value.is_some_and(|v| !v.is_finite() || v <= 0.0 || v > max) {
                return Err(format!(
                    "video {name} must be finite, positive and <= {max}"
                ));
            }
        }
        Ok(())
    }
}

/// Default processor output, in temporal order.
///
/// Each frame has `settings.max_patches()` padded patches and positions.
/// Actual soft-token counts can be below the configured budget because
/// aspect-ratio-preserving rounding leaves unused patch capacity.
#[derive(Debug, Clone)]
pub struct PreparedVideo {
    /// Per-frame patches and positions (frame-major).
    pub frames: Vec<FramePatches>,
    /// Actual number of soft tokens per frame, shared by every frame.
    pub num_soft_tokens: usize,
    /// Original zero-based frame indices; may repeat for sub-1-fps sources.
    pub sampled_indices: Vec<usize>,
}

/// Mirror HF's 1-fps sampling followed by `np.linspace(..., dtype=int)` uniform
/// overflow sampling. No guessed timing is applied to decoded arrays.
///
/// # Errors
/// Rejects empty/oversized frame counts and nonfinite/out-of-bounds timing.
pub fn sample_frames(metadata: VideoMetadata) -> Result<Vec<usize>, String> {
    metadata.validate()?;
    let timing = metadata.fps.zip(metadata.duration);
    let count = timing.map_or(metadata.total_num_frames, |(_, duration)| {
        (duration as usize).max(1)
    });
    let selected = count.min(MAX_SAMPLED_FRAMES);
    let step = (count - 1) as f64 / (selected - 1).max(1) as f64;
    Ok((0..selected)
        .map(|i| {
            let position = if selected == count {
                i
            } else if i + 1 == selected {
                count - 1
            } else {
                (i as f64 * step) as usize
            };
            timing.map_or(position, |(fps, _)| {
                ((position as f64 * fps) as usize).min(metadata.total_num_frames - 1)
            })
        })
        .collect())
}

fn validate_frame(frame: &RgbFrame) -> Result<(), String> {
    let pixels = frame.width.checked_mul(frame.height);
    if frame.width == 0
        || frame.height == 0
        || pixels.is_none_or(|n| n > MAX_PIXELS || n.checked_mul(3) != Some(frame.data.len()))
    {
        return Err(
            "invalid RGB frame: dimensions/buffer must match and contain <= 2,097,152 pixels"
                .into(),
        );
    }
    Ok(())
}

fn validate_settings(settings: &ImageSettings) -> Result<(), String> {
    if settings.patch_size != 16
        || settings.pooling_kernel_size != 3
        || !matches!(settings.max_soft_tokens, 70 | 140 | 280 | 560 | 1120)
        || !settings.rescale_factor.is_finite()
        || settings.rescale_factor <= 0.0
        || settings.rescale_factor > 1.0
    {
        return Err("invalid video settings: expected 16-pixel patches, pooling 3, supported soft-token budget and finite rescale factor".into());
    }
    Ok(())
}

fn prepare_selected(
    frames: &[RgbFrame],
    sampled_indices: Vec<usize>,
    settings: &ImageSettings,
) -> Result<PreparedVideo, String> {
    let patches = frames
        .iter()
        .map(|frame| {
            image::preprocess(
                &image::Rgb {
                    width: frame.width,
                    height: frame.height,
                    data: frame.data.clone(),
                },
                settings,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let num_soft_tokens = patches
        .first()
        .ok_or_else(|| "video has no sampled frames".to_owned())?
        .num_soft_tokens;
    let frames = patches
        .into_iter()
        .map(|p| FramePatches {
            pixel_values: p.pixel_values,
            positions: p.positions,
        })
        .collect();
    Ok(PreparedVideo {
        frames,
        num_soft_tokens,
        sampled_indices,
    })
}

/// Preprocess trusted decoded RGB arrays, with optional caller-supplied timing.
///
/// The resize is torchvision's uint8 antialiased bicubic path, NOT an assumed
/// PIL equivalent. All source frames must have the same dimensions. Settings
/// come from the checkpoint's `video_processor`, not the Python class defaults.
///
/// # Errors
/// Rejects invalid settings/buffers, differing dimensions, invalid timing, or limits.
pub fn preprocess_decoded(
    frames: &[RgbFrame],
    metadata: Option<VideoMetadata>,
    settings: &ImageSettings,
) -> Result<PreparedVideo, String> {
    validate_settings(settings)?;
    let metadata = metadata.unwrap_or(VideoMetadata {
        total_num_frames: frames.len(),
        fps: None,
        duration: None,
    });
    let indices = sample_frames(metadata)?;
    if metadata.total_num_frames != frames.len() {
        return Err("video metadata frame count does not match decoded frames".into());
    }
    let first = &frames[0];
    for frame in frames {
        validate_frame(frame)?;
        if (frame.width, frame.height) != (first.width, first.height) {
            return Err("video frames must share dimensions".into());
        }
    }
    let selected: Vec<_> = indices.iter().map(|&i| frames[i].clone()).collect();
    prepare_selected(&selected, indices, settings)
}

#[derive(Debug)]
struct LocalContainer(PathBuf);

impl LocalContainer {
    fn create(bytes: &[u8]) -> Result<Self, String> {
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        for _ in 0..32 {
            let path = std::env::temp_dir().join(format!(
                "vqtrs-video-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            let mut builder = std::fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt as _;
                builder.mode(0o700);
            }
            match builder.create(&path) {
                Ok(()) => {
                    let guard = Self(path);
                    let mut file = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(guard.path())
                        .map_err(|e| format!("creating temporary video: {e}"))?;
                    file.write_all(bytes)
                        .map_err(|e| format!("writing temporary video: {e}"))?;
                    return Ok(guard);
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(format!("creating private video directory: {e}")),
            }
        }
        Err("cannot create a unique private video directory".into())
    }

    fn path(&self) -> PathBuf {
        self.0.join("container")
    }
}

impl Drop for LocalContainer {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.path());
        let _ = std::fs::remove_dir(&self.0);
    }
}

/// Bound stdout and wall time while draining the pipe concurrently. Stderr is
/// discarded rather than leaking container contents or filling a second pipe.
fn run_bounded(command: &mut Command, cap: usize, started: Instant) -> Result<Vec<u8>, String> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("video decoding requires ffmpeg and ffprobe on PATH: {e}"))?;
    let pipe = child
        .stdout
        .take()
        .ok_or_else(|| "missing video subprocess stdout".to_owned())?;
    let reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        pipe.take((cap + 1) as u64).read_to_end(&mut bytes)?;
        Ok::<_, std::io::Error>(bytes)
    });
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if started.elapsed() < PROCESS_BUDGET => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Ok(None) => break Err("video subprocess exceeded 30-second runtime budget".to_owned()),
            Err(e) => break Err(format!("waiting for video subprocess: {e}")),
        }
    };
    if status.is_err() {
        let _ = child.kill();
    }
    let _ = child.wait();
    let bytes = reader
        .join()
        .map_err(|_| "video stdout reader failed".to_owned())?
        .map_err(|e| format!("reading video stdout: {e}"))?;
    let status = status?;
    if bytes.len() > cap {
        return Err("video subprocess exceeded stdout limit".into());
    }
    if !status.success() {
        return Err("invalid/unsupported video or video decoder failed".into());
    }
    Ok(bytes)
}

#[derive(Deserialize)]
struct Probe {
    streams: Vec<Stream>,
    #[serde(default)]
    format: Format,
}

#[derive(Deserialize)]
struct Stream {
    width: usize,
    height: usize,
    avg_frame_rate: String,
    duration: Option<String>,
    nb_frames: Option<String>,
}

#[derive(Default, Deserialize)]
struct Format {
    duration: Option<String>,
}

#[derive(Deserialize)]
struct FrameList {
    frames: Vec<FrameSize>,
}

#[derive(Deserialize)]
struct FrameSize {
    width: usize,
    height: usize,
}

fn probe_command(local: &LocalContainer) -> Command {
    let mut command = Command::new("ffprobe");
    command.args([
        "-v",
        "error",
        "-max_alloc",
        "67108864",
        "-threads",
        "1",
        "-max_pixels",
        "2097152",
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        FORMATS,
        "-select_streams",
        "v:0",
    ]);
    command.arg("-i").arg(local.path());
    command
}

fn parse_probe(probe: Probe) -> Result<(Stream, f64), String> {
    let stream = probe
        .streams
        .into_iter()
        .next()
        .ok_or_else(|| "video has no video stream".to_owned())?;
    let pixels = stream.width.checked_mul(stream.height);
    if stream.width == 0 || stream.height == 0 || pixels.is_none_or(|p| p > MAX_PIXELS) {
        return Err("video dimensions exceed 2,097,152 pixels".into());
    }
    let (numerator, denominator) = stream
        .avg_frame_rate
        .split_once('/')
        .ok_or_else(|| "invalid video frame rate".to_owned())?;
    let fps = numerator.parse::<f64>().map_err(|e| e.to_string())?
        / denominator.parse::<f64>().map_err(|e| e.to_string())?;
    let duration = stream
        .duration
        .as_ref()
        .or(probe.format.duration.as_ref())
        .ok_or_else(|| "video must declare its duration".to_owned())?
        .parse::<f64>()
        .map_err(|e| format!("invalid video duration: {e}"))?;
    let count = stream
        .nb_frames
        .as_deref()
        .filter(|s| *s != "N/A")
        .map_or(Ok(1), |s| s.parse::<usize>().map_err(|e| e.to_string()))?;
    VideoMetadata {
        total_num_frames: count,
        fps: Some(fps),
        duration: Some(duration),
    }
    .validate()?;
    if duration * fps > MAX_SOURCE_FRAMES as f64 + 1.0 {
        return Err("video duration/rate exceeds 3,600 source frames".into());
    }
    Ok((stream, fps))
}

/// Decode an inline container and preprocess default sampled RGB frames.
///
/// Frame counting validates actual dimensions and count, not just header claims.
/// Sampling duration is actual frame count / average fps, as in HF's `PyAV` path.
///
/// # Errors
/// Rejects invalid settings/containers, unsafe metadata, changing dimensions,
/// missing decoder tools, subprocess failures, stdout overflow, or timeouts.
pub fn decode_container(bytes: &[u8], settings: &ImageSettings) -> Result<PreparedVideo, String> {
    validate_settings(settings)?;
    let (frames, indices) = decode_container_rgb(bytes)?;
    prepare_selected(&frames, indices, settings)
}

fn decode_container_rgb(bytes: &[u8]) -> Result<(Vec<RgbFrame>, Vec<usize>), String> {
    if bytes.is_empty() || bytes.len() > MAX_CONTAINER_BYTES {
        return Err("video container must contain between 1 byte and 16 MiB".into());
    }
    let started = Instant::now();
    let local = LocalContainer::create(bytes)?;
    let mut command = probe_command(&local);
    command.args([
        "-show_entries",
        "stream=width,height,avg_frame_rate,duration,nb_frames:format=duration",
        "-of",
        "json",
    ]);
    let probe: Probe = serde_json::from_slice(&run_bounded(&mut command, 16 * 1024, started)?)
        .map_err(|e| format!("invalid video metadata: {e}"))?;
    let (stream, fps) = parse_probe(probe)?;
    let mut command = probe_command(&local);
    command.args([
        "-show_frames",
        "-show_entries",
        "frame=width,height",
        "-of",
        "json",
    ]);
    let list: FrameList = serde_json::from_slice(&run_bounded(&mut command, 512 * 1024, started)?)
        .map_err(|e| format!("invalid decoded video metadata: {e}"))?;
    let metadata = VideoMetadata {
        total_num_frames: list.frames.len(),
        fps: Some(fps),
        duration: Some(list.frames.len() as f64 / fps),
    };
    let indices = sample_frames(metadata)?;
    if list
        .frames
        .iter()
        .any(|f| (f.width, f.height) != (stream.width, stream.height))
    {
        return Err("video changes frame dimensions".into());
    }
    let mut unique = indices.clone();
    unique.dedup();
    let expression = unique
        .iter()
        .map(|i| format!("eq(n\\,{i})"))
        .collect::<Vec<_>>()
        .join("+");
    let mut command = Command::new("ffmpeg");
    command.args([
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-max_alloc",
        "67108864",
        "-threads",
        "1",
        "-noautorotate",
        "-max_pixels",
        "2097152",
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        FORMATS,
    ]);
    command.arg("-i").arg(local.path());
    command.args([
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-filter_threads",
        "1",
        "-vf",
    ]);
    command.arg(format!("select={expression}"));
    command.args(["-fps_mode", "passthrough", "-frames:v"]);
    command.arg(unique.len().to_string());
    command.args([
        "-threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]);
    let frame_bytes = stream.width * stream.height * 3;
    let raw = run_bounded(&mut command, frame_bytes * unique.len(), started)?;
    if raw.len() != frame_bytes * unique.len() {
        return Err("video decoder returned an incomplete RGB frame sequence".into());
    }
    let selected: Vec<_> = indices
        .iter()
        .map(|i| {
            let offset = unique
                .binary_search(i)
                .map_err(|e| format!("missing sampled frame: {e}"))?;
            Ok(RgbFrame {
                width: stream.width,
                height: stream.height,
                data: raw[offset * frame_bytes..(offset + 1) * frame_bytes].to_vec(),
            })
        })
        .collect::<Result<_, String>>()?;
    Ok((selected, indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint_settings() -> ImageSettings {
        ImageSettings {
            max_soft_tokens: 140,
            ..ImageSettings::default()
        }
    }

    fn metadata(total: usize, fps: Option<f64>, duration: Option<f64>) -> VideoMetadata {
        VideoMetadata {
            total_num_frames: total,
            fps,
            duration,
        }
    }

    #[test]
    fn default_selection_and_missing_timing() {
        assert_eq!(
            sample_frames(metadata(90, Some(30.0), Some(3.0))).unwrap(),
            [0, 30, 60]
        );
        assert_eq!(
            sample_frames(metadata(10, None, None)).unwrap(),
            (0..10).collect::<Vec<_>>()
        );
        assert_eq!(
            sample_frames(metadata(100, Some(25.0), None)).unwrap(),
            sample_frames(metadata(100, None, None)).unwrap()
        );
        assert_eq!(
            sample_frames(metadata(2, Some(0.5), Some(4.0))).unwrap(),
            [0, 0, 1, 1]
        );
        assert_eq!(
            sample_frames(metadata(1, Some(30.0), Some(0.02))).unwrap(),
            [0]
        );
        let indices = sample_frames(metadata(1800, Some(15.0), Some(120.0))).unwrap();
        assert_eq!(indices.len(), 32);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[31], 1785);
        assert_eq!(indices[1], 45);
    }

    #[test]
    fn rejects_unbounded_metadata() {
        for total in [0, 3601, usize::MAX] {
            assert!(sample_frames(metadata(total, None, None)).is_err());
        }
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, 121.0, f64::MAX] {
            assert!(sample_frames(metadata(1, Some(invalid), Some(1.0))).is_err());
            assert!(sample_frames(metadata(1, Some(1.0), Some(invalid))).is_err());
        }
    }

    #[test]
    fn validates_decoded_arrays_and_patch_contract() {
        let mut frame = RgbFrame {
            width: 48,
            height: 48,
            data: vec![255; 48 * 48 * 3],
        };
        let settings = checkpoint_settings();
        let result = preprocess_decoded(&[frame.clone()], None, &settings).unwrap();
        assert_eq!(result.num_soft_tokens, 121);
        assert_eq!(result.frames[0].pixel_values.len(), 1260 * 768);
        assert_eq!(result.frames[0].positions.len(), 1260);
        assert_eq!(result.frames[0].positions[1089], [-1, -1]);
        assert_eq!(
            result.frames[0].pixel_values[0].to_bits(),
            1.0_f32.to_bits()
        );
        assert!(preprocess_decoded(&[], None, &settings).is_err());
        assert!(
            preprocess_decoded(&[frame.clone()], Some(metadata(2, None, None)), &settings).is_err()
        );
        let smaller = ImageSettings {
            max_soft_tokens: 70,
            ..settings
        };
        let small = preprocess_decoded(&[frame.clone()], None, &smaller).unwrap();
        assert_eq!(small.num_soft_tokens, 64);
        assert_eq!(small.frames[0].positions.len(), 630);
        frame.data.pop();
        assert!(preprocess_decoded(&[frame], None, &settings).is_err());
        let frame = RgbFrame {
            width: usize::MAX,
            height: 2,
            data: Vec::new(),
        };
        assert!(preprocess_decoded(&[frame], None, &settings).is_err());
        let hd = RgbFrame {
            width: 1920,
            height: 1080,
            data: vec![0; 1920 * 1080 * 3],
        };
        assert!(validate_frame(&hd).is_ok());
        let over = RgbFrame {
            width: 2048,
            height: 2048,
            data: Vec::new(),
        };
        assert!(validate_frame(&over).is_err());
    }

    #[test]
    fn rejects_invalid_and_oversized_containers() {
        let settings = checkpoint_settings();
        assert!(decode_container(&[], &settings).is_err());
        assert!(decode_container(&vec![0; MAX_CONTAINER_BYTES + 1], &settings).is_err());
        assert!(decode_container(b"https://example.invalid/video.mp4", &settings).is_err());
        assert!(decode_container(b"#EXTM3U\nfile:///etc/passwd", &settings).is_err());
    }

    #[test]
    fn rejects_invalid_settings_before_decoding() {
        for settings in [
            ImageSettings {
                patch_size: 0,
                ..checkpoint_settings()
            },
            ImageSettings {
                pooling_kernel_size: 0,
                ..checkpoint_settings()
            },
            ImageSettings {
                max_soft_tokens: usize::MAX,
                ..checkpoint_settings()
            },
            ImageSettings {
                rescale_factor: f64::NAN,
                ..checkpoint_settings()
            },
        ] {
            assert!(
                decode_container(b"invalid", &settings)
                    .unwrap_err()
                    .contains("settings")
            );
            assert!(
                preprocess_decoded(&[], None, &settings)
                    .unwrap_err()
                    .contains("settings")
            );
        }
    }

    #[derive(Deserialize)]
    struct ReferenceManifest {
        video_settings: ReferenceSettings,
        cases: Vec<ReferenceCase>,
        #[serde(default)]
        rejected_containers: Vec<String>,
    }

    #[derive(Deserialize)]
    struct ReferenceSettings {
        patch_size: usize,
        max_soft_tokens: usize,
        pooling_kernel_size: usize,
        rescale_factor: f64,
    }

    impl ReferenceSettings {
        const fn settings(&self) -> ImageSettings {
            ImageSettings {
                patch_size: self.patch_size,
                max_soft_tokens: self.max_soft_tokens,
                pooling_kernel_size: self.pooling_kernel_size,
                rescale_factor: self.rescale_factor,
            }
        }
    }

    #[derive(Deserialize)]
    struct ReferenceCase {
        name: String,
        width: usize,
        height: usize,
        count: usize,
        fps: Option<f64>,
        duration: Option<f64>,
        frames_file: String,
        container_file: Option<String>,
        sampled_indices: Vec<usize>,
        num_soft_tokens: usize,
        pixels_file: String,
        positions_file: String,
    }

    fn assert_reference(dir: &std::path::Path, case: &ReferenceCase, prepared: &PreparedVideo) {
        assert_eq!(
            prepared.sampled_indices, case.sampled_indices,
            "{} sampling",
            case.name
        );
        assert_eq!(
            prepared.num_soft_tokens, case.num_soft_tokens,
            "{} soft tokens",
            case.name
        );
        let pixels: Vec<u8> = prepared
            .frames
            .iter()
            .flat_map(|f| f.pixel_values.iter().flat_map(|v| v.to_le_bytes()))
            .collect();
        let positions: Vec<u8> = prepared
            .frames
            .iter()
            .flat_map(|f| {
                f.positions
                    .iter()
                    .flat_map(|p| p.iter().flat_map(|v| v.to_le_bytes()))
            })
            .collect();
        let expected_pixels = std::fs::read(dir.join(&case.pixels_file)).unwrap();
        // Summarize mismatches rather than dumping multi-MiB byte arrays.
        assert_eq!(
            pixels.len(),
            expected_pixels.len(),
            "{} pixels length",
            case.name
        );
        assert_eq!(
            pixels
                .iter()
                .zip(&expected_pixels)
                .filter(|(a, b)| a != b)
                .count(),
            0,
            "{} pixel bytes",
            case.name
        );
        let expected_positions = std::fs::read(dir.join(&case.positions_file)).unwrap();
        assert_eq!(
            positions.len(),
            expected_positions.len(),
            "{} positions length",
            case.name
        );
        assert!(positions == expected_positions, "{} positions", case.name);
        println!(
            "{}: {} frames x {} soft tokens, BIT-IDENTICAL pixels and positions",
            case.name,
            prepared.frames.len(),
            prepared.num_soft_tokens
        );
    }

    #[test]
    #[ignore = "needs generated pinned HF fixtures and optional ffmpeg/ffprobe"]
    fn reference_parity() {
        let dir = PathBuf::from(std::env::var("EG2_VIDEO_REFERENCE").expect("EG2_VIDEO_REFERENCE"));
        let manifest: ReferenceManifest =
            serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        let settings = manifest.video_settings.settings();
        assert_eq!(
            settings.max_soft_tokens, 140,
            "canonical checkpoint video budget"
        );
        for file in &manifest.rejected_containers {
            let bytes = std::fs::read(dir.join(file)).unwrap();
            let error = decode_container(&bytes, &settings).unwrap_err();
            println!("{file}: rejected safely ({error})");
        }
        for case in manifest.cases {
            let raw = std::fs::read(dir.join(&case.frames_file)).unwrap();
            let size = case.width * case.height * 3;
            assert_eq!(raw.len(), size * case.count);
            let frames: Vec<_> = raw
                .chunks_exact(size)
                .map(|data| RgbFrame {
                    width: case.width,
                    height: case.height,
                    data: data.to_vec(),
                })
                .collect();
            let metadata = VideoMetadata {
                total_num_frames: case.count,
                fps: case.fps,
                duration: case.duration,
            };
            let prepared = preprocess_decoded(&frames, Some(metadata), &settings).unwrap();
            assert_reference(&dir, &case, &prepared);
            if let Some(file) = &case.container_file {
                let bytes = std::fs::read(dir.join(file)).unwrap();
                let (decoded, indices) = decode_container_rgb(&bytes).unwrap();
                assert_eq!(indices, case.sampled_indices);
                for (frame, &index) in decoded.iter().zip(&indices) {
                    let expected = &frames[index];
                    assert_eq!(
                        (frame.width, frame.height),
                        (expected.width, expected.height)
                    );
                    assert_eq!(frame.data.len(), expected.data.len());
                    assert_eq!(
                        frame
                            .data
                            .iter()
                            .zip(&expected.data)
                            .filter(|(a, b)| a != b)
                            .count(),
                        0,
                        "{} frame {index}: FFmpeg RGB vs independent PyAV RGB bytes",
                        case.name
                    );
                }
                println!(
                    "{}: raw FFmpeg RGB bytes BIT-IDENTICAL to independent PyAV decode",
                    case.name
                );
                let prepared = decode_container(&bytes, &settings).unwrap();
                assert_reference(&dir, &case, &prepared);
            }
        }
    }

    #[test]
    #[ignore = "requires optional ffmpeg command"]
    fn subprocess_output_and_runtime_are_bounded() {
        let mut command = Command::new("ffmpeg");
        command.arg("-version");
        assert!(
            run_bounded(&mut command, 1, Instant::now())
                .unwrap_err()
                .contains("stdout limit")
        );
        let mut command = Command::new("ffmpeg");
        command.args([
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=s=16x16",
            "-f",
            "null",
            "-",
        ]);
        let started = Instant::now().checked_sub(PROCESS_BUDGET).unwrap();
        assert!(
            run_bounded(&mut command, 4096, started)
                .unwrap_err()
                .contains("runtime budget")
        );
    }

    #[test]
    fn temporary_container_is_private_and_removed() {
        let local = LocalContainer::create(b"synthetic private bytes").unwrap();
        let path = local.path();
        let directory = local.0.clone();
        assert_eq!(std::fs::read(&path).unwrap(), b"synthetic private bytes");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                std::fs::metadata(&directory).unwrap().permissions().mode() & 0o777,
                0o700
            );
        }
        drop(local);
        assert!(!path.exists());
        assert!(!directory.exists());
    }

    #[test]
    fn rejects_probe_header_bombs() {
        let parse = |width, count: &str, fps: &str, duration: &str| {
            parse_probe(Probe {
                streams: vec![Stream {
                    width,
                    height: 48,
                    avg_frame_rate: fps.into(),
                    duration: Some(duration.into()),
                    nb_frames: Some(count.into()),
                }],
                format: Format::default(),
            })
        };
        assert!(parse(usize::MAX, "1", "1/1", "1").is_err());
        assert!(parse(48, "999999999999", "1/1", "1").is_err());
        assert!(parse(48, "1", "999999999/1", "1").is_err());
        assert!(parse(48, "1", "1/0", "1").is_err());
        assert!(parse(48, "1", "30/1", "NaN").is_err());
        assert!(parse(48, "1", "60/1", "120").is_err());
    }
}
