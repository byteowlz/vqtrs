//! Video-specific frontend/model parity against the pinned batched CPU HF oracle.
//!
//! Generate fixtures with `scripts/parity/eg2_video_frontend_reference.py`
//! `--model-forward`, then run:
//! `cargo run --release -p vqtrs-core --features embeddinggemma2 --example`
//! `eg2_video_parity -- /tmp/eg2-video-ref <checkpoint-directory> [cpu|metal|cuda]`.
//! CPU requires bit identity; GPU requires final max error <= 2e-4 and
//! cosine at least 0.99999. Preprocessing, sampled indices, positions and IDs stay strict.
//! GPU constructors are explicit (no fallback); enable the corresponding feature.
//! This checks decoded RGB arrays AND encoded containers, without modifying
//! the shared text/image/audio parity runner.

#[cfg(feature = "embeddinggemma2")]
mod enabled {
    use std::collections::BTreeMap;
    use std::path::Path;

    use anyhow::{Context as _, Result, bail, ensure};
    use candle_core::Device;
    use serde::Deserialize;
    use vqtrs_core::{
        EmbeddingGemma2, Forward, Gemma2Input, Gemma2VideoFrame, Gemma2VideoMetadata,
    };

    #[derive(Deserialize)]
    struct Manifest {
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
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
        input_ids: Vec<u32>,
        model_tensors: Option<BTreeMap<String, TensorFile>>,
    }

    #[derive(Deserialize)]
    struct TensorFile {
        file: String,
    }

    struct FloatDiff {
        differing: usize,
        max_abs: f32,
        cosine: f64,
    }

    fn compare_floats(dir: &Path, file: &str, ours: &[f32], label: &str) -> Result<FloatDiff> {
        let reference = std::fs::read(dir.join(file))?;
        ensure!(
            reference.len() == ours.len() * 4,
            "{label}: length mismatch"
        );
        let mut differing = 0;
        let mut max_abs = 0.0_f32;
        let (mut dot, mut ours_norm, mut ref_norm) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (value, bytes) in ours.iter().zip(reference.chunks_exact(4)) {
            let expected = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            ensure!(
                value.is_finite() && expected.is_finite(),
                "{label}: nonfinite value"
            );
            differing += usize::from(value.to_bits() != expected.to_bits());
            max_abs = max_abs.max((value - expected).abs());
            let (a, b) = (f64::from(*value), f64::from(expected));
            dot = a.mul_add(b, dot);
            ours_norm = a.mul_add(a, ours_norm);
            ref_norm = b.mul_add(b, ref_norm);
        }
        let cosine = dot / (ours_norm.sqrt() * ref_norm.sqrt());
        println!(
            "  {label}: differing={differing}/{} max_abs={max_abs:.3e} cosine={cosine:.9}",
            ours.len()
        );
        Ok(FloatDiff {
            differing,
            max_abs,
            cosine,
        })
    }

    fn check_forward(dir: &Path, case: &Case, output: &Forward, strict: bool) -> Result<()> {
        ensure!(
            output.input_ids == case.input_ids,
            "{}: input ids",
            case.name
        );
        ensure!(
            output.sampled_frame_indices == case.sampled_indices,
            "{}: sampled indices",
            case.name
        );
        ensure!(
            output.video_num_soft_tokens == case.num_soft_tokens,
            "{}: soft-token count",
            case.name
        );
        let pixels = compare_floats(
            dir,
            &case.pixels_file,
            &output.video_pixel_values,
            "video pixels",
        )?;
        ensure!(pixels.differing == 0, "video pixels: not bit-identical");
        let positions: Vec<u8> = output
            .video_positions
            .iter()
            .flat_map(|p| p.iter().flat_map(|v| v.to_le_bytes()))
            .collect();
        ensure!(
            positions == std::fs::read(dir.join(&case.positions_file))?,
            "{}: video positions",
            case.name
        );
        let tensors = case
            .model_tensors
            .as_ref()
            .context("missing model tensors")?;
        for (name, values) in [
            ("video_features", output.video_features.as_slice()),
            ("hidden_states", output.hidden_states.as_slice()),
            ("token_embeddings", output.token_embeddings.as_slice()),
            ("embedding", output.embedding.as_slice()),
        ] {
            let entry = tensors
                .get(name)
                .with_context(|| format!("missing {name}"))?;
            let diff = compare_floats(dir, &entry.file, values, name)?;
            if strict {
                ensure!(diff.differing == 0, "{name}: not bit-identical");
            } else if name == "embedding" {
                ensure!(
                    diff.max_abs <= 2e-4 && diff.cosine >= 0.99999,
                    "GPU final embedding outside max_abs<=2e-4 / cosine>=0.99999 tolerance"
                );
            }
        }
        Ok(())
    }

    fn select_device(name: &str) -> Result<Device> {
        match name {
            "cpu" => Ok(Device::Cpu),
            "metal" => Ok(Device::new_metal(0)?),
            "cuda" => Ok(Device::new_cuda(0)?),
            other => bail!("unknown device {other}; expected cpu, metal or cuda"),
        }
    }

    pub fn run() -> Result<()> {
        let mut args = std::env::args().skip(1);
        let reference = args
            .next()
            .context("expected reference directory and checkpoint directory")?;
        let model_dir = args.next().context("expected checkpoint directory")?;
        let device_name = args.next().unwrap_or_else(|| "cpu".to_owned());
        ensure!(args.next().is_none(), "unexpected extra argument");
        let device = select_device(&device_name)?;
        let strict = device_name == "cpu";
        let dir = Path::new(&reference);
        let manifest: Manifest =
            serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
        let model = EmbeddingGemma2::from_dir_on(Path::new(&model_dir), &device)?;
        let mut compared = 0;
        for case in manifest.cases.iter().filter(|c| c.model_tensors.is_some()) {
            let raw = std::fs::read(dir.join(&case.frames_file))?;
            let size = case.width * case.height * 3;
            ensure!(
                raw.len() == size * case.count,
                "invalid fixture frame bytes"
            );
            let frames: Vec<_> = raw
                .chunks_exact(size)
                .map(|data| Gemma2VideoFrame {
                    width: case.width,
                    height: case.height,
                    data: data.to_vec(),
                })
                .collect();
            let metadata = Gemma2VideoMetadata {
                total_num_frames: case.count,
                fps: case.fps,
                duration: case.duration,
            };
            println!("{} decoded:", case.name);
            let output = model.forward(
                Gemma2Input::VideoFrames {
                    frames: &frames,
                    metadata: Some(metadata),
                },
                true,
            )?;
            check_forward(dir, case, &output, strict)?;
            compared += 1;
            if let Some(file) = &case.container_file {
                println!("{} container:", case.name);
                let bytes = std::fs::read(dir.join(file))?;
                let output = model.forward(Gemma2Input::Video(&bytes), true)?;
                check_forward(dir, case, &output, strict)?;
                compared += 1;
            }
        }
        ensure!(
            compared > 0,
            "no model references; generate with --model-forward"
        );
        if strict {
            println!("{compared} video forwards are bit-identical to batched HF CPU");
        } else {
            println!(
                "{compared} {device_name} video forwards pass final embedding tolerances with strict preprocessing"
            );
        }
        Ok(())
    }
}

#[cfg(feature = "embeddinggemma2")]
fn main() -> anyhow::Result<()> {
    enabled::run()
}

#[cfg(not(feature = "embeddinggemma2"))]
fn main() -> std::process::ExitCode {
    eprintln!("eg2_video_parity requires --features embeddinggemma2");
    std::process::ExitCode::FAILURE
}
