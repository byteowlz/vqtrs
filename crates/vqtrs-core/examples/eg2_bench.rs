//! Warm, single-input EmbeddingGemma 2 latency using a local snapshot only.
//!
//! Positional arguments: `snapshot cpu|metal|cuda text|image|audio|video input iterations`.
//! Text input is literal text; other inputs are paths to encoded media files.
//! Iterations must be in `1..=1000`. Three warmups and model loading are untimed.
//! Input file reads are excluded; decoding, preprocessing, inference and the
//! final embedding host copy (which synchronizes GPU work) are timed.
//!
//! Compare the same input in separate CPU/GPU runs, for example:
//! ```text
//! cargo run --release -p vqtrs-core --features embeddinggemma2 --example eg2_bench -- \
//!     /path/to/snapshot cpu text "benchmark sentence" 10
//! cargo run --release -p vqtrs-core --features embeddinggemma2-metal --example eg2_bench -- \
//!     /path/to/snapshot metal text "benchmark sentence" 10
//! ```
//! For CUDA, use feature `embeddinggemma2-cuda` and device `cuda`.
//! GPU selection is explicit and never falls back to CPU. No downloads are made.

#[cfg(feature = "embeddinggemma2")]
mod enabled {
    use std::hint::black_box;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use anyhow::{Context as _, Result, bail, ensure};
    use candle_core::Device;
    use vqtrs_core::{EmbeddingGemma2, Gemma2Input};

    const WARMUPS: usize = 3;
    const MAX_ITERATIONS: usize = 1000;
    const USAGE: &str = "usage: eg2_bench <snapshot> <cpu|metal|cuda> <text|image|audio|video> <input> <iterations: 1..=1000>";

    #[derive(Debug)]
    struct Args {
        snapshot: PathBuf,
        device: String,
        modality: String,
        input: String,
        iterations: usize,
    }

    fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args> {
        let args: Vec<_> = args.into_iter().collect();
        let [snapshot, device, modality, input, iterations]: [String; 5] = args
            .try_into()
            .map_err(|_: Vec<String>| anyhow::anyhow!(USAGE))?;
        ensure!(
            matches!(device.as_str(), "cpu" | "metal" | "cuda"),
            "unknown device {device}; {USAGE}"
        );
        ensure!(
            matches!(modality.as_str(), "text" | "image" | "audio" | "video"),
            "unknown modality {modality}; {USAGE}"
        );
        let iterations = iterations.parse::<usize>().context(USAGE)?;
        ensure!(
            (1..=MAX_ITERATIONS).contains(&iterations),
            "iterations must be in 1..={MAX_ITERATIONS}"
        );
        Ok(Args {
            snapshot: snapshot.into(),
            device,
            modality,
            input,
            iterations,
        })
    }

    /// Benchmark a local snapshot with an explicitly selected device.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid arguments, unavailable hardware, unreadable
    /// local files or failed inference.
    pub fn run() -> Result<()> {
        let args = parse_args(std::env::args().skip(1))?;
        let device = match args.device.as_str() {
            "cpu" => Device::Cpu,
            "metal" => Device::new_metal(0).context("cannot initialize requested Metal device")?,
            "cuda" => Device::new_cuda(0).context("cannot initialize requested CUDA device")?,
            other => bail!("unsupported device {other}"),
        };
        let bytes = if args.modality == "text" {
            Vec::new()
        } else {
            std::fs::read(&args.input).with_context(|| format!("reading {}", args.input))?
        };
        let input = match args.modality.as_str() {
            "text" => Gemma2Input::Text(&args.input),
            "image" => Gemma2Input::Image(&bytes),
            "audio" => Gemma2Input::Audio(&bytes),
            "video" => Gemma2Input::Video(&bytes),
            other => bail!("unsupported modality {other}"),
        };
        let model = EmbeddingGemma2::from_dir_on(&args.snapshot, &device)?;
        println!(
            "device={:?} modality={} iterations={} warmups={WARMUPS}",
            model.device(),
            args.modality,
            args.iterations
        );
        for _ in 0..WARMUPS {
            black_box(model.embed(input)?);
        }

        let mut total = Duration::ZERO;
        let mut min = Duration::MAX;
        let mut max = Duration::ZERO;
        for _ in 0..args.iterations {
            let start = Instant::now();
            // embed returns a host Vec, so queued GPU work has completed here.
            let embedding = black_box(model.embed(input)?);
            let elapsed = start.elapsed();
            drop(embedding);
            total += elapsed;
            min = min.min(elapsed);
            max = max.max(elapsed);
        }
        println!(
            "warm embed ms: mean={:.3} min={:.3} max={:.3}",
            total.as_secs_f64() * 1000.0 / args.iterations as f64,
            min.as_secs_f64() * 1000.0,
            max.as_secs_f64() * 1000.0
        );
        println!(
            "includes decoding/preprocessing and synchronized final host copy; excludes model load, input read and warmups"
        );
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn args(device: &str, modality: &str, iterations: &str) -> [String; 5] {
            ["/local/snapshot", device, modality, "input", iterations].map(str::to_owned)
        }

        #[test]
        fn accepts_all_devices_and_modalities_without_initializing_hardware() -> Result<()> {
            for device in ["cpu", "metal", "cuda"] {
                for modality in ["text", "image", "audio", "video"] {
                    let parsed = parse_args(args(device, modality, "1000"))?;
                    ensure!(parsed.device == device && parsed.modality == modality);
                    ensure!(parsed.iterations == MAX_ITERATIONS);
                }
            }
            ensure!(parse_args(args("cpu", "text", "1"))?.iterations == 1);
            Ok(())
        }

        #[test]
        fn rejects_invalid_iterations_and_selections() -> Result<()> {
            for iterations in ["0", "1001", "-1", "invalid", "18446744073709551616"] {
                ensure!(parse_args(args("cpu", "text", iterations)).is_err());
            }
            ensure!(parse_args(args("auto", "text", "1")).is_err());
            ensure!(parse_args(args("cpu", "unknown", "1")).is_err());
            ensure!(parse_args(Vec::<String>::new()).is_err());
            let mut extra = args("cpu", "text", "1").to_vec();
            extra.push("extra".to_owned());
            ensure!(parse_args(extra).is_err());
            Ok(())
        }
    }
}

#[cfg(feature = "embeddinggemma2")]
fn main() -> std::process::ExitCode {
    match enabled::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(feature = "embeddinggemma2"))]
fn main() -> std::process::ExitCode {
    eprintln!(
        "eg2_bench requires feature embeddinggemma2, embeddinggemma2-metal or embeddinggemma2-cuda"
    );
    std::process::ExitCode::FAILURE
}
