//! Parity check of vqtrs' EmbeddingGemma 2 against the Python reference.
//!
//! Reads a reference directory written by `scripts/parity/eg2_reference.py`
//! and, for every case, compares vqtrs' own tokenization and outputs against
//! the reference tensors: token ids (exact), every text layer, the per-token
//! embeddings and the final embedding (vs both reference paths).
//!
//! ```text
//! cargo run --release -p vqtrs-core --features embeddinggemma2 --example eg2_parity -- \
//!     --reference /tmp/eg2-parity/ref-text --model-dir <snapshot dir>
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use serde_json::Value;
use vqtrs_core::{EmbeddingGemma2, Gemma2Input};

/// Element-wise comparison of two `f32` buffers.
struct Diff {
    max_abs: f32,
    max_ulp: u32,
    identical: usize,
    total: usize,
    cosine: f64,
}

impl Diff {
    fn of(ours: &[f32], reference: &[f32]) -> Result<Self> {
        if ours.len() != reference.len() {
            bail!(
                "length mismatch: ours {} vs reference {}",
                ours.len(),
                reference.len()
            );
        }
        let (mut max_abs, mut max_ulp, mut identical) = (0.0_f32, 0_u32, 0_usize);
        let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (&a, &b) in ours.iter().zip(reference) {
            if a.to_bits() == b.to_bits() {
                identical += 1;
            }
            max_abs = max_abs.max((a - b).abs());
            max_ulp = max_ulp.max(ulp_distance(a, b));
            dot = f64::from(a).mul_add(f64::from(b), dot);
            na = f64::from(a).mul_add(f64::from(a), na);
            nb = f64::from(b).mul_add(f64::from(b), nb);
        }
        Ok(Self {
            max_abs,
            max_ulp,
            identical,
            total: ours.len(),
            cosine: dot / (na.sqrt() * nb.sqrt()),
        })
    }

    const fn bit_identical(&self) -> bool {
        self.identical == self.total
    }

    fn line(&self) -> String {
        format!(
            "max|d|={:.3e} maxULP={:>8} identical={:>6.2}% cos={:.9}",
            self.max_abs,
            self.max_ulp,
            100.0 * self.identical as f64 / self.total.max(1) as f64,
            self.cosine
        )
    }
}

/// Distance in units in the last place, over the ordered integer line of floats.
fn ulp_distance(a: f32, b: f32) -> u32 {
    let key = |x: f32| {
        let bits = x.to_bits() as i32;
        if bits < 0 { i32::MIN - bits } else { bits }
    };
    key(a).abs_diff(key(b))
}

fn read_f32(dir: &Path, entry: &Value) -> Result<Vec<f32>> {
    let file = entry["file"]
        .as_str()
        .context("tensor entry without file")?;
    let bytes = std::fs::read(dir.join(file))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_i64(dir: &Path, entry: &Value) -> Result<Vec<i64>> {
    let file = entry["file"]
        .as_str()
        .context("tensor entry without file")?;
    let bytes = std::fs::read(dir.join(file))?;
    Ok(bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

struct Args {
    reference: PathBuf,
    model_dir: Option<PathBuf>,
    cases: Option<PathBuf>,
    device: candle_core::Device,
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1);
    let (mut reference, mut model_dir, mut cases) = (None, None, None);
    let mut device = candle_core::Device::Cpu;
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .with_context(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--reference" => reference = Some(PathBuf::from(value)),
            "--model-dir" => model_dir = Some(PathBuf::from(value)),
            "--cases" => cases = Some(PathBuf::from(value)),
            "--device" => device = parse_device(&value)?,
            other => bail!("unknown flag {other}"),
        }
    }
    Ok(Args {
        reference: reference.context("--reference is required")?,
        model_dir,
        cases,
        device,
    })
}

fn parse_device(name: &str) -> Result<candle_core::Device> {
    Ok(match name {
        "cpu" => candle_core::Device::Cpu,
        "metal" => candle_core::Device::new_metal(0)?,
        "cuda" => candle_core::Device::new_cuda(0)?,
        other => bail!("unknown device {other}; expected cpu, cuda or metal"),
    })
}

/// Tally of compared tensors.
#[derive(Default)]
struct Tally {
    compared: usize,
    differing: usize,
    failed: usize,
    gpu: bool,
}

impl Tally {
    fn record(&mut self, label: &str, diff: &Diff, show: bool) {
        self.compared += 1;
        if !diff.bit_identical() {
            self.differing += 1;
        }
        // Internal tensor differences are reported, not hidden. GPU acceptance
        // gates exact preprocessing/IDs and final embedding accuracy separately.
        if !self.accepts(label, diff) {
            self.failed += 1;
        }
        if show || !diff.bit_identical() {
            let tag = if diff.bit_identical() {
                "  BIT-IDENTICAL"
            } else {
                ""
            };
            println!("  {label:20}{}{tag}", diff.line());
        }
    }

    fn accepts(&self, label: &str, diff: &Diff) -> bool {
        if !self.gpu {
            return diff.bit_identical();
        }
        match label {
            "embedding vs hf" => {
                diff.max_abs.is_finite() && diff.max_abs <= 2e-4 && diff.cosine >= 0.99999
            }
            "pixel values" | "audio mel features" => diff.bit_identical(),
            _ => true,
        }
    }

    fn exact(&mut self, label: &str, same: bool) {
        self.compared += 1;
        if !same {
            self.differing += 1;
            self.failed += 1;
        }
        println!("  {label:20}{}", if same { "IDENTICAL" } else { "DIFFER" });
    }
}

fn check_audio(
    out: &vqtrs_core::Forward,
    dir: &Path,
    tensors: &Value,
    tally: &mut Tally,
) -> Result<()> {
    tally.record(
        "audio mel features",
        &Diff::of(
            &out.input_features,
            &read_f32(dir, &tensors["input.input_features"])?,
        )?,
        true,
    );
    let mask_file = tensors["input.input_features_mask"]["file"]
        .as_str()
        .context("mask without file")?;
    let mask = std::fs::read(dir.join(mask_file))?;
    tally.exact(
        "audio frame mask",
        out.input_features_mask == mask.iter().map(|&b| b != 0).collect::<Vec<_>>(),
    );
    tally.record(
        "audio soft tokens",
        &Diff::of(
            &out.audio_features,
            &read_f32(dir, &tensors["hf.audio_features"])?,
        )?,
        true,
    );
    for (i, ours) in out.audio_hidden_states.iter().enumerate() {
        let label = if i == 0 {
            "subsample_conv_projection".to_owned()
        } else {
            format!("layers.{}", i - 1)
        };
        let entry = &tensors[format!("hf.audio.{label}")];
        if !entry.is_null() {
            tally.record(
                &format!("audio {label}"),
                &Diff::of(ours, &read_f32(dir, entry)?)?,
                true,
            );
        }
    }
    Ok(())
}

fn check_case(
    model: &EmbeddingGemma2,
    dir: &Path,
    case: &Value,
    input: &Value,
    base: &Path,
    tally: &mut Tally,
) -> Result<()> {
    let name = case["name"].as_str().unwrap_or("?");
    let t = &case["tensors"];
    let bytes;
    let text;
    let input = match case["modality"].as_str() {
        Some("text") => {
            text = format!(
                "{}{}",
                case["prompt"].as_str().unwrap_or(""),
                input["text"].as_str().context("no text")?
            );
            Gemma2Input::Text(&text)
        }
        Some("image") => {
            bytes = std::fs::read(base.join(input["path"].as_str().context("no path")?))?;
            Gemma2Input::Image(&bytes)
        }
        Some("audio") => {
            bytes = std::fs::read(base.join(input["path"].as_str().context("no path")?))?;
            Gemma2Input::Audio(&bytes)
        }
        other => bail!("unsupported modality {other:?}"),
    };
    let out = model.forward(input, true)?;
    println!("{name}: seq={}", out.input_ids.len());

    let ref_ids: Vec<u32> = read_i64(dir, &t["input.input_ids"])?
        .into_iter()
        .map(|v| v as u32)
        .collect();
    tally.exact("token ids", out.input_ids == ref_ids);
    if out.input_ids != ref_ids {
        return Ok(());
    }
    if !t["input.pixel_values"].is_null() {
        tally.record(
            "pixel values",
            &Diff::of(&out.pixel_values, &read_f32(dir, &t["input.pixel_values"])?)?,
            true,
        );
        let ref_pos = read_i64(dir, &t["input.image_position_ids"])?;
        let ours: Vec<i64> = out.image_positions.iter().flatten().copied().collect();
        tally.exact("patch positions", ours == ref_pos);
        tally.record(
            "image soft tokens",
            &Diff::of(
                &out.image_features,
                &read_f32(dir, &t["hf.image_features"])?,
            )?,
            true,
        );
    }
    if !t["input.input_features"].is_null() {
        check_audio(&out, dir, t, tally)?;
    }
    let ref_hidden = read_f32(dir, &t["hf.hidden_states"])?;
    let per_layer = out.input_ids.len() * 512;
    let all = std::env::var_os("EG2_ALL_LAYERS").is_some();
    for (i, (ours, theirs)) in out
        .hidden_states
        .chunks(per_layer)
        .zip(ref_hidden.chunks(per_layer))
        .enumerate()
    {
        let label = if i == 0 {
            "input embeds".to_owned()
        } else {
            format!("after layer {:>2}", i - 1)
        };
        tally.record(&label, &Diff::of(ours, theirs)?, all || i == 0);
    }
    tally.record(
        "token embeddings",
        &Diff::of(
            &out.token_embeddings,
            &read_f32(dir, &t["hf.token_embeddings"])?,
        )?,
        true,
    );
    tally.record(
        "embedding vs hf",
        &Diff::of(&out.embedding, &read_f32(dir, &t["hf.embedding"])?)?,
        true,
    );
    let vs_st = Diff::of(&out.embedding, &read_f32(dir, &t["st.embedding"])?)?;
    println!(
        "  {:20}{}  (sdpa attention; informational)",
        "embedding vs st",
        vs_st.line()
    );
    Ok(())
}

fn run() -> Result<bool> {
    let args = parse_args()?;
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(args.reference.join("manifest.json"))?)?;
    let cases_file = args
        .cases
        .unwrap_or_else(|| PathBuf::from("scripts/parity/eg2_cases_text.json"));
    let cases: Value = serde_json::from_slice(&std::fs::read(&cases_file)?)?;
    let base = cases_file
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    println!("reference versions: {}", manifest["versions"]);

    let model = match &args.model_dir {
        Some(dir) if args.device.is_cpu() => EmbeddingGemma2::from_dir(dir)?,
        Some(dir) => EmbeddingGemma2::from_dir_on(dir, &args.device)?,
        None => EmbeddingGemma2::from_hf_on(vqtrs_core::EMBEDDING_GEMMA2_REPO, &args.device)?,
    };
    println!("actual device: {:?}, dtype: f32", model.device());
    let mut tally = Tally {
        gpu: !model.device().is_cpu(),
        ..Tally::default()
    };
    for case in manifest["cases"]
        .as_array()
        .context("manifest without cases")?
    {
        let input = cases
            .as_array()
            .and_then(|c| c.iter().find(|c| c["name"] == case["name"]))
            .context("case missing from the cases file")?;
        check_case(&model, &args.reference, case, input, &base, &mut tally)?;
    }
    println!(
        "compared {} tensors, {} not bit-identical",
        tally.compared, tally.differing
    );
    println!(
        "{} acceptance failures (GPU: max_abs <= 2e-4, cosine >= 0.99999; CPU: bit identity)",
        tally.failed
    );
    Ok(tally.failed == 0)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => {
            eprintln!("outputs failed the selected device's parity contract");
            ExitCode::FAILURE
        }
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}
