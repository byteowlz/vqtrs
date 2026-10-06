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
use vqtrs_core::EmbeddingGemma2;

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
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1);
    let (mut reference, mut model_dir, mut cases) = (None, None, None);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .with_context(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--reference" => reference = Some(PathBuf::from(value)),
            "--model-dir" => model_dir = Some(PathBuf::from(value)),
            "--cases" => cases = Some(PathBuf::from(value)),
            other => bail!("unknown flag {other}"),
        }
    }
    Ok(Args {
        reference: reference.context("--reference is required")?,
        model_dir,
        cases,
    })
}

/// Returns whether the case's token ids matched exactly.
fn check_text_case(model: &EmbeddingGemma2, dir: &Path, case: &Value, text: &str) -> Result<bool> {
    let name = case["name"].as_str().unwrap_or("?");
    let tensors = &case["tensors"];
    let prompt = case["prompt"].as_str().unwrap_or("");
    let ids = model.tokenize(&format!("{prompt}{text}"))?;
    let ref_ids: Vec<u32> = read_i64(dir, &tensors["input.input_ids"])?
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let ids_ok = ids == ref_ids;
    println!(
        "{name}: seq={} token ids {}",
        ids.len(),
        if ids_ok { "IDENTICAL" } else { "DIFFER" }
    );
    if !ids_ok {
        return Ok(false);
    }

    let out = model.forward_text_ids(&ids, true)?;
    let ref_hidden = read_f32(dir, &tensors["hf.hidden_states"])?;
    let per_layer = out.seq * 512;
    for (i, (ours, theirs)) in out
        .hidden_states
        .chunks(per_layer)
        .zip(ref_hidden.chunks(per_layer))
        .enumerate()
    {
        if std::env::var_os("EG2_ALL_LAYERS").is_some() || i < 3 || i % 6 == 0 || i == 23 {
            let label = if i == 0 {
                "input embeds".to_owned()
            } else {
                format!("after layer {:>2}", i - 1)
            };
            println!("  {label:18}  {}", Diff::of(ours, theirs)?.line());
        }
    }
    let tokens = Diff::of(
        &out.token_embeddings,
        &read_f32(dir, &tensors["hf.token_embeddings"])?,
    )?;
    println!("  token embeddings    {}", tokens.line());
    let vs_hf = Diff::of(&out.embedding, &read_f32(dir, &tensors["hf.embedding"])?)?;
    let vs_st = Diff::of(&out.embedding, &read_f32(dir, &tensors["st.embedding"])?)?;
    println!(
        "  embedding vs hf     {}{}",
        vs_hf.line(),
        if vs_hf.bit_identical() {
            "  BIT-IDENTICAL"
        } else {
            ""
        }
    );
    println!(
        "  embedding vs st     {}{}",
        vs_st.line(),
        if vs_st.bit_identical() {
            "  BIT-IDENTICAL"
        } else {
            ""
        }
    );
    Ok(true)
}

fn run() -> Result<bool> {
    let args = parse_args()?;
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(args.reference.join("manifest.json"))?)?;
    let cases_file = args
        .cases
        .unwrap_or_else(|| PathBuf::from("scripts/parity/eg2_cases_text.json"));
    let cases: Value = serde_json::from_slice(&std::fs::read(&cases_file)?)?;
    println!("reference versions: {}", manifest["versions"]);

    let model = match &args.model_dir {
        Some(dir) => EmbeddingGemma2::from_dir(dir)?,
        None => EmbeddingGemma2::from_hf(vqtrs_core::EMBEDDING_GEMMA2_REPO)?,
    };
    let mut all_ids_ok = true;
    for case in manifest["cases"]
        .as_array()
        .context("manifest without cases")?
    {
        if case["modality"] != "text" {
            continue;
        }
        let input = cases
            .as_array()
            .and_then(|c| c.iter().find(|c| c["name"] == case["name"]))
            .context("case missing from the cases file")?;
        let text = input["text"].as_str().context("text case without text")?;
        all_ids_ok &= check_text_case(&model, &args.reference, case, text)?;
    }
    Ok(all_ids_ok)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => {
            eprintln!("token ids differ from the reference");
            ExitCode::FAILURE
        }
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}
