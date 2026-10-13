//! Experimental Apple runtimes behind a bounded, serialized private worker.

mod prepare;
mod process;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use super::{Gemma2Input, mel};
use crate::catalog::Backend;
use crate::error::{Result, VqtrsError};
use process::Worker;

pub const COREML_ID: &str = "vqtrs/embeddinggemma-2-coreml-fp16-3f978ddf";
pub const MLX_ID: &str = "vqtrs/embeddinggemma-2-mlx-regular-bf16-4f4634bb";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Runtime {
    CoreMl,
    Mlx,
}

impl Runtime {
    const fn limit(self) -> usize {
        match self {
            Self::CoreMl => 512,
            Self::Mlx => 8192,
        }
    }

    pub(crate) const fn identity(self) -> &'static str {
        match self {
            Self::CoreMl => COREML_ID,
            Self::Mlx => MLX_ID,
        }
    }

    pub(crate) const fn backend(self) -> Backend {
        match self {
            Self::CoreMl => Backend::EmbeddingGemma2CoreMl,
            Self::Mlx => Backend::EmbeddingGemma2Mlx,
        }
    }

    const fn argument(self) -> &'static str {
        match self {
            Self::CoreMl => "coreml",
            Self::Mlx => "mlx",
        }
    }
}

pub struct Native {
    runtime: Runtime,
    tokenizer: Tokenizer,
    mel: mel::MelFrontEnd,
    worker: Mutex<Worker>,
}

impl Native {
    pub(crate) fn load(runtime: Runtime) -> Result<Self> {
        if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return Err(fail("native EmbeddingGemma 2 requires Apple Silicon macOS"));
        }
        let root = required_dir(match runtime {
            Runtime::CoreMl => "VQTRS_EG2_COREML_DIR",
            Runtime::Mlx => "VQTRS_EG2_GOOGLE_DIR",
        })?;
        let python = std::env::var_os("VQTRS_NATIVE_PYTHON")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute() && p.is_file())
            .ok_or_else(|| fail("set VQTRS_NATIVE_PYTHON to an absolute private Python3.12 executable; see docs/embeddinggemma2-native.md"))?;
        let mut tokenizer = Tokenizer::from_file(root.join("tokenizer.json")).map_err(error)?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(error)?;
        let worker = Worker::start(runtime, &root, &python)?;
        let mel = mel::MelFrontEnd::new(mel::MelSettings::default(), &candle_core::Device::Cpu)
            .map_err(error)?;
        Ok(Self {
            runtime,
            tokenizer,
            mel,
            worker: Mutex::new(worker),
        })
    }

    pub(crate) fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() || text.len() > 256 * 1024 {
            return Err(fail("native text must be nonempty and <=256KiB"));
        }
        let ids = self
            .tokenizer
            .encode(text, true)
            .map_err(error)?
            .get_ids()
            .to_vec();
        self.check_length(ids.len())?;
        if ids
            .iter()
            .any(|id| (255_999..=256_000).contains(id) || (258_880..=258_884).contains(id))
        {
            return Err(fail(
                "native text cannot contain media placeholder tokens; use typed media inputs",
            ));
        }
        Ok(ids)
    }

    fn check_length(&self, length: usize) -> Result<()> {
        check_length(self.runtime, length)
    }

    pub(crate) fn embed(&self, input: Gemma2Input<'_>) -> Result<Vec<f32>> {
        let prepared = self.prepare(input)?;
        let mut worker = self.worker.lock().map_err(|_| VqtrsError::Poisoned)?;
        worker
            .predict(&[prepared])?
            .pop()
            .ok_or_else(|| fail("native worker returned no vector"))
    }

    pub(crate) fn texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let prepared = texts
            .iter()
            .map(|text| self.prepare(Gemma2Input::Text(text)))
            .collect::<Result<Vec<_>>>()?;
        let mut worker = self.worker.lock().map_err(|_| VqtrsError::Poisoned)?;
        let mut out = Vec::with_capacity(texts.len());
        let mut offset = 0;
        while offset < prepared.len() {
            // Keep regular MLX's long-text activation memory bounded. Short ragged
            // batches preserve the measured regular-backend throughput advantage.
            let mut end = (offset + 16).min(prepared.len());
            if prepared[offset..end].iter().any(|p| p.item.ids.len() > 512) {
                end = offset + 1;
            }
            out.extend(worker.predict(&prepared[offset..end])?);
            offset = end;
        }
        drop(worker);
        Ok(out)
    }

    pub(crate) fn count_tokens(&self, texts: &[String]) -> Option<usize> {
        texts.iter().try_fold(0_usize, |n, text| {
            n.checked_add(self.tokenizer.encode(text.as_str(), true).ok()?.len())
        })
    }
}

#[derive(Serialize)]
struct Request<'a> {
    protocol: u8,
    id: u64,
    items: Vec<&'a Item>,
}

#[derive(Serialize, Default)]
struct Item {
    kind: &'static str,
    ids: Vec<u32>,
    directory: Option<PathBuf>,
    frames: usize,
    patches: usize,
    soft_tokens: usize,
    samples: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reply {
    protocol: u8,
    model: String,
    #[serde(default)]
    id: Option<u64>,
    #[serde(default)]
    vectors: Vec<Vec<f32>>,
}

fn validate(
    reply: Reply,
    runtime: Runtime,
    id: Option<u64>,
    count: usize,
) -> Result<Vec<Vec<f32>>> {
    if reply.protocol != 1
        || reply.model != runtime.identity()
        || reply.id != id
        || reply.vectors.len() != count
    {
        return Err(fail("native worker identity/protocol/order/count mismatch"));
    }
    for vector in &reply.vectors {
        let norm = vector
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt();
        if vector.len() != 768 || !norm.is_finite() || (norm - 1.0).abs() > 2e-6 {
            return Err(fail(
                "native worker returned invalid/nonfinite/unnormalized768d vector",
            ));
        }
    }
    Ok(reply.vectors)
}

fn check_length(runtime: Runtime, length: usize) -> Result<()> {
    if length == 0 || length > runtime.limit() {
        return Err(fail(format!(
            "{} input has {length} tokens; explicit limit is {}; split the input (no truncation or fallback)",
            runtime.identity(),
            runtime.limit()
        )));
    }
    Ok(())
}

fn required_dir(name: &str) -> Result<PathBuf> {
    std::env::var_os(name).map(PathBuf::from).filter(|p| p.is_absolute() && p.is_dir()).ok_or_else(|| fail(format!("set {name} to an absolute pinned asset directory; see docs/embeddinggemma2-native.md")))
}

fn error(error: impl std::fmt::Display) -> VqtrsError {
    fail(error.to_string())
}
fn fail(message: impl Into<String>) -> VqtrsError {
    VqtrsError::Backend(message.into())
}

fn write_floats(path: &Path, values: &[f32]) -> Result<()> {
    use std::io::Write as _;
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    for value in values {
        if !value.is_finite() {
            return Err(fail("nonfinite preprocessing tensor"));
        }
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reply(vector: Vec<f32>) -> Reply {
        Reply {
            protocol: 1,
            model: COREML_ID.into(),
            id: Some(7),
            vectors: vec![vector],
        }
    }

    #[test]
    fn exact_context_boundaries_do_not_truncate() {
        for (runtime, limit) in [(Runtime::CoreMl, 512), (Runtime::Mlx, 8192)] {
            assert!(check_length(runtime, 0).is_err());
            assert!(check_length(runtime, limit - 1).is_ok());
            assert!(check_length(runtime, limit).is_ok());
            assert!(check_length(runtime, limit + 1).is_err());
        }
    }

    #[test]
    fn validates_identity_order_shape_finiteness_and_norm() {
        let mut vector = vec![0.0; 768];
        vector[0] = 1.0;
        assert!(validate(reply(vector.clone()), Runtime::CoreMl, Some(7), 1).is_ok());
        assert!(validate(reply(vector.clone()), Runtime::Mlx, Some(7), 1).is_err());
        assert!(validate(reply(vector.clone()), Runtime::CoreMl, Some(8), 1).is_err());
        assert!(validate(reply(vector), Runtime::CoreMl, Some(7), 2).is_err());
        for bad in [
            vec![1.0],
            vec![0.0; 768],
            vec![f32::NAN; 768],
            vec![f32::INFINITY; 768],
            vec![1.0; 768],
        ] {
            assert!(validate(reply(bad), Runtime::CoreMl, Some(7), 1).is_err());
        }
    }
}
