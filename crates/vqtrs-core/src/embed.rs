//! Dense text-embedding engine over the ONNX and Qwen3 backends.

use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "qwen3")]
use fastembed::Qwen3TextEmbedding;
use fastembed::{InitOptions, TextEmbedding};
use tokenizers::Tokenizer;

use crate::cache::model_cache_dir;
#[cfg(feature = "qwen3")]
use crate::catalog::QWEN3_MAX_LENGTH;
use crate::catalog::{Backend, Resolved, hf_repo, resolve_dense};
use crate::error::{Result, VqtrsError};

/// Loaded backend instance. Variants are boxed so the enum stays pointer-sized
/// regardless of how the two backend structs differ in size.
///
/// fastembed's ONNX `embed` takes `&mut self`, so it is guarded by a [`Mutex`]
/// to keep [`Engine`] inference `&self` (and therefore `Arc`-shareable). The
/// Qwen3 backend embeds through `&self` and needs no lock.
enum Inner {
    Onnx(Box<Mutex<TextEmbedding>>),
    #[cfg(feature = "qwen3")]
    Qwen3(Box<Qwen3TextEmbedding>),
}

/// A loaded text-embedding model.
///
/// Inference takes `&self`, so an `Engine` can be wrapped in an `Arc` and
/// shared across threads. Loading downloads the model on first use and caches
/// it under the OS cache directory.
pub struct Engine {
    model: String,
    dimensions: usize,
    backend: Backend,
    inner: Inner,
    tokenizer: OnceLock<Option<Arc<Tokenizer>>>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("model", &self.model)
            .field("dimensions", &self.dimensions)
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl Engine {
    /// Load the model identified by `model` (a catalog code or variant name).
    ///
    /// # Errors
    ///
    /// Returns an error if the name cannot be resolved, the cache directory
    /// cannot be created, or the backend fails to initialise the model.
    pub fn load(model: &str) -> Result<Self> {
        match resolve_dense(model)? {
            Resolved::Onnx {
                model: variant,
                dimensions,
            } => {
                let opts = InitOptions::new(variant)
                    .with_cache_dir(model_cache_dir()?)
                    .with_show_download_progress(true)
                    .with_execution_providers(crate::accel::execution_providers());
                let mut inner = TextEmbedding::try_new(opts).map_err(backend_err)?;
                let dimensions = match dimensions {
                    Some(d) => d,
                    None => onnx_dimensions(&mut inner)?,
                };
                Ok(Self {
                    model: model.to_owned(),
                    dimensions,
                    backend: Backend::Onnx,
                    inner: Inner::Onnx(Box::new(Mutex::new(inner))),
                    tokenizer: OnceLock::new(),
                })
            }
            #[cfg(feature = "qwen3")]
            Resolved::Qwen3 { repo, dimensions } => {
                let (device, dtype) = crate::accel::qwen3_device();
                let inner = Qwen3TextEmbedding::from_hf(&repo, &device, dtype, QWEN3_MAX_LENGTH)
                    .map_err(backend_err)?;
                Ok(Self {
                    model: model.to_owned(),
                    dimensions,
                    backend: Backend::Qwen3,
                    inner: Inner::Qwen3(Box::new(inner)),
                    tokenizer: OnceLock::new(),
                })
            }
        }
    }

    /// Embed a single string.
    ///
    /// # Errors
    ///
    /// Returns an error if backend inference fails or yields no vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let owned = [text.to_owned()];
        self.embed_batch(&owned)?
            .pop()
            .ok_or_else(|| VqtrsError::Backend("model returned no embedding".to_owned()))
    }

    /// Embed a batch of strings, preserving input order.
    ///
    /// # Errors
    ///
    /// Returns an error if backend inference fails.
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        match &self.inner {
            Inner::Onnx(m) => {
                let mut guard = m.lock().map_err(|_| VqtrsError::Poisoned)?;
                guard.embed(texts, None).map_err(backend_err)
            }
            #[cfg(feature = "qwen3")]
            Inner::Qwen3(m) => m.embed(texts).map_err(backend_err),
        }
    }

    /// The resolved model name this engine was loaded with.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Output vector dimensionality.
    #[must_use]
    pub const fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// The backend serving this model.
    #[must_use]
    pub const fn backend(&self) -> Backend {
        self.backend
    }

    /// Count tokens across `texts` with the model's tokenizer.
    ///
    /// Returns `None` when the tokenizer cannot be located in the cache, so the
    /// caller can fall back to an estimate.
    #[must_use]
    pub fn count_tokens(&self, texts: &[String]) -> Option<usize> {
        let tokenizer = self
            .tokenizer
            .get_or_init(|| load_tokenizer(&self.model))
            .as_ref()?;
        let mut total = 0;
        for text in texts {
            total += tokenizer.encode(text.as_str(), true).ok()?.len();
        }
        Some(total)
    }
}

/// Probe an ONNX model's output dimensionality with a one-token forward pass.
fn onnx_dimensions(model: &mut TextEmbedding) -> Result<usize> {
    let probe = model
        .embed(vec![" ".to_owned()], None)
        .map_err(backend_err)?;
    probe
        .first()
        .map(Vec::len)
        .ok_or_else(|| VqtrsError::Backend("dimension probe returned no vector".to_owned()))
}

/// Locate and load a model's `tokenizer.json` from the Hugging Face hub cache.
fn load_tokenizer(model: &str) -> Option<Arc<Tokenizer>> {
    let repo = hf_repo(model)?;
    let snapshots = model_cache_dir()
        .ok()?
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    for entry in std::fs::read_dir(snapshots).ok()?.flatten() {
        let path = entry.path().join("tokenizer.json");
        if path.exists() {
            let mut tokenizer = Tokenizer::from_file(&path).ok()?;
            // Count real input tokens, not padding/truncation artefacts.
            tokenizer.with_padding(None);
            tokenizer.with_truncation(None).ok()?;
            return Some(Arc::new(tokenizer));
        }
    }
    None
}

/// Map any backend error into [`VqtrsError::Backend`].
fn backend_err<E: std::fmt::Display>(err: E) -> VqtrsError {
    VqtrsError::Backend(err.to_string())
}
