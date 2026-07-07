//! Sparse and joint dense+sparse embedding engines.
//!
//! - [`SparseEngine`] runs sparse models (SPLADE, BGE-M3 sparse) and yields
//!   [`SparseVector`]s (term index → weight).
//! - [`M3Engine`] runs BGE-M3, which produces a dense vector and a sparse vector
//!   from a single forward pass ([`DenseSparse`]).

use std::sync::Mutex;

use fastembed::{Bgem3Embedding, Bgem3InitOptions, SparseInitOptions, SparseTextEmbedding};

use crate::cache::model_cache_dir;
use crate::catalog::{resolve_m3, resolve_sparse};
use crate::error::{Result, VqtrsError};

/// A sparse embedding: parallel arrays of term indices and their weights.
#[derive(Debug, Clone)]
pub struct SparseVector {
    /// Vocabulary indices of the non-zero terms.
    pub indices: Vec<usize>,
    /// Weight of each term, aligned with [`SparseVector::indices`].
    pub values: Vec<f32>,
}

/// A joint dense + sparse embedding from a single BGE-M3 pass.
#[derive(Debug, Clone)]
pub struct DenseSparse {
    /// The dense vector.
    pub dense: Vec<f32>,
    /// The sparse vector.
    pub sparse: SparseVector,
}

/// A loaded sparse-embedding model.
///
/// fastembed's sparse `embed` takes `&mut self`, so it is guarded by a
/// [`Mutex`]; inference is `&self` and the engine is `Send + Sync`.
pub struct SparseEngine {
    model: String,
    inner: Mutex<SparseTextEmbedding>,
}

impl std::fmt::Debug for SparseEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparseEngine")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl SparseEngine {
    /// Load the sparse model identified by `model` (a catalog code or variant).
    ///
    /// # Errors
    ///
    /// Returns an error if the name cannot be resolved, the cache directory
    /// cannot be created, or the backend fails to initialise the model.
    pub fn load(model: &str) -> Result<Self> {
        let resolved = resolve_sparse(model)?;
        let opts = SparseInitOptions::new(resolved)
            .with_cache_dir(model_cache_dir()?)
            .with_show_download_progress(true)
            .with_execution_providers(crate::accel::execution_providers());
        let inner = SparseTextEmbedding::try_new(opts).map_err(backend_err)?;
        Ok(Self {
            model: model.to_owned(),
            inner: Mutex::new(inner),
        })
    }

    /// Embed a single string into a sparse vector.
    ///
    /// # Errors
    ///
    /// Returns an error if backend inference fails or yields no vector.
    pub fn embed(&self, text: &str) -> Result<SparseVector> {
        let owned = [text.to_owned()];
        self.embed_batch(&owned)?
            .pop()
            .ok_or_else(|| VqtrsError::Backend("model returned no embedding".to_owned()))
    }

    /// Embed a batch of strings into sparse vectors, preserving input order.
    ///
    /// # Errors
    ///
    /// Returns an error if the lock is poisoned or backend inference fails.
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<SparseVector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let raw = {
            let mut guard = self.inner.lock().map_err(|_| VqtrsError::Poisoned)?;
            guard.embed(texts, None).map_err(backend_err)?
        };
        Ok(raw.into_iter().map(into_sparse).collect())
    }

    /// The resolved model name this engine was loaded with.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }
}

/// A loaded BGE-M3 joint dense+sparse model.
pub struct M3Engine {
    model: String,
    inner: Mutex<Bgem3Embedding>,
}

impl std::fmt::Debug for M3Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("M3Engine")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl M3Engine {
    /// Load the BGE-M3 model identified by `model` (a catalog code or variant).
    ///
    /// # Errors
    ///
    /// Returns an error if the name cannot be resolved, the cache directory
    /// cannot be created, or the backend fails to initialise the model.
    pub fn load(model: &str) -> Result<Self> {
        let resolved = resolve_m3(model)?;
        let opts = Bgem3InitOptions::new(resolved)
            .with_cache_dir(model_cache_dir()?)
            .with_show_download_progress(true)
            .with_execution_providers(crate::accel::execution_providers());
        let inner = Bgem3Embedding::try_new(opts).map_err(backend_err)?;
        Ok(Self {
            model: model.to_owned(),
            inner: Mutex::new(inner),
        })
    }

    /// Embed a single string into a joint dense+sparse vector.
    ///
    /// # Errors
    ///
    /// Returns an error if backend inference fails or yields no vector.
    pub fn embed(&self, text: &str) -> Result<DenseSparse> {
        let owned = [text.to_owned()];
        self.embed_batch(&owned)?
            .pop()
            .ok_or_else(|| VqtrsError::Backend("model returned no embedding".to_owned()))
    }

    /// Embed a batch of strings into joint dense+sparse vectors, preserving
    /// input order. The BGE-M3 `ColBERT` output is not exposed.
    ///
    /// # Errors
    ///
    /// Returns an error if the lock is poisoned or backend inference fails.
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<DenseSparse>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let output = {
            let mut guard = self.inner.lock().map_err(|_| VqtrsError::Poisoned)?;
            guard.embed(texts, None).map_err(backend_err)?
        };
        Ok(output
            .dense
            .into_iter()
            .zip(output.sparse)
            .map(|(dense, sparse)| DenseSparse {
                dense,
                sparse: into_sparse(sparse),
            })
            .collect())
    }

    /// The resolved model name this engine was loaded with.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }
}

/// Convert a fastembed sparse embedding into our [`SparseVector`].
fn into_sparse(raw: fastembed::SparseEmbedding) -> SparseVector {
    SparseVector {
        indices: raw.indices,
        values: raw.values,
    }
}

/// Map any backend error into [`VqtrsError::Backend`], preserving the full
/// cause chain so the underlying failure (e.g. a hf-hub transport error) is
/// not hidden behind fastembed's outer "Failed to retrieve …" context.
#[expect(
    clippy::needless_pass_by_value,
    reason = "map_err passes an owned anyhow::Error via FnOnce; borrowing would force closure wrappers at every call site"
)]
fn backend_err(err: anyhow::Error) -> VqtrsError {
    let mut msg = err.to_string();
    for cause in err.chain().skip(1) {
        msg.push_str(": ");
        msg.push_str(&cause.to_string());
    }
    VqtrsError::Backend(msg)
}
