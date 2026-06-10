//! Cross-encoder reranking over the fastembed/ONNX backend.

use std::sync::Mutex;

use fastembed::{RerankInitOptions, TextRerank};

use crate::cache::model_cache_dir;
use crate::catalog::resolve_reranker;
use crate::error::{Result, VqtrsError};

/// One scored document produced by [`Reranker::rerank`].
#[derive(Debug, Clone)]
pub struct Ranked {
    /// Index of this document in the original input list.
    pub index: usize,
    /// Relevance score; higher is more relevant.
    pub score: f32,
    /// The document text, present only when `return_documents` was set.
    pub document: Option<String>,
}

/// A loaded cross-encoder reranker.
///
/// The underlying fastembed reranker requires `&mut self` for inference, so it
/// is guarded by a [`Mutex`]; a `Reranker` is therefore safe to share across
/// threads behind an `Arc`, serialising rerank calls.
pub struct Reranker {
    model: String,
    inner: Mutex<TextRerank>,
}

impl std::fmt::Debug for Reranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reranker")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl Reranker {
    /// Load the reranker identified by `model` (a catalog code or variant name).
    ///
    /// # Errors
    ///
    /// Returns an error if the name cannot be resolved, the cache directory
    /// cannot be created, or the backend fails to initialise the model.
    pub fn load(model: &str) -> Result<Self> {
        let resolved = resolve_reranker(model)?;
        let opts = RerankInitOptions::new(resolved)
            .with_cache_dir(model_cache_dir()?)
            .with_show_download_progress(true)
            .with_execution_providers(crate::accel::execution_providers());
        let inner = TextRerank::try_new(opts).map_err(backend_err)?;
        Ok(Self {
            model: model.to_owned(),
            inner: Mutex::new(inner),
        })
    }

    /// Rank `documents` by relevance to `query`, returning results sorted by
    /// descending score.
    ///
    /// Set `return_documents` to include the document text in each result.
    /// `top_k` truncates to the highest-scoring `k` results when provided.
    ///
    /// # Errors
    ///
    /// Returns an error if the lock is poisoned or backend inference fails.
    pub fn rerank(
        &self,
        query: &str,
        documents: &[String],
        return_documents: bool,
        top_k: Option<usize>,
    ) -> Result<Vec<Ranked>> {
        let docs: Vec<&str> = documents.iter().map(String::as_str).collect();
        let mut results = {
            let mut guard = self.inner.lock().map_err(|_| VqtrsError::Poisoned)?;
            guard
                .rerank(query, &docs, return_documents, None)
                .map_err(backend_err)?
        };
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(k) = top_k {
            results.truncate(k);
        }
        Ok(results
            .into_iter()
            .map(|r| Ranked {
                index: r.index,
                score: r.score,
                document: r.document,
            })
            .collect())
    }

    /// The resolved reranker model name this instance was loaded with.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }
}

/// Map any backend error into [`VqtrsError::Backend`].
fn backend_err<E: std::fmt::Display>(err: E) -> VqtrsError {
    VqtrsError::Backend(err.to_string())
}
