//! Error and result types for the vqtrs-core engine.

use thiserror::Error;

/// Errors produced by the embeddings and reranking engine.
#[derive(Debug, Error)]
pub enum VqtrsError {
    /// The underlying inference backend (fastembed/ort or candle) failed.
    #[error("backend error: {0}")]
    Backend(String),

    /// The requested model name could not be resolved to a known model.
    #[error("unknown model: {0}")]
    UnknownModel(String),

    /// A filesystem operation (e.g. preparing the cache directory) failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// A shared backend lock was poisoned by a panic in another thread.
    #[error("engine lock poisoned")]
    Poisoned,
}

/// Convenience result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, VqtrsError>;
