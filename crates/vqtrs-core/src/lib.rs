//! `vqtrs-core` — a lean local embeddings and reranking engine.
//!
//! `vqtrs-core` does embeddings and reranking and nothing else. It wraps
//! [`fastembed`] with two backends behind one small API:
//!
//! - **ONNX** (via `ort`) for the catalog of small, fast sentence-transformer
//!   models.
//! - **Qwen3** (via the candle backend) for the SOTA open-weight
//!   Qwen3-Embedding models (0.6B / 4B / 8B).
//!
//! Plus cross-encoder reranking ([`Reranker`]) over the ONNX BGE/Jina rerankers.
//!
//! Inference takes `&self` (embeddings) or serialises through an internal lock
//! (reranking), so both [`Engine`] and [`Reranker`] are `Send + Sync` and meant
//! to be loaded once and shared behind an `Arc`.
//!
//! ```no_run
//! # fn main() -> vqtrs_core::Result<()> {
//! let engine = vqtrs_core::Engine::load("Qwen/Qwen3-Embedding-0.6B")?;
//! let vectors = engine.embed_batch(&["hello".to_owned(), "world".to_owned()])?;
//! assert_eq!(vectors.len(), 2);
//! # Ok(())
//! # }
//! ```

mod accel;
mod cache;
mod catalog;
mod embed;
mod error;
mod rerank;
mod sparse;

pub use catalog::{
    Backend, ModelInfo, QWEN3_MAX_LENGTH, RerankInfo, SparseInfo, dense_models, rerank_models,
    sparse_models,
};
pub use embed::Engine;
pub use error::{Result, VqtrsError};
pub use rerank::{Ranked, Reranker};
pub use sparse::{DenseSparse, M3Engine, SparseEngine, SparseVector};
