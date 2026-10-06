//! EmbeddingGemma 2 (`google/embeddinggemma-2`) on vqtrs' own candle backend.
//!
//! A port of the transformers `embedding_gemma2` reference (v5.19.0) plus the
//! sentence-transformers head (mean pooling, L2 normalisation). Neither
//! fastembed nor candle-transformers ships this architecture.

mod config;
mod ops;
#[cfg(test)]
mod probe;
mod sleef;
mod sleef_rempitab;
mod text;
mod torch_cpu;

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::cache::model_cache_dir;
use crate::error::{Result, VqtrsError};
use config::Config;
use ops::mean_pool_normalize;
use text::TextModel;

/// Hugging Face repository id of the reference checkpoint.
pub const EMBEDDING_GEMMA2_REPO: &str = "google/embeddinggemma-2";

/// Output width of EmbeddingGemma 2 embeddings.
pub const EMBEDDING_GEMMA2_DIMENSIONS: usize = 768;

/// Raw outputs of one text forward pass, for diagnostics and parity checks.
#[derive(Debug, Clone)]
pub struct TextForward {
    /// Sequence length (number of tokens).
    pub seq: usize,
    /// Per-token embeddings, row-major `(seq, embedding_dim)`.
    pub token_embeddings: Vec<f32>,
    /// Output of every text layer, row-major `(layers, seq, hidden)`; empty
    /// unless requested.
    pub hidden_states: Vec<f32>,
    /// Mean-pooled, L2-normalised sentence embedding.
    pub embedding: Vec<f32>,
}

/// A loaded EmbeddingGemma 2 model.
pub struct EmbeddingGemma2 {
    text: TextModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl std::fmt::Debug for EmbeddingGemma2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGemma2").finish_non_exhaustive()
    }
}

impl EmbeddingGemma2 {
    /// Download (or reuse from the Hugging Face cache) and load `repo`.
    ///
    /// # Errors
    ///
    /// Returns an error if a file cannot be fetched or the checkpoint is not a
    /// valid EmbeddingGemma 2 model.
    pub fn from_hf(repo: &str) -> Result<Self> {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(model_cache_dir()?)
            .with_progress(true)
            .build()
            .map_err(backend)?;
        let repo = api.model(repo.to_owned());
        let fetch = |name: &str| repo.get(name).map_err(backend);
        Self::from_files(
            &fetch("config.json")?,
            &fetch("tokenizer.json")?,
            &fetch("model.safetensors")?,
        )
    }

    /// Load from a local directory holding `config.json`, `tokenizer.json`
    /// and `model.safetensors`.
    ///
    /// # Errors
    ///
    /// Returns an error if a file is missing or invalid.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::from_files(
            &dir.join("config.json"),
            &dir.join("tokenizer.json"),
            &dir.join("model.safetensors"),
        )
    }

    fn from_files(config: &PathBuf, tokenizer: &PathBuf, weights: &PathBuf) -> Result<Self> {
        let cfg: Config = serde_json::from_slice(&std::fs::read(config)?).map_err(backend)?;
        let tokenizer = Tokenizer::from_file(tokenizer).map_err(backend)?;
        let device = Device::Cpu;
        // Buffered (not mmapped) safetensors: the crate forbids `unsafe`. Only
        // the tensors a tower asks for are converted; the buffer is dropped
        // once the towers are built.
        let vb =
            VarBuilder::from_buffered_safetensors(std::fs::read(weights)?, DType::F32, &device)
                .map_err(backend)?;
        let text = TextModel::load(&vb.pp("language_model"), &cfg.text_config).map_err(backend)?;
        Ok(Self {
            text,
            tokenizer,
            device,
        })
    }

    /// Token ids for `text`, with the tokenizer's special tokens (`<bos>` … `<eos>`).
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, true).map_err(backend)?;
        Ok(encoding.get_ids().to_vec())
    }

    /// The sentence embedding of `text` (no prompt is added).
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or inference fails.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.tokenize(text)?;
        Ok(self.forward_text_ids(&ids, false)?.embedding)
    }

    /// Run the text tower on `ids`, optionally keeping every layer output.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    pub fn forward_text_ids(&self, ids: &[u32], keep_hidden_states: bool) -> Result<TextForward> {
        let run = || -> candle_core::Result<TextForward> {
            let ids = Tensor::new(ids, &self.device)?;
            let embeds = self.text.embed(&ids)?;
            let mut hidden = keep_hidden_states.then(Vec::new);
            let tokens = self.text.forward(&embeds, hidden.as_mut())?;
            let embedding = mean_pool_normalize(&tokens)?.to_vec1::<f32>()?;
            let hidden_states = match hidden {
                Some(layers) => Tensor::stack(&layers, 0)?.flatten_all()?.to_vec1::<f32>()?,
                None => Vec::new(),
            };
            Ok(TextForward {
                seq: tokens.dim(0)?,
                token_embeddings: tokens.flatten_all()?.to_vec1::<f32>()?,
                hidden_states,
                embedding,
            })
        };
        run().map_err(backend)
    }
}

fn backend(err: impl std::fmt::Display) -> VqtrsError {
    VqtrsError::Backend(err.to_string())
}
