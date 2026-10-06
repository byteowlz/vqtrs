//! EmbeddingGemma 2 (`google/embeddinggemma-2`) on vqtrs' own candle backend.
//!
//! A port of the transformers `embedding_gemma2` reference (v5.19.0) plus the
//! sentence-transformers head (mean pooling, L2 normalisation). Neither
//! fastembed nor candle-transformers ships this architecture.
//!
//! On aarch64 macOS the output is bit-identical to the PyTorch reference
//! (eager attention, `f32`, CPU): see `ops` and `torch_cpu`.

mod config;
mod image;
mod jpeg;
mod ops;
#[cfg(test)]
mod probe;
mod sleef;
mod sleef_rempitab;
mod text;
mod torch_cpu;
mod vision;

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::cache::model_cache_dir;
use crate::error::{Result, VqtrsError};
use config::Config;
use image::ImageSettings;
use ops::mean_pool_normalize;
use text::TextModel;
use vision::VisionModel;

/// Hugging Face repository id of the reference checkpoint.
pub const EMBEDDING_GEMMA2_REPO: &str = "google/embeddinggemma-2";

/// Output width of EmbeddingGemma 2 embeddings.
pub const EMBEDDING_GEMMA2_DIMENSIONS: usize = 768;

/// One input to embed.
#[derive(Debug, Clone, Copy)]
pub enum Gemma2Input<'a> {
    /// Text, embedded as given (no prompt is added).
    Text(&'a str),
    /// An encoded image (PNG, JPEG, ...).
    Image(&'a [u8]),
}

/// Raw outputs of one forward pass, for diagnostics and parity checks.
#[derive(Debug, Clone, Default)]
pub struct Forward {
    /// Token ids fed to the text tower.
    pub input_ids: Vec<u32>,
    /// Image processor output `(max_patches, patch_dim)`, empty for text.
    pub pixel_values: Vec<f32>,
    /// Patch positions `(max_patches, 2)`, empty for text.
    pub image_positions: Vec<[i64; 2]>,
    /// Image soft tokens in text space `(num_soft_tokens, hidden)`.
    pub image_features: Vec<f32>,
    /// Per-token embeddings, row-major `(seq, embedding_dim)`.
    pub token_embeddings: Vec<f32>,
    /// Input embeddings then every text layer output but the last,
    /// row-major `(layers, seq, hidden)`; empty unless requested.
    pub hidden_states: Vec<f32>,
    /// Mean-pooled, L2-normalised embedding.
    pub embedding: Vec<f32>,
}

/// A loaded EmbeddingGemma 2 model.
pub struct EmbeddingGemma2 {
    text: TextModel,
    vision: Option<VisionModel>,
    image_settings: ImageSettings,
    tokenizer: Tokenizer,
    cfg: Config,
    device: Device,
}

impl std::fmt::Debug for EmbeddingGemma2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGemma2").finish_non_exhaustive()
    }
}

/// Files that make up a checkpoint.
struct CheckpointFiles {
    config: PathBuf,
    tokenizer: PathBuf,
    weights: PathBuf,
    processor: Option<PathBuf>,
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
        Self::load(&CheckpointFiles {
            config: fetch("config.json")?,
            tokenizer: fetch("tokenizer.json")?,
            weights: fetch("model.safetensors")?,
            processor: repo.get("processor_config.json").ok(),
        })
    }

    /// Load from a local directory holding `config.json`, `tokenizer.json`,
    /// `model.safetensors` and optionally `processor_config.json`.
    ///
    /// # Errors
    ///
    /// Returns an error if a file is missing or invalid.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let processor = dir.join("processor_config.json");
        Self::load(&CheckpointFiles {
            config: dir.join("config.json"),
            tokenizer: dir.join("tokenizer.json"),
            weights: dir.join("model.safetensors"),
            processor: processor.exists().then_some(processor),
        })
    }

    fn load(files: &CheckpointFiles) -> Result<Self> {
        let cfg: Config =
            serde_json::from_slice(&std::fs::read(&files.config)?).map_err(backend)?;
        let tokenizer = Tokenizer::from_file(&files.tokenizer).map_err(backend)?;
        let image_settings = match &files.processor {
            Some(path) => image_settings(&std::fs::read(path)?)?,
            None => ImageSettings::default(),
        };
        let device = Device::Cpu;
        // Buffered (not mmapped) safetensors: the crate forbids `unsafe`. Only
        // the tensors a tower asks for are converted; the buffer is dropped
        // once the towers are built.
        let vb = VarBuilder::from_buffered_safetensors(
            std::fs::read(&files.weights)?,
            DType::F32,
            &device,
        )
        .map_err(backend)?;
        let text = TextModel::load(&vb.pp("language_model"), &cfg.text).map_err(backend)?;
        let vision = cfg
            .vision
            .as_ref()
            .map(|vc| VisionModel::load(&vb, vc, cfg.text.hidden_size))
            .transpose()
            .map_err(backend)?;
        Ok(Self {
            text,
            vision,
            image_settings,
            tokenizer,
            cfg,
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

    /// The embedding of one input.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding, tokenization or inference fails.
    pub fn embed(&self, input: Gemma2Input<'_>) -> Result<Vec<f32>> {
        Ok(self.forward(input, false)?.embedding)
    }

    /// The embedding of `text` (no prompt is added).
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or inference fails.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(Gemma2Input::Text(text))
    }

    /// Run the model on one input, optionally keeping every text layer output.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding, tokenization or inference fails.
    pub fn forward(&self, input: Gemma2Input<'_>, keep_hidden_states: bool) -> Result<Forward> {
        match input {
            Gemma2Input::Text(text) => {
                let ids = self.tokenize(text)?;
                self.forward_sequence(&ids, None, keep_hidden_states, Forward::default())
            }
            Gemma2Input::Image(bytes) => self.forward_image(bytes, keep_hidden_states),
        }
    }

    fn forward_image(&self, bytes: &[u8], keep_hidden_states: bool) -> Result<Forward> {
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| VqtrsError::Backend("checkpoint has no vision tower".into()))?;
        let rgb = image::decode_rgb(bytes).map_err(VqtrsError::Backend)?;
        let patches = image::preprocess(&rgb, &self.image_settings).map_err(VqtrsError::Backend)?;
        let s = &self.image_settings;
        let features = (|| {
            let pv = Tensor::from_slice(
                &patches.pixel_values,
                (s.max_patches(), s.patch_dim()),
                &self.device,
            )?;
            vision.forward(&pv, &patches.positions)
        })()
        .map_err(backend)?;
        let token = |id: u32| self.tokenizer.id_to_token(id).unwrap_or_default();
        let prompt = format!(
            "{}{}{}",
            token(self.cfg.boi_token_id),
            token(self.cfg.image_token_id).repeat(patches.num_soft_tokens),
            token(self.cfg.eoi_token_id)
        );
        let ids = self.tokenize(&prompt)?;
        let out = Forward {
            pixel_values: patches.pixel_values,
            image_positions: patches.positions,
            image_features: features
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(backend)?,
            ..Forward::default()
        };
        self.forward_sequence(
            &ids,
            Some((&features, self.cfg.image_token_id)),
            keep_hidden_states,
            out,
        )
    }

    /// Embed `ids`, scatter `soft` tokens into the rows holding the
    /// placeholder id (which is embedded as the pad token), run the text
    /// tower and pool.
    fn forward_sequence(
        &self,
        ids: &[u32],
        soft: Option<(&Tensor, u32)>,
        keep_hidden_states: bool,
        mut out: Forward,
    ) -> Result<Forward> {
        let run = || -> candle_core::Result<Forward> {
            let pad = self.cfg.text.pad_token_id;
            let placeholder = soft.map(|(_, id)| id);
            let llm_ids: Vec<u32> = ids
                .iter()
                .map(|&id| if Some(id) == placeholder { pad } else { id })
                .collect();
            let mut embeds = self
                .text
                .embed(&Tensor::new(llm_ids.as_slice(), &self.device)?)?;
            if let Some((features, id)) = soft {
                embeds = masked_scatter(&embeds, ids, id, features)?;
            }
            let mut hidden = keep_hidden_states.then(Vec::new);
            let tokens = self.text.forward(&embeds, hidden.as_mut())?;
            out.input_ids = ids.to_vec();
            out.embedding = mean_pool_normalize(&tokens)?.to_vec1::<f32>()?;
            out.token_embeddings = tokens.flatten_all()?.to_vec1::<f32>()?;
            if let Some(layers) = hidden {
                out.hidden_states = Tensor::stack(&layers, 0)?.flatten_all()?.to_vec1::<f32>()?;
            }
            Ok(out)
        };
        run().map_err(backend)
    }
}

/// `inputs_embeds.masked_scatter(ids == id, features)`: the rows at the
/// placeholder positions take the feature rows in order.
fn masked_scatter(
    embeds: &Tensor,
    ids: &[u32],
    id: u32,
    features: &Tensor,
) -> candle_core::Result<Tensor> {
    let seq = ids.len();
    let wanted = ids.iter().filter(|&&t| t == id).count();
    if wanted != features.dim(0)? {
        return Err(candle_core::Error::Msg(format!(
            "{wanted} placeholder tokens but {} soft tokens",
            features.dim(0)?
        )));
    }
    let mut next = seq;
    let index: Vec<u32> = ids
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            if t == id {
                next += 1;
                (next - 1) as u32
            } else {
                i as u32
            }
        })
        .collect();
    Tensor::cat(&[embeds, features], 0)?.index_select(&Tensor::new(index, embeds.device())?, 0)
}

/// `image_processor` settings from `processor_config.json`.
fn image_settings(bytes: &[u8]) -> Result<ImageSettings> {
    let json: serde_json::Value = serde_json::from_slice(bytes).map_err(backend)?;
    let ip = &json["image_processor"];
    let defaults = ImageSettings::default();
    let get = |key: &str, default: usize| ip[key].as_u64().map_or(default, |v| v as usize);
    Ok(ImageSettings {
        patch_size: get("patch_size", defaults.patch_size),
        max_soft_tokens: get("max_soft_tokens", defaults.max_soft_tokens),
        pooling_kernel_size: get("pooling_kernel_size", defaults.pooling_kernel_size),
        rescale_factor: ip["rescale_factor"]
            .as_f64()
            .unwrap_or(defaults.rescale_factor),
    })
}

fn backend(err: impl std::fmt::Display) -> VqtrsError {
    VqtrsError::Backend(err.to_string())
}
