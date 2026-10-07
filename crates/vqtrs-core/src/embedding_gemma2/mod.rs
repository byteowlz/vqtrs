//! EmbeddingGemma 2 (`google/embeddinggemma-2`) on vqtrs' own candle backend.
//!
//! A port of the transformers `embedding_gemma2` reference (v5.19.0) plus the
//! sentence-transformers head (mean pooling, L2 normalisation). Neither
//! fastembed nor candle-transformers ships this architecture.
//!
//! On aarch64 macOS the tested outputs are bit-identical to the PyTorch
//! reference (`f32`, CPU; text/vision eager attention and audio boolean masks):
//! see `ops` and `torch_cpu`. This is not a cross-platform bit-identity guarantee.

mod audio;
mod config;
mod image;
mod jpeg;
mod layer_norm;
mod mel;
mod ops;
mod pocketfft;
#[cfg(test)]
mod probe;
mod sleef;
mod sleef_rempitab;
mod text;
mod torch_cpu;
mod video;
mod vision;

pub use video::{RgbFrame as Gemma2VideoFrame, VideoMetadata as Gemma2VideoMetadata};

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::cache::model_cache_dir;
use crate::error::{Result, VqtrsError};
use audio::AudioModel;
use config::Config;
use image::ImageSettings;
use mel::{MelFrontEnd, MelSettings};
use ops::mean_pool_normalize;
use text::TextModel;
use vision::VisionModel;

/// Hugging Face repository id of the reference checkpoint.
pub const EMBEDDING_GEMMA2_REPO: &str = "google/embeddinggemma-2";

/// Output width of EmbeddingGemma 2 embeddings.
pub const EMBEDDING_GEMMA2_DIMENSIONS: usize = 768;

/// Practical limit for the joint sequence, including special/media tokens.
///
/// This backend materializes attention matrices; it does not implement the
/// checkpoint's full 262,144-token context efficiently.
pub const EMBEDDING_GEMMA2_MAX_TOKENS: usize = 8192;

/// One input to embed.
#[derive(Debug, Clone, Copy)]
pub enum Gemma2Input<'a> {
    /// Text, embedded as given (no prompt is added).
    Text(&'a str),
    /// An encoded image (PNG, JPEG, ...).
    Image(&'a [u8]),
    /// An encoded WAV clip at 16 kHz (mono or stereo).
    /// Clips longer than 30 seconds are truncated, as in the reference.
    Audio(&'a [u8]),
    /// An encoded MP4/MOV or Matroska/WebM video; requires local `FFmpeg` tools.
    /// Uses 1-fps, at-most-32-frame visual-only prompts and checkpoint patch budgets.
    Video(&'a [u8]),
    /// Trusted decoded RGB frames, with optional timing for frame sampling.
    /// Without timing, keeps every frame up to the uniform 32-frame budget.
    VideoFrames {
        /// Frames in source temporal order, all sharing dimensions.
        frames: &'a [Gemma2VideoFrame],
        /// Optional source frame count, rate and duration.
        metadata: Option<Gemma2VideoMetadata>,
    },
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
    /// Frame-major video patches `(frames, max_patches, patch_dim)`.
    pub video_pixel_values: Vec<f32>,
    /// Frame-major video patch positions `(frames, max_patches, 2)`.
    pub video_positions: Vec<[i64; 2]>,
    /// Frame-major video soft tokens in text space.
    pub video_features: Vec<f32>,
    /// Original frame indices chosen by the video processor.
    pub sampled_frame_indices: Vec<usize>,
    /// Actual soft-token count per sampled frame (within the checkpoint budget).
    pub video_num_soft_tokens: usize,
    /// Audio processor output `(frames, mel_bins)`, empty for other inputs.
    pub input_features: Vec<f32>,
    /// Audio processor's frame-validity mask.
    pub input_features_mask: Vec<bool>,
    /// Valid audio soft tokens in text space `(num_soft_tokens, hidden)`.
    pub audio_features: Vec<f32>,
    /// Subsample projection and audio layer outputs; empty unless requested.
    pub audio_hidden_states: Vec<Vec<f32>>,
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
    audio: Option<AudioModel>,
    mel: MelFrontEnd,
    image_settings: ImageSettings,
    video_settings: ImageSettings,
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
        Self::from_hf_on(repo, &crate::accel::embeddinggemma2_device())
    }

    /// Fetch and load a checkpoint on an explicitly selected device.
    ///
    /// # Errors
    ///
    /// Returns an error if fetching, checkpoint parsing or device loading fails.
    pub fn from_hf_on(repo: &str, device: &Device) -> Result<Self> {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(model_cache_dir()?)
            .with_progress(true)
            .build()
            .map_err(backend)?;
        let repo = api.model(repo.to_owned());
        let fetch = |name: &str| repo.get(name).map_err(backend);
        Self::load(
            &CheckpointFiles {
                config: fetch("config.json")?,
                tokenizer: fetch("tokenizer.json")?,
                weights: fetch("model.safetensors")?,
                processor: Some(fetch("processor_config.json")?),
            },
            device,
        )
    }

    /// Load from a local directory holding `config.json`, `tokenizer.json`,
    /// `model.safetensors` and optionally `processor_config.json`.
    ///
    /// # Errors
    ///
    /// Returns an error if a file is missing or invalid.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::from_dir_on(dir, &Device::Cpu)
    }

    /// Load a local checkpoint on an explicitly selected CPU, CUDA or Metal device.
    /// All towers use `f32`; GPU results are not promised bit-identical to CPU.
    ///
    /// # Errors
    ///
    /// Returns an error if a file is missing, invalid or cannot be loaded on the device.
    pub fn from_dir_on(dir: &Path, device: &Device) -> Result<Self> {
        let processor = dir.join("processor_config.json");
        Self::load(
            &CheckpointFiles {
                config: dir.join("config.json"),
                tokenizer: dir.join("tokenizer.json"),
                weights: dir.join("model.safetensors"),
                processor: processor.exists().then_some(processor),
            },
            device,
        )
    }

    /// The actual device holding all three model towers.
    #[must_use]
    pub const fn device(&self) -> &Device {
        &self.device
    }

    fn load(files: &CheckpointFiles, device: &Device) -> Result<Self> {
        let cfg: Config =
            serde_json::from_slice(&std::fs::read(&files.config)?).map_err(backend)?;
        let tokenizer = Tokenizer::from_file(&files.tokenizer).map_err(backend)?;
        let defaults = ImageSettings::default();
        let video_defaults = ImageSettings {
            max_soft_tokens: 70,
            ..defaults
        };
        let (image_settings, video_settings) = match &files.processor {
            Some(path) => {
                let bytes = std::fs::read(path)?;
                (
                    processor_settings(&bytes, "image_processor", defaults)?,
                    processor_settings(&bytes, "video_processor", video_defaults)?,
                )
            }
            None => (defaults, video_defaults),
        };
        // Buffered (not mmapped) safetensors: the crate forbids `unsafe`. Only
        // the tensors a tower asks for are converted; the buffer is dropped
        // once the towers are built.
        let vb = VarBuilder::from_buffered_safetensors(
            std::fs::read(&files.weights)?,
            DType::F32,
            device,
        )
        .map_err(backend)?;
        let text = TextModel::load(&vb.pp("language_model"), &cfg.text).map_err(backend)?;
        let vision = cfg
            .vision
            .as_ref()
            .map(|vc| VisionModel::load(&vb, vc, cfg.text.hidden_size))
            .transpose()
            .map_err(backend)?;
        let audio = cfg
            .audio
            .as_ref()
            .map(|ac| AudioModel::load(&vb, ac, cfg.text.hidden_size))
            .transpose()
            .map_err(backend)?;
        // CPU preprocessing deliberately retains f64 NumPy-compatible mel GEMM;
        // Metal does not support f64. Only completed features are uploaded.
        let mel = MelFrontEnd::new(MelSettings::default(), &Device::Cpu).map_err(backend)?;
        Ok(Self {
            text,
            vision,
            audio,
            mel,
            image_settings,
            video_settings,
            tokenizer,
            cfg,
            device: device.clone(),
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
        Ok(self.run(input, false, false)?.embedding)
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
        self.run(input, keep_hidden_states, true)
    }

    fn run(
        &self,
        input: Gemma2Input<'_>,
        keep_hidden_states: bool,
        capture: bool,
    ) -> Result<Forward> {
        match input {
            Gemma2Input::Text(text) => {
                let ids = self.tokenize(text)?;
                self.forward_sequence(&ids, None, keep_hidden_states, Forward::default(), capture)
            }
            Gemma2Input::Image(bytes) => self.forward_image(bytes, keep_hidden_states, capture),
            Gemma2Input::Audio(bytes) => self.forward_audio(bytes, keep_hidden_states, capture),
            Gemma2Input::Video(bytes) => self.forward_video(
                video::decode_container(bytes, &self.video_settings)
                    .map_err(VqtrsError::Backend)?,
                keep_hidden_states,
                capture,
            ),
            Gemma2Input::VideoFrames { frames, metadata } => self.forward_video(
                video::preprocess_decoded(frames, metadata, &self.video_settings)
                    .map_err(VqtrsError::Backend)?,
                keep_hidden_states,
                capture,
            ),
        }
    }

    fn forward_audio(
        &self,
        bytes: &[u8],
        keep_hidden_states: bool,
        capture: bool,
    ) -> Result<Forward> {
        let audio = self
            .audio
            .as_ref()
            .ok_or_else(|| VqtrsError::Backend("checkpoint has no audio tower".into()))?;
        let (wave, rate) = mel::decode_wav(bytes).map_err(VqtrsError::Backend)?;
        let settings = self.mel.settings();
        if rate != settings.sampling_rate {
            return Err(VqtrsError::Backend(format!(
                "expected {} Hz WAV, got {rate} Hz; resample before embedding",
                settings.sampling_rate
            )));
        }
        let mel = self.mel.features(&wave).map_err(backend)?;
        if mel.mask.iter().all(|&valid| !valid) {
            return Err(VqtrsError::Backend(
                "audio clip is too short to produce a valid frame".into(),
            ));
        }
        let input = Tensor::from_slice(
            &mel.features,
            (mel.mask.len(), settings.feature_size),
            &self.device,
        )
        .map_err(backend)?;
        let mut hidden = keep_hidden_states.then(Vec::new);
        let features = audio
            .forward(&input, &mel.mask, hidden.as_mut())
            .map_err(backend)?;
        let token = |id: u32| self.tokenizer.id_to_token(id).unwrap_or_default();
        let prompt = format!(
            "{}{}{}",
            token(self.cfg.boa_token_id),
            token(self.cfg.audio_token_id).repeat(features.dim(0).map_err(backend)?),
            token(self.cfg.eoa_token_id)
        );
        let ids = self.tokenize(&prompt)?;
        let out = Forward {
            input_features: mel.features,
            input_features_mask: mel.mask,
            audio_features: snapshot(&features, capture)?,
            audio_hidden_states: hidden
                .unwrap_or_default()
                .iter()
                .map(|tensor| snapshot(tensor, true))
                .collect::<Result<_>>()?,
            ..Forward::default()
        };
        self.forward_sequence(
            &ids,
            Some((&features, self.cfg.audio_token_id)),
            keep_hidden_states,
            out,
            capture,
        )
    }

    fn forward_image(
        &self,
        bytes: &[u8],
        keep_hidden_states: bool,
        capture: bool,
    ) -> Result<Forward> {
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
            image_features: snapshot(&features, capture)?,
            ..Forward::default()
        };
        self.forward_sequence(
            &ids,
            Some((&features, self.cfg.image_token_id)),
            keep_hidden_states,
            out,
            capture,
        )
    }

    fn forward_video(
        &self,
        prepared: video::PreparedVideo,
        keep_hidden_states: bool,
        capture: bool,
    ) -> Result<Forward> {
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| VqtrsError::Backend("checkpoint has no vision tower".into()))?;
        let features = prepared
            .frames
            .iter()
            .map(|frame| {
                let pv = Tensor::from_slice(
                    &frame.pixel_values,
                    (frame.positions.len(), self.video_settings.patch_dim()),
                    &self.device,
                )?;
                vision.forward(&pv, &frame.positions)
            })
            .collect::<candle_core::Result<Vec<_>>>()
            .map_err(backend)?;
        let features = Tensor::cat(&features, 0).map_err(backend)?;
        let token = |id| self.tokenizer.id_to_token(id).unwrap_or_default();
        let frame_prompt = format!(
            "{}{}{}",
            token(self.cfg.boi_token_id),
            token(self.cfg.video_token_id).repeat(prepared.num_soft_tokens),
            token(self.cfg.eoi_token_id)
        );
        let ids = self.tokenize(&frame_prompt.repeat(prepared.frames.len()))?;
        let out = Forward {
            video_pixel_values: if capture {
                prepared
                    .frames
                    .iter()
                    .flat_map(|frame| frame.pixel_values.iter().copied())
                    .collect()
            } else {
                Vec::new()
            },
            video_positions: if capture {
                prepared
                    .frames
                    .iter()
                    .flat_map(|frame| frame.positions.iter().copied())
                    .collect()
            } else {
                Vec::new()
            },
            video_features: snapshot(&features, capture)?,
            sampled_frame_indices: prepared.sampled_indices,
            video_num_soft_tokens: prepared.num_soft_tokens,
            ..Forward::default()
        };
        self.forward_sequence(
            &ids,
            Some((&features, self.cfg.video_token_id)),
            keep_hidden_states,
            out,
            capture,
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
        capture: bool,
    ) -> Result<Forward> {
        check_sequence_length(ids.len())?;
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
            if out.embedding.iter().any(|v| !v.is_finite()) {
                return Err(candle_core::Error::Msg(
                    "model produced a non-finite embedding".into(),
                ));
            }
            if capture {
                out.token_embeddings = tokens.flatten_all()?.to_vec1::<f32>()?;
            }
            if let Some(layers) = hidden {
                out.hidden_states = Tensor::stack(&layers, 0)?.flatten_all()?.to_vec1::<f32>()?;
            }
            Ok(out)
        };
        run().map_err(backend)
    }
}

/// Download a diagnostic tensor only when capture is requested.
fn snapshot(tensor: &Tensor, capture: bool) -> Result<Vec<f32>> {
    if !capture {
        return Ok(Vec::new());
    }
    tensor
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(backend)
}

fn check_sequence_length(count: usize) -> Result<()> {
    if count > EMBEDDING_GEMMA2_MAX_TOKENS {
        return Err(VqtrsError::Backend(format!(
            "EmbeddingGemma 2 input has {count} tokens; limit is {EMBEDDING_GEMMA2_MAX_TOKENS}; split the input"
        )));
    }
    Ok(())
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

/// Image/video settings from `processor_config.json`.
fn processor_settings(bytes: &[u8], key: &str, defaults: ImageSettings) -> Result<ImageSettings> {
    let json: serde_json::Value = serde_json::from_slice(bytes).map_err(backend)?;
    let ip = &json[key];
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

#[cfg(test)]
mod limits_tests {
    use super::*;

    #[test]
    fn rejects_sequences_above_cpu_limit() {
        assert!(check_sequence_length(EMBEDDING_GEMMA2_MAX_TOKENS).is_ok());
        assert!(check_sequence_length(EMBEDDING_GEMMA2_MAX_TOKENS + 1).is_err());
    }

    #[test]
    fn checkpoint_video_budget_overrides_class_default() {
        let bytes = br#"{"image_processor":{"max_soft_tokens":280},"video_processor":{"max_soft_tokens":140}}"#;
        let defaults = ImageSettings::default();
        let video = ImageSettings {
            max_soft_tokens: 70,
            ..defaults
        };
        assert_eq!(
            processor_settings(bytes, "image_processor", defaults)
                .unwrap()
                .max_soft_tokens,
            280
        );
        assert_eq!(
            processor_settings(bytes, "video_processor", video)
                .unwrap()
                .max_soft_tokens,
            140
        );
        assert_eq!(
            processor_settings(b"{}", "video_processor", video)
                .unwrap()
                .max_soft_tokens,
            70
        );
    }
}
