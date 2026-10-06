//! `config.json` of an EmbeddingGemma 2 checkpoint.

use std::collections::HashMap;

use serde::Deserialize;

/// Attention pattern of one text layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    /// Bidirectional window of radius `sliding_window`.
    SlidingAttention,
    /// Bidirectional attention over the whole sequence.
    FullAttention,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    pub rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayerOverride {
    pub head_dim: usize,
    pub num_key_value_heads: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    pub layer_types: Vec<LayerType>,
    pub hidden_size_per_layer_input: usize,
    pub embedding_dim: usize,
    pub pad_token_id: u32,
    pub rope_parameters: HashMap<String, RopeParams>,
    /// Per-layer overrides, keyed by zero-padded layer index (`"05"`).
    #[serde(default)]
    pub per_layer_config: HashMap<String, LayerOverride>,
}

/// Resolved attention geometry of one layer.
#[derive(Debug, Clone, Copy)]
pub struct LayerGeometry {
    pub layer_type: LayerType,
    pub head_dim: usize,
    pub num_kv_heads: usize,
}

impl TextConfig {
    /// The geometry of layer `idx`, with `per_layer_config` applied.
    pub fn layer(&self, idx: usize) -> Option<LayerGeometry> {
        let layer_type = *self.layer_types.get(idx)?;
        let over = self
            .per_layer_config
            .iter()
            .find(|(key, _)| key.parse::<usize>().ok() == Some(idx))
            .map(|(_, v)| v);
        Some(LayerGeometry {
            layer_type,
            head_dim: over.map_or(self.head_dim, |o| o.head_dim),
            num_kv_heads: over.map_or(self.num_key_value_heads, |o| o.num_key_value_heads),
        })
    }

    /// RoPE base for a layer type.
    pub fn rope_theta(&self, layer_type: LayerType) -> Option<f64> {
        let key = match layer_type {
            LayerType::SlidingAttention => "sliding_attention",
            LayerType::FullAttention => "full_attention",
        };
        self.rope_parameters.get(key).map(|p| p.rope_theta)
    }

    /// Head dimension used by every layer of `layer_type` (they share RoPE tables).
    pub fn head_dim_for(&self, layer_type: LayerType) -> Option<usize> {
        (0..self.num_hidden_layers)
            .filter_map(|i| self.layer(i))
            .find(|g| g.layer_type == layer_type)
            .map(|g| g.head_dim)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(rename = "text_config")]
    pub text: TextConfig,
    #[serde(default, rename = "vision_config")]
    pub vision: Option<super::vision::VisionConfig>,
    pub image_token_id: u32,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
}
