//! Model-free exact CUDA positional-cache geometry tests; no checkpoint downloads.

use std::collections::HashMap;

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::super::config::{LayerType, RopeParams, TextConfig};
use super::super::ops::rope_tables;
use super::{ROPE_CACHE_LENGTH, TextModel};

fn config(head_dim: usize) -> TextConfig {
    TextConfig {
        vocab_size: 4,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim,
        rms_norm_eps: 1e-6,
        sliding_window: 512,
        layer_types: vec![LayerType::SlidingAttention, LayerType::FullAttention],
        hidden_size_per_layer_input: 4,
        embedding_dim: 8,
        pad_token_id: 0,
        rope_parameters: HashMap::from([
            (
                "sliding_attention".into(),
                RopeParams {
                    rope_theta: 10_000.0,
                },
            ),
            (
                "full_attention".into(),
                RopeParams {
                    rope_theta: 1_000_000.0,
                },
            ),
        ]),
        per_layer_config: HashMap::new(),
    }
}

fn check(a: &Tensor, b: &Tensor) -> Result<()> {
    ensure!(
        a.dims() == b.dims() && a.dtype() == b.dtype(),
        "shape/dtype"
    );
    let a = a.flatten_all()?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_vec1::<f32>()?;
    ensure!(
        a.iter()
            .map(|v| v.to_bits())
            .eq(b.iter().map(|v| v.to_bits())),
        "changed positional bits"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA device; no models or downloads"]
fn cuda_positional_cache_preserves_prefixes_fallback_and_scalar_tail_geometries() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for dim in [6, 8, 256, 512, 520] {
        let cfg = config(dim);
        let model = TextModel::load(&VarBuilder::zeros(DType::F32, &device), &cfg)?;
        ensure!(
            model.rope_cache.is_some() == (dim % 8 == 0 && dim <= 512),
            "cache eligibility"
        );
        for kind in [LayerType::SlidingAttention, LayerType::FullAttention] {
            let theta = cfg
                .rope_theta(kind)
                .ok_or_else(|| anyhow::anyhow!("missing theta"))?;
            for seq in [
                1,
                2,
                3,
                7,
                16,
                31,
                127,
                128,
                255,
                256,
                511,
                ROPE_CACHE_LENGTH,
                ROPE_CACHE_LENGTH + 1,
            ] {
                let cached = model.rope(kind, seq, &device)?;
                let original = rope_tables(seq, dim, theta, &device)?;
                check(&cached.0, &original.0)?;
                check(&cached.1, &original.1)?;
            }
        }
    }
    let mut missing = config(8);
    missing.rope_parameters.remove("full_attention");
    let model = TextModel::load(&VarBuilder::zeros(DType::F32, &device), &missing)?;
    ensure!(
        model.rope_cache.is_none(),
        "missing parameters must keep original lazy error"
    );
    ensure!(model.rope(LayerType::FullAttention, 1, &device).is_err());
    let cpu = Device::Cpu;
    let cfg = config(8);
    let model = TextModel::load(&VarBuilder::zeros(DType::F32, &cpu), &cfg)?;
    ensure!(model.rope_cache.is_none(), "CPU must keep original path");
    Ok(())
}
