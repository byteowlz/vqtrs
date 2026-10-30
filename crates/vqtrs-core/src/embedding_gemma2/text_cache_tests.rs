//! Model-free exact CUDA positional-cache geometry tests; no checkpoint downloads.

use std::collections::HashMap;

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::super::config::{LayerType, RopeParams, TextConfig};
use super::super::ops::{apply_rope, folded_rope_tables, rope_tables};
use super::{ROPE_CACHE_LENGTH, RotaryTables, TextModel};

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

fn check_tables(cached: &RotaryTables, original: &(Tensor, Tensor)) -> Result<()> {
    check(&cached.cos, &original.0)?;
    let (seq, dim) = original.1.dims2()?;
    let mut expected = original.1.to_vec2::<f32>()?;
    if cached.folded {
        for row in &mut expected {
            for value in &mut row[..dim / 2] {
                *value = -*value;
            }
        }
    }
    let expected = Tensor::from_vec(expected.concat(), (seq, dim), cached.sin.device())?;
    check(&cached.sin, &expected)
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
                512,
                513,
                1024,
                2048,
                ROPE_CACHE_LENGTH - 1,
                ROPE_CACHE_LENGTH,
                ROPE_CACHE_LENGTH + 1,
            ] {
                let cached = model.rope(kind, seq, &device)?;
                ensure!(
                    cached.folded == (seq <= ROPE_CACHE_LENGTH && model.rope_cache.is_some()),
                    "folded layout eligibility"
                );
                let original = rope_tables(seq, dim, theta, &device)?;
                check_tables(&cached, &original)?;
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

#[test]
#[ignore = "requires CUDA; no checkpoint or downloads"]
fn cuda_folded_rope_keeps_separate_product_and_add_bits() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for (heads, seq, dim) in [
        (2, 3, 6),
        (2, 3, 8),
        (4, 17, 256),
        (2, 129, 512),
        (1, 513, 8),
    ] {
        let special = [
            0.0,
            -0.0,
            1e-40,
            -1e-40,
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_1234),
        ];
        let values = (0..heads * seq * dim)
            .map(|i| {
                if i % 17 < special.len() {
                    special[i % 17]
                } else {
                    ((i % 101) as f32).mul_add(0.125, -6.0)
                }
            })
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(values, (heads, seq, dim), &device)?;
        for theta in [10_000.0, 1_000_000.0] {
            let (cos, sin) = rope_tables(seq, dim, theta, &device)?;
            let expected = apply_rope(&input, &cos, &sin)?;
            let (cos, sin) = folded_rope_tables(seq, dim, theta, &device)?;
            let folded = RotaryTables {
                cos,
                sin,
                folded: true,
            };
            check(&expected, &folded.apply(&input)?)?;
        }
    }
    Ok(())
}
