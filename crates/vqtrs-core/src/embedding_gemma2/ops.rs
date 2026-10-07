//! Tensor operations with bit-exact CPU and device-native GPU paths.
//!
//! CPU reductions and transcendental operations use [`torch_cpu`] to match
//! the PyTorch reference's evaluation order. Non-CPU inputs stay on device
//! using candle tensor operations and fused NN kernels; their results are
//! numerically equivalent, not bit-identical. GPU activations and weights
//! are F32. RoPE constants are computed on CPU and uploaded once.

use candle_core::{D, DType, Device, Result, Tensor};

use super::torch_cpu;

/// `t * s` against a 0-d `f32` tensor, i.e. exactly one rounding (no `affine`).
pub fn mul_scalar(t: &Tensor, s: f32) -> Result<Tensor> {
    t.broadcast_mul(&Tensor::new(s, t.device())?)
}

/// Contiguous host copy of `t` and its shape.
fn host(t: &Tensor) -> Result<(Vec<f32>, Vec<usize>)> {
    let dims = t.dims().to_vec();
    Ok((t.flatten_all()?.to_vec1::<f32>()?, dims))
}

fn from_host(data: Vec<f32>, dims: &[usize], dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(data, dims, dev)
}

/// Gemma RMS norm over the last dimension:
/// `x * pow(sum(x^2) / n + eps, -0.5) [* weight]`; `pow(., -0.5)` is the
/// reference kernel's `1 / sqrt(.)`.
pub fn rms_norm(x: &Tensor, weight: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    if !x.device().is_cpu() {
        let n = x.dim(D::Minus1)?;
        // Attention heads can be strided and have arbitrary leading dimensions.
        // The fused kernel operates on contiguous rows with F32 weights.
        let rows = x.contiguous()?.reshape(((), n))?;
        let weight = match weight {
            Some(weight) => weight.contiguous()?,
            None => Tensor::ones(n, DType::F32, x.device())?,
        };
        return candle_nn::ops::rms_norm(&rows, &weight, eps)?.reshape(x.shape());
    }
    let (mut data, dims) = host(x)?;
    let n = *dims
        .last()
        .ok_or_else(|| candle_core::Error::Msg("rms_norm of a scalar".into()))?;
    let weight = weight.map(Tensor::to_vec1::<f32>).transpose()?;
    let mut squares = vec![0.0_f32; n];
    for row in data.chunks_exact_mut(n) {
        for (sq, &v) in squares.iter_mut().zip(row.iter()) {
            *sq = v * v;
        }
        let mean = torch_cpu::sum_row(&squares) / n as f32;
        let inv = 1.0_f32 / (mean + eps).sqrt();
        for v in row.iter_mut() {
            *v *= inv;
        }
        if let Some(w) = &weight {
            for (v, &w) in row.iter_mut().zip(w) {
                *v *= w;
            }
        }
    }
    from_host(data, &dims, x.device())
}

/// `gelu(approximate="tanh")`.
pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    if !x.device().is_cpu() {
        return x.gelu();
    }
    let (mut data, dims) = host(x)?;
    torch_cpu::gelu_tanh(&mut data);
    from_host(data, &dims, x.device())
}

/// Softmax over the last dimension.
pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    if !x.device().is_cpu() {
        return candle_nn::ops::softmax_last_dim(&x.contiguous()?);
    }
    let (mut data, dims) = host(x)?;
    let n = *dims
        .last()
        .ok_or_else(|| candle_core::Error::Msg("softmax of a scalar".into()))?;
    for row in data.chunks_exact_mut(n) {
        torch_cpu::softmax_row(row);
    }
    from_host(data, &dims, x.device())
}

/// `cat(-x[..., d/2:], x[..., :d/2])`.
pub fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// `x * cos + rotate_half(x) * sin`, with `cos`/`sin` broadcast over heads.
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(cos)?
        .add(&rotate_half(x)?.broadcast_mul(sin)?)
}

/// RoPE tables of shape `(seq, head_dim)` for positions `0..seq`:
/// `inv_freq = 1 / base ** (arange(0, d, 2) / d)`, `freqs = pos * inv_freq`,
/// `emb = cat(freqs, freqs)`, then `cos(emb)`, `sin(emb)`.
pub fn rope_tables(
    seq: usize,
    head_dim: usize,
    base: f64,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut inv_freq: Vec<f32> = (0..half)
        .map(|i| (2 * i) as f32 / head_dim as f32)
        .collect();
    torch_cpu::pow_scalar_base(base as f32, &mut inv_freq);
    // `1.0 / t` is `t.reciprocal() * 1.0` in PyTorch; `* 1.0` is exact.
    for v in &mut inv_freq {
        *v = 1.0_f32 / *v;
    }
    let mut emb = Vec::with_capacity(seq * head_dim);
    for pos in 0..seq {
        let p = pos as f32;
        emb.extend(inv_freq.iter().map(|f| p * f));
        emb.extend_from_within(emb.len() - half..);
    }
    let mut cos = emb.clone();
    torch_cpu::cos(&mut cos);
    torch_cpu::sin(&mut emb);
    Ok((
        from_host(cos, &[seq, head_dim], dev)?,
        from_host(emb, &[seq, head_dim], dev)?,
    ))
}

/// `matmul(x, w^T)` for a 2-D `x` of shape `(n, in)` and `w` of `(out, in)`.
pub fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.matmul(&w.t()?)
}

/// sentence-transformers' `Pooling(mean)` + `Normalize` over `(seq, dim)`
/// token embeddings: `sum(0) / seq`, then `x / max(norm(x), 1e-12)`.
pub fn mean_pool_normalize(tokens: &Tensor) -> Result<Tensor> {
    let (seq, dim) = tokens.dims2()?;
    if !tokens.device().is_cpu() {
        let pooled = tokens.to_dtype(DType::F32)?.mean(0)?;
        let norm = pooled.sqr()?.sum_keepdim(0)?.sqrt()?;
        let eps = Tensor::full(1e-12_f32, norm.shape(), tokens.device())?;
        // A comparison/selection preserves NaN, unlike GPU maximum kernels.
        let denom = norm.lt(&eps)?.where_cond(&eps, &norm)?;
        return pooled.broadcast_div(&denom);
    }
    let (data, _) = host(&tokens.to_dtype(DType::F32)?)?;
    let count = seq as f32;
    let pooled: Vec<f32> = torch_cpu::sum_cols(&data, seq, dim)
        .into_iter()
        .map(|s| s / count)
        .collect();
    // `clamp_min(eps)`, which (unlike `f32::max`) propagates NaN.
    let norm = torch_cpu::l2_norm_row(&pooled);
    let eps = 1e-12_f64 as f32;
    let denom = if norm < eps { eps } else { norm };
    let normalized = pooled.into_iter().map(|v| v / denom).collect();
    from_host(normalized, &[dim], tokens.device())
}

#[cfg(all(
    test,
    any(feature = "embeddinggemma2-metal", feature = "embeddinggemma2-cuda")
))]
mod tests {
    use super::*;

    fn synthetic(count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.125)
            .collect()
    }

    // Downloads belong only in the test oracle, never in the native paths.
    fn check_close(cpu: &Tensor, metal: &Tensor, tolerance: f32) -> Result<()> {
        if metal.device().is_cpu() || cpu.dims() != metal.dims() || metal.dtype() != DType::F32 {
            candle_core::bail!("native output must preserve shape, F32 dtype and GPU residency");
        }
        let expected = cpu.flatten_all()?.to_vec1::<f32>()?;
        let actual = metal.flatten_all()?.to_vec1::<f32>()?;
        for (i, (&expected, &actual)) in expected.iter().zip(&actual).enumerate() {
            if expected.to_bits() == actual.to_bits() || (expected.is_nan() && actual.is_nan()) {
                continue;
            }
            if !expected.is_finite()
                || !actual.is_finite()
                || (expected - actual).abs() > tolerance * (1.0 + expected.abs())
            {
                candle_core::bail!("element {i}: CPU {expected}, Metal {actual}");
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU hardware and embeddinggemma2-metal or embeddinggemma2-cuda"]
    fn gpu_rms_norm_matches_cpu() -> Result<()> {
        let dev = crate::accel::embeddinggemma2_device();
        for dims in [vec![7], vec![3, 64], vec![2, 3, 257], vec![2, 3, 4, 64]] {
            let cpu = Tensor::from_vec(
                synthetic(dims.iter().product()),
                dims.as_slice(),
                &Device::Cpu,
            )?;
            // Exercise both contiguous input and the strided attention-head layout.
            let inputs = if dims.len() > 1 {
                vec![cpu.clone(), cpu.transpose(0, 1)?]
            } else {
                vec![cpu]
            };
            for cpu in inputs {
                let metal = cpu.to_device(&dev)?;
                let n = cpu.dim(D::Minus1)?;
                let weight = Tensor::from_vec(synthetic(n), n, &Device::Cpu)?;
                let metal_weight = weight.to_device(&dev)?;
                for (weight, metal_weight) in [(None, None), (Some(&weight), Some(&metal_weight))] {
                    check_close(
                        &rms_norm(&cpu, weight, 1e-6)?,
                        &rms_norm(&metal, metal_weight, 1e-6)?,
                        2e-5,
                    )?;
                }
            }
        }
        let cpu = Tensor::zeros((2, 7), DType::F32, &Device::Cpu)?;
        check_close(
            &rms_norm(&cpu, None, 1e-6)?,
            &rms_norm(&cpu.to_device(&dev)?, None, 1e-6)?,
            2e-5,
        )
    }

    #[test]
    #[ignore = "requires GPU hardware and embeddinggemma2-metal or embeddinggemma2-cuda"]
    fn gpu_gelu_tanh_matches_cpu() -> Result<()> {
        let dev = crate::accel::embeddinggemma2_device();
        let cpu = Tensor::from_vec(synthetic(3 * 257), (3, 257), &Device::Cpu)?.t()?;
        check_close(&gelu_tanh(&cpu)?, &gelu_tanh(&cpu.to_device(&dev)?)?, 2e-5)
    }

    #[test]
    #[ignore = "requires GPU hardware and embeddinggemma2-metal or embeddinggemma2-cuda"]
    fn gpu_softmax_matches_cpu() -> Result<()> {
        let dev = crate::accel::embeddinggemma2_device();
        for width in [7, 64, 257] {
            let values: Vec<f32> = synthetic(2 * 3 * 4 * width)
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    if i % 5 == 0 {
                        f32::NEG_INFINITY
                    } else {
                        v + 1000.0
                    }
                })
                .collect();
            let cpu = Tensor::from_vec(values, (2, 3, 4, width), &Device::Cpu)?.transpose(1, 2)?;
            check_close(
                &softmax_last_dim(&cpu)?,
                &softmax_last_dim(&cpu.to_device(&dev)?)?,
                2e-5,
            )?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU hardware and embeddinggemma2-metal or embeddinggemma2-cuda"]
    fn gpu_mean_pool_normalize_matches_cpu() -> Result<()> {
        let dev = crate::accel::embeddinggemma2_device();
        for seq in [1, 7, 33] {
            let cpu = Tensor::from_vec(synthetic(seq * 17), (17, seq), &Device::Cpu)?.t()?;
            check_close(
                &mean_pool_normalize(&cpu)?,
                &mean_pool_normalize(&cpu.to_device(&dev)?)?,
                2e-5,
            )?;
        }
        // Zero norms, norms below the clamp and NaN propagation.
        for value in [0.0_f32, 1e-14, f32::NAN] {
            let cpu = Tensor::full(value, (3, 7), &Device::Cpu)?;
            check_close(
                &mean_pool_normalize(&cpu)?,
                &mean_pool_normalize(&cpu.to_device(&dev)?)?,
                2e-5,
            )?;
        }
        Ok(())
    }
}
