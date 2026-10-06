//! Tensor operations reproducing the PyTorch reference bit for bit.
//!
//! GEMMs and single-rounding elementwise operations run on candle (their
//! results already match the reference exactly). Everything whose result
//! depends on evaluation order or on a transcendental implementation —
//! reductions, softmax, `tanh`, `sin`/`cos`, `pow` — runs through
//! [`torch_cpu`], which emulates the reference kernels.

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
    let (mut data, dims) = host(x)?;
    torch_cpu::gelu_tanh(&mut data);
    from_host(data, &dims, x.device())
}

/// Softmax over the last dimension.
pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
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
