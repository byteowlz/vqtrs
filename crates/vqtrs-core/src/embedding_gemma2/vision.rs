//! Gemma 4 vision tower (`gemma4_vision`, reused verbatim by EmbeddingGemma 2)
//! plus EmbeddingGemma 2's `embed_vision` projection into the text space.

use candle_core::{D, Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use super::ops::{apply_rope, gelu_tanh, linear, mul_scalar, rms_norm, softmax_last_dim};
use super::torch_cpu;

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: VisionRope,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionRope {
    pub rope_theta: f64,
}

struct Layer {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    input_ln: Tensor,
    post_attn_ln: Tensor,
    pre_ff_ln: Tensor,
    post_ff_ln: Tensor,
}

impl Layer {
    fn load(vb: &VarBuilder<'_>, c: &VisionConfig) -> Result<Self> {
        let (h, i, hd) = (c.hidden_size, c.intermediate_size, c.head_dim);
        let proj = c.num_attention_heads * hd;
        Ok(Self {
            q: vb.get((proj, h), "self_attn.q_proj.linear.weight")?,
            k: vb.get((proj, h), "self_attn.k_proj.linear.weight")?,
            v: vb.get((proj, h), "self_attn.v_proj.linear.weight")?,
            o: vb.get((h, proj), "self_attn.o_proj.linear.weight")?,
            q_norm: vb.get(hd, "self_attn.q_norm.weight")?,
            k_norm: vb.get(hd, "self_attn.k_norm.weight")?,
            gate: vb.get((i, h), "mlp.gate_proj.linear.weight")?,
            up: vb.get((i, h), "mlp.up_proj.linear.weight")?,
            down: vb.get((h, i), "mlp.down_proj.linear.weight")?,
            input_ln: vb.get(h, "input_layernorm.weight")?,
            post_attn_ln: vb.get(h, "post_attention_layernorm.weight")?,
            pre_ff_ln: vb.get(h, "pre_feedforward_layernorm.weight")?,
            post_ff_ln: vb.get(h, "post_feedforward_layernorm.weight")?,
        })
    }

    fn attention(
        &self,
        x: &Tensor,
        rope: &Rope2d,
        mask: Option<&Tensor>,
        nh: usize,
        eps: f32,
    ) -> Result<Tensor> {
        let (seq, _) = x.dims2()?;
        let hd = rope.cos.dim(D::Minus1)?;
        // (seq, heads, head_dim), normed and rotated before the head transpose.
        let heads = |w: &Tensor| linear(x, w)?.reshape((seq, nh, hd));
        let q = rope.apply(&rms_norm(&heads(&self.q)?, Some(&self.q_norm), eps)?)?;
        let k = rope.apply(&rms_norm(&heads(&self.k)?, Some(&self.k_norm), eps)?)?;
        let v = rms_norm(&heads(&self.v)?, None, eps)?;
        let (q, k, v) = (
            q.transpose(0, 1)?.contiguous()?,
            k.transpose(0, 1)?.contiguous()?,
            v.transpose(0, 1)?.contiguous()?,
        );
        let mut scores = q.matmul(&k.t()?)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }
        let out = softmax_last_dim(&scores)?
            .matmul(&v)?
            .transpose(0, 1)?
            .contiguous()?;
        linear(&out.reshape((seq, nh * hd))?, &self.o)
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &Rope2d,
        mask: Option<&Tensor>,
        nh: usize,
        eps: f32,
    ) -> Result<Tensor> {
        let attn = self.attention(
            &rms_norm(x, Some(&self.input_ln), eps)?,
            rope,
            mask,
            nh,
            eps,
        )?;
        let x = x.add(&rms_norm(&attn, Some(&self.post_attn_ln), eps)?)?;
        let h = rms_norm(&x, Some(&self.pre_ff_ln), eps)?;
        let mlp = linear(
            &gelu_tanh(&linear(&h, &self.gate)?)?.mul(&linear(&h, &self.up)?)?,
            &self.down,
        )?;
        x.add(&rms_norm(&mlp, Some(&self.post_ff_ln), eps)?)
    }
}

/// Axial 2-D RoPE tables, recomposed `H-H-W-W` over the head dimension.
struct Rope2d {
    cos: Tensor,
    sin: Tensor,
}

impl Rope2d {
    /// `freqs = pos[..., None] * inv_freq` over a `(seq, 2, head_dim / 4)`
    /// buffer (padding positions are `-1`, unclamped), `cos`/`sin` over that
    /// buffer, then `cat([f_x, f_x, f_y, f_y])`.
    fn new(positions: &[[i64; 2]], head_dim: usize, theta: f64, dev: &Device) -> Result<Self> {
        let spatial = head_dim / 2;
        let mut inv_freq: Vec<f32> = (0..spatial / 2)
            .map(|i| (2 * i) as f32 / spatial as f32)
            .collect();
        torch_cpu::pow_scalar_base(theta as f32, &mut inv_freq);
        for v in &mut inv_freq {
            *v = 1.0_f32 / *v;
        }
        let freqs: Vec<f32> = positions
            .iter()
            .flat_map(|p| {
                p.iter()
                    .flat_map(|&axis| inv_freq.iter().map(move |f| axis as f32 * f))
            })
            .collect();
        let (mut cos, mut sin) = (freqs.clone(), freqs);
        torch_cpu::cos(&mut cos);
        torch_cpu::sin(&mut sin);
        let q = inv_freq.len();
        let recompose = |t: &[f32]| -> Vec<f32> {
            t.chunks_exact(2 * q)
                .flat_map(|row| {
                    let (fx, fy) = row.split_at(q);
                    fx.iter()
                        .chain(fx)
                        .chain(fy)
                        .chain(fy)
                        .copied()
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        let seq = positions.len();
        Ok(Self {
            cos: Tensor::from_vec(recompose(&cos), (seq, head_dim), dev)?,
            sin: Tensor::from_vec(recompose(&sin), (seq, head_dim), dev)?,
        })
    }

    /// `apply_multidimensional_rope` on `(seq, heads, head_dim)`: rotate each
    /// half of the head dimension with its own half of the tables.
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let half = x.dim(D::Minus1)? / 2;
        let parts = (0..2)
            .map(|i| {
                let cos = self.cos.narrow(1, i * half, half)?.unsqueeze(1)?;
                let sin = self.sin.narrow(1, i * half, half)?.unsqueeze(1)?;
                apply_rope(&x.narrow(D::Minus1, i * half, half)?, &cos, &sin)
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&parts, D::Minus1)
    }
}

/// The vision tower and the `embed_vision` projection.
pub struct VisionModel {
    input_proj: Tensor,
    position_table: Tensor,
    layers: Vec<Layer>,
    embed_projection: Tensor,
    cfg: VisionConfig,
    eps: f32,
}

impl VisionModel {
    /// `vb` is the checkpoint root (`vision_tower.*`, `embed_vision.*`).
    pub fn load(vb: &VarBuilder<'_>, cfg: &VisionConfig, text_hidden: usize) -> Result<Self> {
        let tower = vb.pp("vision_tower");
        let h = cfg.hidden_size;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::load(&tower.pp(format!("encoder.layers.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input_proj: tower.get(
                (h, 3 * cfg.patch_size * cfg.patch_size),
                "patch_embedder.input_proj.weight",
            )?,
            position_table: tower.get(
                (2, cfg.position_embedding_size, h),
                "patch_embedder.position_embedding_table",
            )?,
            layers,
            embed_projection: vb
                .get((text_hidden, h), "embed_vision.embedding_projection.weight")?,
            eps: cfg.rms_norm_eps as f32,
            cfg: cfg.clone(),
        })
    }

    /// Image soft tokens `(num_soft_tokens, text_hidden)` from one image's
    /// padded patches `(max_patches, patch_dim)` and positions.
    pub fn forward(&self, pixel_values: &Tensor, positions: &[[i64; 2]]) -> Result<Tensor> {
        let dev = pixel_values.device();
        let padding: Vec<bool> = positions.iter().map(|p| p[0] == -1 && p[1] == -1).collect();
        let mut x = self.embed_patches(pixel_values, positions, &padding)?;
        let rope = Rope2d::new(
            positions,
            self.cfg.head_dim,
            self.cfg.rope_parameters.rope_theta,
            dev,
        )?;
        let mask = padding_mask(&padding, dev)?;
        for layer in &self.layers {
            x = layer.forward(
                &x,
                &rope,
                mask.as_ref(),
                self.cfg.num_attention_heads,
                self.eps,
            )?;
        }
        let pooled = self.pool(&x, positions, &padding)?;
        linear(&rms_norm(&pooled, None, self.eps)?, &self.embed_projection)
    }

    /// `input_proj(2 * (pv - 0.5)) + (x_emb + y_emb)`, positions zeroed on padding.
    fn embed_patches(
        &self,
        pixel_values: &Tensor,
        positions: &[[i64; 2]],
        padding: &[bool],
    ) -> Result<Tensor> {
        let dev = pixel_values.device();
        let shifted = mul_scalar(
            &pixel_values.broadcast_sub(&Tensor::new(0.5_f32, dev)?)?,
            2.0,
        )?;
        let proj = linear(&shifted, &self.input_proj)?;
        let index = |axis: usize| -> Result<Tensor> {
            let ids: Vec<u32> = positions.iter().map(|p| p[axis].max(0) as u32).collect();
            self.position_table
                .get(axis)?
                .index_select(&Tensor::new(ids, dev)?, 0)
        };
        let pos = index(0)?.add(&index(1)?)?;
        // `where(padding, 0.0, pos)`.
        proj.add(&zero_rows(&pos, padding)?)
    }

    /// `Gemma4VisionPooler`: zero padding rows, average `k x k` patch blocks
    /// via a one-hot GEMM, keep blocks that received weight, scale by
    /// `sqrt(hidden_size)`.
    fn pool(&self, x: &Tensor, positions: &[[i64; 2]], padding: &[bool]) -> Result<Tensor> {
        let dev = x.device();
        let (seq, _) = x.dims2()?;
        let k = self.cfg.pooling_kernel_size;
        let length = seq / (k * k);
        // `masked_fill(padding, 0.0)`.
        let x = zero_rows(x, padding)?;
        let clamped: Vec<[usize; 2]> = positions
            .iter()
            .map(|p| [p[0].max(0) as usize, p[1].max(0) as usize])
            .collect();
        let max_x = clamped.iter().map(|p| p[0]).max().unwrap_or(0) + 1;
        // `weights^T`, built contiguous: the reference copies it to contiguous
        // before the GEMM (its leading dimension is smaller than `k`).
        let w = 1.0_f32 / (k * k) as f32;
        let mut weights_t = vec![0.0_f32; length * seq];
        let mut used = vec![false; length];
        for (s, p) in clamped.iter().enumerate() {
            let block = p[0] / k + (max_x / k) * (p[1] / k);
            weights_t[block * seq + s] = w;
            used[block] = true;
        }
        let pooled = Tensor::from_vec(weights_t, (length, seq), dev)?.matmul(&x)?;
        let root = (self.cfg.hidden_size as f64).sqrt() as f32;
        let pooled = mul_scalar(&pooled, root)?;
        let rows: Vec<u32> = (0..length).filter(|&i| used[i]).map(|i| i as u32).collect();
        pooled.index_select(&Tensor::new(rows, dev)?, 0)
    }
}

/// Rows of `t` where `rows[i]` holds replaced by `+0.0` (a select, not a
/// multiply, so no `-0.0` appears).
fn zero_rows(t: &Tensor, rows: &[bool]) -> Result<Tensor> {
    let mask: Vec<u8> = rows.iter().map(|&r| u8::from(r)).collect();
    let mask = Tensor::from_vec(mask, (rows.len(), 1), t.device())?.broadcast_as(t.shape())?;
    mask.where_cond(&t.zeros_like()?, t)
}

/// Additive key-padding mask (`f32::MIN` on padded keys), or `None`.
fn padding_mask(padding: &[bool], dev: &Device) -> Result<Option<Tensor>> {
    if !padding.iter().any(|&p| p) {
        return Ok(None);
    }
    let row: Vec<f32> = padding
        .iter()
        .map(|&p| if p { f32::MIN } else { 0.0 })
        .collect();
    Tensor::from_vec(row, (1, padding.len()), dev).map(Some)
}
