//! EmbeddingGemma 2 text backbone: a bidirectional Gemma encoder with
//! per-layer embeddings (PLE) and a final projection to the embedding width.

use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::config::{LayerGeometry, LayerType, TextConfig};
use super::ops::{
    apply_rope, gelu_tanh, linear, mul_scalar, rms_norm, rope_tables, softmax_last_dim,
};

struct Attention {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    num_heads: usize,
    geo: LayerGeometry,
}

impl Attention {
    fn load(vb: &VarBuilder<'_>, cfg: &TextConfig, geo: LayerGeometry) -> Result<Self> {
        let (h, nh, hd, nkv) = (
            cfg.hidden_size,
            cfg.num_attention_heads,
            geo.head_dim,
            geo.num_kv_heads,
        );
        Ok(Self {
            q: vb.get((nh * hd, h), "q_proj.weight")?,
            k: vb.get((nkv * hd, h), "k_proj.weight")?,
            v: vb.get((nkv * hd, h), "v_proj.weight")?,
            o: vb.get((h, nh * hd), "o_proj.weight")?,
            q_norm: vb.get(hd, "q_norm.weight")?,
            k_norm: vb.get(hd, "k_norm.weight")?,
            num_heads: nh,
            geo,
        })
    }

    /// `x`: `(seq, hidden)`; `mask`: optional additive `(seq, seq)`.
    fn forward(
        &self,
        x: &Tensor,
        rope: &(Tensor, Tensor),
        mask: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        let seq = x.dim(0)?;
        let (hd, nkv) = (self.geo.head_dim, self.geo.num_kv_heads);
        let heads = |w: &Tensor, n: usize| linear(x, w)?.reshape((seq, n, hd))?.transpose(0, 1);
        let q = rms_norm(&heads(&self.q, self.num_heads)?, Some(&self.q_norm), eps)?;
        let k = rms_norm(&heads(&self.k, nkv)?, Some(&self.k_norm), eps)?;
        let v = rms_norm(&heads(&self.v, nkv)?, None, eps)?;
        let q = apply_rope(&q, &rope.0, &rope.1)?.contiguous()?;
        let k = repeat_kv(&apply_rope(&k, &rope.0, &rope.1)?, self.num_heads / nkv)?;
        let v = repeat_kv(&v, self.num_heads / nkv)?;
        // EmbeddingGemma 2 attends with scaling = 1.0. `k^T` stays a transposed
        // view, so the GEMM is called with `transa = 'T'` like the reference.
        let mut scores = q.matmul(&k.t()?)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }
        let probs = softmax_last_dim(&scores)?;
        let out = probs.matmul(&v)?.transpose(0, 1)?.contiguous()?;
        linear(&out.reshape((seq, self.num_heads * hd))?, &self.o)
    }
}

/// `(kv_heads, seq, d)` -> `(kv_heads * rep, seq, d)`, each head repeated in place.
pub(super) fn repeat_kv(x: &Tensor, rep: usize) -> Result<Tensor> {
    if rep == 1 {
        return x.contiguous();
    }
    let (nkv, seq, d) = x.dims3()?;
    x.unsqueeze(1)?
        .broadcast_as((nkv, rep, seq, d))?
        .contiguous()?
        .reshape((nkv * rep, seq, d))
}

struct Layer {
    attn: Attention,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    input_ln: Tensor,
    post_attn_ln: Tensor,
    pre_ff_ln: Tensor,
    post_ff_ln: Tensor,
    ple_gate: Tensor,
    ple_proj: Tensor,
    ple_post_ln: Tensor,
    scalar: Tensor,
}

impl Layer {
    fn load(vb: &VarBuilder<'_>, cfg: &TextConfig, geo: LayerGeometry) -> Result<Self> {
        let (h, i, p) = (
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.hidden_size_per_layer_input,
        );
        Ok(Self {
            attn: Attention::load(&vb.pp("self_attn"), cfg, geo)?,
            gate: vb.get((i, h), "mlp.gate_proj.weight")?,
            up: vb.get((i, h), "mlp.up_proj.weight")?,
            down: vb.get((h, i), "mlp.down_proj.weight")?,
            input_ln: vb.get(h, "input_layernorm.weight")?,
            post_attn_ln: vb.get(h, "post_attention_layernorm.weight")?,
            pre_ff_ln: vb.get(h, "pre_feedforward_layernorm.weight")?,
            post_ff_ln: vb.get(h, "post_feedforward_layernorm.weight")?,
            ple_gate: vb.get((p, h), "ple_block.per_layer_input_gate.weight")?,
            ple_proj: vb.get((h, p), "ple_block.per_layer_projection.weight")?,
            ple_post_ln: vb.get(h, "ple_block.post_per_layer_input_norm.weight")?,
            scalar: vb.get(1, "layer_scalar")?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        ple_in: &Tensor,
        rope: &(Tensor, Tensor),
        mask: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        let attn = self
            .attn
            .forward(&rms_norm(x, Some(&self.input_ln), eps)?, rope, mask, eps)?;
        let x = x.add(&rms_norm(&attn, Some(&self.post_attn_ln), eps)?)?;

        let h = rms_norm(&x, Some(&self.pre_ff_ln), eps)?;
        let mlp = linear(
            &gelu_tanh(&linear(&h, &self.gate)?)?.mul(&linear(&h, &self.up)?)?,
            &self.down,
        )?;
        let x = x.add(&rms_norm(&mlp, Some(&self.post_ff_ln), eps)?)?;

        let g = gelu_tanh(&linear(&x, &self.ple_gate)?)?.mul(ple_in)?;
        let g = rms_norm(&linear(&g, &self.ple_proj)?, Some(&self.ple_post_ln), eps)?;
        x.add(&g)?.broadcast_mul(&self.scalar)
    }
}

/// The text backbone, producing per-token embeddings of width `embedding_dim`.
pub struct TextModel {
    embed_tokens: Tensor,
    embed_scale: f32,
    ple_projection: Tensor,
    ple_scale: f32,
    ple_norm: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    embedding_projection: Tensor,
    cfg: TextConfig,
    eps: f32,
}

impl TextModel {
    pub fn load(vb: &VarBuilder<'_>, cfg: &TextConfig) -> Result<Self> {
        let (h, l, p) = (
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.hidden_size_per_layer_input,
        );
        let layers = (0..l)
            .map(|i| {
                let geo = cfg.layer(i).ok_or_else(|| {
                    candle_core::Error::Msg(format!("no layer type for layer {i}"))
                })?;
                Layer::load(&vb.pp(format!("layers.{i}")), cfg, geo)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: vb.get((cfg.vocab_size, h), "embed_tokens.weight")?,
            // sqrt and pow(-0.5) in double precision, rounded to f32 like the reference buffers.
            embed_scale: (h as f64).sqrt() as f32,
            ple_projection: vb.get((l * p, h), "ple.per_layer_model_projection.weight")?,
            ple_scale: (h as f64).powf(-0.5) as f32,
            ple_norm: vb.get(p, "ple.per_layer_projection_norm.weight")?,
            layers,
            norm: vb.get(h, "norm.weight")?,
            embedding_projection: vb.get((cfg.embedding_dim, h), "embedding_projection.weight")?,
            eps: cfg.rms_norm_eps as f32,
            cfg: cfg.clone(),
        })
    }

    /// Scaled token embeddings `(seq, hidden)` for `ids` of shape `(seq,)`.
    pub fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        mul_scalar(&self.embed_tokens.index_select(ids, 0)?, self.embed_scale)
    }

    /// Per-token embeddings `(seq, embedding_dim)` from input embeddings
    /// `(seq, hidden)`. When `hidden_states` is given it receives the
    /// reference layout: the input embeddings, then the output of every layer
    /// but the last (for parity diagnostics).
    pub fn forward(
        &self,
        embeds: &Tensor,
        mut hidden_states: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        if let Some(out) = hidden_states.as_deref_mut() {
            out.push(embeds.clone());
        }
        let (seq, _) = embeds.dims2()?;
        let (l, p) = (
            self.cfg.num_hidden_layers,
            self.cfg.hidden_size_per_layer_input,
        );
        let ple = mul_scalar(&linear(embeds, &self.ple_projection)?, self.ple_scale)?
            .reshape((seq, l, p))?;
        let ple = rms_norm(&ple, Some(&self.ple_norm), self.eps)?;

        let dev = embeds.device();
        let rope_sliding = self.rope(LayerType::SlidingAttention, seq, dev)?;
        let rope_full = self.rope(LayerType::FullAttention, seq, dev)?;
        let window_mask = sliding_window_mask(seq, self.cfg.sliding_window, dev)?;

        let mut x = embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let (rope, mask) = match layer.attn.geo.layer_type {
                LayerType::SlidingAttention => (&rope_sliding, window_mask.as_ref()),
                LayerType::FullAttention => (&rope_full, None),
            };
            let ple_in = ple.narrow(1, i, 1)?.squeeze(1)?;
            x = layer.forward(&x, &ple_in, rope, mask, self.eps)?;
            if i + 1 < self.layers.len()
                && let Some(out) = hidden_states.as_deref_mut()
            {
                out.push(x.clone());
            }
        }
        linear(
            &rms_norm(&x, Some(&self.norm), self.eps)?,
            &self.embedding_projection,
        )
    }

    fn rope(&self, layer_type: LayerType, seq: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
        let missing = || candle_core::Error::Msg(format!("no RoPE parameters for {layer_type:?}"));
        let theta = self.cfg.rope_theta(layer_type).ok_or_else(missing)?;
        let head_dim = self.cfg.head_dim_for(layer_type).ok_or_else(missing)?;
        rope_tables(seq, head_dim, theta, dev)
    }
}

/// Additive bidirectional window mask (`|q - kv| <= window` allowed), or
/// `None` when the window already covers the whole sequence. Disallowed
/// positions get `f32::MIN`, the reference's `finfo(float32).min`.
pub(super) fn sliding_window_mask(
    seq: usize,
    window: usize,
    dev: &Device,
) -> Result<Option<Tensor>> {
    if seq <= window + 1 {
        return Ok(None);
    }
    let mask: Vec<f32> = (0..seq)
        .flat_map(|q| {
            (0..seq).map(move |k| {
                if q.abs_diff(k) <= window {
                    0.0
                } else {
                    f32::MIN
                }
            })
        })
        .collect();
    Tensor::from_vec(mask, (seq, seq), dev).map(Some)
}
