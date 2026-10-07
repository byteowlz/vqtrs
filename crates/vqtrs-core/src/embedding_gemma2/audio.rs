//! Gemma 4 conformer audio tower and EmbeddingGemma 2's audio projection.
//! One clip at a time, f32. CPU retains the exact PyTorch reference kernels;
//! GPU inference stays on device using native tensor kernels. Local attention
//! uses boolean SDPA mask semantics, not SDPA kernels.

use candle_core::{D, Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use super::layer_norm::layer_norm_rows;
use super::ops::{linear, mul_scalar, rms_norm, softmax_last_dim};
use super::{sleef, torch_cpu};

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub hidden_act: String,
    pub subsampling_conv_channels: [usize; 2],
    pub conv_kernel_size: usize,
    pub residual_weight: f32,
    pub attention_chunk_size: usize,
    pub attention_context_left: usize,
    pub attention_context_right: usize,
    pub attention_logit_cap: f32,
    pub attention_invalid_logits_value: f32,
    pub use_clipped_linears: bool,
    pub rms_norm_eps: f32,
    pub gradient_clipping: f32,
    pub output_proj_dims: usize,
}

fn map(x: &Tensor, f: impl Fn(&mut [f32])) -> Result<Tensor> {
    let mut data = x.flatten_all()?.to_vec1::<f32>()?;
    f(&mut data);
    Tensor::from_vec(data, x.dims(), x.device())
}

fn clamp(x: &Tensor, min: f32, max: f32) -> Result<Tensor> {
    if !x.device().is_cpu() {
        return x.clamp(min, max);
    }
    map(x, |data| {
        for v in data {
            *v = v.max(min).min(max);
        }
    })
}

fn silu(x: &Tensor) -> Result<Tensor> {
    if !x.device().is_cpu() {
        return candle_nn::ops::silu(x);
    }
    map(x, |data| {
        torch_cpu::vectorized_loop(
            data,
            |v| v.map(|x| x / (1.0 + sleef::expf(-x))),
            |x| x / (1.0 + (-x).exp()),
        );
    })
}

struct ClippedLinear {
    weight: Tensor,
    input: (f32, f32),
    output: (f32, f32),
}

impl ClippedLinear {
    fn load(vb: &VarBuilder<'_>, input: usize, output: usize, clipped: bool) -> Result<Self> {
        let scalar = |name| vb.get((), name)?.to_scalar::<f32>();
        Ok(Self {
            weight: vb.get((output, input), "linear.weight")?,
            input: if clipped {
                (scalar("input_min")?, scalar("input_max")?)
            } else {
                (f32::NEG_INFINITY, f32::INFINITY)
            },
            output: if clipped {
                (scalar("output_min")?, scalar("output_max")?)
            } else {
                (f32::NEG_INFINITY, f32::INFINITY)
            },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        clamp(
            &linear(&clamp(x, self.input.0, self.input.1)?, &self.weight)?,
            self.output.0,
            self.output.1,
        )
    }
}

struct SubsampleLayer {
    weight: Tensor,
    norm: Vec<f32>,
    native_norm: Tensor,
    eps: f32,
}

impl SubsampleLayer {
    fn load(vb: &VarBuilder<'_>, input: usize, output: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: vb.get((output, input, 3, 3), "conv.weight")?,
            norm: vb.get(output, "norm.weight")?.to_vec1::<f32>()?,
            native_norm: vb.get(output, "norm.weight")?,
            eps,
        })
    }

    /// Input/output is `(channels, frames, frequency)`. Torch's slow CPU
    /// Conv2d uses contiguous im2col then `weight @ columns`.
    fn forward(&self, x: &Tensor, mask: &[bool]) -> Result<(Tensor, Vec<bool>)> {
        if !x.device().is_cpu() {
            return self.forward_native(x, mask);
        }
        let (channels, time, freq) = x.dims3()?;
        let (ot, of) = (time.div_ceil(2), freq.div_ceil(2));
        let positions = ot * of;
        let data = x.flatten_all()?.to_vec1::<f32>()?;
        let mut col = vec![0.0_f32; channels * 9 * positions];
        for c in 0..channels {
            for kt in 0..3 {
                for kf in 0..3 {
                    let row = (c * 9 + kt * 3 + kf) * positions;
                    for t in 0..ot {
                        for f in 0..of {
                            let it = (2 * t + kt).checked_sub(1);
                            let iff = (2 * f + kf).checked_sub(1);
                            if let (Some(it), Some(iff)) = (it, iff)
                                && it < time
                                && iff < freq
                            {
                                col[row + t * of + f] =
                                    data[(c * time + it) * freq + iff] * f32::from(mask[it]);
                            }
                        }
                    }
                }
            }
        }
        let dev = x.device();
        let out_c = self.norm.len();
        let col = Tensor::from_vec(col, (channels * 9, positions), dev)?;
        let y = self.weight.reshape((out_c, channels * 9))?.matmul(&col)?;
        let mut rows = y.t()?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        layer_norm_rows(&mut rows, out_c, &self.norm, self.eps);
        for v in &mut rows {
            *v = v.max(0.0);
        }
        let y = Tensor::from_vec(rows, (positions, out_c), dev)?
            .t()?
            .contiguous()?
            .reshape((out_c, ot, of))?;
        Ok((y, mask.iter().step_by(2).copied().collect()))
    }

    /// Gather a stride-two im2col on device, then GEMM and channel `LayerNorm`.
    /// Only the input validity mask originates on the host.
    fn forward_native(&self, x: &Tensor, mask: &[bool]) -> Result<(Tensor, Vec<bool>)> {
        let (channels, time, freq) = x.dims3()?;
        let (ot, of) = (time.div_ceil(2), freq.div_ceil(2));
        let positions = ot * of;
        let validity: Vec<f32> = mask.iter().map(|&v| f32::from(v)).collect();
        let x = x.broadcast_mul(&Tensor::from_vec(validity, (1, time, 1), x.device())?)?;
        let padded = x.pad_with_zeros(1, 1, 1)?.pad_with_zeros(2, 1, 1)?;
        let kernel = Tensor::arange(0_u32, 3_u32, x.device())?;
        let stride = Tensor::new(2_u32, x.device())?;
        let times = Tensor::arange(0_u32, ot as u32, x.device())?
            .broadcast_mul(&stride)?
            .reshape((1, 1, ot, 1))?
            .broadcast_add(&kernel.reshape((3, 1, 1, 1))?)?
            .broadcast_mul(&Tensor::new((freq + 2) as u32, x.device())?)?;
        let freqs = Tensor::arange(0_u32, of as u32, x.device())?
            .broadcast_mul(&stride)?
            .reshape((1, 1, 1, of))?
            .broadcast_add(&kernel.reshape((1, 3, 1, 1))?)?;
        let indices = times.broadcast_add(&freqs)?.flatten_all()?;
        let col = padded
            .reshape((channels, (time + 2) * (freq + 2)))?
            .index_select(&indices, 1)?
            .reshape((channels * 9, positions))?;
        let out_c = self.native_norm.dim(0)?;
        let rows = self
            .weight
            .reshape((out_c, channels * 9))?
            .matmul(&col)?
            .t()?
            .contiguous()?;
        let centered = rows.broadcast_sub(&rows.mean_keepdim(D::Minus1)?)?;
        let inv = (centered.sqr()?.mean_keepdim(D::Minus1)? + f64::from(self.eps))?
            .sqrt()?
            .recip()?;
        let y = centered
            .broadcast_mul(&inv)?
            .broadcast_mul(&self.native_norm)?
            .relu()?
            .t()?
            .contiguous()?
            .reshape((out_c, ot, of))?;
        Ok((y, mask.iter().step_by(2).copied().collect()))
    }
}

struct FeedForward {
    first: ClippedLinear,
    second: ClippedLinear,
    pre: Tensor,
    post: Tensor,
    scale: f32,
    clip: f32,
}

impl FeedForward {
    fn load(vb: &VarBuilder<'_>, c: &AudioConfig) -> Result<Self> {
        let h = c.hidden_size;
        Ok(Self {
            first: ClippedLinear::load(&vb.pp("ffw_layer_1"), h, h * 4, c.use_clipped_linears)?,
            second: ClippedLinear::load(&vb.pp("ffw_layer_2"), h * 4, h, c.use_clipped_linears)?,
            pre: vb.get(h, "pre_layer_norm.weight")?,
            post: vb.get(h, "post_layer_norm.weight")?,
            scale: c.residual_weight,
            clip: c.gradient_clipping,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // These FFN RMSNorms use the constructor's default eps, not config.
        let y = rms_norm(&clamp(x, -self.clip, self.clip)?, Some(&self.pre), 1e-6)?;
        let y = self.second.forward(&silu(&self.first.forward(&y)?)?)?;
        let y = rms_norm(&clamp(&y, -self.clip, self.clip)?, Some(&self.post), 1e-6)?;
        mul_scalar(&y, self.scale)?.add(x)
    }
}

struct LightConv {
    start: ClippedLinear,
    end: ClippedLinear,
    depthwise: Tensor,
    pre: Tensor,
    norm: Tensor,
    eps: f32,
    clip: f32,
}

impl LightConv {
    fn load(vb: &VarBuilder<'_>, c: &AudioConfig) -> Result<Self> {
        let h = c.hidden_size;
        Ok(Self {
            start: ClippedLinear::load(&vb.pp("linear_start"), h, h * 2, c.use_clipped_linears)?,
            end: ClippedLinear::load(&vb.pp("linear_end"), h, h, c.use_clipped_linears)?,
            depthwise: vb.get((h, 1, c.conv_kernel_size), "depthwise_conv1d.weight")?,
            pre: vb.get(h, "pre_layer_norm.weight")?,
            norm: vb.get(h, "conv_norm.weight")?,
            eps: c.rms_norm_eps,
            clip: c.gradient_clipping,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if !x.device().is_cpu() {
            return self.forward_native(x);
        }
        let (seq, h) = x.dims2()?;
        let y = self
            .start
            .forward(&rms_norm(x, Some(&self.pre), self.eps)?)?
            .to_vec2::<f32>()?;
        let mut glu = Vec::with_capacity(seq * h);
        for row in &y {
            glu.extend((0..h).map(|i| row[i] * (1.0 / (1.0 + sleef::expf(-row[h + i])))));
        }
        let y = self.depthwise(&glu, seq, h, x.device())?;
        let y = self.depthwise_norm(&clamp(&y, -self.clip, self.clip)?)?;
        self.end.forward(&silu(&y)?)?.add(x)
    }

    fn forward_native(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.dim(1)?;
        let y = self
            .start
            .forward(&rms_norm(x, Some(&self.pre), self.eps)?)?;
        let glu = y
            .narrow(1, 0, h)?
            .mul(&candle_nn::ops::sigmoid(&y.narrow(1, h, h)?)?)?;
        let y = self.depthwise_native(&glu)?;
        let y = self.depthwise_norm(&clamp(&y, -self.clip, self.clip)?)?;
        self.end.forward(&silu(&y)?)?.add(x)
    }

    /// Causal im2col followed by one batched GEMM for all channels, avoiding
    /// one convolution/GEMM launch per group. No activation leaves the device.
    fn depthwise_native(&self, glu: &Tensor) -> Result<Tensor> {
        let (seq, h) = glu.dims2()?;
        let kernel = self.depthwise.dim(2)?;
        let padded = glu.t()?.contiguous()?.pad_with_zeros(1, kernel - 1, 0)?;
        let indices = Tensor::arange(0_u32, kernel as u32, glu.device())?
            .reshape((kernel, 1))?
            .broadcast_add(&Tensor::arange(0_u32, seq as u32, glu.device())?.reshape((1, seq))?)?
            .flatten_all()?;
        let col = padded
            .index_select(&indices, 1)?
            .reshape((h, kernel, seq))?;
        self.depthwise
            .matmul(&col)?
            .reshape((h, seq))?
            .t()?
            .contiguous()
    }

    /// `RMSNorm` receives a transposed `(channels, time)` convolution result
    /// in PyTorch. Its squared values retain channel-major strides, so the
    /// reduction uses the outer (column-wise) cascade, not `sum_row`.
    fn depthwise_norm(&self, x: &Tensor) -> Result<Tensor> {
        if !x.device().is_cpu() {
            return rms_norm(x, Some(&self.norm), self.eps);
        }
        let (seq, h) = x.dims2()?;
        let mut data = x.flatten_all()?.to_vec1::<f32>()?;
        let mut squares = vec![0.0_f32; data.len()];
        for t in 0..seq {
            for c in 0..h {
                squares[c * seq + t] = data[t * h + c] * data[t * h + c];
            }
        }
        let sums = torch_cpu::sum_cols(&squares, h, seq);
        let weight = self.norm.to_vec1::<f32>()?;
        for (t, row) in data.chunks_exact_mut(h).enumerate() {
            let inv = 1.0_f32 / (sums[t] / h as f32 + self.eps).sqrt();
            for (c, v) in row.iter_mut().enumerate() {
                *v = (*v * inv) * weight[c];
            }
        }
        Tensor::from_vec(data, (seq, h), x.device())
    }

    fn depthwise(
        &self,
        glu: &[f32],
        seq: usize,
        h: usize,
        dev: &candle_core::Device,
    ) -> Result<Tensor> {
        let kernel = self.depthwise.dim(2)?;
        let mut groups = Vec::with_capacity(h);
        // Per-group slow Conv1d: a GEMM, not a scalar dot reduction.
        for c in 0..h {
            let mut col = vec![0.0_f32; kernel * seq];
            for k in 0..kernel {
                for t in 0..seq {
                    if let Some(it) = (t + k).checked_sub(kernel - 1) {
                        col[k * seq + t] = glu[it * h + c];
                    }
                }
            }
            let col = Tensor::from_vec(col, (kernel, seq), dev)?;
            let w = self.depthwise.narrow(0, c, 1)?.reshape((1, kernel))?;
            groups.push(w.matmul(&col)?);
        }
        Tensor::cat(&groups, 0)?.t()?.contiguous()
    }
}

struct Attention {
    q: ClippedLinear,
    k: ClippedLinear,
    v: ClippedLinear,
    post: ClippedLinear,
    relative: Tensor,
    scale: Vec<f32>,
    native_scale: Tensor,
}

/// Per-clip control tensors shared by every GPU attention layer.
struct AttentionLayout {
    context_indices: Tensor,
    valid: Tensor,
    seq: usize,
    blocks: usize,
    chunk: usize,
    context: usize,
}

impl AttentionLayout {
    fn new(mask: &[bool], c: &AudioConfig, dev: &Device) -> Result<Self> {
        let seq = mask.len();
        let chunk = c.attention_chunk_size;
        let blocks = seq.div_ceil(chunk);
        let past = c.attention_context_left - 1;
        let context = chunk + past + c.attention_context_right;
        // A sentinel row containing zeros represents both left and right pad.
        let indices: Vec<u32> = (0..blocks)
            .flat_map(|block| {
                (0..context).map(move |at| {
                    (block * chunk + at)
                        .checked_sub(past)
                        .filter(|&i| i < seq)
                        .map_or(seq as u32, |i| i as u32)
                })
            })
            .collect();
        let valid: Vec<u8> = (0..blocks)
            .flat_map(|block| {
                (0..chunk).flat_map(move |row| {
                    let qi = block * chunk + row;
                    (0..context).map(move |at| {
                        let ki = (block * chunk + at).checked_sub(past);
                        u8::from(
                            qi < seq
                                && ki.is_some_and(|ki| {
                                    ki < seq && mask[ki] && qi >= ki && qi - ki < past
                                }),
                        )
                    })
                })
            })
            .collect();
        Ok(Self {
            context_indices: Tensor::new(indices.as_slice(), dev)?,
            valid: Tensor::from_vec(valid, (1, blocks, chunk, context), dev)?,
            seq,
            blocks,
            chunk,
            context,
        })
    }

    fn blocked(&self, t: &Tensor, heads: usize, is_context: bool) -> Result<Tensor> {
        let hidden = t.dim(1)?;
        let width = if is_context { self.context } else { self.chunk };
        let t = if is_context {
            t.pad_with_zeros(0, 0, 1)?
                .index_select(&self.context_indices, 0)?
        } else {
            t.pad_with_zeros(0, 0, self.blocks * self.chunk - self.seq)?
        };
        t.reshape((self.blocks, width, heads, hidden / heads))?
            .permute((2, 0, 1, 3))?
            .contiguous()
    }

    /// Match PyTorch's pad/flatten/truncate relative shift without host data.
    fn relative_shift(&self, bd: &Tensor) -> Result<Tensor> {
        let heads = bd.dim(0)?;
        let plen = bd.dim(3)?;
        bd.pad_with_zeros(3, 0, self.context + 1 - plen)?
            .reshape((heads, self.blocks, self.chunk * (self.context + 1)))?
            .narrow(2, 0, self.chunk * self.context)?
            .contiguous()?
            .reshape((heads, self.blocks, self.chunk, self.context))
    }
}

impl Attention {
    fn load(vb: &VarBuilder<'_>, c: &AudioConfig) -> Result<Self> {
        let h = c.hidden_size;
        let proj = |name| ClippedLinear::load(&vb.pp(name), h, h, c.use_clipped_linears);
        let mut scale = vb
            .get(h / c.num_attention_heads, "per_dim_scale")?
            .to_vec1::<f32>()?;
        torch_cpu::vectorized_loop(
            &mut scale,
            |v| {
                v.map(|x| {
                    if x > 20.0 {
                        x
                    } else {
                        sleef::log1pf_positive(sleef::expf(x))
                    }
                })
            },
            |x| if x > 20.0 { x } else { x.exp().ln_1p() },
        );
        Ok(Self {
            q: proj("q_proj")?,
            k: proj("k_proj")?,
            v: proj("v_proj")?,
            post: proj("post")?,
            relative: vb.get((h, h), "relative_k_proj.weight")?,
            native_scale: Tensor::from_slice(&scale, (1, h / c.num_attention_heads), vb.device())?
                .repeat((1, c.num_attention_heads))?,
            scale,
        })
    }

    #[expect(
        clippy::many_single_char_names,
        reason = "Q/K/V names follow the attention equations"
    )]
    #[expect(
        clippy::imprecise_flops,
        reason = "the scale reproduces Python math.log(1 + math.e), not log1p"
    )]
    fn forward(&self, x: &Tensor, pos: &Tensor, mask: &[bool], c: &AudioConfig) -> Result<Tensor> {
        if !x.device().is_cpu() {
            return self.forward_native(x, pos, &AttentionLayout::new(mask, c, x.device())?, c);
        }
        let (seq, hidden) = x.dims2()?;
        let (nh, chunk) = (c.num_attention_heads, c.attention_chunk_size);
        let hd = hidden / nh;
        let past = c.attention_context_left - 1;
        let context = chunk + past + c.attention_context_right;
        let blocks = seq.div_ceil(chunk);
        let q_scale = (hd as f64).powf(-0.5) / std::f64::consts::LN_2;
        let k_scale = (1.0 + std::f64::consts::E).ln() / std::f64::consts::LN_2;
        let q = mul_scalar(&self.q.forward(x)?, q_scale as f32)?.broadcast_mul(
            &Tensor::from_slice(&self.scale, (1, 1, hd), x.device())?
                .reshape((1, hd))?
                .repeat((1, nh))?,
        )?;
        let k = mul_scalar(&self.k.forward(x)?, k_scale as f32)?;
        let v = self.v.forward(x)?;
        let blocked = |t: &Tensor, is_context: bool| -> Result<Tensor> {
            let values = t.to_vec2::<f32>()?;
            let width = if is_context { context } else { chunk };
            let mut out = vec![0.0_f32; nh * blocks * width * hd];
            for head in 0..nh {
                for block in 0..blocks {
                    for at in 0..width {
                        let idx =
                            (block * chunk + at).checked_sub(if is_context { past } else { 0 });
                        if let Some(idx) = idx.filter(|&i| i < seq) {
                            let dst = ((head * blocks + block) * width + at) * hd;
                            out[dst..dst + hd]
                                .copy_from_slice(&values[idx][head * hd..(head + 1) * hd]);
                        }
                    }
                }
            }
            Tensor::from_vec(out, (nh, blocks, width, hd), x.device())
        };
        let q = blocked(&q, false)?;
        let k = blocked(&k, true)?;
        let v = blocked(&v, true)?;
        let ac = q.matmul(&k.transpose(2, 3)?)?;
        let plen = pos.dim(0)?;
        let relative = linear(pos, &self.relative)?
            .reshape((plen, nh, hd))?
            .permute((1, 2, 0))?
            .contiguous()?;
        let bd = q
            .reshape((nh, blocks * chunk, hd))?
            .matmul(&relative)?
            .reshape((nh, blocks, chunk, plen))?;
        let bd = bd.flatten_all()?.to_vec1::<f32>()?;
        let mut scores = ac.flatten_all()?.to_vec1::<f32>()?;
        for head in 0..nh {
            for block in 0..blocks {
                for row in 0..chunk {
                    for at in 0..context {
                        // _rel_shift: zero-pad to context+1 then flatten,
                        // truncate and reshape back to (chunk, context).
                        let flat = row * context + at;
                        let (sr, sc) = (flat / (context + 1), flat % (context + 1));
                        let dest = ((head * blocks + block) * chunk + row) * context + at;
                        if sc < plen {
                            scores[dest] += bd[((head * blocks + block) * chunk + sr) * plen + sc];
                        }
                    }
                }
            }
        }
        torch_cpu::vectorized_loop(
            &mut scores,
            |v| v.map(|s| sleef::tanhf(s / c.attention_logit_cap) * c.attention_logit_cap),
            |s| (s / c.attention_logit_cap).tanh() * c.attention_logit_cap,
        );
        for head in 0..nh {
            for block in 0..blocks {
                for row in 0..chunk {
                    let qi = block * chunk + row;
                    for at in 0..context {
                        let ki = (block * chunk + at).checked_sub(past);
                        let valid = qi < seq
                            && ki.is_some_and(|ki| {
                                ki < seq && mask[ki] && qi >= ki && qi - ki < past
                            });
                        if !valid {
                            scores[((head * blocks + block) * chunk + row) * context + at] =
                                c.attention_invalid_logits_value;
                        }
                    }
                }
            }
        }
        let scores = Tensor::from_vec(scores, (nh, blocks, chunk, context), x.device())?;
        let out = softmax_last_dim(&scores)?
            .matmul(&v)?
            .permute((1, 2, 0, 3))?
            .contiguous()?
            .reshape((blocks * chunk, hidden))?
            .narrow(0, 0, seq)?
            .contiguous()?;
        self.post.forward(&out)
    }

    #[expect(
        clippy::imprecise_flops,
        reason = "the scale reproduces Python math.log(1 + math.e), not log1p"
    )]
    fn forward_native(
        &self,
        x: &Tensor,
        pos: &Tensor,
        layout: &AttentionLayout,
        c: &AudioConfig,
    ) -> Result<Tensor> {
        let hidden = x.dim(1)?;
        let heads = c.num_attention_heads;
        let hd = hidden / heads;
        let q_scale = (hd as f64).powf(-0.5) / std::f64::consts::LN_2;
        let k_scale = (1.0 + std::f64::consts::E).ln() / std::f64::consts::LN_2;
        let query =
            mul_scalar(&self.q.forward(x)?, q_scale as f32)?.broadcast_mul(&self.native_scale)?;
        let query = layout.blocked(&query, heads, false)?;
        let key = layout.blocked(
            &mul_scalar(&self.k.forward(x)?, k_scale as f32)?,
            heads,
            true,
        )?;
        let value = layout.blocked(&self.v.forward(x)?, heads, true)?;
        let ac = query.matmul(&key.transpose(2, 3)?)?;
        let plen = pos.dim(0)?;
        let relative = linear(pos, &self.relative)?
            .reshape((plen, heads, hd))?
            .permute((1, 2, 0))?
            .contiguous()?;
        let bd = query
            .reshape((heads, layout.blocks * layout.chunk, hd))?
            .matmul(&relative)?
            .reshape((heads, layout.blocks, layout.chunk, plen))?;
        let scores = ac.add(&layout.relative_shift(&bd)?)?;
        let scores = scores
            .broadcast_div(&Tensor::new(c.attention_logit_cap, x.device())?)?
            .tanh()?;
        let scores = mul_scalar(&scores, c.attention_logit_cap)?;
        let invalid = Tensor::full(c.attention_invalid_logits_value, scores.shape(), x.device())?;
        let scores = layout
            .valid
            .broadcast_as(scores.shape())?
            .where_cond(&scores, &invalid)?;
        let out = softmax_last_dim(&scores)?
            .matmul(&value)?
            .permute((1, 2, 0, 3))?
            .contiguous()?
            .reshape((layout.blocks * layout.chunk, hidden))?
            .narrow(0, 0, layout.seq)?
            .contiguous()?;
        self.post.forward(&out)
    }
}

struct Layer {
    ff1: FeedForward,
    ff2: FeedForward,
    conv: LightConv,
    attn: Attention,
    pre: Tensor,
    post: Tensor,
    out: Tensor,
}

impl Layer {
    fn load(vb: &VarBuilder<'_>, c: &AudioConfig) -> Result<Self> {
        let h = c.hidden_size;
        Ok(Self {
            ff1: FeedForward::load(&vb.pp("feed_forward1"), c)?,
            ff2: FeedForward::load(&vb.pp("feed_forward2"), c)?,
            conv: LightConv::load(&vb.pp("lconv1d"), c)?,
            attn: Attention::load(&vb.pp("self_attn"), c)?,
            pre: vb.get(h, "norm_pre_attn.weight")?,
            post: vb.get(h, "norm_post_attn.weight")?,
            out: vb.get(h, "norm_out.weight")?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        pos: &Tensor,
        mask: &[bool],
        c: &AudioConfig,
        layout: Option<&AttentionLayout>,
    ) -> Result<Tensor> {
        let x = self.ff1.forward(x)?;
        let y = rms_norm(
            &clamp(&x, -c.gradient_clipping, c.gradient_clipping)?,
            Some(&self.pre),
            1e-6,
        )?;
        let y = if let Some(layout) = layout {
            self.attn.forward_native(&y, pos, layout, c)?
        } else {
            self.attn.forward(&y, pos, mask, c)?
        };
        let y = rms_norm(
            &clamp(&y, -c.gradient_clipping, c.gradient_clipping)?,
            Some(&self.post),
            1e-6,
        )?
        .add(&x)?;
        let y = self.ff2.forward(&self.conv.forward(&y)?)?;
        rms_norm(
            &clamp(&y, -c.gradient_clipping, c.gradient_clipping)?,
            Some(&self.out),
            1e-6,
        )
    }
}

/// A loaded conformer tower plus the projection into text hidden space.
pub struct AudioModel {
    first: SubsampleLayer,
    second: SubsampleLayer,
    input: Tensor,
    positions: Tensor,
    layers: Vec<Layer>,
    output: Tensor,
    bias: Tensor,
    embed: Tensor,
    cfg: AudioConfig,
}

impl AudioModel {
    pub fn load(vb: &VarBuilder<'_>, c: &AudioConfig, text_hidden: usize) -> Result<Self> {
        if c.num_attention_heads == 0
            || c.hidden_size < 4
            || !c.hidden_size.is_multiple_of(c.num_attention_heads)
            || !c.hidden_size.is_multiple_of(2)
            || c.attention_chunk_size == 0
            || c.attention_context_left < 2
            || c.attention_context_right != 0
            || c.subsampling_conv_channels[0] != 128
            || c.subsampling_conv_channels[1] == 0
            || c.conv_kernel_size == 0
            || c.hidden_act != "silu"
            || !c.attention_logit_cap.is_finite()
            || c.attention_logit_cap <= 0.0
        {
            return Err(candle_core::Error::Msg(
                "unsupported or invalid Gemma 4 audio geometry/activation".into(),
            ));
        }
        let tower = vb.pp("audio_tower");
        let sub = tower.pp("subsample_conv_projection");
        let channels = c.subsampling_conv_channels;
        let half = c.hidden_size / 2;
        let increment = 10_000.0_f64.ln() / (half - 1) as f64;
        let mut inv: Vec<f32> = (0..half).map(|i| i as f32 * (-increment as f32)).collect();
        torch_cpu::vectorized_loop(&mut inv, |v| v.map(sleef::expf), f32::exp);
        let context =
            c.attention_chunk_size + c.attention_context_left - 1 + c.attention_context_right;
        let plen = context / 2 + 1;
        let mut scaled: Vec<f32> = (0..plen)
            .rev()
            .flat_map(|p| inv.iter().map(move |&v| p as f32 * v))
            .collect();
        let mut cos = scaled.clone();
        torch_cpu::sin(&mut scaled);
        torch_cpu::cos(&mut cos);
        let mut positions = Vec::with_capacity(plen * c.hidden_size);
        for (sin, cos) in scaled.chunks_exact(half).zip(cos.chunks_exact(half)) {
            positions.extend_from_slice(sin);
            positions.extend_from_slice(cos);
        }
        Ok(Self {
            first: SubsampleLayer::load(&sub.pp("layer0"), 1, channels[0], c.rms_norm_eps)?,
            second: SubsampleLayer::load(
                &sub.pp("layer1"),
                channels[0],
                channels[1],
                c.rms_norm_eps,
            )?,
            input: sub.get(
                (c.hidden_size, (channels[0] / 4) * channels[1]),
                "input_proj_linear.weight",
            )?,
            positions: Tensor::from_vec(positions, (plen, c.hidden_size), vb.device())?,
            layers: (0..c.num_hidden_layers)
                .map(|i| Layer::load(&tower.pp(format!("layers.{i}")), c))
                .collect::<Result<_>>()?,
            output: tower.get((c.output_proj_dims, c.hidden_size), "output_proj.weight")?,
            bias: tower.get(c.output_proj_dims, "output_proj.bias")?,
            embed: vb.get(
                (text_hidden, c.output_proj_dims),
                "embed_audio.embedding_projection.weight",
            )?,
            cfg: c.clone(),
        })
    }

    #[cfg(target_os = "macos")]
    fn project_output(&self, x: &Tensor) -> Result<Tensor> {
        use ndarray::{Array2, ArrayView2};
        if !x.device().is_cpu() {
            return linear(x, &self.output)?.broadcast_add(&self.bias);
        }
        let (seq, input) = x.dims2()?;
        let output = self.bias.dim(0)?;
        let values = x.flatten_all()?.to_vec1::<f32>()?;
        let weights = self.output.flatten_all()?.to_vec1::<f32>()?;
        let bias = self.bias.to_vec1::<f32>()?;
        let a = ArrayView2::from_shape((seq, input), &values)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let w = ArrayView2::from_shape((output, input), &weights)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut out = Array2::from_shape_fn((seq, output), |(_, col)| bias[col]);
        // PyTorch nn.Linear with bias dispatches addmm: beta=1 accumulates
        // into the broadcast bias, rather than rounding GEMM then adding.
        ndarray::linalg::general_mat_mul(1.0, &a, &w.t(), 1.0, &mut out);
        Tensor::from_vec(out.into_raw_vec_and_offset().0, (seq, output), x.device())
    }

    #[cfg(not(target_os = "macos"))]
    fn project_output(&self, x: &Tensor) -> Result<Tensor> {
        linear(x, &self.output)?.broadcast_add(&self.bias)
    }

    pub fn forward(
        &self,
        features: &Tensor,
        mask: &[bool],
        mut hidden: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let (seq, freq) = features.dims2()?;
        let (x, mask) = self
            .first
            .forward(&features.reshape((1, seq, freq))?, mask)?;
        let (x, mask) = self.second.forward(&x, &mask)?;
        let (channels, seq, freq) = x.dims3()?;
        let mut x = linear(
            &x.permute((1, 2, 0))?
                .contiguous()?
                .reshape((seq, freq * channels))?,
            &self.input,
        )?;
        if let Some(h) = hidden.as_mut() {
            h.push(x.clone());
        }
        let layout = if features.device().is_cpu() {
            None
        } else {
            Some(AttentionLayout::new(&mask, &self.cfg, features.device())?)
        };
        for layer in &self.layers {
            x = layer.forward(&x, &self.positions, &mask, &self.cfg, layout.as_ref())?;
            if let Some(h) = hidden.as_mut() {
                h.push(x.clone());
            }
        }
        let output = self.project_output(&x)?;
        let embed = linear(
            &rms_norm(&output, None, self.cfg.rms_norm_eps)?,
            &self.embed,
        )?;
        let indices: Vec<u32> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| v.then_some(i as u32))
            .collect();
        if !features.device().is_cpu() && indices.is_empty() {
            // Metal cannot allocate an empty index buffer; retain a zero-length
            // view of the existing device allocation instead.
            return embed.narrow(0, 0, 0);
        }
        embed.index_select(&Tensor::new(indices.as_slice(), features.device())?, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_tensor(shape: &[usize], dev: &Device) -> Tensor {
        let values: Vec<f32> = (0..shape.iter().product())
            .map(|i| (((i * 17 + 3) % 41) as f32 - 20.0) * 0.025)
            .collect();
        Tensor::from_vec(values, shape, dev).unwrap()
    }

    fn fixture_linear(
        weights: &mut std::collections::HashMap<String, Tensor>,
        prefix: &str,
        input: usize,
        output: usize,
        dev: &Device,
    ) {
        weights.insert(
            format!("{prefix}.linear.weight"),
            fixture_tensor(&[output, input], dev),
        );
        for (name, value) in [
            ("input_min", -0.7_f32),
            ("input_max", 0.9),
            ("output_min", -0.8),
            ("output_max", 0.6),
        ] {
            weights.insert(format!("{prefix}.{name}"), Tensor::new(value, dev).unwrap());
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "synthetic checkpoint specifies all audio tower weights"
    )]
    fn fixture_model(dev: &Device) -> AudioModel {
        let c = AudioConfig {
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            hidden_act: "silu".into(),
            subsampling_conv_channels: [128, 4],
            conv_kernel_size: 5,
            residual_weight: 0.5,
            attention_chunk_size: 4,
            attention_context_left: 6,
            attention_context_right: 0,
            attention_logit_cap: 3.0,
            attention_invalid_logits_value: -1e9,
            use_clipped_linears: true,
            rms_norm_eps: 1e-6,
            gradient_clipping: 0.75,
            output_proj_dims: 6,
        };
        let mut weights = std::collections::HashMap::new();
        let sub = "audio_tower.subsample_conv_projection";
        for (i, input, output) in [(0, 1, 128), (1, 128, 4)] {
            weights.insert(
                format!("{sub}.layer{i}.conv.weight"),
                fixture_tensor(&[output, input, 3, 3], dev),
            );
            weights.insert(
                format!("{sub}.layer{i}.norm.weight"),
                Tensor::ones(output, candle_core::DType::F32, dev).unwrap(),
            );
        }
        weights.insert(
            format!("{sub}.input_proj_linear.weight"),
            fixture_tensor(&[8, 128], dev),
        );
        for i in 0..c.num_hidden_layers {
            let layer = format!("audio_tower.layers.{i}");
            for ff in ["feed_forward1", "feed_forward2"] {
                fixture_linear(
                    &mut weights,
                    &format!("{layer}.{ff}.ffw_layer_1"),
                    8,
                    32,
                    dev,
                );
                fixture_linear(
                    &mut weights,
                    &format!("{layer}.{ff}.ffw_layer_2"),
                    32,
                    8,
                    dev,
                );
                for norm in ["pre_layer_norm", "post_layer_norm"] {
                    weights.insert(
                        format!("{layer}.{ff}.{norm}.weight"),
                        Tensor::ones(8, candle_core::DType::F32, dev).unwrap(),
                    );
                }
            }
            for proj in ["q_proj", "k_proj", "v_proj", "post"] {
                fixture_linear(
                    &mut weights,
                    &format!("{layer}.self_attn.{proj}"),
                    8,
                    8,
                    dev,
                );
            }
            weights.insert(
                format!("{layer}.self_attn.relative_k_proj.weight"),
                fixture_tensor(&[8, 8], dev),
            );
            weights.insert(
                format!("{layer}.self_attn.per_dim_scale"),
                fixture_tensor(&[4], dev),
            );
            fixture_linear(
                &mut weights,
                &format!("{layer}.lconv1d.linear_start"),
                8,
                16,
                dev,
            );
            fixture_linear(
                &mut weights,
                &format!("{layer}.lconv1d.linear_end"),
                8,
                8,
                dev,
            );
            weights.insert(
                format!("{layer}.lconv1d.depthwise_conv1d.weight"),
                fixture_tensor(&[8, 1, 5], dev),
            );
            for norm in [
                "lconv1d.pre_layer_norm",
                "lconv1d.conv_norm",
                "norm_pre_attn",
                "norm_post_attn",
                "norm_out",
            ] {
                weights.insert(
                    format!("{layer}.{norm}.weight"),
                    Tensor::ones(8, candle_core::DType::F32, dev).unwrap(),
                );
            }
        }
        weights.insert(
            "audio_tower.output_proj.weight".into(),
            fixture_tensor(&[6, 8], dev),
        );
        weights.insert(
            "audio_tower.output_proj.bias".into(),
            fixture_tensor(&[6], dev),
        );
        weights.insert(
            "embed_audio.embedding_projection.weight".into(),
            fixture_tensor(&[10, 6], dev),
        );
        let vb = VarBuilder::from_tensors(weights, candle_core::DType::F32, dev);
        AudioModel::load(&vb, &c, 10).unwrap()
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32) {
        assert_eq!(actual.dims(), expected.dims());
        let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() <= tolerance,
                "element {i}: {a} != {b} (tol {tolerance})"
            );
        }
    }

    #[test]
    fn native_audio_stages_match_cpu() {
        let dev = Device::Cpu;
        let audio = fixture_model(&dev);
        for seq in [1, 3, 4, 5, 9, 17] {
            // Holes as well as trailing padding; odd subsampling geometries.
            let mask: Vec<bool> = (0..seq).map(|i| i != 2 && i != seq - 1).collect();
            let features = fixture_tensor(&[1, seq, 128], &dev);
            let (cpu, next_mask) = audio.first.forward(&features, &mask).unwrap();
            let (native, native_mask) = audio.first.forward_native(&features, &mask).unwrap();
            assert_eq!(next_mask, native_mask);
            assert_close(&native, &cpu, 2e-5);
            let (cpu, _) = audio.second.forward(&cpu, &next_mask).unwrap();
            let (native, _) = audio.second.forward_native(&native, &native_mask).unwrap();
            assert_close(&native, &cpu, 3e-5);
            let x = fixture_tensor(&[seq, 8], &dev);
            let layer = &audio.layers[0];
            let glu = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_close(
                &layer.conv.depthwise_native(&x).unwrap(),
                &layer.conv.depthwise(&glu, seq, 8, &dev).unwrap(),
                1e-6,
            );
            assert_close(
                &layer.conv.forward_native(&x).unwrap(),
                &layer.conv.forward(&x).unwrap(),
                2e-6,
            );
            let layout = AttentionLayout::new(&mask, &audio.cfg, &dev).unwrap();
            assert_close(
                &layer
                    .attn
                    .forward_native(&x, &audio.positions, &layout, &audio.cfg)
                    .unwrap(),
                &layer
                    .attn
                    .forward(&x, &audio.positions, &mask, &audio.cfg)
                    .unwrap(),
                2e-6,
            );
        }
    }

    #[cfg(any(feature = "embeddinggemma2-metal", feature = "embeddinggemma2-cuda"))]
    #[test]
    #[ignore = "requires an enabled Metal or CUDA feature and a GPU"]
    fn gpu_audio_synthetic_matches_cpu() {
        let dev = crate::accel::embeddinggemma2_device();
        assert!(!dev.is_cpu(), "GPU test must not silently fall back to CPU");
        let gpu = fixture_model(&dev);
        let cpu = fixture_model(&Device::Cpu);
        for seq in [1, 15, 16, 17, 37, 65] {
            let mask: Vec<bool> = (0..seq).map(|i| i != 2 && i != seq - 1).collect();
            let features = fixture_tensor(&[seq, 128], &Device::Cpu);
            let mut cpu_hidden = Vec::new();
            let mut gpu_hidden = Vec::new();
            let expected = cpu
                .forward(&features, &mask, Some(&mut cpu_hidden))
                .unwrap();
            let actual = gpu
                .forward(
                    &features.to_device(&dev).unwrap(),
                    &mask,
                    Some(&mut gpu_hidden),
                )
                .unwrap();
            assert_close(&actual, &expected, 3e-4);
            for (actual, expected) in gpu_hidden.iter().zip(&cpu_hidden) {
                assert_close(actual, expected, 3e-4);
            }
            // Production path without diagnostic snapshots must remain native too.
            assert_close(
                &gpu.forward(&features.to_device(&dev).unwrap(), &mask, None)
                    .unwrap(),
                &actual,
                1e-6,
            );
        }
    }

    #[test]
    #[ignore = "needs audio tower Python trace and checkpoint"]
    #[expect(
        clippy::too_many_lines,
        reason = "linear diagnostic probe covers each reference stage in order"
    )]
    fn audio_stage_probe() {
        let dir = std::path::PathBuf::from(std::env::var("EG2_AUDIO_TRACE_DIR").unwrap());
        let model = std::path::PathBuf::from(std::env::var("EG2_MODEL_DIR").unwrap());
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        let tensors = &manifest["cases"][0]["tensors"];
        let dev = candle_core::Device::Cpu;
        let read = |key: &str| -> Tensor {
            let entry = &tensors[format!("hf.audio.{key}")];
            let bytes = std::fs::read(dir.join(entry["file"].as_str().unwrap())).unwrap();
            let data: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let mut shape: Vec<usize> = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            if shape.len() == 3 && shape[0] == 1 {
                shape.remove(0);
            }
            Tensor::from_vec(data, shape.as_slice(), &dev).unwrap()
        };
        let compare = |label: &str, out: &Tensor| {
            let reference = read(label).flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let ours = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(ours.len(), reference.len(), "{label}");
            let same = ours
                .iter()
                .zip(&reference)
                .filter(|(a, b)| a.to_bits() == b.to_bits())
                .count();
            let max = ours
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            println!("{label}: identical {same}/{}, max={max:e}", ours.len());
            assert_eq!(same, reference.len(), "{label}");
        };
        let cfg: super::super::config::Config =
            serde_json::from_slice(&std::fs::read(model.join("config.json")).unwrap()).unwrap();
        let c = cfg.audio.unwrap();
        let vb = VarBuilder::from_buffered_safetensors(
            std::fs::read(model.join("model.safetensors")).unwrap(),
            candle_core::DType::F32,
            &dev,
        )
        .unwrap();
        let audio = AudioModel::load(&vb, &c, cfg.text.hidden_size).unwrap();
        compare("rel_pos_enc", &audio.positions);
        compare(
            "output_proj",
            &audio.project_output(&read("layers.11")).unwrap(),
        );
        let layer = &audio.layers[0];
        let ff = layer
            .ff1
            .forward(&read("subsample_conv_projection"))
            .unwrap();
        compare("layers.0.feed_forward1", &ff);
        let pre = rms_norm(
            &clamp(
                &read("layers.0.feed_forward1"),
                -c.gradient_clipping,
                c.gradient_clipping,
            )
            .unwrap(),
            Some(&layer.pre),
            1e-6,
        )
        .unwrap();
        compare("layers.0.norm_pre_attn", &pre);
        let seq = pre.dim(0).unwrap();
        let attn = layer
            .attn
            .forward(
                &read("layers.0.norm_pre_attn"),
                &read("rel_pos_enc"),
                &{
                    let entry = &tensors["input.input_features_mask"];
                    std::fs::read(dir.join(entry["file"].as_str().unwrap()))
                        .unwrap()
                        .iter()
                        .step_by(4)
                        .map(|&b| b != 0)
                        .collect::<Vec<_>>()
                },
                &c,
            )
            .unwrap();
        compare("layers.0.self_attn", &attn);
        let post = rms_norm(&read("layers.0.self_attn"), Some(&layer.post), 1e-6).unwrap();
        compare("layers.0.norm_post_attn", &post);
        let residual = read("layers.0.norm_post_attn")
            .add(&read("layers.0.feed_forward1"))
            .unwrap();
        compare(
            "layers.0.lconv1d.pre_layer_norm",
            &rms_norm(&residual, Some(&layer.conv.pre), layer.conv.eps).unwrap(),
        );
        let start = layer
            .conv
            .start
            .forward(&read("layers.0.lconv1d.pre_layer_norm"))
            .unwrap();
        compare("layers.0.lconv1d.linear_start", &start);
        let start = read("layers.0.lconv1d.linear_start")
            .to_vec2::<f32>()
            .unwrap();
        let h = c.hidden_size;
        let glu: Vec<f32> = start
            .iter()
            .flat_map(|r| (0..h).map(move |i| r[i] * (1.0 / (1.0 + sleef::expf(-r[h + i])))))
            .collect();
        let depthwise = layer.conv.depthwise(&glu, seq, h, &dev).unwrap();
        compare("layers.0.lconv1d.depthwise_conv1d", &depthwise.t().unwrap());
        let norm = layer
            .conv
            .depthwise_norm(
                &read("layers.0.lconv1d.depthwise_conv1d")
                    .t()
                    .unwrap()
                    .contiguous()
                    .unwrap(),
            )
            .unwrap();
        compare("layers.0.lconv1d.conv_norm", &norm);
        compare(
            "layers.0.lconv1d.act_fn",
            &silu(&read("layers.0.lconv1d.conv_norm")).unwrap(),
        );
        compare(
            "layers.0.lconv1d.linear_end",
            &layer
                .conv
                .end
                .forward(&read("layers.0.lconv1d.act_fn"))
                .unwrap(),
        );
        let conv = layer.conv.forward(&residual).unwrap();
        compare("layers.0.lconv1d", &conv);
        let ff = layer.ff2.forward(&read("layers.0.lconv1d")).unwrap();
        compare("layers.0.feed_forward2", &ff);
    }
}
