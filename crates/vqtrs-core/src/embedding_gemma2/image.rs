//! Image preprocessing, reproducing the reference pipeline bit for bit:
//! PIL `convert("RGB")`, transformers' `Gemma4ImageProcessor` (aspect-ratio
//! preserving size), PyTorch's uint8 antialiased bicubic resize (the aarch64
//! NEON kernel in `UpSampleKernelNEONAntialias.h`, whose integer arithmetic
//! equals the generic separable path), the `1/255` rescale, patchification
//! and padding.

use image::DynamicImage;

/// Image-processor settings (`processor_config.json`, `image_processor`).
#[derive(Debug, Clone, Copy)]
pub struct ImageSettings {
    pub patch_size: usize,
    pub max_soft_tokens: usize,
    pub pooling_kernel_size: usize,
    pub rescale_factor: f64,
}

impl Default for ImageSettings {
    fn default() -> Self {
        Self {
            patch_size: 16,
            max_soft_tokens: 280,
            pooling_kernel_size: 3,
            rescale_factor: 0.003_921_568_627_450_98,
        }
    }
}

impl ImageSettings {
    pub const fn max_patches(&self) -> usize {
        self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size
    }

    pub const fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }
}

/// An interleaved RGB8 image (`height * width * 3` bytes).
#[derive(Debug, Clone)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// Decode `bytes` and convert like PIL `Image.convert("RGB")`: alpha is
/// dropped (not composited), grey is replicated, 16-bit samples keep their
/// high byte.
pub fn decode_rgb(bytes: &[u8]) -> Result<Rgb, String> {
    // JPEG through the libjpeg-turbo-exact decoder (Pillow's decoder).
    if let Some(d) = super::jpeg::decode(bytes) {
        let data = if d.channels == 1 {
            d.data.into_iter().flat_map(|v| [v, v, v]).collect()
        } else {
            d.data
        };
        return Ok(Rgb {
            width: d.width,
            height: d.height,
            data,
        });
    }
    let img = image::load_from_memory(bytes).map_err(|e| format!("cannot decode image: {e}"))?;
    let (width, height) = (img.width() as usize, img.height() as usize);
    let data = match img {
        DynamicImage::ImageRgb8(buf) => buf.into_raw(),
        DynamicImage::ImageRgba8(buf) => buf
            .into_raw()
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        DynamicImage::ImageLuma8(buf) => {
            buf.into_raw().into_iter().flat_map(|v| [v, v, v]).collect()
        }
        DynamicImage::ImageLumaA8(buf) => buf
            .into_raw()
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0]])
            .collect(),
        DynamicImage::ImageRgb16(buf) => {
            buf.into_raw().into_iter().map(|v| (v >> 8) as u8).collect()
        }
        DynamicImage::ImageRgba16(buf) => buf
            .into_raw()
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]].map(|v| (v >> 8) as u8))
            .collect(),
        other => {
            return Err(format!(
                "unsupported image pixel format {:?}",
                other.color()
            ));
        }
    };
    Ok(Rgb {
        width,
        height,
        data,
    })
}

/// `get_aspect_ratio_preserving_size`: the largest `(height, width)`, both
/// multiples of `pooling_kernel_size * patch_size`, within the patch budget.
pub fn target_size(
    height: usize,
    width: usize,
    s: &ImageSettings,
) -> Result<(usize, usize), String> {
    let total_px = (height * width) as f64;
    let target_px = s.max_patches() * s.patch_size * s.patch_size;
    let factor = (target_px as f64 / total_px).sqrt();
    let side = s.pooling_kernel_size * s.patch_size;
    let snap = |ideal: f64| (ideal / side as f64).floor() as usize * side;
    let (mut th, mut tw) = (snap(factor * height as f64), snap(factor * width as f64));
    if th == 0 && tw == 0 {
        return Err("image too small to resize to a whole number of patches".into());
    }
    let max_side = (s.max_patches() / (s.pooling_kernel_size * s.pooling_kernel_size)) * side;
    if th == 0 {
        th = side;
        tw = ((width as f64 / height as f64).floor() as usize * side).min(max_side);
    } else if tw == 0 {
        tw = side;
        th = ((height as f64 / width as f64).floor() as usize * side).min(max_side);
    }
    if th * tw > target_px {
        return Err(format!(
            "resizing {height}x{width} to {th}x{tw} exceeds the patch budget"
        ));
    }
    Ok((th, tw))
}

/// Keys cubic filter (`a = -0.5`) in `double`, as `aa_filter<double, true>`.
/// The reference build (clang, `-ffp-contract=on`) fuses the multiply-adds
/// inside each expression of `cubic_convolution1/2`.
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        // ((A + 2) * x - (A + 3)) * x * x + 1
        let t = (A + 2.0).mul_add(x, -(A + 3.0));
        (t * x).mul_add(x, 1.0)
    } else if x < 2.0 {
        // ((A * x - 5 * A) * x + 8 * A) * x - 4 * A
        let t = A.mul_add(x, -(5.0 * A));
        let t = t.mul_add(x, 8.0 * A);
        t.mul_add(x, -(4.0 * A))
    } else {
        0.0
    }
}

/// Per-output-pixel source window and int16 weights for one axis.
struct AxisWeights {
    xmin: Vec<usize>,
    xsize: Vec<usize>,
    ksize: usize,
    weights: Vec<i16>,
    precision: u32,
}

/// `_compute_index_ranges_int16_weights` (antialias, bicubic).
fn axis_weights(input: usize, output: usize) -> AxisWeights {
    let scale = input as f64 / output as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let ksize = (support.ceil() as usize) * 2 + 1;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut weights_f64 = vec![0.0_f64; output * ksize];
    let (mut xmin, mut xsize) = (vec![0; output], vec![0; output]);
    let mut wt_max = 0.0_f64;
    for i in 0..output {
        let center = scale * (i as f64 + 0.5);
        // C `static_cast<int64_t>` truncates toward zero.
        let lo = ((center - support + 0.5) as i64).max(0);
        let hi = ((center + support + 0.5) as i64).min(input as i64);
        let size = (hi - lo).clamp(0, ksize as i64) as usize;
        let lo = lo as usize;
        let w = &mut weights_f64[i * ksize..(i + 1) * ksize];
        let mut total = 0.0_f64;
        for (j, slot) in w.iter_mut().enumerate().take(size) {
            *slot = cubic(((j + lo) as f64 - center + 0.5) * invscale);
            total += *slot;
        }
        if total != 0.0 {
            for slot in w.iter_mut().take(size) {
                *slot /= total;
                wt_max = wt_max.max(*slot);
            }
        }
        xmin[i] = lo;
        xsize[i] = size;
    }
    let mut precision = 0_u32;
    while precision < 22 {
        // `0.5 + wt_max * 2^(p+1)`: the product is exact, so fusing is too.
        let next = wt_max.mul_add(f64::from(1_u32 << (precision + 1)), 0.5) as i32;
        if next >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    let scale_i = f64::from(1_u32 << precision);
    let weights = weights_f64
        .into_iter()
        .map(|w| {
            let v = w * scale_i;
            (if v < 0.0 { -0.5 + v } else { 0.5 + v }) as i32 as i16
        })
        .collect();
    AxisWeights {
        xmin,
        xsize,
        ksize,
        weights,
        precision,
    }
}

/// `clamp((2^(p-1) + sum(w * px)) >> p, 0, 255)`.
fn convolve(taps: impl Iterator<Item = (i16, u8)>, precision: u32) -> u8 {
    let mut sum: i32 = 1 << (precision - 1);
    for (w, px) in taps {
        sum += i32::from(w) * i32::from(px);
    }
    (sum >> precision).clamp(0, 255) as u8
}

/// PyTorch's uint8 antialiased bicubic resize of an RGB image: the
/// horizontal pass into a uint8 buffer, then the vertical pass.
pub fn resize(src: &Rgb, out_h: usize, out_w: usize) -> Rgb {
    let mut img = src.clone();
    if out_w != img.width {
        let aw = axis_weights(img.width, out_w);
        let mut data = vec![0_u8; img.height * out_w * 3];
        for y in 0..img.height {
            let row = &img.data[y * img.width * 3..(y + 1) * img.width * 3];
            for x in 0..out_w {
                let k = &aw.weights[x * aw.ksize..x * aw.ksize + aw.xsize[x]];
                for c in 0..3 {
                    let taps = k
                        .iter()
                        .enumerate()
                        .map(|(j, &w)| (w, row[(aw.xmin[x] + j) * 3 + c]));
                    data[(y * out_w + x) * 3 + c] = convolve(taps, aw.precision);
                }
            }
        }
        img = Rgb {
            width: out_w,
            height: img.height,
            data,
        };
    }
    if out_h != img.height {
        let aw = axis_weights(img.height, out_h);
        let stride = img.width * 3;
        let mut data = vec![0_u8; out_h * stride];
        for y in 0..out_h {
            let k = &aw.weights[y * aw.ksize..y * aw.ksize + aw.xsize[y]];
            for i in 0..stride {
                let taps = k
                    .iter()
                    .enumerate()
                    .map(|(j, &w)| (w, img.data[(aw.xmin[y] + j) * stride + i]));
                data[y * stride + i] = convolve(taps, aw.precision);
            }
        }
        img = Rgb {
            width: img.width,
            height: out_h,
            data,
        };
    }
    img
}

/// Processor output for one image.
#[derive(Debug, Clone)]
pub struct Patches {
    /// `(max_patches, patch_dim)` row-major, zero-padded.
    pub pixel_values: Vec<f32>,
    /// `(max_patches, 2)` `(x, y)` patch positions, `-1` for padding.
    pub positions: Vec<[i64; 2]>,
    /// Number of soft tokens the image becomes.
    pub num_soft_tokens: usize,
}

/// Full `Gemma4ImageProcessor` pipeline for one decoded RGB image.
pub fn preprocess(img: &Rgb, s: &ImageSettings) -> Result<Patches, String> {
    let (th, tw) = target_size(img.height, img.width, s)?;
    let img = if (th, tw) == (img.height, img.width) {
        img.clone()
    } else {
        resize(img, th, tw)
    };
    let scale = s.rescale_factor as f32;
    let (p, dim) = (s.patch_size, s.patch_dim());
    let (ph, pw) = (th / p, tw / p);
    let mut pixel_values = vec![0.0_f32; s.max_patches() * dim];
    let mut positions = vec![[-1_i64, -1]; s.max_patches()];
    for py in 0..ph {
        for px in 0..pw {
            let patch = py * pw + px;
            positions[patch] = [px as i64, py as i64];
            let out = &mut pixel_values[patch * dim..(patch + 1) * dim];
            for dy in 0..p {
                for dx in 0..p {
                    let src = ((py * p + dy) * tw + px * p + dx) * 3;
                    for c in 0..3 {
                        out[(dy * p + dx) * 3 + c] = f32::from(img.data[src + c]) * scale;
                    }
                }
            }
        }
    }
    let k2 = s.pooling_kernel_size * s.pooling_kernel_size;
    Ok(Patches {
        pixel_values,
        positions,
        num_soft_tokens: ph * pw / k2,
    })
}
