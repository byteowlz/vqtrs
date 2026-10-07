//! `PyTorch` 2.14.1 CPU `LayerNorm` for contiguous `f32` rows, affine weight,
//! and no bias, matching the aarch64 macOS single-threaded kernels.
//!
//! References: `aten/src/ATen/native/cpu/moments_utils.h`
//! (`RowwiseMoments`, `UpdateMomentsVec`, `AddMomentsVec`, `AddMoments`) and
//! `aten/src/ATen/native/cpu/layer_norm_kernel.cpp` (`LayerNormSecondPass`).
//! Four NEON lanes are emulated with safe scalar arithmetic; explicit
//! `mul_add` calls preserve fused operations without requiring ARM hardware.
//! The audio subsampler's checkpoint widths are 128 and 32, with `eps = 1e-6`.
//!
//! Parity was verified against the macOS arm64 wheel (torch commit
//! `5c4886908584029761b579af026dcfb627c84070`), with one thread: 169,916 output
//! values across 35 synthetic shapes and both checkpoint subsample layers,
//! using real convolution/norm weights with synthetic spectrogram inputs.
//! Every output bit matched in debug and optimized builds. The self-contained
//! tests retain synthetic output digests and individual mean/rstd bit goldens.

const LANES: usize = 4;
const CHUNK_VECTORS: usize = 16;

#[derive(Clone, Copy, Debug, Default)]
struct VectorMoments {
    count: usize,
    mean: [f32; LANES],
    m2: [f32; LANES],
}

impl VectorMoments {
    /// `AddMomentsVec`: only the final variance multiply-add is fused.
    fn merge(&mut self, other: Self) {
        let count = self.count + other.count;
        let ratio = if count == 0 {
            0.0
        } else {
            other.count as f32 / count as f32
        };
        for lane in 0..LANES {
            let delta = other.mean[lane] - self.mean[lane];
            let scaled_delta = ratio * delta;
            let count_delta = delta * self.count as f32;
            self.mean[lane] += scaled_delta;
            self.m2[lane] = count_delta.mul_add(scaled_delta, self.m2[lane] + other.m2[lane]);
        }
        self.count = count;
    }

    /// `UpdateMomentsVec`: independent Welford streams for the four lanes.
    fn chunk(data: &[f32]) -> Self {
        let mut moments = Self::default();
        for (index, vector) in data.chunks_exact(LANES).enumerate() {
            let reciprocal = 1.0 / (index + 1) as f32;
            for (lane, &x) in vector.iter().enumerate() {
                let delta = x - moments.mean[lane];
                moments.mean[lane] = reciprocal.mul_add(delta, moments.mean[lane]);
                moments.m2[lane] = delta.mul_add(x - moments.mean[lane], moments.m2[lane]);
            }
            moments.count += 1;
        }
        moments
    }
}

/// `RowwiseMoments` with population variance (`ddof = 0`).
fn rowwise_moments(row: &[f32]) -> (f32, f32) {
    let vectors = row.len() / LANES;
    let chunks = vectors.div_ceil(CHUNK_VECTORS);
    // ATen's CeilLog2 is at least one, including for zero chunks.
    let depth = if chunks <= 2 {
        1
    } else {
        (usize::BITS - (chunks - 1).leading_zeros()) as usize
    };
    let mut stack = [VectorMoments::default(); usize::BITS as usize];
    let body = vectors * LANES;
    for (index, chunk) in row[..body].chunks(CHUNK_VECTORS * LANES).enumerate() {
        stack[0].merge(VectorMoments::chunk(chunk));
        let mut mask = index + 1;
        for level in 1..depth {
            if mask & 1 != 0 {
                break;
            }
            let lower = stack[level - 1];
            stack[level].merge(lower);
            stack[level - 1] = VectorMoments::default();
            mask >>= 1;
        }
    }
    for level in 1..depth {
        let upper = stack[level];
        stack[0].merge(upper);
    }

    // Scalar tail precedes lane merges, even when it is shorter than a vector.
    let mut count = 0;
    let mut mean = 0.0_f32;
    let mut m2 = 0.0_f32;
    for &x in &row[body..] {
        let delta = x - mean;
        count += 1;
        mean += delta / count as f32;
        m2 = delta.mul_add(x - mean, m2);
    }
    for lane in 0..LANES {
        // AddMoments, compiled with Clang's scalar FP contraction. Unlike
        // AddMomentsVec, this squares delta before scaling by the ratio.
        let total = count + vectors;
        let ratio = if total == 0 {
            0.0
        } else {
            vectors as f32 / total as f32
        };
        let delta = stack[0].mean[lane] - mean;
        mean = ratio.mul_add(delta, mean);
        m2 += (delta * delta * ratio).mul_add(count as f32, stack[0].m2[lane]);
        count = total;
    }
    (mean, m2 / row.len() as f32)
}

/// Normalize each contiguous last-dimension row in place, with affine weight
/// and no additive bias, as in the `EmbeddingGemma2` audio subsampler.
///
/// Matches `PyTorch` 2.14.1 CPU `f32` arithmetic on aarch64 macOS with one thread:
/// four-lane chunked Welford moments, population variance, and
/// `((x - mean) * (1 / sqrt(variance + eps))) * weight + 0`.
/// This is not `RMSNorm` and does not apply the subsampler's subsequent `ReLU`.
/// Empty batches are accepted when the row shape is valid.
///
/// # Panics
///
/// Panics if `cols` is zero, `data.len()` is not a multiple of `cols`, or
/// `weight.len()` differs from `cols`.
pub fn layer_norm_rows(data: &mut [f32], cols: usize, weight: &[f32], eps: f32) {
    assert!(cols > 0, "LayerNorm row width must be nonzero");
    assert!(
        data.len().is_multiple_of(cols),
        "LayerNorm rows must be complete"
    );
    assert_eq!(weight.len(), cols, "LayerNorm weight must match row width");
    for row in data.chunks_exact_mut(cols) {
        let (mean, variance) = rowwise_moments(row);
        let scale = 1.0 / (variance + eps).sqrt();
        for (x, &gamma) in row.iter_mut().zip(weight) {
            // LayerNormSecondPass with beta=null: center before scaling, and
            // retain the +0 (in particular its effect on signed zero).
            *x = ((*x - mean) * scale).mul_add(gamma, 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{layer_norm_rows, rowwise_moments};

    fn sample(index: usize) -> f32 {
        f32::from_bits(0x3f00_0000 | ((index as u32 * 104_729 + 12_345) & 0x007f_ffff)) - 0.75
    }

    #[expect(
        clippy::suboptimal_flops,
        reason = "fixture generation mirrors separately rounded PyTorch tensor operations"
    )]
    fn fixture(cols: usize) -> (Vec<f32>, Vec<f32>) {
        let data = (0..4 * cols)
            .map(|index| match index / cols {
                0 => sample(index) * 8.0,
                1 => 1024.0 + sample(index) * 0.25,
                2 => sample(index) * (1.0 / 1_048_576.0),
                _ => -3.0,
            })
            .collect();
        let weight = (0..cols)
            .map(|index| {
                let value =
                    f32::from_bits(0x3f00_0000 | ((index as u32 * 7_919 + 999) & 0x007f_ffff));
                if index % 17 == 0 {
                    -0.0
                } else if index % 3 == 1 {
                    -value
                } else {
                    value
                }
            })
            .collect();
        (data, weight)
    }

    // FNV-1a over u32 words (not bytes): every output bit contributes.
    fn digest(data: &[f32]) -> u64 {
        data.iter().fold(0xcbf2_9ce4_8422_2325, |hash, x| {
            (hash ^ u64::from(x.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Goldens dumped with `uv run --no-project --with torch==2.14.1 python -I`
    /// on macOS arm64, `torch.set_num_threads(1)`, `torch.native_layer_norm`
    /// with contiguous `f32` inputs, weight, no bias and `eps=1e-6`.
    /// Rows cover ordinary values, large offsets, tiny variance and constants;
    /// weights include negative values and negative zero. Widths straddle
    /// vector, chunk and cascade boundaries, not just checkpoint widths.
    #[test]
    fn torch_cpu_output_goldens() {
        let goldens = [
            (1, 0x4d25_767f_9dce_13f5),
            (2, 0x3a49_ee86_9822_1a46),
            (3, 0xf74d_42e8_952c_b5cc),
            (4, 0x607b_5dd2_dcd0_3225),
            (5, 0x74a7_0f94_f242_9846),
            (7, 0xccf9_50d2_b882_a852),
            (8, 0xe296_7030_bfe5_e6b8),
            (15, 0x9c36_bb12_a721_8978),
            (16, 0xcf62_f6bf_e979_e36c),
            (17, 0xdd3d_c68c_6819_3258),
            (31, 0xab71_2109_506b_4394),
            (32, 0x8d75_5da3_0217_4981),
            (33, 0x4bbf_1586_748b_22a7),
            (63, 0xf367_1330_0b72_61a0),
            (64, 0xfb6d_b42f_f0df_de1e),
            (65, 0x11f1_459b_b681_a80f),
            (127, 0xf420_09d4_28c7_8c90),
            (128, 0x27e0_2bc4_b198_0ab3),
            (129, 0xd0f9_134f_9416_ecd1),
            (191, 0x6dc8_6c44_1c03_17d5),
            (192, 0x3df6_c9e2_2205_cad2),
            (193, 0x1bdb_7a75_2843_58c6),
            (255, 0xd248_1637_8110_c982),
            (256, 0x1f31_171c_f6ba_64f5),
            (257, 0x3484_2c53_458c_2f19),
            (319, 0xd793_c844_43d2_915a),
            (320, 0xc2c1_a612_864d_8431),
            (321, 0xa2cc_46b5_9296_65e3),
            (511, 0x2610_157d_59c6_edbe),
            (512, 0x83a9_06b1_8634_225c),
            (513, 0xaf91_c62f_d9a6_c535),
            (1023, 0x990e_e560_2754_0a2b),
            (1024, 0x3d10_ff29_da42_1802),
            (1025, 0x0abf_3551_7edf_0a56),
            (4097, 0x184c_af30_e302_a05c),
        ];
        for (cols, expected) in goldens {
            let (mut data, weight) = fixture(cols);
            layer_norm_rows(&mut data, cols, &weight, 1e-6);
            assert_eq!(digest(&data), expected, "width {cols}");
        }
    }

    #[test]
    fn torch_cpu_moments_goldens() {
        // native_layer_norm's mean/rstd bits for the first fixture row.
        let goldens = [
            (3, 3_220_757_176, 1_103_373_325),
            (32, 3_214_682_894, 1_074_449_723),
            (128, 3_198_509_368, 1_064_139_054),
            (193, 3_193_051_938, 1_063_028_437),
            (4097, 3_147_617_316, 1_063_079_315),
        ];
        for (cols, mean_bits, rstd_bits) in goldens {
            let (data, _) = fixture(cols);
            let (mean, variance) = rowwise_moments(&data[..cols]);
            assert_eq!(mean.to_bits(), mean_bits, "mean at width {cols}");
            let rstd = 1.0 / (variance + 1e-6).sqrt();
            assert_eq!(rstd.to_bits(), rstd_bits, "rstd at width {cols}");
        }
    }

    #[test]
    fn epsilon_weight_and_empty_batch() {
        let mut data = [-1.0, 1.0];
        layer_norm_rows(&mut data, 2, &[2.0, -2.0], 3.0);
        assert_eq!(data.map(f32::to_bits), [(-1.0_f32).to_bits(); 2]);
        layer_norm_rows(&mut [], 32, &[1.0; 32], 1e-6);
    }

    #[test]
    fn constant_rows_produce_positive_zero_even_with_negative_weight() {
        let mut data = [-3.0; 128];
        layer_norm_rows(&mut data, 32, &[-1.0; 32], 1e-6);
        assert!(data.iter().all(|x| x.to_bits() == 0));
    }

    #[test]
    fn nonfinite_rows_propagate_nan() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut data = [1.0, 2.0, value, 4.0, 5.0];
            layer_norm_rows(&mut data, 5, &[1.0; 5], 1e-6);
            assert!(data.iter().all(|x| x.is_nan()));
        }
    }

    #[test]
    #[should_panic(expected = "row width must be nonzero")]
    fn zero_width_is_rejected() {
        layer_norm_rows(&mut [], 0, &[], 1e-6);
    }

    #[test]
    #[should_panic(expected = "rows must be complete")]
    fn incomplete_row_is_rejected() {
        layer_norm_rows(&mut [0.0; 3], 2, &[1.0; 2], 1e-6);
    }

    #[test]
    #[should_panic(expected = "weight must match row width")]
    fn incorrect_weight_is_rejected() {
        layer_norm_rows(&mut [0.0; 2], 2, &[1.0], 1e-6);
    }
}
