//! Reusable native CPU arithmetic mechanics, independent of checkpoint weights.
//!
//! These preserve the inspected ARM Torch vector/scalar boundaries. Their reuse
//! does not qualify a different model, dtype policy, runtime or hardware profile.
use crate::embedding_gemma2::{layer_norm, torch_cpu};

/// Explicit elementwise operation; never an embedding-model selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum F32Op {
    /// SLEEF vector cosine and scalar-tail cosine.
    Cos,
    /// SLEEF vector sine and scalar-tail sine.
    Sin,
    /// Tanh-approximate GELU with Torch's operation boundaries.
    GeluTanh,
    /// A fixed scalar base raised to each F32 input exponent.
    PowScalarBase(f32),
}

/// Apply the selected F32 operation in place, preserving vector/scalar tails.
pub fn apply(op: F32Op, values: &mut [f32]) {
    let kernel: fn(&mut [f32]) = match op {
        F32Op::PowScalarBase(base) => return torch_cpu::pow_scalar_base(base, values),
        F32Op::Cos => torch_cpu::cos,
        F32Op::Sin => torch_cpu::sin,
        F32Op::GeluTanh => torch_cpu::gelu_tanh,
    };
    kernel(values);
}

/// BF16-input layer-norm arithmetic using lossless F32 representations.
///
/// This executes paired Welford streams and affine arithmetic in F32. The caller
/// must supply BF16-representable inputs/parameters and narrow the result to BF16.
/// It does not replace the independently qualified F32 audio subsampler norm.
///
/// # Errors
/// Rejects malformed row/affine dimensions, nonpositive/nonfinite epsilon, or
/// nonfinite input/parameters before modifying the input buffer.
pub fn bf16_layer_norm_accumulation(
    values: &mut [f32],
    width: usize,
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
) -> Result<(), String> {
    if width == 0
        || !values.len().is_multiple_of(width)
        || gamma.len() != width
        || beta.len() != width
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || !values
            .iter()
            .chain(gamma)
            .chain(beta)
            .all(|v| v.is_finite())
    {
        return Err("invalid BF16 layer-norm rows, affine parameters or epsilon".into());
    }
    layer_norm::bf16_layer_norm_rows(values, width, gamma, beta, epsilon);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn bf16_norm_matches_cf30153_arm_torch_with_bias_and_tails() {
        // Torch 2.13.0 cf30153, CPU ARM, one thread, eps=1e-6. Inputs and
        // affine parameters below are exactly BF16-representable binary fractions.
        for (width, digest) in [
            (8, 0x0812_b8da_d1d1_2305_u64),
            (9, 0x1b30_901b_b8d7_5a57),
            (31, 0x9256_3cee_63ee_cbef),
            (32, 0x6368_c7c2_0328_5aa5),
            (128, 0x95d2_6f9e_e92b_8125),
            (1152, 0x9b40_d8f0_d349_7125),
        ] {
            let mut values: Vec<f32> = (0..width * 3)
                .map(|i| (i as i32 % 127 - 63) as f32 / 32.0)
                .collect();
            let gamma: Vec<f32> = (0..width).map(|i| (i * 3 % 13 + 1) as f32 / 8.0).collect();
            let beta: Vec<f32> = (0..width)
                .map(|i| (i as i32 * 7 % 29 - 14) as f32 / 32.0)
                .collect();
            bf16_layer_norm_accumulation(&mut values, width, &gamma, &beta, 1e-6).unwrap();
            let rounded = Tensor::new(values, &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            // FNV-1a over u32 words, matching the existing F32 norm goldens.
            let actual = rounded.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, v| {
                (h ^ u64::from(v.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
            });
            assert_eq!(actual, digest, "width {width}");
        }
    }

    #[test]
    fn dispatch_preserves_the_existing_vector_and_scalar_kernels() {
        for (op, kernel) in [
            (F32Op::Cos, torch_cpu::cos as fn(&mut [f32])),
            (F32Op::Sin, torch_cpu::sin),
            (F32Op::GeluTanh, torch_cpu::gelu_tanh),
        ] {
            let mut values = [-2_f32, -1., -0.0, 0., 0.1, 0.5, 1., 2., 3.];
            let mut expected = values;
            kernel(&mut expected);
            apply(op, &mut values);
            assert_eq!(values.map(f32::to_bits), expected.map(f32::to_bits));
        }
        let mut powers = [-2_f32, -0.0, 0., 0.5, 1., 2.];
        let mut expected = powers;
        torch_cpu::pow_scalar_base(10.0, &mut expected);
        apply(F32Op::PowScalarBase(10.0), &mut powers);
        assert_eq!(powers.map(f32::to_bits), expected.map(f32::to_bits));
    }

    #[test]
    fn rgb_facade_rejects_malformed_and_large_intermediate_buffers() {
        let pixel = crate::Gemma2VideoFrame {
            width: 1,
            height: 1,
            data: vec![1, 2, 3],
        };
        let resized = crate::resize_rgb_image(&pixel, 2, 2).unwrap();
        assert_eq!((resized.height, resized.width), (2, 2));
        assert_eq!(resized.data, [1, 2, 3].repeat(4));
        assert!(crate::resize_rgb_image(&pixel, 0, 2).is_err());
        assert!(crate::resize_rgb_image(&pixel, 1, usize::MAX).is_err());
        let tall = crate::Gemma2VideoFrame {
            width: 1,
            height: 4096,
            data: vec![0; 4096 * 3],
        };
        assert!(crate::resize_rgb_image(&tall, 1, 8192).is_err());
        assert!(crate::decode_rgb_image(b"invalid image", 16_000_000).is_err());
    }

    #[test]
    fn malformed_bf16_norm_is_rejected_before_modification() {
        let original = [1_f32, 2., 3.];
        let mut values = original;
        assert!(bf16_layer_norm_accumulation(&mut values, 2, &[1., 1.], &[0., 0.], 1e-6).is_err());
        assert_eq!(values.map(f32::to_bits), original.map(f32::to_bits));
        assert!(bf16_layer_norm_accumulation(&mut values, 0, &[], &[], 1e-6).is_err());
        assert!(
            bf16_layer_norm_accumulation(&mut values, 3, &[1.; 3], &[0.; 3], f32::NAN).is_err()
        );
    }
}
