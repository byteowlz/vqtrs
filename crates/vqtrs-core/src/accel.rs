//! Hardware acceleration selection for the ONNX and Qwen3 backends.
//!
//! ONNX models register `ort` execution providers (CUDA / TensorRT / CoreML /
//! DirectML) according to the enabled cargo features; the Qwen3 candle backend
//! picks a GPU [`candle_core::Device`] when a `qwen3-*` feature is enabled and
//! the device is available. With no acceleration feature, both run on the CPU.

use fastembed::ExecutionProviderDispatch;

/// ONNX execution providers to register, highest priority first.
///
/// Empty when no ONNX acceleration feature is enabled, which leaves fastembed on
/// its default CPU provider. Unavailable providers fall back to CPU at runtime.
#[cfg(not(any(
    feature = "cuda",
    feature = "tensorrt",
    feature = "coreml",
    feature = "directml"
)))]
#[must_use]
pub const fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    Vec::new()
}

/// ONNX execution providers to register, highest priority first.
#[cfg(any(
    feature = "cuda",
    feature = "tensorrt",
    feature = "coreml",
    feature = "directml"
))]
#[must_use]
pub fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    let mut providers: Vec<ExecutionProviderDispatch> = Vec::new();
    #[cfg(feature = "cuda")]
    providers.push(ort::execution_providers::CUDAExecutionProvider::default().build());
    #[cfg(feature = "tensorrt")]
    providers.push(ort::execution_providers::TensorRTExecutionProvider::default().build());
    #[cfg(feature = "coreml")]
    providers.push(ort::execution_providers::CoreMLExecutionProvider::default().build());
    #[cfg(feature = "directml")]
    providers.push(ort::execution_providers::DirectMLExecutionProvider::default().build());
    providers
}

/// Candle device and dtype for the Qwen3 backend. Plain CPU/F32 when no
/// `qwen3-*` GPU feature is enabled.
#[cfg(all(
    feature = "qwen3",
    not(any(feature = "qwen3-cuda", feature = "qwen3-metal"))
))]
#[must_use]
pub const fn qwen3_device() -> (candle_core::Device, candle_core::DType) {
    (candle_core::Device::Cpu, candle_core::DType::F32)
}

/// Candle device and dtype for the Qwen3 backend, preferring a GPU when a
/// `qwen3-*` feature is enabled and the device initialises, else CPU/F32.
#[cfg(any(feature = "qwen3-cuda", feature = "qwen3-metal"))]
#[must_use]
pub fn qwen3_device() -> (candle_core::Device, candle_core::DType) {
    use candle_core::{DType, Device};

    #[cfg(feature = "qwen3-cuda")]
    if let Ok(device) = Device::new_cuda(0) {
        return (device, DType::BF16);
    }
    #[cfg(feature = "qwen3-metal")]
    if let Ok(device) = Device::new_metal(0) {
        return (device, DType::F16);
    }
    (Device::Cpu, DType::F32)
}
