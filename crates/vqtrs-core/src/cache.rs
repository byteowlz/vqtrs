//! Cache directory resolution for downloaded ONNX model files.

use std::env;
use std::path::PathBuf;

use crate::error::Result;

/// Directory where ONNX model files are cached, created if missing.
///
/// Resolves to the standard Hugging Face hub cache, so ONNX models share one
/// location with the Qwen3 candle backend (which downloads via `hf-hub`) and
/// any other Hugging Face tooling. Resolution mirrors `hf-hub`/`huggingface_hub`:
/// `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `<os-cache-dir>/huggingface/hub`.
///
/// # Errors
///
/// Returns an error if the directory cannot be created.
pub fn model_cache_dir() -> Result<PathBuf> {
    let path = hf_hub_cache_dir();
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

/// Resolve the Hugging Face hub cache directory (without creating it).
fn hf_hub_cache_dir() -> PathBuf {
    if let Some(dir) = env::var_os("HF_HUB_CACHE") {
        return PathBuf::from(dir);
    }
    if let Some(home) = env::var_os("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    let base = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("."));
    base.join("huggingface").join("hub")
}
