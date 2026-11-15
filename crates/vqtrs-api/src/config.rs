//! Server configuration: a TOML file overlaid by CLI/env flags.
//!
//! Precedence (highest first): CLI flag / env var → config file → built-in
//! defaults. The config file lives at `$XDG_CONFIG_HOME/vqtrs/config.toml` by
//! default, or wherever `--config` points.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Default cap on models kept loaded per backend.
///
/// A server with no config file must not accumulate every model a workload
/// touches: an unbounded registry reached ~91-97 GB on the Studio (vqtrs-616k).
pub const DEFAULT_MAX_LOADED: usize = 4;

/// On-disk server configuration. Every field has a default, so a partial or
/// absent file is fine.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct ServerConfig {
    /// Embedding model loaded at startup (catalog code or fastembed variant).
    pub model: String,
    /// Reranker model, loaded lazily on the first `/rerank` request.
    pub rerank_model: String,
    /// Sparse model, loaded lazily on the first `/embeddings/sparse` request.
    pub sparse_model: String,
    /// BGE-M3 model, loaded lazily on the first `/embeddings/m3` request.
    pub m3_model: String,
    /// Address to bind the TCP listener to.
    pub host: String,
    /// TCP port to listen on.
    pub port: u16,
    /// Unix socket path; empty means the default (`$XDG_RUNTIME_DIR/vqtrs.sock`).
    pub socket: String,
    /// Disable the Unix socket entirely (TCP only).
    pub no_socket: bool,
    /// Extra embedding models to preload at startup and keep warm (never
    /// evicted). The default `model` is always warm.
    pub warm: Vec<String>,
    /// Max models kept loaded per backend (dense, rerank, sparse, BGE-M3), evicted
    /// least-recently-used beyond the cap. Warm models are pinned: they are never
    /// evicted, but they do occupy slots. `0` disables the bound and is only safe
    /// for a deliberately fixed model set, since resident memory then grows with
    /// every distinct model requested.
    ///
    /// This caps model *count*, not bytes: size it against the heaviest model you
    /// allow (one Qwen3-Embedding-8B on the CPU F32 path is tens of GB alone).
    pub max_loaded: usize,
    /// Max inference tasks in flight across all endpoints. Requests beyond the
    /// cap are rejected with `429` + `Retry-After` instead of queueing, so the
    /// blocking thread pool can never saturate (the vqtrs-76mr wedge). Values
    /// below 1 are treated as 1.
    ///
    /// Resident memory is bounded by `max_inflight x max_batch_texts`: the ONNX
    /// runtime retains arena memory for the largest batch shape it has run, so
    /// size both knobs together when budgeting.
    pub max_inflight: usize,
    /// Max texts accepted in one request. Larger batches are rejected with
    /// `413` instead of being attempted, bounding ONNX arena growth (which is
    /// retained for the lifetime of the process). Values below 1 are treated
    /// as 1.
    pub max_batch_texts: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model: "Qdrant/all-MiniLM-L6-v2-onnx".to_owned(),
            rerank_model: "BAAI/bge-reranker-base".to_owned(),
            sparse_model: "Qdrant/Splade_PP_en_v1".to_owned(),
            m3_model: "BAAI/bge-m3".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 8430,
            socket: String::new(),
            no_socket: false,
            warm: Vec::new(),
            max_loaded: DEFAULT_MAX_LOADED,
            max_inflight: 4,
            max_batch_texts: 256,
        }
    }
}

impl ServerConfig {
    /// Load from `path`, else the default config location, else built-in
    /// defaults when no file exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be read or parsed.
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let resolved = path.map(Path::to_path_buf).or_else(default_config_path);
        if let Some(file) = resolved
            && file.exists()
        {
            let text = std::fs::read_to_string(&file)
                .with_context(|| format!("reading {}", file.display()))?;
            return toml::from_str(&text).with_context(|| format!("parsing {}", file.display()));
        }
        Ok(Self::default())
    }

    /// The JSON schema for this config, pretty-printed.
    ///
    /// # Errors
    ///
    /// Returns an error if the schema cannot be serialized.
    pub fn json_schema() -> Result<String> {
        let schema = schemars::schema_for!(Self);
        serde_json::to_string_pretty(&schema).context("serializing JSON schema")
    }
}

/// `$XDG_CONFIG_HOME/vqtrs/config.toml` (or the platform config dir).
fn default_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|dir| dir.join("vqtrs").join("config.toml"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_and_partial_configs_are_bounded() {
        assert_eq!(ServerConfig::default().max_loaded, DEFAULT_MAX_LOADED);
        let partial: ServerConfig = toml::from_str("port = 9000").unwrap();
        assert_eq!(partial.max_loaded, DEFAULT_MAX_LOADED);
        assert_ne!(DEFAULT_MAX_LOADED, 0, "the safe default must be bounded");
    }

    #[test]
    fn missing_config_file_falls_back_to_bounded_defaults() {
        let missing = std::env::temp_dir().join("vqtrs-616k-definitely-missing/config.toml");
        assert!(!missing.exists());
        let cfg = ServerConfig::load(Some(&missing)).unwrap();
        assert_eq!(cfg.max_loaded, DEFAULT_MAX_LOADED);
    }

    #[test]
    fn explicit_values_including_zero_are_respected() {
        let cfg: ServerConfig = toml::from_str("max_loaded = 8").unwrap();
        assert_eq!(cfg.max_loaded, 8);
        let opt_out: ServerConfig = toml::from_str("max_loaded = 0").unwrap();
        assert_eq!(opt_out.max_loaded, 0);
    }
}
