//! Server configuration: a TOML file overlaid by CLI/env flags.
//!
//! Precedence (highest first): CLI flag / env var → config file → built-in
//! defaults. The config file lives at `$XDG_CONFIG_HOME/vqtrs/config.toml` by
//! default, or wherever `--config` points.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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
    /// Max models kept loaded per backend; `0` = unbounded. Warm models are
    /// pinned and do not count toward eviction.
    pub max_loaded: usize,
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
            max_loaded: 0,
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
