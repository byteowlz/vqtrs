//! HTTP API server for vqtrs: OpenAI-compatible embeddings plus reranking,
//! backed by vqtrs-core.

mod config;
mod registry;

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::config::ServerConfig;
use crate::registry::Registry;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use clap::Parser;
use log::info;
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use vqtrs_core::{
    Engine, M3Engine, Reranker, SparseEngine, dense_models, rerank_models, sparse_models,
};

fn main() -> Result<()> {
    try_main()
}

#[tokio::main]
async fn try_main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    if cli.print_schema {
        println!("{}", ServerConfig::json_schema()?);
        return Ok(());
    }

    // Effective settings: CLI flag / env > config file > built-in default.
    let cfg = ServerConfig::load(cli.config.as_deref())?;
    let model = cli.model.unwrap_or(cfg.model);
    let rerank_model = cli.rerank_model.unwrap_or(cfg.rerank_model);
    let sparse_model = cli.sparse_model.unwrap_or(cfg.sparse_model);
    let m3_model = cli.m3_model.unwrap_or(cfg.m3_model);
    let host: IpAddr = cli.host.unwrap_or_else(|| {
        cfg.host
            .parse()
            .unwrap_or_else(|_| IpAddr::from([127, 0, 0, 1]))
    });
    let port = cli.port.unwrap_or(cfg.port);
    let no_socket = cli.no_socket || cfg.no_socket;
    let socket_path = cli
        .socket
        .or_else(|| (!cfg.socket.is_empty()).then(|| PathBuf::from(&cfg.socket)))
        .unwrap_or_else(default_socket_path);

    let max_loaded = cfg.max_loaded;
    let warm = cfg.warm;

    let dense = Arc::new(Registry::new(max_loaded));
    info!("loading embedding model: {model}");
    let engine = dense.load_pinned(&model, || Engine::load(&model).map_err(Into::into))?;
    info!(
        "loaded {} ({} dims, {:?})",
        engine.model(),
        engine.dimensions(),
        engine.backend()
    );
    for warm_model in &warm {
        if warm_model != &model {
            info!("warming model: {warm_model}");
            dense.load_pinned(warm_model, || Engine::load(warm_model).map_err(Into::into))?;
        }
    }

    let state = AppState {
        default_model: Arc::new(model),
        dense,
        default_rerank: Arc::new(rerank_model),
        rerankers: Arc::new(Registry::new(max_loaded)),
        default_sparse: Arc::new(sparse_model),
        sparse: Arc::new(Registry::new(max_loaded)),
        default_m3: Arc::new(m3_model),
        m3: Arc::new(Registry::new(max_loaded)),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/embeddings", post(embeddings))
        .route("/embeddings/sparse", post(sparse_embeddings))
        .route("/embeddings/m3", post(m3_embeddings))
        .route("/rerank", post(rerank))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Local channel: a Unix-domain socket serving the same routes. No TCP, no
    // port management, filesystem-permission auth, and the model stays warm.
    if !no_socket {
        let _ = std::fs::remove_file(&socket_path);
        let uds = tokio::net::UnixListener::bind(&socket_path)
            .with_context(|| format!("binding unix socket {}", socket_path.display()))?;
        info!("serving on unix:{}", socket_path.display());
        let uds_app = app.clone();
        tokio::spawn(async move {
            if let Err(err) = axum::serve(uds, uds_app).await {
                log::error!("unix socket server error: {err}");
            }
        });
    }

    let addr = SocketAddr::new(host, port);
    info!("serving on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Default local socket path: `$XDG_RUNTIME_DIR/vqtrs.sock` when that directory
/// exists, else `<temp-dir>/vqtrs.sock`.
fn default_socket_path() -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .unwrap_or_else(std::env::temp_dir)
        .join("vqtrs.sock")
}

#[derive(Debug, Parser)]
#[command(author, version, about = "vqtrs embeddings + reranking server")]
struct Cli {
    /// Config file path (default: `vqtrs/config.toml` under `$XDG_CONFIG_HOME`)
    #[arg(long, env = "VQTRS_CONFIG")]
    config: Option<PathBuf>,
    /// Print the config JSON schema and exit
    #[arg(long)]
    print_schema: bool,
    /// Embedding model to load (catalog code or fastembed variant)
    #[arg(short, long, env = "VQTRS_MODEL")]
    model: Option<String>,
    /// Reranker model, loaded lazily on first /rerank request
    #[arg(long, env = "VQTRS_RERANK_MODEL")]
    rerank_model: Option<String>,
    /// Sparse model, loaded lazily on first /embeddings/sparse request
    #[arg(long, env = "VQTRS_SPARSE_MODEL")]
    sparse_model: Option<String>,
    /// BGE-M3 model, loaded lazily on first /embeddings/m3 request
    #[arg(long, env = "VQTRS_M3_MODEL")]
    m3_model: Option<String>,
    /// Address to bind
    #[arg(long)]
    host: Option<IpAddr>,
    /// Port to listen on
    #[arg(short, long, env = "VQTRS_PORT")]
    port: Option<u16>,
    /// Override the Unix socket path (default under `$XDG_RUNTIME_DIR`)
    #[arg(long, env = "VQTRS_SOCKET")]
    socket: Option<PathBuf>,
    /// Do not bind a Unix socket (TCP only)
    #[arg(long)]
    no_socket: bool,
}

/// Shared server state: a keep-warm registry per backend plus the default model
/// for requests that omit one.
#[derive(Clone)]
struct AppState {
    default_model: Arc<String>,
    dense: Arc<Registry<Engine>>,
    default_rerank: Arc<String>,
    rerankers: Arc<Registry<Reranker>>,
    default_sparse: Arc<String>,
    sparse: Arc<Registry<SparseEngine>>,
    default_m3: Arc<String>,
    m3: Arc<Registry<M3Engine>>,
}

/// Choose the request's model when present and non-empty, else the default.
const fn pick<'a>(requested: Option<&'a str>, default: &'a str) -> &'a str {
    match requested {
        Some(model) if !model.is_empty() => model,
        _ => default,
    }
}

/// An error returned to an HTTP client as a JSON body.
struct ApiError {
    status: StatusCode,
    message: String,
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: err.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(ErrorEnvelope {
            error: ErrorBody {
                message: self.message,
            },
        });
        (self.status, body).into_response()
    }
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
}

#[derive(Serialize)]
struct RootResponse {
    name: &'static str,
    version: &'static str,
}

async fn root() -> Json<RootResponse> {
    Json(RootResponse {
        name: env!("CARGO_PKG_NAME"),
        version: env!("CARGO_PKG_VERSION"),
    })
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
struct ModelEntry {
    id: &'static str,
    object: &'static str,
    owned_by: &'static str,
    task: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

async fn list_models() -> Json<ModelList> {
    let mut data: Vec<ModelEntry> = dense_models()
        .iter()
        .map(|m| ModelEntry {
            id: m.code,
            object: "model",
            owned_by: "vqtrs",
            task: "embedding",
            dimensions: Some(m.dimensions),
        })
        .collect();
    data.extend(sparse_models().iter().map(|m| ModelEntry {
        id: m.code,
        object: "model",
        owned_by: "vqtrs",
        task: if m.joint_dense {
            "dense+sparse"
        } else {
            "sparse"
        },
        dimensions: None,
    }));
    data.extend(rerank_models().iter().map(|m| ModelEntry {
        id: m.code,
        object: "model",
        owned_by: "vqtrs",
        task: "rerank",
        dimensions: None,
    }));
    Json(ModelList {
        object: "list",
        data,
    })
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbedInput {
    One(String),
    Many(Vec<String>),
}

impl EmbedInput {
    /// Normalise the OpenAI string-or-array input into a list of texts.
    fn into_texts(self) -> Vec<String> {
        match self {
            Self::One(text) => vec![text],
            Self::Many(texts) => texts,
        }
    }
}

#[derive(Deserialize)]
struct EmbedRequest {
    #[serde(default)]
    model: Option<String>,
    input: EmbedInput,
}

#[derive(Serialize)]
struct EmbedResponse {
    object: &'static str,
    data: Vec<EmbedData>,
    model: String,
    usage: Usage,
}

#[derive(Serialize)]
struct EmbedData {
    object: &'static str,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, ApiError> {
    let texts = req.input.into_texts();
    let approx_tokens: usize = texts.iter().map(|t| t.len().div_ceil(4)).sum();

    let (model, vectors, tokens) = tokio::task::spawn_blocking(move || -> Result<_> {
        let name = pick(req.model.as_deref(), &state.default_model);
        let engine = state
            .dense
            .get_or_load(name, || Engine::load(name).map_err(Into::into))?;
        let tokens = engine.count_tokens(&texts);
        let vectors = engine.embed_batch(&texts)?;
        Ok((engine.model().to_owned(), vectors, tokens))
    })
    .await
    .map_err(|e| join_error(&e))??;

    let prompt_tokens = tokens.unwrap_or(approx_tokens);
    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbedData {
            object: "embedding",
            embedding,
            index,
        })
        .collect();

    Ok(Json(EmbedResponse {
        object: "list",
        data,
        model,
        usage: Usage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    }))
}

#[derive(Serialize)]
struct SparseData {
    index: usize,
    indices: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Serialize)]
struct SparseResponse {
    object: &'static str,
    data: Vec<SparseData>,
    model: String,
}

async fn sparse_embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<SparseResponse>, ApiError> {
    let texts = req.input.into_texts();
    let (model, vectors) = tokio::task::spawn_blocking(move || -> Result<_> {
        let name = pick(req.model.as_deref(), &state.default_sparse);
        let engine = state
            .sparse
            .get_or_load(name, || SparseEngine::load(name).map_err(Into::into))?;
        let vectors = engine.embed_batch(&texts)?;
        Ok((engine.model().to_owned(), vectors))
    })
    .await
    .map_err(|e| join_error(&e))??;

    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, v)| SparseData {
            index,
            indices: v.indices,
            values: v.values,
        })
        .collect();

    Ok(Json(SparseResponse {
        object: "list",
        data,
        model,
    }))
}

#[derive(Serialize)]
struct M3Data {
    index: usize,
    dense: Vec<f32>,
    sparse: SparsePair,
}

#[derive(Serialize)]
struct SparsePair {
    indices: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Serialize)]
struct M3Response {
    object: &'static str,
    data: Vec<M3Data>,
    model: String,
}

async fn m3_embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<M3Response>, ApiError> {
    let texts = req.input.into_texts();
    let (model, vectors) = tokio::task::spawn_blocking(move || -> Result<_> {
        let name = pick(req.model.as_deref(), &state.default_m3);
        let engine = state
            .m3
            .get_or_load(name, || M3Engine::load(name).map_err(Into::into))?;
        let vectors = engine.embed_batch(&texts)?;
        Ok((engine.model().to_owned(), vectors))
    })
    .await
    .map_err(|e| join_error(&e))??;

    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, v)| M3Data {
            index,
            dense: v.dense,
            sparse: SparsePair {
                indices: v.sparse.indices,
                values: v.sparse.values,
            },
        })
        .collect();

    Ok(Json(M3Response {
        object: "list",
        data,
        model,
    }))
}

#[derive(Deserialize)]
struct RerankRequest {
    #[serde(default)]
    model: Option<String>,
    query: String,
    documents: Vec<String>,
    #[serde(default)]
    top_n: Option<usize>,
    #[serde(default)]
    return_documents: bool,
}

#[derive(Serialize)]
struct RerankResponse {
    model: String,
    results: Vec<RerankResultBody>,
}

#[derive(Serialize)]
struct RerankResultBody {
    index: usize,
    relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    document: Option<RerankDoc>,
}

#[derive(Serialize)]
struct RerankDoc {
    text: String,
}

async fn rerank(
    State(state): State<AppState>,
    Json(req): Json<RerankRequest>,
) -> Result<Json<RerankResponse>, ApiError> {
    let ranked = tokio::task::spawn_blocking(move || -> Result<_> {
        let name = pick(req.model.as_deref(), &state.default_rerank);
        let reranker = state
            .rerankers
            .get_or_load(name, || Reranker::load(name).map_err(Into::into))?;
        let ranked =
            reranker.rerank(&req.query, &req.documents, req.return_documents, req.top_n)?;
        Ok((reranker.model().to_owned(), ranked))
    })
    .await
    .map_err(|e| join_error(&e))??;

    let (model, ranked) = ranked;
    let results = ranked
        .into_iter()
        .map(|r| RerankResultBody {
            index: r.index,
            relevance_score: r.score,
            document: r.document.map(|text| RerankDoc { text }),
        })
        .collect();

    Ok(Json(RerankResponse { model, results }))
}

/// Map a `spawn_blocking` join failure into an API error.
fn join_error(err: &tokio::task::JoinError) -> ApiError {
    ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("inference task failed: {err}"),
    }
}
