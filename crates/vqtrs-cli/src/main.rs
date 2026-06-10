//! `vqtrs` — command-line embeddings and reranking over the vqtrs-core engine.

mod daemon;
mod pull;
mod service;

use std::io::{Read, Write};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use vqtrs_core::{
    Engine, M3Engine, Reranker, SparseEngine, dense_models, rerank_models, sparse_models,
};

use crate::service::{ServiceManager, StartOutcome, Status};

const DEFAULT_MODEL: &str = "Qdrant/all-MiniLM-L6-v2-onnx";
const DEFAULT_RERANK_MODEL: &str = "BAAI/bge-reranker-base";
const DEFAULT_SPARSE_MODEL: &str = "Qdrant/Splade_PP_en_v1";
const DEFAULT_M3_MODEL: &str = "BAAI/bge-m3";

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Embed(args) => embed(args),
        Command::Sparse(args) => sparse(args),
        Command::M3(args) => m3(args),
        Command::Rerank(args) => rerank(args),
        Command::Models(args) => models(&args),
        Command::Pull(args) => pull::run(args.query.as_deref()),
        Command::Service { action } => handle_service(action),
    }
}

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Embeddings + reranking over the vqtrs engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Embed text; reads positional args or, if none, lines from stdin
    Embed(EmbedArgs),
    /// Sparse-embed text (SPLADE / BGE-M3 sparse)
    Sparse(SparseArgs),
    /// Joint dense+sparse embed via BGE-M3
    M3(M3Args),
    /// Rerank documents against a query
    Rerank(RerankArgs),
    /// List available embedding and reranker models
    Models(ModelsArgs),
    /// Fuzzy-pick a model (fzf) and pre-download it into the cache
    Pull(PullArgs),
    /// Manage the vqtrs-api server daemon
    Service {
        #[command(subcommand)]
        action: ServiceAction,
    },
}

#[derive(Debug, Subcommand)]
enum ServiceAction {
    /// Start the server in the background (detached)
    Start(ServiceOpts),
    /// Run the server in the foreground (blocking)
    Run(ServiceOpts),
    /// Stop the background server
    Stop,
    /// Restart the background server
    Restart(ServiceOpts),
    /// Show server status
    Status,
    /// Install + enable a systemd user unit (autostart on login)
    Enable(ServiceOpts),
    /// Disable + remove the systemd user unit
    Disable,
}

#[derive(Debug, Args)]
struct ServiceOpts {
    /// Address to bind
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Port to listen on
    #[arg(short, long, default_value = "3000")]
    port: u16,
    /// Embedding model (else the server default / `VQTRS_MODEL`)
    #[arg(long)]
    model: Option<String>,
    /// Reranker model override
    #[arg(long)]
    rerank_model: Option<String>,
    /// Sparse model override
    #[arg(long)]
    sparse_model: Option<String>,
    /// BGE-M3 model override
    #[arg(long)]
    m3_model: Option<String>,
}

impl ServiceOpts {
    fn to_args(&self) -> Vec<String> {
        service::api_args(
            &self.host,
            self.port,
            self.model.as_deref(),
            self.rerank_model.as_deref(),
            self.sparse_model.as_deref(),
            self.m3_model.as_deref(),
        )
    }
}

#[derive(Debug, Args)]
struct EmbedArgs {
    /// Texts to embed; if omitted, each stdin line is embedded
    texts: Vec<String>,
    /// Model code or fastembed variant (default: a running daemon's model, else the built-in default)
    #[arg(short, long, env = "VQTRS_MODEL")]
    model: Option<String>,
    /// Always load in-process; never use a running daemon
    #[arg(long)]
    no_daemon: bool,
    /// Pretty-print JSON output
    #[arg(long)]
    pretty: bool,
}

#[derive(Debug, Args)]
struct SparseArgs {
    /// Texts to embed; if omitted, each stdin line is embedded
    texts: Vec<String>,
    /// Sparse model code or fastembed variant
    #[arg(short, long, env = "VQTRS_SPARSE_MODEL")]
    model: Option<String>,
    /// Always load in-process; never use a running daemon
    #[arg(long)]
    no_daemon: bool,
    /// Pretty-print JSON output
    #[arg(long)]
    pretty: bool,
}

#[derive(Debug, Args)]
struct M3Args {
    /// Texts to embed; if omitted, each stdin line is embedded
    texts: Vec<String>,
    /// BGE-M3 model code or fastembed variant
    #[arg(short, long, env = "VQTRS_M3_MODEL")]
    model: Option<String>,
    /// Always load in-process; never use a running daemon
    #[arg(long)]
    no_daemon: bool,
    /// Pretty-print JSON output
    #[arg(long)]
    pretty: bool,
}

#[derive(Debug, Args)]
struct RerankArgs {
    /// The query to rank documents against
    #[arg(short, long)]
    query: String,
    /// Documents to rank; if omitted, each stdin line is a document
    documents: Vec<String>,
    /// Reranker model code or fastembed variant
    #[arg(short, long, env = "VQTRS_RERANK_MODEL")]
    model: Option<String>,
    /// Keep only the top N results
    #[arg(long)]
    top_n: Option<usize>,
    /// Include document text in the output
    #[arg(long)]
    return_documents: bool,
    /// Always load in-process; never use a running daemon
    #[arg(long)]
    no_daemon: bool,
    /// Pretty-print JSON output
    #[arg(long)]
    pretty: bool,
}

#[derive(Debug, Args)]
struct ModelsArgs {
    /// Emit JSON instead of a table
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct PullArgs {
    /// Exact model code/variant to download, or an initial fzf filter
    query: Option<String>,
}

/// POST `body` to a warm daemon over its Unix socket when `enabled`; returns the
/// raw response body, or `None` to fall back to in-process loading.
fn try_daemon<B: Serialize>(route: &str, body: &B, enabled: bool) -> Option<String> {
    if !enabled {
        return None;
    }
    let socket = daemon::live()?;
    let payload = serde_json::to_string(body).ok()?;
    daemon::post_json(&socket, route, &payload).ok()
}

#[derive(Serialize)]
struct InputBody<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    input: &'a [String],
}

#[derive(Serialize)]
struct EmbedOutput {
    model: String,
    dimensions: usize,
    data: Vec<EmbedItem>,
}

#[derive(Serialize, Deserialize)]
struct EmbedItem {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct DaemonEmbed {
    model: String,
    data: Vec<EmbedItem>,
}

fn embed(args: EmbedArgs) -> Result<()> {
    let texts = inputs_or_stdin(args.texts)?;

    let body = InputBody {
        model: args.model.as_deref(),
        input: &texts,
    };
    if let Some(resp) = try_daemon("/v1/embeddings", &body, !args.no_daemon) {
        let parsed: DaemonEmbed = serde_json::from_str(&resp).context("parsing daemon response")?;
        let dimensions = parsed.data.first().map_or(0, |i| i.embedding.len());
        let output = EmbedOutput {
            model: parsed.model,
            dimensions,
            data: parsed.data,
        };
        return print_json(&output, args.pretty);
    }

    let model = args.model.as_deref().unwrap_or(DEFAULT_MODEL);
    let engine = Engine::load(model).context("loading embedding model")?;
    let vectors = engine.embed_batch(&texts).context("embedding")?;

    let output = EmbedOutput {
        model: engine.model().to_owned(),
        dimensions: engine.dimensions(),
        data: vectors
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| EmbedItem { index, embedding })
            .collect(),
    };
    print_json(&output, args.pretty)
}

#[derive(Serialize, Deserialize)]
struct SparseItem {
    index: usize,
    indices: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct SparseOutput {
    model: String,
    data: Vec<SparseItem>,
}

fn sparse(args: SparseArgs) -> Result<()> {
    let texts = inputs_or_stdin(args.texts)?;

    let body = InputBody {
        model: args.model.as_deref(),
        input: &texts,
    };
    if let Some(resp) = try_daemon("/embeddings/sparse", &body, !args.no_daemon) {
        let output: SparseOutput =
            serde_json::from_str(&resp).context("parsing daemon response")?;
        return print_json(&output, args.pretty);
    }

    let model = args.model.as_deref().unwrap_or(DEFAULT_SPARSE_MODEL);
    let engine = SparseEngine::load(model).context("loading sparse model")?;
    let vectors = engine.embed_batch(&texts).context("sparse embedding")?;

    let output = SparseOutput {
        model: engine.model().to_owned(),
        data: vectors
            .into_iter()
            .enumerate()
            .map(|(index, v)| SparseItem {
                index,
                indices: v.indices,
                values: v.values,
            })
            .collect(),
    };
    print_json(&output, args.pretty)
}

#[derive(Serialize, Deserialize)]
struct M3Item {
    index: usize,
    dense: Vec<f32>,
    sparse: SparsePair,
}

#[derive(Serialize, Deserialize)]
struct SparsePair {
    indices: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct M3Output {
    model: String,
    data: Vec<M3Item>,
}

fn m3(args: M3Args) -> Result<()> {
    let texts = inputs_or_stdin(args.texts)?;

    let body = InputBody {
        model: args.model.as_deref(),
        input: &texts,
    };
    if let Some(resp) = try_daemon("/embeddings/m3", &body, !args.no_daemon) {
        let output: M3Output = serde_json::from_str(&resp).context("parsing daemon response")?;
        return print_json(&output, args.pretty);
    }

    let model = args.model.as_deref().unwrap_or(DEFAULT_M3_MODEL);
    let engine = M3Engine::load(model).context("loading BGE-M3 model")?;
    let vectors = engine.embed_batch(&texts).context("m3 embedding")?;

    let output = M3Output {
        model: engine.model().to_owned(),
        data: vectors
            .into_iter()
            .enumerate()
            .map(|(index, v)| M3Item {
                index,
                dense: v.dense,
                sparse: SparsePair {
                    indices: v.sparse.indices,
                    values: v.sparse.values,
                },
            })
            .collect(),
    };
    print_json(&output, args.pretty)
}

#[derive(Serialize)]
struct RerankOutput {
    model: String,
    results: Vec<RerankItem>,
}

#[derive(Serialize)]
struct RerankItem {
    index: usize,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    document: Option<String>,
}

#[derive(Serialize)]
struct RerankBody<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    query: &'a str,
    documents: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    top_n: Option<usize>,
    return_documents: bool,
}

#[derive(Deserialize)]
struct DaemonRerank {
    model: String,
    results: Vec<DaemonRerankItem>,
}

#[derive(Deserialize)]
struct DaemonRerankItem {
    index: usize,
    relevance_score: f32,
    #[serde(default)]
    document: Option<DaemonDoc>,
}

#[derive(Deserialize)]
struct DaemonDoc {
    text: String,
}

fn rerank(args: RerankArgs) -> Result<()> {
    let documents = inputs_or_stdin(args.documents)?;

    let body = RerankBody {
        model: args.model.as_deref(),
        query: &args.query,
        documents: &documents,
        top_n: args.top_n,
        return_documents: args.return_documents,
    };
    if let Some(resp) = try_daemon("/rerank", &body, !args.no_daemon) {
        let parsed: DaemonRerank =
            serde_json::from_str(&resp).context("parsing daemon response")?;
        let output = RerankOutput {
            model: parsed.model,
            results: parsed
                .results
                .into_iter()
                .map(|r| RerankItem {
                    index: r.index,
                    score: r.relevance_score,
                    document: r.document.map(|d| d.text),
                })
                .collect(),
        };
        return print_json(&output, args.pretty);
    }

    let model = args.model.as_deref().unwrap_or(DEFAULT_RERANK_MODEL);
    let reranker = Reranker::load(model).context("loading reranker model")?;
    let ranked = reranker
        .rerank(&args.query, &documents, args.return_documents, args.top_n)
        .context("reranking")?;

    let output = RerankOutput {
        model: reranker.model().to_owned(),
        results: ranked
            .into_iter()
            .map(|r| RerankItem {
                index: r.index,
                score: r.score,
                document: r.document,
            })
            .collect(),
    };
    print_json(&output, args.pretty)
}

fn models(args: &ModelsArgs) -> Result<()> {
    if args.json {
        let entries: Vec<ModelRow> = model_rows();
        return print_json(&entries, true);
    }

    println!("EMBEDDING MODELS");
    for m in dense_models() {
        println!(
            "  {:<48} {:>5}d  {:<6} {}",
            m.code,
            m.dimensions,
            format!("{:?}", m.backend),
            m.description
        );
    }
    println!("\nSPARSE MODELS");
    for m in sparse_models() {
        let kind = if m.joint_dense {
            "dense+sparse"
        } else {
            "sparse"
        };
        println!("  {:<48} {:<12} {}", m.code, kind, m.description);
    }
    println!("\nRERANKER MODELS");
    for m in rerank_models() {
        println!("  {:<48} {}", m.code, m.description);
    }
    Ok(())
}

#[derive(Serialize)]
struct ModelRow {
    code: &'static str,
    task: &'static str,
    backend: Option<String>,
    dimensions: Option<usize>,
    description: &'static str,
}

fn model_rows() -> Vec<ModelRow> {
    let mut rows: Vec<ModelRow> = dense_models()
        .iter()
        .map(|m| ModelRow {
            code: m.code,
            task: "embedding",
            backend: Some(format!("{:?}", m.backend)),
            dimensions: Some(m.dimensions),
            description: m.description,
        })
        .collect();
    rows.extend(sparse_models().iter().map(|m| ModelRow {
        code: m.code,
        task: if m.joint_dense {
            "dense+sparse"
        } else {
            "sparse"
        },
        backend: None,
        dimensions: None,
        description: m.description,
    }));
    rows.extend(rerank_models().iter().map(|m| ModelRow {
        code: m.code,
        task: "rerank",
        backend: None,
        dimensions: None,
        description: m.description,
    }));
    rows
}

fn handle_service(action: ServiceAction) -> Result<()> {
    let mgr = ServiceManager::new()?;
    match action {
        ServiceAction::Start(opts) => {
            let outcome = mgr.start(&opts.to_args(), opts.port)?;
            report_start(&outcome, &opts.host, &mgr);
            Ok(())
        }
        ServiceAction::Run(opts) => service::run(&opts.to_args()),
        ServiceAction::Stop => {
            mgr.stop()?;
            println!("✓ server stopped");
            Ok(())
        }
        ServiceAction::Restart(opts) => {
            let outcome = mgr.restart(&opts.to_args(), opts.port)?;
            report_start(&outcome, &opts.host, &mgr);
            Ok(())
        }
        ServiceAction::Status => {
            print_status(&mgr);
            Ok(())
        }
        ServiceAction::Enable(opts) => {
            let path = service::enable(&opts.to_args())?;
            println!("✓ enabled systemd user unit: {}", path.display());
            Ok(())
        }
        ServiceAction::Disable => {
            service::disable()?;
            println!("✓ disabled systemd user unit");
            Ok(())
        }
    }
}

fn report_start(outcome: &StartOutcome, host: &str, mgr: &ServiceManager) {
    match outcome {
        StartOutcome::Ready { pid, port } => {
            println!("✓ server running (pid {pid}) on http://{host}:{port}");
        }
        StartOutcome::Initializing { pid } => {
            println!(
                "✓ server starting (pid {pid}); model still loading — check `vqtrs server status`\n  logs: {}",
                mgr.log_file().display()
            );
        }
    }
}

fn print_status(mgr: &ServiceManager) {
    match mgr.status() {
        Status::Running { pid, port } => {
            let port = port.map_or_else(|| "?".to_owned(), |p| p.to_string());
            println!("running   pid {pid}, port {port}");
        }
        Status::Dead { pid } => println!("dead      stale pid {pid} (process gone)"),
        Status::Stopped => println!("stopped"),
    }
    #[cfg(target_os = "linux")]
    if let Some(state) = service::systemd_state() {
        println!("systemd   {} / {}", state.enabled, state.active);
    }
    if let Some(socket) = daemon::live() {
        println!("socket    {} (accepting)", socket.display());
    }
}

/// Use the provided inputs, or read newline-separated entries from stdin when
/// the list is empty.
fn inputs_or_stdin(provided: Vec<String>) -> Result<Vec<String>> {
    if !provided.is_empty() {
        return Ok(provided);
    }
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .context("reading stdin")?;
    Ok(buf
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect())
}

/// Serialize `value` as JSON to stdout.
fn print_json<T: Serialize>(value: &T, pretty: bool) -> Result<()> {
    let text = if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .context("serializing output")?;
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(text.as_bytes())
        .context("writing output")?;
    stdout.write_all(b"\n").context("writing output")?;
    Ok(())
}
