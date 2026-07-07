//! `vqtrs pull` — fuzzy-pick a model with fzf and pre-download it into the cache.

use std::io::{Write, stdin};
use std::process::{Command, Stdio};

use anyhow::{Context, Result};
use vqtrs_core::{
    Engine, M3Engine, Reranker, SparseEngine, dense_models, rerank_models, sparse_models,
};

/// A catalog row offered for download.
#[derive(Clone)]
struct Row {
    code: &'static str,
    variant: &'static str,
    task: &'static str,
    line: String,
}

/// Pick a model (exact arg, fzf, or numbered prompt) and download it.
///
/// # Errors
///
/// Returns an error if the download fails.
pub fn run(query: Option<&str>) -> Result<()> {
    let rows = rows();

    // An exact code/variant downloads directly; otherwise fuzzy-pick.
    let selected = match query {
        Some(q) => match find_exact(&rows, q) {
            Some(row) => Some(row),
            None => pick(&rows, Some(q))?,
        },
        None => pick(&rows, None)?,
    };

    let Some(row) = selected else {
        eprintln!("nothing selected");
        return Ok(());
    };

    eprintln!("downloading {} ({})…", row.code, row.task);
    download(&row)?;
    println!("{}", row.code);
    eprintln!("✓ cached {}", row.code);
    Ok(())
}

fn find_exact(rows: &[Row], q: &str) -> Option<Row> {
    rows.iter()
        .find(|r| r.code == q || (!r.variant.is_empty() && r.variant == q))
        .cloned()
}

fn pick(rows: &[Row], query: Option<&str>) -> Result<Option<Row>> {
    let lines: Vec<&str> = rows.iter().map(|r| r.line.as_str()).collect();
    run_fzf(&lines, query).map_or_else(
        |_| pick_numbered(rows), // fzf not available
        |selection| Ok(selection.and_then(|s| rows.iter().find(|r| r.line == s).cloned())),
    )
}

/// Run fzf over `lines`; `Ok(None)` means the user cancelled. Errors only if fzf
/// cannot be spawned (so the caller can fall back).
fn run_fzf(lines: &[&str], query: Option<&str>) -> Result<Option<String>> {
    let mut cmd = Command::new("fzf");
    cmd.arg("--prompt=model> ")
        .arg("--height=40%")
        .arg("--reverse")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped());
    if let Some(q) = query {
        cmd.arg(format!("--query={q}"));
    }
    let mut child = cmd.spawn().context("spawning fzf")?;
    {
        let mut child_stdin = child.stdin.take().context("opening fzf stdin")?;
        child_stdin
            .write_all(lines.join("\n").as_bytes())
            .context("writing models to fzf")?;
    }
    let output = child.wait_with_output().context("running fzf")?;
    if !output.status.success() {
        return Ok(None);
    }
    let selection = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    Ok((!selection.is_empty()).then_some(selection))
}

fn pick_numbered(rows: &[Row]) -> Result<Option<Row>> {
    eprintln!("(fzf not found — pick a number)");
    for (i, row) in rows.iter().enumerate() {
        eprintln!("{:>3}  {}", i + 1, row.line);
    }
    eprint!("model number: ");
    std::io::stderr().flush().ok();
    let mut buf = String::new();
    stdin().read_line(&mut buf).context("reading selection")?;
    let choice: usize = match buf.trim().parse() {
        Ok(n) => n,
        Err(_) => return Ok(None),
    };
    Ok(choice.checked_sub(1).and_then(|i| rows.get(i)).cloned())
}

fn download(row: &Row) -> Result<()> {
    match row.task {
        "rerank" => {
            Reranker::load(row.code).context("downloading reranker")?;
        }
        "sparse" => {
            SparseEngine::load(row.code).context("downloading sparse model")?;
        }
        "dense+sparse" => {
            M3Engine::load(row.code).context("downloading BGE-M3 model")?;
        }
        _ => {
            Engine::load(row.code).context("downloading embedding model")?;
        }
    }
    Ok(())
}

/// Format a fixed-width row so columns line up in fzf.
fn row_line(code: &str, dims: &str, backend: &str, task: &str, desc: &str) -> String {
    format!("{code:<52} {dims:>6}  {backend:<6} {task:<12} {desc}")
}

fn rows() -> Vec<Row> {
    let mut rows: Vec<Row> = dense_models()
        .iter()
        .map(|m| Row {
            code: m.code,
            variant: m.variant,
            task: "embedding",
            line: row_line(
                m.code,
                &format!("{}d", m.dimensions),
                &format!("{:?}", m.backend),
                "embedding",
                m.description,
            ),
        })
        .collect();
    rows.extend(sparse_models().iter().map(|m| {
        let task = if m.joint_dense {
            "dense+sparse"
        } else {
            "sparse"
        };
        Row {
            code: m.code,
            variant: m.variant,
            task,
            line: row_line(m.code, "", "", task, m.description),
        }
    }));
    rows.extend(rerank_models().iter().map(|m| Row {
        code: m.code,
        variant: m.variant,
        task: "rerank",
        line: row_line(m.code, "", "", "rerank", m.description),
    }));
    rows
}
