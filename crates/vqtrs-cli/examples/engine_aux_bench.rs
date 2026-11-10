//! Bounded differential benchmark for the sparse, BGE-M3 and rerank engines.
//! Usage: `engine_aux_bench sparse|m3|rerank MODEL CASES_JSON REPORT_JSON [all|short]`.
//! Warm models, all text work included, exact output bits checked on every timed call.

use std::{hint::black_box, path::Path, time::Instant};

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use vqtrs_core::{M3Engine, Reranker, SparseEngine};

#[derive(Deserialize)]
struct Input {
    name: String,
    prompt: String,
    text: String,
}

/// One model output in a form that supports exact comparison.
#[derive(PartialEq, Serialize)]
enum Output {
    Sparse(Vec<(usize, u32)>),
    DenseSparse(Vec<u32>, Vec<(usize, u32)>),
    Ranked(Vec<(usize, u32)>),
}

#[derive(Serialize)]
struct Case {
    id: String,
    rows: usize,
    ms: Vec<f64>,
    iterations: Vec<usize>,
    output: Output,
}

#[derive(Serialize)]
struct Report {
    kind: String,
    model: String,
    load_ms: f64,
    warmups: usize,
    trials: usize,
    cases: Vec<Case>,
}

fn sparse_bits(vector: &vqtrs_core::SparseVector) -> Vec<(usize, u32)> {
    vector
        .indices
        .iter()
        .copied()
        .zip(vector.values.iter().map(|v| v.to_bits()))
        .collect()
}

fn finite(bits: &[(usize, u32)]) -> bool {
    bits.iter().all(|(_, raw)| f32::from_bits(*raw).is_finite())
}

fn run_sparse(engine: &SparseEngine, texts: &[String]) -> Result<Output> {
    let vectors = engine.embed_batch(texts)?;
    ensure!(vectors.len() == texts.len(), "sparse row count changed");
    let mut rows = Vec::new();
    for vector in vectors {
        ensure!(
            vector.indices.len() == vector.values.len() && !vector.indices.is_empty(),
            "malformed sparse vector"
        );
        let bits = sparse_bits(&vector);
        ensure!(finite(&bits), "non-finite sparse weight");
        rows.extend(bits);
    }
    Ok(Output::Sparse(rows))
}

fn run_m3(engine: &M3Engine, texts: &[String]) -> Result<Output> {
    let vectors = engine.embed_batch(texts)?;
    ensure!(vectors.len() == texts.len(), "m3 row count changed");
    let mut dense = Vec::new();
    let mut sparse = Vec::new();
    for vector in vectors {
        ensure!(
            vector.sparse.indices.len() == vector.sparse.values.len(),
            "malformed m3 sparse vector"
        );
        let row: Vec<u32> = vector.dense.iter().map(|v| v.to_bits()).collect();
        ensure!(
            row.iter().all(|raw| f32::from_bits(*raw).is_finite()),
            "non-finite m3 dense value"
        );
        dense.extend(row);
        sparse.extend(sparse_bits(&vector.sparse));
    }
    ensure!(finite(&sparse), "non-finite m3 sparse weight");
    Ok(Output::DenseSparse(dense, sparse))
}

fn run_rerank(engine: &Reranker, query: &str, documents: &[String]) -> Result<Output> {
    let ranked = engine.rerank(query, documents, false, None)?;
    ensure!(ranked.len() == documents.len(), "rerank row count changed");
    ensure!(
        ranked.iter().all(|r| r.index < documents.len()),
        "rerank index out of range"
    );
    let bits: Vec<(usize, u32)> = ranked
        .iter()
        .map(|r| (r.index, r.score.to_bits()))
        .collect();
    ensure!(finite(&bits), "non-finite rerank score");
    Ok(Output::Ranked(bits))
}

fn timed(mut call: impl FnMut() -> Result<Output>, id: &str, rows: usize) -> Result<Case> {
    let expected = call()?;
    for _ in 0..2 {
        black_box(call()?);
    }
    let mut ms = Vec::with_capacity(5);
    let mut iterations = Vec::with_capacity(5);
    for _ in 0..5 {
        let mut elapsed = std::time::Duration::ZERO;
        let mut count = 0_usize;
        while elapsed.as_millis() < 200 {
            let start = Instant::now();
            let output = black_box(call()?);
            elapsed += start.elapsed();
            count += 1;
            ensure!(output == expected, "non-deterministic output for {id}");
        }
        ms.push(elapsed.as_secs_f64() * 1000.0 / count as f64);
        iterations.push(count);
    }
    eprintln!("{id}: {ms:?} ms");
    Ok(Case {
        id: id.to_owned(),
        rows,
        ms,
        iterations,
        output: expected,
    })
}

fn texts(inputs: &[Input], short: bool) -> Vec<(String, String)> {
    inputs
        .iter()
        .filter(|i| !(short && i.name == "t_long"))
        .map(|i| (i.name.clone(), format!("{}{}", i.prompt, i.text)))
        .collect()
}

fn run_sparse_suite(args: &[String], inputs: &[Input], short: bool) -> Result<()> {
    let start = Instant::now();
    let engine = SparseEngine::load(&args[1])?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut cases = Vec::new();
    let mut batch = Vec::new();
    for (name, text) in texts(inputs, short) {
        cases.push(timed(
            || run_sparse(&engine, std::slice::from_ref(&text)),
            &name,
            1,
        )?);
        batch.push(text);
    }
    cases.push(timed(
        || run_sparse(&engine, &batch),
        "ragged_batch",
        batch.len(),
    )?);
    batch.reverse();
    cases.push(timed(
        || run_sparse(&engine, &batch),
        "reversed_batch",
        batch.len(),
    )?);
    write(&args[3], "sparse", &args[1], load_ms, cases)
}

fn run_m3_suite(args: &[String], inputs: &[Input], short: bool) -> Result<()> {
    let start = Instant::now();
    let engine = M3Engine::load(&args[1])?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut cases = Vec::new();
    let mut batch = Vec::new();
    for (name, text) in texts(inputs, short) {
        cases.push(timed(
            || run_m3(&engine, std::slice::from_ref(&text)),
            &name,
            1,
        )?);
        batch.push(text);
    }
    cases.push(timed(
        || run_m3(&engine, &batch),
        "ragged_batch",
        batch.len(),
    )?);
    batch.reverse();
    cases.push(timed(
        || run_m3(&engine, &batch),
        "reversed_batch",
        batch.len(),
    )?);
    write(&args[3], "m3", &args[1], load_ms, cases)
}

fn run_rerank_suite(args: &[String], inputs: &[Input], short: bool) -> Result<()> {
    let start = Instant::now();
    let engine = Reranker::load(&args[1])?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut cases = Vec::new();
    let mut batch = Vec::new();
    for (name, text) in texts(inputs, short) {
        let query = inputs
            .iter()
            .find(|i| i.name == name)
            .map_or_else(String::new, |i| i.prompt.clone());
        cases.push(timed(
            || run_rerank(&engine, &query, std::slice::from_ref(&text)),
            &name,
            1,
        )?);
        batch.push(text);
    }
    let query = inputs
        .first()
        .map_or_else(String::new, |i| i.prompt.clone());
    cases.push(timed(
        || run_rerank(&engine, &query, &batch),
        "ragged_batch",
        batch.len(),
    )?);
    batch.reverse();
    cases.push(timed(
        || run_rerank(&engine, &query, &batch),
        "reversed_batch",
        batch.len(),
    )?);
    write(&args[3], "rerank", &args[1], load_ms, cases)
}

fn write(path: &str, kind: &str, model: &str, load_ms: f64, cases: Vec<Case>) -> Result<()> {
    let report = Report {
        kind: kind.to_owned(),
        model: model.to_owned(),
        load_ms,
        warmups: 3,
        trials: 5,
        cases,
    };
    std::fs::write(Path::new(path), serde_json::to_vec(&report)?)?;
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        (4..=5).contains(&args.len()),
        "usage: sparse|m3|rerank MODEL CASES_JSON REPORT_JSON [all|short]"
    );
    let short = match args.get(4).map_or("all", String::as_str) {
        "all" => false,
        "short" => true,
        other => anyhow::bail!("suite must be all or short, got {other}"),
    };
    let inputs: Vec<Input> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    ensure!(
        !inputs.is_empty() && inputs.len() <= 32,
        "invalid case count"
    );
    ensure!(
        inputs
            .iter()
            .all(|i| i.prompt.len() + i.text.len() <= 32_768),
        "fixture text exceeds byte bound"
    );
    match args[0].as_str() {
        "sparse" => run_sparse_suite(&args, &inputs, short),
        "m3" => run_m3_suite(&args, &inputs, short),
        "rerank" => run_rerank_suite(&args, &inputs, short),
        other => anyhow::bail!("kind must be sparse, m3 or rerank, got {other}"),
    }
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
