//! Bounded dense-engine differential benchmark using existing parity text cases.
//! Usage: `engine_bench MODEL CASES_JSON REPORT_JSON [all|short]`.
//! Uses warm models, includes token accounting, and records exact output bits.

use std::{hint::black_box, path::Path, time::Instant};

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use vqtrs_core::Engine;

#[derive(Deserialize)]
struct Input {
    name: String,
    prompt: String,
    text: String,
}

#[derive(Serialize)]
struct Output {
    tokens: Option<usize>,
    bits: Vec<Vec<u32>>,
    #[serde(skip)]
    token_ms: f64,
}

#[derive(Serialize)]
struct Case {
    id: String,
    rows: usize,
    ms: Vec<f64>,
    token_ms: Vec<f64>,
    iterations: Vec<usize>,
    output: Output,
}

#[derive(Serialize)]
struct Report {
    model: String,
    backend: String,
    load_ms: f64,
    warmups: usize,
    trials: usize,
    cases: Vec<Case>,
}

fn infer(engine: &Engine, texts: &[String]) -> Result<Output> {
    let start = Instant::now();
    let tokens = engine.count_tokens(texts);
    let token_ms = start.elapsed().as_secs_f64() * 1000.0;
    let vectors = engine.embed_batch(texts)?;
    ensure!(vectors.len() == texts.len(), "output row count changed");
    let mut bits = Vec::with_capacity(vectors.len());
    for row in vectors {
        ensure!(row.len() == engine.dimensions(), "wrong dimensions");
        ensure!(row.iter().all(|v| v.is_finite()), "non-finite output");
        bits.push(row.into_iter().map(f32::to_bits).collect());
    }
    Ok(Output {
        tokens,
        bits,
        token_ms,
    })
}

fn benchmark(engine: &Engine, id: String, texts: &[String]) -> Result<Case> {
    let expected = infer(engine, texts)?;
    for _ in 0..2 {
        black_box(infer(engine, texts)?);
    }
    let mut ms = Vec::with_capacity(5);
    let mut token_ms = Vec::with_capacity(5);
    let mut iterations = Vec::with_capacity(5);
    for _ in 0..5 {
        let mut elapsed = std::time::Duration::ZERO;
        let mut token_elapsed = 0.0;
        let mut count = 0;
        while elapsed.as_millis() < 200 {
            let start = Instant::now();
            let output = black_box(infer(engine, texts)?);
            elapsed += start.elapsed();
            token_elapsed += output.token_ms;
            count += 1;
            ensure!(
                output.tokens == expected.tokens && output.bits == expected.bits,
                "non-deterministic output for {id}"
            );
        }
        ms.push(elapsed.as_secs_f64() * 1000.0 / count as f64);
        token_ms.push(token_elapsed / count as f64);
        iterations.push(count);
    }
    eprintln!("{id}: {ms:?} ms");
    Ok(Case {
        id,
        rows: texts.len(),
        ms,
        token_ms,
        iterations,
        output: expected,
    })
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        (3..=4).contains(&args.len()),
        "usage: MODEL CASES_JSON REPORT_JSON [all|short]"
    );
    let short = match args.get(3).map_or("all", String::as_str) {
        "all" => false,
        "short" => true,
        _ => anyhow::bail!("suite must be all or short"),
    };
    let inputs: Vec<Input> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
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
    let start = Instant::now();
    let engine = Engine::load(&args[0])?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    ensure!(engine.embed_batch(&[])?.is_empty(), "empty batch changed");
    let mut cases = Vec::new();
    let mut batch = Vec::new();
    for input in inputs {
        if short && input.name == "t_long" {
            continue;
        }
        let text = format!("{}{}", input.prompt, input.text);
        cases.push(benchmark(&engine, input.name, std::slice::from_ref(&text))?);
        batch.push(text);
    }
    cases.push(benchmark(&engine, "ragged_batch".into(), &batch)?);
    batch.reverse();
    // Order restoration and neighbor isolation are recorded as a separate case.
    cases.push(benchmark(&engine, "reversed_batch".into(), &batch)?);
    let report = Report {
        model: engine.model().into(),
        backend: format!("{:?}", engine.backend()),
        load_ms,
        warmups: 3,
        trials: 5,
        cases,
    };
    std::fs::write(Path::new(&args[2]), serde_json::to_vec(&report)?)?;
    Ok(())
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
