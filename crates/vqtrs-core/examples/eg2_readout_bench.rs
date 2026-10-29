//! Paired same-process CUDA readout timings against unchanged sequential inference.
//! Usage: `eg2_readout_bench snapshot fixture.json output-prefix`.
//! Existing fixtures only; explicit CUDA, exact output bits, no inference cache.

use std::{hint::black_box, time::Instant};

use anyhow::{Result, ensure};
use candle_core::Device;
use serde::Deserialize;
use serde_json::json;
use vqtrs_core::EmbeddingGemma2;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    texts: Vec<String>,
    ids: Vec<Vec<u32>>,
}

fn encode(model: &EmbeddingGemma2, texts: &[String], batch: bool) -> Result<Vec<Vec<f32>>> {
    if batch {
        model.embed_text_batch(texts).map_err(Into::into)
    } else {
        texts
            .iter()
            .map(|text| model.embed_text(text).map_err(Into::into))
            .collect()
    }
}

fn check(expected: &[Vec<f32>], actual: &[Vec<f32>]) -> Result<()> {
    ensure!(expected.len() == actual.len(), "row count changed");
    for row in expected.iter().chain(actual) {
        ensure!(row.len() == 768, "dimension changed");
        ensure!(row.iter().all(|v| v.is_finite()), "non-finite output");
    }
    ensure!(
        expected
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .eq(actual.iter().flatten().map(|v| v.to_bits())),
        "F32 bit gate failed"
    );
    Ok(())
}

fn measure(model: &EmbeddingGemma2, case: &Case) -> Result<[serde_json::Value; 2]> {
    let expected = encode(model, &case.texts, false)?;
    for _ in 0..3 {
        check(&expected, &encode(model, &case.texts, false)?)?;
        check(&expected, &encode(model, &case.texts, true)?)?;
    }
    let mut times = [Vec::new(), Vec::new()];
    let mut iterations = [Vec::new(), Vec::new()];
    for trial in 0..6 {
        for index in [trial % 2, 1 - trial % 2] {
            let mut elapsed = std::time::Duration::ZERO;
            let mut count = 0_usize;
            while elapsed.as_millis() < 200 {
                let start = Instant::now();
                let actual = black_box(encode(model, &case.texts, index == 1)?);
                elapsed += start.elapsed();
                count += 1;
                check(&expected, &actual)?;
            }
            times[index].push(elapsed.as_secs_f64() * 1000.0 / count as f64);
            iterations[index].push(count);
        }
    }
    eprintln!(
        "{}: sequential {:?}; deferred {:?}",
        case.id, times[0], times[1]
    );
    Ok(std::array::from_fn(
        |i| json!({"id":case.id,"ms":times[i],"iterations":iterations[i],"vectors":expected}),
    ))
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "usage: snapshot fixture.json output-prefix"
    );
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    ensure!(
        !fixture.cases.is_empty() && fixture.cases.len() <= 16,
        "case bound"
    );
    let model =
        EmbeddingGemma2::from_dir_on(std::path::Path::new(&args[0]), &Device::new_cuda(0)?)?;
    let mut results = [Vec::new(), Vec::new()];
    for case in &fixture.cases {
        ensure!(
            !case.texts.is_empty() && case.texts.len() <= 32 && case.texts.len() == case.ids.len(),
            "row bound"
        );
        for (text, ids) in case.texts.iter().zip(&case.ids) {
            ensure!(
                text.len() <= 16_384 && !ids.is_empty() && ids.len() <= 512,
                "input bound"
            );
            ensure!(model.tokenize(text)? == *ids, "token identity changed");
        }
        for (out, row) in results.iter_mut().zip(measure(&model, case)?) {
            out.push(row);
        }
    }
    for (i, name) in ["baseline", "candidate"].iter().enumerate() {
        let report = json!({"runtime":"candle-cuda-f32","dimensions":768,"inference_cache":false,"warmups_per_arm":3,"paired_trials":6,"cases":results[i]});
        std::fs::write(
            format!("{}-{name}.json", args[2]),
            serde_json::to_vec(&report)?,
        )?;
    }
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
