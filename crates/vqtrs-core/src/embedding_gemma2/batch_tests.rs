//! Exact same-device CUDA batch readout regression; local model assets only.

use anyhow::{Result, ensure};
use candle_core::Device;
use serde::Deserialize;

use super::{EMBEDDING_GEMMA2_MAX_TOKENS, EmbeddingGemma2};

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    texts: Vec<String>,
}

fn exact(expected: &[Vec<f32>], actual: &[Vec<f32>]) -> bool {
    let words = |rows: &[Vec<f32>]| {
        rows.iter()
            .map(|row| row.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    words(expected) == words(actual)
}

#[test]
#[ignore = "requires local EG2_MODEL_DIR and CUDA hardware; no downloads or CPU fallback"]
fn cuda_batch_readout_preserves_singleton_bits_order_boundaries_and_errors() -> Result<()> {
    let root = std::env::var("EG2_MODEL_DIR")?;
    let model = EmbeddingGemma2::from_dir_on(std::path::Path::new(&root), &Device::new_cuda(0)?)?;
    ensure!(model.device().is_cuda(), "CUDA required");
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../../docs/benchmarks/apple-text-fixtures.json"
    ))?;
    let texts: Vec<_> = fixture
        .cases
        .into_iter()
        .flat_map(|case| case.texts)
        .collect();
    ensure!(texts.len() >= 32, "fixture coverage changed");
    let reference = texts
        .iter()
        .map(|text| model.embed_text(text))
        .collect::<super::Result<Vec<_>>>()?;
    for count in [0, 1, 2, 7, 8, 9, 16, 17, 32] {
        let mut input = texts[..count].to_vec();
        let mut expected = reference[..count].to_vec();
        ensure!(
            exact(&expected, &model.embed_text_batch(&input)?),
            "batch {count}"
        );
        input.reverse();
        expected.reverse();
        ensure!(
            exact(&expected, &model.embed_text_batch(&input)?),
            "reversed batch {count}"
        );
    }
    let duplicate = vec![texts[0].clone(); 9];
    let expected = vec![reference[0].clone(); 9];
    ensure!(
        exact(&expected, &model.embed_text_batch(&duplicate)?),
        "duplicate rows"
    );
    check_limits(&model, &texts, &reference)?;
    Ok(())
}

fn check_limits(model: &EmbeddingGemma2, texts: &[String], reference: &[Vec<f32>]) -> Result<()> {
    let long = texts[0].repeat(64);
    let count = model.tokenize(&long)?.len();
    ensure!(
        (513..=EMBEDDING_GEMMA2_MAX_TOKENS).contains(&count),
        "long fixture"
    );
    let long_batch = vec![long.clone(), texts[0].clone(), long];
    let long_reference = long_batch
        .iter()
        .map(|text| model.embed_text(text))
        .collect::<super::Result<Vec<_>>>()?;
    ensure!(
        exact(&long_reference, &model.embed_text_batch(&long_batch)?),
        "long ragged batch"
    );
    let oversized = texts[0].repeat(1024);
    ensure!(
        model.tokenize(&oversized)?.len() > EMBEDDING_GEMMA2_MAX_TOKENS,
        "oversize fixture"
    );
    ensure!(
        model
            .embed_text_batch(&[texts[0].clone(), oversized, texts[1].clone()])
            .is_err(),
        "oversize accepted"
    );
    ensure!(
        exact(&reference[..2], &model.embed_text_batch(&texts[..2])?),
        "error recovery"
    );
    Ok(())
}
