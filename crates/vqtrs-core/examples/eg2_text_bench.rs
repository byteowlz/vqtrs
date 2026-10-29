//! Bounded local, sequential text-batch benchmark for explicit Candle devices.
//! Arguments: snapshot cpu|metal|cuda fixture.json output.json [max-tokens].
//! Each case has `id`, prepared `texts` and matching tokenizer `ids`. At most 32
//! rows per case; default 512 tokens per row, explicit cap up to 4096. Three
//! warmups, five trials; no downloads, implicit truncation or device fallback.
//! This is an experimental benchmark, not a native CoreML or MLX backend.

#[cfg(feature = "embeddinggemma2")]
mod enabled {
    use std::{hint::black_box, time::Instant};

    use anyhow::{Result, bail, ensure};
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

    /// Run a fixed, bounded local benchmark with exact token identity checks.
    ///
    /// # Errors
    /// Returns an error for invalid fixtures, unavailable hardware or inference.
    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        ensure!(
            (4..=5).contains(&args.len()),
            "usage: snapshot cpu|metal|cuda fixture.json output.json [max-tokens:1..=4096]"
        );
        let max_tokens = args
            .get(4)
            .map_or(Ok(512), |value| value.parse::<usize>())?;
        ensure!(
            (1..=4096).contains(&max_tokens),
            "benchmark token cap must be 1..=4096"
        );
        let fixture: Fixture = serde_json::from_slice(&std::fs::read(&args[2])?)?;
        ensure!(
            !fixture.cases.is_empty() && fixture.cases.len() <= 16,
            "invalid case count"
        );
        for case in &fixture.cases {
            validate(case, max_tokens)?;
        }
        let start = Instant::now();
        let device = match args[1].as_str() {
            "cpu" => Device::Cpu,
            "metal" => Device::new_metal(0)?,
            "cuda" => Device::new_cuda(0)?,
            other => bail!("unsupported device: {other}"),
        };
        let model = EmbeddingGemma2::from_dir_on(std::path::Path::new(&args[0]), &device)?;
        let load_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut results = Vec::new();
        for case in fixture.cases {
            results.push(benchmark_case(&model, &case)?);
        }
        let result = json!({"runtime":format!("candle-{}-f32", args[1]), "load_ms":load_ms,
            "warmups":3,"trials":5,"dimensions":768,"inference_cache":false,"max_tokens":max_tokens,
            "timing":"prepared text -> tokenization + sequential inference + synchronized host copy",
            "cases":results});
        std::fs::write(&args[3], serde_json::to_vec_pretty(&result)?)?;
        Ok(())
    }

    fn benchmark_case(model: &EmbeddingGemma2, case: &Case) -> Result<serde_json::Value> {
        for (text, ids) in case.texts.iter().zip(&case.ids) {
            ensure!(model.tokenize(text)? == *ids, "token mismatch: {}", case.id);
        }
        let mut vectors = Vec::new();
        for _ in 0..3 {
            vectors = black_box(encode(model, &case.texts)?);
        }
        let mut times = Vec::new();
        let mut iterations = Vec::new();
        for _ in 0..5 {
            let mut elapsed = std::time::Duration::ZERO;
            let mut count = 0_usize;
            while elapsed.as_millis() < 200 {
                let start = Instant::now();
                let actual = black_box(encode(model, &case.texts)?);
                elapsed += start.elapsed();
                count += 1;
                ensure!(
                    same_vectors(&vectors, &actual),
                    "non-finite or non-deterministic output: {}",
                    case.id
                );
                vectors = actual;
            }
            times.push(elapsed.as_secs_f64() * 1000.0 / count as f64);
            iterations.push(count);
        }
        eprintln!("{}: {:?} ms", case.id, times);
        Ok(json!({"id":case.id,"ms":times,"iterations":iterations,"vectors":vectors}))
    }

    fn same_vectors(expected: &[Vec<f32>], actual: &[Vec<f32>]) -> bool {
        !expected.is_empty()
            && expected.len() == actual.len()
            && expected.iter().zip(actual).all(|(a, b)| {
                a.len() == 768
                    && b.len() == 768
                    && a.iter()
                        .zip(b)
                        .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits())
            })
    }

    fn validate(case: &Case, max_tokens: usize) -> Result<()> {
        let max_bytes = if max_tokens <= 512 { 16_384 } else { 131_072 };
        ensure!(
            !case.texts.is_empty() && case.texts.len() <= 32,
            "invalid row count"
        );
        ensure!(
            case.texts.len() == case.ids.len(),
            "row/token count mismatch"
        );
        ensure!(
            case.texts.iter().all(|s| s.len() <= max_bytes),
            "text exceeds byte bound"
        );
        ensure!(
            case.ids
                .iter()
                .all(|v| !v.is_empty() && v.len() <= max_tokens),
            "token bound"
        );
        Ok(())
    }

    fn encode(model: &EmbeddingGemma2, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        model.embed_text_batch(texts).map_err(Into::into)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn exact_vector_gate_rejects_bits_shape_and_non_finite_values() {
            let expected = vec![vec![0.0_f32; 768]];
            assert!(same_vectors(&expected, &expected));
            assert!(!same_vectors(&expected, &[]));
            assert!(!same_vectors(&expected, &[vec![0.0; 767]]));
            let mut changed = expected.clone();
            changed[0][0] = -0.0;
            assert!(!same_vectors(&expected, &changed));
            changed[0][0] = f32::NAN;
            assert!(!same_vectors(&changed, &changed));
        }

        #[test]
        fn refuses_oversized_or_mismatched_inputs() {
            let mut case = Case {
                id: "test".into(),
                texts: vec!["text".into()],
                ids: vec![vec![2, 1]],
            };
            assert!(validate(&case, 512).is_ok());
            case.ids[0] = vec![2; 513];
            assert!(validate(&case, 512).is_err());
            assert!(validate(&case, 4096).is_ok());
            case.ids[0] = vec![2; 4097];
            assert!(validate(&case, 4096).is_err());
            case.ids.clear();
            assert!(validate(&case, 4096).is_err());
        }
    }
}

fn main() -> std::process::ExitCode {
    #[cfg(feature = "embeddinggemma2")]
    match enabled::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            std::process::ExitCode::FAILURE
        }
    }
    #[cfg(not(feature = "embeddinggemma2"))]
    {
        eprintln!("requires embeddinggemma2 (CPU) or embeddinggemma2-metal");
        std::process::ExitCode::FAILURE
    }
}
