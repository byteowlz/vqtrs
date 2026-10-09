//! Bounded local, sequential text-batch benchmark for Apple runtime comparisons.
//! Arguments: snapshot cpu|metal fixture.json output.json. Each fixture case has
//! `id`, prepared `texts` and matching tokenizer `ids`. At most 32 rows per case,
//! 512 tokens per row; three warmups, five trials; no downloads or fallback.
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
            args.len() == 4,
            "usage: snapshot cpu|metal fixture.json output.json"
        );
        let fixture: Fixture = serde_json::from_slice(&std::fs::read(&args[2])?)?;
        ensure!(
            !fixture.cases.is_empty() && fixture.cases.len() <= 16,
            "invalid case count"
        );
        for case in &fixture.cases {
            validate(case)?;
        }
        let start = Instant::now();
        let device = match args[1].as_str() {
            "cpu" => Device::Cpu,
            "metal" => Device::new_metal(0)?,
            other => bail!("unsupported device: {other}"),
        };
        let model = EmbeddingGemma2::from_dir_on(std::path::Path::new(&args[0]), &device)?;
        let load_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut results = Vec::new();
        for case in fixture.cases {
            for (text, ids) in case.texts.iter().zip(&case.ids) {
                ensure!(model.tokenize(text)? == *ids, "token mismatch: {}", case.id);
            }
            for _ in 0..3 {
                black_box(encode(&model, &case.texts)?);
            }
            let mut times = Vec::new();
            let mut vectors = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                vectors = encode(&model, &case.texts)?;
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            eprintln!("{}: {:?} ms", case.id, times);
            results.push(json!({"id":case.id,"ms":times,"vectors":vectors}));
        }
        let result = json!({"runtime":format!("candle-{}-f32", args[1]), "load_ms":load_ms,
            "warmups":3,"trials":5,"dimensions":768,"inference_cache":false,
            "timing":"prepared text -> tokenization + sequential inference + synchronized host copy",
            "cases":results});
        std::fs::write(&args[3], serde_json::to_vec_pretty(&result)?)?;
        Ok(())
    }

    fn validate(case: &Case) -> Result<()> {
        ensure!(
            !case.texts.is_empty() && case.texts.len() <= 32,
            "invalid row count"
        );
        ensure!(
            case.texts.len() == case.ids.len(),
            "row/token count mismatch"
        );
        ensure!(
            case.texts.iter().all(|s| s.len() <= 16_384),
            "text exceeds byte bound"
        );
        ensure!(
            case.ids.iter().all(|v| !v.is_empty() && v.len() <= 512),
            "token bound"
        );
        Ok(())
    }

    fn encode(model: &EmbeddingGemma2, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts
            .iter()
            .map(|text| model.embed_text(text).map_err(Into::into))
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn refuses_oversized_or_mismatched_inputs() {
            let mut case = Case {
                id: "test".into(),
                texts: vec!["text".into()],
                ids: vec![vec![2, 1]],
            };
            assert!(validate(&case).is_ok());
            case.ids[0] = vec![2; 513];
            assert!(validate(&case).is_err());
            case.ids.clear();
            assert!(validate(&case).is_err());
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
