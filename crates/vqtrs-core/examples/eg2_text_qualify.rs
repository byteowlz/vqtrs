//! Single-pass bounded text-vector export for native-runtime retrieval qualification.
//! Usage: snapshot cpu|metal fixture.json output.json. No downloads or fallback.

#[cfg(feature = "embeddinggemma2")]
mod enabled {
    use std::path::Path;

    use anyhow::{Result, bail, ensure};
    use candle_core::Device;
    use serde::Deserialize;
    use serde_json::json;
    use vqtrs_core::EmbeddingGemma2;

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Fixture {
        rows: Vec<Row>,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Row {
        id: String,
        text: String,
        ids: Vec<u32>,
    }

    fn validate(fixture: &Fixture) -> Result<()> {
        ensure!(
            !fixture.rows.is_empty() && fixture.rows.len() <= 2048,
            "expected 1..2048 rows"
        );
        for row in &fixture.rows {
            ensure!(row.id.len() <= 128, "oversized row identifier");
            ensure!(row.text.len() <= 16384, "oversized text");
            ensure!(
                !row.ids.is_empty() && row.ids.len() <= 512,
                "expected 1..512 tokens"
            );
        }
        Ok(())
    }

    /// Export vectors for one bounded, pretokenized qualification fixture.
    ///
    /// # Errors
    /// Returns an error for invalid input, unavailable devices or inference failure.
    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        ensure!(args.len() == 4, "usage: snapshot cpu|metal fixture output");
        ensure!(
            std::fs::metadata(&args[2])?.len() <= 16 * 1024 * 1024,
            "fixture exceeds 16 MiB"
        );
        let fixture: Fixture = serde_json::from_slice(&std::fs::read(&args[2])?)?;
        validate(&fixture)?;
        let device = match args[1].as_str() {
            "cpu" => Device::Cpu,
            "metal" => Device::new_metal(0)?,
            other => bail!("unsupported device: {other}"),
        };
        let model = EmbeddingGemma2::from_dir_on(Path::new(&args[0]), &device)?;
        for row in &fixture.rows {
            ensure!(model.tokenize(&row.text)? == row.ids, "token mismatch");
        }
        let mut vectors = Vec::with_capacity(fixture.rows.len());
        for row in fixture.rows {
            let vector = model.embed_text(&row.text)?;
            ensure!(
                vector.len() == 768 && vector.iter().all(|v| v.is_finite()),
                "invalid embedding"
            );
            vectors.push(json!({"id":row.id,"vector":vector}));
        }
        std::fs::write(
            &args[3],
            serde_json::to_vec(&json!({"runtime":format!("candle-{}-f32",args[1]),
                "dimensions":768,"passes":1,"rows":vectors}))?,
        )?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::{Fixture, Row, validate};

        #[test]
        fn refuses_unbounded_rows_and_context() {
            assert!(validate(&Fixture { rows: Vec::new() }).is_err());
            assert!(
                validate(&Fixture {
                    rows: vec![Row {
                        id: "overflow".into(),
                        text: "text".into(),
                        ids: vec![2; 513],
                    }],
                })
                .is_err()
            );
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
        eprintln!("requires embeddinggemma2 or embeddinggemma2-metal");
        std::process::ExitCode::FAILURE
    }
}
