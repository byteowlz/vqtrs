//! Image/WAV embeddings using a warm daemon or the in-process Gemma backend.

use std::path::PathBuf;

use anyhow::{Context, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use clap::{Args, ValueEnum};
use serde::Serialize;
use vqtrs_core::{EMBEDDING_GEMMA2_REPO, Engine, Gemma2Input};

use super::{DaemonEmbed, EmbedItem, EmbedOutput, print_json, try_daemon};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Modality {
    Image,
    Audio,
}

#[derive(Debug, Args)]
pub struct MediaArgs {
    /// Image files or 16 kHz mono/stereo WAV clips (one embedding per file)
    #[arg(required = true)]
    paths: Vec<PathBuf>,
    /// The modality shared by these files
    #[arg(long, value_enum)]
    modality: Modality,
    /// Model repository id
    #[arg(short, long, env = "VQTRS_MODEL", default_value = EMBEDDING_GEMMA2_REPO)]
    model: String,
    /// Always load in-process; never use a running daemon
    #[arg(long)]
    no_daemon: bool,
    /// Pretty-print JSON output
    #[arg(long)]
    pretty: bool,
}

#[derive(Serialize)]
#[serde(tag = "modality", rename_all = "snake_case")]
enum Encoded {
    Image { data: String },
    Audio { data: String },
}

#[derive(Serialize)]
struct Body<'a> {
    model: &'a str,
    input: Vec<Encoded>,
}

pub fn run(args: &MediaArgs) -> Result<()> {
    let files: Vec<Vec<u8>> = args
        .paths
        .iter()
        .map(|p| std::fs::read(p).with_context(|| format!("reading {}", p.display())))
        .collect::<Result<_>>()?;
    if !args.no_daemon {
        let input = files
            .iter()
            .map(|bytes| {
                let data = STANDARD.encode(bytes);
                match args.modality {
                    Modality::Image => Encoded::Image { data },
                    Modality::Audio => Encoded::Audio { data },
                }
            })
            .collect();
        let body = Body {
            model: &args.model,
            input,
        };
        if let Some(response) = try_daemon("/embeddings/multimodal", &body, true) {
            let parsed: DaemonEmbed =
                serde_json::from_str(&response).context("parsing daemon response")?;
            let output = EmbedOutput {
                model: parsed.model,
                dimensions: parsed.data.first().map_or(0, |v| v.embedding.len()),
                data: parsed.data,
            };
            return print_json(&output, args.pretty);
        }
    }
    let engine = Engine::load(&args.model).context("loading multimodal embedding model")?;
    let inputs: Vec<_> = files
        .iter()
        .map(|bytes| match args.modality {
            Modality::Image => Gemma2Input::Image(bytes),
            Modality::Audio => Gemma2Input::Audio(bytes),
        })
        .collect();
    let vectors = engine
        .embed_multimodal(&inputs)
        .context("embedding media")?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn cli_requires_modality_and_paths() {
        assert!(
            super::super::Cli::try_parse_from(["vqtrs", "embed-media", "--modality", "audio"])
                .is_err()
        );
        assert!(super::super::Cli::try_parse_from(["vqtrs", "embed-media", "a.wav"]).is_err());
        let cli = super::super::Cli::try_parse_from([
            "vqtrs",
            "embed-media",
            "--modality",
            "audio",
            "a.wav",
            "b.wav",
            "--no-daemon",
        ])
        .unwrap();
        let super::super::Command::EmbedMedia(args) = cli.command else {
            panic!("wrong command");
        };
        assert_eq!(args.paths.len(), 2);
        assert_eq!(args.model, EMBEDDING_GEMMA2_REPO);
        assert!(args.no_daemon);
    }
}
