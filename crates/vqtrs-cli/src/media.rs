//! Image/WAV/video embeddings using a daemon or the in-process Gemma backend.

use std::io::Read as _;
use std::path::{Path, PathBuf};

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
    Video,
}

#[derive(Debug, Args)]
pub struct MediaArgs {
    /// Images, 16 kHz WAV clips, or MP4/MKV videos (one embedding per file).
    /// Video requires ffmpeg/ffprobe; see the bounded video decoder contract.
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
    Video { data: String },
}

#[derive(Serialize)]
struct Body<'a> {
    model: &'a str,
    input: Vec<Encoded>,
}

fn read_media_file(path: &Path, modality: Modality) -> Result<Vec<u8>> {
    if !matches!(modality, Modality::Video) {
        return std::fs::read(path).with_context(|| format!("reading {}", path.display()));
    }
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut bytes = Vec::new();
    // Bound the read itself, including growing/nonregular caller-local files.
    file.take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading {}", path.display()))?;
    anyhow::ensure!(
        bytes.len() <= 16 * 1024 * 1024,
        "{}: video exceeds 16 MiB",
        path.display()
    );
    Ok(bytes)
}

pub fn run(args: &MediaArgs) -> Result<()> {
    let files: Vec<Vec<u8>> = args
        .paths
        .iter()
        .map(|p| read_media_file(p, args.modality))
        .collect::<Result<_>>()?;
    if !args.no_daemon {
        let input = files
            .iter()
            .map(|bytes| {
                let data = STANDARD.encode(bytes);
                match args.modality {
                    Modality::Image => Encoded::Image { data },
                    Modality::Audio => Encoded::Audio { data },
                    Modality::Video => Encoded::Video { data },
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
            Modality::Video => Gemma2Input::Video(bytes),
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

    #[test]
    fn oversized_video_read_is_rejected_before_model_loading() {
        let path =
            std::env::temp_dir().join(format!("vqtrs-cli-video-limit-{}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.set_len(16 * 1024 * 1024 + 1).unwrap();
        let result = read_media_file(&path, Modality::Video);
        drop(file);
        std::fs::remove_file(&path).unwrap();
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("video exceeds 16 MiB")
        );
    }

    #[test]
    fn video_modality_and_daemon_encoding() {
        let cli = super::super::Cli::try_parse_from([
            "vqtrs",
            "embed-media",
            "--modality",
            "video",
            "clip.mp4",
            "--no-daemon",
        ])
        .unwrap();
        let super::super::Command::EmbedMedia(args) = cli.command else {
            panic!("wrong command");
        };
        assert!(matches!(args.modality, Modality::Video));
        let json = serde_json::to_value(Encoded::Video {
            data: STANDARD.encode([1, 2, 3]),
        })
        .unwrap();
        assert_eq!(json, serde_json::json!({"modality":"video", "data":"AQID"}));
    }
}
