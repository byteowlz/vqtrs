//! Bounded library verification of opt-in native models against the original F32 engine.

use std::path::PathBuf;

use anyhow::{Context as _, Result, ensure};
use candle_core::Device;
use serde_json::json;
use vqtrs_core::{EmbeddingGemma2, Engine, Gemma2Input, Gemma2VideoFrame};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 4,
        "usage: eg2_native_smoke <native-model> <Google-snapshot> <synthetic-media-dir> <report>"
    );
    let media = PathBuf::from(&args[2]);
    let native = Engine::load(&args[0])?;
    ensure!(
        native.model() == args[0] && native.dimensions() == 768,
        "native identity mismatch"
    );
    #[cfg(feature = "embeddinggemma2-metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "embeddinggemma2-metal"))]
    let device = Device::Cpu;
    let reference = EmbeddingGemma2::from_dir_on(&PathBuf::from(&args[1]), &device)?;
    let mut reports = Vec::new();
    let text = "task: search result | query: What powers the Sun?";
    let image = std::fs::read(media.join("small_64x48.png"))?;
    let audio = std::fs::read(media.join("short_300ms.wav"))?;
    let chirp = std::fs::read(media.join("chirp_2370ms.wav"))?;
    let frames = [Gemma2VideoFrame {
        width: 64,
        height: 48,
        data: (0..64 * 48 * 3).map(|n| (n % 251) as u8).collect(),
    }];
    let inputs = [
        Gemma2Input::Text(text),
        Gemma2Input::Image(&image),
        Gemma2Input::Audio(&audio),
        Gemma2Input::Audio(&chirp),
        Gemma2Input::VideoFrames {
            frames: &frames,
            metadata: None,
        },
    ];
    let names = [
        "text",
        "image",
        "short_audio",
        "chirp_audio",
        "decoded_video",
    ];
    let actual = native.embed_multimodal(&inputs)?;
    ensure!(actual.len() == inputs.len(), "library order/count mismatch");
    for ((name, input), vector) in names.into_iter().zip(inputs).zip(&actual) {
        let expected = reference.embed(input)?;
        let norm = |values: &[f32]| {
            values
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let cosine = vector
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>()
            / (norm(vector) * norm(&expected));
        let max_abs = vector
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        ensure!(
            vector.len() == 768
                && vector.iter().all(|x| x.is_finite())
                && (norm(vector) - 1.0).abs() < 2e-6,
            "invalid library vector"
        );
        // Fixture regression envelope, NOT the unchanged F32-equivalence gate,
        // not a general native precision/task-quality claim.
        ensure!(
            cosine >= 0.99 && max_abs <= 0.02,
            "native fixture drift regression: {name} maxabs={max_abs} cosine={cosine}"
        );
        reports.push(json!({"case":name,"max_abs_vs_f32":max_abs,"cosine_vs_f32":cosine,"norm":norm(vector)}));
    }
    let repeated = native.embed_multimodal(&inputs)?;
    ensure!(
        actual == repeated,
        "serialized repeated library inference differs"
    );
    ensure!(
        native.count_tokens(&[text.to_owned()]).is_some(),
        "missing native token accounting"
    );
    reject_limits(&native, &media, &frames[0])?;
    let maximum_video = maximum_video(&native, &frames[0])?;
    let report = json!({"model":native.model(),"reference_device":format!("{device:?}"),"cases":reports,"repeated_exact":true,"limit_rejections":true,"additional_decoded_video_frames":maximum_video,"f32_equivalence_claimed":false});
    std::fs::write(&args[3], serde_json::to_vec_pretty(&report)?)
        .context("writing public-safe report")?;
    println!("{report}");
    Ok(())
}

fn maximum_video(native: &Engine, frame: &Gemma2VideoFrame) -> Result<Option<usize>> {
    if !native.model().contains("mlx") {
        return Ok(None);
    }
    let frames = vec![frame.clone(); 32];
    let vectors = native.embed_multimodal(&[Gemma2Input::VideoFrames {
        frames: &frames,
        metadata: None,
    }])?;
    ensure!(
        vectors.len() == 1 && vectors[0].len() == 768,
        "MLX32-frame video failed"
    );
    Ok(Some(32))
}

fn reject_limits(native: &Engine, media: &std::path::Path, frame: &Gemma2VideoFrame) -> Result<()> {
    let long = std::fs::read(media.join("long_31500ms_pcm16.wav"))?;
    ensure!(
        native
            .embed_multimodal(&[Gemma2Input::Audio(&long)])
            .is_err(),
        "long audio was silently truncated"
    );
    ensure!(
        native.embed(&"x ".repeat(9000)).is_err(),
        "long text was silently truncated"
    );
    ensure!(
        native.embed("<|image|>").is_err(),
        "untyped media placeholder accepted"
    );
    if native.model().contains("coreml") {
        let too_many = vec![frame.clone(); 4];
        ensure!(
            native
                .embed_multimodal(&[Gemma2Input::VideoFrames {
                    frames: &too_many,
                    metadata: None
                }])
                .is_err(),
            "CoreML video overflow was silently truncated"
        );
    }
    Ok(())
}
