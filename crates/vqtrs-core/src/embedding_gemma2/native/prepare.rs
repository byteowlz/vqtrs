//! Reuse the original Rust decoder/frontend; never load a second model for preprocessing.

use std::io::Write as _;

use tempfile::TempDir;

use super::{Item, Native, Runtime, error, fail, write_floats};
use crate::embedding_gemma2::{Gemma2Input, image, mel, video};
use crate::error::Result;

pub(super) struct Prepared {
    pub(super) item: Item,
    _directory: Option<TempDir>,
}

impl Native {
    pub(super) fn prepare(&self, input: Gemma2Input<'_>) -> Result<Prepared> {
        if let Gemma2Input::Text(text) = input {
            return Ok(Prepared {
                item: Item {
                    kind: "text",
                    ids: self.tokens(text)?,
                    ..Item::default()
                },
                _directory: None,
            });
        }
        let directory = tempfile::tempdir()?;
        let mut item = Item {
            directory: Some(directory.path().to_owned()),
            ..Item::default()
        };
        match input {
            Gemma2Input::Image(bytes) => {
                let rgb = image::decode_rgb(bytes).map_err(fail)?;
                let patches =
                    image::preprocess(&rgb, &image::ImageSettings::default()).map_err(fail)?;
                item.kind = "image";
                item.frames = 1;
                item.patches = patches.positions.len();
                item.soft_tokens = patches.num_soft_tokens;
                item.ids = media_ids(255_999, 258_880, 258_882, item.soft_tokens, 1);
                write_floats(&directory.path().join("pixels"), &patches.pixel_values)?;
                write_positions(&directory.path().join("positions"), &patches.positions)?;
            }
            Gemma2Input::Audio(bytes) => {
                let (wave, rate) = mel::decode_wav(bytes).map_err(fail)?;
                let max = if self.runtime == Runtime::CoreMl {
                    160_000
                } else {
                    480_000
                };
                if rate != 16_000 || wave.len() > max || wave.len() < 161 {
                    return Err(fail(format!(
                        "native audio requires16kHz WAV with161..={max} samples; split longer audio explicitly (no truncation)"
                    )));
                }
                item.kind = "audio";
                item.samples = wave.len();
                if self.runtime == Runtime::CoreMl {
                    item.soft_tokens = ((160 + wave.len() - 321) / 160 + 1).div_ceil(4);
                    write_floats(&directory.path().join("wave"), &wave)?;
                } else {
                    let features = self.mel.features(&wave).map_err(error)?;
                    item.frames = features.mask.len();
                    item.soft_tokens = features.mask.iter().filter(|&&v| v).count().div_ceil(4);
                    write_floats(&directory.path().join("mel"), &features.features)?;
                    std::fs::write(
                        directory.path().join("mask"),
                        features
                            .mask
                            .iter()
                            .map(|&x| u8::from(x))
                            .collect::<Vec<_>>(),
                    )?;
                }
                item.ids = media_ids(256_000, 258_881, 258_883, item.soft_tokens, 1);
            }
            Gemma2Input::Video(bytes) => {
                self.prepare_video(
                    &video::decode_container(bytes, &video_settings()).map_err(fail)?,
                    &directory,
                    &mut item,
                )?;
            }
            Gemma2Input::VideoFrames { frames, metadata } => {
                self.prepare_video(
                    &video::preprocess_decoded(frames, metadata, &video_settings())
                        .map_err(fail)?,
                    &directory,
                    &mut item,
                )?;
            }
            Gemma2Input::Text(_) => return Err(fail("unreachable text preparation branch")),
        }
        self.check_length(item.ids.len())?;
        Ok(Prepared {
            item,
            _directory: Some(directory),
        })
    }

    fn prepare_video(
        &self,
        prepared: &video::PreparedVideo,
        directory: &TempDir,
        item: &mut Item,
    ) -> Result<()> {
        item.kind = "video";
        item.frames = prepared.frames.len();
        item.soft_tokens = prepared.num_soft_tokens;
        item.ids = media_ids(255_999, 258_884, 258_882, item.soft_tokens, item.frames);
        self.check_length(item.ids.len())?;
        item.patches = prepared
            .frames
            .first()
            .ok_or_else(|| fail("no video frames"))?
            .positions
            .len();
        let pixels: Vec<_> = prepared
            .frames
            .iter()
            .flat_map(|f| f.pixel_values.iter().copied())
            .collect();
        let positions: Vec<_> = prepared
            .frames
            .iter()
            .flat_map(|f| f.positions.iter().copied())
            .collect();
        write_floats(&directory.path().join("pixels"), &pixels)?;
        write_positions(&directory.path().join("positions"), &positions)?;
        Ok(())
    }
}

fn video_settings() -> image::ImageSettings {
    image::ImageSettings {
        max_soft_tokens: 140,
        ..image::ImageSettings::default()
    }
}

fn media_ids(start: u32, marker: u32, end: u32, soft: usize, frames: usize) -> Vec<u32> {
    let mut ids = vec![2];
    for _ in 0..frames {
        ids.push(start);
        ids.extend(std::iter::repeat_n(marker, soft));
        ids.push(end);
    }
    ids.push(1);
    ids
}

fn write_positions(path: &std::path::Path, positions: &[[i64; 2]]) -> Result<()> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    for position in positions {
        for value in position {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_sequences_preserve_bos_eos_frame_and_placeholder_order() {
        assert_eq!(
            media_ids(255_999, 258_884, 258_882, 2, 2),
            [
                2, 255_999, 258_884, 258_884, 258_882, 255_999, 258_884, 258_884, 258_882, 1
            ]
        );
        assert_eq!(
            media_ids(256_000, 258_881, 258_883, 1, 1),
            [2, 256_000, 258_881, 258_883, 1]
        );
        assert_eq!(media_ids(255_999, 258_880, 258_882, 140, 3).len(), 428);
        assert_eq!(media_ids(255_999, 258_880, 258_882, 140, 4).len(), 570);
    }
}
