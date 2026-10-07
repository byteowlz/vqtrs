//! Multimodal embeddings over the shared bounded dense-model registry.
//! Inline base64 only: the server never fetches a URL or opens a client path.

use axum::{Json, extract::State, http::StatusCode};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use vqtrs_core::{Backend, EMBEDDING_GEMMA2_REPO, Engine, Gemma2Input};

use super::{ApiError, AppState, EmbedData, InFlight, check_batch_len, join_error};

#[derive(Deserialize)]
#[serde(tag = "modality", rename_all = "snake_case", deny_unknown_fields)]
enum Input {
    Text { text: String },
    Image { data: String },
    Audio { data: String },
    Video { data: String },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    #[serde(default)]
    model: Option<String>,
    input: Vec<Input>,
}

/// Embedding list; media token accounting is not available, so usage is omitted.
#[derive(Serialize)]
pub struct Response {
    object: &'static str,
    data: Vec<EmbedData>,
    model: String,
}

enum Decoded {
    Text(String),
    Image(Vec<u8>),
    Audio(Vec<u8>),
    Video(Vec<u8>),
}

impl Decoded {
    fn decode(input: Input) -> Result<Self, ApiError> {
        let decode = |encoded: String| {
            STANDARD
                .decode(encoded)
                .map_err(|e| bad_request(format!("invalid base64 media: {e}")))
        };
        match input {
            Input::Text { text } => Ok(Self::Text(text)),
            Input::Image { data } => Ok(Self::Image(decode(data)?)),
            Input::Audio { data } => Ok(Self::Audio(decode(data)?)),
            Input::Video { data } => Ok(Self::Video(decode(data)?)),
        }
    }

    fn input(&self) -> Gemma2Input<'_> {
        match self {
            Self::Text(text) => Gemma2Input::Text(text),
            Self::Image(bytes) => Gemma2Input::Image(bytes),
            Self::Audio(bytes) => Gemma2Input::Audio(bytes),
            Self::Video(bytes) => Gemma2Input::Video(bytes),
        }
    }
}

const fn bad_request(message: String) -> ApiError {
    ApiError::new(StatusCode::BAD_REQUEST, message)
}

pub async fn embeddings(
    State(state): State<AppState>,
    InFlight(permit): InFlight,
    Json(req): Json<Request>,
) -> Result<Json<Response>, ApiError> {
    check_batch_len(
        req.input.len(),
        state.max_batch_texts,
        "multimodal input batch",
    )?;
    if req.input.is_empty() {
        return Err(bad_request("input must contain at least one item".into()));
    }
    let (model, vectors) = tokio::task::spawn_blocking(move || -> Result<_, ApiError> {
        // Keep the permit in the worker even if the HTTP client disconnects.
        let _permit = permit;
        let decoded = req
            .input
            .into_iter()
            .map(Decoded::decode)
            .collect::<Result<Vec<_>, _>>()?;
        let name = req.model.as_deref().unwrap_or(EMBEDDING_GEMMA2_REPO);
        let model_info = vqtrs_core::dense_models()
            .iter()
            .find(|m| m.code.eq_ignore_ascii_case(name));
        if model_info.is_none_or(|m| m.backend != Backend::EmbeddingGemma2) {
            return Err(bad_request(
                "multimodal input requires google/embeddinggemma-2".into(),
            ));
        }
        let engine = state
            .dense
            .get_or_load(name, || Engine::load(name).map_err(Into::into))?;
        let inputs: Vec<_> = decoded.iter().map(Decoded::input).collect();
        let vectors = engine
            .embed_multimodal(&inputs)
            .map_err(|e| bad_request(e.to_string()))?;
        Ok((engine.model().to_owned(), vectors))
    })
    .await
    .map_err(|e| join_error(&e))??;
    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbedData {
            object: "embedding",
            embedding,
            index,
        })
        .collect();
    // Token accounting is not yet defined for media; never report an invented
    // text-byte estimate as the audio/image/video token count.
    Ok(Json(Response {
        object: "list",
        data,
        model,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_inputs_decode_in_order() {
        let req: Request = serde_json::from_str(r#"{"input":[{"modality":"text","text":"hello"},{"modality":"image","data":"AQID"},{"modality":"audio","data":"BAU="},{"modality":"video","data":"Bgc="}]}"#).unwrap();
        let decoded: Vec<_> = req
            .input
            .into_iter()
            .map(Decoded::decode)
            .collect::<Result<_, _>>()
            .unwrap_or_else(|e| panic!("{}", e.message));
        assert!(matches!(decoded[0].input(), Gemma2Input::Text("hello")));
        assert!(matches!(decoded[1].input(), Gemma2Input::Image([1, 2, 3])));
        assert!(matches!(decoded[2].input(), Gemma2Input::Audio([4, 5])));
        assert!(matches!(decoded[3].input(), Gemma2Input::Video([6, 7])));
    }

    #[test]
    fn rejects_invalid_base64_urls_and_paths() {
        let result = Decoded::decode(Input::Image {
            data: "not base64!".into(),
        });
        assert!(matches!(
            result,
            Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                ..
            })
        ));
        for modality in ["image", "audio", "video"] {
            for field in ["url", "path"] {
                let json =
                    format!(r#"{{"input":[{{"modality":"{modality}","{field}":"file-or-url"}}]}}"#);
                assert!(serde_json::from_str::<Request>(&json).is_err());
                let json = format!(
                    r#"{{"input":[{{"modality":"{modality}","data":"AQ==","{field}":"file-or-url"}}]}}"#
                );
                assert!(serde_json::from_str::<Request>(&json).is_err());
            }
        }
        assert!(
            Decoded::decode(Input::Video {
                data: "data:video/mp4;base64,AQ==".into()
            })
            .is_err()
        );
        assert!(
            Decoded::decode(Input::Video {
                data: "https://example.invalid/clip.mp4".into()
            })
            .is_err()
        );
    }
}
