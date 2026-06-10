//! Model catalog and name resolution.
//!
//! Two backends are exposed behind one catalog:
//! - [`Backend::Onnx`] — fastembed/ONNX models selected via the
//!   [`fastembed::EmbeddingModel`] enum.
//! - [`Backend::Qwen3`] — Qwen3-Embedding models run through the candle
//!   backend, selected by their Hugging Face repository id.
//!
//! Names resolve either by Hugging Face `code` (e.g.
//! `"intfloat/multilingual-e5-small"`) or by fastembed `variant`
//! (e.g. `"MultilingualE5Small"`).

use fastembed::{Bgem3Model, EmbeddingModel, RerankerModel, SparseModel};

use crate::error::{Result, VqtrsError};

/// Inference backend that serves a given model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// fastembed running ONNX models through `ort`.
    Onnx,
    /// Qwen3-Embedding running through the candle backend.
    Qwen3,
}

/// A single text-embedding model in the catalog.
#[derive(Debug, Clone, Copy)]
pub struct ModelInfo {
    /// Hugging Face style code (also the repo id for Qwen3 models).
    pub code: &'static str,
    /// fastembed variant name (empty for Qwen3 models).
    pub variant: &'static str,
    /// Output vector dimensionality.
    pub dimensions: usize,
    /// Short human-readable description.
    pub description: &'static str,
    /// Backend that serves this model.
    pub backend: Backend,
}

/// A single reranker model in the catalog.
#[derive(Debug, Clone, Copy)]
pub struct RerankInfo {
    /// Hugging Face style code.
    pub code: &'static str,
    /// fastembed variant name.
    pub variant: &'static str,
    /// Short human-readable description.
    pub description: &'static str,
}

/// Default maximum sequence length used for Qwen3 embedding models.
pub const QWEN3_MAX_LENGTH: usize = 8192;

/// Resolved model ready to be loaded by the engine.
pub enum Resolved {
    /// An ONNX model variant plus its catalog dimensionality, if known.
    Onnx {
        model: EmbeddingModel,
        dimensions: Option<usize>,
    },
    /// A Qwen3 model identified by repository id, with known dimensionality.
    #[cfg(feature = "qwen3")]
    Qwen3 { repo: String, dimensions: usize },
}

/// The full text-embedding catalog (ONNX models plus Qwen3 candle models).
#[must_use]
pub const fn dense_models() -> &'static [ModelInfo] {
    DENSE_MODELS
}

/// The full reranker catalog.
#[must_use]
pub const fn rerank_models() -> &'static [RerankInfo] {
    RERANK_MODELS
}

/// The Hugging Face repository id for a catalog model named `name` (by code or
/// variant), used to locate its cached `tokenizer.json`.
#[must_use]
pub fn hf_repo(name: &str) -> Option<&'static str> {
    DENSE_MODELS
        .iter()
        .find(|m| m.code == name || (!m.variant.is_empty() && m.variant == name))
        .map(|m| m.code)
}

const DENSE_MODELS: &[ModelInfo] = &[
    // --- MiniLM / mpnet ---
    ModelInfo {
        code: "Qdrant/all-MiniLM-L6-v2-onnx",
        variant: "AllMiniLML6V2",
        dimensions: 384,
        description: "Fast, lightweight model (default)",
        backend: Backend::Onnx,
    },
    ModelInfo {
        code: "Xenova/all-MiniLM-L6-v2",
        variant: "AllMiniLML6V2Q",
        dimensions: 384,
        description: "Quantized MiniLM-L6-v2",
        backend: Backend::Onnx,
    },
    ModelInfo {
        code: "Xenova/all-MiniLM-L12-v2",
        variant: "AllMiniLML12V2",
        dimensions: 384,
        description: "Larger MiniLM variant",
        backend: Backend::Onnx,
    },
    ModelInfo {
        code: "Xenova/all-mpnet-base-v2",
        variant: "AllMpnetBaseV2",
        dimensions: 768,
        description: "Sentence Transformer mpnet-base-v2",
        backend: Backend::Onnx,
    },
    // --- BGE (English) ---
    ModelInfo {
        code: "Xenova/bge-small-en-v1.5",
        variant: "BGESmallENV15",
        dimensions: 384,
        description: "BGE small English v1.5",
        backend: Backend::Onnx,
    },
    ModelInfo {
        code: "Xenova/bge-base-en-v1.5",
        variant: "BGEBaseENV15",
        dimensions: 768,
        description: "BGE base English v1.5",
        backend: Backend::Onnx,
    },
    ModelInfo {
        code: "Xenova/bge-large-en-v1.5",
        variant: "BGELargeENV15",
        dimensions: 1024,
        description: "BGE large English v1.5",
        backend: Backend::Onnx,
    },
    // --- BGE (Chinese) ---
    ModelInfo {
        code: "Xenova/bge-small-zh-v1.5",
        variant: "BGESmallZHV15",
        dimensions: 512,
        description: "BGE small Chinese v1.5",
        backend: Backend::Onnx,
    },
    ModelInfo {
        code: "Xenova/bge-large-zh-v1.5",
        variant: "BGELargeZHV15",
        dimensions: 1024,
        description: "BGE large Chinese v1.5",
        backend: Backend::Onnx,
    },
    // --- BGE-M3 (multilingual) ---
    ModelInfo {
        code: "BAAI/bge-m3",
        variant: "BGEM3",
        dimensions: 1024,
        description: "BGE-M3 multilingual (100+ languages)",
        backend: Backend::Onnx,
    },
    // --- GTE ---
    ModelInfo {
        code: "Alibaba-NLP/gte-base-en-v1.5",
        variant: "GTEBaseENV15",
        dimensions: 768,
        description: "GTE base English v1.5",
        backend: Backend::Onnx,
    },
    ModelInfo {
        code: "Alibaba-NLP/gte-large-en-v1.5",
        variant: "GTELargeENV15",
        dimensions: 1024,
        description: "GTE large English v1.5",
        backend: Backend::Onnx,
    },
    // --- Nomic ---
    ModelInfo {
        code: "nomic-ai/nomic-embed-text-v1",
        variant: "NomicEmbedTextV1",
        dimensions: 768,
        description: "Nomic Embed Text v1",
        backend: Backend::Onnx,
    },
    ModelInfo {
        code: "nomic-ai/nomic-embed-text-v1.5",
        variant: "NomicEmbedTextV15",
        dimensions: 768,
        description: "Nomic Embed Text v1.5 (8192 context)",
        backend: Backend::Onnx,
    },
    // --- MixedBread ---
    ModelInfo {
        code: "mixedbread-ai/mxbai-embed-large-v1",
        variant: "MxbaiEmbedLargeV1",
        dimensions: 1024,
        description: "MixedBread AI large model",
        backend: Backend::Onnx,
    },
    // --- ModernBERT ---
    ModelInfo {
        code: "lightonai/modernbert-embed-large",
        variant: "ModernBertEmbedLarge",
        dimensions: 1024,
        description: "ModernBERT embedding model",
        backend: Backend::Onnx,
    },
    // --- Multilingual E5 ---
    ModelInfo {
        code: "intfloat/multilingual-e5-small",
        variant: "MultilingualE5Small",
        dimensions: 384,
        description: "Multilingual E5 small",
        backend: Backend::Onnx,
    },
    ModelInfo {
        code: "intfloat/multilingual-e5-base",
        variant: "MultilingualE5Base",
        dimensions: 768,
        description: "Multilingual E5 base",
        backend: Backend::Onnx,
    },
    ModelInfo {
        code: "Qdrant/multilingual-e5-large-onnx",
        variant: "MultilingualE5Large",
        dimensions: 1024,
        description: "Multilingual E5 large",
        backend: Backend::Onnx,
    },
    // --- Paraphrase ---
    ModelInfo {
        code: "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q",
        variant: "ParaphraseMLMiniLML12V2",
        dimensions: 384,
        description: "Paraphrase multilingual MiniLM-L12 (quantized)",
        backend: Backend::Onnx,
    },
    ModelInfo {
        code: "Xenova/paraphrase-multilingual-mpnet-base-v2",
        variant: "ParaphraseMLMpnetBaseV2",
        dimensions: 768,
        description: "Paraphrase multilingual mpnet-base-v2",
        backend: Backend::Onnx,
    },
    // --- Jina ---
    ModelInfo {
        code: "jinaai/jina-embeddings-v2-base-code",
        variant: "JinaEmbeddingsV2BaseCode",
        dimensions: 768,
        description: "Jina v2 code embedding",
        backend: Backend::Onnx,
    },
    ModelInfo {
        code: "jinaai/jina-embeddings-v2-base-en",
        variant: "JinaEmbeddingsV2BaseEN",
        dimensions: 768,
        description: "Jina v2 base English",
        backend: Backend::Onnx,
    },
    // --- Gemma ---
    ModelInfo {
        code: "onnx-community/embeddinggemma-300m-ONNX",
        variant: "EmbeddingGemma300M",
        dimensions: 768,
        description: "Gemma 300M embedding model",
        backend: Backend::Onnx,
    },
    // --- Snowflake Arctic ---
    ModelInfo {
        code: "snowflake/snowflake-arctic-embed-xs",
        variant: "SnowflakeArcticEmbedXS",
        dimensions: 384,
        description: "Snowflake Arctic Embed XS",
        backend: Backend::Onnx,
    },
    ModelInfo {
        code: "Snowflake/snowflake-arctic-embed-m",
        variant: "SnowflakeArcticEmbedM",
        dimensions: 768,
        description: "Snowflake Arctic Embed M",
        backend: Backend::Onnx,
    },
    ModelInfo {
        code: "snowflake/snowflake-arctic-embed-l",
        variant: "SnowflakeArcticEmbedL",
        dimensions: 1024,
        description: "Snowflake Arctic Embed L",
        backend: Backend::Onnx,
    },
    // --- Qwen3-Embedding (candle backend, SOTA open-weight) ---
    ModelInfo {
        code: "Qwen/Qwen3-Embedding-0.6B",
        variant: "",
        dimensions: 1024,
        description: "Qwen3-Embedding 0.6B (candle)",
        backend: Backend::Qwen3,
    },
    ModelInfo {
        code: "Qwen/Qwen3-Embedding-4B",
        variant: "",
        dimensions: 2560,
        description: "Qwen3-Embedding 4B (candle)",
        backend: Backend::Qwen3,
    },
    ModelInfo {
        code: "Qwen/Qwen3-Embedding-8B",
        variant: "",
        dimensions: 4096,
        description: "Qwen3-Embedding 8B (candle, top of MTEB)",
        backend: Backend::Qwen3,
    },
];

const RERANK_MODELS: &[RerankInfo] = &[
    RerankInfo {
        code: "BAAI/bge-reranker-base",
        variant: "BGERerankerBase",
        description: "BGE reranker base (default)",
    },
    RerankInfo {
        code: "rozgo/bge-reranker-v2-m3",
        variant: "BGERerankerV2M3",
        description: "BGE reranker v2-m3 (multilingual)",
    },
    RerankInfo {
        code: "jinaai/jina-reranker-v1-turbo-en",
        variant: "JINARerankerV1TurboEn",
        description: "Jina reranker v1 turbo (English)",
    },
    RerankInfo {
        code: "jinaai/jina-reranker-v2-base-multilingual",
        variant: "JINARerankerV2BaseMultiligual",
        description: "Jina reranker v2 base (multilingual)",
    },
];

/// Resolve a user-supplied name to a loadable text-embedding model.
///
/// Accepts a Hugging Face code or fastembed variant from [`dense_models`].
/// Unknown names fall back to parsing as an ONNX variant directly, and only
/// error if that also fails.
///
/// # Errors
///
/// Returns [`VqtrsError::UnknownModel`] if the name matches no catalog entry and
/// is not a parseable fastembed variant.
pub fn resolve_dense(name: &str) -> Result<Resolved> {
    if let Some(info) = DENSE_MODELS
        .iter()
        .find(|m| m.code == name || (!m.variant.is_empty() && m.variant == name))
    {
        return match info.backend {
            #[cfg(feature = "qwen3")]
            Backend::Qwen3 => Ok(Resolved::Qwen3 {
                repo: info.code.to_owned(),
                dimensions: info.dimensions,
            }),
            #[cfg(not(feature = "qwen3"))]
            Backend::Qwen3 => Err(VqtrsError::Backend(format!(
                "model `{name}` needs the `qwen3` feature (rebuild with --features qwen3)"
            ))),
            Backend::Onnx => info.variant.parse::<EmbeddingModel>().map_or_else(
                |_| Err(VqtrsError::UnknownModel(name.to_owned())),
                |model| {
                    Ok(Resolved::Onnx {
                        model,
                        dimensions: Some(info.dimensions),
                    })
                },
            ),
        };
    }

    name.parse::<EmbeddingModel>().map_or_else(
        |_| Err(VqtrsError::UnknownModel(name.to_owned())),
        |model| {
            Ok(Resolved::Onnx {
                model,
                dimensions: None,
            })
        },
    )
}

/// Resolve a user-supplied name to a [`RerankerModel`].
///
/// Accepts a Hugging Face code or fastembed variant from [`rerank_models`].
///
/// # Errors
///
/// Returns [`VqtrsError::UnknownModel`] if the name matches no known reranker.
pub fn resolve_reranker(name: &str) -> Result<RerankerModel> {
    match name {
        "BAAI/bge-reranker-base" | "BGERerankerBase" => Ok(RerankerModel::BGERerankerBase),
        "rozgo/bge-reranker-v2-m3" | "BGERerankerV2M3" => Ok(RerankerModel::BGERerankerV2M3),
        "jinaai/jina-reranker-v1-turbo-en" | "JINARerankerV1TurboEn" => {
            Ok(RerankerModel::JINARerankerV1TurboEn)
        }
        "jinaai/jina-reranker-v2-base-multilingual" | "JINARerankerV2BaseMultiligual" => {
            Ok(RerankerModel::JINARerankerV2BaseMultiligual)
        }
        other => Err(VqtrsError::UnknownModel(other.to_owned())),
    }
}

/// A single sparse-embedding model in the catalog.
#[derive(Debug, Clone, Copy)]
pub struct SparseInfo {
    /// Hugging Face style code.
    pub code: &'static str,
    /// fastembed variant name.
    pub variant: &'static str,
    /// Whether this model also produces a dense vector (BGE-M3 via `M3Engine`).
    pub joint_dense: bool,
    /// Short human-readable description.
    pub description: &'static str,
}

const SPARSE_MODELS: &[SparseInfo] = &[
    SparseInfo {
        code: "Qdrant/Splade_PP_en_v1",
        variant: "SPLADEPPV1",
        joint_dense: false,
        description: "SPLADE++ sparse model (English, default)",
    },
    SparseInfo {
        code: "BAAI/bge-m3",
        variant: "BGEM3",
        joint_dense: true,
        description: "BGE-M3 dense+sparse, 100+ languages, 8192 context",
    },
];

/// The full sparse-embedding catalog.
#[must_use]
pub const fn sparse_models() -> &'static [SparseInfo] {
    SPARSE_MODELS
}

/// Resolve a user-supplied name to a [`SparseModel`].
///
/// Accepts a Hugging Face code or fastembed variant from [`sparse_models`].
///
/// # Errors
///
/// Returns [`VqtrsError::UnknownModel`] if the name matches no known sparse model.
pub fn resolve_sparse(name: &str) -> Result<SparseModel> {
    match name {
        "Qdrant/Splade_PP_en_v1" | "prithivida/Splade_PP_en_v1" | "SPLADEPPV1" | "splade" => {
            Ok(SparseModel::SPLADEPPV1)
        }
        "BAAI/bge-m3" | "BGEM3" => Ok(SparseModel::BGEM3),
        other => Err(VqtrsError::UnknownModel(other.to_owned())),
    }
}

/// Resolve a user-supplied name to a [`Bgem3Model`] for joint dense+sparse.
///
/// # Errors
///
/// Returns [`VqtrsError::UnknownModel`] if the name is not a known BGE-M3 model.
pub fn resolve_m3(name: &str) -> Result<Bgem3Model> {
    match name {
        "BAAI/bge-m3" | "BGEM3Q" | "BGEM3" | "bge-m3" => Ok(Bgem3Model::BGEM3Q),
        other => Err(VqtrsError::UnknownModel(other.to_owned())),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Backend, DENSE_MODELS, Resolved, dense_models, rerank_models, resolve_dense, resolve_m3,
        resolve_reranker, resolve_sparse, sparse_models,
    };

    #[test]
    fn resolves_onnx_by_code() {
        let resolved = resolve_dense("intfloat/multilingual-e5-small").expect("known code");
        match resolved {
            Resolved::Onnx { dimensions, .. } => assert_eq!(dimensions, Some(384)),
            #[cfg(feature = "qwen3")]
            Resolved::Qwen3 { .. } => panic!("expected ONNX backend"),
        }
    }

    #[test]
    fn resolves_onnx_by_variant() {
        let resolved = resolve_dense("MultilingualE5Small").expect("known variant");
        assert!(matches!(resolved, Resolved::Onnx { .. }));
    }

    #[cfg(feature = "qwen3")]
    #[test]
    fn resolves_qwen3_by_repo() {
        let resolved = resolve_dense("Qwen/Qwen3-Embedding-0.6B").expect("known qwen3");
        match resolved {
            Resolved::Qwen3 { repo, dimensions } => {
                assert_eq!(repo, "Qwen/Qwen3-Embedding-0.6B");
                assert_eq!(dimensions, 1024);
            }
            Resolved::Onnx { .. } => panic!("expected Qwen3 backend"),
        }
    }

    #[cfg(not(feature = "qwen3"))]
    #[test]
    fn qwen3_model_errors_without_feature() {
        assert!(resolve_dense("Qwen/Qwen3-Embedding-0.6B").is_err());
    }

    #[test]
    fn unknown_model_errors() {
        assert!(resolve_dense("definitely-not-a-real-model").is_err());
    }

    #[test]
    fn resolves_reranker_by_code_and_variant() {
        assert!(resolve_reranker("BAAI/bge-reranker-base").is_ok());
        assert!(resolve_reranker("JINARerankerV2BaseMultiligual").is_ok());
        assert!(resolve_reranker("nope").is_err());
    }

    #[test]
    fn every_catalog_entry_resolves() {
        for m in dense_models() {
            let resolved = resolve_dense(m.code);
            // Qwen3 entries only resolve when the `qwen3` feature is built in.
            if m.backend == Backend::Qwen3 && cfg!(not(feature = "qwen3")) {
                assert!(
                    resolved.is_err(),
                    "`{}` must need the qwen3 feature",
                    m.code
                );
            } else {
                resolved.unwrap_or_else(|_| panic!("code `{}` must resolve", m.code));
            }
            assert_eq!(
                m.variant.is_empty(),
                m.backend == Backend::Qwen3,
                "variant emptiness must match Qwen3 backend for `{}`",
                m.code,
            );
        }
        for m in rerank_models() {
            resolve_reranker(m.code).unwrap_or_else(|_| panic!("code `{}` must resolve", m.code));
        }
        for m in sparse_models() {
            resolve_sparse(m.code).unwrap_or_else(|_| panic!("code `{}` must resolve", m.code));
        }
    }

    #[test]
    fn resolves_sparse_and_m3() {
        assert!(resolve_sparse("Qdrant/Splade_PP_en_v1").is_ok());
        assert!(resolve_sparse("splade").is_ok());
        assert!(resolve_sparse("BAAI/bge-m3").is_ok());
        assert!(resolve_sparse("nope").is_err());
        assert!(resolve_m3("BAAI/bge-m3").is_ok());
        assert!(resolve_m3("nope").is_err());
    }

    #[test]
    fn catalog_is_non_empty() {
        assert!(!DENSE_MODELS.is_empty());
    }
}
