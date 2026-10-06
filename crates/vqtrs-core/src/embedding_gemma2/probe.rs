//! Per-operation parity probe against PyTorch (run on demand, not in CI).
//!
//! Replays every operation captured by `scripts/parity/eg2_op_probe.py` on
//! the captured input and reports how much of the output is bit-identical.
//!
//! ```text
//! EG2_PROBE_DIR=/tmp/eg2-parity/probe EG2_MODEL_DIR=<snapshot> \
//!   cargo test --release -p vqtrs-core --features embeddinggemma2 probe -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::ops::{apply_rope, gelu_tanh, linear, rms_norm, rope_tables, softmax_last_dim};

fn load(dir: &Path, name: &str) -> Tensor {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let entry = &manifest["tensors"][name];
    let shape: Vec<usize> = entry["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let bytes = std::fs::read(dir.join(entry["file"].as_str().unwrap())).unwrap();
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

fn report(label: &str, ours: &Tensor, reference: &Tensor) {
    let a = ours.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: shape mismatch {:?} vs {:?}",
        ours.shape(),
        reference.shape()
    );
    let same = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() == y.to_bits())
        .count();
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    let tag = if same == a.len() { "BIT-IDENTICAL" } else { "" };
    println!(
        "{label:44} identical {:>7.3}%  max|d| {max:.3e}  {tag}",
        100.0 * same as f64 / a.len() as f64
    );
}

#[test]
#[ignore = "needs a captured probe directory and the model weights"]
fn probe() {
    let dir = PathBuf::from(std::env::var("EG2_PROBE_DIR").expect("EG2_PROBE_DIR"));
    let model = PathBuf::from(std::env::var("EG2_MODEL_DIR").expect("EG2_MODEL_DIR"));
    let vb = VarBuilder::from_buffered_safetensors(
        std::fs::read(model.join("model.safetensors")).unwrap(),
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let l0 = vb.pp("language_model.layers.0");
    let w = |n: &str| l0.get_unchecked(n).unwrap();
    let t = |n: &str| load(&dir, n);
    let eps = 1e-6_f32;

    println!("--- GEMM (x @ W^T) ---");
    for (name, key) in [
        ("self_attn.q_proj", "self_attn.q_proj.weight"),
        ("self_attn.k_proj", "self_attn.k_proj.weight"),
        ("self_attn.o_proj", "self_attn.o_proj.weight"),
        ("mlp.gate_proj", "mlp.gate_proj.weight"),
        ("mlp.down_proj", "mlp.down_proj.weight"),
    ] {
        report(
            name,
            &linear(&t(&format!("{name}.in")), &w(key)).unwrap(),
            &t(&format!("{name}.out")),
        );
    }

    println!("--- RMS norm (sum reduction + 1/sqrt) ---");
    report(
        "input_layernorm",
        &rms_norm(
            &t("input_layernorm.in"),
            Some(&w("input_layernorm.weight")),
            eps,
        )
        .unwrap(),
        &t("input_layernorm.out"),
    );
    report(
        "self_attn.q_norm (per head)",
        &rms_norm(
            &t("self_attn.q_norm.in"),
            Some(&w("self_attn.q_norm.weight")),
            eps,
        )
        .unwrap(),
        &t("self_attn.q_norm.out"),
    );
    report(
        "self_attn.v_norm (no scale)",
        &rms_norm(&t("self_attn.v_norm.in"), None, eps).unwrap(),
        &t("self_attn.v_norm.out"),
    );

    println!("--- elementwise transcendentals ---");
    report(
        "gelu_tanh (tanh)",
        &gelu_tanh(&t("mlp.act_fn.in")).unwrap(),
        &t("mlp.act_fn.out"),
    );
    let cos = t("rope.cos");
    let (seq, hd) = cos.dims2().unwrap();
    // Sliding layers use head_dim 256 / theta 1e4, full layers 512 / 1e6.
    let theta = if hd == 256 { 10_000.0 } else { 1_000_000.0 };
    let (our_cos, our_sin) = rope_tables(seq, hd, theta, &Device::Cpu).unwrap();
    report("rope cos table (powf + cos)", &our_cos, &cos);
    report("rope sin table (powf + sin)", &our_sin, &t("rope.sin"));
    report(
        "apply_rope (given tables)",
        &apply_rope(&t("rope.q_in"), &cos, &t("rope.sin")).unwrap(),
        &t("rope.q_out"),
    );

    println!("--- attention ---");
    let (q, k) = (t("attn.q"), t("attn.k"));
    let rep = q.dim(0).unwrap() / k.dim(0).unwrap();
    let k_rep = super::text::repeat_kv(&k, rep).unwrap();
    report(
        "scores = q @ k^T (batched GEMM)",
        &q.matmul(&k_rep.t().unwrap()).unwrap(),
        &t("attn.scores"),
    );
    let manifest = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    if manifest.contains("\"attn.mask\"") {
        let seq = t("attn.mask").dim(0).unwrap();
        let ours = super::text::sliding_window_mask(seq, 512, &Device::Cpu)
            .unwrap()
            .unwrap();
        report("sliding-window mask", &ours, &t("attn.mask"));
        let masked = t("attn.scores").broadcast_add(&ours).unwrap();
        report("scores + mask", &masked, &t("attn.scores_masked"));
    }
    report(
        "softmax (max, exp, sum, recip)",
        &softmax_last_dim(&t("attn.scores_masked")).unwrap(),
        &t("attn.probs"),
    );
    let v_rep = super::text::repeat_kv(&t("attn.v"), rep).unwrap();
    let out = t("attn.probs")
        .matmul(&v_rep)
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    report("probs @ v (batched GEMM)", &out, &t("attn.out"));
}
