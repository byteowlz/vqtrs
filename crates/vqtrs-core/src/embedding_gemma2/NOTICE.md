# Third-party notices

This is a modified Rust implementation, not an unmodified distribution of
any of the projects below. Source comments identify the reference files and
versions; retain these notices and license files when redistributing.

- transformers 5.19.0 model/processor code: Copyright 2026 Google LLC and
  Copyright 2026 The HuggingFace Team, Apache License 2.0
  (`transformers.LICENSE`). The text, vision and audio architecture and
  preprocessing were ported to candle/safe Rust.
- PyTorch 2.14.1 CPU arithmetic: copyright holders listed in
  `pytorch.LICENSE`, BSD 3-clause. `torch_cpu.rs`, `layer_norm.rs` and the
  resize/arithmetic routines emulate the CPU kernels in safe scalar Rust.
- SLEEF commit 5a1d179d: Copyright Naoki Shibata and contributors, Boost
  Software License 1.0 (`sleef.LICENSE`). The AdvSIMD transcendental routines
  and range-reduction table were ported to scalar Rust with explicit FMA.
- NumPy 2.4.6: Copyright 2005-2025 NumPy Developers, BSD 3-clause
  (`numpy.LICENSE`). Audio table generation, framing and complex magnitude
  arithmetic reproduce the reference NumPy operations.
- pocketfft commit 33ae5dc9: copyright holders and BSD 3-clause terms in
  `pocketfft.LICENSE`. Only the power-of-two real forward transform was
  ported, with f64 computation narrowed to complex64.
- libjpeg-turbo 3.1.4.1: original source notices in `jpeg.NOTICE`, license
  terms in `libjpeg-turbo.LICENSE.md` and the unmodified `README.ijg`.
  `jpeg.rs` is a modified safe Rust decoder with ported IDCT, upsampling and
  color-conversion arithmetic.

This software is based in part on the work of the Independent JPEG Group.
