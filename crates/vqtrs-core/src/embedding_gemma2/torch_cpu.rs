//! Bit-exact emulation of the PyTorch 2.14 CPU kernels (aarch64, 4-lane
//! `Vectorized<float>`, SLEEF) whose results depend on evaluation order:
//! reductions, softmax, the L2 norm, and the vectorised elementwise loop.
//!
//! References (pytorch v2.14.1): `aten/src/ATen/native/cpu/SumKernel.cpp`
//! (`cascade_sum`), `cpu/vec/functional_base.h` (`reduce_all`, `map`),
//! `native/cpu/SoftMaxKernel.cpp` (`_vec_softmax_lastdim`),
//! `native/cpu/ReduceOpsKernel.cpp` (vectorised `norm` for `p = 2`) and
//! `native/cpu/Loops.h` (`vectorized_loop`).
//!
//! Single-threaded semantics throughout: the reference is run with
//! `torch.set_num_threads(1)`; more threads change where PyTorch splits work.

#![expect(
    clippy::float_cmp,
    reason = "exact comparisons mirror the reference kernels"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "unfused multiply-adds are rounded separately on purpose, as in the reference"
)]

use super::sleef;

/// Lanes of `Vectorized<float>` on aarch64 (`float32x4_t`).
const LANES: usize = 4;
type V = [f32; LANES];

fn vadd(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

/// `CeilLog2` from `ATen/native/cpu/utils.h`.
const fn ceil_log2(x: usize) -> usize {
    if x <= 2 {
        1
    } else {
        (usize::BITS - (x - 1).leading_zeros()) as usize
    }
}

/// `multi_row_sum<V, NROWS>`: `NROWS` interleaved accumulators, each a
/// four-level cascade that flushes every `2^level_power` steps.
fn multi_row_sum<const NROWS: usize>(load: impl Fn(usize, usize) -> V, size: usize) -> [V; NROWS] {
    const LEVELS: usize = 4;
    let level_power = 4.max(ceil_log2(size) / LEVELS);
    let level_step = 1 << level_power;
    let level_mask = level_step - 1;
    let mut acc = [[[0.0_f32; LANES]; NROWS]; LEVELS];

    let mut i = 0;
    while i + level_step <= size {
        for _ in 0..level_step {
            for (k, slot) in acc[0].iter_mut().enumerate() {
                *slot = vadd(*slot, load(i, k));
            }
            i += 1;
        }
        for j in 1..LEVELS {
            let (lower, upper) = acc.split_at_mut(j);
            for (hi, lo) in upper[0].iter_mut().zip(lower[j - 1].iter_mut()) {
                *hi = vadd(*hi, *lo);
                *lo = [0.0; LANES];
            }
            if i & (level_mask << (j * level_power)) != 0 {
                break;
            }
        }
    }
    for i in i..size {
        for (k, slot) in acc[0].iter_mut().enumerate() {
            *slot = vadd(*slot, load(i, k));
        }
    }
    let [mut total, rest @ ..] = acc;
    for level in rest {
        for (t, v) in total.iter_mut().zip(level) {
            *t = vadd(*t, v);
        }
    }
    total
}

/// `row_sum<V>`: a 4-way ILP split over `size` items, then the partials in order.
fn row_sum(load: impl Fn(usize) -> V, size: usize) -> V {
    const ILP: usize = 4;
    let size_ilp = size / ILP;
    let mut partial = multi_row_sum::<ILP>(|i, k| load(i * ILP + k), size_ilp);
    for i in size_ilp * ILP..size {
        partial[0] = vadd(partial[0], load(i));
    }
    for k in 1..ILP {
        partial[0] = vadd(partial[0], partial[k]);
    }
    partial[0]
}

/// `sum(-1)` of one contiguous row (`vectorized_inner_sum` for `n >= 4`,
/// `scalar_inner_sum` otherwise), stored by accumulating into `0.0`.
pub fn sum_row(row: &[f32]) -> f32 {
    let n = row.len();
    let total = if n >= LANES {
        let vecs = n / LANES;
        let acc = row_sum(
            |i| [row[i * 4], row[i * 4 + 1], row[i * 4 + 2], row[i * 4 + 3]],
            vecs,
        );
        let mut total = 0.0_f32;
        for &x in &row[vecs * LANES..] {
            total += x;
        }
        for lane in acc {
            total += lane;
        }
        total
    } else {
        row_sum(|i| [row[i], 0.0, 0.0, 0.0], n)[0]
    };
    0.0 + total
}

/// `sum(0)` of a row-major `(rows, cols)` matrix whose columns are contiguous
/// (`vectorized_outer_sum`): 16-column blocks, then 4-column, then scalars.
pub fn sum_cols(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let at = |r: usize, c: usize| data[r * cols + c];
    let vec_at = |r: usize, c: usize| [at(r, c), at(r, c + 1), at(r, c + 2), at(r, c + 3)];
    let mut out = vec![0.0_f32; cols];
    let mut j = 0;
    while j + 4 * LANES <= cols {
        let sums = multi_row_sum::<4>(|i, k| vec_at(i, j + k * LANES), rows);
        for (k, s) in sums.iter().enumerate() {
            out[j + k * LANES..j + (k + 1) * LANES].copy_from_slice(s);
        }
        j += 4 * LANES;
    }
    while j + LANES <= cols {
        out[j..j + LANES].copy_from_slice(&row_sum(|i| vec_at(i, j), rows));
        j += LANES;
    }
    for (c, slot) in out.iter_mut().enumerate().skip(j) {
        *slot = row_sum(|i| [at(i, c), 0.0, 0.0, 0.0], rows)[0];
    }
    out.iter().map(|s| 0.0 + s).collect()
}

/// `vmaxq_f32` lane semantics: NaN-propagating, `+0` over `-0`.
fn vmax(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        f32::from_bits(a.to_bits() & b.to_bits())
    } else if a > b {
        a
    } else {
        b
    }
}

/// `vec::reduce_all` with the aarch64 horizontal step for a generic operator:
/// `op(op(a0, a2), op(a1, a3))`.
fn reduce_all(data: &[f32], op: fn(f32, f32) -> f32) -> f32 {
    let n = data.len();
    let load = |d: usize, count: usize| {
        let mut v = [0.0_f32; LANES];
        v[..count].copy_from_slice(&data[d..d + count]);
        v
    };
    if n < LANES {
        // Slow path: lane 0 folds the remaining lanes in order.
        let lanes = load(0, n);
        let mut acc = lanes[0];
        for &lane in &lanes[1..n] {
            acc = op(acc, lane);
        }
        return acc;
    }
    let mut acc = load(0, LANES);
    let mut d = LANES;
    while d < n - n % LANES {
        let v = load(d, LANES);
        acc = std::array::from_fn(|l| op(acc[l], v[l]));
        d += LANES;
    }
    if n > d {
        let v = load(d, n - d);
        for l in 0..n - d {
            acc[l] = op(acc[l], v[l]);
        }
    }
    op(op(acc[0], acc[2]), op(acc[1], acc[3]))
}

/// `softmax(-1)` of one row: `e = exp(x - max)`, `e * (1 / sum(e))`.
pub fn softmax_row(row: &mut [f32]) {
    let max = reduce_all(row, vmax);
    for x in row.iter_mut() {
        *x = sleef::expf(*x - max);
    }
    let inv = 1.0_f32 / reduce_all(row, |a, b| a + b);
    for x in row.iter_mut() {
        *x *= inv;
    }
}

/// Vectorised `p = 2` norm of one contiguous row: four lane accumulators of
/// `x * x` (separately rounded), lanes summed in order, a fused scalar tail.
pub fn l2_norm_row(row: &[f32]) -> f32 {
    let n = row.len();
    let mut acc = [0.0_f32; LANES];
    let body = n - n % LANES;
    for chunk in row[..body].chunks_exact(LANES) {
        for (a, &x) in acc.iter_mut().zip(chunk) {
            *a += x * x;
        }
    }
    let mut total = acc[0];
    for &lane in &acc[1..] {
        total += lane;
    }
    for &x in &row[body..] {
        total = x.mul_add(x, total);
    }
    total.sqrt()
}

/// `cpu_kernel_vec`'s `vectorized_loop`: pairs of 4-lane vectors, then a
/// scalar tail of up to 7 elements through `scalar`.
pub fn vectorized_loop(data: &mut [f32], vector: impl Fn(V) -> V, scalar: impl Fn(f32) -> f32) {
    let body = data.len() - data.len() % (2 * LANES);
    for chunk in data[..body].chunks_exact_mut(LANES) {
        let v = vector([chunk[0], chunk[1], chunk[2], chunk[3]]);
        chunk.copy_from_slice(&v);
    }
    for x in &mut data[body..] {
        *x = scalar(*x);
    }
}

/// `gelu(approximate="tanh")` over a contiguous buffer.
pub fn gelu_tanh(data: &mut [f32]) {
    let beta = (std::f64::consts::SQRT_2 * std::f64::consts::FRAC_2_SQRT_PI * 0.5) as f32;
    let kappa = 0.044_715_f64 as f32;
    vectorized_loop(
        data,
        |v| {
            v.map(|x| {
                let x_cube = x * x * x;
                let inner = beta * (x + kappa * x_cube);
                0.5 * x * (1.0 + sleef::tanhf(inner))
            })
        },
        // The scalar tail is C++ compiled with `-ffp-contract=on`, which fuses
        // `x + kappa * x_cube`, and calls libm `tanhf`.
        |x| {
            let x_cube = x * x * x;
            let inner = beta * kappa.mul_add(x_cube, x);
            0.5 * x * (1.0 + inner.tanh())
        },
    );
}

/// `cos` over a contiguous buffer.
pub fn cos(data: &mut [f32]) {
    vectorized_loop(data, sleef::cos4, f32::cos);
}

/// `sin` over a contiguous buffer.
pub fn sin(data: &mut [f32]) {
    vectorized_loop(data, sleef::sin4, f32::sin);
}

/// `base ** exps` (`torch.pow(Scalar, Tensor)`) over a contiguous buffer.
pub fn pow_scalar_base(base: f32, exps: &mut [f32]) {
    vectorized_loop(exps, |v| v.map(|e| sleef::powf(base, e)), |e| base.powf(e));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_log2_matches_aten() {
        assert_eq!(ceil_log2(1), 1);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(16), 4);
        assert_eq!(ceil_log2(17), 5);
    }

    #[test]
    fn sums_are_exact_for_integers() {
        let row: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        assert_eq!(sum_row(&row), 499_500.0);
        let cols = sum_cols(&row, 50, 20);
        assert_eq!(cols[0], (0..50).map(|r| (r * 20) as f32).sum::<f32>());
    }

    #[test]
    fn softmax_row_sums_to_one() {
        let mut row = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        softmax_row(&mut row);
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
