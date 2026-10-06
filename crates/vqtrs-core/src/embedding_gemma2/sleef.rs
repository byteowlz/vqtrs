//! Bit-exact scalar port of the SLEEF AdvSIMD single-precision routines that
//! PyTorch 2.14 (aarch64, `AT_BUILD_ARM_VEC256_WITH_SLEEF`) calls from its
//! vectorised CPU kernels: `Sleef_{tanh,exp,sin,cos,pow}f4_u10`.
//!
//! Ported from SLEEF commit 5a1d179d (`src/libm/sleefsimdsp.c`,
//! `src/common/df.h`, `src/arch/helperadvsimd.h`, `CONFIG == 1`), copyright
//! Naoki Shibata and contributors, Boost Software License 1.0.
//!
//! SIMD lanes are independent, so each routine is ported as a scalar function
//! of one lane, except `sin`/`cos`, whose argument-reduction path is chosen
//! once per 4-lane vector (see [`sin4`]). Where SLEEF uses `vfmaq_f32`/
//! `vfmsq_f32` the port uses [`f32::mul_add`] (one rounding); everywhere else
//! each operation is rounded separately, exactly as in the C source.
//!
//! Constants keep SLEEF's literal types: `f`-suffixed literals are `f32`
//! literals, unsuffixed ones are `double` literals narrowed to `float`
//! (written `x_f64 as f32`); the two can round differently.

#![expect(
    clippy::excessive_precision,
    reason = "constants are kept verbatim from the SLEEF source for auditability"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "variable names follow the SLEEF source line by line"
)]
#![expect(
    clippy::float_cmp,
    reason = "exact comparisons are part of the ported algorithm"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "unfused multiply-adds are rounded separately on purpose, as in SLEEF"
)]

use super::sleef_rempitab::REMPITAB_SP;

/// A double-float value `x + y` (SLEEF `vfloat2`).
#[derive(Clone, Copy)]
struct F2 {
    x: f32,
    y: f32,
}

const fn f2(x: f32, y: f32) -> F2 {
    F2 { x, y }
}

/// `R_LN2f`, the same `f32` as SLEEF's literal.
const R_LN2F: f32 = std::f32::consts::LOG2_E;
const L2UF: f32 = 0.693_145_751_953_125_f32;
const L2LF: f32 = 1.428_606_765_330_187_045e-06_f32;
const PI_A2F: f32 = 3.141_479_492_187_5_f32;
const PI_B2F: f32 = 0.000_113_159_418_106_079_101_56_f32;
const PI_C2F: f32 = 1.984_187_258_941_005_893_6e-09_f32;
const TRIGRANGEMAX2F: f32 = 125.0_f32;
/// `(float)M_1_PI`: a `double` constant narrowed at the `vcast_vf_f` call.
const M_1_PI_F: f32 = std::f64::consts::FRAC_1_PI as f32;
const FLT_MIN: f32 = f32::MIN_POSITIVE;

// --- AdvSIMD helpers (helperadvsimd.h / sleefsimdsp.c) ---

/// `vmla_vf_vf_vf_vf(x, y, z)` = `z + x * y`, fused (`vfmaq_f32`).
const fn mla(x: f32, y: f32, z: f32) -> f32 {
    x.mul_add(y, z)
}

/// `vfmanp_vf_vf_vf_vf(x, y, z)` = `z - x * y`, fused (`vfmsq_f32`).
fn fmanp(x: f32, y: f32, z: f32) -> f32 {
    (-x).mul_add(y, z)
}

/// `vfmapn_vf_vf_vf_vf(x, y, z)` = `x * y - z`, fused.
fn fmapn(x: f32, y: f32, z: f32) -> f32 {
    x.mul_add(y, -z)
}

/// `vrint_vf_vf`: round to nearest, ties to even (`vrndnq_f32`).
const fn rint(x: f32) -> f32 {
    x.round_ties_even()
}

/// `vrint_vi2_vf`: `vcvtq_s32_f32(vrndnq_f32(x))` (saturating).
const fn rint_i(x: f32) -> i32 {
    rint(x) as i32
}

/// `vmulsign_vf_vf_vf`: `x` with its sign flipped where `y` is negative.
const fn mulsign(x: f32, y: f32) -> f32 {
    f32::from_bits(x.to_bits() ^ (y.to_bits() & 0x8000_0000))
}

/// `vor_vm_vo32_vm(mask, x)`: all bits set (a NaN) where `mask` holds.
const fn or_all_bits(mask: bool, x: f32) -> f32 {
    if mask { f32::from_bits(u32::MAX) } else { x }
}

/// `vandnot_vm_vo32_vm(mask, x)`: `+0.0` where `mask` holds.
const fn and_not(mask: bool, x: f32) -> f32 {
    if mask { 0.0 } else { x }
}

/// `vpow2i_vf_vi2`: `2^q` by building the exponent field.
const fn pow2i(q: i32) -> f32 {
    f32::from_bits(q.wrapping_add(0x7f).wrapping_shl(23) as u32)
}

/// `vldexp2_vf_vf_vi2`.
fn ldexp2(d: f32, e: i32) -> f32 {
    d * pow2i(e >> 1) * pow2i(e - (e >> 1))
}

/// `vldexp3_vf_vf_vi2`: add `q` to the exponent field directly.
const fn ldexp3(d: f32, q: i32) -> f32 {
    f32::from_bits((d.to_bits() as i32).wrapping_add(q.wrapping_shl(23)) as u32)
}

/// `vldexp_vf_vf_vi2` (the four-multiply variant).
fn ldexp(x: f32, q: i32) -> f32 {
    let mut m = q >> 31;
    m = (((m + q) >> 6) - m) << 4;
    let q = q - (m << 2);
    m += 0x7f;
    m = if m > 0 { m } else { 0 };
    m = if m > 0xff { 0xff } else { m };
    let u = f32::from_bits(m.wrapping_shl(23) as u32);
    let x = x * u * u * u * u;
    x * f32::from_bits(q.wrapping_add(0x7f).wrapping_shl(23) as u32)
}

/// `vilogb2k_vi2_vf`: unbiased exponent field.
const fn ilogb2k(d: f32) -> i32 {
    ((d.to_bits() >> 23) & 0xff) as i32 - 0x7f
}

// --- double-float arithmetic (df.h, ENABLE_FMA_SP) ---

fn dfneg(x: F2) -> F2 {
    f2(-x.x, -x.y)
}

fn dfnormalize(t: F2) -> F2 {
    let s = t.x + t.y;
    f2(s, (t.x - s) + t.y)
}

fn dfscale(d: F2, s: f32) -> F2 {
    f2(d.x * s, d.y * s)
}

fn dfadd_vf_vf(x: f32, y: f32) -> F2 {
    let s = x + y;
    f2(s, (x - s) + y)
}

fn dfadd2_vf_vf(x: f32, y: f32) -> F2 {
    let s = x + y;
    let v = s - x;
    f2(s, (x - (s - v)) + (y - v))
}

fn dfadd_vf2_vf(x: F2, y: f32) -> F2 {
    let s = x.x + y;
    f2(s, ((x.x - s) + y) + x.y)
}

fn dfadd2_vf2_vf(x: F2, y: f32) -> F2 {
    let s = x.x + y;
    let v = s - x.x;
    let t = (x.x - (s - v)) + (y - v);
    f2(s, t + x.y)
}

fn dfadd_vf_vf2(x: f32, y: F2) -> F2 {
    let s = x + y.x;
    f2(s, ((x - s) + y.x) + y.y)
}

fn dfadd_vf2_vf2(x: F2, y: F2) -> F2 {
    let s = x.x + y.x;
    f2(s, (((x.x - s) + y.x) + x.y) + y.y)
}

fn dfadd2_vf2_vf2(x: F2, y: F2) -> F2 {
    let s = x.x + y.x;
    let v = s - x.x;
    let t = (x.x - (s - v)) + (y.x - v);
    f2(s, t + (x.y + y.y))
}

fn dfdiv(n: F2, d: F2) -> F2 {
    let t = 1.0_f32 / d.x;
    let s = n.x * t;
    let u = fmapn(t, n.x, s);
    let v = fmanp(d.y, t, fmanp(d.x, t, 1.0));
    f2(s, s.mul_add(v, n.y.mul_add(t, u)))
}

fn dfmul_vf_vf(x: f32, y: f32) -> F2 {
    let s = x * y;
    f2(s, fmapn(x, y, s))
}

fn dfsqu(x: F2) -> F2 {
    let s = x.x * x.x;
    f2(s, (x.x + x.x).mul_add(x.y, fmapn(x.x, x.x, s)))
}

fn dfmul_vf2_vf2(x: F2, y: F2) -> F2 {
    let s = x.x * y.x;
    f2(s, x.x.mul_add(y.y, x.y.mul_add(y.x, fmapn(x.x, y.x, s))))
}

/// `dfmul_vf_vf2_vf2`: the product collapsed to one `f32`.
fn dfmul_to_f(x: F2, y: F2) -> f32 {
    x.x.mul_add(y.x, x.y.mul_add(y.x, x.x * y.y))
}

fn dfmul_vf2_vf(x: F2, y: f32) -> F2 {
    let s = x.x * y;
    f2(s, x.y.mul_add(y, fmapn(x.x, y, s)))
}

fn dfrec_vf2(d: F2) -> F2 {
    let s = 1.0_f32 / d.x;
    f2(s, s * fmanp(d.y, s, fmanp(d.x, s, 1.0)))
}

// --- exp / tanh ---

/// `xexpf` = `Sleef_expf4_u10`.
pub fn expf(d: f32) -> f32 {
    let q = rint_i(d * R_LN2F);
    let qf = q as f32;
    let s = mla(qf, -L2UF, d);
    let s = mla(qf, -L2LF, s);
    let mut u = 0.000_198_527_617_612_853_646_278_381_f64 as f32;
    u = mla(u, s, 0.001_393_043_552_525_341_510_772_71_f64 as f32);
    u = mla(u, s, 0.008_333_360_776_305_198_669_433_59_f64 as f32);
    u = mla(u, s, 0.041_666_485_369_205_474_853_515_6_f64 as f32);
    u = mla(u, s, 0.166_666_671_633_720_397_949_219_f64 as f32);
    u = mla(u, s, 0.5_f64 as f32);
    u = 1.0_f32 + mla(s * s, u, s);
    u = ldexp2(u, q);
    u = and_not(d < -104.0, u);
    if 100.0 < d { f32::INFINITY } else { u }
}

fn expk2f(d: F2) -> F2 {
    let u = (d.x + d.y) * R_LN2F;
    let q = rint_i(u);
    let qf = q as f32;
    let s = dfadd2_vf2_vf(d, qf * -L2UF);
    let s = dfadd2_vf2_vf(s, qf * -L2LF);

    let mut u = 0.198_096_022_4e-3_f32;
    u = mla(u, s.x, 0.139_425_648_4e-2_f32);
    u = mla(u, s.x, 0.833_345_670_3e-2_f32);
    u = mla(u, s.x, 0.416_663_736_1e-1_f32);

    let t = dfadd2_vf2_vf(
        dfmul_vf2_vf(s, u),
        0.166_666_659_414_234_244_790_680_580_464e+0_f32,
    );
    let t = dfadd2_vf2_vf(dfmul_vf2_vf2(s, t), 0.5);
    let t = dfadd2_vf2_vf2(s, dfmul_vf2_vf2(dfsqu(s), t));
    let t = dfadd_vf_vf2(1.0, t);
    let t = f2(ldexp2(t.x, q), ldexp2(t.y, q));
    let below = d.x < -104.0;
    f2(and_not(below, t.x), and_not(below, t.y))
}

/// `xtanhf` = `Sleef_tanhf4_u10`.
pub fn tanhf(x: f32) -> f32 {
    let y = x.abs();
    let d = expk2f(f2(y, 0.0));
    let e = dfrec_vf2(d);
    let d = dfdiv(dfadd_vf2_vf2(d, dfneg(e)), dfadd_vf2_vf2(d, e));
    let mut y = d.x + d.y;
    if x.abs() > 8.664_339_742_f32 || y.is_nan() {
        y = 1.0;
    }
    let y = mulsign(y, x);
    or_all_bits(x.is_nan(), y)
}

// --- pow ---

fn logkf(d: f32) -> F2 {
    let o = d < FLT_MIN;
    let two64 = (1_u64 << 32) as f32 * (1_u64 << 32) as f32;
    let d = if o { d * two64 } else { d };
    let mut e = ilogb2k(d * (1.0_f32 / 0.75_f32));
    let m = ldexp3(d, -e);
    if o {
        e -= 64;
    }
    let x = dfdiv(dfadd2_vf_vf(-1.0, m), dfadd2_vf_vf(1.0, m));
    let x2 = dfsqu(x);
    let mut t = 0.240_320_354_700_088_500_976_562_f64 as f32;
    t = mla(t, x2.x, 0.285_112_679_004_669_189_453_125_f64 as f32);
    t = mla(t, x2.x, 0.400_007_992_982_864_379_882_812_f64 as f32);
    let c = f2(
        0.666_666_626_930_236_816_406_25_f32,
        3.691_838_612_596_143_320_843_11e-09_f32,
    );
    let s = dfmul_vf2_vf(
        f2(
            0.693_147_182_464_599_609_38_f32,
            -1.904_654_323_148_236_017e-09_f32,
        ),
        e as f32,
    );
    let s = dfadd_vf2_vf2(s, dfscale(x, 2.0));
    dfadd_vf2_vf2(
        s,
        dfmul_vf2_vf2(dfmul_vf2_vf2(x2, x), dfadd2_vf2_vf2(dfmul_vf2_vf(x2, t), c)),
    )
}

fn expkf(d: F2) -> f32 {
    let u = (d.x + d.y) * R_LN2F;
    let q = rint_i(u);
    let qf = q as f32;
    let s = dfadd2_vf2_vf(d, qf * -L2UF);
    let s = dfadd2_vf2_vf(s, qf * -L2LF);
    let s = dfnormalize(s);

    let mut u = 0.001_363_246_468_827_128_410_339_36_f32;
    u = mla(u, s.x, 0.008_365_969_173_610_210_418_701_17_f32);
    u = mla(u, s.x, 0.041_671_082_377_433_776_855_468_8_f32);
    u = mla(u, s.x, 0.166_665_524_244_308_471_679_688_f32);
    u = mla(u, s.x, 0.499_999_850_988_388_061_523_438_f32);

    let t = dfadd_vf2_vf2(s, dfmul_vf2_vf(dfsqu(s), u));
    let t = dfadd_vf_vf2(1.0, t);
    let u = ldexp(t.x + t.y, q);
    and_not(d.x < -104.0, u)
}

/// `xpowf` = `Sleef_powf4_u10`.
pub fn powf(x: f32, y: f32) -> f32 {
    let y_trunc = y.trunc();
    let yisint = y_trunc == y || y.abs() > (1_i32 << 24) as f32;
    let yisodd = ((y as i32) & 1) == 1 && yisint && y.abs() < (1_i32 << 24) as f32;

    let mut result = expkf(dfmul_vf2_vf(logkf(x.abs()), y));
    if result.is_nan() {
        result = f32::INFINITY;
    }
    let sign = if x > 0.0 {
        1.0
    } else if yisint {
        if yisodd { -1.0 } else { 1.0 }
    } else {
        f32::NAN
    };
    result *= sign;

    let efx = mulsign(x.abs() - 1.0, y);
    if y.is_infinite() {
        result = and_not(efx < 0.0, if efx == 0.0 { 1.0 } else { f32::INFINITY });
    }
    if x.is_infinite() || x == 0.0 {
        let mag = if (y.to_bits() >> 31 == 1) ^ (x == 0.0) {
            0.0
        } else {
            f32::INFINITY
        };
        result = mulsign(mag, if yisodd { x } else { 1.0 });
    }
    result = or_all_bits(x.is_nan() || y.is_nan(), result);
    if y == 0.0 || x == 1.0 { 1.0 } else { result }
}

// --- sin / cos ---

fn rempisubf(x: f32) -> (f32, i32) {
    let y = rint(x * 4.0);
    let vi = (y - rint(x) * 4.0) as i32;
    (x - y * 0.25, vi)
}

/// `rempif`: Payne-Hanek reduction of `a` by `pi/2`.
fn rempif(a: f32) -> (F2, i32) {
    let mut ex = ilogb2k(a) - 25;
    let q = if ex > 90 - 25 { -64 } else { 0 };
    let a = ldexp3(a, q);
    ex = if ex < 0 { 0 } else { ex };
    let ex = (ex << 2) as usize;

    let mut x = dfmul_vf_vf(a, REMPITAB_SP[ex]);
    let (d, mut q) = rempisubf(x.x);
    x.x = d;
    x = dfnormalize(x);
    let y = dfmul_vf_vf(a, REMPITAB_SP[ex + 1]);
    x = dfadd2_vf2_vf2(x, y);
    let (d, qi) = rempisubf(x.x);
    q += qi;
    x.x = d;
    x = dfnormalize(x);
    let y = dfmul_vf2_vf(f2(REMPITAB_SP[ex + 2], REMPITAB_SP[ex + 3]), a);
    x = dfadd2_vf2_vf2(x, y);
    x = dfnormalize(x);
    x = dfmul_vf2_vf2(
        x,
        f2(
            3.141_592_741_012_573_242_2_f32 * 2.0,
            -8.742_277_657_347_585_773_1e-08_f32 * 2.0,
        ),
    );
    if a.abs() < 0.7 {
        x = f2(a, 0.0);
    }
    (x, q)
}

/// Shared polynomial tail of `xsinf_u1` / `xcosf_u1`.
fn sincos_poly(s: F2) -> f32 {
    let t = s;
    let s = dfsqu(s);
    let mut u = 2.608_315_980_978_659_354_150_3e-06_f32;
    u = mla(u, s.x, -0.000_198_106_907_191_686_332_225_8_f32);
    u = mla(u, s.x, 0.008_333_078_585_565_090_179_443_36_f32);
    let x = dfadd_vf_vf2(
        1.0,
        dfmul_vf2_vf2(
            dfadd_vf_vf(-0.166_666_597_127_914_428_710_938_f32, u * s.x),
            s,
        ),
    );
    dfmul_to_f(t, x)
}

/// `(pi/2) * -0.5` halves of the large-argument correction, narrowed like
/// `vcast_vf_f(3.1415927410125732422f*-0.5)` (a `double` product).
fn half_pi_parts() -> (f32, f32) {
    (
        (f64::from(3.141_592_741_012_573_242_2_f32) * -0.5) as f32,
        (f64::from(-8.742_277_657_347_585_773_1e-08_f32) * -0.5) as f32,
    )
}

fn sin_lane_small(d: f32) -> (F2, i32) {
    let u = rint(d * M_1_PI_F);
    let q = rint_i(u);
    let v = mla(u, -PI_A2F, d);
    let s = dfadd2_vf_vf(v, u * -PI_B2F);
    (dfadd_vf2_vf(s, u * -PI_C2F), q)
}

fn sin_lane_large(d: f32) -> (F2, i32) {
    let (df, di) = rempif(d);
    let mut q = di & 3;
    q = q + q + if df.x > 0.0 { 2 } else { 1 };
    q >>= 2;
    let (hi, lo) = half_pi_parts();
    let x = dfadd2_vf2_vf2(df, f2(mulsign(hi, df.x), mulsign(lo, df.x)));
    let df = if (di & 1) == 1 { x } else { df };
    let mut s = dfnormalize(df);
    s.x = or_all_bits(d.is_infinite() || d.is_nan(), s.x);
    (s, q)
}

fn cos_lane_small(d: f32) -> (F2, i32) {
    let dq = mla(rint(mla(d, M_1_PI_F, -0.5)), 2.0, 1.0);
    let q = rint_i(dq);
    let s = dfadd2_vf_vf(d, dq * (-PI_A2F * 0.5));
    let s = dfadd2_vf2_vf(s, dq * (-PI_B2F * 0.5));
    (dfadd2_vf2_vf(s, dq * (-PI_C2F * 0.5)), q)
}

fn cos_lane_large(d: f32) -> (F2, i32) {
    let (df, di) = rempif(d);
    let mut q = di & 3;
    q = q + q + if df.x > 0.0 { 8 } else { 7 };
    q >>= 1;
    let y = if df.x > 0.0 { 0.0 } else { -1.0 };
    let (hi, lo) = half_pi_parts();
    let x = dfadd2_vf2_vf2(df, f2(mulsign(hi, y), mulsign(lo, y)));
    let df = if (di & 1) == 0 { x } else { df };
    let mut s = dfnormalize(df);
    s.x = or_all_bits(d.is_infinite() || d.is_nan(), s.x);
    (s, q)
}

/// Whether a 4-lane vector takes the small-argument path: SLEEF tests all
/// lanes at once (`vtestallones_i_vo32`), so one large lane moves the whole
/// vector to the Payne-Hanek path.
fn all_small(d: &[f32; 4]) -> bool {
    d.iter().all(|v| v.abs() < TRIGRANGEMAX2F)
}

/// `xsinf_u1` = `Sleef_sinf4_u10` on one 4-lane vector.
pub fn sin4(d: [f32; 4]) -> [f32; 4] {
    let small = all_small(&d);
    d.map(|d| {
        let (s, q) = if small {
            sin_lane_small(d)
        } else {
            sin_lane_large(d)
        };
        let u = sincos_poly(s);
        let u = if (q & 1) == 1 { -u } else { u };
        if d == 0.0 && d.is_sign_negative() {
            d
        } else {
            u
        }
    })
}

/// `xcosf_u1` = `Sleef_cosf4_u10` on one 4-lane vector.
pub fn cos4(d: [f32; 4]) -> [f32; 4] {
    let small = all_small(&d);
    d.map(|d| {
        let (s, q) = if small {
            cos_lane_small(d)
        } else {
            cos_lane_large(d)
        };
        let u = sincos_poly(s);
        if (q & 2) == 0 { -u } else { u }
    })
}
