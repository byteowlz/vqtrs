//! Real forward FFT reproducing `NumPy` 2.4's `np.fft.rfft` on `float32` input
//! bit for bit. Measured: `NumPy`'s result equals the `float64` transform of
//! the (exactly widened) input rounded to `complex64`, so this is a port of
//! pocketfft's `rfftp` (`pocketfft_hdronly.h`, commit 33ae5dc9, as vendored
//! by `NumPy`) in `double` for power-of-two lengths -- factor ordering,
//! `sincos_2pibyn` twiddles, the `radf4` / `radf2` butterflies -- with a
//! final narrowing to `f32`. See `pocketfft.LICENSE` for the BSD 3-clause notice.

#![expect(
    clippy::similar_names,
    reason = "CC and CH indexers retain the pocketfft source names"
)]
//!
//! Where the C++ has `a * b +/- c * d` in one expression, the reference
//! build (clang, `-ffp-contract=on`) fuses the first product; the port does
//! the same with [`f32::mul_add`] / [`f64::mul_add`].

/// One factor of the transform with its twiddles (`(ip - 1) * (ido - 1)`).
struct Factor {
    ip: usize,
    tw: Vec<f64>,
}

/// A planned real FFT of a power-of-two length.
pub struct Rfft {
    length: usize,
    factors: Vec<Factor>,
}

/// `sincos_2pibyn<float>`: twiddles computed in `double`, from a split
/// pair of tables, narrowed to `float`.
struct SinCos {
    n: usize,
    mask: usize,
    shift: usize,
    v1: Vec<(f64, f64)>,
    v2: Vec<(f64, f64)>,
}

impl SinCos {
    fn calc(x: usize, n: usize, ang: f64) -> (f64, f64) {
        let x = x << 3;
        let f = |v: usize| v as f64 * ang;
        if x < 4 * n {
            if x < 2 * n {
                if x < n {
                    return (f(x).cos(), f(x).sin());
                }
                return (f(2 * n - x).sin(), f(2 * n - x).cos());
            }
            let x = x - 2 * n;
            if x < n {
                return (-f(x).sin(), f(x).cos());
            }
            return (-f(2 * n - x).cos(), f(2 * n - x).sin());
        }
        let x = 8 * n - x;
        if x < 2 * n {
            if x < n {
                return (f(x).cos(), -f(x).sin());
            }
            return (f(2 * n - x).sin(), -f(2 * n - x).cos());
        }
        let x = x - 2 * n;
        if x < n {
            return (-f(x).sin(), -f(x).cos());
        }
        (-f(2 * n - x).cos(), -f(2 * n - x).sin())
    }

    fn new(n: usize) -> Self {
        // `0.25L * pi / n` with `long double == double` on aarch64 macOS.
        let ang = 0.25 * std::f64::consts::PI / n as f64;
        let nval = n / 2 + 1;
        let mut shift = 1;
        while (1 << shift) * (1 << shift) < nval {
            shift += 1;
        }
        let mask = (1 << shift) - 1;
        let v1 = (0..=mask)
            .map(|i| {
                if i == 0 {
                    (1.0, 0.0)
                } else {
                    Self::calc(i, n, ang)
                }
            })
            .collect();
        let v2 = (0..nval.div_ceil(mask + 1))
            .map(|i| {
                if i == 0 {
                    (1.0, 0.0)
                } else {
                    Self::calc(i * (mask + 1), n, ang)
                }
            })
            .collect();
        Self {
            n,
            mask,
            shift,
            v1,
            v2,
        }
    }

    fn get(&self, idx: usize) -> (f64, f64) {
        let (idx, conj) = if 2 * idx <= self.n {
            (idx, false)
        } else {
            (self.n - idx, true)
        };
        let (a, b) = (self.v1[idx & self.mask], self.v2[idx >> self.shift]);
        let re = a.0.mul_add(b.0, -(a.1 * b.1));
        let im = a.0.mul_add(b.1, a.1 * b.0);
        (re, if conj { -im } else { im })
    }
}

/// `MULPM`: `(c*e + d*f, c*f - d*e)`, first product fused.
fn mulpm(c: f64, d: f64, e: f64, f: f64) -> (f64, f64) {
    (c.mul_add(e, d * f), c.mul_add(f, -(d * e)))
}

impl Rfft {
    /// Plan a transform of `length`, a power of two.
    pub fn new(length: usize) -> Self {
        // rfftp::factorize: 4s first, then one 2 moved to the front.
        let mut ips = Vec::new();
        let mut len = length;
        while len.is_multiple_of(4) && len > 1 {
            ips.push(4);
            len >>= 2;
        }
        if len.is_multiple_of(2) {
            ips.push(2);
            let last = ips.len() - 1;
            ips.swap(0, last);
        }
        // rfftp::comp_twiddle.
        let twid = SinCos::new(length);
        let mut l1 = 1;
        let nf = ips.len();
        let factors = ips
            .iter()
            .enumerate()
            .map(|(k, &ip)| {
                let ido = length / (l1 * ip);
                let mut tw = Vec::new();
                if k + 1 < nf {
                    tw = vec![0.0; (ip - 1) * (ido - 1)];
                    for j in 1..ip {
                        for i in 1..=(ido - 1) / 2 {
                            let (re, im) = twid.get(j * l1 * i);
                            tw[(j - 1) * (ido - 1) + 2 * i - 2] = re;
                            tw[(j - 1) * (ido - 1) + 2 * i - 1] = im;
                        }
                    }
                }
                l1 *= ip;
                Factor { ip, tw }
            })
            .collect();
        Self { length, factors }
    }

    /// Forward transform of `c` (length `self.length`) in place, FFTPACK
    /// half-complex order `r0, r1, i1, ..., r(n/2)`.
    pub fn forward(&self, c: &mut [f64]) {
        let n = self.length;
        let mut a = c.to_vec();
        let mut b = vec![0.0_f64; n];
        let mut l1 = n;
        for factor in self.factors.iter().rev() {
            let ido = n / l1;
            l1 /= factor.ip;
            if factor.ip == 4 {
                radf4(ido, l1, &a, &mut b, &factor.tw);
            } else {
                radf2(ido, l1, &a, &mut b, &factor.tw);
            }
            std::mem::swap(&mut a, &mut b);
        }
        c.copy_from_slice(&a);
    }

    /// `np.fft.rfft(frame, n)` for a real frame (zero-padded to `n`):
    /// `n / 2 + 1` complex bins.
    pub fn rfft(&self, frame: &[f32]) -> Vec<(f32, f32)> {
        let n = self.length;
        let mut buf = vec![0.0_f64; n];
        for (b, &x) in buf.iter_mut().zip(frame) {
            *b = f64::from(x);
        }
        self.forward(&mut buf);
        let mut out = Vec::with_capacity(n / 2 + 1);
        out.push((buf[0] as f32, 0.0));
        for k in 1..n / 2 {
            out.push((buf[2 * k - 1] as f32, buf[2 * k] as f32));
        }
        out.push((buf[n - 1] as f32, 0.0));
        out
    }
}

fn radf2(ido: usize, l1: usize, cc: &[f64], ch: &mut [f64], wa: &[f64]) {
    let cc_at = |a: usize, b: usize, c: usize| cc[a + ido * (b + l1 * c)];
    let ch_at = |a: usize, b: usize, c: usize| a + ido * (b + 2 * c);
    for k in 0..l1 {
        ch[ch_at(0, 0, k)] = cc_at(0, k, 0) + cc_at(0, k, 1);
        ch[ch_at(ido - 1, 1, k)] = cc_at(0, k, 0) - cc_at(0, k, 1);
    }
    if ido.is_multiple_of(2) {
        for k in 0..l1 {
            ch[ch_at(0, 1, k)] = -cc_at(ido - 1, k, 1);
            ch[ch_at(ido - 1, 0, k)] = cc_at(ido - 1, k, 0);
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let (tr2, ti2) = mulpm(wa[i - 2], wa[i - 1], cc_at(i - 1, k, 1), cc_at(i, k, 1));
            ch[ch_at(i - 1, 0, k)] = cc_at(i - 1, k, 0) + tr2;
            ch[ch_at(ic - 1, 1, k)] = cc_at(i - 1, k, 0) - tr2;
            ch[ch_at(i, 0, k)] = ti2 + cc_at(i, k, 0);
            ch[ch_at(ic, 1, k)] = ti2 - cc_at(i, k, 0);
        }
    }
}

fn radf4(ido: usize, l1: usize, cc: &[f64], ch: &mut [f64], wa: &[f64]) {
    let hsqt2 = std::f64::consts::FRAC_1_SQRT_2;
    let cc_at = |a: usize, b: usize, c: usize| cc[a + ido * (b + l1 * c)];
    let ch_at = |a: usize, b: usize, c: usize| a + ido * (b + 4 * c);
    let wa_at = |x: usize, i: usize| wa[i + x * (ido - 1)];
    for k in 0..l1 {
        let (tr1, ch02) = (
            cc_at(0, k, 3) + cc_at(0, k, 1),
            cc_at(0, k, 3) - cc_at(0, k, 1),
        );
        let (tr2, ch11) = (
            cc_at(0, k, 0) + cc_at(0, k, 2),
            cc_at(0, k, 0) - cc_at(0, k, 2),
        );
        ch[ch_at(0, 2, k)] = ch02;
        ch[ch_at(ido - 1, 1, k)] = ch11;
        ch[ch_at(0, 0, k)] = tr2 + tr1;
        ch[ch_at(ido - 1, 3, k)] = tr2 - tr1;
    }
    if ido.is_multiple_of(2) {
        for k in 0..l1 {
            let ti1 = -hsqt2 * (cc_at(ido - 1, k, 1) + cc_at(ido - 1, k, 3));
            let tr1 = hsqt2 * (cc_at(ido - 1, k, 1) - cc_at(ido - 1, k, 3));
            ch[ch_at(ido - 1, 0, k)] = cc_at(ido - 1, k, 0) + tr1;
            ch[ch_at(ido - 1, 2, k)] = cc_at(ido - 1, k, 0) - tr1;
            ch[ch_at(0, 3, k)] = ti1 + cc_at(ido - 1, k, 2);
            ch[ch_at(0, 1, k)] = ti1 - cc_at(ido - 1, k, 2);
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let (cr2, ci2) = mulpm(
                wa_at(0, i - 2),
                wa_at(0, i - 1),
                cc_at(i - 1, k, 1),
                cc_at(i, k, 1),
            );
            let (cr3, ci3) = mulpm(
                wa_at(1, i - 2),
                wa_at(1, i - 1),
                cc_at(i - 1, k, 2),
                cc_at(i, k, 2),
            );
            let (cr4, ci4) = mulpm(
                wa_at(2, i - 2),
                wa_at(2, i - 1),
                cc_at(i - 1, k, 3),
                cc_at(i, k, 3),
            );
            let (tr1, tr4) = (cr4 + cr2, cr4 - cr2);
            let (ti1, ti4) = (ci2 + ci4, ci2 - ci4);
            let (tr2, tr3) = (cc_at(i - 1, k, 0) + cr3, cc_at(i - 1, k, 0) - cr3);
            let (ti2, ti3) = (cc_at(i, k, 0) + ci3, cc_at(i, k, 0) - ci3);
            ch[ch_at(i - 1, 0, k)] = tr2 + tr1;
            ch[ch_at(ic - 1, 3, k)] = tr2 - tr1;
            ch[ch_at(i, 0, k)] = ti1 + ti2;
            ch[ch_at(ic, 3, k)] = ti1 - ti2;
            ch[ch_at(i - 1, 2, k)] = tr3 + ti4;
            ch[ch_at(ic - 1, 1, k)] = tr3 - ti4;
            ch[ch_at(i, 2, k)] = tr4 + ti3;
            ch[ch_at(ic, 1, k)] = tr4 - ti3;
        }
    }
}

/// `NumPy`'s SIMD `np.abs` for `complex64`: `larger * sqrt(fma(r, r, 1))`
/// with `r = smaller / larger` (0 when `larger == 0`).
pub fn cabs(re: f32, im: f32) -> f32 {
    let (re, im) = (re.abs(), im.abs());
    if re.is_infinite() || im.is_infinite() {
        return f32::INFINITY;
    }
    if re.is_nan() || im.is_nan() {
        return f32::NAN;
    }
    let (larger, smaller) = if re >= im { (re, im) } else { (im, re) };
    let ratio = if larger == 0.0 { 0.0 } else { smaller / larger };
    ratio.mul_add(ratio, 1.0).sqrt() * larger
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "these elementary transforms have exactly representable outputs"
    )]
    fn impulse_and_constant_transforms() {
        let plan = Rfft::new(512);
        let mut impulse = [0.0_f32; 320];
        impulse[0] = 1.0;
        let spectrum = plan.rfft(&impulse);
        assert_eq!(spectrum.len(), 257);
        for (re, im) in spectrum {
            assert_eq!(re, 1.0);
            assert_eq!(im, 0.0);
        }
        let spectrum = plan.rfft(&[1.0; 512]);
        assert_eq!(spectrum[0], (512.0, 0.0));
        for (re, im) in &spectrum[1..] {
            assert_eq!(*re, 0.0);
            assert_eq!(*im, 0.0);
        }
        assert_eq!(cabs(0.0, -0.0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(cabs(3.0, 4.0), 5.0);
        assert!(cabs(f32::NAN, 1.0).is_nan());
        assert!(cabs(f32::INFINITY, f32::NAN).is_infinite());
    }

    /// Against `np.fft.rfft` / `np.abs` outputs dumped by the parity scripts.
    #[test]
    #[ignore = "needs NumPy reference dumps"]
    fn matches_numpy() {
        let dir = std::path::PathBuf::from(std::env::var("EG2_MEL_DIR").expect("EG2_MEL_DIR"));
        let read = |name: &str| -> Vec<f32> {
            std::fs::read(dir.join(name))
                .unwrap()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let frames = read("fft_frames_f32.bin");
        let reference = read("fft_out_c64.bin");
        let abs_ref = read("fft_abs_f32.bin");
        let plan = Rfft::new(512);
        let (mut same, mut total, mut abs_same) = (0, 0, 0);
        for (f, frame) in frames.chunks_exact(320).enumerate() {
            for (k, (re, im)) in plan.rfft(frame).into_iter().enumerate() {
                let at = (f * 257 + k) * 2;
                same += usize::from(re.to_bits() == reference[at].to_bits());
                same += usize::from(im.to_bits() == reference[at + 1].to_bits());
                abs_same += usize::from(cabs(re, im).to_bits() == abs_ref[f * 257 + k].to_bits());
                total += 2;
            }
        }
        println!(
            "rfft identical {same}/{total}, abs identical {abs_same}/{}",
            total / 2
        );
        assert_eq!(same, total);
        assert_eq!(abs_same, total / 2);
    }
}
