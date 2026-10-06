//! A JPEG decoder whose output is bit-identical to libjpeg-turbo 3.1 with
//! its default settings, as used by Pillow (`islow` IDCT, fancy upsampling,
//! no merged upsampler, YCbCr->RGB via the `jdcolor.c` fixed-point tables).
//!
//! Entropy decoding (baseline and progressive Huffman) is exact in any
//! correct decoder; everything after it is ported from libjpeg-turbo:
//! `jidctint.c`, `jdsample.c` (with `jdmainct.c`'s edge-row replication),
//! `jdcolor.c` and the range-limit table of `jdmaster.c`. libjpeg-turbo is
//! (c) the libjpeg-turbo project and the Independent JPEG Group, under the
//! IJG and BSD-style licenses.
//!
//! Only 8-bit Huffman JPEGs with 1 (grey) or 3 (YCbCr/RGB) components are
//! handled; anything else yields `None` so the caller can fall back.

/// Zig-zag index -> natural (row-major) index, padded like libjpeg's
/// `jpeg_natural_order` so corrupt run lengths land on index 63.
const NATURAL: [usize; 80] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63,
];

/// A decoded image: `channels` is 1 (grey) or 3 (RGB), interleaved.
#[derive(Debug, Clone)]
pub struct Decoded {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<u8>,
}

#[derive(Clone, Default)]
struct Huffman {
    maxcode: [i32; 18],
    valoffset: [i32; 17],
    values: Vec<u8>,
}

impl Huffman {
    /// Canonical code construction (ITU T.81, F.2.2.3 / `jpeg_make_d_derived_tbl`).
    fn new(counts: &[u8; 16], values: Vec<u8>) -> Self {
        let mut t = Self {
            values,
            ..Self::default()
        };
        let mut code = 0_i32;
        let mut p = 0_i32;
        for l in 1..=16 {
            let n = i32::from(counts[l - 1]);
            if n > 0 {
                t.valoffset[l] = p - code;
                p += n;
                code += n;
                t.maxcode[l] = code - 1;
            } else {
                t.maxcode[l] = -1;
            }
            code <<= 1;
        }
        t.maxcode[17] = i32::MAX;
        t
    }
}

/// Entropy-coded-segment bit reader: unstuffs `FF 00`, stops at markers and
/// then supplies zeros (libjpeg's behaviour on premature markers).
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u64,
    count: u32,
    hit_marker: bool,
}

impl<'a> Bits<'a> {
    const fn new(data: &'a [u8], pos: usize) -> Self {
        Self {
            data,
            pos,
            acc: 0,
            count: 0,
            hit_marker: false,
        }
    }

    fn fill(&mut self) {
        while self.count <= 56 {
            let mut byte = 0_u8;
            if !self.hit_marker && self.pos < self.data.len() {
                byte = self.data[self.pos];
                if byte == 0xFF {
                    if self.data.get(self.pos + 1) == Some(&0x00) {
                        self.pos += 2;
                    } else {
                        self.hit_marker = true;
                        byte = 0;
                    }
                } else {
                    self.pos += 1;
                }
            }
            self.acc |= u64::from(byte) << (56 - self.count);
            self.count += 8;
        }
    }

    fn bits(&mut self, n: u32) -> i32 {
        if n == 0 {
            return 0;
        }
        if self.count < n {
            self.fill();
        }
        let v = (self.acc >> (64 - n)) as i32;
        self.acc <<= n;
        self.count -= n;
        v
    }

    fn bit(&mut self) -> i32 {
        self.bits(1)
    }

    fn decode(&mut self, t: &Huffman) -> u8 {
        let mut code = self.bit();
        let mut l = 1;
        while l < 17 && code > t.maxcode[l] {
            code = (code << 1) | self.bit();
            l += 1;
        }
        if l > 16 {
            return 0; // corrupt: libjpeg warns and uses 0
        }
        t.values
            .get((code + t.valoffset[l]) as usize)
            .copied()
            .unwrap_or(0)
    }

    /// `HUFF_EXTEND(GET_BITS(s), s)`.
    fn receive_extend(&mut self, s: u32) -> i32 {
        if s == 0 {
            return 0;
        }
        let x = self.bits(s);
        if x < (1 << (s - 1)) {
            x + ((-1) << s) + 1
        } else {
            x
        }
    }

    /// Skip to the next byte boundary and past an `RSTn` marker.
    fn restart(&mut self) {
        self.acc = 0;
        self.count = 0;
        self.hit_marker = false;
        while self.pos + 1 < self.data.len() {
            if self.data[self.pos] == 0xFF && (0xD0..=0xD7).contains(&self.data[self.pos + 1]) {
                self.pos += 2;
                return;
            }
            self.pos += 1;
        }
    }
}

struct Component {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    /// `downsampled_width` / `downsampled_height`.
    width: usize,
    height: usize,
    /// Blocks covering the component (`width_in_blocks`, `height_in_blocks`).
    bw: usize,
    bh: usize,
    /// Coefficient-buffer stride in blocks (whole MCUs).
    stride: usize,
    coefs: Vec<i16>,
    dc_table: usize,
    ac_table: usize,
}

impl Component {
    fn block(&mut self, bx: usize, by: usize) -> &mut [i16] {
        let at = (by * self.stride + bx) * 64;
        &mut self.coefs[at..at + 64]
    }
}

struct Frame {
    width: usize,
    height: usize,
    progressive: bool,
    comps: Vec<Component>,
    hmax: usize,
    vmax: usize,
    mcus_x: usize,
    mcus_y: usize,
}

struct Scan {
    comps: Vec<usize>,
    ss: usize,
    se: usize,
    ah: u32,
    al: u32,
}

/// Decoder state carried across markers.
struct Decoder<'a> {
    data: &'a [u8],
    qt: [[u16; 64]; 4],
    dc: [Huffman; 4],
    ac: [Huffman; 4],
    restart_interval: usize,
    frame: Option<Frame>,
    jfif: bool,
    adobe_transform: Option<u8>,
    eobrun: u32,
}

fn be16(d: &[u8], at: usize) -> Option<usize> {
    Some(usize::from(*d.get(at)?) << 8 | usize::from(*d.get(at + 1)?))
}

/// Decode `bytes`; `None` if the file uses features outside the supported
/// subset (or is not a JPEG).
pub fn decode(bytes: &[u8]) -> Option<Decoded> {
    let mut dec = Decoder {
        data: bytes,
        qt: [[0; 64]; 4],
        dc: Default::default(),
        ac: Default::default(),
        restart_interval: 0,
        frame: None,
        jfif: false,
        adobe_transform: None,
        eobrun: 0,
    };
    if bytes.get(..2)? != [0xFF, 0xD8] {
        return None;
    }
    let mut pos = 2;
    loop {
        while *bytes.get(pos)? != 0xFF {
            pos += 1;
        }
        while *bytes.get(pos)? == 0xFF {
            pos += 1;
        }
        let marker = *bytes.get(pos)?;
        pos += 1;
        if marker == 0xD9 {
            break;
        }
        if (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            continue;
        }
        let len = be16(bytes, pos)?;
        let body = bytes.get(pos + 2..pos + len)?;
        match marker {
            0xC0..=0xC2 => dec.frame = Some(parse_frame(body, marker == 0xC2)?),
            0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => return None,
            0xC4 => dec.parse_dht(body)?,
            0xDB => dec.parse_dqt(body)?,
            0xDD => dec.restart_interval = be16(body, 0)?,
            0xE0 => dec.jfif |= body.starts_with(b"JFIF\0"),
            0xEE if body.starts_with(b"Adobe") => dec.adobe_transform = body.get(11).copied(),
            0xDA => {
                pos = dec.scan(body, pos + len)?;
                continue;
            }
            _ => {}
        }
        pos += len;
    }
    dec.finish()
}

fn parse_frame(body: &[u8], progressive: bool) -> Option<Frame> {
    if *body.first()? != 8 {
        return None; // only 8-bit precision
    }
    let height = be16(body, 1)?;
    let width = be16(body, 3)?;
    let n = usize::from(*body.get(5)?);
    if !(n == 1 || n == 3) || width == 0 || height == 0 {
        return None;
    }
    let mut comps = Vec::with_capacity(n);
    for i in 0..n {
        let c = body.get(6 + i * 3..9 + i * 3)?;
        let (h, v) = (usize::from(c[1] >> 4), usize::from(c[1] & 15));
        if h == 0 || v == 0 || h > 4 || v > 4 || c[2] > 3 {
            return None;
        }
        comps.push(Component {
            id: c[0],
            h,
            v,
            tq: usize::from(c[2]),
            width: 0,
            height: 0,
            bw: 0,
            bh: 0,
            stride: 0,
            coefs: Vec::new(),
            dc_table: 0,
            ac_table: 0,
        });
    }
    let hmax = comps.iter().map(|c| c.h).max()?;
    let vmax = comps.iter().map(|c| c.v).max()?;
    let mcus_x = width.div_ceil(8 * hmax);
    let mcus_y = height.div_ceil(8 * vmax);
    for c in &mut comps {
        c.width = (width * c.h).div_ceil(hmax);
        c.height = (height * c.v).div_ceil(vmax);
        c.bw = c.width.div_ceil(8);
        c.bh = c.height.div_ceil(8);
        c.stride = mcus_x * c.h;
        c.coefs = vec![0; c.stride * mcus_y * c.v * 64];
    }
    Some(Frame {
        width,
        height,
        progressive,
        comps,
        hmax,
        vmax,
        mcus_x,
        mcus_y,
    })
}

impl Decoder<'_> {
    fn parse_dht(&mut self, mut body: &[u8]) -> Option<()> {
        while !body.is_empty() {
            let class_id = body[0];
            let counts: [u8; 16] = body.get(1..17)?.try_into().ok()?;
            let total: usize = counts.iter().map(|&c| usize::from(c)).sum();
            let values = body.get(17..17 + total)?.to_vec();
            let table = Huffman::new(&counts, values);
            let id = usize::from(class_id & 15).min(3);
            if class_id >> 4 == 0 {
                self.dc[id] = table;
            } else {
                self.ac[id] = table;
            }
            body = &body[17 + total..];
        }
        Some(())
    }

    fn parse_dqt(&mut self, mut body: &[u8]) -> Option<()> {
        while !body.is_empty() {
            let (pq, tq) = (body[0] >> 4, usize::from(body[0] & 15).min(3));
            let size = if pq == 0 { 64 } else { 128 };
            let raw = body.get(1..1 + size)?;
            for k in 0..64 {
                let v = if pq == 0 {
                    u16::from(raw[k])
                } else {
                    u16::from(raw[2 * k]) << 8 | u16::from(raw[2 * k + 1])
                };
                self.qt[tq][NATURAL[k]] = v;
            }
            body = &body[1 + size..];
        }
        Some(())
    }

    /// Decode one scan whose entropy-coded data starts at `start`; returns
    /// the offset of the marker that ends it.
    fn scan(&mut self, body: &[u8], start: usize) -> Option<usize> {
        let frame = self.frame.as_mut()?;
        let ns = usize::from(*body.first()?);
        let mut comps = Vec::with_capacity(ns);
        for i in 0..ns {
            let (cs, t) = (*body.get(1 + i * 2)?, *body.get(2 + i * 2)?);
            let ci = frame.comps.iter().position(|c| c.id == cs)?;
            frame.comps[ci].dc_table = usize::from(t >> 4).min(3);
            frame.comps[ci].ac_table = usize::from(t & 15).min(3);
            comps.push(ci);
        }
        let tail = body.get(1 + ns * 2..4 + ns * 2)?;
        let scan = Scan {
            comps,
            ss: usize::from(tail[0]),
            se: usize::from(tail[1]).min(63),
            ah: u32::from(tail[2] >> 4),
            al: u32::from(tail[2] & 15),
        };
        let mut bits = Bits::new(self.data, start);
        self.eobrun = 0;
        self.decode_scan(&scan, &mut bits);
        // The scan ends at the first marker that is not a restart marker.
        let mut pos = bits.pos;
        while pos + 1 < self.data.len() {
            if self.data[pos] == 0xFF
                && self.data[pos + 1] != 0x00
                && !(0xD0..=0xD7).contains(&self.data[pos + 1])
            {
                return Some(pos);
            }
            pos += 1;
        }
        Some(self.data.len())
    }

    fn decode_scan(&mut self, scan: &Scan, bits: &mut Bits<'_>) {
        let Some(frame) = self.frame.as_mut() else {
            return;
        };
        let progressive = frame.progressive;
        let mut dc_pred = vec![0_i32; frame.comps.len()];
        // Units: blocks of a single-component scan, MCUs otherwise.
        let single = scan.comps.len() == 1;
        let (units_x, units_y) = if single {
            let c = &frame.comps[scan.comps[0]];
            (c.bw, c.bh)
        } else {
            (frame.mcus_x, frame.mcus_y)
        };
        let mut done = 0_usize;
        for uy in 0..units_y {
            for ux in 0..units_x {
                if self.restart_interval > 0
                    && done > 0
                    && done.is_multiple_of(self.restart_interval)
                {
                    bits.restart();
                    dc_pred.fill(0);
                    self.eobrun = 0;
                }
                for &ci in &scan.comps {
                    let (h, v) = if single {
                        (1, 1)
                    } else {
                        (frame.comps[ci].h, frame.comps[ci].v)
                    };
                    for by in 0..v {
                        for bx in 0..h {
                            let (x, y) = if single {
                                (ux, uy)
                            } else {
                                (ux * h + bx, uy * v + by)
                            };
                            let comp = &mut frame.comps[ci];
                            let (dct, act) = (comp.dc_table, comp.ac_table);
                            let block = comp.block(x, y);
                            let pred = &mut dc_pred[ci];
                            if !progressive {
                                decode_baseline(bits, block, &self.dc[dct], &self.ac[act], pred);
                            } else if scan.ss == 0 {
                                decode_dc(bits, block, &self.dc[dct], pred, scan.ah, scan.al);
                            } else if scan.ah == 0 {
                                decode_ac_first(bits, block, &self.ac[act], scan, &mut self.eobrun);
                            } else {
                                decode_ac_refine(
                                    bits,
                                    block,
                                    &self.ac[act],
                                    scan,
                                    &mut self.eobrun,
                                );
                            }
                        }
                    }
                }
                done += 1;
            }
        }
    }

    fn finish(self) -> Option<Decoded> {
        let frame = self.frame?;
        let planes: Vec<Vec<u8>> = frame
            .comps
            .iter()
            .map(|c| idct_plane(c, &self.qt[c.tq]))
            .collect();
        let full: Vec<Vec<u8>> = frame
            .comps
            .iter()
            .zip(&planes)
            .map(|(c, p)| upsample(c, p, &frame))
            .collect::<Option<_>>()?;
        let (w, h) = (frame.width, frame.height);
        if full.len() == 1 {
            return Some(Decoded {
                width: w,
                height: h,
                channels: 1,
                data: full[0].clone(),
            });
        }
        let rgb_input = match self.adobe_transform {
            _ if self.jfif => false,
            Some(0) => true,
            Some(_) => false,
            None => {
                frame.comps[0].id == b'R' && frame.comps[1].id == b'G' && frame.comps[2].id == b'B'
            }
        };
        let mut data = vec![0_u8; w * h * 3];
        if rgb_input {
            for (i, px) in data.chunks_exact_mut(3).enumerate() {
                px.copy_from_slice(&[full[0][i], full[1][i], full[2][i]]);
            }
        } else {
            ycc_to_rgb(&full, &mut data);
        }
        Some(Decoded {
            width: w,
            height: h,
            channels: 3,
            data,
        })
    }
}

fn decode_baseline(
    bits: &mut Bits<'_>,
    block: &mut [i16],
    dc: &Huffman,
    ac: &Huffman,
    pred: &mut i32,
) {
    let s = u32::from(bits.decode(dc));
    *pred += bits.receive_extend(s);
    block[0] = *pred as i16;
    let mut k = 1;
    while k < 64 {
        let rs = bits.decode(ac);
        let (r, s) = (usize::from(rs >> 4), u32::from(rs & 15));
        if s != 0 {
            k += r;
            block[NATURAL[k]] = bits.receive_extend(s) as i16;
        } else if r != 15 {
            break;
        } else {
            k += 15;
        }
        k += 1;
    }
}

fn decode_dc(
    bits: &mut Bits<'_>,
    block: &mut [i16],
    dc: &Huffman,
    pred: &mut i32,
    ah: u32,
    al: u32,
) {
    if ah == 0 {
        let s = u32::from(bits.decode(dc));
        *pred += bits.receive_extend(s);
        block[0] = ((*pred as u32) << al) as i16;
    } else if bits.bit() != 0 {
        block[0] |= 1 << al;
    }
}

fn decode_ac_first(
    bits: &mut Bits<'_>,
    block: &mut [i16],
    ac: &Huffman,
    scan: &Scan,
    eobrun: &mut u32,
) {
    if *eobrun > 0 {
        *eobrun -= 1;
        return;
    }
    let mut k = scan.ss;
    while k <= scan.se {
        let rs = bits.decode(ac);
        let (r, s) = (u32::from(rs >> 4), u32::from(rs & 15));
        if s != 0 {
            k += r as usize;
            block[NATURAL[k]] = ((bits.receive_extend(s) as u32) << scan.al) as i16;
        } else if r == 15 {
            k += 15;
        } else {
            *eobrun = (1 << r) + bits.bits(r) as u32 - 1;
            break;
        }
        k += 1;
    }
}

/// `decode_mcu_AC_refine` (jdphuff.c).
fn decode_ac_refine(
    bits: &mut Bits<'_>,
    block: &mut [i16],
    ac: &Huffman,
    scan: &Scan,
    eobrun: &mut u32,
) {
    let p1: i16 = 1 << scan.al;
    let m1: i16 = (-1_i16) << scan.al;
    let correct = |bits: &mut Bits<'_>, coef: &mut i16| {
        if bits.bit() != 0 && (*coef & p1) == 0 {
            *coef += if *coef >= 0 { p1 } else { m1 };
        }
    };
    let mut k = scan.ss;
    if *eobrun == 0 {
        while k <= scan.se {
            let rs = bits.decode(ac);
            let (mut r, s) = (i32::from(rs >> 4), rs & 15);
            let mut value = 0_i16;
            if s != 0 {
                value = if bits.bit() != 0 { p1 } else { m1 };
            } else if r != 15 {
                *eobrun = (1 << r) + bits.bits(r as u32) as u32;
                break;
            }
            while k <= scan.se {
                let coef = &mut block[NATURAL[k]];
                if *coef != 0 {
                    correct(bits, coef);
                } else {
                    r -= 1;
                    if r < 0 {
                        break;
                    }
                }
                k += 1;
            }
            if value != 0 {
                block[NATURAL[k]] = value;
            }
            k += 1;
        }
    }
    if *eobrun > 0 {
        while k <= scan.se {
            let coef = &mut block[NATURAL[k]];
            if *coef != 0 {
                correct(bits, coef);
            }
            k += 1;
        }
        *eobrun -= 1;
    }
}

/// `DESCALE(x, n)`: round and arithmetic-shift.
const fn descale(x: i64, n: u32) -> i64 {
    (x + (1 << (n - 1))) >> n
}

/// Post-IDCT `range_limit[x & RANGE_MASK]`: `clamp(x + 128, 0, 255)` over
/// the 10-bit wrapped index, exactly as libjpeg's table.
const fn range_limit_idct(x: i64) -> u8 {
    let t = x & 1023;
    if t < 128 {
        (t + 128) as u8
    } else if t < 512 {
        255
    } else if t < 896 {
        0
    } else {
        (t - 896) as u8
    }
}

/// `jpeg_idct_islow`: dequantise and inverse-DCT one block into `out`
/// (8 rows of `stride` bytes).
fn idct_islow(coef: &[i16], q: &[u16; 64], out: &mut [u8], stride: usize) {
    const CB: u32 = 13;
    const P1: u32 = 2;
    let mut ws = [0_i64; 64];
    for col in 0..8 {
        let c = |r: usize| i64::from(coef[r * 8 + col]) * i64::from(q[r * 8 + col]);
        if (1..8).all(|r| coef[r * 8 + col] == 0) {
            let dc = c(0) << P1;
            for r in 0..8 {
                ws[r * 8 + col] = dc;
            }
            continue;
        }
        let v = idct_1d([c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7)]);
        for r in 0..8 {
            ws[r * 8 + col] = descale(v[r], CB - P1);
        }
    }
    for row in 0..8 {
        let w = &ws[row * 8..row * 8 + 8];
        let o = &mut out[row * stride..row * stride + 8];
        if w[1..].iter().all(|&x| x == 0) {
            o.fill(range_limit_idct(descale(w[0], P1 + 3)));
            continue;
        }
        let v = idct_1d([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]);
        for (o, v) in o.iter_mut().zip(v) {
            *o = range_limit_idct(descale(v, CB + P1 + 3));
        }
    }
}

/// The shared even/odd butterfly of both `jpeg_idct_islow` passes, before
/// descaling. Inputs are the eight (dequantised or pass-1) values.
const fn idct_1d(x: [i64; 8]) -> [i64; 8] {
    const CB: u32 = 13;
    let (z2, z3) = (x[2], x[6]);
    let z1 = (z2 + z3) * 4433;
    let tmp2 = z1 + z3 * -15137;
    let tmp3 = z1 + z2 * 6270;
    let tmp0 = (x[0] + x[4]) << CB;
    let tmp1 = (x[0] - x[4]) << CB;
    let (t10, t13, t11, t12) = (tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2);

    let (o0, o1, o2, o3) = (x[7], x[5], x[3], x[1]);
    let (z1, z2, z3, z4) = (o0 + o3, o1 + o2, o0 + o2, o1 + o3);
    let z5 = (z3 + z4) * 9633;
    let (mut a0, mut a1, mut a2, mut a3) = (o0 * 2446, o1 * 16819, o2 * 25172, o3 * 12299);
    let (z1, z2) = (z1 * -7373, z2 * -20995);
    let z3 = z3 * -16069 + z5;
    let z4 = z4 * -3196 + z5;
    a0 += z1 + z3;
    a1 += z2 + z4;
    a2 += z2 + z3;
    a3 += z1 + z4;
    [
        t10 + a3,
        t11 + a2,
        t12 + a1,
        t13 + a0,
        t13 - a0,
        t12 - a1,
        t11 - a2,
        t10 - a3,
    ]
}

/// IDCT every block covering the component into a `bw*8 x bh*8` plane.
fn idct_plane(c: &Component, q: &[u16; 64]) -> Vec<u8> {
    let stride = c.bw * 8;
    let mut plane = vec![0_u8; stride * c.bh * 8];
    for by in 0..c.bh {
        for bx in 0..c.bw {
            let at = (by * c.stride + bx) * 64;
            let out = &mut plane[by * 8 * stride + bx * 8..];
            idct_islow(&c.coefs[at..at + 64], q, out, stride);
        }
    }
    plane
}

/// Upsample one component plane to the full `width x height` image, with
/// libjpeg-turbo's method selection (`jinit_upsampler`) and edge handling.
fn upsample(comp: &Component, plane: &[u8], frame: &Frame) -> Option<Vec<u8>> {
    let (width, height) = (frame.width, frame.height);
    let stride = comp.bw * 8;
    // jdmainct.c: rows above the image repeat row 0, rows below repeat the
    // last real row.
    let row = |y: isize| -> &[u8] {
        let y = y.clamp(0, comp.height as isize - 1) as usize;
        &plane[y * stride..y * stride + stride]
    };
    let fancy_w = comp.width > 2;
    let mut out = vec![0_u8; width * height];
    if comp.h == frame.hmax && comp.v == frame.vmax {
        for y in 0..height {
            out[y * width..(y + 1) * width].copy_from_slice(&row(y as isize)[..width]);
        }
    } else if 2 * comp.h == frame.hmax && comp.v == frame.vmax && fancy_w {
        for y in 0..height {
            out[y * width..(y + 1) * width]
                .copy_from_slice(&h2_fancy(row(y as isize), comp.width)[..width]);
        }
    } else if comp.h == frame.hmax && 2 * comp.v == frame.vmax {
        for y in 0..height {
            let (near, far, bias) = v2_rows(&row, y);
            for x in 0..width {
                out[y * width + x] =
                    ((i32::from(near[x]) * 3 + i32::from(far[x]) + bias) >> 2) as u8;
            }
        }
    } else if 2 * comp.h == frame.hmax && 2 * comp.v == frame.vmax && fancy_w {
        for y in 0..height {
            let (near, far, _) = v2_rows(&row, y);
            out[y * width..(y + 1) * width]
                .copy_from_slice(&h2v2_fancy(near, far, comp.width)[..width]);
        }
    } else if frame.hmax.is_multiple_of(comp.h) && frame.vmax.is_multiple_of(comp.v) {
        // h2v1_upsample, h2v2_upsample and int_upsample all replicate.
        let (hx, vx) = (frame.hmax / comp.h, frame.vmax / comp.v);
        for y in 0..height {
            let src = row((y / vx) as isize);
            for x in 0..width {
                out[y * width + x] = src[x / hx];
            }
        }
    } else {
        return None;
    }
    Some(out)
}

/// Rows feeding output row `y` of a vertical 2x fancy upsample: the nearest
/// input row, the next nearest (above for even rows, below for odd) and the
/// rounding bias (`1` above, `2` below).
fn v2_rows<'p>(row: &impl Fn(isize) -> &'p [u8], y: usize) -> (&'p [u8], &'p [u8], i32) {
    let near = (y / 2) as isize;
    if y.is_multiple_of(2) {
        (row(near), row(near - 1), 1)
    } else {
        (row(near), row(near + 1), 2)
    }
}

/// `h2v1_fancy_upsample` of one row of `n` samples.
fn h2_fancy(input: &[u8], n: usize) -> Vec<u8> {
    let s = |i: usize| i32::from(input[i]);
    let mut out = Vec::with_capacity(2 * n);
    out.push(input[0]);
    out.push(((s(0) * 3 + s(1) + 2) >> 2) as u8);
    for i in 1..n - 1 {
        let v = s(i) * 3;
        out.push(((v + s(i - 1) + 1) >> 2) as u8);
        out.push(((v + s(i + 1) + 2) >> 2) as u8);
    }
    out.push(((s(n - 1) * 3 + s(n - 2) + 1) >> 2) as u8);
    out.push(input[n - 1]);
    out
}

/// One output row of `h2v2_fancy_upsample` from its nearest and next-nearest
/// input rows.
fn h2v2_fancy(near: &[u8], far: &[u8], n: usize) -> Vec<u8> {
    let col = |i: usize| i32::from(near[i]) * 3 + i32::from(far[i]);
    let mut out = Vec::with_capacity(2 * n);
    let (mut this, next) = (col(0), col(1));
    out.push(((this * 4 + 8) >> 4) as u8);
    out.push(((this * 3 + next + 7) >> 4) as u8);
    let mut last = this;
    this = next;
    for i in 2..n {
        let next = col(i);
        out.push(((this * 3 + last + 8) >> 4) as u8);
        out.push(((this * 3 + next + 7) >> 4) as u8);
        last = this;
        this = next;
    }
    out.push(((this * 3 + last + 8) >> 4) as u8);
    out.push(((this * 4 + 7) >> 4) as u8);
    out
}

/// `ycc_rgb_convert` with `build_ycc_rgb_table`'s 16-bit fixed point.
fn ycc_to_rgb(planes: &[Vec<u8>], out: &mut [u8]) {
    const SCALEBITS: u32 = 16;
    const ONE_HALF: i64 = 1 << (SCALEBITS - 1);
    // `FIX(x)`; scaling by 2^16 is exact, so fusing changes nothing.
    let fix = |x: f64| x.mul_add(f64::from(1_u32 << SCALEBITS), 0.5) as i64;
    let red_cr = fix(1.40200);
    let blue_cb = fix(1.77200);
    let green_from_red = fix(0.71414);
    let green_from_blue = fix(0.34414);
    let clamp = |v: i64| v.clamp(0, 255) as u8;
    for (i, px) in out.chunks_exact_mut(3).enumerate() {
        let y = i64::from(planes[0][i]);
        let cb = i64::from(planes[1][i]) - 128;
        let cr = i64::from(planes[2][i]) - 128;
        let r = (red_cr * cr + ONE_HALF) >> SCALEBITS;
        let g = (-green_from_blue * cb + ONE_HALF + -green_from_red * cr) >> SCALEBITS;
        let b = (blue_cb * cb + ONE_HALF) >> SCALEBITS;
        px.copy_from_slice(&[clamp(y + r), clamp(y + g), clamp(y + b)]);
    }
}

#[cfg(test)]
mod tests {
    use super::decode;

    /// Compares against Pillow-decoded `.raw` files written next to each
    /// `.jpg` (see the parity scripts); run on demand.
    #[test]
    #[ignore = "needs a directory of JPEGs decoded by Pillow"]
    fn matches_pillow() {
        let dir = std::path::PathBuf::from(std::env::var("EG2_JPEG_DIR").expect("EG2_JPEG_DIR"));
        let cases: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("cases.json")).unwrap()).unwrap();
        let mut failed = 0;
        for case in cases.as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let jpg = std::fs::read(dir.join(format!("{name}.jpg"))).unwrap();
            let reference = std::fs::read(dir.join(format!("{name}.raw"))).unwrap();
            let Some(d) = decode(&jpg) else {
                println!("{name:20} UNSUPPORTED");
                failed += 1;
                continue;
            };
            let same = d
                .data
                .iter()
                .zip(&reference)
                .filter(|(a, b)| a == b)
                .count();
            let ok = d.data.len() == reference.len() && same == reference.len();
            failed += usize::from(!ok);
            println!(
                "{name:20} {}x{}x{}  identical {:7.3}%  {}",
                d.width,
                d.height,
                d.channels,
                100.0 * same as f64 / reference.len() as f64,
                if ok { "BIT-IDENTICAL" } else { "DIFFER" }
            );
        }
        assert_eq!(failed, 0, "{failed} JPEGs differ from Pillow");
    }
}
