//! Independent scalar mask oracle; no models, downloads or GPU required.

use candle_core::Device;

use super::sliding_window_mask;

#[test]
fn prefilled_window_mask_matches_every_scalar_cell() {
    for seq in 0_usize..37 {
        for window in [0, 1, 2, 7, 16, 32, 64] {
            let actual = sliding_window_mask(seq, window, &Device::Cpu).unwrap();
            if seq <= window + 1 {
                assert!(actual.is_none());
                continue;
            }
            let rows = actual.unwrap().to_vec2::<f32>().unwrap();
            assert_eq!(rows.len(), seq);
            for (q, row) in rows.iter().enumerate() {
                assert_eq!(row.len(), seq);
                for (k, &value) in row.iter().enumerate() {
                    let expected = if q.abs_diff(k) <= window {
                        0.0_f32
                    } else {
                        f32::MIN
                    };
                    assert_eq!(
                        value.to_bits(),
                        expected.to_bits(),
                        "seq={seq} window={window} q={q} k={k}"
                    );
                }
            }
        }
    }
}
