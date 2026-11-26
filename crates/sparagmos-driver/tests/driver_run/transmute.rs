//! Transmute shapes that only optimized MIR produces: an
//! inlined small-array integer comparison transmutes a
//! one-field `#[repr(simd)]` wrapper straight to the integer
//! of its array's total width, and the reverse construction
//! builds the wrapper back from an integer. The wrapper's
//! storage length is a const parameter the type
//! representation does not carry, and sibling instances of
//! the same wrapper (here `Simd<u8, 4>` next to
//! `Simd<u8, 2>`) share the type-argument key the recorded
//! shapes are filed under, so the pair is decided by the
//! bit-stream reinterpretation whose element count comes
//! from the integer's width.

use super::{fixture, run_driver_extra, temp_dir};

const SIMD_INT_TRANSMUTE_SRC: &str = r#"
#![feature(repr_simd, core_intrinsics, adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;
use std::intrinsics::simd::simd_shuffle;

#[derive(Copy, Clone, ConstParamTy, PartialEq, Eq)]
#[repr(simd)]
struct Simd<T, const N: usize>([T; N]);

fn into_array<T, const N: usize>(v: Simd<T, N>) -> [T; N] {
    unsafe { std::intrinsics::transmute_unchecked(v) }
}

#[inline(never)]
fn get_u16() -> u16 {
    // The guest's argv is just its own path, so this is
    // 0x0501: the little-endian bytes [1, 5].
    0x0500 + std::env::args().count() as u16
}

fn main() {
    const I2: Simd<u32, 2> = Simd([1, 5]);
    let a = Simd::<u8, 4>([0, 1, 2, 3]);
    let b = Simd::<u8, 4>([4, 5, 6, 7]);
    unsafe {
        let y: Simd<u8, 2> = simd_shuffle(a, b, I2);
        assert_eq!(into_array(y), [1, 5]);
        let w: Simd<u8, 2> =
            std::intrinsics::transmute_unchecked(get_u16());
        assert_eq!(into_array(w), [1, 5]);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_simd_wrapper_int_transmute_o2() {
    let dir = temp_dir("simd-int-transmute");
    let src = fixture(&dir, "simd_int_transmute.rs", SIMD_INT_TRANSMUTE_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-O"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the wrapper <-> integer transmutes must convert and run; stderr={stderr}"
    );
    assert_eq!(stdout, "ok\n", "stdout={stdout} stderr={stderr}");
}
