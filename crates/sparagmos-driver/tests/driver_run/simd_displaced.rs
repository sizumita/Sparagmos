//! Whole-array stores and loads through a pointer displaced
//! by a byte offset from a value boundary: a reference
//! transmuted to an array type at a byte offset inside
//! differently typed storage (the shape of
//! tests/ui/simd/issue-32947.rs, where a `[u8; 32]` is written
//! one byte into a `[u64; 8]`). The position crosses the base
//! storage's element boundaries, so no structural projection
//! denotes it; the byte view splices the value's byte image
//! at the displaced position instead.

use super::{fixture, run_driver_extra, temp_dir};

fn check(name: &str, src: &str) {
    let dir = temp_dir(name);
    let src = fixture(&dir, "displaced.rs", src);
    for opt in [false, true] {
        let out = if opt {
            run_driver_extra(&src, &["--rustc-arg", "-O"])
        } else {
            run_driver_extra(&src, &[])
        };
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(0),
            "displaced array access {name} opt={opt} stderr={stderr}"
        );
        assert_eq!(
            stdout, "ok\n",
            "displaced array access {name} opt={opt} stdout={stdout} stderr={stderr}"
        );
    }
}

const ISSUE_32947_SHAPE: &str = r#"
#![feature(repr_simd)]
use std::mem::transmute;
#[repr(simd)]
#[derive(Copy, Clone)]
pub struct Mu64(pub [u64; 4]);
fn main() {
    unsafe {
        let mut arr = [0u64; 8];
        let memory = &mut arr as *mut _ as *mut u8;
        let misaligned_ptr: &mut [u8; 32] = transmute(memory.offset(1));
        *misaligned_ptr = transmute(Mu64([1, 1, 1, 1]));
        // The written array is the little-endian bytes of four
        // 1u64 lanes, spliced at byte 1 of the storage.
        assert_eq!(arr[0], 0x0100);
        assert_eq!(arr[1], 0x0100);
        assert_eq!(arr[2], 0x0100);
        assert_eq!(arr[3], 0x0100);
        assert_eq!(arr[4], 0);
        // A direct copy back through the same reference reads
        // the spliced bytes as the array again.
        let back: [u8; 32] = *misaligned_ptr;
        assert_eq!(back[0], 1);
        assert_eq!(back[1], 0);
        assert_eq!(back[8], 1);
        assert_eq!(back[31], 0);
    }
    println!("ok");
}
"#;

const DISPLACED_BYTE_ARRAY: &str = r#"
fn main() {
    let mut arr = [0u64; 8];
    unsafe {
        let base = &mut arr as *mut u64 as *mut u8;
        let dst = base.add(1) as *mut [u8; 32];
        *dst = [7u8; 32];
        let back = *(base.add(1) as *const [u8; 32]);
        assert_eq!(back, [7u8; 32]);
        assert_eq!(arr[0] & 0xff, 0);
        assert_eq!(arr[0] >> 8, 0x07070707070707);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_simd_issue_32947_displaced_array_store() {
    check("simd-displaced-32947", ISSUE_32947_SHAPE);
}

#[test]
fn run_mode_displaced_byte_array_store() {
    check("simd-displaced-byte-array", DISPLACED_BYTE_ARRAY);
}
