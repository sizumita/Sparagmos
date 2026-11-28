//! The `std::simd` value form: a `Simd` value crossing
//! between its array denomination and its aggregate one
//! through core's `MaybeUninit` load/store copies
//! (`Simd::from_array` loads the array into a fresh
//! `MaybeUninit<Simd>` slot with a count-1 `copy_nonoverlapping`;
//! `Simd::to_array` stores the vector into a fresh
//! `MaybeUninit<[T; N]>` slot the same way). The slot's stored
//! form is denominated in the payload's own slots — an array
//! payload's slot reads back as a tuple of uninitialised
//! elements — so the copy's destination judgement must still
//! take the position for the one element it denotes; writing
//! the element as slot zero of a fresh sequence wraps it in a
//! one-element tuple no consumer denominates as the element
//! ("expected Array(Uint(U8), 2), found tuple"). The same
//! judgement serves any whole-value payload through a fresh
//! `MaybeUninit` slot, and the guards at the bottom pin what
//! it must not claim: byte copies and multi-element copies
//! into fresh slots still fill them element by element.

use super::{fixture, run_driver_extra, temp_dir};

fn check(name: &str, src: &str) {
    let dir = temp_dir(name);
    let src = fixture(&dir, "value_form.rs", src);
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
            "simd value form {name} opt={opt} stderr={stderr}"
        );
        assert_eq!(
            stdout, "ok\n",
            "simd value form {name} opt={opt} stdout={stdout} stderr={stderr}"
        );
    }
}

const FROM_ARRAY_TO_ARRAY: &str = r#"
#![feature(portable_simd)]
use std::simd::prelude::*;

fn main() {
    let v = Simd::from_array([7u8, 9]);
    let a = v.to_array();
    assert_eq!(a, [7, 9]);
    let w = u32x4::from_array([1, 2, 3, 4]);
    let b = w.to_array();
    assert_eq!(b, [1, 2, 3, 4]);
    println!("ok");
}
"#;

const TRANSMUTE_UNCHECKED: &str = r#"
#![feature(portable_simd, core_intrinsics)]
use std::simd::prelude::*;

fn main() {
    let v = Simd::from_array([7u8, 9]);
    let a: [u8; 2] = unsafe { std::intrinsics::transmute_unchecked(v) };
    assert_eq!(a, [7, 9]);
    println!("ok");
}
"#;

const SPLAT_ADD_TO_ARRAY: &str = r#"
#![feature(portable_simd)]
use std::simd::prelude::*;

fn main() {
    let v = u8x2::splat(3);
    let w = v + v;
    assert_eq!(w.to_array(), [6, 6]);
    let v2 = Simd::from_array([10u8, 20]);
    assert!(v2[0] == 10 && v2[1] == 20);
    println!("ok");
}
"#;

const CONST_AND_MAYBEUNINIT: &str = r#"
#![feature(portable_simd)]
use std::mem::MaybeUninit;
use std::simd::prelude::*;

const C: Simd<u8, 2> = Simd::from_array([7, 9]);

fn main() {
    let a = C.to_array();
    assert_eq!(a, [7, 9]);
    let x = MaybeUninit::new(C);
    let v = unsafe { x.assume_init() };
    assert!(v[0] == 7 && v[1] == 9);
    assert_eq!(v.to_array(), [7, 9]);
    println!("ok");
}
"#;

const MAYBEUNINIT_COPIES: &str = r#"
use std::mem::MaybeUninit;

fn main() {
    unsafe {
        // A whole-array payload filled into a fresh
        // MaybeUninit slot by a count-1 bulk copy.
        let src: [u32; 3] = [10, 20, 30];
        let mut slot: MaybeUninit<[u32; 3]> = MaybeUninit::uninit();
        std::ptr::copy_nonoverlapping(
            &src as *const [u32; 3],
            slot.as_mut_ptr(),
            1,
        );
        assert_eq!(slot.assume_init(), [10, 20, 30]);

        // A byte copy into the same kind of slot still lands
        // element by element.
        let bytes: [u8; 2] = [5, 6];
        let mut bslot: MaybeUninit<[u8; 2]> = MaybeUninit::uninit();
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            bslot.as_mut_ptr() as *mut u8,
            2,
        );
        assert_eq!(bslot.assume_init(), [5, 6]);

        // A count-2 copy of aggregate elements into a fresh
        // buffer still fills the sequence.
        let srcs: [[u8; 2]; 2] = [[1, 2], [3, 4]];
        let mut buf: MaybeUninit<[[u8; 2]; 2]> = MaybeUninit::uninit();
        std::ptr::copy_nonoverlapping(
            srcs.as_ptr(),
            buf.as_mut_ptr() as *mut [u8; 2],
            2,
        );
        assert_eq!(buf.assume_init(), [[1, 2], [3, 4]]);
    }
    println!("ok");
}
"#;

#[test]
fn simd_from_array_to_array() {
    check("from_array_to_array", FROM_ARRAY_TO_ARRAY);
}

#[test]
fn simd_transmute_unchecked_to_array() {
    check("transmute_unchecked", TRANSMUTE_UNCHECKED);
}

#[test]
fn simd_splat_add_to_array() {
    check("splat_add", SPLAT_ADD_TO_ARRAY);
}

#[test]
fn simd_const_and_maybeuninit() {
    check("const_and_maybeuninit", CONST_AND_MAYBEUNINIT);
}

#[test]
fn maybeuninit_whole_and_element_copies() {
    check("maybeuninit_copies", MAYBEUNINIT_COPIES);
}
