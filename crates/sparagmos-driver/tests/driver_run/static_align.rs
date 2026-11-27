//! Statics whose own alignment attribute exceeds their type's
//! alignment (`#[rustc_align_static(N)]`, the shape of
//! tests/ui/static/static-align.rs in the rustc suite): the
//! storage's address must be a multiple of the declared
//! alignment, for a plain static, for an exported static, for
//! a foreign declaration bound to that exported static's
//! storage, and for a thread-local static (Issue #395).

use super::*;

const ALIGN_SRC: &str = r##"
#![feature(static_align)]

#[rustc_align_static(64)]
static A: u8 = 0;

#[rustc_align_static(4096)]
static B: u8 = 0;

#[rustc_align_static(128)]
#[no_mangle]
static EXPORTED: u64 = 0;

unsafe extern "C" {
    #[rustc_align_static(128)]
    #[link_name = "EXPORTED"]
    static C: u64;
}

thread_local! {
    #[rustc_align_static(4096)]
    static LOCAL: u64 = 0;
}

fn main() {
    assert!(core::ptr::from_ref(&A).addr().is_multiple_of(64));
    assert!(core::ptr::from_ref(&B).addr().is_multiple_of(4096));
    assert!(core::ptr::from_ref(&EXPORTED).addr().is_multiple_of(128));
    unsafe {
        assert!(core::ptr::from_ref(&C).addr().is_multiple_of(128));
        // The foreign declaration denotes the definition's
        // storage, so the two addresses are one.
        assert_eq!(
            core::ptr::from_ref(&C).addr(),
            core::ptr::from_ref(&EXPORTED).addr()
        );
    }
    LOCAL.with(|local| {
        assert!(core::ptr::from_ref(local).addr().is_multiple_of(4096));
    });
    println!("aligned");
}
"##;

#[test]
fn run_mode_static_align_attribute_is_honoured() {
    let dir = temp_dir("static-align");
    let src = fixture(&dir, "align.rs", ALIGN_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "aligned statics must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "aligned\n",
        "stdout mismatch; stderr={stderr}"
    );
}
