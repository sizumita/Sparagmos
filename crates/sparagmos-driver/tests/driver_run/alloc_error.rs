use super::{fixture, run_driver, run_driver_extra, temp_dir};

// ----- allocation-error handling (`handle_alloc_error` / `handle_error`) --
//
// `std::alloc::handle_alloc_error` and `alloc::raw_vec::handle_error`
// have no Stable MIR body; the interpreter models them. With std
// linked, a failure prints the default hook's report and aborts;
// a guest `#[alloc_error_handler]` runs instead when one exists;
// in a `no_std` program with neither, the failure is raised to
// the guest's panic handler as a `PanicInfo`, like alloc's
// `__rdl_alloc_error_handler`.

const STD_HANDLE_ALLOC_ERROR_SRC: &str = r#"
fn main() {
    let layout = std::alloc::Layout::new::<[u8; 42]>();
    std::alloc::handle_alloc_error(layout);
}
"#;

#[test]
fn run_mode_std_handle_alloc_error_reports_and_aborts() {
    let dir = temp_dir("alloc-error-std");
    let src = fixture(&dir, "oom.rs", STD_HANDLE_ALLOC_ERROR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(134),
        "an allocation error aborts the guest; stderr={stderr}"
    );
    assert!(
        stderr.contains("memory allocation of 42 bytes failed"),
        "the default hook reports the failed size: {stderr}"
    );
}

const CAPACITY_OVERFLOW_SRC: &str = r#"
fn main() {
    std::collections::VecDeque::<String>::with_capacity(!0);
}
"#;

#[test]
fn run_mode_capacity_overflow_panics_at_raw_vec_site() {
    let dir = temp_dir("alloc-error-capacity");
    let src = fixture(&dir, "capacity.rs", CAPACITY_OVERFLOW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "capacity overflow is a guest panic; stderr={stderr}"
    );
    assert!(
        stderr.contains("capacity overflow"),
        "the panic payload is the capacity-overflow message: {stderr}"
    );
    assert!(
        stderr.contains("library/alloc/src/raw_vec/mod.rs"),
        "the reported location is the panic site inside RawVec: {stderr}"
    );
}

const NO_STD_CUSTOM_HANDLER_SRC: &str = r#"
#![no_std]
#![no_main]
#![feature(alloc_error_handler)]
extern crate alloc;
use core::alloc::{GlobalAlloc, Layout};
use alloc::boxed::Box;

struct FailAlloc;
unsafe impl GlobalAlloc for FailAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= 4096 { core::ptr::null_mut() } else { core::ptr::without_provenance_mut(0x1000) }
    }
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}
#[global_allocator]
static GLOBAL: FailAlloc = FailAlloc;

#[alloc_error_handler]
fn my_oom(layout: Layout) -> ! {
    panic!("oom-marker size={} align={}", layout.size(), layout.align())
}

#[panic_handler]
fn ph(_info: &core::panic::PanicInfo) -> ! { loop {} }

#[no_mangle]
extern "C" fn main() -> i32 {
    let b = Box::new([0u64; 512]);
    b[0] as i32
}
"#;

#[test]
fn run_mode_no_std_custom_alloc_error_handler_gets_layout() {
    let dir = temp_dir("alloc-error-custom");
    let src = fixture(&dir, "custom.rs", NO_STD_CUSTOM_HANDLER_SRC);
    let out = run_driver_extra(
        &src,
        &["--edition", "2021", "--rustc-arg", "-C", "--rustc-arg", "panic=abort"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("oom-marker size=4096 align=8"),
        "the guest alloc error handler runs with the failed layout: {stderr}"
    );
}

const NO_STD_DEFAULT_HANDLER_SRC: &str = r#"
#![no_std]
#![no_main]
extern crate alloc;
use core::alloc::{GlobalAlloc, Layout};
use alloc::boxed::Box;

struct FailAlloc;
unsafe impl GlobalAlloc for FailAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= 4096 { core::ptr::null_mut() } else { core::ptr::without_provenance_mut(0x1000) }
    }
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}
#[global_allocator]
static GLOBAL: FailAlloc = FailAlloc;

#[panic_handler]
fn ph(info: &core::panic::PanicInfo) -> ! {
    panic!("handler got: {}", info.message())
}

#[no_mangle]
extern "C" fn main() -> i32 {
    let b = Box::new([0u64; 512]);
    b[0] as i32
}
"#;

#[test]
fn run_mode_no_std_default_alloc_error_reaches_panic_handler() {
    let dir = temp_dir("alloc-error-default");
    let src = fixture(&dir, "default.rs", NO_STD_DEFAULT_HANDLER_SRC);
    let out = run_driver_extra(
        &src,
        &["--edition", "2021", "--rustc-arg", "-C", "--rustc-arg", "panic=abort"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("handler got: memory allocation of 4096 bytes failed"),
        "with no alloc error handler, the failure reaches the panic \
         handler as a PanicInfo carrying the allocation message: {stderr}"
    );
}
