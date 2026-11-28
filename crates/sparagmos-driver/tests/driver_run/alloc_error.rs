use super::{fixture, run_driver, run_driver_extra, run_ok, temp_dir};

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


// ----- the handler-side string's length slot -----
//
// The two `tests/ui/allocator/no_std-alloc-error-handler-*`
// records read the length of a `String` the interpreter holds as
// bare string data: in the default-handler record the panic
// handler takes `panic_info.to_string()` and passes `s.as_ptr()`
// / `s.len()` to `libc::write`; in the custom-handler record the
// alloc error handler builds its message with `fmt::write` and
// reads the same slots. MIR inlines `String::len` to the place
// projection `_s.0.1` (the `vec` field's `len` slot), and the
// borrowing projector must answer field 1 of a bare string with
// its length, as the owning projector does.

const NO_STD_TOSTRING_LEN_SRC: &str = r#"
#![no_std]
#![no_main]
extern crate alloc;
use core::alloc::{GlobalAlloc, Layout};
use alloc::boxed::Box;
use alloc::string::ToString;

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
    let s = info.to_string();
    let l = s.len();
    let p = s.as_ptr();
    panic!("handler len={} ptr_nonnull={}", l, !p.is_null())
}

#[no_mangle]
extern "C" fn main() -> i32 {
    let b = Box::new([0u64; 512]);
    b[0] as i32
}
"#;

#[test]
fn run_mode_no_std_panic_handler_reads_tostring_string_len() {
    let dir = temp_dir("alloc-error-tostring-len");
    let src = fixture(&dir, "tostring_len.rs", NO_STD_TOSTRING_LEN_SRC);
    let out = run_driver_extra(
        &src,
        &["--edition", "2021", "--rustc-arg", "-C", "--rustc-arg", "panic=abort"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("handler len=84 ptr_nonnull=true"),
        "the panic handler reads the to_string String's length slot \
         (the Display text is 84 bytes): {stderr}"
    );
}

const NO_STD_BUILT_STRING_LEN_SRC: &str = r#"
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
    use alloc::fmt::write;
    let size = layout.size();
    let mut s = alloc::string::String::new();
    write(&mut s, format_args!("My OOM: failed to allocate {} bytes!\n", size)).unwrap();
    let l = s.len();
    let p = s.as_ptr();
    panic!("oom len={} ptr_nonnull={}", l, !p.is_null())
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
fn run_mode_no_std_alloc_error_handler_reads_built_string_len() {
    let dir = temp_dir("alloc-error-built-len");
    let src = fixture(&dir, "built_len.rs", NO_STD_BUILT_STRING_LEN_SRC);
    let out = run_driver_extra(
        &src,
        &["--edition", "2021", "--rustc-arg", "-C", "--rustc-arg", "panic=abort"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("oom len=39 ptr_nonnull=true"),
        "the alloc error handler reads the fmt::write-built String's \
         length slot (the message is 39 bytes): {stderr}"
    );
}


// ----- allocation-error hook (`set_alloc_error_hook`) -----
//
// `std::alloc::set_alloc_error_hook` / `take_alloc_error_hook`
// have no Stable MIR body; the interpreter keeps the hook in
// its state and `run_alloc_error` invokes it with a
// fabricated `Layout` value in place of the default hook.

const HOOK_FIRES_SRC: &str = r#"
#![feature(alloc_error_hook)]
use std::panic::catch_unwind;
fn main() {
    std::alloc::set_alloc_error_hook(|layout| {
        println!("hook saw size {}", layout.size());
        panic!("hook panics");
    });
    let caught = catch_unwind(|| {
        let layout = std::alloc::Layout::from_size_align(4096, 8).unwrap();
        std::alloc::handle_alloc_error(layout);
    });
    println!("caught: {}", caught.is_err());
}
"#;

#[test]
fn run_mode_alloc_error_hook_fires_with_the_layout_and_panics() {
    let dir = temp_dir("alloc-hook-fires");
    let src = fixture(&dir, "case.rs", HOOK_FIRES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hook saw size 4096\ncaught: true\n",
        "stderr={stderr}"
    );
    assert!(
        stderr.contains("hook panics"),
        "the hook's panic is the one caught: {stderr}"
    );
}

const HOOK_TAKE_SRC: &str = r#"
#![feature(alloc_error_hook)]
use std::alloc::Layout;
fn my_hook(layout: Layout) { println!("custom hook {}", layout.size()); }
fn main() {
    std::alloc::set_alloc_error_hook(my_hook);
    let taken = std::alloc::take_alloc_error_hook();
    taken(Layout::new::<[u8; 7]>());
    println!("after taken call");
    let mine: fn(Layout) = my_hook;
    let def = std::alloc::take_alloc_error_hook();
    println!("default differs: {}", !std::ptr::fn_addr_eq(def, mine));
}
"#;

#[test]
fn run_mode_alloc_error_hook_take_returns_installed_then_default() {
    run_ok(
        "alloc-hook-take",
        HOOK_TAKE_SRC,
        "custom hook 7\nafter taken call\ndefault differs: true\n",
    );
}

const HOOK_DEFAULT_CALL_SRC: &str = r#"
#![feature(alloc_error_hook)]
fn main() {
    let def = std::alloc::take_alloc_error_hook();
    def(std::alloc::Layout::new::<[u8; 42]>());
    println!("never");
}
"#;

#[test]
fn run_mode_alloc_error_hook_default_hook_reports_and_aborts() {
    let dir = temp_dir("alloc-hook-default");
    let src = fixture(&dir, "case.rs", HOOK_DEFAULT_CALL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(134),
        "calling the default hook aborts the guest; stderr={stderr}"
    );
    assert!(
        stderr.contains("memory allocation of 42 bytes failed"),
        "the default hook reports the failed size: {stderr}"
    );
}
