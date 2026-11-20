//! extern "C" surface for JIT-generated code. No unwinding across FFI:
//! failures are recorded (see error::last_panic) and signalled by null / -1.
use crate::alloc::{with_global_arena, Allocator};
use crate::error::record_panic;
use crate::layout::Layout;
use std::ffi::c_void;

/// Allocate `size` bytes with `align` from the global arena (leaked by design v1).
/// Returns null on failure / invalid align.
#[no_mangle]
pub extern "C" fn sparagmos_alloc(size: usize, align: usize) -> *mut c_void {
    if size == 0 {
        return align as *mut c_void;
    }
    let layout = match Layout::new(size, if align == 0 { 1 } else { align }) {
        Ok(l) => l,
        Err(e) => {
            record_panic(e.to_string(), None);
            return std::ptr::null_mut();
        }
    };
    with_global_arena(|a| {
        unsafe { a.alloc(layout) }
            .map(|p| p.as_ptr() as *mut c_void)
            .unwrap_or_else(|e| {
                record_panic(e.to_string(), None);
                std::ptr::null_mut()
            })
    })
}
#[no_mangle]
pub extern "C" fn sparagmos_alloc_zeroed(size: usize, align: usize) -> *mut c_void {
    let p = sparagmos_alloc(size, align);
    if !p.is_null() && size > 0 {
        unsafe {
            std::ptr::write_bytes(p as *mut u8, 0, size);
        }
    }
    p
}
/// Dealloc counterpart. Arena v1 leaks, so this is intentionally a no-op;
/// the symbol exists so JIT code can emit paired alloc/dealloc and the
/// backend can later be swapped for SystemAllocator without changing call sites.
/// # Safety: ptr must have come from sparagmos_alloc with same size/align.
#[no_mangle]
pub unsafe extern "C" fn sparagmos_dealloc(_ptr: *mut c_void, _size: usize, _align: usize) {}
/// Record a panic from JIT code. `msg` is UTF-8, not NUL-terminated. Returns -1 always.
/// # Safety: msg must be valid for len bytes (or null when len==0).
#[no_mangle]
pub unsafe extern "C" fn sparagmos_panic(msg: *const u8, len: usize) -> i32 {
    let s = if msg.is_null() {
        "<null panic message>".to_string()
    } else {
        String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(msg, len) }).into_owned()
    };
    record_panic(s, None);
    -1
}
/// Bounds check helper: 0 if index < len, else records panic and returns -1.
#[no_mangle]
pub extern "C" fn sparagmos_bounds_check(index: usize, len: usize) -> i32 {
    if index < len {
        0
    } else {
        record_panic(
            format!("index out of bounds: index {index} len {len}"),
            None,
        );
        -1
    }
}
/// Checked division helper: writes quotient to `out`, 0 on success, -1 on div-by-zero.
#[no_mangle]
pub unsafe extern "C" fn sparagmos_checked_div_i64(a: i64, b: i64, out: *mut i64) -> i32 {
    if b == 0 {
        record_panic("attempt to divide by zero", None);
        return -1;
    }
    match a.checked_div(b) {
        Some(v) => {
            if !out.is_null() {
                unsafe {
                    *out = v;
                }
            }
            0
        }
        None => {
            record_panic("arithmetic overflow in div", None);
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{clear_last_panic, last_panic};
    #[test]
    fn ffi_alloc_write_read() {
        let p = sparagmos_alloc(8, 8);
        assert!(!p.is_null());
        unsafe {
            *(p as *mut u64) = 123;
            assert_eq!(*(p as *mut u64), 123);
            sparagmos_dealloc(p, 8, 8);
        }
    }
    #[test]
    fn ffi_bounds_and_panic() {
        let _g = crate::error::PANIC_TEST_LOCK.lock().unwrap();
        clear_last_panic();
        assert_eq!(sparagmos_bounds_check(1, 3), 0);
        assert_eq!(sparagmos_bounds_check(3, 3), -1);
        assert!(last_panic().unwrap().message.contains("out of bounds"));
        clear_last_panic();
        unsafe {
            assert_eq!(sparagmos_panic(b"kaboom".as_ptr(), 6), -1);
        }
        assert_eq!(last_panic().unwrap().message, "kaboom");
        clear_last_panic();
    }
    #[test]
    fn ffi_zeroed_and_div() {
        let _g = crate::error::PANIC_TEST_LOCK.lock().unwrap();
        let p = sparagmos_alloc_zeroed(16, 8);
        unsafe {
            assert_eq!(*(p as *mut u64), 0);
        }
        let mut out = 0i64;
        unsafe {
            assert_eq!(sparagmos_checked_div_i64(10, 2, &mut out), 0);
            assert_eq!(out, 5);
            assert_eq!(sparagmos_checked_div_i64(1, 0, &mut out), -1);
        }
        clear_last_panic();
    }
}
