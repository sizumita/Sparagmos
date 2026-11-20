//! Executable memory management.
//!
//! NOTE on the `libc` dependency: `libc` is used ONLY here, and only for the
//! OS primitives `mmap` / `mprotect` / `munmap` / `sysconf` (allocating pages
//! and flipping them between writable and executable). `libc` is an OS
//! foundation crate, not a code generator — all machine code bytes are
//! encoded by hand in [`crate::emitter`]. The mapping follows W^X: pages are
//! mapped RW, code is written, then they are flipped to RX before execution.

use std::ffi::c_void;
use std::ptr;

/// A page-aligned region of memory that holds JIT-compiled code.
///
/// Lifecycle: allocate (RW) -> write code -> `make_executable` (RX) -> call.
/// Freed with `munmap` on drop.
pub struct JitMemory {
    ptr: *mut u8,
    /// Mapped length (page-rounded).
    len: usize,
    /// Number of code bytes written.
    used: usize,
    executable: bool,
}

// The region is owned exclusively by this value; moving it between threads
// is sound because execution only happens through &self reads of immutable
// (post-seal) code.
unsafe impl Send for JitMemory {}
unsafe impl Sync for JitMemory {}

fn page_size() -> usize {
    // SAFETY: sysconf with a valid name returns the page size (or -1).
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v > 0 {
        v as usize
    } else {
        4096
    }
}

fn round_up(v: usize, align: usize) -> usize {
    v.div_ceil(align) * align
}

impl JitMemory {
    /// Allocate at least `capacity` bytes of RW (not yet executable) memory.
    pub fn new(capacity: usize) -> std::io::Result<Self> {
        let len = round_up(capacity.max(1), page_size());
        // SAFETY: standard anonymous private mmap; result checked below.
        let p = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self {
            ptr: p as *mut u8,
            len,
            used: 0,
            executable: false,
        })
    }

    pub fn capacity(&self) -> usize {
        self.len
    }
    pub fn used(&self) -> usize {
        self.used
    }
    pub fn is_executable(&self) -> bool {
        self.executable
    }
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Copy code bytes into the region. Must be called before
    /// `make_executable`.
    pub fn write_code(&mut self, code: &[u8]) -> std::io::Result<()> {
        if self.executable {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "JitMemory is already sealed executable (W^X)",
            ));
        }
        if code.len() > self.len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "code does not fit in JitMemory",
            ));
        }
        // SAFETY: ptr is a valid writable mapping of self.len bytes and the
        // length was checked above; regions do not overlap (code is a slice).
        unsafe {
            ptr::copy_nonoverlapping(code.as_ptr(), self.ptr, code.len());
        }
        self.used = code.len();
        Ok(())
    }

    /// Flip the region from RW to RX so the code can be executed.
    pub fn make_executable(&mut self) -> std::io::Result<()> {
        if self.executable {
            return Ok(());
        }
        // SAFETY: ptr/len describe our own live mapping.
        let rc = unsafe {
            libc::mprotect(
                self.ptr as *mut c_void,
                self.len,
                libc::PROT_READ | libc::PROT_EXEC,
            )
        };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        self.executable = true;
        Ok(())
    }

    /// Reinterpret the start of the region as a function pointer of type `F`
    /// (e.g. `unsafe extern "C" fn(i64, i64) -> i64`).
    ///
    /// # Safety
    /// The caller must guarantee: the memory has been made executable, the
    /// code written there really is a function matching signature `F`
    /// (including the System V AMD64 calling convention), and `F` is a
    /// function pointer type. The returned pointer must not be called after
    /// this `JitMemory` is dropped.
    pub unsafe fn function_ptr<F: Copy>(&self) -> F {
        assert!(self.executable, "JitMemory must be executable before use");
        assert_eq!(
            std::mem::size_of::<F>(),
            std::mem::size_of::<usize>(),
            "F must be a function pointer"
        );
        let addr = self.ptr as usize;
        // SAFETY: per the caller's contract F is a function pointer; we copy
        // the address bits into it.
        unsafe { std::mem::transmute_copy(&addr) }
    }
}

impl Drop for JitMemory {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr/len describe our own live mapping, freed once here.
            unsafe {
                libc::munmap(self.ptr as *mut c_void, self.len);
            }
            self.ptr = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_write_seal() {
        let mut mem = JitMemory::new(16).unwrap();
        assert!(mem.capacity() >= 4096);
        // mov eax, 42 ; ret  (B8 2A 00 00 00 C3) — 32-bit form is fine here.
        mem.write_code(&[0xB8, 42, 0, 0, 0, 0xC3]).unwrap();
        assert_eq!(mem.used(), 6);
        mem.make_executable().unwrap();
        assert!(mem.is_executable());
        // SAFETY: bytes above are exactly `fn() -> i32 { 42 }`.
        let f: unsafe extern "C" fn() -> i32 = unsafe { mem.function_ptr() };
        let v = unsafe { f() };
        assert_eq!(v, 42);
    }

    #[test]
    fn write_after_seal_is_rejected() {
        let mut mem = JitMemory::new(16).unwrap();
        mem.write_code(&[0xC3]).unwrap();
        mem.make_executable().unwrap();
        assert!(mem.write_code(&[0xC3]).is_err());
    }
}
