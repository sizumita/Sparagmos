//! Executable memory management.
//!
//! NOTE on the `libc` dependency: `libc` is used ONLY here, and only for the
//! OS primitives that allocate pages and flip them between writable and
//! executable: `mmap` / `mprotect` / `munmap` / `sysconf`. On Windows the
//! corresponding primitives (`VirtualAlloc` / `VirtualProtect` /
//! `VirtualFree` / `GetSystemInfo`) are declared directly below, as
//! `libc`'s Windows bindings do not carry them. `libc` is an OS
//! foundation crate, not a code generator — all machine code bytes are
//! encoded by hand in [`crate::emitter`]. The mapping follows W^X: pages are
//! mapped RW, code is written, then they are flipped to RX before execution.

use std::ffi::c_void;
use std::ptr;

/// A page-aligned region of memory that holds JIT-compiled code.
///
/// Lifecycle: allocate (RW) -> write code -> `make_executable` (RX) -> call.
/// Freed on drop (`munmap`, or `VirtualFree` on Windows).
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

#[cfg(not(windows))]
fn page_size() -> usize {
    // SAFETY: sysconf with a valid name returns the page size (or -1).
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v > 0 {
        v as usize
    } else {
        4096
    }
}

/// The Windows memory primitives, declared directly:
/// `libc`'s Windows bindings do not carry them, and this
/// crate takes no `windows-sys` dependency for four calls.
#[cfg(windows)]
mod win32 {
    use std::ffi::c_void;

    pub const MEM_COMMIT: u32 = 0x1000;
    pub const MEM_RESERVE: u32 = 0x2000;
    pub const MEM_RELEASE: u32 = 0x8000;
    pub const PAGE_READWRITE: u32 = 0x04;
    pub const PAGE_EXECUTE_READ: u32 = 0x20;

    /// `SYSTEM_INFO`, up to the fields this module reads;
    /// the layout is the documented one.
    #[repr(C)]
    pub struct SystemInfo {
        processor_architecture: u16,
        reserved: u16,
        pub page_size: u32,
        minimum_application_address: *mut c_void,
        maximum_application_address: *mut c_void,
        active_processor_mask: usize,
        number_of_processors: u32,
        processor_type: u32,
        allocation_granularity: u32,
        processor_level: u16,
        processor_revision: u16,
    }

    extern "system" {
        pub fn VirtualAlloc(
            addr: *mut c_void,
            size: usize,
            alloc_type: u32,
            protect: u32,
        ) -> *mut c_void;
        pub fn VirtualProtect(
            addr: *mut c_void,
            size: usize,
            new_protect: u32,
            old_protect: *mut u32,
        ) -> i32;
        pub fn VirtualFree(addr: *mut c_void, size: usize, free_type: u32) -> i32;
        pub fn GetSystemInfo(info: *mut SystemInfo);
    }
}

#[cfg(windows)]
fn page_size() -> usize {
    // SAFETY: GetSystemInfo fills the whole struct it is given;
    // its page_size field is the host page size.
    let v = unsafe {
        let mut info = std::mem::zeroed::<win32::SystemInfo>();
        win32::GetSystemInfo(&mut info);
        info.page_size
    };
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
        #[cfg(not(windows))]
        let p = {
            // SAFETY: standard anonymous private mmap; result
            // checked below.
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
            p as *mut u8
        };
        #[cfg(windows)]
        let p = {
            // SAFETY: standard reserve-and-commit VirtualAlloc
            // of `len` bytes; result checked below.
            let p = unsafe {
                win32::VirtualAlloc(
                    ptr::null_mut(),
                    len,
                    win32::MEM_COMMIT | win32::MEM_RESERVE,
                    win32::PAGE_READWRITE,
                )
            };
            if p.is_null() {
                return Err(std::io::Error::last_os_error());
            }
            p as *mut u8
        };
        Ok(Self {
            ptr: p,
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
        #[cfg(not(windows))]
        {
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
        }
        #[cfg(windows)]
        {
            // SAFETY: ptr/len describe our own live allocation.
            let mut old = 0;
            let rc = unsafe {
                win32::VirtualProtect(
                    self.ptr as *mut c_void,
                    self.len,
                    win32::PAGE_EXECUTE_READ,
                    &mut old,
                )
            };
            if rc == 0 {
                return Err(std::io::Error::last_os_error());
            }
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
            #[cfg(not(windows))]
            {
                // SAFETY: ptr/len describe our own live mapping, freed once here.
                unsafe {
                    libc::munmap(self.ptr as *mut c_void, self.len);
                }
            }
            #[cfg(windows)]
            {
                // SAFETY: ptr describes our own live allocation,
                // released once here (MEM_RELEASE takes size 0).
                unsafe {
                    win32::VirtualFree(
                        self.ptr as *mut c_void,
                        0,
                        win32::MEM_RELEASE,
                    );
                }
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
