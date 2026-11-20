//! Allocator layer: alloc/dealloc separated so the backend can be swapped.
//! First backend: bump Arena (leaks by design; dealloc is a no-op but present).
//! Second backend: SystemAllocator delegating to std::alloc (real dealloc).
use crate::error::{RuntimeError, RuntimeResult};
use crate::layout::Layout;
use std::alloc as sys;
use std::ptr::NonNull;
use std::sync::Mutex;

pub trait Allocator: Send {
    /// # Safety: layout must be valid; returned ptr is uninitialised.
    unsafe fn alloc(&mut self, layout: Layout) -> RuntimeResult<NonNull<u8>>;
    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> RuntimeResult<NonNull<u8>> {
        let p = unsafe { self.alloc(layout)? };
        unsafe {
            std::ptr::write_bytes(p.as_ptr(), 0, layout.size);
        }
        Ok(p)
    }
    /// # Safety: ptr must come from this allocator with same layout.
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout);
    fn used(&self) -> usize {
        0
    }
}

const CHUNK: usize = 64 * 1024;
struct Chunk {
    ptr: *mut u8,
    cap: usize,
    used: usize,
}
pub struct Arena {
    chunks: Vec<Chunk>,
    current: usize,
}
impl Arena {
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            current: 0,
        }
    }
    pub fn total_used(&self) -> usize {
        self.chunks.iter().map(|c| c.used).sum()
    }
    pub fn reset(&mut self) {
        for c in &mut self.chunks {
            c.used = 0;
        }
        self.current = 0;
    }
    fn grow(&mut self, min: usize) -> RuntimeResult<()> {
        let cap = min.max(CHUNK);
        let lay = sys::Layout::from_size_align(cap, 16).map_err(|_| RuntimeError::AllocFailed {
            size: cap,
            align: 16,
        })?;
        let ptr = unsafe { sys::alloc(lay) };
        if ptr.is_null() {
            return Err(RuntimeError::AllocFailed {
                size: cap,
                align: 16,
            });
        }
        self.chunks.push(Chunk { ptr, cap, used: 0 });
        self.current = self.chunks.len() - 1;
        Ok(())
    }
}
impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for Arena {
    fn drop(&mut self) {
        for c in &self.chunks {
            let lay = sys::Layout::from_size_align(c.cap, 16).unwrap();
            unsafe {
                sys::dealloc(c.ptr, lay);
            }
        }
    }
}
impl Allocator for Arena {
    unsafe fn alloc(&mut self, layout: Layout) -> RuntimeResult<NonNull<u8>> {
        if layout.size == 0 {
            return NonNull::new(layout.align as *mut u8).ok_or(RuntimeError::AllocFailed {
                size: 0,
                align: layout.align,
            });
        }
        loop {
            if self.chunks.is_empty() {
                self.grow(layout.size)?;
            }
            let idx = self.current;
            let (base, cap, used) = {
                let c = &self.chunks[idx];
                (c.ptr, c.cap, c.used)
            };
            let start = unsafe { base.add(used) } as usize;
            let aligned = crate::layout::align_up(start, layout.align);
            let offset = aligned - base as usize;
            if offset.checked_add(layout.size).map_or(false, |e| e <= cap) {
                self.chunks[idx].used = offset + layout.size;
                return NonNull::new(aligned as *mut u8).ok_or(RuntimeError::AllocFailed {
                    size: layout.size,
                    align: layout.align,
                });
            }
            self.grow(layout.size)?;
        }
    }
    unsafe fn dealloc(&mut self, _ptr: NonNull<u8>, _layout: Layout) { /* leak by design: arena */
    }
    fn used(&self) -> usize {
        self.total_used()
    }
}

pub struct SystemAllocator;
impl Allocator for SystemAllocator {
    unsafe fn alloc(&mut self, layout: Layout) -> RuntimeResult<NonNull<u8>> {
        if layout.size == 0 {
            return NonNull::new(layout.align as *mut u8).ok_or(RuntimeError::AllocFailed {
                size: 0,
                align: layout.align,
            });
        }
        let lay = sys::Layout::from_size_align(layout.size, layout.align).map_err(|_| {
            RuntimeError::AllocFailed {
                size: layout.size,
                align: layout.align,
            }
        })?;
        let p = unsafe { sys::alloc(lay) };
        NonNull::new(p).ok_or(RuntimeError::AllocFailed {
            size: layout.size,
            align: layout.align,
        })
    }
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size == 0 {
            return;
        }
        if let Ok(lay) = sys::Layout::from_size_align(layout.size, layout.align) {
            unsafe {
                sys::dealloc(ptr.as_ptr(), lay);
            }
        }
    }
}

/// Global arena used by the extern "C" entry points.
pub static GLOBAL_ARENA: Mutex<Option<Arena>> = Mutex::new(None);
pub fn with_global_arena<R>(f: impl FnOnce(&mut Arena) -> R) -> R {
    let mut g = GLOBAL_ARENA.lock().expect("arena poisoned");
    f(g.get_or_insert_with(Arena::new))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arena_read_write() {
        let mut a = Arena::new();
        let lay = Layout { size: 16, align: 8 };
        unsafe {
            let p = a.alloc(lay).unwrap();
            std::ptr::write(p.as_ptr() as *mut u64, 0xdeadbeef);
            assert_eq!(std::ptr::read(p.as_ptr() as *mut u64), 0xdeadbeef);
            a.dealloc(p, lay);
        }
        assert!(a.used() >= 16);
    }
    #[test]
    fn arena_alignment() {
        let mut a = Arena::new();
        unsafe {
            let p1 = a.alloc(Layout { size: 1, align: 1 }).unwrap();
            let p2 = a.alloc(Layout { size: 8, align: 8 }).unwrap();
            assert_eq!(p2.as_ptr() as usize % 8, 0);
            let _ = p1;
        }
    }
    #[test]
    fn arena_large_and_zeroed() {
        let mut a = Arena::new();
        unsafe {
            let p = a
                .alloc_zeroed(Layout {
                    size: 200_000,
                    align: 8,
                })
                .unwrap();
            assert_eq!(*p.as_ptr(), 0);
            assert_eq!(*p.as_ptr().add(199_999), 0);
        }
    }
    #[test]
    fn system_alloc_roundtrip() {
        let mut s = SystemAllocator;
        unsafe {
            let lay = Layout { size: 32, align: 8 };
            let p = s.alloc_zeroed(lay).unwrap();
            std::ptr::write(p.as_ptr() as *mut u64, 42);
            assert_eq!(std::ptr::read(p.as_ptr() as *mut u64), 42);
            s.dealloc(p, lay);
        }
    }
    #[test]
    fn aggregate_bytes() {
        // simulate array/tuple/string storage
        let mut a = Arena::new();
        unsafe {
            let lay = Layout { size: 5, align: 1 };
            let p = a.alloc(lay).unwrap();
            std::ptr::copy_nonoverlapping(b"hello".as_ptr(), p.as_ptr(), 5);
            assert_eq!(std::slice::from_raw_parts(p.as_ptr(), 5), b"hello");
        }
    }
}
// Arena owns its chunks exclusively (guarded by &mut self / Mutex at the global site).
unsafe impl Send for Arena {}
