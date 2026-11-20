# sparagmos-runtime design notes

## Target version premise (corrected 2026-11-20)

- The project target is **Rust 1.100.0**, released 2026-11-12 and already
  installed and verified on the development host `fleur-du-lapin`
  (`rustc 1.100.0 (0ef1bc386 2026-11-09)`, toolchain `1.100.0-x86_64-unknown-linux-gnu`).
- All coverage denominators for Sparagmos are unified on 1.100.0.
  Do not treat 1.100.0 as a future release, and do not substitute
  1.99.1 (or any other version) as the reference denominator.

## Future scope: Allocator API (Rust 1.100.0)

- In Rust 1.100.0 the core of the **Allocator API** is stabilised:
  `std::alloc::Allocator`, and allocator-aware constructors such as
  `Box::new_in` / `Vec::new_in` (the `new_in` family).
- Supporting that API is **in the future target scope of this runtime lane**:
  JIT/interpreter execution of code that allocates through a custom
  `Allocator`, passes allocators into `Box`/`Vec`, and relies on the
  stabilised `Allocator` contract (allocate / deallocate / grow / shrink).
- Current status (first implementation, unchanged by this note): the runtime
  provides its own allocator layer in `src/alloc.rs` — the `Allocator` trait
  in this crate (alloc/dealloc separated, backend swappable) with a leaking
  bump `Arena` as v1 and `SystemAllocator` as the real-dealloc alternative —
  plus the `sparagmos_alloc` / `sparagmos_dealloc` extern "C" surface.
  This crate-local trait is **not yet** `std::alloc::Allocator`; bridging /
  implementing the stabilised std Allocator API for JIT-executed code, and
  handling `Box::new_in` / `Vec::new_in` MIR patterns, is a named next step,
  not implemented scope.
- Design constraint for that next step: keep the alloc/dealloc separation
  and the backend-swap point introduced in v1, so the std Allocator API can
  be layered on top without changing JIT call sites (`sparagmos_alloc` etc.).
