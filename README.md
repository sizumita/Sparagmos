# Sparagmos

A completely original JIT runtime based on Rust MIR, written in Rust itself.

Goal: execute Rust 1.100.0 at the 99% level — ultimately running `rustc` itself
and its test suite on top of Sparagmos. See `docs/DESIGN.md` for the
architecture and `docs/COVERAGE.md` for how the 99% figure is measured
(honestly, in parts, never as one blended number).

No existing code generator / JIT / VM is used for the core (no Cranelift, no
LLVM, no Wasmtime): Sparagmos emits x86_64 machine code itself.

## Layout

- `crates/sparagmos-mir` — MIR type definitions (our own model of rustc MIR)
- `crates/sparagmos-driver` — rustc driver that obtains MIR via Stable MIR (rustc_public) and converts it to our model; builds separately with the 1.100.0 toolchain
- `crates/sparagmos-interp` — MIR interpreter (correctness oracle)
- `crates/sparagmos-jit` — x86_64 baseline JIT (own assembler/emitter)
- `crates/sparagmos-runtime` — allocation, panic, intrinsics, std support
- `crates/sparagmos-cli` — the `sparagmos` driver binary
- `crates/sparagmos-harness` — corpus runner and coverage measurement
- `tests/corpus/` — `.rs` programs used as the first coverage corpus

Development happens on the `fleur-du-lapin` host; progress and feature
coverage are tracked in GitHub Issues.
