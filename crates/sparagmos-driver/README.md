# sparagmos-driver

The Sparagmos rustc driver — the project's only MIR input path.

It compiles a `.rs` file with the pinned **Rust 1.100.0** toolchain,
obtains MIR through **Stable MIR** (`rustc_public`), converts it into
`sparagmos_mir::Body`, and prints a per-function summary. Functions that
contain constructs the converter does not handle yet are reported as
SKIPPED with per-variant counts — nothing is silently dropped.

This crate is a **separate workspace** (see the `[workspace]` table in its
`Cargo.toml` and the `exclude` entry in the root `Cargo.toml`): it must be
built with the 1.100.0 toolchain while the rest of Sparagmos builds with
the regular stable toolchain. The root workspace build/test is unaffected
by this crate.

## Build

Requires the `rustc-dev` component of the 1.100.0 toolchain
(`rustup component add rustc-dev --toolchain 1.100.0`).

```sh
cd crates/sparagmos-driver
RUSTC_BOOTSTRAP=1 cargo +1.100.0 build
RUSTC_BOOTSTRAP=1 cargo +1.100.0 test
```

`RUSTC_BOOTSTRAP=1` is needed because `#![feature(rustc_private)]` is a
feature gate and 1.100.0 is a release toolchain.

The build script captures the building rustc's sysroot and

- adds `<sysroot>/lib` to the binary's rpath and linker search path (for
  `librustc_driver` / `libLLVM`), and
- bakes the sysroot in as the default `--sysroot` for the compilation the
  driver runs (override with `--sysroot <path>` or `$SPARAGMOS_SYSROOT`).

So the built binary always drives the 1.100.0 compiler, regardless of the
caller's `PATH` or default toolchain.

## Run

```sh
./target/debug/sparagmos-driver <file.rs> [--dump <path>] [--edition 2021]
```

Example (from the repository root):

```sh
crates/sparagmos-driver/target/debug/sparagmos-driver tests/corpus/arith_int.rs
```

Output: one line per local function (`[converted]` with block/local/arg
counts, or `SKIPPED` with reasons), totals, and an `unsupported:` table
aggregating the unhandled variants — this feeds the project's coverage
measurement.

`--dump <path>` writes the converted bodies in the crate's text dump
format v0 (see `src/dump.rs`); the harness will read it in a later lane.

## Notes / limitations (v0)

- `Instance::body()` returns the monomorphized, optimized MIR — the same
  body a codegen backend would consume.
- Generic functions cannot be monomorphized without concrete arguments
  and are reported as skipped.
- Sparagmos `DefId`s are assigned sequentially within a run (Stable MIR
  does not expose rustc's raw `(krate, index)` pair); the dump lists the
  name for each assigned id.
- Scalar constants are stored with a fixed size of 8 bytes for now
  (the constant's own type is not tracked yet).
