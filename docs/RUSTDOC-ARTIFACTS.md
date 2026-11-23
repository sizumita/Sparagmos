# Rustdoc artifact tree

The rustdoc category of the rustc test suite needs `sparagmos-driver`
to convert functions across the whole `librustdoc` dependency
closure. That does not work against the installed sysroot: the
shipped compiler crates carry no MIR bodies for non-generic,
non-`#[inline]` functions. Measured on the sysroot build of
`rustc_lexer`:

```
fn probe_lexer::rustc_lexer::is_id_start: SKIPPED (no MIR body available x1)
```

The artifact tree is that closure rebuilt from the matching Rust
source with every function body encoded, so conversion can cross
crate boundaries the same way it does inside one crate.

## Contents

Built for Rust 1.100.0 (`x86_64-unknown-linux-gnu`):

- 312 artifact files, 273 unique crate names, 1,877,402,672 bytes
  in total (`MANIFEST.txt` in the tree carries the same numbers,
  and `externs.txt` lists every artifact as `name=path`).
- The tree holds `librustdoc.rlib`, `librustc_driver.so`, every
  `rustc_*` compiler crate, and the full crates.io dependency
  closure of both — including the six crates the sysroot could
  not supply for the scoping build (`askama`, `minifier`,
  `proc-macro2`, `stringdex`, `threadpool`,
  `unicode-segmentation`).
- A crate name can appear as several builds: the compiler graph
  and the rustdoc graph enable different feature sets of shared
  crates.io dependencies, and a few crates exist in two versions.
  Every distinct build is kept, because each consumer's metadata
  records the exact build it was compiled against and the crate
  loader disambiguates same-name candidates in the `-L` directory
  by that recorded hash. Collapsing the tree to one build per
  name loses builds that recorded hashes point at, and loading
  then fails with E0460.

The tree lives outside the repository at
`~/rustdoc-artifacts-1.100.0/` on the build host; only the build
script and this document are in the repository.

## Regenerating

```
scripts/build-rustdoc-artifacts.sh all   # phase1 + phase2 + harvest
```

Prerequisites: the `1.100.0` toolchain with the `rustc-dev` and
`rust-src` components, and the full extracted Rust source tree
(the script reads `RUST_SRC_DIR`, default `~/rust-src-1.100.0`,
and writes the tree to `RUSTDOC_ARTIFACTS_DIR`, default
`~/rustdoc-artifacts-1.100.0`). Re-running from a clean state
reproduces the same tree: the sorted artifact file list and
`MANIFEST.txt` are identical across clean rebuilds. The harvest
step can also be re-run over an existing tree: artifacts already
in place are skipped and stale ones are replaced.

How it works:

- Everything is compiled with
  `RUSTFLAGS="-Zalways-encode-mir -Zforce-unstable-if-unmarked"`,
  so every function — not just generic or `#[inline]` ones — has
  its MIR body encoded in the artifact metadata.
- The compiler crates read bootstrap values through `env!`
  (`CFG_RELEASE`, `CFG_VER_HASH`, `RUSTC_INSTALL_BINDIR`, ...).
  The script exports the same set rustc's bootstrap passes to
  cargo; without it the build fails inside `rustc_span` /
  `rustc_interface`.
- Dependencies resolve offline against the source tree's own
  `vendor/` directory at the versions its `Cargo.lock` pins.
- Phase 1 builds `rustc_driver`, which produces
  `librustc_driver.so` and every `rustc_*` rlib.
- Phase 2 builds the `rustdoc` package. `librustdoc`'s `rustc_*`
  dependencies are not Cargo dependencies — its `lib.rs` binds
  them with `extern crate` items that normally resolve from the
  sysroot — so a generated `RUSTC_WRAPPER` injects
  `--extern <name>=<phase-1 artifact>` for the compiler crates
  into the `rustdoc` crate's own rustc invocation only, plus
  `-L dependency=<dir of all phase-1 artifacts>` so the
  transitive dependencies recorded in those artifacts resolve to
  the phase-1 builds by hash. (`-L` alone does not override the
  sysroot for a direct `extern crate` binding; the explicit
  `--extern` is what swaps the direct binding.)
- The harvest collects every artifact file named by the two
  cargo builds' JSON output into the flat output tree
  (hardlinked when on the same filesystem).

## Using the tree with the driver

Point the session's direct externs at the tree and put the tree
on the dependency search path; transitive dependencies then
resolve inside the tree by recorded hash:

```
sparagmos-driver run src/tools/rustdoc/main.rs --edition 2024 \
  --rustc-arg --extern \
  --rustc-arg rustdoc=$HOME/rustdoc-artifacts-1.100.0/librustdoc.rlib \
  --rustc-arg --extern \
  --rustc-arg rustc_driver=$HOME/rustdoc-artifacts-1.100.0/librustc_driver.so \
  --rustc-arg -L \
  --rustc-arg dependency=$HOME/rustdoc-artifacts-1.100.0 \
  --guest-arg --version
```

## Conversion status

With this tree, a driver session on the rustdoc entry compiles
the entry crate against the tree (no E0460, no missing-body
failures at the crate boundary) and converts the reachable
closure across `librustdoc`, the compiler crates and the
crates.io dependencies. Measured on the entry
(`src/tools/rustdoc/main.rs`, convert mode): 2,650 functions
converted, 2,503 skipped. The skip reasons are the driver's own
collection cap (`collection limit reached`, 2,299 — the
per-owner and instantiation-size limits in the collector),
interpreter builtins (150), dynamic dispatch targets (24),
allocated constants (2), and 28 functions still without a MIR
body.

The prerequisite this tree removes is the missing-MIR-body wall
at the sysroot boundary. The probe from the conversion scoping —
a minimal program calling `rustc_lexer::is_id_start`, which
skips against the sysroot — converts `is_id_start` from the
tree build and reaches the first real converter blocker, the
by-reference constant in `unicode_ident`'s XID tables:

```
sparagmos-driver run: unsupported: call to `unicode_ident::is_xid_start`, whose body could not be converted: unsupported constant: by-reference constant
```

That blocker and the ones behind it belong to the follow-up
converter issues, not to this tree.
