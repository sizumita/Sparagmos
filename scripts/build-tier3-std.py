#!/usr/bin/env python3
"""Build std from source for tier-3 targets the channel does not
distribute, and install the result into the pinned toolchain's
sysroot (Issue #516).

The subset runner screens a record's compile-flags --target
against the triples that have a std in the sysroot
(`installed_targets()` in rustc-subset-run.py scans
`<sysroot>/lib/rustlib/*/lib`). rustup distributes prebuilt std
for 119 triples of the 1.100.0 channel; the remaining known
triples are tier-3 targets with no prebuilt std at all, so every
record naming one stays "not attempted (compile-flags)" (158
records). For those triples the only source of a std the pinned
compiler will accept is the compiler itself: rustc refuses rlibs
whose metadata another compiler version produced, so the std
must be built by the pinned 1.100.0 compiler from its own
rust-src component. That is what `-Z build-std` does; the flag
is nightly-gated, so the build runs under RUSTC_BOOTSTRAP=1,
which is contained to this script's cargo invocation.

The build profile is `--release`: the distributed std is a
release build, and a debug std would change std's own
debug-assertion behaviour under the guest, which is a judging
difference, not just a size one.

Each table entry needs one green build-std run as its entry
ticket: per-triple buildability varies (full std for
m68k-unknown-linux-gnu, for example, fails inside LLVM codegen
with this compiler, so that triple stays blocked regardless of
this pipeline). Core-only triples (the `-none` family and
similar) have no std to build; `core,alloc` is their ceiling,
and records whose test files use std then fail at the guest
compile on their merits instead of staying unattempted.

Two targets (amdgcn-amd-amdhsa, avr-none) refuse to compile
anything at all without an explicit `-C target-cpu`, so their
entries also pin one -- the cpu the blocked records themselves
name in their compile-flags (see TIER3_TARGET_CPU below). A
record naming a different cpu for one of those triples then
fails at the guest compile with a target-cpu ABI mismatch
instead of staying unattempted: again a verdict on its merits,
not a screening artefact.

Usage: python3 scripts/build-tier3-std.py [--triple T] [--force]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLCHAIN = "1.100.0"

# triple -> the build-std crate set to build for it. "core" is in
# every set (build-std requires it as the root). Order inside a
# set does not matter; build-std resolves the dependency closure.
TIER3_STD = {
    "aarch64v8r-unknown-none": ("core", "alloc"),
    "aarch64-unknown-linux-pauthtest": ("core", "std", "panic_unwind"),
    "amdgcn-amd-amdhsa": ("core", "alloc"),
    "avr-none": ("core", "alloc"),
    "riscv32e-unknown-none-elf": ("core", "alloc"),
}

# triple -> the `-C target-cpu` its build-std run must carry.
# These targets have no default cpu: rustc refuses to compile
# even core without an explicit one. The pinned cpu is the one
# the records blocked by the triple themselves specify in their
# compile-flags -- gfx900 in 15 of the 16 amdgcn flag sets,
# atmega328p in 15 of the 17 avr ones (the exceptions are one
# gfx1100, one atmega328 and one attiny11 flag set; those
# records meet the ABI-mismatch error described above).
TIER3_TARGET_CPU = {
    "amdgcn-amd-amdhsa": "gfx900",
    "avr-none": "atmega328p",
}

PROBE_CRATE = "tier3-std-probe"


def rustup_cmd():
    rustup = Path.home() / ".cargo" / "bin" / "rustup"
    return str(rustup) if rustup.exists() else "rustup"


def toolchain_cmd(tool):
    return [rustup_cmd(), "run", TOOLCHAIN, tool]


def sysroot():
    out = subprocess.run(
        toolchain_cmd("rustc") + ["--print", "sysroot"],
        capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"rustc --print sysroot failed: {out.stderr}")
    return Path(out.stdout.strip())


def probe_source(crates):
    """The probe crate body: it must name the top crate of the set
    so the build-std closure is exercised end to end."""
    if "std" in crates:
        return ("pub fn probe() -> usize {\n"
                "    std::vec::Vec::<u8>::new().len()\n"
                "}\n")
    if "alloc" in crates:
        return ("#![no_std]\n"
                "extern crate alloc;\n"
                "use alloc::vec::Vec;\n"
                "pub fn probe(v: &mut Vec<u32>) {\n"
                "    v.push(1);\n"
                "}\n")
    return ("#![no_std]\n"
            "pub fn probe(x: u32) -> u32 {\n"
            "    x.wrapping_add(1)\n"
            "}\n")


def collect_rlibs(build_root, triple):
    """The rlibs cargo's build-std layout leaves under
    target/<triple>/release/build/<crate>/<hash>/out/, excluding
    the probe crate's own output."""
    out_dirs = (build_root / triple / "release" / "build").glob("*/**/out")
    rlibs = []
    for out in out_dirs:
        if out.parent.parent.name == PROBE_CRATE:
            continue
        rlibs.extend(out.glob("*.rlib"))
    return sorted(rlibs)


def target_cpu_env(triple):
    """The extra cargo env pinning `-C target-cpu` for the
    triples in TIER3_TARGET_CPU, scoped to that target only via
    CARGO_TARGET_<TRIPLE>_RUSTFLAGS, so host-side compilations
    in the same cargo run are unaffected."""
    cpu = TIER3_TARGET_CPU.get(triple)
    if cpu is None:
        return {}
    var = "CARGO_TARGET_" + triple.upper().replace("-", "_") \
        + "_RUSTFLAGS"
    return {var: f"-C target-cpu={cpu}"}


def installed_core(sysroot_dir, triple):
    lib = sysroot_dir / "lib" / "rustlib" / triple / "lib"
    return lib.is_dir() and any(lib.glob("libcore-*.rlib"))


def build_and_install(triple, crates, sysroot_dir, force=False):
    dest = sysroot_dir / "lib" / "rustlib" / triple / "lib"
    if not force and installed_core(sysroot_dir, triple):
        print(f"{triple}: already installed in sysroot, skipping")
        return 0
    with tempfile.TemporaryDirectory(prefix="tier3-std-") as td:
        work = Path(td)
        src = work / "src"
        src.mkdir()
        (work / "Cargo.toml").write_text(
            "[package]\n"
            f"name = \"{PROBE_CRATE}\"\n"
            "version = \"0.0.0\"\n"
            "edition = \"2021\"\n")
        (src / "lib.rs").write_text(probe_source(crates))
        env = dict(os.environ, RUSTC_BOOTSTRAP="1")
        env.update(target_cpu_env(triple))
        cmd = toolchain_cmd("cargo") + [
            "build", "--release",
            "-Z", "build-std=" + ",".join(crates),
            "--target", triple,
            "--manifest-path", str(work / "Cargo.toml"),
        ]
        print(f"{triple}: building {','.join(crates)} from source "
              f"(build-std, {TOOLCHAIN})")
        run = subprocess.run(cmd, env=env, capture_output=True,
                             text=True, timeout=3600)
        if run.returncode != 0:
            print(run.stdout)
            print(run.stderr, file=sys.stderr)
            raise RuntimeError(f"build-std failed for {triple}")
        rlibs = collect_rlibs(work / "target", triple)
        if not rlibs:
            raise RuntimeError(
                f"build-std for {triple} produced no rlibs")
        dest.mkdir(parents=True, exist_ok=True)
        for rlib in rlibs:
            shutil.copy2(rlib, dest / rlib.name)
        print(f"{triple}: installed {len(rlibs)} rlibs into {dest}")
        return len(rlibs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triple", help="build only this table entry")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if a std is already installed")
    args = ap.parse_args()
    entries = dict(TIER3_STD)
    if args.triple:
        if args.triple not in entries:
            print(f"{args.triple}: not in the tier-3 table",
                  file=sys.stderr)
            return 2
        entries = {args.triple: entries[args.triple]}
    root = sysroot()
    for triple, crates in entries.items():
        build_and_install(triple, crates, root, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
