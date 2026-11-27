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
record naming one stays "not attempted (compile-flags)". The
table below covers every triple that blocks more than one
record: at the latest official measurement before the fourth
increment, 57 compile-check records were still blocked by
triples outside the table, and that increment's 43 entries make
51 of them attemptable. The six records that remain blocked
name xtensa-esp32-none-elf (three; that backend cannot build
even core, see below) or are all-targets census tests naming
~200 triples apiece, most of them one-offs. For those triples the only source of a std the pinned
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
ticket: per-triple buildability varies. Full std for
m68k-unknown-linux-gnu fails inside LLVM codegen with this
compiler (`Cannot select` in std's main_thread), so that
triple is registered with the `core,alloc` set that does
build; xtensa-esp32-none-elf fails the same way inside
compiler_builtins, so not even core builds for it and it
stays blocked regardless of this pipeline. Core-only triples
(the `-none` family and similar) have no std to build;
`core,alloc` is their ceiling, and records whose test files
use std then fail at the guest compile on their merits
instead of staying unattempted. The OS triples in the table
(the linux gnu/musl/uclibc family across aarch64, csky,
mips/mipsel/mips64/mips64el and the mipsisa r6 variants,
sparc linux, hexagon musl, powerpc gnuspe and powerpc64
AIX, riscv32gc linux, fuchsia, hermit, the Apple
ios/watchos/darwin triples, the uwp windows-gnu pair,
wasm64) carry the full `core,std,panic_unwind` set, matching
what a distributed std for those targets would provide. Apart from the two targets
called out below, no entry needs an explicit `-C target-cpu`
to build: each entry's ticket build ran without one.

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
    "aarch64-unknown-linux-pauthtest": ("core", "std", "panic_unwind"),
    "aarch64-unknown-trusty": ("core", "alloc"),
    "aarch64_be-unknown-linux-gnu": ("core", "std", "panic_unwind"),
    "aarch64v8r-unknown-none": ("core", "alloc"),
    "aarch64v8r-unknown-none-softfloat": ("core", "alloc"),
    "amdgcn-amd-amdhsa": ("core", "alloc"),
    "armebv7r-none-eabi": ("core", "alloc"),
    "armebv7r-none-eabihf": ("core", "alloc"),
    "armv5te-none-eabi": ("core", "alloc"),
    "armv7k-apple-watchos": ("core", "std", "panic_unwind"),
    "armv7s-apple-ios": ("core", "std", "panic_unwind"),
    "avr-none": ("core", "alloc"),
    "bpfeb-unknown-none": ("core", "alloc"),
    "bpfel-unknown-none": ("core", "alloc"),
    "csky-unknown-linux-gnuabiv2": ("core", "std", "panic_unwind"),
    "hexagon-unknown-linux-musl": ("core", "std", "panic_unwind"),
    "i386-apple-ios": ("core", "std", "panic_unwind"),
    "i686-apple-darwin": ("core", "std", "panic_unwind"),
    "i686-uwp-windows-gnu": ("core", "std", "panic_unwind"),
    "m68k-unknown-linux-gnu": ("core", "alloc"),
    "m68k-unknown-none-elf": ("core", "alloc"),
    "mips-mti-none-elf": ("core", "alloc"),
    "mips-unknown-linux-gnu": ("core", "std", "panic_unwind"),
    "mips-unknown-linux-musl": ("core", "std", "panic_unwind"),
    "mips-unknown-linux-uclibc": ("core", "std", "panic_unwind"),
    "mips64-openwrt-linux-musl": ("core", "std", "panic_unwind"),
    "mips64-unknown-linux-gnuabi64": ("core", "std", "panic_unwind"),
    "mips64-unknown-linux-muslabi64": ("core", "std", "panic_unwind"),
    "mips64el-unknown-linux-gnuabi64": ("core", "std", "panic_unwind"),
    "mips64el-unknown-linux-muslabi64": ("core", "std", "panic_unwind"),
    "mipsel-mti-none-elf": ("core", "alloc"),
    "mipsel-sony-psp": ("core", "alloc"),
    "mipsel-sony-psx": ("core", "alloc"),
    "mipsel-unknown-linux-gnu": ("core", "std", "panic_unwind"),
    "mipsel-unknown-linux-musl": ("core", "std", "panic_unwind"),
    "mipsel-unknown-linux-uclibc": ("core", "std", "panic_unwind"),
    "mipsel-unknown-netbsd": ("core", "std", "panic_unwind"),
    "mipsel-unknown-none": ("core", "alloc"),
    "mipsisa32r6-unknown-linux-gnu": ("core", "std", "panic_unwind"),
    "mipsisa32r6el-unknown-linux-gnu": ("core", "std", "panic_unwind"),
    "mipsisa64r6-unknown-linux-gnuabi64": ("core", "std", "panic_unwind"),
    "mipsisa64r6el-unknown-linux-gnuabi64": ("core", "std", "panic_unwind"),
    "msp430-none-elf": ("core", "alloc"),
    "powerpc-unknown-linux-gnuspe": ("core", "std", "panic_unwind"),
    "powerpc64-ibm-aix": ("core", "std", "panic_unwind"),
    "riscv32e-unknown-none-elf": ("core", "alloc"),
    "riscv32em-unknown-none-elf": ("core", "alloc"),
    "riscv32emc-unknown-none-elf": ("core", "alloc"),
    "riscv32gc-unknown-linux-gnu": ("core", "std", "panic_unwind"),
    "riscv32gc-unknown-linux-musl": ("core", "std", "panic_unwind"),
    "riscv64gc-unknown-fuchsia": ("core", "std", "panic_unwind"),
    "s390x-unknown-none-softfloat": ("core", "alloc"),
    "sparc-unknown-linux-gnu": ("core", "std", "panic_unwind"),
    "sparc-unknown-none-elf": ("core", "alloc"),
    "thumbv4t-none-eabi": ("core", "alloc"),
    "thumbv5te-none-eabi": ("core", "alloc"),
    "wasm64-unknown-unknown": ("core", "std", "panic_unwind"),
    "x86_64-unknown-hermit": ("core", "std", "panic_unwind"),
    "x86_64-uwp-windows-gnu": ("core", "std", "panic_unwind"),
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
