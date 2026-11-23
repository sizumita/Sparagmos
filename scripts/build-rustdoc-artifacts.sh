#!/bin/bash
# Build the MIR-carrying artifact tree that lets Sparagmos convert and
# run rustdoc itself (Issue #361).
#
# The distributed rustc-dev sysroot ships the compiler crates as
# metadata without MIR bodies for non-generic functions ("no MIR body
# available"), so the driver's Stable MIR session cannot convert any
# function of rustdoc's dependency closure from the sysroot builds.
# This script rebuilds the whole closure from the rust-lang/rust
# 1.100.0 sources with the same flags the driver's `compile_aux`
# already uses for auxiliary crates:
#
#     -Zalways-encode-mir -Zforce-unstable-if-unmarked
#     (under RUSTC_BOOTSTRAP=1)
#
# Build shape (two cargo invocations, one target directory):
#
#   Phase 1 builds the compiler-crate DAG (package `rustc_driver`,
#   which pulls in rustc_driver_impl and every compiler crate under
#   it) plus the crates.io dependencies of that DAG, all from the
#   vendored sources of the official source tarball, offline.
#
#   Phase 2 builds package `rustdoc` (librustdoc). librustdoc's
#   rustc_* dependencies are not Cargo dependencies -- its lib.rs
#   binds them with `extern crate` items that rustc normally resolves
#   from the sysroot. To bind the phase-1 builds instead, phase 2
#   runs rustc through a wrapper (generated into the build
#   directory) that, only for the compilation whose --crate-name is
#   exactly `rustdoc`, appends `--extern <name>=<phase-1 artifact>`
#   for every rustc_* crate plus `-L dependency=<flat dir>` holding
#   every phase-1 library artifact. The --extern flags bind the
#   direct `extern crate` items; the -L directory is what the crate
#   loader searches for the *transitive* dependencies recorded in
#   those crates' metadata (without it, the loader falls back to
#   the sysroot builds and aborts with E0460 hash mismatches).
#   Both phases use identical RUSTFLAGS and environment. A shared
#   crates.io dependency can still be built more than once (the two
#   graphs enable different feature sets, and a few crates exist in
#   two versions); the harvest keeps every distinct build, so each
#   recorded dependency hash resolves to an artifact in the tree.
#
# librustdoc reads DOC_RUST_LANG_ORG_CHANNEL through `env!`; for the
# stable 1.100.0 channel bootstrap sets it to
# https://doc.rust-lang.org/1.100.0 (see src/bootstrap's
# doc_rust_lang_org_channel), and so does this script. librustdoc's
# build.rs (the static-files .min/.sha256 generation) runs as part of
# the phase-2 cargo build, so the OUT_DIR products exist exactly as
# in a bootstrap build.
#
# Outputs (default locations; override with the environment):
#   RUSTDOC_ARTIFACTS_DIR  (default ~/rustdoc-artifacts-1.100.0)
#     the harvested .rlib/.so files, externs.txt (name=path lines
#     for the driver session) and MANIFEST.txt.
#   RUSTDOC_BUILD_DIR      (default ~/rustdoc-artifacts-build)
#     cargo target directory, per-phase artifact JSON, logs, the
#     generated wrapper and the phase-2 extern list. Deleting this
#     directory and re-running the script regenerates an identical
#     artifact tree.
#   RUST_SRC_DIR           (default ~/rust-src-1.100.0)
#     an extracted rustc-1.100.0-src.tar.xz tree (its vendor/
#     directory provides every crates.io source, offline).
#
# Usage: build-rustdoc-artifacts.sh [phase1|phase2|harvest|all]
# (default: all). CARGO_BUILD_JOBS defaults to 4.
set -euo pipefail

RUST_VERSION=1.100.0
TARGET_TRIPLE=x86_64-unknown-linux-gnu
SRC=${RUST_SRC_DIR:-$HOME/rust-src-1.100.0}
BUILD=${RUSTDOC_BUILD_DIR:-$HOME/rustdoc-artifacts-build}
OUT=${RUSTDOC_ARTIFACTS_DIR:-$HOME/rustdoc-artifacts-1.100.0}
TARGET_DIR=$BUILD/target
MIR_FLAGS="-Zalways-encode-mir -Zforce-unstable-if-unmarked"

export PATH=$HOME/.cargo/bin:$PATH
export RUSTC_BOOTSTRAP=1
export DOC_RUST_LANG_ORG_CHANNEL="https://doc.rust-lang.org/${RUST_VERSION}"
# The compiler crates read these through `env!`; bootstrap sets them
# in `rustc_cargo_env` (src/bootstrap/src/core/build_steps/
# compile.rs). Values mirror the official 1.100.0 release build:
# CFG_VERSION is the distributed `rustc --version` string, and
# CFG_VER_HASH / CFG_VER_DATE come from the source tree's
# git-commit-info file.
export CFG_RELEASE="$RUST_VERSION"
export CFG_RELEASE_CHANNEL="stable"
export CFG_VERSION="$(rustc +"$RUST_VERSION" --version | sed 's/^rustc //')"
export CFG_VER_HASH="$(sed -n '1p' "$SRC/git-commit-info")"
export CFG_VER_DATE="$(sed -n '3p' "$SRC/git-commit-info")"
export CFG_COMPILER_HOST_TRIPLE="$TARGET_TRIPLE"
export CFG_DEFAULT_CODEGEN_BACKEND="llvm"
export CFG_LIBDIR_RELATIVE="lib"
export CFG_DISABLE_UNSTABLE_FEATURES="1"
export RUSTC_INSTALL_BINDIR="bin"
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4}
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=0

CARGO_OFFLINE=(--offline
    --config "source.crates-io.replace-with=\"vendored-sources\""
    --config "source.vendored-sources.directory=\"$SRC/vendor\"")

write_wrapper() {
    # Pass-through rustc wrapper; see the header comment. The file
    # must not change between phase 1 and phase 2.
    cat > "$BUILD/rustc-wrapper.sh" <<'WRAPPER'
#!/bin/bash
# Generated by build-rustdoc-artifacts.sh. Invoked by cargo as
# `wrapper <rustc> <args...>`. Appends the phase-1 --extern flags
# listed in $SPARAGMOS_EXTERNS_FILE (one name=path per line) only
# when the crate being compiled is named exactly `rustdoc` and
# SPARAGMOS_INJECT_RUSTDOC_EXTERNS=1.
RUSTC_BIN="$1"; shift
if [ "${SPARAGMOS_INJECT_RUSTDOC_EXTERNS:-0}" = "1" ] && [ -n "${SPARAGMOS_EXTERNS_FILE:-}" ]; then
    prev=""; is_rustdoc=0
    for a in "$@"; do
        if [ "$prev" = "--crate-name" ] && [ "$a" = "rustdoc" ]; then is_rustdoc=1; fi
        prev="$a"
    done
    if [ "$is_rustdoc" = "1" ]; then
        extra=()
        while IFS= read -r line; do
            [ -n "$line" ] && extra+=(--extern "$line")
        done < "$SPARAGMOS_EXTERNS_FILE"
        if [ -n "${SPARAGMOS_L_DIR:-}" ]; then
            extra+=(-L "dependency=$SPARAGMOS_L_DIR")
        fi
        exec "$RUSTC_BIN" "$@" "${extra[@]}"
    fi
fi
exec "$RUSTC_BIN" "$@"
WRAPPER
    chmod +x "$BUILD/rustc-wrapper.sh"
}

phase1() {
    mkdir -p "$BUILD"
    write_wrapper
    export RUSTC_WRAPPER=$BUILD/rustc-wrapper.sh
    export RUSTFLAGS="$MIR_FLAGS"
    echo "phase1: cargo build -p rustc_driver (start $(date))"
    cargo +$RUST_VERSION build "${CARGO_OFFLINE[@]}" \
        --manifest-path "$SRC/Cargo.toml" \
        --target-dir "$TARGET_DIR" \
        -p rustc_driver \
        --message-format=json > "$BUILD/phase1-artifacts.json"
    echo "phase1: done $(date)"
}

gen_externs() {
    # From the phase-1 artifact JSON, list name=path for every
    # rustc_* library artifact (rlib or shared object). These are
    # the crates librustdoc binds with `extern crate` items.
    python3 - "$BUILD/phase1-artifacts.json" "$BUILD/phase2-externs.txt" <<'PYEOF'
import json, sys
seen = {}
with open(sys.argv[1]) as f:
    for line in f:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("reason") != "compiler-artifact":
            continue
        target = msg.get("target", {})
        name = target.get("name", "")
        if not name.startswith("rustc"):
            continue
        kinds = set(target.get("kind", []))
        if not (kinds & {"lib", "rlib", "dylib", "proc-macro"}):
            continue
        for path in msg.get("filenames", []):
            if path.endswith((".rlib", ".so")):
                seen[name] = path
with open(sys.argv[2], "w") as f:
    for name in sorted(seen):
        f.write(f"{name}={seen[name]}\n")
print(f"phase2 externs: {len(seen)} crates")
PYEOF
}

make_flatdir() {
    # A flat directory with a symlink to every phase-1 library
    # artifact. The crate loader resolves the transitive
    # dependencies recorded in a crate's metadata by searching
    # -L dependency directories for lib<name>-*.rlib/.so and
    # matching the recorded hash; the phase-1 artifacts live in
    # per-unit output directories, so they are gathered here once.
    python3 - "$BUILD/phase1-artifacts.json" "$BUILD/deps-flat" <<'PYEOF'
import json, os, sys
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
count = 0
with open(sys.argv[1]) as f:
    for line in f:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("reason") != "compiler-artifact" or msg.get("executable"):
            continue
        kinds = set(msg.get("target", {}).get("kind", []))
        if not (kinds & {"lib", "rlib", "dylib", "proc-macro"}):
            continue
        for path in msg.get("filenames", []):
            if path.endswith((".rlib", ".so")):
                dst = os.path.join(out, os.path.basename(path))
                if not os.path.exists(dst):
                    os.symlink(path, dst)
                    count += 1
print(f"deps-flat: {count} artifacts linked")
PYEOF
}

phase2() {
    write_wrapper
    gen_externs
    make_flatdir
    export RUSTC_WRAPPER=$BUILD/rustc-wrapper.sh
    export RUSTFLAGS="$MIR_FLAGS"
    export SPARAGMOS_INJECT_RUSTDOC_EXTERNS=1
    export SPARAGMOS_EXTERNS_FILE=$BUILD/phase2-externs.txt
    export SPARAGMOS_L_DIR=$BUILD/deps-flat
    echo "phase2: cargo build -p rustdoc (start $(date))"
    cargo +$RUST_VERSION build "${CARGO_OFFLINE[@]}" \
        --manifest-path "$SRC/Cargo.toml" \
        --target-dir "$TARGET_DIR" \
        -p rustdoc \
        --message-format=json > "$BUILD/phase2-artifacts.json"
    echo "phase2: done $(date)"
}

harvest() {
    python3 - "$BUILD/phase1-artifacts.json" "$BUILD/phase2-artifacts.json" "$OUT" "$RUST_VERSION" <<'PYEOF'
import json, os, shutil, sys
phase_files = sys.argv[1:3]
out = sys.argv[3]
version = sys.argv[4]
os.makedirs(out, exist_ok=True)
# Every distinct artifact file is harvested. A crate name can have
# several builds (the compiler graph and the rustdoc graph enable
# different feature sets of shared crates.io dependencies, and a
# few crates exist in two versions); each consumer's metadata
# records the exact build it was compiled against, and the crate
# loader disambiguates same-name candidates in the -L directory
# by that recorded hash. Collapsing to one build per name would
# lose builds that recorded hashes point at (E0460).
chosen = {}
for path in phase_files:
    with open(path) as f:
        for line in f:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("reason") != "compiler-artifact":
                continue
            if msg.get("executable"):
                continue
            target = msg.get("target", {})
            name = target.get("name", "")
            kinds = set(target.get("kind", []))
            if not (kinds & {"lib", "rlib", "dylib", "proc-macro"}):
                continue
            for fn in msg.get("filenames", []):
                if fn.endswith((".rlib", ".so")):
                    chosen[fn] = name
externs = []
total = 0
for src in sorted(chosen):
    base = os.path.basename(src)
    dst = os.path.join(out, base)
    # Hardlink when possible (same filesystem): the tree then costs
    # no extra space while the build target exists, and stays valid
    # after the target directory is removed.
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    total += os.path.getsize(dst)
    externs.append(f"{chosen[src]}={dst}")
with open(os.path.join(out, "externs.txt"), "w") as f:
    f.write("\n".join(sorted(externs)) + "\n")
with open(os.path.join(out, "MANIFEST.txt"), "w") as f:
    f.write(f"rustdoc artifact tree for Rust {version}\n")
    f.write(f"artifact files: {len(chosen)}\n")
    f.write(f"unique crate names: {len(set(chosen.values()))}\n")
    f.write(f"total bytes: {total}\n")
    f.write("built by scripts/build-rustdoc-artifacts.sh "
            "(-Zalways-encode-mir -Zforce-unstable-if-unmarked)\n")
print(f"harvested {len(chosen)} artifacts "
      f"({len(set(chosen.values()))} unique names), {total} bytes -> {out}")
PYEOF
}

case "${1:-all}" in
    phase1) phase1 ;;
    phase2) phase2 ;;
    harvest) harvest ;;
    all) phase1; phase2; harvest ;;
    *) echo "usage: $0 [phase1|phase2|harvest|all]" >&2; exit 2 ;;
esac
