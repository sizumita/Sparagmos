#!/usr/bin/env bash
# coverage-report.sh — per-crate implementation-marker and test counts.
#
# For every crate in this Cargo workspace, count:
#   * TODO / FIXME / XXX markers
#   * unimplemented!() / todo!() call sites
#   * `#[test]` functions (unit + integration tests under the crate)
#   * source lines (as a rough size signal)
# and print a Markdown table. This is a *marker* report, not the Sparagmos
# coverage definition itself — see docs/COVERAGE.md for the three real
# metrics (MIR variant coverage / corpus pass rate / std feature coverage).
#
# Usage: scripts/coverage-report.sh [workspace-root]
# Output: Markdown on stdout (suitable for $GITHUB_STEP_SUMMARY / Issues).

set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

count() {
    # count <pattern> <paths...> ; prints 0 instead of failing on no match
    # (grep exits 1 on no match; with `set -o pipefail` that would kill the
    # script, so the grep is explicitly allowed to fail here)
    local pattern="$1"; shift
    { grep -rE --include='*.rs' -- "$pattern" "$@" 2>/dev/null || true; } | wc -l | tr -d ' '
}

crate_dirs=()
if [ -f Cargo.toml ] && grep -q '^\[workspace\]' Cargo.toml; then
    # Workspace members are listed as "crates/<name>" in Cargo.toml.
    while IFS= read -r member; do
        [ -n "$member" ] && crate_dirs+=("$member")
    done < <(grep -oE '"crates/[^"]+"' Cargo.toml | tr -d '"' | sort -u)
fi
# Fallback: any directory under crates/ that has a Cargo.toml.
if [ "${#crate_dirs[@]}" -eq 0 ] && [ -d crates ]; then
    for d in crates/*/; do
        [ -f "${d}Cargo.toml" ] && crate_dirs+=("${d%/}")
    done
fi

echo "# Sparagmos coverage marker report"
echo
echo "Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')  "
echo "Workspace: \`$ROOT\`"
echo
echo "| Crate | Source files | LOC | TODO/FIXME | unimplemented!/todo! | Tests (\`#[test]\`) |"
echo "|---|---:|---:|---:|---:|---:|"

total_files=0 total_loc=0 total_todo=0 total_unimpl=0 total_tests=0

for crate in "${crate_dirs[@]}"; do
    name="$(basename "$crate")"
    src_paths=()
    [ -d "$crate/src" ] && src_paths+=("$crate/src")
    [ -d "$crate/tests" ] && src_paths+=("$crate/tests")
    if [ "${#src_paths[@]}" -eq 0 ]; then
        echo "| $name | 0 | 0 | 0 | 0 | 0 |"
        continue
    fi
    files=$(find "${src_paths[@]}" -name '*.rs' -type f | wc -l | tr -d ' ')
    loc=$(find "${src_paths[@]}" -name '*.rs' -type f -exec cat {} + 2>/dev/null | wc -l | tr -d ' ')
    todos=$(count 'TODO|FIXME|XXX' "${src_paths[@]}")
    unimpl=$(count 'unimplemented!|todo!' "${src_paths[@]}")
    tests=$(count '#\[test\]' "${src_paths[@]}")
    echo "| $name | $files | $loc | $todos | $unimpl | $tests |"
    total_files=$((total_files + files))
    total_loc=$((total_loc + loc))
    total_todo=$((total_todo + todos))
    total_unimpl=$((total_unimpl + unimpl))
    total_tests=$((total_tests + tests))
done

# Workspace-level integration tests (tests/ at the repo root), if any.
root_tests=0
if [ -d tests ]; then
    root_tests=$(count '#\[test\]' tests)
fi

echo "| **TOTAL (crates)** | **$total_files** | **$total_loc** | **$total_todo** | **$total_unimpl** | **$total_tests** |"
echo
echo "- Integration \`#[test]\` functions under top-level \`tests/\`: $root_tests"
echo "- Marker counts are a progress signal only; the authoritative coverage"
echo "  definition lives in \`docs/COVERAGE.md\` (three separate metrics)."
