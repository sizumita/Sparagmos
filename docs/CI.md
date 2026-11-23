# CI and the pull-request workflow

All verification for this repository runs in GitHub Actions on
`ubuntu-latest`. Nothing is measured by hand on a dev host: the
official measurement is a workflow too.

## How work lands

1. A lane works on a branch, pushes it, and opens a pull request.
   Opening the PR is the lane's last step.
2. On the PR, two things report back:
   - the **CI** workflow's status checks (build, tests, corpus,
     script tests), and
   - the **Flip Diff** workflow's comment with the gain/loss result,
     when the lane asks for one (see below).
   A Codex review may also appear on the PR; it is enabled on the
   GitHub side, not by any workflow in this repository.
3. The integrator reviews the code, the CI results, and the Codex
   review, then merges PRs one at a time, in dependency order.
   Merged branches are deleted automatically.

## Workflows

### CI (`.github/workflows/ci.yml`) — per-PR verification

Runs on every pull request and on pushes to `main`.

- `build / test / fmt / clippy`: builds and tests the workspace
  (stable toolchain, plus Rust 1.100.0 for the harness's native
  oracle), runs the `scripts/` unit tests with pytest, and reports
  formatting and clippy as advisory (non-blocking) checks.
- `driver / corpus (Rust 1.100.0)`: builds and tests
  `sparagmos-driver` with Rust 1.100.0 + rustc-dev under
  `RUSTC_BOOTSTRAP=1`, builds the CLI, and runs the corpus
  measurement. The job fails unless every corpus case passes
  (metric 2(a), currently 47/47).

Both jobs appear as status checks on the PR.

### Flip Diff (`.github/workflows/flip-diff.yml`) — per-PR subset check

Compares the PR against its base over a topical population of the
metric-2(b) subset: the subset runner runs once with the base driver
and once with the head driver, and each record that flips is reported
as a gain or a loss (a record passes when its bucket is empty).

To request it, add a label named `flip-diff:<kw1+kw2+...>` to the PR
(the separator is `+` because label names cannot contain commas).
The keywords are matched as substrings of the test paths, so
`flip-diff:path+os-str+ffi` selects every record whose path mentions
one of them (a few hundred to a couple of thousand records is the
intended scale).

It can also be started manually from the Actions tab
(workflow_dispatch) with explicit keywords and a target: a PR number
(head and base are resolved from the PR), or a head ref plus a base
ref. The manual form measures branches that do not carry this
workflow file yet — the measurement scripts come from the ref the
workflow was dispatched on, and only the code under test comes from
the target ref. When a PR number is given, the result comment is
posted to that PR.

Results land in the job summary and as a comment on the PR. The job
does not fail on losses: some flips are known flakes, and whether a
loss blocks the merge is the integrator's call. Flip-diff runs are
serialised repository-wide (one at a time).

### Official Measurement (`.github/workflows/official-measure.yml`)

The full official measurement over the whole 24,937-record
denominator. Runs on every push to `main` and on manual dispatch.

- The survey (`scripts/rustc-subset-survey.py`) classifies the
  Rust 1.100.0 test suite; the source tree and the survey JSON are
  cached between runs.
- The survey is split into shards
  (`scripts/ci-subset-tools.py shard`, round-robin over path order so
  shards are balanced), and each shard runs as one matrix job with
  `scripts/rustc-subset-run.py`. A second matrix repeats the pass
  with `--jit` for metric 4.
- The aggregate job merges the shard results
  (`scripts/ci-subset-tools.py merge`; aggregates are recomputed from
  the merged records), derives metric 3 with
  `scripts/std-coverage.py`, and publishes metrics 2(b), 3, and 4 to
  the job summary. The merged result JSONs are kept as run artifacts.

## Hosted-runner calibration

Hosted runners are smaller and slower than the dev host (4 vCPU,
16 GB class), so the measurement workflows:

- run the subset runner with `--jobs 2`;
- raise the runner's per-record timeouts through the
  `SPARAGMOS_NATIVE_COMPILE_TIMEOUT`, `SPARAGMOS_NATIVE_RUN_TIMEOUT`,
  `SPARAGMOS_DRIVER_TIMEOUT`, and `SPARAGMOS_CONVERT_TIMEOUT`
  environment variables (the runner's built-in defaults are
  unchanged, so dev-host runs behave as before);
- free disk space before measuring and cache the Rust source tree,
  the survey, and cargo build outputs;
- give every job an explicit `timeout-minutes` below the 6-hour
  hosted limit.
