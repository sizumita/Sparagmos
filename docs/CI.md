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
  `scripts/rustc-subset-run.py`. The metric-2(b) pass runs the
  runner with `--stages all`: stages 1 and 2 plus the stage-3
  procedure harness, which attempts the `procedure` records
  (run-make recipes — each recipe runs natively as the oracle, then
  under the driver with tool interception; the cargo suites
  `run-make-cargo` and `build-std` are out of stage-3 scope and
  recorded as not attempted, because their recipes drive cargo
  itself and cargo orchestration — workspace resolution,
  build-script and proc-macro processes, and the real compiler
  artifacts cargo consumes — is not modelled by the harness). A second matrix repeats
  the pass with `--jit` for metric 4; the JIT pass does not attempt
  procedure records (the runner never combines stage 3 with
  `--jit`), so they remain in metric 4's denominator as not
  attempted.
- The `rustdoc-check` records (1,735) are not attempted by any
  stage yet. The rustdoc-html (802), rustdoc-ui (440), and
  rustdoc-json (194) suites wait for the runner's rustdoc
  stage (issue #364) and keep the generic out-of-scope bucket
  until it dispatches them. The remaining 299 -- rustdoc-js
  (79), rustdoc-js-std (68), and rustdoc-gui (152) -- are
  judged by host-side tooling even in compiletest itself: the
  js suites run the test's paired .js assertions under
  Node.js (`src/tools/rustdoc-js/tester.js`) against the
  search index the tested rustdoc generated, and the gui
  suite drives a headless browser through .goml scripts
  (the browser-ui-test npm package, installed with yarn and
  run under Node.js). The measurement workflow provisions
  none of that tooling (no Node.js setup, no browser, no npm
  install), and the judged artifacts must be generated by
  rustdoc running under Sparagmos in any case, so the runner
  records these suites as not attempted under their own
  buckets -- `not attempted (host JavaScript engine)` and
  `not attempted (host browser)` (issue #365). This records
  the current harness, not an unreachability claim: the
  suites become attemptable once the rustdoc execution chain
  produces the artifacts and the host tooling is wired in.
  The denominator is unaffected; all 1,735 stay in it as
  non-passes.
- Inside a stage-3 recipe run, the driver's tool interception
  answers the recipe's compiler invocations with its own
  conversion machinery and records each produced artifact
  (path, source tree, edition, crate type, the
  library bindings -- each pinned to the dependency tree
  the build bound -- and compilation options the producing
  invocation carried, the environment it ran under, and
  the directory it ran in),
  leaving a placeholder file
  at the artifact's path -- executable, for a binary -- so the
  recipe's file and spawn checks find it. When the recipe then
  executes a binary it produced earlier in the same run, the
  interception answers that too: the recorded program is run
  through the driver in run mode with the recorded bindings
  and options and the recipe's arguments, and the fabricated process
  outcome is the interpreter's exit code and output. The run
  is reconstructed from the producing invocation, not from
  the file system as it later stands: the conversion reads
  the recorded source tree (or its recorded snapshot, once
  the recipe has rewritten or deleted any of its files),
  the artifact a bare program name denotes is the one the
  invocation's own `PATH` search finds, the artifact a path
  denotes is the one whose recorded file identity (device
  and inode) the file now at that path carries -- so
  renaming one built binary over another makes the path run
  the replacement, and a path whose file no recorded
  identity matches is refused rather than answered with a
  stale program -- the program's
  argv[0] is the name the recipe invoked it under, the
  re-conversion runs under the producing invocation's
  compilation environment (so `option_env!` and its kin see
  the build's values), and the program itself runs under
  the environment its invocation carried, applied as a
  complete assignment once the conversion is done. That
  assignment travels to the run subprocess in a file
  created owner-only under an unpredictable name in the
  shared temporary directory, since it is the driver's
  whole environment, inherited credentials included. The
  program's stderr likewise travels through a channel
  file created owner-only under an unpredictable name,
  since a recipe's diagnostics can carry the same kind
  of contents. The re-conversion likewise runs in the
  producing
  invocation's working directory, so its relative
  compilation paths (a forwarded `--extern` path, a `-L`
  directory) denote the files they denoted at build time,
  and the program enters its invocation's directory once
  the conversion is done; a conversion that reads a
  snapshot remaps the snapshot's directory back to the
  recorded source's directory, so the program's `file!()`
  spellings stay the build's. The snapshot itself is
  materialised owner-only in the shared temporary
  directory -- it copies every recorded neighbouring file
  verbatim, private contents included, so neither its
  directories nor its files are readable by other
  accounts. A run through
  `Command::status()` inherits the recipe's streams, so
  the program's output stands in them in invocation
  order; a run through `Command::output()` collects it
  into the fabricated `Output` instead. This is a
  deliberate reading of "execute the built binary": what is
  judged is the program's input/output behaviour under
  interpretation, not native code generation -- the category
  these recipes belong to tests code generation, so a stage-3
  pass for such a recipe is weaker evidence than a native
  run-make pass. Recipes whose checks need the real artifact
  (native linking, object or DWARF inspection), recipes that
  run a program the run did not produce, and recipes that
  spawn a child to feed its stdin (the process model
  fabricates no child handle, and tool requests carry no
  stdin) stay unsupported.
- The aggregate job merges the shard results
  (`scripts/ci-subset-tools.py merge`; aggregates are recomputed from
  the merged records), derives metric 3 with
  `scripts/std-coverage.py`, and publishes metrics 2(b), 3, and 4 to
  the job summary. The merged result JSONs are kept as run artifacts.
- The denominator is asserted at merge time, not taken from the
  merged records: both merges (metrics 2(b) and 4) run with
  `--expect-total 24937`, the per-nature expectations
  (`exec` 4,149; `procedure` 551; `exec-debugger` 170;
  `compile-check` 18,332; `rustdoc-check` 1,735), and `--survey`
  pointing at the survey the shards were cut from. The merge fails
  without writing output when the survey's total or per-nature
  counts, or the merged records' total or per-nature counts, differ
  from the expectations (survey counts exclude its `env_excluded`
  tests, as the denominator definition does), so a survey
  regenerated through the cache with a different population cannot
  silently become the official denominator. Attempting the
  procedure records through stage 3 changes no count here: the
  denominator stays 24,937 and `procedure` stays 551.
- Environment applicability is judged against one reference
  environment: the host target `x86_64-unknown-linux-gnu` and the
  pinned official Rust 1.100.0 release toolchain (stable channel,
  by its own version string; the release build configuration).
  The runner evaluates `//@ ignore-*`, `only-*`, and `needs-*`
  directives the way compiletest evaluates them for that
  environment -- the host cfg set, compiletest's target rules,
  queries to the pinned rustc, and the release configuration --
  and records a test whose conditions do not hold as not
  attempted (bucket `out of environment`, with the unmet
  condition as the reason); it stays in the denominator as a
  fail, and a condition that cannot be decided is recorded as
  `not attempted (unevaluatable condition)`. The hosted
  environment has the same properties (`ubuntu-latest` is
  x86_64-linux-gnu and the toolchain is the pinned release), so
  the unmet prerequisites -- other targets, the nightly channel,
  a rustc built with debug assertions, Enzyme, or offload
  support, the host target's unsupported sanitizers, and
  dlltool's windows-gnu target rule -- fail identically in CI
  and no workflow change satisfies them. Judging `only-*`
  records under their own target, or replacing the pinned
  toolchain with a differently configured build, would change
  the environment the records are judged against; that is a
  methodology change, not an environment fix (issue #382).
- The cached source tree's sparse checkout additionally includes
  `src/tools/run-make-support`, `src/build_helper`, and
  `src/rustdoc-json-types` (the prepare job widens it before the
  cache is saved): the stage-3 harness extracts those sources from
  the tree to build run-make-support. The survey itself is
  unaffected — it reads only `tests/` and the compiletest /
  bootstrap sources.

## Hosted-runner calibration

Hosted runners are smaller and slower than the dev host (4 vCPU,
16 GB class), so the measurement workflows:

- run the subset runner with `--jobs 2`;
- raise the runner's per-record timeouts through environment
  variables: `SPARAGMOS_NATIVE_COMPILE_TIMEOUT=240`,
  `SPARAGMOS_DRIVER_TIMEOUT=240`, and
  `SPARAGMOS_CONVERT_TIMEOUT=240` (and
  `SPARAGMOS_NATIVE_RUN_TIMEOUT=60`), where the runner's built-in
  defaults -- the conditions of the manual `both`-series
  measurements on the dev host -- are 60-second class
  (compile 60, driver 60, convert 60, run 30). The raised values
  are a calibration for the slower hosted CPUs, nothing more: a
  record near the timeout boundary can pass in CI and time out
  under the `both`-series conditions, or the reverse. The official
  values published by CI are therefore not a same-conditions time
  series with the `both` series, and the two are not mixed when
  comparing measurements over time;
- stage-3 records use the runner's built-in timeouts under both
  the dev-host and CI conditions (300 s for the native recipe
  compile and for the native recipe run, 600 s for the driver run,
  1800 s for the one-time run-make-support build per job); they are
  not raised through the `SPARAGMOS_*_TIMEOUT` variables;
- free disk space before measuring and cache the Rust source tree,
  the survey, and cargo build outputs;
- give every job an explicit `timeout-minutes` below the 6-hour
  hosted limit.
