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
- The driver is built once per host class per run (the `build`
  job for Linux x86_64, the `build-host` jobs for the other
  classes), with the driver crate's `measure` profile
  (`crates/sparagmos-driver/Cargo.toml`: the dev profile with the
  optimizer on; debug assertions and overflow checks stay enabled,
  so a check that panics under a plain `cargo build` panics here
  too). Every judging job downloads its class's binary as an
  artifact, so no shard compiles the driver and all shards of a
  class run the same build.
- The survey is split into shards
  (`scripts/ci-subset-tools.py shard`, round-robin over path order so
  shards are balanced), and each shard runs as one matrix job with
  `scripts/rustc-subset-run.py`. The metric-2(b) pass is cut into 12
  shards and the JIT pass into 4 by default (the dispatch inputs
  `shards` and `jit_shards`): the metric-2(b) pass carries the
  stage-3 and stage-4 records and is several times heavier. The metric-2(b) pass runs the
  runner with `--stages all`: stages 1 and 2, the stage-3
  procedure harness, which attempts the `procedure` records
  (run-make recipes — each recipe runs natively as the oracle, then
  under the driver with tool interception; the cargo suites
  `run-make-cargo` and `build-std` are out of stage-3 scope and
  recorded as not attempted, because their recipes drive cargo
  itself and cargo orchestration — workspace resolution,
  build-script and proc-macro processes, and the real compiler
  artifacts cargo consumes — is not modelled by the harness),
  and the stage-4 rustdoc harness, which attempts the
  `rustdoc-check` records (see "Stage 4" below). A second
  matrix repeats the pass with `--jit` for metric 4; the JIT
  pass attempts neither procedure nor rustdoc records (the
  runner never combines stages 3 or 4 with `--jit`), so they
  remain in metric 4's denominator as not
  attempted.
- Stage-1 and stage-2 program runs pass `--guest-process-host`
  to the driver: a guest's `std::process` children really start
  on the host (the interpreter's `Child` model is performed
  against live host children), and a child naming the guest's
  own executable -- or a byte-identical copy of it -- re-runs
  the guest program under the driver with the child's
  arguments, environment and working directory. Stage-3 runs
  configure tool interception instead and install no process
  host, and the JIT pass (`run-jit`) installs none either.
- The `rustdoc-check` records (1,735) are attempted by the
  stage-4 rustdoc harness for the rustdoc-html (802),
  rustdoc-ui (440), and rustdoc-json (194) suites (see
  "Stage 4" below). The remaining 299 -- rustdoc-js
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
- The `exec` records of the `coverage` suite (108) and the
  `coverage-run-rustdoc` suite (1) are attempted by no stage.
  compiletest judges them by coverage output rather than by
  program output: the coverage-map mode compares the
  coverage-dump tool's output over the test's saved LLVM IR
  against its `.cov-map` snapshot, and the coverage-run mode
  compares the `llvm-cov` text report rendered from an
  instrumented run's merged counters against its `.coverage`
  snapshot (the rustdoc suite's record also runs the file's
  doctests under instrumentation and merges their counters).
  Every revision of every one of these tests keeps at least
  one of the two comparisons active, so no verdict reduces to
  the plain execution stage 1 judges, and Sparagmos implements
  neither the coverage counters/regions rustc-codegen-llvm
  materialises nor the coverage-mapping dump and llvm-cov
  report path, so the runner records them as not attempted
  under the bucket `not attempted (coverage instrumentation)`
  (issue #514). The denominator is unaffected; all 109 stay
  in it as non-passes.
- An executed run's verdict follows compiletest's own for the
  run (runtest/ui.rs): the run's exit must classify as the
  mode requires (run-pass: pass class, exit 0; run-fail:
  failure class, any exit in 1..=127 -- the class, not one
  fixed code: a test's `failure-status` directive constrains
  the compile phase's status only, and a sanitizer runtime's
  exit or a plain `process::exit(1)` is the same class as a
  panic's 101), the run's output must contain the test's
  `error-pattern` / `regex-error-pattern` directives (checked
  against the run's stderr, or its stdout followed by stderr
  under `check-stdout`), and the run's streams are compared
  only when the test declares `check-run-results` -- the one
  directive under which compiletest compares run output at
  all -- after compiletest's output normalization (path
  rules, the thread-id rule, the test's own
  `normalize-stdout` / `normalize-stderr` rules). A native run
  failing the class or pattern checks is a broken oracle for
  the record and is recorded as a native mismatch, not judged.
  Without `check-run-results` the streams are not compared:
  libtest's parallel completion order, thread interleaving
  and address-valued prints make a run's output
  non-reproducible in ways compiletest never judges. The
  guest's stderr reaches the judge on its own channel
  (`--guest-stderr-file`), apart from the driver process's
  diagnostics; the JIT run shares the driver's channel, so
  its check-run-results verdict compares stdout only. Both
  runs also receive `RUST_TEST_TMPDIR` (the attempt's temp
  directory), which bootstrap exports to the whole test
  process tree and tests read at run time.
- The run-fail exec records compiled with a
  `-Z sanitizer=...` flag (the `tests/ui/sanitizer` family)
  are attempted by no stage. compiletest judges them by
  the sanitizer runtime's own report on the instrumented
  binary -- its exit class and the diagnostic its
  `error-pattern` names -- and the instrumentation is
  LLVM-level: no MIR the interpreter executes carries it
  and the guest program is never instrumented, so no
  verdict reduces to plain execution. The runner records
  them as not attempted under the bucket
  `not attempted (sanitizer instrumentation)` (issue #532),
  the same shape as the coverage records above. The
  denominator is unaffected. A run-pass sanitizer test
  (the cfi family, memory-passing, dataflow) is judged on
  the instrumented program running clean, which plain
  execution reproduces, so those stay attempted.
- An `//@ aux-bin:` file is built as compiletest builds it:
  after the test's auxiliary crates, into
  `auxiliary/bin/` of the attempt directory, with its own
  header's flags and environment -- including the auxiliary
  crates its own `//@ aux-crate:` / `//@ aux-build:`
  directives name, which are resolved and built for the bin
  (the on-broken-pipe bins name sigpipe_utils this way, a
  crate the test itself never declares). Both runs work in
  the attempt directory, so both spawn the same binaries.
  The runs' library path is compiletest's as well -- the
  sysroot lib dir, the target lib dir holding the shared
  std, and the attempt's auxiliary directory -- on the
  native run and on the driver process alike, whose host
  children load the same shared libraries. An auxiliary
  build's environment is the auxiliary file's own
  (`--aux-env-scrub`): the test's `//@ rustc-env:` variables
  ride the driver process's environment into the main
  conversion, where `env!` / `option_env!` read them, and
  are scrubbed from the driver's auxiliary builds, whose
  own `//@ rustc-env:` directives are applied after the
  scrub.
- The exec records whose mode is run-crash are attempted by
  stages 1 and 2 and judged by crash class, following
  compiletest's own classification of a finished run
  (runtest/ui.rs's RunResult): exit 0 is a pass-class run, an
  exit code in 1..=127 is a failure-class run, and death by
  signal or a code above 127 is a crash. The record passes
  when the native run classifies as a crash, the Sparagmos
  run classifies as a crash too, and the guest's stdout
  agrees with the native run's; when the native run does not
  crash, the record is not judged and keeps the
  native-expectation bucket, as a native mismatch does for
  the other modes. To make the classes distinguishable,
  the driver reports a guest crash -- a guest panic that may
  not unwind (panic=abort, a nounwind assertion failure, a
  second panic while a panic is in flight) -- with exit code
  134 and a `guest crashed` marker line, while an unwinding
  guest panic keeps exit 101 and a completed run keeps 0.
  A guest failure the interpreter cannot exhibit as a crash
  (a hook panic the interpreter unwinds where native code
  aborts, a conversion gap) fails under its own bucket; the
  verdict is never widened to cover it.
- The compile-check records of the `crashes` suite carry
  compiletest's crash expectation (runtest/crashes.rs): the
  compile passes the test exactly when the compiler process
  ends with an exit code other than 0 or 1, and no output is
  compared. The runner's crash judging implements exactly
  that condition, so these records are judged, not waived:
  the driver drives rustc only as far as Stable MIR, the
  internal compiler errors these tests reproduce live in
  later compiler phases, and a driver that converts the
  input or rejects it cleanly is recorded as failing the
  crash expectation (`no crash (converted successfully)` /
  `no crash (clean rejection)`) until the crash itself is
  reproduced.
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
- Metric 3 runs in its own job (`metric3`), alongside the shards:
  `scripts/std-coverage.py --survey-json` takes the stage-1
  candidates (exec, single-file, ui / ui-fulldeps) from the survey
  instead of from the merged results, since the predicate reads only
  properties the survey classifies. The aggregate job publishes the
  value only after `scripts/ci-subset-tools.py check-candidates` has
  asserted that the measured candidate set is exactly the one the
  merged metric-2(b) results define.
- The aggregate job merges the shard results
  (`scripts/ci-subset-tools.py merge`; aggregates are recomputed from
  the merged records), checks and adds the metric-3 result, and
  publishes metrics 2(b), 3, and 4 to the job summary. It runs no
  toolchain or build. The merged result JSONs are kept as run
  artifacts.
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
  denominator stays 24,937 and `procedure` stays 551. The same
  holds for stage 4: the denominator stays 24,937 and
  `rustdoc-check` stays 1,735.
- Environment applicability is judged per host class (issue
  #517). The metric-2(b) pass partitions the survey by host class
  (`scripts/ci-subset-tools.py partition`, run by the prepare
  job): each record is assigned to the first class whose host
  model admits it, evaluating the record's headers with the
  runner's own applicability code under that class's host
  profile -- the target's cfg condition set and compiletest's
  target rules for it, with the target-spec facts taken from the
  pinned toolchain's own spec document and the live-probe facts
  taken as satisfied (a satisfied probe can only keep a record
  attemptable, never exclude it, so no record the Linux host
  attempts is moved away, and the judging host's runner
  re-evaluates the probes for real). The classes are disjoint
  and cover the survey exactly once -- the partition command
  refuses to write output otherwise -- so the aggregate's
  denominator assertion (24,937 with the per-nature split) is
  unchanged and no record is double-counted or dropped. Records
  no enabled class admits stay in the first class (Linux
  x86_64) as the residual class and keep the verdict its runner
  gives them today (bucket `out of environment`, with the unmet
  condition as the reason; a condition that cannot be decided
  is `not attempted (unevaluatable condition)`).
- The enabled classes are `linux-x86_64` (the `measure` shards
  on `ubuntu-latest`) and `darwin-x86_64` (`macos-15-intel`);
  the latter runs as the `measure-host` jobs, with a driver the
  `build-host` job builds natively on the class's runner and the
  runner's host model selected through
  `SPARAGMOS_HOST_TARGET`. On the survey of run 45564001677 the
  partition moves 29 records off the Linux class to
  `darwin-x86_64`; every other record is judged exactly where it
  was judged before.
  The host jobs do not restore the rustdoc artifact tree (it is
  a Linux-built tree), so a class record whose stage 4 needs it
  records the same not-attempted verdict the Linux jobs record
  on a cache miss. Metric 4 and metric 3 are not partitioned:
  they run on the Linux host over the full survey (the JIT
  lowers x86_64 machine code, and metric 3's candidate
  predicate reads only survey-classified properties).
- What no host class changes: conditions that are not host
  properties. The pinned official Rust 1.100.0 release toolchain
  (stable channel, by its own version string; the release build
  configuration) is the same build on every runner, so the
  nightly channel, a rustc built with debug assertions, Enzyme,
  and offload support fail identically on all of them, as do the
  runner-configuration conditions (wasmtime, clang-based
  tests). Conditions that name a compilation target rather
  than an execution host (the wasm32 and nvptx64 families,
  32-bit and other non-hosted targets) are likewise not host
  cases: compiletest matches them against the target, and they
  belong with cross-target support. `ignore-auxiliary` is not a
  host case either: compiletest registers the `auxiliary`
  condition as unconditionally true, so a file carrying it is
  never run as a test on any host -- it is a module input to
  another test, which carries its own record -- and its record
  is classified out of environment. The remaining classes are
  defined in the runner's host profiles and in the partition
  rule but have no jobs yet, so their records stay in the
  residual class: the aarch64 classes
  (`aarch64-unknown-linux-gnu`, `aarch64-apple-darwin`) wait on
  the driver building for aarch64 -- sparagmos-jit's entry
  trampoline (`crates/sparagmos-jit/src/program.rs`) is x86_64
  naked assembly with no aarch64 counterpart, and the driver
  links the JIT crate on every host (the class builds in run
  45575305504 fail with 90 inline-asm errors each) -- and the
  Windows classes (`x86_64-pc-windows-msvc`,
  `x86_64-pc-windows-gnu`) wait on that plus the runner's
  process layer (process groups, `/proc`-based memory
  accounting), which is POSIX-only, so a Windows runner would
  record internal errors instead of verdicts (issue #517).
- Auxiliary crates are built the way compiletest builds them:
  with the test's own compile arguments, which carry the
  revision's `--cfg` for a revised test (an incremental test's
  auxiliary crate selects its revision's definitions through
  `#[cfg(rpass1)]` / `#[cfg(rpass2)]`; the runner stamps the
  revision onto each auxiliary spec and the driver forwards it
  to the auxiliary build as `--cfg`), and a proc-macro
  auxiliary chain links the same native test-helpers library
  the native auxiliary builds link. A test whose header
  combines `add-minicore` with `run-pass` is not attempted:
  compiletest itself panics on the combination when parsing
  the directives (`add-minicore` cannot be used to run the
  test binary), so the file is never compiled — the one suite
  file with both is a should-fail self-test of that rejection.
- The cached source tree's sparse checkout additionally includes
  `src/tools/run-make-support`, `src/build_helper`, and
  `src/rustdoc-json-types` (the prepare job widens it before the
  cache is saved): the stage-3 harness extracts those sources from
  the tree to build run-make-support. It also includes `src/etc`
  (run-make-support's `htmldocck()` helper executes
  `src/etc/htmldocck.py` from the source root) and
  `src/librustdoc/html/static/css` (read directly by the rustdoc
  themes recipe); without them those recipes fail on the native
  side of stage 3. It also includes `src/tools/jsondocck`,
  `src/tools/jsondoclint`, and `src/tools/rustdoc`: the stage-4
  harness extracts `htmldocck.py` and the jsondocck /
  jsondoclint sources from the tree and runs the guest rustdoc
  from the tree's `src/tools/rustdoc/main.rs`. The survey itself
  is unaffected — it reads only `tests/` and the
  compiletest / bootstrap sources.
- The stage-3 recipe environment carries
  `DOC_RUST_LANG_ORG_CHANNEL` (`https://doc.rust-lang.org/` plus
  the pinned version, matching what bootstrap exports for
  compiletest on the stable channel): `htmldocck.py` refuses to
  start without it.

### Stage 4 (the rustdoc harness)

Stage 4 attempts the `rustdoc-check` records (1,735) for the
suites that need no JavaScript or browser environment:

- `rustdoc-html` (802): the pinned rustdoc documents the test
  file (compiletest's `document()`), and the tree's own
  `src/etc/htmldocck.py` checks the generated HTML.
- `rustdoc-json` (194): `document()` with `--output-format
  json`, then the tree's own `jsondocck` and `jsondoclint`
  check the produced JSON. Both tools are built once per job
  from the tree's sources, against a `Cargo.lock` cut from the
  tree's own lock (their crates.io dependencies are pinned by
  the tree; a fresh resolution would drift to incompatible
  versions).
- `rustdoc-ui` (440): the pinned rustdoc compiles the test file
  with compiletest's UI argument set; the verdict is
  compiletest's UI verdict (pass/fail mode and failure status,
  snapshot compare of the rendered diagnostics, `//~`
  annotations, error patterns). A `rustdoc-ui` record
  declaring `build-fail` is recorded as out of stage-4 scope:
  compiletest's build-fail prerequisite is a check build
  (`--emit=metadata`) that must pass while the judged build
  fails, a distinction that exists only for rustc --
  compiletest passes no `--emit` to rustdoc, and rustdoc
  rejects the flag -- so the mode has no prerequisite run to
  judge against for rustdoc.

The `rustdoc-gui`, `rustdoc-js`, and `rustdoc-js-std` suites
(299 records) run in a browser / JavaScript environment; stage 4
records them as out of scope with that reason, in the
denominator (Issue #365 tracks the environment).

The native rustdoc run is the oracle: a record whose native run
does not satisfy compiletest's rules for its suite is a plain
failure. Only when the native side passes does the guest side
run — rustdoc itself, converted and executed by the driver from
the rustdoc artifact tree (`scripts/build-rustdoc-artifacts.sh`;
`docs/RUSTDOC-ARTIFACTS.md`). The guest program is the same for
every record and its conversion does not depend on the record's
input, so the harness measures the guest's startup once per job
with a `--version` probe and attributes the outcome to each
record whose oracle passed: while the rustdoc execution chain
stops during startup, those records are `unsupported` with that
measured reason; once the probe passes, each record's guest run
is made for real and judged by the same rules.

Failure buckets follow the rule of the other stages: a judged
failure never carries the empty bucket (the pass arithmetic
counts an empty bucket as a pass). A native-side failure is
bucketed `native rustdoc failure`; a guest failure is bucketed
by what mismatched -- `exit-code mismatch` for the run's exit
status, `stdout mismatch` for a rustdoc-ui snapshot,
annotation, or pattern mismatch, and `output mismatch` for a
rustdoc-html / rustdoc-json checker failure over the guest's
output directory.

The artifact tree is not built inside the measurement jobs:
building it compiles the whole compiler-crate closure from
source (the tree is ~1.9 GB), which does not fit the prepare
job's budget. The measure jobs restore it from a dedicated
cache (`rustdoc-artifacts-1.100.0-v1`) into
`.ci/rustdoc-artifacts-1.100.0` and point the runner at it with
`SPARAGMOS_RUSTDOC_ARTIFACTS`. Until that cache is populated,
official runs record each oracle-passing record as not
attempted with the reason "rustdoc artifact tree unavailable" —
the measured state of that environment, in the denominator.
Local runs use the tree at `~/rustdoc-artifacts-1.100.0` (or
`--rustdoc-artifacts`) and record the guest's measured startup
outcome the same way.

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
- stage-4 records likewise use built-in timeouts (240 s for a
  native rustdoc run, 1800 s for a guest run and for the one-time
  per-job startup probe, 1200 s for a checker-tool build);
  `SPARAGMOS_RUSTDOC_TIMEOUT` and its siblings can raise them,
  but the workflows do not;
- bound every driver subprocess's resident memory: the runner
  polls each driver process tree's RSS while it runs and kills
  the tree when it crosses `SPARAGMOS_DRIVER_MEM_LIMIT_MB`
  (built-in default 5120 MB; the workflows do not raise it),
  recording the record as that record's own failure with bucket
  `memory limit` and a reason naming the bound. The per-record
  timeouts cannot express this bound: a driver that balloons
  faster than its timeout (Issue #480's debugger session grew
  at ~190 MB/s against a 240 s timeout) exhausts the hosted
  runner's 16 GB and kills the whole shard job, and since a
  shard's results are written only when it finishes, every
  result in the shard is lost (Issue #483). The bound sits at
  roughly twice the largest legitimate driver RSS measured
  (~2.6 GB for a stage-4 guest run), and two `--jobs` slots at
  the bound plus the native side and the runner stay inside
  the 16 GB runner. Only driver subprocesses are bounded; the
  native oracle's compiles and runs are the reference
  toolchain's own and are not the runner's to cut off. The
  bound is on RSS, not address space: the driver's virtual
  size is ~10 GB at startup, so an address-space cap tight
  enough to matter would stop it spawning its own threads;
- in the official measurement, run the driver built with the
  `measure` profile (optimized, with the dev profile's debug
  assertions and overflow checks; flip-diff still compares two dev
  builds, so its base and head stay like for like). Like the
  raised timeouts, this only changes how fast a record runs, and so
  which records near a timeout boundary finish in time; official
  values from runs before and after the switch are not a
  same-conditions series either;
- cache the Rust source tree, the survey, and cargo build outputs
  (the hosted image's disk has ample room for the toolchain and the
  tree, so no preinstalled software is removed first);
- give every job an explicit `timeout-minutes` below the 6-hour
  hosted limit.
