# Performance baseline

First measured execution-performance baseline for Sparagmos:
native code (rustc 1.100.0, `-O`) against the Sparagmos
interpreter and the Sparagmos JIT, on identical computations.
All figures are medians of repeated runs; the raw samples are
kept in `tests/bench/results.json`.

## How to reproduce

```sh
# builds the driver and the stage-timing helper in release mode,
# then measures every program in tests/bench/ (takes ~50 minutes,
# most of it interpreter execution; run it detached)
nohup python3 scripts/bench.py --out bench-results.json > bench.log 2>&1 &
```

Requirements: the pinned 1.100.0 toolchain (the driver is built
with `RUSTC_BOOTSTRAP=1 cargo +1.100.0 build --release`).

## What is measured

For each bench program:

| figure | meaning |
|---|---|
| native compile | `rustc --edition 2021 -O` wall time, once |
| native exec | wall time of the compiled binary, per run |
| driver conversion | `sparagmos-driver <file> --dump <path>` wall time: process start, Stable MIR conversion, dump write |
| driver run total | `sparagmos-driver run <file>` wall time: the interpreter path end to end, with the interpreter's default limits |
| dump parse | dump text back into a program (harness `parse_dump`), in-process |
| MIR conversion | sparagmos-mir program -> interpreter IR (`convert_program`), in-process |
| interp exec | interpreter execution of the converted program, in-process |
| JIT compile | whole-program JIT compilation (`JitProgram::compile`), in-process |
| JIT exec | one call of the compiled `main`, in-process |

The stage figures come from `scripts/bench-helper/`, a small
standalone binary that loads a driver dump once and times each
stage with `Instant`. The dump round-trip is the representation
the harness runners already use; the interpreter stages timed
there are the same `convert_program` + `Interpreter::run` calls
the driver's run mode makes after its conversion.

Two limits shape what can be measured, and both are part of the
baseline:

* The interpreter's default budget is 1,000,000 steps
  (statements + terminators) and 1,024 call frames, and the
  driver's run mode uses those defaults. Programs past the
  budget stop with `step limit exceeded` end to end. Each
  kernel therefore has two sizes: the plain file (sized for
  pure execution timing) and a `_s` file (sized to complete
  within the default budget). Pure interpreter execution of
  the large sizes is timed with the budget raised inside the
  helper; the end-to-end column records what the default
  budget does to them.
* Floating-point values are not modelled yet: the interpreter
  rejects them at MIR conversion and the JIT at compile time.
  `float_loop.rs` is a native-only reference; its Sparagmos
  outcomes are the explicit unsupported reasons, recorded
  below, not passes.

Bench programs use `while` loops and scalar locals throughout:
arrays and range iterators are outside the JIT path's current
range, and the kernels are written to the intersection of the
three paths so all three execute the same computation. The
matrix-shaped kernel derives element values from indices
instead of loading arrays; its loop shape and arithmetic mix
are those of a scalar matrix multiply's inner loop. Every
program ends with a sentinel branch on its result so the
native optimizer cannot delete the computation (the branch is
never taken).

## Environment

Measured 2026-11-21, 01:46–02:37 JST, on fleur-du-lapin
(Intel Xeon Gold 6140 @ 2.30 GHz, 72 logical CPUs, Linux
6.1.0-55-amd64, Debian 12) with rustc 1.100.0 (0ef1bc386
2026-11-09), at bench-branch commit `7674759` (engine code
identical to main `dfce498`; the branch adds only the bench
files). Driver and helper were built with `cargo --release`;
native guests with `rustc -O`.

The machine was shared with many other parallel jobs for the
whole session: the 1-minute load average ranged from about 20
to about 1,076 (start [1075.6, 943.9, 672.5], end [183.5,
202.8, 274.8] for the 1/5/15-minute averages). Absolute times
therefore carry heavy contention and are best read as
order-of-magnitude; ratios between engines measured in the
same pass are tighter. A back-to-back spot check after the
main run (driver run total vs the sum of stage medians for
the two end-to-end sizes) agreed within about 15%, while the
same measurement repeated at a different time of day differed
by up to about 2x.

Repetitions: native exec 7, driver conversion 5, driver run
total 3, interpreter stages 2, JIT stages 5.

## Results

End-to-end figures (medians):

| program | native compile (once) | native exec | driver conversion | driver run total | driver run exit |
|---|---|---|---|---|---|
| fib_rec.rs | 0.168 s | 7.2 ms | 43 ms | 8.86 s | 2 (step limit) |
| fib_rec_s.rs | 0.117 s | 1.4 ms | 32 ms | 2.07 s | 0 |
| float_loop.rs | 0.117 s | 246.1 ms | 39 ms | 0.037 s | 2 (float unsupported) |
| loop_sum.rs | 0.116 s | 4.2 ms | 33 ms | 8.50 s | 2 (step limit) |
| loop_sum_s.rs | 0.203 s | 2.8 ms | 49 ms | 15.03 s | 0 |
| matmul_kernel.rs | 0.202 s | 4.3 ms | 58 ms | 30.30 s | 2 (step limit) |
| matmul_kernel_s.rs | 0.300 s | 1.5 ms | 49 ms | 21.42 s | 0 |

Stage figures (medians, in-process; ratios against the native
exec median of the same program):

| program | dump parse | MIR conversion | interp exec | vs native | JIT compile | JIT exec | vs native |
|---|---|---|---|---|---|---|---|
| fib_rec.rs | 0.63 ms | 0.03 ms | 122.12 s | ≈1.69×10⁴ | 0.02 ms | 16.5 ms | ≈2.3× |
| fib_rec_s.rs | 1.76 ms | 0.04 ms | 1.94 s | ≈1.36×10³ | 0.05 ms | 0.53 ms | ≈0.4× * |
| float_loop.rs | 0.88 ms | rejected: `Float (F64)` local | — | — | rejected: floating-point type | — | — |
| loop_sum.rs | 1.96 ms | 0.03 ms | 651.87 s | ≈1.54×10⁵ | 0.05 ms | 99.4 ms | ≈24× |
| loop_sum_s.rs | 3.38 ms | 0.07 ms | 11.18 s | ≈3.99×10³ | 0.04 ms | 1.62 ms | ≈0.6× * |
| matmul_kernel.rs | 2.54 ms | 0.04 ms | 549.93 s | ≈1.27×10⁵ | 0.06 ms | 44.9 ms | ≈10× |
| matmul_kernel_s.rs | 5.39 ms | 0.10 ms | 14.02 s | ≈9.34×10³ | 0.06 ms | 0.97 ms | ≈0.6× * |

\* The `_s` programs' native exec times (1.4–2.8 ms) are
process startup, not computation — the computations themselves
are microseconds at that size — so ratios against them do not
compare engines. The large sizes are the engine comparison.

Sample spread in the main run: interpreter exec samples for
the large sizes were 104.8–139.5 s (fib_rec), 623.1–680.6 s
(loop_sum) and 495.6–604.2 s (matmul_kernel); JIT exec
samples stayed within about ±5% of their medians; native
exec samples for the large sizes were 6.3–11.0 ms (fib_rec),
3.7–4.9 ms (loop_sum), 3.4–6.1 ms (matmul_kernel).

## Reading the numbers

* Conversion is not the bottleneck at this scale. Driver
  conversion end to end is 32–58 ms per program, MIR->IR
  conversion at most 0.10 ms, dump parse at most 5.4 ms, and
  JIT compilation at most 0.06 ms median (worst sample
  0.49 ms).
* The interpreter's per-step cost dominates everything else:
  pure execution is between about 1.4×10³ and 1.5×10⁵ times
  the native run time, depending on the kernel. The cost per
  step varies with statement mix; the remainder- and
  checked-arithmetic-heavy loops are the most expensive. As a
  scale reference, one loop_sum iteration costs between ~17
  and 50 interpreter steps (20,000 iterations complete within
  the 1,000,000-step budget; 60,000 do not).
* End to end, the interpreter currently cannot finish the
  large sizes at all: the default step budget stops them
  (exit 2) after 8.5–30.3 s of execution. For long-running
  programs the budget is the first blocker, ahead of speed.
* JIT execution on the large sizes is ≈2.3× native for the
  call-heavy recursive kernel, ≈10× for the dense
  multiply-add kernel and ≈24× for the remainder-heavy loop.
  The backend assigns every local a stack slot (no register
  allocation) and checked arithmetic emits test-and-branch
  sequences, which is the expected shape for this baseline.
* Counting the native compile time, a single cold run through
  the JIT path (driver conversion + parse + JIT compile + one
  execution) totals ≈0.06 s (fib_rec), ≈0.13 s (loop_sum) and
  ≈0.11 s (matmul_kernel), against ≈0.18 s, ≈0.12 s and
  ≈0.21 s for native compile + one run. The compiled native
  binary pulls ahead as soon as it is run repeatedly; the
  interpreter path's cold total for the sizes it can finish
  is its execution time (seconds) plus ~0.05 s of conversion.

## Known gaps this baseline exposes

* The interpreter's 1,000,000-step default budget blocks
  end-to-end execution of long-running programs, and its raw
  speed is three to five orders of magnitude below native.
* No floating point in the interpreter or JIT value models:
  `float_loop.rs` stops with `unsupported type in local _1 of
  float_loop::main: Float (F64)` (interpreter, at MIR
  conversion) and `function float_loop::main:
  floating-point type` (JIT, at compile time).
* The JIT path cannot run programs that use arrays,
  references or ADT values at all yet; the bench kernels are
  written to the scalar intersection of the three paths.
* Native exec of the `_s` sizes measures process startup, so
  small-size ratios are not engine comparisons (see the note
  under the stage table).
* All numbers are from one shared machine under heavy,
  varying load (see Environment). Before any absolute figure
  is quoted elsewhere, re-measure on an idle machine with
  `scripts/bench.py`; expect the ratios to hold up better
  than the absolute times.


## Update 2026-11-21 (gpu-3070): interpreter clone elimination

Measured on gpu-3070 (AMD Ryzen 9 5900X, 24 threads) with the
same bench helper and median-of-runs method, before and after a
semantics-preserving optimisation pass over the interpreter.
A `perf record` profile of the pre-pass interpreter on
`loop_sum` attributed roughly 55-60% of execution time to
clone/drop/alloc work: the run loop cloned the whole current
function body on every step, the plain-local write path cloned
the destination body on every write, place descriptions were
formatted on every read, and the Vec/String push builtins
cloned the whole aggregate to append one element.

The pass, in order:

1. The run loop borrows the current body (resolved by id on
   frame switches) instead of cloning it per step; statements
   and terminators are borrowed too.
2. The write path resolves the destination body's declared
   types by borrow instead of cloning the body per write.
3. Place descriptions are built lazily (only when an error
   needs one) and in a single allocation; the error text is
   byte-identical.
4. `Vec::push`, `String::push_str` and `String::push` mutate
   the stored aggregate in place.

Interpreter execution medians (helper, identical conditions):

| kernel | before | after | speedup |
|---|---|---|---|
| loop_sum_s (2M iterations) | 4,946 ms | 126.5 ms | 39x |
| matmul_kernel_s (32^3) | 7,156 ms | 128.7 ms | 56x |
| fib_rec_s (fib 25) | 1,412 ms | 179.2 ms | 7.9x |
| fib_rec (fib 30) | 65,791 ms | 8,689 ms | 7.6x |
| vec_push_sum_s (2k pushes) | 554 ms | 87.4 ms | 6.3x |
| string_concat_s (1k push_str) | 405 ms | 15.0 ms | 27x |

The interpreter is still far from native (post-pass ratios on
the small kernels: roughly 19x-221x; float_loop 957x), and
`float_loop` now executes in the interpreter, which the
baseline above recorded as rejected. The JIT is unchanged by
this pass and its numbers are unchanged within noise.

Remaining measured hotspots, in order: read paths still clone
whole aggregates (`resolve_value` clones non-reference inputs;
`slice_elements` copies out Vec elements), which dominates the
large vec kernel's ratio; `resolve_place` construction; and
the JIT backend still has no register allocation. Semantics
are unchanged: the full test suites pass and a subset
flip-diff against the pre-pass tree shows no gains or losses
attributable to the pass.

## Update 2026-11-22 (gpu-3070): borrowing read paths

The read side of the clone problem named in the previous
update: `IndexIndex` element reads cloned the whole aggregate
twice per element (an O(n^2) read-back loop), and slice
comparisons cloned both operands elements. The interpreter

## Update 2026-11-22 (gpu-3070): borrowing read paths

The read side of the clone problem named in the previous
update: `IndexIndex` element reads cloned the whole aggregate
twice per element (an O(n^2) read-back loop), and slice
comparisons cloned both operands' elements. The interpreter
now has borrowing read paths (`project_value_ref` /
`read_resolved_ref`, returning `Cow`), builtins resolve
references and slice elements through them (`slice_elements`
rebuilt on top, two clones down to one), indexed reads clone
only the target element, and slice equality/comparison
borrows and stops at the first deciding element.

Bench-helper medians against the pre-pass tree:

| kernel | speedup |
|---|---|
| vec_push_sum (20k pushes) | 11.06x |
| vec_push_sum_s | 2.39x |
| vec_eq | 1.56x |

Integration notes: union projections delegate to the owned
read (union field reads need the typed projector), and the
borrowing path mirrors the owned read's zero-sized-field
rescue for never-written fields, so read semantics are
unchanged. Semantics are unchanged: the full test suites
pass and the subset flip-diff for this integration is
recorded with it.

## Update 2026-11-22 (gpu-3070): pre-decode dispatch layer

Statement execution now runs through a per-body pre-decode
cache (keyed by body address): projection-free locals are
accessed by direct slot, projections and place descriptions
are pre-resolved/prebuilt, constants are pre-evaluated
(except static references), and `SwitchInt` uses binary
search when all patterns are unique. Shapes the decoder
does not specialise delegate to the same engine functions
as before; there is no second execution path. A function
pointer dispatch table was prototyped and measured slower
than the match it would replace (54.56 ns/op against
53.87 ns/op), so the match stays; the prototype remains as
an ignored test in decode.rs.

Bench-helper medians against the pre-pass tree:

| kernel | speedup |
|---|---|
| loop_sum | 1.64x |
| string_concat | 1.52x |
| fib_rec | 1.32x |
| matmul | 1.30x |
| vec_push_sum | 1.26x |
| statement-heavy small kernels | 1.44-1.47x |

Call- and builtin-heavy kernels are unchanged (about 1.0x),
as expected: their cost sits below the statement layer.

## Update 2026-11-22 (fleur): no body clones on call and return

Re-measured all bench kernels on the current main and profiled
with a sampling profiler (perf is blocked on this host, so an
LD_PRELOAD SIGPROF sampler was used). On the call-heavy kernel
the largest single cost was not dispatch or operand evaluation:
`Body::clone` took 39% inclusive of fib_rec_s and dropping those
clones another 21%. `start_call_stamped`, `start_detached_call`
and `exec_return` cloned the callee/caller body (every block,
statement and type) on every call and every return, only to read
fields and pass `&Body` to the write/jump paths. The interpreter
borrows the program (`&'a Program`), so the bodies are available
by reference; the three clones are gone.

Bench-helper medians against the pre-pass tree:

| kernel | speedup |
|---|---|
| fib_rec | 2.50x |
| fib_rec_s | 2.37x |
| vec_push_sum_s | 1.08x |
| loop_sum / vec_push_sum / matmul / string kernels | about 1.0x |

Semantics are unchanged: the full test suites pass (workspace
580, driver 321, corpus 47/47, pytest 349). The profile also
named the next two costs for later passes, both unchanged by
this one: terminator operand evaluation in the run loop goes
through the general `eval_operand` path (59.6% inclusive on
loop_sum, with place-description formatting at 21.2%), and the
`Vec` builtins' `resolve_value` clones the whole vector
(about 50% inclusive on vec_push_sum_s).

## Update 2026-11-22 (gpu-3070): iter-map-fold regression — foreign-closure cap verdict, memoized type walks

`tests/ui/iterators/iter-map-fold-type-length.rs` passed at
both18 (39.5s on this host) but at both19/both20 exceeded the
60s driver timeout; given a longer budget it finished at
121.8s with a spurious instantiation-size-limit failure. A
per-commit bisect plus a sampling profile (LD_PRELOAD SIGPROF,
as in the fleur pass above) split the regression into three
independent wave-12 changes:

- HH (closure demand passes): a foreign closure demanded
  with oversized but finite monomorphic arguments (std
  `map_fold`'s closure over the twelve-deep `Map` tower)
  tripped the local expanded-size cap and set the
  compilation-wide recursion-limit failure. A foreign
  instance is finite by construction — the compiler already
  monomorphized it — and the reachability collection already
  exempts foreign demand from the cap for exactly this
  reason.
- HN (closure layout recording): +46s. The rendered instance
  key is a rendering of the whole type tree, and it was
  computed on every closure-type encounter before the
  per-instance memo could apply.
- The HS+HR segment: +33s, profile-attributed to HR's
  guarded kind query — every structural type walk pays a
  mutex lock and a `catch_unwind` per node; the
  expanded-size guard walk alone was 41% inclusive.

Fixes, all in the driver:

- The closure demand passes now fail only the demand itself
  (recorded under the recursion-limit reason) when a
  *foreign* closure trips the expanded-size cap; the
  compilation-wide failure is reserved for locally generated
  expansion. Local behaviour is unchanged: the
  size-increasing tuple test still fails in 0.33s.
- `record_layout` / `record_coroutine_layout` dedup attempts
  by the compiler type handle before rendering the key.
- `generic_args_within_expanded_size` memoizes exact
  expanded sizes per type handle across all checks of a
  compilation (the argument graph is a DAG; only fully
  measured sizes are stored, so verdicts are unchanged).
- `ty_mentions_param` is memoized per type handle, cleared
  per compilation.

Result on this host: 121.8s / exit 2 -> 49.1s / exit 0 (the
remaining distance to the 39.5s base is wave 12's added
collection and recording work, now paid once per distinct
instance). Suites: workspace 602, driver 376, corpus 47/47,
pytest 349.
