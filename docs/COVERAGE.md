# Coverage: how "99%" is measured

Placeholder — owned by the design/coverage lane.

Three separate metrics (never blended into one number):
1. MIR variant coverage (implemented variants / all variants in the target
   rustc's `rustc_middle::mir`, counted mechanically).
2. Corpus / rustc test-suite pass rate (JIT result == native result).
3. std feature coverage required by the corpus.

Note: Rust 1.100.0 was released on 2026-11-12
(https://blog.rust-lang.org/2026/11/12/Rust-1.100.0/) and is installed on
the dev host via rustup (verified: rustc/cargo 1.100.0, --emit=mir works).
The MIR reference and all denominators are Rust 1.100.0 from the start.
