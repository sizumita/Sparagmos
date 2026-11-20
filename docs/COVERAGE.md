# Coverage: how "99%" is measured

Placeholder — owned by the design/coverage lane.

Three separate metrics (never blended into one number):
1. MIR variant coverage (implemented variants / all variants in the target
   rustc's `rustc_middle::mir`, counted mechanically).
2. Corpus / rustc test-suite pass rate (JIT result == native result).
3. std feature coverage required by the corpus.

Note: "Rust 1.100.0" is a future release. Until it ships, the MIR reference
is Rust 1.99.1 (the closest toolchain available on the dev host); the
denominators will be swapped when 1.100.0 is released.
