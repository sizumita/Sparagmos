// bench (end-to-end size): same computation as matmul_kernel.rs,
// sized to finish within the interpreter's default step budget
// (1,000,000 statements + terminators) so the whole driver run
// path can complete it. See docs/PERFORMANCE.md.
fn main() {
    let dim: i64 = 16;
    let passes: i64 = 3;
    let mut acc: i64 = 0;
    let mut p: i64 = 0;
    while p < passes {
        let mut i: i64 = 0;
        while i < dim {
            let mut j: i64 = 0;
            while j < dim {
                let mut k: i64 = 0;
                while k < dim {
                    let a = (i * 7 + k * 13) % 97;
                    let b = (k * 11 + j * 5) % 89;
                    acc += a * b;
                    k += 1;
                }
                j += 1;
            }
            i += 1;
        }
        p += 1;
    }
    if acc == i64::MAX {
        loop {}
    }
}
