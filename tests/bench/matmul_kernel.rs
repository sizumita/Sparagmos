// bench: dense multiply-add kernel with the triple-nested loop
// shape of a matrix product. Element values are derived from the
// indices by small formulas instead of array loads, because arrays
// are outside the JIT path's current range; the loop control and
// arithmetic mix match the inner loop of a scalar matrix multiply.
// All intermediate values stay far inside i64 range.
// The trailing sentinel branch keeps the result observable to the
// native optimizer (never taken at runtime).
fn main() {
    let dim: i64 = 40;
    let passes: i64 = 8;
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
