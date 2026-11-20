// bench: floating-point iteration — two coupled f64 recurrences.
// Native-only reference: the interpreter and JIT value models have
// no floating-point values yet, so on those paths this program is
// expected to stop with an explicit unsupported reason (recorded in
// docs/PERFORMANCE.md), never a silent pass.
// The trailing sentinel branch keeps the result observable to the
// native optimizer (never taken at runtime).
fn main() {
    let mut x: f64 = 1.0;
    let mut y: f64 = 0.5;
    let mut i: i64 = 0;
    let n: i64 = 50_000_000;
    while i < n {
        x = x * 1.0000000001 + y * 0.0000000001;
        y = y * 0.9999999999 + x * 0.0000000001;
        i += 1;
    }
    if x.is_nan() {
        loop {}
    }
}
