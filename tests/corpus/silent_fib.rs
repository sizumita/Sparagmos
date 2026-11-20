// corpus: recursion without any output — fibonacci by self-recursion.
// The result is verified against its expected value before the
// program ends, so a wrong result from the engine under test surfaces
// as a panic (exit 101) where the native run exits 0.
//
// The check is arithmetic rather than assert!: dividing 1 by
// (actual == expected) divides by 1 when the values agree and by 0
// when they differ, and the division-by-zero Assert is a construct
// both execution paths implement. assert! would put a call to
// core::panicking::panic in the body, which the JIT path cannot
// compile yet.
fn expect_eq(actual: u64, expected: u64) {
    let _guard = 1 / ((actual == expected) as u64);
}

fn fib(n: u64) -> u64 {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}
fn main() {
    expect_eq(fib(15), 610);
}
