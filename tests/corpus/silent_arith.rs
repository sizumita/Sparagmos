// corpus: integer arithmetic without any output — the same
// computations as arith_int.rs, but nothing is printed. Each result
// is verified against its expected value before the program ends, so
// a wrong result from the engine under test surfaces as a panic
// (exit 101) where the native run exits 0.
//
// The check is arithmetic rather than assert!: dividing 1 by
// (actual == expected) divides by 1 when the values agree and by 0
// when they differ, and the division-by-zero Assert is a construct
// both execution paths implement. assert! would put a call to
// core::panicking::panic in the body, which the JIT path cannot
// compile yet.
fn expect_eq(actual: i64, expected: i64) {
    let _guard = 1 / ((actual == expected) as i64);
}

fn main() {
    let a: i32 = 40 + 2;
    let b: i64 = 1_000_000 * 3;
    let c = (a - 10) / 4;
    let d = 17 % 5;
    let total = b + c as i64 + d as i64;
    expect_eq(a as i64, 42);
    expect_eq(b, 3_000_000);
    expect_eq(c as i64, 8);
    expect_eq(d as i64, 2);
    expect_eq(total, 3_000_010);
}
