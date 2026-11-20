// corpus: while loops without any output — accumulation and a nested
// loop with manual counters. The accumulated results are verified
// against their expected values before the program ends, so a wrong
// result from the engine under test surfaces as a panic (exit 101)
// where the native run exits 0.
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
    let mut sum: i64 = 0;
    let mut i: i64 = 0;
    while i < 100 {
        sum += i;
        i += 1;
    }
    let mut grid: i64 = 0;
    let mut x: i64 = 0;
    while x < 10 {
        let mut y: i64 = 0;
        while y < 10 {
            grid += x * y;
            y += 1;
        }
        x += 1;
    }
    expect_eq(sum, 4_950);
    expect_eq(grid, 2_025);
    expect_eq(sum + grid, 6_975);
}
