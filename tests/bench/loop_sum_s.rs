// bench (end-to-end size): same computation as loop_sum.rs, sized
// to finish within the interpreter's default step budget
// (1,000,000 statements + terminators) so the whole driver run
// path can complete it. See docs/PERFORMANCE.md.
fn main() {
    let mut acc: i64 = 0;
    let mut i: i64 = 0;
    let n: i64 = 20_000;
    while i < n {
        acc += (i % 997) * (i % 991) + (i % 7);
        i += 1;
    }
    if acc == i64::MAX {
        loop {}
    }
}
