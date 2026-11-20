// bench: one long-running accumulation loop — integer remainder,
// multiply and add per iteration, no calls. Loop-control and
// arithmetic throughput dominate. All intermediate values stay far
// inside i64 range, so checked and unchecked arithmetic agree.
// The trailing sentinel branch keeps the result observable to the
// native optimizer (never taken at runtime).
fn main() {
    let mut acc: i64 = 0;
    let mut i: i64 = 0;
    let n: i64 = 30_000_000;
    while i < n {
        acc += (i % 997) * (i % 991) + (i % 7);
        i += 1;
    }
    if acc == i64::MAX {
        loop {}
    }
}
