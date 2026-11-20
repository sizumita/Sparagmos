// bench (end-to-end size): same computation as fib_rec.rs, sized
// to finish within the interpreter's default step budget
// (1,000,000 statements + terminators) so the whole driver run
// path can complete it. See docs/PERFORMANCE.md.
fn fib(n: u64) -> u64 {
    if n < 2 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

fn main() {
    let r = fib(22);
    if r == u64::MAX {
        loop {}
    }
}
