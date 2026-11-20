// bench: recursion-heavy integer computation — fibonacci by
// self-recursion. Function-call and branching cost dominates.
// The trailing sentinel branch keeps the result observable to the
// native optimizer (it is never taken at runtime); without a use of
// the result, an optimizing compiler may delete the whole call.
fn fib(n: u64) -> u64 {
    if n < 2 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

fn main() {
    let r = fib(30);
    if r == u64::MAX {
        loop {}
    }
}
