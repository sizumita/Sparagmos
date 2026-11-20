// corpus: recursion without any output — fibonacci by self-recursion,
// result discarded, ending silently.
fn fib(n: u64) -> u64 {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}
fn main() {
    let _ = fib(15);
}
