// corpus: integer arithmetic without any output — the same
// computations as arith_int.rs, but nothing is printed and the
// program ends silently. Exercises the execution path for programs
// whose only observable behaviour is normal termination.
fn main() {
    let a: i32 = 40 + 2;
    let b: i64 = 1_000_000 * 3;
    let c = (a - 10) / 4;
    let d = 17 % 5;
    let _total = b + c as i64 + d as i64;
}
