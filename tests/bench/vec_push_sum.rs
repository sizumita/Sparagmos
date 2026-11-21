// bench: Vec growth and sequential read-back — push in a loop,
// then index-sum the elements. Exercises the Vec value model,
// method-call dispatch and the index builtins. The trailing
// sentinel branch keeps the result observable to the native
// optimizer (never taken at runtime).
fn main() {
    let mut v: Vec<i64> = Vec::new();
    let mut i: i64 = 0;
    while i < 20_000 {
        v.push(i % 1000);
        i += 1;
    }
    let mut acc: i64 = 0;
    let mut j: usize = 0;
    while j < v.len() {
        acc += v[j];
        j += 1;
    }
    if acc == i64::MAX {
        loop {}
    }
}
