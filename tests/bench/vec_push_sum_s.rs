// bench (small): Vec growth and sequential read-back, sized to
// complete within the interpreter's default step budget. See
// vec_push_sum.rs for the kernel description.
fn main() {
    let mut v: Vec<i64> = Vec::new();
    let mut i: i64 = 0;
    while i < 2_000 {
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
