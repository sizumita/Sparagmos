// bench: String growth by repeated push_str, then a byte-wise
// checksum pass over the result. Exercises the String value model
// and the growth path of the string builtins. The trailing
// sentinel branch keeps the result observable to the native
// optimizer (never taken at runtime).
fn main() {
    let mut s = String::new();
    let mut i: i64 = 0;
    while i < 10_000 {
        s.push_str("abcd");
        i += 1;
    }
    let mut acc: i64 = 0;
    let b = s.as_bytes();
    let mut j: usize = 0;
    while j < b.len() {
        acc += b[j] as i64;
        j += 1;
    }
    if acc == i64::MAX {
        loop {}
    }
}
