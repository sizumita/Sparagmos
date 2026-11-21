// bench (small): String growth by repeated push_str plus a
// byte-wise checksum, sized to complete within the interpreter's
// default step budget. See string_concat.rs for the kernel
// description.
fn main() {
    let mut s = String::new();
    let mut i: i64 = 0;
    while i < 1_000 {
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
