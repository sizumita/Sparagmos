// corpus: tuples — field access, swap, tuples as arguments and results.
fn swap(p: (i32, i32)) -> (i32, i32) {
    (p.1, p.0)
}
fn stats() -> (i32, i32, i32) {
    (3, 1, 4)
}
fn main() {
    let t = swap((10, 20));
    let s = stats();
    let nested = ((1, 2), (3, 4));
    println!("{} {} {} {}", t.0 + t.1, s.0 + s.1 + s.2, nested.0 .1, nested.1 .0);
}
