// corpus: mutable references — write through deref, reborrow, ref parameters.
fn inc(r: &mut i32) {
    *r += 1;
}
fn get(r: &i32) -> i32 {
    *r
}
fn main() {
    let mut x = 41;
    inc(&mut x);
    let y = get(&x);
    let r = &mut x;
    *r *= 2;
    println!("{} {}", x, y);
}
