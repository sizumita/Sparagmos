// corpus: signed division and remainder — truncation toward zero, negative operands.
fn main() {
    let a = -7 / 2;
    let b = -7 % 2;
    let c = 7 / -2;
    let d = 7 % -2;
    let e: i64 = -100 / 7;
    let f: i64 = -100 % 7;
    println!("{} {} {} {} {} {}", a, b, c, d, e, f);
}
