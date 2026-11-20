// corpus: floating-point arithmetic (f64) — four operations and
// compound assignment; 0.1 + 0.2 shows the binary rounding.
fn main() {
    let a = 1.5f64;
    let b = 2.25f64;
    let mut x = a + b;
    x *= 2.0;
    x -= 0.5;
    x /= 3.0;
    println!("{} {} {} {}", a + b, a - b, a * b, b / a);
    println!("{} {}", x, 0.1 + 0.2);
}
