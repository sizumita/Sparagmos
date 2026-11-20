// corpus: f32 arithmetic and f32/f64 conversions — widening is
// exact, narrowing rounds (0.1f32 widened shows the f32 rounding).
fn main() {
    let a = 0.1f32;
    let b = 0.2f32;
    println!("{} {}", a + b, a as f64);
    println!("{} {}", 3.4028235e38f64 as f32, 1.5f32 as f64 * 2.0);
    let c: f32 = 1.0 / 3.0;
    println!("{} {}", c, c as f64);
}
