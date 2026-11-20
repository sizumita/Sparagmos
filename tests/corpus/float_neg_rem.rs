// corpus: float negation and remainder — the remainder keeps the
// dividend sign, and -0.0 compares equal to 0.0.
fn main() {
    println!("{} {} {}", -1.5f64, 5.5f64 % 2.0, -5.5f64 % 2.0);
    println!("{} {}", -0.0f64 == 0.0f64, -(2.0f64 - 5.0));
}
