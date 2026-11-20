// corpus: float formatting — Display and Debug of plain values,
// negative zero, NaN and an integral float.
fn main() {
    println!("{} {:?} {} {:?}", 1.5f64, 1.5f64, 3.0f64, 3.0f64);
    println!("{} {:?} {}", -0.0f64, -0.0f64, 0.25f32);
    println!("{:?} {}", 0.0f64 / 0.0, -2.75f64);
}
