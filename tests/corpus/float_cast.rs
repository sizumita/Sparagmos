// corpus: casts between floats and integers — Rust float-to-int
// casts saturate, NaN becomes 0; int-to-float rounds to nearest.
fn main() {
    println!("{} {} {}", 3.99f64 as i32, -3.99f64 as i32, 12.9f64 as u8);
    println!("{} {} {}", 1e300f64 as i64, -1e300f64 as i64, -5.0f64 as u8);
    println!("{} {}", (0.0f64 / 0.0) as i32, 42i32 as f64 + 0.5);
    println!("{} {}", 16777217i64 as f32, 9007199254740993u64 as f64);
    println!("{} {} {}", 200.0f64 as u8, 300.7f64 as u8, 70000.0f64 as u16);
    println!("{} {} {}", -300.7f64 as u16, (0.0f64 / 0.0) as u8, 1e300f64 as u64);
    println!("{} {}", 200.0f32 as u8, 1e10f32 as u32);
}
