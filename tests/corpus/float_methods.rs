// corpus: primitive float methods (f64 and f32) — predicates,
// rounding, powers, min/max with NaN, and bit casts. NaN and the
// infinities are produced at run time (0.0/0.0, 1.0/0.0).
fn main() {
    let nan = 0.0f64 / 0.0;
    let inf = 1.0f64 / 0.0;
    println!("{} {} {}", nan.is_nan(), 1.0f64.is_nan(), inf.is_infinite());
    println!("{} {} {}", inf.is_finite(), 2.5f64.is_finite(), (-0.0f64).is_sign_negative());
    println!("{} {} {}", (-0.0f64).is_sign_positive(), (-3.5f64).abs(), (-0.0f64).abs());
    println!("{} {}", 2.0f64.sqrt(), (-4.0f64).sqrt());
    println!("{} {} {} {}", 2.5f64.floor(), (-2.5f64).floor(), 2.5f64.ceil(), (-2.5f64).ceil());
    println!("{} {} {} {}", 2.5f64.round(), (-2.5f64).round(), 2.5f64.trunc(), (-2.5f64).trunc());
    println!("{} {} {}", 2.75f64.fract(), 4.0f64.recip(), (-0.5f64).signum());
    println!("{} {}", nan.signum(), nan.min(1.0));
    println!("{} {}", 1.0f64.max(nan), 1.5f64.powi(3));
    println!("{} {}", 2.0f64.powf(0.5), 2.0f64.powi(-2));
    println!("{} {}", std::f64::consts::PI.to_degrees(), 180.0f64.to_radians());
    println!("{} {}", 1.0f64.to_bits(), f64::from_bits(4607182418800017408u64));
    let a = 2.0f32;
    let b = -2.5f32;
    println!("{} {} {}", a.sqrt(), b.abs(), b.floor());
    println!("{} {} {}", b.ceil(), b.round(), a.powi(2));
    println!("{} {} {}", a.powf(0.5f32), a.max(b), a.min(b));
    println!("{} {}", (0.0f32 / 0.0).is_nan(), 1.0f32.to_bits());
    println!("{}", f32::from_bits(1065353216u32));
}
