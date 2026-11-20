// corpus: floating-point comparisons, including NaN and infinity
// (every comparison with NaN is false except !=).
fn main() {
    let nan = 0.0f64 / 0.0;
    let inf = 1.0f64 / 0.0;
    let x = 2.5f64;
    println!("{} {} {} {}", x < 3.0, x >= 2.5, x == 2.5, x != 2.5);
    println!("{} {} {} {}", nan == nan, nan != nan, nan < x, nan >= x);
    println!("{} {} {}", inf > 1e300, -inf < -1e300, inf == inf);
}
