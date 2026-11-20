// corpus: floats in loops — repeated accumulation shows rounding,
// a halving loop runs while the value stays above a bound.
fn main() {
    let mut sum = 0.0f64;
    for _ in 0..10 {
        sum += 0.1;
    }
    let mut v = 100.0f64;
    let mut n = 0;
    while v > 0.01 {
        v /= 2.0;
        n += 1;
    }
    println!("{} {} {}", sum, n, v);
}
