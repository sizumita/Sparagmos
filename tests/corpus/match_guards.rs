// corpus: match guards, range patterns and bindings.
fn classify(n: i32) -> i32 {
    match n {
        x if x < 0 => -1,
        0 => 0,
        1..=9 if n % 2 == 0 => 2,
        1..=9 => 1,
        _ => 3,
    }
}
fn main() {
    let mut out = 0;
    for n in [-5, 0, 4, 7, 50] {
        out = out * 10 + classify(n) + 5;
    }
    println!("{}", out);
}
