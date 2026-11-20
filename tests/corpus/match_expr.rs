// corpus: match expression — literal patterns, wildcard, value binding result.
fn describe(n: i32) -> &'static str {
    match n {
        0 => "zero",
        1 | 2 => "small",
        3..=9 => "digit",
        _ => "many",
    }
}
fn main() {
    for n in [0, 2, 5, 42] {
        println!("{}:{}", n, describe(n));
    }
}
