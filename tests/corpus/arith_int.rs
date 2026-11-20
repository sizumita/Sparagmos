// corpus: integer arithmetic — add/sub/mul/div/mod, operator precedence, i32/i64 widths.
fn main() {
    let a: i32 = 40 + 2;
    let b: i64 = 1_000_000 * 3;
    let c = (a - 10) / 4;
    let d = 17 % 5;
    println!("{} {} {} {}", a, b, c, d);
}
