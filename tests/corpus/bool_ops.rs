// corpus: boolean logic — short-circuit evaluation with function calls.
fn yes() -> bool { true }
fn no() -> bool { false }
fn main() {
    let a = yes() && no();
    let b = yes() || no();
    let c = !a && (b || no());
    let d = if c { 1 } else { 0 };
    println!("{} {} {} {}", a, b, c, d);
}
