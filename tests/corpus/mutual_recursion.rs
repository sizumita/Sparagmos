// corpus: mutual recursion — is_even / is_odd calling each other.
fn is_even(n: u32) -> bool {
    if n == 0 { true } else { is_odd(n - 1) }
}
fn is_odd(n: u32) -> bool {
    if n == 0 { false } else { is_even(n - 1) }
}
fn main() {
    println!("{} {} {}", is_even(10), is_odd(7), is_even(3));
}
