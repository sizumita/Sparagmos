// corpus: compound assignment operators on plain locals.
fn main() {
    let mut a: i32 = 100;
    a += 5;
    a -= 3;
    a *= 2;
    a /= 4;
    a %= 30;
    let mut b: u32 = 0b1010;
    b &= 0b1100;
    b |= 0b0011;
    b ^= 0b1111;
    b <<= 2;
    b >>= 1;
    println!("{} {}", a, b);
}
