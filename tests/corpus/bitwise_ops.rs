// corpus: bitwise operators — and, or, xor, not, same-type shifts.
fn main() {
    let a: u32 = 0b1100;
    let b: u32 = 0b1010;
    let c = (a & b) | (a ^ b);
    let d = !a & 0xFF;
    let e = (a << 2) >> 1;
    println!("{} {} {}", c, d, e);
}
