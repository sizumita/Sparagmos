// corpus: char values — literals, comparison, casts to integers, printing.
fn main() {
    let a: char = 'x';
    let b: char = 'A';
    let diff = a as u32 - b as u32;
    let next = (b as u8 + 1) as char;
    let is_upper = b >= 'A' && b <= 'Z';
    println!("{} {} {} {}", diff, next, is_upper, a);
}
