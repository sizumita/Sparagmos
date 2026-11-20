// corpus: `as` casts between integer types, bool and char conversions.
fn main() {
    let a: i64 = 300;
    let b = a as i32;
    let c = (a as u8) as u32;
    let d = true as u8;
    let e = 'a' as u32;
    let f = 65u8 as char;
    let g = -1i32 as u32;
    println!("{} {} {} {} {}", b, c, d as u32 + e, f, g > 100);
}
