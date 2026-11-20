// corpus: nested Option — Some(Some(v)) pattern matching.
fn main() {
    let nested = Some(Some(7u32));
    let c = match nested {
        Some(Some(v)) => v,
        _ => 999,
    };
    let flat = Some(5u32);
    let d = match flat {
        Some(v) => v * 2,
        None => 0,
    };
    println!("{} {}", c, d);
}
