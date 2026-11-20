// corpus: Option<u32> — passed by value into a matcher with a guard arm.
fn describe(o: Option<u32>) -> u32 {
    match o {
        Some(v) if v > 10 => 2,
        Some(_) => 1,
        None => 0,
    }
}
fn main() {
    let b = describe(Some(20)) * 100 + describe(Some(3)) * 10 + describe(None);
    println!("{}", b);
}
