// corpus: Option<u32> — search function returning Option, matched by caller.
fn find_first_even(xs: [u32; 4]) -> Option<u32> {
    for x in xs {
        if x % 2 == 0 {
            return Some(x);
        }
    }
    None
}
fn main() {
    let a = match find_first_even([1, 3, 4, 7]) {
        Some(v) => v,
        None => 999,
    };
    let b = match find_first_even([1, 3, 5, 7]) {
        Some(v) => v,
        None => 999,
    };
    println!("{} {}", a, b);
}
