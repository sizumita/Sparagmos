// corpus: mutability, shadowing, tuples and destructuring.
fn main() {
    let x = 5;
    let x = x + 1;
    let mut y = x;
    y *= 10;
    let (a, b) = (y, x);
    println!("{} {} {}", x, a, b);
}
