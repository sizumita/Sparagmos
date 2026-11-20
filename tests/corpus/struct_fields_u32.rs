// corpus: struct construction and field access (unsigned fields).
struct Point {
    x: u32,
    y: u32,
}
fn main() {
    let mut p = Point { x: 3, y: 4 };
    p.x += 10;
    let q = Point { x: p.x, y: p.y * 2 };
    println!("{} {}", q.x, q.y);
}
