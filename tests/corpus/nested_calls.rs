// corpus: deeply nested calls and a five-argument function.
fn f1(x: i32) -> i32 { x + 1 }
fn f2(x: i32) -> i32 { f1(x) * 2 }
fn f3(x: i32) -> i32 { f2(x) - 3 }
fn sum5(a: i32, b: i32, c: i32, d: i32, e: i32) -> i32 {
    a + b + c + d + e
}
fn main() {
    let v = f3(f2(f1(5)));
    let w = sum5(f1(1), f1(2), f1(3), f1(4), f1(5));
    println!("{} {}", v, w);
}
