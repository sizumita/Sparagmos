// corpus: function calls, multiple args, return values, nested calls.
fn add(a: i32, b: i32) -> i32 { a + b }
fn square(x: i32) -> i32 { x * x }
fn main() {
    println!("{}", add(square(3), square(4)));
}
