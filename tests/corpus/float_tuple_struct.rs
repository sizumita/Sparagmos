// corpus: floats inside tuples, arrays and a struct with a method.
struct Vec2 {
    x: f64,
    y: f64,
}
impl Vec2 {
    fn len_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }
}
fn main() {
    let v = Vec2 { x: 3.0, y: 4.0 };
    let t: (f64, i32) = (2.5, 7);
    let arr = [0.5f64, 1.5, 2.5];
    println!("{} {} {}", v.len_sq(), t.0 * t.1 as f64, arr[0] + arr[1] + arr[2]);
}
