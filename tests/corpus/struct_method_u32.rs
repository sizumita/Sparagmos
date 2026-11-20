// corpus: struct methods taking &self (unsigned fields).
struct Rect {
    w: u32,
    h: u32,
}
impl Rect {
    fn area(&self) -> u32 {
        self.w * self.h
    }
    fn perimeter(&self) -> u32 {
        2 * (self.w + self.h)
    }
}
fn main() {
    let r = Rect { w: 6, h: 7 };
    println!("{} {}", r.area(), r.perimeter());
}
