fn neg(x: i32) -> i32 { -x }
fn copy_demo(x: i32) -> i32 { let y = x; y }
fn move_demo(v: Vec<i32>) -> Vec<i32> { let w = v; w }
fn main() { let _a = neg(3); let _b = copy_demo(4); }
