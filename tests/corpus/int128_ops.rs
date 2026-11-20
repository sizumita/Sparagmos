// corpus: 128-bit integers — i128/u128 arithmetic at the extremes,
// casts in both directions, promoted array constants, comparisons.
fn get() -> i128 {
    -12345
}

fn main() {
    let a = get();
    println!("{} {}", a * 3 + 1, a / 100);
    println!("{}", i128::MAX - 1);
    println!("{}", u128::MAX - 1);
    println!("{}", (1u128 << 100) + 7);
    println!("{} {}", (300i128 as i8), (-1i128 as u128));
    println!("{} {}", (1e30f64 as i128), (16777217i128 as f32));
    println!("{}", [5i128, -7i128][1]);
    println!("{}", [u128::MAX, 1][0]);
    let arr: [i128; 2] = [5, -7];
    println!("{} {}", arr == [5i128, -7i128], arr == [5i128, 7i128]);
    let rep = [i128::MIN; 1];
    println!("{}", rep[0]);
    println!("{} {}", a < i128::MIN + 20000, u128::MAX > (1u128 << 127));
    println!("{} {}", (7i128 << 120), (-16i128 >> 2));
}
