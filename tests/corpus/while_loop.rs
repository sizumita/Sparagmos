// corpus: while loop with mutation and break.
fn main() {
    let mut n = 1;
    let mut count = 0;
    while n < 1000 {
        n *= 2;
        count += 1;
        if count > 20 { break; }
    }
    println!("{} {}", n, count);
}
