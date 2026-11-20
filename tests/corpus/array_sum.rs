// corpus: arrays — literal, indexing, iteration, len.
fn main() {
    let xs = [3, 1, 4, 1, 5, 9, 2, 6];
    let mut sum = 0;
    for i in 0..xs.len() {
        sum += xs[i];
    }
    println!("{} {}", sum, xs[3]);
}
