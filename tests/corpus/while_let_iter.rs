// corpus: while let over an iterator — Range next with &mut writeback.
fn main() {
    let mut it = 0..5;
    let mut sum = 0;
    while let Some(x) = it.next() {
        sum += x;
    }
    let mut it2 = 1..=4;
    let mut prod = 1;
    while let Some(x) = it2.next() {
        prod *= x;
    }
    println!("{} {}", sum, prod);
}
