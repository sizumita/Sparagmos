// corpus: if/else branching and comparison operators.
fn main() {
    let x = 7;
    if x > 10 {
        println!("big");
    } else if x > 5 {
        println!("medium");
    } else {
        println!("small");
    }
    let y = if x == 7 { 100 } else { 0 };
    println!("{}", y);
}
