// corpus: array forms — repeat syntax, nested arrays, by-value iteration, len.
fn main() {
    let zeros = [0; 5];
    let grid = [[1, 2], [3, 4], [5, 6]];
    let mut total = zeros.len() as i32;
    for row in grid {
        for v in row {
            total += v;
        }
    }
    let mut rev_sum = 0;
    let xs = [10, 20, 30];
    let mut i = xs.len();
    while i > 0 {
        i -= 1;
        rev_sum += xs[i];
    }
    println!("{} {}", total, rev_sum);
}
