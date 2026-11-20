// corpus: while loops without any output — accumulation and a nested
// loop with manual counters, ending silently.
fn main() {
    let mut sum: i64 = 0;
    let mut i: i64 = 0;
    while i < 100 {
        sum += i;
        i += 1;
    }
    let mut grid: i64 = 0;
    let mut x: i64 = 0;
    while x < 10 {
        let mut y: i64 = 0;
        while y < 10 {
            grid += x * y;
            y += 1;
        }
        x += 1;
    }
    let _ = sum + grid;
}
