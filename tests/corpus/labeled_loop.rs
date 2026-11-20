// corpus: labeled loops — break with value, labeled continue, nested loops.
fn main() {
    let mut found = 0;
    'outer: for i in 0..10 {
        for j in 0..10 {
            if i * j == 12 {
                found = i * 100 + j;
                break 'outer;
            }
        }
    }
    let mut sum = 0;
    let mut n = 0;
    'count: loop {
        n += 1;
        if n % 3 == 0 {
            continue 'count;
        }
        if n > 10 {
            break 'count;
        }
        sum += n;
    }
    let val = loop {
        break 42;
    };
    println!("{} {} {}", found, sum, val);
}
