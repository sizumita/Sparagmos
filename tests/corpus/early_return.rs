// corpus: early returns and guard clauses across several functions.
fn first_positive(xs: [i32; 5]) -> i32 {
    for x in xs {
        if x > 0 {
            return x;
        }
    }
    -1
}
fn grade(score: i32) -> i32 {
    if score < 0 {
        return -1;
    }
    if score >= 90 {
        return 4;
    }
    if score >= 50 {
        return 1;
    }
    0
}
fn main() {
    let a = first_positive([0, -2, 7, 8, 9]);
    let b = grade(95) * 100 + grade(60) * 10 + grade(10);
    println!("{} {}", a, b);
}
