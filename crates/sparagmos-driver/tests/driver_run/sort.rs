// Slice sort tests.

use super::run_ok;

// A 400-element vector in the shape a round-robin collect
// from four producers leaves behind (the `concurrent_writes`
// shape of tests/ui/threads-sendsync/mpsc_stress.rs): each
// producer's sequence in short ascending runs, interleaved.
const SORT_MANY_RUNS_SRC: &str = r#"
fn main() {
    for _ in 0..6 {
        let mut v: Vec<usize> = Vec::new();
        let mut done = 0;
        while done < 100 {
            for j in 0..4 {
                for i in done..(done + 8).min(100) {
                    v.push(j * 1000 + i);
                }
            }
            done += 8;
        }
        assert_eq!(v.len(), 400);
        v.sort();
        let sum: usize = v.iter().sum();
        println!(
            "len={} sum={} first={} at100={} last={} sorted={}",
            v.len(),
            sum,
            v[0],
            v[100],
            v[399],
            v.windows(2).all(|w| w[0] <= w[1])
        );
    }
}
"#;

#[test]
fn run_mode_sort_many_short_runs() {
    // Regression for #399: sorting a buffer laid out in many
    // short runs is pointer-offset work over the buffer on
    // every merge step. Judging each offset cloned the whole
    // element container (the offset projection's shape checks
    // and container snapshots), so one 400-element sort cost
    // seconds of host time and long-running programs built
    // on sorts/collects of this shape met the record timeout
    // (mpsc_stress). The offsets now judge the container by
    // reference; the sorts must stay correct and cheap.
    run_ok(
        "sort-many-short-runs",
        SORT_MANY_RUNS_SRC,
        "len=400 sum=619800 first=0 at100=1000 last=3099 sorted=true\n\
         len=400 sum=619800 first=0 at100=1000 last=3099 sorted=true\n\
         len=400 sum=619800 first=0 at100=1000 last=3099 sorted=true\n\
         len=400 sum=619800 first=0 at100=1000 last=3099 sorted=true\n\
         len=400 sum=619800 first=0 at100=1000 last=3099 sorted=true\n\
         len=400 sum=619800 first=0 at100=1000 last=3099 sorted=true\n",
    );
}
