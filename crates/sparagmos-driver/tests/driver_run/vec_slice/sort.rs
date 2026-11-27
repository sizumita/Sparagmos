// Sort tests.

const VEC_STRING_SORT_SRC: &str = r#"
fn main() {
    let mut v: Vec<String> = vec![
        "banana".to_string(),
        "apple".to_string(),
        "cherry".to_string(),
        "apple".to_string(),
    ];
    v.sort();
    for s in &v { println!("sorted: {s}"); }
    v.reverse();
    for s in &v { println!("reversed: {s}"); }
    v.sort_unstable();
    for s in &v { println!("unstable: {s}"); }
    println!("binsearch apple: {:?}", v.binary_search(&"apple".to_string()));
    v.dedup();
    for s in &v { println!("dedup: {s}"); }
    let mut refs: Vec<&str> = vec!["pear", "fig", "plum", "fig"];
    refs.sort();
    for s in &refs { println!("refs: {s}"); }
    let mut w: Vec<String> = vec!["a".to_string(), "bb".to_string(), "ccc".to_string()];
    w.sort_by_key(|s| s.len());
    for s in &w { println!("by len: {s}"); }
    println!("vec-string-sort-ok");
}
"#;

#[test]
fn run_mode_vec_string_sort() {
    run_ok(
        "vec-string-sort",
        VEC_STRING_SORT_SRC,
        "sorted: apple\nsorted: apple\nsorted: banana\nsorted: cherry\nreversed: cherry\nreversed: banana\nreversed: apple\nreversed: apple\nunstable: apple\nunstable: apple\nunstable: banana\nunstable: cherry\nbinsearch apple: Ok(1)\ndedup: apple\ndedup: banana\ndedup: cherry\nrefs: fig\nrefs: fig\nrefs: pear\nrefs: plum\nby len: a\nby len: bb\nby len: ccc\nvec-string-sort-ok\n",
    );
}

const SORT_INTERLEAVED_SCRATCH_SRC: &str = r#"
fn main() {
    let mut v: Vec<usize> = Vec::new();
    for i in 0..8usize {
        for p in 0..4usize {
            v.push(p * 1000 + i);
        }
    }
    v.sort();
    println!("sorted: {v:?}");
}
"#;

#[test]
fn run_mode_sort_interleaved_scratch_partition() {
    // Driftsort on interleaved runs takes the stable-quicksort
    // path: `stable_partition` copies elements one at a time
    // between the vector buffer and the stack scratch array
    // (`AlignedStorage`, a byte array field reinterpreted as
    // elements). The scratch base pointer is a bare field
    // projection, and `partition_one` offsets it by an element
    // count (`scratch_base.add(num_left)`). That offset used to
    // fall out of the declared-type array judgment (which
    // required a trailing index step) into the value-shape
    // absorption, which denominated the position in the
    // offset's element size over a byte container: element 1
    // landed at byte 1, the address was misaligned, and the
    // guest died in the `copy_nonoverlapping` precondition
    // check (exit 134). The bare field base now judges as
    // element 0 of the field's sequence, so the partition
    // copies land on element boundaries and the sort runs.
    run_ok(
        "sort-interleaved-scratch",
        SORT_INTERLEAVED_SCRATCH_SRC,
        "sorted: [0, 1, 2, 3, 4, 5, 6, 7, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007]\n",
    );
}

const SORT_INTERLEAVED_FORTY_SRC: &str = r#"
fn main() {
    let mut v: Vec<usize> = Vec::new();
    for i in 0..10usize {
        for p in 0..4usize {
            v.push(p * 1000 + i);
        }
    }
    v.sort();
    let ok = v.windows(2).all(|w| w[0] <= w[1]);
    println!("len {} sorted {} first {} last {}", v.len(), ok, v[0], v[v.len()-1]);
}
"#;

#[test]
fn run_mode_sort_interleaved_forty_scratch_offset_from() {
    // At forty elements driftsort's merge measures the saved
    // run against the scratch base with `ptr_offset_from`:
    // one pointer denotes an element through the storage
    // field's index step, the other is the bare field base the
    // cast chain left behind. The distance split read the
    // trailing field step as the element index under an empty
    // prefix key, so the two never shared a key and the
    // intrinsic reported unrelated pointers. The split now
    // also offers the bare-base reading (the field path as the
    // key, index 0), so the pair measures.
    run_ok(
        "sort-interleaved-forty",
        SORT_INTERLEAVED_FORTY_SRC,
        "len 40 sorted true first 0 last 3009\n",
    );
}

const SORT_UB_CHECK_FOUR_HUNDRED_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::new();
    for k in 0..400usize {
        let j = (k * 7 + 1) % 4;
        v.push((j * 1000 + (k / 4)) as i32);
    }
    v.sort();
    let mut sorted = v.len() == 400;
    for i in 1..v.len() { if v[i - 1] > v[i] { sorted = false; } }
    let mut sum: i64 = 0;
    for x in &v { sum += *x as i64; }
    println!("stable sorted {} sum {} first {} last {}", sorted, sum, v[0], v[399]);
    let mut w: Vec<i32> = Vec::new();
    for i in (0..400usize).rev() { w.push(v[i]); }
    w.sort_unstable();
    let mut same = w.len() == v.len();
    for i in 0..w.len() { if w[i] != v[i] { same = false; } }
    println!("unstable same {}", same);
}
"#;

#[test]
fn run_mode_sort_ub_check_predicates_four_hundred() {
    // Four hundred elements in the interleaved shape of the
    // `mpsc_stress` sort: the stable and unstable sorts drive
    // thousands of `ptr::copy_nonoverlapping` / `ptr::copy`
    // calls inside core, each gated by a `core::ub_checks`
    // precondition (non-overlap, alignment, non-null) that the
    // interpreter now evaluates natively at the call instead of
    // executing the collected check bodies. The verdicts must
    // accept every one of the sort's own copies — a false
    // positive aborts the run in the precondition check (exit
    // 134) — and the sorted results must be exact: the sum and
    // the endpoints pin the whole multiset.
    run_ok(
        "sort-ub-check-four-hundred",
        SORT_UB_CHECK_FOUR_HUNDRED_SRC,
        "stable sorted true sum 619800 first 0 last 3099\nunstable same true\n",
    );
}

const SORT_UB_CHECK_ZERO_SIZE_SRC: &str = r#"
fn main() {
    let mut z: Vec<()> = vec![(); 64];
    z.sort();
    println!("zst len {}", z.len());
    let mut e: Vec<i32> = Vec::new();
    e.sort();
    println!("empty len {}", e.len());
    let mut one: Vec<i32> = vec![42];
    one.sort();
    println!("one {:?}", one);
    let a: Vec<i32> = (0..10).collect();
    let mut b: Vec<i32> = vec![0; 10];
    b[..0].copy_from_slice(&a[..0]);
    b.copy_from_slice(&a);
    println!("copied {:?}", b);
}
"#;

#[test]
fn run_mode_sort_ub_check_zero_size_paths() {
    // The zero-sized corner of the same preconditions: a ZST
    // sort, an empty sort, and a zero-length slice copy. The
    // aligned-and-non-null predicate must accept these through
    // its zero-size arm (a dangling or never-dereferenced
    // pointer is fine when nothing is copied), exactly as the
    // collected check bodies accept them.
    run_ok(
        "sort-ub-check-zero-size",
        SORT_UB_CHECK_ZERO_SIZE_SRC,
        "zst len 64\nempty len 0\none [42]\ncopied [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n",
    );
}

const SORT_TWO_SORTED_RUNS_SRC: &str = r#"
fn main() {
    let mut v: Vec<usize> = Vec::new();
    for i in 0..16usize {
        v.push(i * 2);
    }
    for i in 0..16usize {
        v.push(i * 2 + 1);
    }
    v.sort();
    let ok = v.windows(2).all(|w| w[0] <= w[1]);
    println!("len {} sorted {} first {} last {}", v.len(), ok, v[0], v[v.len()-1]);
}
"#;

#[test]
fn run_mode_sort_two_sorted_runs_merge_scratch_first_use() {
    // Two pre-sorted runs of sixteen: driftsort finds both
    // runs already sorted, so `merge` is the first user of a
    // pristine scratch. Three denomination gaps used to break
    // it in sequence: the stored scratch pointer is the bare
    // field base (its first-element step stripped by the cast
    // chain), so the merge's first comparison read the whole
    // storage array instead of element 0; the run copy into
    // the byte-denominated storage landed element `i` in byte
    // slot `i` instead of slot `i * size_of::<T>()`; and the
    // elements the copy moved can carry the `MaybeUninit`
    // union form, which a scalar comparison now reads through.
    // The dereference of a bare array base denotes element 0
    // for an element or scalar pointee, bulk sequence access
    // prices its stride from the recorded field element size,
    // and the comparison unwraps a one-sided wrapper form.
    run_ok(
        "sort-two-sorted-runs",
        SORT_TWO_SORTED_RUNS_SRC,
        "len 32 sorted true first 0 last 31\n",
    );
}

const SORT_INTERLEAVED_SIXTYFOUR_SRC: &str = r#"
fn main() {
    let mut v: Vec<usize> = Vec::new();
    for i in 0..16usize {
        for p in 0..4usize {
            v.push(p * 1000 + i);
        }
    }
    v.sort();
    let ok = v.windows(2).all(|w| w[0] <= w[1]);
    println!("len {} sorted {} first {} last {}", v.len(), ok, v[0], v[v.len()-1]);
}
"#;

#[test]
fn run_mode_sort_interleaved_sixtyfour_scratch_merge() {
    // The sixty-four-element interleaved sort: the scratch
    // merge path at the size where the saved run no longer
    // fits the small-sort special cases (see
    // `run_mode_sort_two_sorted_runs_merge_scratch_first_use`
    // for the denomination gaps in this path).
    run_ok(
        "sort-interleaved-sixtyfour",
        SORT_INTERLEAVED_SIXTYFOUR_SRC,
        "len 64 sorted true first 0 last 3015\n",
    );
}

const SORT_INTERLEAVED_FOUR_HUNDRED_SRC: &str = r#"
fn main() {
    let mut v: Vec<usize> = Vec::new();
    for i in 0..100usize {
        for p in 0..4usize {
            v.push(p * 1000 + i);
        }
    }
    v.sort();
    let ok = v.windows(2).all(|w| w[0] <= w[1]);
    println!("len {} sorted {} first {} last {}", v.len(), ok, v[0], v[v.len()-1]);
}
"#;

#[test]
fn run_mode_sort_interleaved_four_hundred_scratch_merge() {
    // The four-hundred-element interleaved sort: driftsort's
    // merge walks the scratch with pointers the cast chain
    // left entered into element 0's `MaybeUninit` payload
    // (`[Field(1), Index(0), Field(0)]` over the byte storage
    // field). Two denomination gaps used to break it in
    // sequence. The pointer offset over that entered position
    // fell out of the declared-type array judgment (which
    // required the index step to be trailing) into the
    // value-shape absorption, which priced the offset in the
    // offset's own element size over the byte-denominated
    // container: `add(1)` on the `*mut usize` landed on byte
    // slot 1 instead of slot 8, and the merge's comparisons
    // read the union wrapper's storage as a unit value. The
    // array judgment now splits a position into its index step
    // and the whole-value entry steps behind it, prices the
    // offset by the sequence's recorded element size, and
    // keeps the entry steps. And the verification's `windows`
    // read the sorted elements in the `MaybeUninit` union form
    // the bulk copies left behind: an index run over the
    // materialised window tuple refused to compose because
    // the head element was an enum, so the second index
    // ranged inside the one-field wrapper and reported a
    // position past its end. A one-field variant-0 wrapper
    // around a scalar payload is the scalar in wrapper form,
    // not an element aggregate, so the run now composes.
    run_ok(
        "sort-interleaved-four-hundred",
        SORT_INTERLEAVED_FOUR_HUNDRED_SRC,
        "len 400 sorted true first 0 last 3099\n",
    );
}

const MERGE_UP_BYTE_STORAGE_SRC: &str = r#"
use std::mem::MaybeUninit;

#[repr(C)]
struct AlignedStorage {
    _align: [usize; 0],
    storage: [MaybeUninit<u8>; 4096],
}

impl AlignedStorage {
    fn new() -> Self {
        Self { _align: [], storage: [const { MaybeUninit::uninit() }; 4096] }
    }
    fn as_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<usize>] {
        let len = 4096 / std::mem::size_of::<usize>();
        unsafe { core::slice::from_raw_parts_mut(self.storage.as_mut_ptr().cast(), len) }
    }
}

struct MergeState {
    start: *mut usize,
    end: *mut usize,
    dst: *mut usize,
}

impl MergeState {
    unsafe fn merge_up(&mut self, mut right: *const usize, right_end: *const usize) {
        unsafe {
            let left = &mut self.start;
            let out = &mut self.dst;
            while *left != self.end && right as *const usize != right_end {
                let consume_left = !(*right < **left);
                let src = if consume_left { *left } else { right };
                std::ptr::copy_nonoverlapping(src, *out, 1);
                *left = left.add(consume_left as usize);
                right = right.add(!consume_left as usize);
                *out = out.add(1);
            }
        }
    }
}

fn do_merge(v: &mut [usize], scratch: &mut [MaybeUninit<usize>], mid: usize) {
    unsafe {
        let buf = scratch.as_mut_ptr().cast::<usize>();
        let v_base = v.as_mut_ptr();
        let v_mid = v_base.add(mid);
        let v_end = v_base.add(v.len());
        std::ptr::copy_nonoverlapping(v_base, buf, mid);
        let mut merge_state = MergeState { start: buf, end: buf.add(mid), dst: v_base };
        merge_state.merge_up(v_mid, v_end);
    }
}

fn main() {
    let mut v: Vec<usize> = Vec::new();
    for i in 0..16usize {
        for p in 0..4usize {
            v.push(p * 1000 + i);
        }
    }
    let mut stack_buf = AlignedStorage::new();
    let scratch = stack_buf.as_uninit_slice_mut();
    do_merge(&mut v, scratch, 32);
    let mut wsum: usize = 0;
    for (i, x) in v.iter().enumerate() {
        wsum += (i + 1) * x;
    }
    println!("len {} first {} last {} wsum {}", v.len(), v[0], v[63], wsum);
}
"#;

#[test]
fn run_mode_merge_up_over_byte_storage_scratch() {
    // The hand-written merge over a user-defined byte-storage
    // scratch (the shape std's driftsort scratch takes, in
    // user MIR): the scratch pointers are denominated through
    // a `[MaybeUninit<u8>]` field reinterpreted as
    // `[MaybeUninit<usize>]`, so the base pointer denotes
    // element 0 entered through the union payload and every
    // `add` must be priced in the storage's own byte slots
    // (see `run_mode_sort_interleaved_four_hundred_scratch_merge`).
    // The merge used to stop at the second element with a
    // unit value assigned to a `usize` place.
    run_ok(
        "merge-up-byte-storage",
        MERGE_UP_BYTE_STORAGE_SRC,
        "len 64 first 0 last 3015 wsum 3272656\n",
    );
}

const DRIFT_SORT_LARGE_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = (0..5000).rev().collect();
    v.sort();
    println!("{} {} {}", v[0], v[2500], v[4999]);
}
"#;

#[test]
fn run_mode_sort_large_input_crosses_the_drift_sqrt_threshold() {
    // Past 4096 elements driftsort's run-length heuristic calls
    // `sqrt_approx`, which has no demandable body of its own.
    run_ok("drift-sort-large", DRIFT_SORT_LARGE_SRC, "0 2500 4999\n");
}

const SORT_ORD_VIOLATION_SRC: &str = r#"
use std::cell::Cell;
fn main() {
    let flip = Cell::new(false);
    let mut v: Vec<i32> = (0..50).collect();
    v.sort_by(|a, b| {
        flip.set(!flip.get());
        if flip.get() { a.cmp(b) } else { b.cmp(a) }
    });
    println!("sorted {:?}", &v[..3]);
}
"#;

#[test]
fn run_mode_sort_ord_violation_panics_with_core_message() {
    let dir = temp_dir("sort-ord-violation");
    let src = fixture(&dir, "sortviol.rs", SORT_ORD_VIOLATION_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stderr={stderr}");
    assert!(
        stderr.contains(
            "user-provided comparison function does not correctly implement a total order"
        ),
        "stderr={stderr}"
    );
}
