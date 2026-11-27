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

const OSSTR_REF_SORT_SRC: &str = r#"
use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStringExt;

fn main() {
    let a = OsString::from_vec(b"alpha".to_vec());
    let b = OsString::from_vec(b"bravo".to_vec());
    let c = OsString::from_vec(b"charlie".to_vec());
    let d = OsString::from_vec(b"delta".to_vec());
    let ra: &OsStr = &a;
    let rb: &OsStr = &b;
    let rc: &OsStr = &c;
    let rd: &OsStr = &d;
    let mut v: Vec<&OsStr> = vec![rd, ra, rc, rb];
    v.sort();
    for e in &v {
        println!("{:?} len={} bytes={:?}", e, e.len(), e.as_encoded_bytes());
    }
}
"#;

#[test]
fn run_mode_sort_vec_osstr_refs_utf8() {
    // Regression for the defect pinned on #513 (comment
    // 7224146657): a `Vec<&OsStr>` whose elements are UTF-8
    // (the `Str` value form) broke in the sort's element
    // moves -- the two-element case stopped with `type
    // mismatch in assignment: expected Ref(Adt("std::ffi::
    // OsStr")), found unsigned integer`, and longer inputs
    // corrupted an element that a later comparison then met
    // as a string against an unsigned integer. The copy
    // machinery judged a single-element source by shape
    // alone, and a `Str` value did not count as a value of
    // the `&OsStr` element type, so the copy read the
    // string's first byte as the element. (`&str` elements,
    // `&[u8]` elements and the non-UTF-8 `FatRef` form of
    // `&OsStr` all sorted fine.) The post-sort reads (`len`,
    // `as_encoded_bytes`, `Debug`) pin the moved elements'
    // contents, not just the ordering.
    run_ok(
        "sort-vec-osstr-refs-utf8",
        OSSTR_REF_SORT_SRC,
        "\"alpha\" len=5 bytes=[97, 108, 112, 104, 97]\n\
         \"bravo\" len=5 bytes=[98, 114, 97, 118, 111]\n\
         \"charlie\" len=7 bytes=[99, 104, 97, 114, 108, 105, 101]\n\
         \"delta\" len=5 bytes=[100, 101, 108, 116, 97]\n",
    );
}

const PATH_REF_SORT_SRC: &str = r#"
use std::path::{Path, PathBuf};

fn main() {
    let a = PathBuf::from("alpha");
    let b = PathBuf::from("beta");
    let z = PathBuf::from("zeta");
    let pa: &Path = &a;
    let pb: &Path = &b;
    let pz: &Path = &z;
    let mut v: Vec<&Path> = vec![pz, pb, pa];
    v.sort();
    for e in &v {
        println!("{:?} len={}", e, e.as_os_str().len());
    }
}
"#;

#[test]
fn run_mode_sort_vec_path_refs_utf8() {
    // The `&Path` pointee shares the `&OsStr` representation
    // (a UTF-8 value is the `Str` form), so a `Vec<&Path>`
    // sort crossed the same element-move judgement and
    // failed the same way.
    run_ok(
        "sort-vec-path-refs-utf8",
        PATH_REF_SORT_SRC,
        "\"alpha\" len=5\n\
         \"beta\" len=4\n\
         \"zeta\" len=4\n",
    );
}

const OSSTR_LITERAL_SORT_SRC: &str = r#"
use std::ffi::OsStr;

fn main() {
    let mut v: Vec<&OsStr> = vec![OsStr::new("banana"), OsStr::new("apple")];
    v.sort();
    println!("{:?} {:?}", v[0], v[1]);
    let mut w: Vec<&OsStr> = vec![
        OsStr::new("cherry"),
        OsStr::new("apple"),
        OsStr::new("banana"),
    ];
    w.sort_unstable();
    println!("{:?} {:?} {:?}", w[0], w[1], w[2]);
    println!("lens {} {}", v[0].len(), w[2].len());
}
"#;

#[test]
fn run_mode_sort_vec_osstr_refs_literal() {
    // Regression for the second defect of the #513 pin
    // family: with `OsStr::new` elements the sort itself
    // stopped — the unstable sort's `insert_tail` copy met
    // its own `copy_nonoverlapping` precondition check
    // (`requires that both pointer arguments are aligned and
    // non-null and the specified memory ranges do not
    // overlap`). The element pointers were priced 8 bytes
    // apart instead of 16: the tag layout table priced a
    // reference to a dynamically sized ADT (`OsStr`, `Path`,
    // `CStr`, any DST struct) as a thin pointer, knowing
    // only `str` and slices as fat pointees, so a buffer of
    // `&OsStr` elements was denominated at half its true
    // stride wherever the tag walk priced it (the buffer
    // price behind `as_mut_ptr`, `offset_from`, and the
    // range checks). `layout_of` now judges a reference or
    // raw-pointer tag's pointee against the program's DST
    // records before the static table.
    run_ok(
        "sort-vec-osstr-refs-literal",
        OSSTR_LITERAL_SORT_SRC,
        "\"apple\" \"banana\"\n\
         \"apple\" \"banana\" \"cherry\"\n\
         lens 5 6\n",
    );
}

const PATH_LITERAL_SORT_SRC: &str = r#"
use std::path::Path;

fn main() {
    let mut v: Vec<&Path> = vec![Path::new("banana"), Path::new("apple")];
    v.sort();
    println!("{:?} {:?}", v[0], v[1]);
    let mut w: Vec<&Path> = vec![Path::new("cherry"), Path::new("apple")];
    w.sort_unstable();
    println!("{:?} {:?}", w[0], w[1]);
}
"#;

#[test]
fn run_mode_sort_vec_path_refs_literal() {
    // The `&Path` literal form crossed the same thin
    // pricing as `&OsStr` (see
    // `run_mode_sort_vec_osstr_refs_literal`): element
    // pointers into a `Vec<&Path>` buffer priced 8 apart.
    run_ok(
        "sort-vec-path-refs-literal",
        PATH_LITERAL_SORT_SRC,
        "\"apple\" \"banana\"\n\
         \"apple\" \"cherry\"\n",
    );
}
