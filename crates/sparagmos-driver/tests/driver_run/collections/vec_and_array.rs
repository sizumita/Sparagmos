// Vec, slice chunks/windows and array into-iterator tests.


const CHUNKS_ITERATION_SRC: &str = r#"
fn main() {
    let v = [1, 2, 3, 4, 5];
    let mut c = v.chunks(2);
    println!("{:?}", c.next());
    println!("{:?}", c.next_back());
    println!("{:?}", c.next());
    println!("{:?}", c.next());
    let r: Vec<&[i32]> = v.rchunks(2).collect();
    println!("{:?}", r);
    let ce: Vec<&[i32]> = v.chunks_exact(2).collect();
    println!("{:?}", ce);
    println!("rem {:?}", v.chunks_exact(2).remainder());
    let mut rc = v.rchunks_exact(2);
    println!("{:?}", rc.next());
    println!("{:?}", rc.next_back());
    println!("{:?}", rc.next());
    println!("count {}", v.chunks(3).count());
    let s: i32 = v.chunks(2).map(|c| c.iter().sum::<i32>()).sum();
    println!("sum {}", s);
}
"#;

#[test]
fn run_mode_chunks_family_iteration() {
    // The slice chunk family (`Chunks` / `RChunks` and their
    // `Exact` variants) had a modelled `size_hint` but no
    // modelled step: every `next` on their real-layout state
    // ([remaining slice, chunk size], the `Exact` variants
    // adding a remainder slice) died as not runtime-supported.
    // The state machine now steps them, including the short
    // chunk each family yields last in its primary direction.
    // The outputs are native rustc 1.100.0's.
    let dir = temp_dir("chunks-iteration");
    let src = fixture(&dir, "chunks_iteration.rs", CHUNKS_ITERATION_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "chunks iteration must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "Some([1, 2])\nSome([5])\nSome([3, 4])\nNone\n[[4, 5], [2, 3], [1]]\n[[1, 2], [3, 4]]\nrem [5]\nSome([4, 5])\nSome([2, 3])\nNone\ncount 2\nsum 15\n"
    );
}
const SPARSE_VEC_HUGE_SLICE_SRC: &str = r#"
fn main() {
    let mut v: Vec<u8> = Vec::with_capacity(1_000_000_000);
    unsafe { v.set_len(1_000_000_000) }
    let s = &v[..];
    println!("len {}", s.len());
    let tail = &s[999_999_995..];
    println!("tail {}", tail.len());
    let head = &s[..7];
    println!("head {}", head.len());
    let mut w: Vec<u8> = Vec::with_capacity(16);
    w.extend_from_slice(&[1, 2, 3]);
    unsafe { w.set_len(1_000_000_000) }
    let win = &w[1..3];
    println!("win {} {}", win[0], win[1]);
    let past = &w[3..];
    println!("past {}", past.len());
    let past2 = &w[2..];
    println!("past2 {} {}", past2.len(), past2[0]);
}
"#;

#[test]
fn run_mode_sparse_vec_huge_slice_windows() {
    // A vector whose declared length runs past its stored
    // elements (the sparse form, from a `with_capacity` +
    // `set_len` into the billions): a sub-slice window inside
    // the stored elements reads as the dense slice value it
    // always was, and a window running past them keeps the
    // sparse form -- the stored prefix rebased to the window,
    // the declared length the window's -- instead of
    // materialising one host slot per element of the window
    // (a billion-slot allocation that exhausts the host).
    // Windows of a window compose onto the same forms.
    let dir = temp_dir("sparsevechugeslice");
    let src = fixture(&dir, "sparsevechugeslice.rs", SPARSE_VEC_HUGE_SLICE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "slicing a huge sparse vector must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 1000000000\ntail 5\nhead 7\nwin 2 3\npast 999999997\npast2 999999998 3\n"
    );
}

const ARRAY_INTO_ITER_CONSUMERS_O3_SRC: &str = r#"
fn main() {
    let v: Vec<i32> = [4, 5, 6].into_iter().collect();
    println!("{}", v.len());
    let mut s = 0;
    for x in &v { s += x; }
    println!("{}", s);
    let total: i32 = [1, 2, 3].into_iter().sum();
    println!("{}", total);
    println!("{}", [1, 2, 3].into_iter().count());
    let e: Vec<i32> = [].into_iter().collect();
    println!("{}", e.len());
    let w: Vec<String> = ["a".to_string(), "bb".to_string()].into_iter().collect();
    println!("{}", w.len());
    println!("{}", w[0].len() + w[1].len());
}
"#;

#[test]
fn run_mode_array_into_iter_consumers_o3() {
    // At opt-level 3 the `into_iter` construction in `main`
    // inlines, so the state a consumer builtin (`collect`,
    // `sum`, `count`) receives is core's real nested layout
    // (`IntoIter { inner: ManuallyDrop { value:
    // PolymorphicIter { alive, data } } }`), not the modelled
    // [items, next index] state the `IntoIterator` builtin
    // builds at opt-level 0. The modelled step rejected that
    // layout ("expected [items, next index], found 1
    // field(s)"); the step now decodes the real layout and
    // rebuilds it in the same shape.
    let dir = temp_dir("arrayintoiterconsumers");
    let src = fixture(&dir, "arrayintoiterconsumers.rs", ARRAY_INTO_ITER_CONSUMERS_O3_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "array into_iter consumers at -O3 must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "3\n15\n6\n3\n0\n2\n3\n"
    );
}

const ARRAY_INTO_ITER_EXTEND_O3_SRC: &str = r#"
use std::collections::VecDeque;
fn main() {
    let mut s = String::new();
    s.extend(['a', 'b', 'c']);
    println!("{}", s);
    let mut d: VecDeque<i32> = VecDeque::new();
    d.extend([1, 2, 3]);
    println!("{}", d.len());
    println!("{}", d[0] + d[1] + d[2]);
}
"#;

#[test]
fn run_mode_array_into_iter_extend_o3() {
    // The `extend` consumers drain the same real-layout
    // `array::IntoIter` state at opt-level 3 (see
    // `run_mode_array_into_iter_consumers_o3`): `String`'s
    // char extend and `VecDeque`'s extend both step it
    // through the consumer machinery.
    let dir = temp_dir("arrayintoiterextend");
    let src = fixture(&dir, "arrayintoiterextend.rs", ARRAY_INTO_ITER_EXTEND_O3_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "array into_iter extend at -O3 must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "abc\n3\n6\n");
}

const ARRAY_INTO_ITER_BY_REF_O3_SRC: &str = r#"
fn main() {
    let mut it = [1, 2, 3].into_iter();
    let v: Vec<i32> = it.by_ref().take(2).collect();
    println!("{}", v.len());
    println!("{}", it.next().unwrap());
    println!("{}", it.next().is_none());
    let mut it2 = [4, 5, 6].into_iter();
    println!("{}", it2.next_back().unwrap());
    println!("{}", it2.next().unwrap());
    println!("{}", it2.next_back().unwrap());
    println!("{}", it2.next().is_none());
}
"#;

#[test]
fn run_mode_array_into_iter_by_ref_rebuild_o3() {
    // A builtin consumer that partially drains a real-layout
    // `array::IntoIter` (here `by_ref().take(2).collect()`)
    // writes the stepped state back in the same real layout,
    // so the guest's own inlined `next` that runs afterwards
    // reads the narrowed alive range -- and the back step
    // (`next_back`) narrows it from the other end, the two
    // composing on one state.
    let dir = temp_dir("arrayintoiterbyref");
    let src = fixture(&dir, "arrayintoiterbyref.rs", ARRAY_INTO_ITER_BY_REF_O3_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "array into_iter by_ref/back steps at -O3 must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "2\n3\ntrue\n6\n4\n5\ntrue\n"
    );
}