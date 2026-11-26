// Tests run at MIR opt level 3.

const SLICE_ITER_ENUMERATE_CONST_PTR_SRC: &str = r#"
trait Dimension {
    fn slice(&self) -> &[usize];
}
impl Dimension for () {
    fn slice(&self) -> &[usize] { &[] }
}
impl Dimension for usize {
    fn slice(&self) -> &[usize] {
        unsafe { std::slice::from_raw_parts(self, 1) }
    }
}
fn main() {
    let tup = ();
    let mut it = tup.slice().iter();
    for (i, _) in 3usize.slice().iter().enumerate() {
        if i == 0 { continue; }
        it.next();
    }
    println!("done");
}
"#;

#[test]
fn run_mode_slice_iter_enumerate_const_ptr_o3() {
    // The boundary of the scalar-backed first-element rule:
    // a slice formed over a pointer to a constant value (the
    // `from_raw_parts` over a promoted `&3` shape, at the
    // opt level the shape's regression test compiles at) is
    // not storage-backed — its element position is the
    // appended index the offset machinery advances, and
    // iterating it must keep working.
    let dir = temp_dir("slice-iter-enumerate-const-ptr");
    let src = fixture(&dir, "case.rs", SLICE_ITER_ENUMERATE_CONST_PTR_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "done\n", "stdout={stdout}");
}

const VEC_DEBUG_O3_SRC: &str = r#"
fn main() {
    let v = vec![1, 2, 3];
    println!("{:?}", v);
    println!("{:#?}", v);
}
"#;

#[test]
fn run_mode_vec_debug_o3() {
    // At opt-level 3 the inlined construction leaves the
    // vector in its real aggregate form ([RawVec, len]); its
    // `Debug` rendering must read the elements back through
    // the buffer pointer instead of treating the fields as
    // the element list.
    let dir = temp_dir("vec-debug-o3");
    let src = fixture(&dir, "case.rs", VEC_DEBUG_O3_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "[1, 2, 3]\n[\n    1,\n    2,\n    3,\n]\n", "stdout={stdout}");
}

const VEC_DEBUG_STRING_O3_SRC: &str = r#"
fn main() {
    let v = vec![String::from("a"), String::from("bb")];
    println!("{:?}", v);
    println!("{:#?}", v);
}
"#;

#[test]
fn run_mode_vec_debug_string_elements_o3() {
    // The elements of a real-form vector of `String`s are
    // themselves aggregates in the buffer's storage; the
    // typed element read must rebuild them (a bulk snapshot
    // of the storage flattens their fields instead).
    let dir = temp_dir("vec-debug-string-o3");
    let src = fixture(&dir, "case.rs", VEC_DEBUG_STRING_O3_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "[\"a\", \"bb\"]\n[\n    \"a\",\n    \"bb\",\n]\n",
        "stdout={stdout}"
    );
}

const VEC_DEBUG_NESTED_O3_SRC: &str = r#"
fn main() {
    let v = vec![vec![1, 2], vec![3]];
    println!("{:?}", v);
    let e: Vec<i32> = Vec::new();
    println!("{:?}", e);
}
"#;

#[test]
fn run_mode_vec_debug_nested_and_empty_o3() {
    // A real-form vector of vectors: each element read back
    // is itself a real-form vector and renders through the
    // same path. An empty vector's dangling buffer pointer
    // denotes no elements.
    let dir = temp_dir("vec-debug-nested-o3");
    let src = fixture(&dir, "case.rs", VEC_DEBUG_NESTED_O3_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "[[1, 2], [3]]\n[]\n", "stdout={stdout}");
}

const VEC_RESIZE_OPT_SRC: &str = r#"
fn main() {
    // At opt-level 3 the reserve and the element writes are
    // inlined into one body: the vector value the guest built
    // is the real RawVec aggregate, whose buffer pointer
    // field still holds the dangling pointer after the
    // capacity grows -- the growth must install storage
    // behind it for the writes in extend_with to land (the
    // shape tests/ui/moves/bycopy_untupled.rs exercises in
    // its opt revision).
    let mut v: Vec<usize> = Vec::new();
    v.resize(2, 1);
    assert_eq!(v.len(), 2);
    assert_eq!(v[0], 1);
    assert_eq!(v[1], 1);
    println!("vec-resize-opt-ok");
}
"#;

#[test]
fn run_mode_vec_resize_opt_level_3() {
    let dir = temp_dir("vec-resize-opt");
    let src = fixture(&dir, "vec_resize_opt.rs", VEC_RESIZE_OPT_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec resize at opt-level 3 must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "vec-resize-opt-ok\n");
}

const VEC_PUSH_GROW_O3_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    let ok = v.len() == 3
        && v.capacity() >= 3
        && v[0] == 1
        && v[1] == 2
        && v[2] == 3;
    std::process::exit(if ok { 0 } else { 3 });
}
"#;

/// At opt-level 3 a `Vec::push` that reallocates runs the
/// collected `RawVec::grow_one` body, which reads the capacity
/// through the `&mut RawVec` / `&mut RawVecInner` references the
/// caller formed over the vector's buffer field and writes the
/// grown capacity and buffer pointer back through them. The
/// vector a builtin built is in its real aggregate form, so
/// those references must denote the aggregate's own field
/// storage: completing the buffer-chain walk as the bare
/// buffer pointer loses the vector place, the capacity read
/// fabricates an empty aggregate for the `UsizeNoHighBit`
/// slot, and the run stops with `type mismatch in assignment:
/// expected Uint(Usize), found ADT value`.
#[test]
fn run_mode_vec_push_grows_from_empty_o3() {
    let dir = temp_dir("vec-push-grow-o3");
    let src = fixture(&dir, "case.rs", VEC_PUSH_GROW_O3_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "O3 push with reallocation must run to completion; stderr={stderr}"
    );
}

const VEC_PUSH_GROW_WITH_CAPACITY_O3_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::with_capacity(2);
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    let ok = v.len() == 4
        && v.capacity() >= 4
        && v[0] == 1
        && v[1] == 2
        && v[2] == 3
        && v[3] == 4;
    std::process::exit(if ok { 0 } else { 3 });
}
"#;

/// The same growth through a vector whose buffer already lives
/// in a heap allocation: the reallocation must copy the stored
/// elements into the new buffer and record the new capacity in
/// the vector's aggregate.
#[test]
fn run_mode_vec_push_grows_past_with_capacity_o3() {
    let dir = temp_dir("vec-push-grow-with-capacity-o3");
    let src = fixture(&dir, "case.rs", VEC_PUSH_GROW_WITH_CAPACITY_O3_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "O3 push past with_capacity must run to completion; stderr={stderr}"
    );
}

const VEC_RESERVE_GROW_O3_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::new();
    v.push(1);
    v.reserve(10);
    let mut ok = v.capacity() >= 11 && v.len() == 1 && v[0] == 1;
    for i in 0..10 {
        v.push(i);
    }
    ok = ok && v.len() == 11 && v[10] == 9;
    std::process::exit(if ok { 0 } else { 3 });
}
"#;

/// `reserve` grows through the same `RawVec` growth path as a
/// reallocating `push` (here without a following reallocation:
/// the pushes afterwards fit the reserved capacity).
#[test]
fn run_mode_vec_reserve_grows_o3() {
    let dir = temp_dir("vec-reserve-grow-o3");
    let src = fixture(&dir, "case.rs", VEC_RESERVE_GROW_O3_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "O3 reserve growth must run to completion; stderr={stderr}"
    );
}

const VEC_MANY_PUSHES_O3_SRC: &str = r#"
fn main() {
    let mut v: Vec<u64> = Vec::new();
    for i in 0..100u64 {
        v.push(i * 3);
    }
    let mut ok = v.len() == 100 && v.capacity() >= 100;
    for i in 0..100u64 {
        if v[i as usize] != i * 3 {
            ok = false;
        }
    }
    std::process::exit(if ok { 0 } else { 3 });
}
"#;

/// Repeated growth: every reallocation moves the stored
/// elements to the new buffer, so a hundred pushes read back
/// exactly what was pushed.
#[test]
fn run_mode_vec_many_pushes_realloc_o3() {
    let dir = temp_dir("vec-many-pushes-o3");
    let src = fixture(&dir, "case.rs", VEC_MANY_PUSHES_O3_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "O3 repeated push growth must run to completion; stderr={stderr}"
    );
}

const VEC_EXTEND_RANGE_O3_SRC: &str = r#"
fn main() {
    let mut v = vec![0];
    v.extend(1..4);
    let mut ok = v.len() == 4 && v[0] == 0 && v[1] == 1 && v[2] == 2 && v[3] == 3;
    let mut w: Vec<i32> = Vec::new();
    w.extend(1..4);
    ok = ok && w.len() == 3 && w[0] == 1 && w[2] == 3;
    let mut u = vec![0, 0];
    u.extend(1..4);
    ok = ok && u.len() == 5 && u[4] == 3;
    v.extend(4..6);
    ok = ok && v.len() == 6 && v[4] == 4 && v[5] == 5;
    std::process::exit(if ok { 0 } else { 3 });
}
"#;

/// `Vec::extend` over a `Range` at opt-level 3: the vector a
/// converted body built stands in its real aggregate form, and
/// the append must write the new elements into the buffer
/// storage and advance the aggregate's length field. Appending
/// them as fields of the aggregate itself left the length
/// reading as the field count (a three-element extend of
/// `vec![0]` reported length 5) and the buffer aggregate where
/// the first element belongs.
#[test]
fn run_mode_vec_extend_range_o3() {
    let dir = temp_dir("vec-extend-range-o3");
    let src = fixture(&dir, "case.rs", VEC_EXTEND_RANGE_O3_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "O3 extend over a Range must keep the length and elements; stderr={stderr}"
    );
}

const VEC_EXTEND_SLICE_O3_SRC: &str = r#"
fn main() {
    let mut v = vec![0];
    v.extend([1, 2, 3].iter().copied());
    let mut ok = v.len() == 4 && v[0] == 0 && v[1] == 1 && v[2] == 2 && v[3] == 3;
    let mut w = vec![0];
    w.extend_from_slice(&[1, 2, 3]);
    ok = ok && w.len() == 4 && w[0] == 0 && w[1] == 1 && w[2] == 2 && w[3] == 3;
    std::process::exit(if ok { 0 } else { 3 });
}
"#;

/// The slice-shaped appends at opt-level 3 (an iterator over a
/// slice, and `extend_from_slice`) share the extend append's
/// duty on a real-form vector (see
/// `run_mode_vec_extend_range_o3`).
#[test]
fn run_mode_vec_extend_slice_o3() {
    let dir = temp_dir("vec-extend-slice-o3");
    let src = fixture(&dir, "case.rs", VEC_EXTEND_SLICE_O3_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "O3 slice extends must keep the length and elements; stderr={stderr}"
    );
}

const VEC_WRITE_FMT_O3_SRC: &str = r#"
use std::io::Write;
fn main() {
    let mut v: Vec<u8> = Vec::new();
    let _ = write!(v, "hello");
    let sum: u32 = v.iter().map(|b| *b as u32).sum();
    let mut ok = v.len() == 5 && sum == 532;
    let mut w = vec![b'x'];
    let _ = write!(w, "yz");
    let wsum: u32 = w.iter().map(|b| *b as u32).sum();
    ok = ok && w.len() == 3 && wsum == 363;
    std::process::exit(if ok { 0 } else { 3 });
}
"#;

/// `write!` into a `Vec<u8>` at opt-level 3 appends through
/// the same vector append as `extend` (see
/// `run_mode_vec_extend_range_o3`): on a real-form vector the
/// bytes land in the buffer storage and the length advances
/// by the byte count.
#[test]
fn run_mode_vec_write_fmt_o3() {
    let dir = temp_dir("vec-write-fmt-o3");
    let src = fixture(&dir, "case.rs", VEC_WRITE_FMT_O3_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "O3 write! into a Vec<u8> must append the bytes; stderr={stderr}"
    );
}
