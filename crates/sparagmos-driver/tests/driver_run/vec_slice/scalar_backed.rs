// Scalar-backed slice tests.

const SLICE_ITER_MUT_SCALAR_BACKED_SRC: &str = r#"
use std::iter::ExactSizeIterator;
fn main() {
    let mut x = 41i32;
    let mut it = std::slice::from_mut(&mut x).iter_mut();
    println!("len {}", ExactSizeIterator::len(&it));
    println!("view {}", it.as_slice().len());
    let rest = it.into_slice();
    println!("rest {}", rest.len());
    let mut w = 900u64;
    let itw = std::slice::from_mut(&mut w).iter_mut();
    println!("wlen {}", ExactSizeIterator::len(&itw));
    println!("wview {}", itw.as_slice().len());
}
"#;

#[test]
fn run_mode_slice_iter_mut_scalar_backed_len() {
    // PR #387, Codex round 2: over a scalar-backed slice
    // (`slice::from_mut(&mut x)`), the iterator's front
    // pointer denotes the scalar place while the end pointer
    // denotes the same place with a trailing
    // `ByteOffset(size_of::<T>())`. The index reading of the
    // unified bounds drops that offset and reports an empty
    // iterator, and the snapshot-address fallback cannot read
    // a storage-backed pair at all, so the modelled `len`
    // and slice views failed instead of returning the
    // one-element result. They now read the pair's distance
    // through the same storage-backed offset reading as
    // `offset_from` before falling back to the bounds.
    let dir = temp_dir("slice-iter-mut-scalar-backed");
    let src = fixture(&dir, "case.rs", SLICE_ITER_MUT_SCALAR_BACKED_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "len 1\nview 1\nrest 1\nwlen 1\nwview 1\n",
        "stdout={stdout}"
    );
}

const SLICE_SCALAR_BACKED_ACCESS_SRC: &str = r#"
fn main() {
    let mut x: i32 = 41;
    {
        let mut it = std::slice::from_mut(&mut x).iter_mut();
        println!("n1 {:?}", it.next().map(|v| *v));
        println!("n2 {:?}", it.next());
        println!("nb {:?}", it.next_back());
    }
    println!("x {}", x);
    let mut y: u64 = 900;
    {
        let mut it = std::slice::from_mut(&mut y).iter_mut();
        if let Some(v) = it.next() { *v = 1234; }
        println!("after {:?}", it.next());
    }
    println!("y {}", y);
    let mut z: u64 = 77;
    {
        let s = std::slice::from_mut(&mut z);
        let mut it = s.iter();
        println!("hint {:?}", it.size_hint());
        println!("back {:?}", it.next_back().map(|v| *v));
        println!("front {:?}", it.next().map(|v| *v));
        println!("len {}", std::iter::ExactSizeIterator::len(&it));
    }
    let w: u64 = 900;
    let sr = std::slice::from_ref(&w);
    println!("idx {}", sr[0]);
    println!("ptr {}", unsafe { *sr.as_ptr() });
    let mut sum = 0u64;
    for v in sr { sum += *v; }
    println!("sum {}", sum);
    let mut it = sr.iter();
    println!("first {:?}", it.next().map(|v| *v));
    println!("second {:?}", it.next());
    let n: i32 = -5;
    println!("neg {}", std::slice::from_ref(&n)[0]);
    let mut m: u32 = 0x01020304;
    {
        let sm = std::slice::from_mut(&mut m);
        sm[0] = 0xAABBCCDD;
    }
    println!("m {:x}", m);
}
"#;

#[test]
fn run_mode_slice_scalar_backed_access() {
    // Issue #413: over a slice formed on a scalar's own
    // storage (`slice::from_ref` / `slice::from_mut` of a
    // scalar place), three readings disagreed with the shape.
    // The step machine's index-pair shortcut read the
    // iterator's front and end pointers by their trailing
    // index step only, dropping the end pointer's byte offset,
    // so `next()` reported the one element as consumed. An
    // element access appended a bare index step to the scalar
    // place, which the projectors read as a byte of the
    // storage (900 read back as 132). And the shared `Iter`
    // construction rejected the scalar the snapshot read found
    // where a container was expected. Expected output is the
    // native result of the same source.
    let dir = temp_dir("slice-scalar-backed-access");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_ACCESS_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "n1 Some(41)\nn2 None\nnb None\nx 41\nafter None\ny 1234\nhint (1, Some(1))\nback Some(77)\nfront None\nlen 0\nidx 900\nptr 900\nsum 900\nfirst Some(900)\nsecond None\nneg -5\nm aabbccdd\n",
        "stdout={stdout}"
    );
}

const SLICE_SCALAR_BACKED_BYTE_VIEW_SRC: &str = r#"
fn main() {
    let w: u64 = 900;
    let b0 = unsafe { *(&w as *const u64 as *const u8) };
    let b1 = unsafe { *(&w as *const u64 as *const u8).add(1) };
    println!("bytes {} {}", b0, b1);
    let bs = unsafe { std::slice::from_raw_parts(&w as *const u64 as *const u8, 8) };
    let mut itb = bs.iter();
    println!("b0 {:?}", itb.next());
    println!("b1 {:?}", itb.next());
    println!("blen {}", bs.len());
    println!("bidx {}", bs[1]);
}
"#;

#[test]
fn run_mode_slice_scalar_backed_byte_view() {
    // The byte views of a scalar's storage keep their
    // byte-wise reading (a byte pointer into the storage, and
    // a byte slice over it): only an element the width of the
    // storage denotes the scalar itself. Expected output is
    // the native result of the same source.
    let dir = temp_dir("slice-scalar-backed-byte-view");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_BYTE_VIEW_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "bytes 132 3\nb0 Some(132)\nb1 Some(3)\nblen 8\nbidx 3\n",
        "stdout={stdout}"
    );
}

const SLICE_SCALAR_BACKED_PUNNED_SRC: &str = r#"
fn main() {
    let w: u64 = 0x1_0000_0302;
    println!("idx {}", unsafe { std::slice::from_raw_parts(&w as *const u64 as *const u32, 2) }[0]);
}
"#;

#[test]
fn run_mode_slice_scalar_backed_punned_view_declines() {
    // A punned element width over a scalar's storage (u32
    // elements over a u64) holds no elements as values: the
    // access declines with an explicit error instead of
    // reading a byte of the storage (the pre-#413 reading
    // printed 2 here, the low byte, where the native element
    // is 770).
    let dir = temp_dir("slice-scalar-backed-punned");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_PUNNED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "stderr={stderr}");
    assert!(
        stderr.contains("punned element width"),
        "stderr={stderr}"
    );
}

const SLICE_SCALAR_BACKED_BOOL_SRC: &str = r#"
fn main() {
    let t = true;
    let s = std::slice::from_ref(&t);
    let mut it = s.iter();
    println!("first {:?}", it.next());
    println!("second {:?}", it.next());
    let f = false;
    let mut sum = 0;
    for v in std::slice::from_ref(&f) {
        if *v {
            sum += 1;
        }
    }
    println!("sum {sum}");
    println!("idx {}", s[0]);
}
"#;

#[test]
fn run_mode_slice_scalar_backed_bool_element() {
    // A one-byte element of the storage's own type is the
    // scalar itself, not its byte: `slice::from_ref(&true)`
    // iterated as a byte snapshot yields `Uint(1)`, whose
    // pointee cannot be read as the `bool` the iterator
    // promises. Expected output is the native result of the
    // same source.
    let dir = temp_dir("slice-scalar-backed-bool");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_BOOL_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "first Some(true)\nsecond None\nsum 0\nidx true\n",
        "stdout={stdout}"
    );
}

const SLICE_SCALAR_BACKED_FLOAT_VIEW_SRC: &str = r#"
#![feature(f16)]
fn main() {
    let x: f32 = 1.0;
    let bs = unsafe { std::slice::from_raw_parts(&x as *const f32 as *const u8, 4) };
    let mut it = bs.iter();
    println!("b0 {:?}", it.next());
    println!("b1 {:?}", it.next());
    println!("b2 {:?}", it.next());
    println!("b3 {:?}", it.next());
    let y: f64 = 1.0;
    let bs8 = unsafe { std::slice::from_raw_parts(&y as *const f64 as *const u8, 8) };
    let v: Vec<u8> = bs8.iter().copied().collect();
    println!("f64 {v:?}");
    let h: f16 = 1.0;
    let bsh = unsafe { std::slice::from_raw_parts(&h as *const f16 as *const u8, 2) };
    let vh: Vec<u8> = bsh.iter().copied().collect();
    println!("f16 {vh:?}");
}
"#;

#[test]
fn run_mode_slice_scalar_backed_float_byte_view() {
    // A byte view over a float's storage reads the bytes of
    // the float's own width under its declared layout, not
    // the widened model form's: a four-byte view of `1.0f32`
    // iterates as [0, 0, 128, 63]. Expected output is the
    // native result of the same source.
    let dir = temp_dir("slice-scalar-backed-float-view");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_FLOAT_VIEW_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "b0 Some(0)\nb1 Some(0)\nb2 Some(128)\nb3 Some(63)\nf64 [0, 0, 0, 0, 0, 0, 240, 63]\nf16 [0, 60]\n",
        "stdout={stdout}"
    );
}

const SLICE_SCALAR_BACKED_SIGNED_BYTE_SRC: &str = r#"
fn main() {
    let n = -1i8;
    let s = std::slice::from_ref(&n);
    let mut it = s.iter();
    println!("first {:?}", it.next());
    println!("second {:?}", it.next());
    let m = -128i8;
    for v in std::slice::from_ref(&m) {
        println!("elem {v}");
    }
    let x: u32 = 0x80FF_7F01;
    let bs = unsafe { std::slice::from_raw_parts(&x as *const u32 as *const i8, 4) };
    let v: Vec<i8> = bs.iter().copied().collect();
    println!("view {v:?}");
    let p = 127i8;
    let s2 = std::slice::from_ref(&p);
    println!("pos {:?}", s2.iter().next());
}
"#;

#[test]
fn run_mode_slice_scalar_backed_signed_byte_element() {
    // A byte element of a signed type denotes the byte's
    // signed value: `slice::from_ref(&-1i8)` snapshotted as
    // an unsigned byte yields `Uint(255)`, which a consumer
    // resolving the yielded reference directly (the
    // formatting path) prints as 255 instead of -1. The same
    // holds for a signed byte view over a wider storage.
    // Expected output is the native result of the same
    // source.
    let dir = temp_dir("slice-scalar-backed-signed-byte");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_SIGNED_BYTE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "first Some(-1)\nsecond None\nelem -128\nview [1, 127, -1, -128]\npos Some(127)\n",
        "stdout={stdout}"
    );
}

const SLICE_SCALAR_BACKED_BYTE_VIEW_OF_BOOL_SRC: &str = r#"
fn main() {
    let b = true;
    let bytes = unsafe { std::slice::from_raw_parts(&b as *const bool as *const u8, 1) };
    let n: u8 = bytes[0];
    println!("n={n}");
    println!("r={:?}", &bytes[0]);
    let b2 = false;
    let bytes2 = unsafe { std::slice::from_raw_parts(&b2 as *const bool as *const u8, 1) };
    println!("n2={}", bytes2[0]);
    println!("r2={:?}", &bytes2[0]);
    let m = -1i8;
    let sbytes = unsafe { std::slice::from_raw_parts(&m as *const i8 as *const u8, 1) };
    println!("s={:?}", &sbytes[0]);
    let bb = unsafe { std::slice::from_raw_parts(&b as *const bool as *const bool, 1) };
    println!("same={} {:?}", bb[0], &bb[0]);
}
"#;

#[test]
fn run_mode_slice_scalar_backed_byte_view_of_bool() {
    // A byte view of a boolean's storage denotes the storage's
    // byte, not the boolean: the same-width shortcut in place
    // resolution left the element position on the boolean
    // itself, so a consumer resolving the element reference
    // directly (the formatting path) printed `true` instead of
    // `1`. The same held for a byte view of an `i8` (printed
    // `-1` instead of `255`). A same-kind view still denotes
    // the scalar itself. Expected output is the native result
    // of the same source.
    let dir = temp_dir("slice-scalar-backed-byte-view-of-bool");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_BYTE_VIEW_OF_BOOL_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "n=1\nr=1\nn2=0\nr2=0\ns=255\nsame=true true\n",
        "stdout={stdout}"
    );
}

const SLICE_SCALAR_BACKED_BYTE_VIEW_AS_PTR_SRC: &str = r#"
fn main() {
    let b = true;
    let bytes = unsafe { std::slice::from_raw_parts(&b as *const bool as *const u8, 1) };
    let p: *const u8 = bytes.as_ptr();
    let n: u8 = unsafe { *p };
    println!("n={n}");
    let r: &u8 = unsafe { &*bytes.as_ptr() };
    println!("r={r}");
    let b2 = false;
    let bytes2 = unsafe { std::slice::from_raw_parts(&b2 as *const bool as *const u8, 1) };
    let r2: &u8 = unsafe { &*bytes2.as_ptr() };
    println!("r2={r2}");
    let m = -1i8;
    let sbytes = unsafe { std::slice::from_raw_parts(&m as *const i8 as *const u8, 1) };
    let sr: &u8 = unsafe { &*sbytes.as_ptr() };
    println!("s={sr}");
    let same = unsafe { std::slice::from_raw_parts(&b as *const bool as *const bool, 1) };
    let br: &bool = unsafe { &*same.as_ptr() };
    println!("same={br}");
    let mut b3 = false;
    let mbytes = unsafe { std::slice::from_raw_parts_mut(&mut b3 as *mut bool as *mut u8, 1) };
    unsafe { *mbytes.as_mut_ptr() = 1u8; }
    println!("w={b3}");
}
"#;

#[test]
fn run_mode_slice_scalar_backed_byte_view_as_ptr() {
    // The first-element pointer a byte view of a scalar's
    // storage extracts (`as_ptr` / `as_mut_ptr`) denotes the
    // storage's byte, not the scalar itself: the scalar
    // shortcut in the pointer extraction returned the storage
    // pointer as it stands, so a consumer resolving the
    // reference the pointer forms directly printed the boolean
    // (`true` instead of `1`), and a byte write through
    // `as_mut_ptr` met the boolean place's type check and
    // failed. A same-kind view's pointer still denotes the
    // scalar itself. Expected output is the native result of
    // the same source.
    let dir = temp_dir("slice-scalar-backed-byte-view-as-ptr");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_BYTE_VIEW_AS_PTR_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "n=1\nr=1\nr2=0\ns=255\nsame=true\nw=true\n",
        "stdout={stdout}"
    );
}
