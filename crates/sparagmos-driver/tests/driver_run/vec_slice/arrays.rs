// Array tests.

const ARRAY_PTR_COPY_ADT_SRC: &str = r#"
struct E(i32);
fn main() {
    // The element copy behind `Vec::insert`'s right shift and
    // `Vec::remove`'s left shift, directly over an array of ADT
    // elements: elements move whole, never byte-composed.
    let mut a: [E; 6] = [E(0), E(1), E(2), E(3), E(4), E(5)];
    unsafe {
        let p = a.as_mut_ptr();
        std::ptr::copy(p.add(2), p.add(1), 4);
    }
    assert_eq!((a[0].0, a[1].0, a[2].0, a[3].0, a[4].0, a[5].0), (0, 2, 3, 4, 5, 5));
    let mut b: [E; 6] = [E(0), E(1), E(2), E(3), E(4), E(5)];
    unsafe {
        let p = b.as_mut_ptr();
        std::ptr::copy(p.add(1), p.add(2), 4);
        std::ptr::write(p.add(1), E(99));
    }
    assert_eq!((b[0].0, b[1].0, b[2].0, b[3].0, b[4].0, b[5].0), (0, 99, 1, 2, 3, 4));
    println!("array-copy ok");
}
"#;

#[test]
fn run_mode_ptr_copy_over_adt_array_elements() {
    let dir = temp_dir("array-ptr-copy-adt");
    let src = fixture(&dir, "array_ptr_copy_adt.rs", ARRAY_PTR_COPY_ADT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "ptr::copy over ADT elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "array-copy ok\n");
}

const HUGE_ZST_ARRAY_CONST_SRC: &str = r#"
fn main() {
    println!("{}", [(); usize::MAX].len());
}
"#;

#[test]
fn convert_mode_huge_zst_array_constant_converts() {
    // A constant of a huge zero-sized array type (`[(); usize::MAX]`,
    // the shape of rustc's own compile-time regression test 68010)
    // has an empty allocation; decoding it element by element would
    // materialise `usize::MAX` values and abort the host process.
    // The constant is not materialised past the decode bound, and
    // the program converts.
    let dir = temp_dir("mono-huge-zst-array-const");
    let src = fixture(&dir, "huge_zst_array_const.rs", HUGE_ZST_ARRAY_CONST_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a huge ZST array constant must not be materialised element \
         by element and the program must convert; stderr={stderr}"
    );
}

const TRANSMUTE_ARRAY_SRC: &str = r#"
fn inp() -> u32 { let mut s = 0u32; let mut i = 0; while i < 41 { s += 1; i += 1; } s }
fn main() {
    let n = inp() + 0x04030200 - 41;
    let a: [u8; 4] = unsafe { std::mem::transmute(n) };
    let back: u32 = unsafe { std::mem::transmute(a) };
    let wide: u64 = unsafe { std::mem::transmute([1u16, 2, 3, 4]) };
    let pair: u64 = unsafe { std::mem::transmute([n, n + 1]) };
    let halves: [u32; 2] = unsafe { std::mem::transmute(pair) };
    let fpair: u64 = unsafe { std::mem::transmute([1.5f32, 2.5f32]) };
    println!("{:?} {:x} {:x} {:x} {:?} {:x}", a, back, wide, pair, halves, fpair);
}
"#;

#[test]
fn run_mode_transmute_arrays_and_scalars() {
    run_ok(
        "transmute-arrays",
        TRANSMUTE_ARRAY_SRC,
        "[0, 2, 3, 4] 4030200 4000300020001 403020104030200 [67305984, 67305985] 402000003fc00000\n",
    );
}

const TRANSMUTE_LONG_SCALAR_ARRAY_SRC: &str = r#"
fn main() {
    let mut a: [u32; 40] = [0; 40];
    for i in 0..40 { a[i] = 0x3fc00000 + (i as u32) * 0x10000; }
    let f: [f32; 40] = unsafe { std::mem::transmute(a) };
    println!("{} {} {}", f[0], f[1], f[39]);
    let back: [u32; 40] = unsafe { std::mem::transmute(f) };
    println!("{:x} {:x}", back[0], back[39]);
}
"#;

#[test]
fn run_mode_transmute_long_scalar_arrays() {
    // The same over-budget length with element shapes
    // that do not line up structurally (integer ->
    // float): the evaluation reinterprets element by
    // element instead of packing the whole array into
    // one 128-bit pattern.
    run_ok(
        "transmute-long-scalar-arrays",
        TRANSMUTE_LONG_SCALAR_ARRAY_SRC,
        "1.5 1.5078125 1.8046875\n3fc00000 3fe70000\n",
    );
}

// ----- large array fills (Issue #140) ------------------------------------------

const LARGE_FILL_SRC: &str = r#"
use std::mem::MaybeUninit;

fn main() {
    // `mem::zeroed` of a large scalar array materialises the
    // whole zero tree.
    let z: [u64; 5000] = unsafe { std::mem::zeroed() };
    assert_eq!(z[4999], 0);
    let nested: [[u16; 300]; 300] = unsafe { std::mem::zeroed() };
    assert_eq!(nested[299][299], 0);
    // `write_bytes` whose element type is the whole array (the
    // shape `mem::zeroed` takes under optimisation) fills it in
    // one operation.
    let mut un: MaybeUninit<[u8; 1048576]> = MaybeUninit::uninit();
    unsafe {
        std::ptr::write_bytes(un.as_mut_ptr(), 0u8, 1);
        let arr = un.assume_init_read();
        assert_eq!(arr[0], 0);
        assert_eq!(arr[1048575], 0);
        std::ptr::write_bytes(un.as_mut_ptr(), 0x5Au8, 1);
        let arr = un.assume_init_read();
        assert_eq!(arr[524288], 0x5A);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_large_array_zeroed_and_write_bytes_execute() {
    let dir = temp_dir("large-fill");
    let src = fixture(&dir, "large_fill.rs", LARGE_FILL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "large array zeroed / write_bytes fills must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

// ----- Nested arrays: an element that IS an array ---------------------------

const NESTED_ARRAY_ITER_READ_SRC: &str = r#"
fn fill(a: &mut [[u64; 3]; 4]) {
    for i in 0..4 { for j in 0..3 { a[i][j] = (10 * i + j) as u64; } }
}
fn main() {
    let mut a: [[u64; 3]; 4] = [[0; 3]; 4];
    fill(&mut a);
    for e in &a { println!("{} {} {}", e[0], e[1], e[2]); }
    for e in a.iter() { println!("{} {} {}", e[0], e[1], e[2]); }
}
"#;

#[test]
fn run_mode_nested_array_iter_index_reads() {
    // Indexing through the references a slice iterator over a
    // nested array yields must denominate by the inner array:
    // `e[k]` is element k of the inner array `e` denotes, never
    // the k-th following element of the outer array.
    let dir = temp_dir("nested-array-iter-read");
    let src = fixture(&dir, "nested_array_iter_read.rs", NESTED_ARRAY_ITER_READ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested array iteration must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "0 1 2\n10 11 12\n20 21 22\n30 31 32\n0 1 2\n10 11 12\n20 21 22\n30 31 32\n"
    );
}

const NESTED_ARRAY_ELEMENT_REF_SRC: &str = r#"
fn fill(a: &mut [[u64; 3]; 4]) {
    for i in 0..4 { for j in 0..3 { a[i][j] = (10 * i + j) as u64; } }
}
fn show(e: &[u64; 3]) { println!("{} {} {}", e[0], e[1], e[2]); }
fn main() {
    let mut a: [[u64; 3]; 4] = [[0; 3]; 4];
    fill(&mut a);
    show(&a[0]);
    show(&a[1]);
    show(&a[3]);
    let e: &mut [u64; 3] = &mut a[1];
    e[1] = 555;
    show(&a[1]);
}
"#;

#[test]
fn run_mode_nested_array_element_ref_index() {
    // A plain borrow of one inner array (`&a[i]`), indexed
    // through, at the container's first, middle and last
    // positions, and a mutable borrow's write landing in the
    // borrowed element alone.
    let dir = temp_dir("nested-array-element-ref");
    let src = fixture(&dir, "nested_array_element_ref.rs", NESTED_ARRAY_ELEMENT_REF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "indexing a borrowed inner array must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "0 1 2\n10 11 12\n30 31 32\n10 555 12\n"
    );
}

const NESTED_ARRAY_ITER_MUT_WRITE_SRC: &str = r#"
fn main() {
    let mut a: [[u64; 3]; 4] = [[0; 3]; 4];
    for i in 0..4 { for j in 0..3 { a[i][j] = (10 * i + j) as u64; } }
    for (i, e) in a.iter_mut().enumerate() { e[1] = 1000 + i as u64; e[2] = 2000 + i as u64; }
    for i in 0..4 { println!("{} {} {}", a[i][0], a[i][1], a[i][2]); }
}
"#;

#[test]
fn run_mode_nested_array_iter_mut_writes_land_in_element() {
    // Writes through `iter_mut` element references land in the
    // element the reference denotes, at the written component.
    let dir = temp_dir("nested-array-iter-mut-write");
    let src = fixture(&dir, "nested_array_iter_mut_write.rs", NESTED_ARRAY_ITER_MUT_WRITE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iter_mut over a nested array must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "0 1000 2000\n10 1001 2001\n20 1002 2002\n30 1003 2003\n"
    );
}

const NESTED_ARRAY_BOXED_COPY_SRC: &str = r#"
fn main() {
    let mut b: Box<[[u64; 3]; 4]> = Box::new([[0; 3]; 4]);
    for i in 0..4 { for j in 0..3 { b[i][j] = (10 * i + j) as u64; } }
    for e in &*b { println!("{} {} {}", e[0], e[1], e[2]); }
    let mut a: [[u64; 3]; 4] = [[0; 3]; 4];
    for i in 0..4 { for j in 0..3 { a[i][j] = (10 * i + j) as u64; } }
    for e in &a { let v: [u64; 3] = *e; println!("{} {} {}", v[0], v[1], v[2]); }
}
"#;

#[test]
fn run_mode_nested_array_boxed_iter_and_element_copy() {
    // The boxed nested array iterates with the same inner
    // denomination, and a whole-value copy of a yielded element
    // (`let v = *e`) copies that element alone.
    let dir = temp_dir("nested-array-boxed-copy");
    let src = fixture(&dir, "nested_array_boxed_copy.rs", NESTED_ARRAY_BOXED_COPY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "boxed nested array iteration must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "0 1 2\n10 11 12\n20 21 22\n30 31 32\n0 1 2\n10 11 12\n20 21 22\n30 31 32\n"
    );
}

const FLAT_BUFFER_ARRAY_CARVE_SRC: &str = r#"
fn main() {
    let buf: [u64; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    unsafe {
        let a0 = &*(buf.as_ptr() as *const [u64; 3]);
        println!("{} {} {}", a0[0], a0[1], a0[2]);
        let a1 = &*(buf.as_ptr().add(4) as *const [u64; 3]);
        println!("{} {} {}", a1[0], a1[1], a1[2]);
        let s = &buf[4..7];
        let a2 = &*(s.as_ptr() as *const [u64; 3]);
        println!("{} {} {}", a2[0], a2[1], a2[2]);
    }
}
"#;

#[test]
fn run_mode_flat_buffer_array_carve_stays_windowed() {
    // The other side of the denomination: an array pointer
    // cast over a flat scalar buffer denotes the declared
    // array as a window of the buffer's sequence starting at
    // the pointer's element. That reading must survive the
    // nested-array judgement.
    let dir = temp_dir("flat-buffer-array-carve");
    let src = fixture(&dir, "flat_buffer_array_carve.rs", FLAT_BUFFER_ARRAY_CARVE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "array carve over a flat buffer must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "0 1 2\n4 5 6\n4 5 6\n"
    );
}
