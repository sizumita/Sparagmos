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
