// Unsize and fat-pointer metadata tests.

// Slice unsizing of pointer-carrying values: a reference-counted
// handle is itself the pointer (its pointee lives in the
// allocation store), a custom DST propagates the metadata
// through its trailing field, and a byte-slice constant in
// pointer position is the fat pointer, not the pointee data.

const UNSIZE_RC_SLICE_SRC: &str = r#"
use std::rc::Rc;
fn main() {
    let a: Rc<[i32; 3]> = Rc::new([1, 2, 3]);
    let b: Rc<[i32]> = a;
    println!("len={} {} {} {}", b.len(), b[0], b[1], b[2]);
    let c = b.clone();
    println!("strong={} clen={}", Rc::strong_count(&b), c.len());
}
"#;

#[test]
fn run_mode_unsize_rc_slice_handle() {
    run_ok("unsize-rc-slice", UNSIZE_RC_SLICE_SRC, "len=3 1 2 3\nstrong=2 clen=3\n");
}

const UNSIZE_BYTE_STATICS_SRC: &str = r#"
static BYTES: &[u8] = &[1, 2, 3];
const CB: &[u8] = b"ab";
fn take(s: &[u8]) { println!("take {} {}", s.len(), s[2]); }
fn main() {
    println!("st {} {}", BYTES.len(), BYTES[1]);
    take(BYTES);
    println!("cb {} {}", CB.len(), CB[0]);
}
"#;

#[test]
fn run_mode_unsize_byte_slice_statics_are_fat_pointers() {
    run_ok(
        "unsize-byte-statics",
        UNSIZE_BYTE_STATICS_SRC,
        "st 3 2\ntake 3 3\ncb 2 97\n",
    );
}

const UNSIZE_RAW_ONE_ELEM_SRC: &str = r#"
fn main() {
    let b = Box::new([5u8]);
    let raw = Box::into_raw(b);
    let raw2 = raw as *mut [u8];
    let s: &[u8] = unsafe { &*raw2 };
    println!("len={} v={}", s.len(), s[0]);
    let b2 = unsafe { Box::from_raw(raw2) };
    drop(b2);
    println!("done");
}
"#;

#[test]
fn run_mode_unsize_raw_pointer_slice_view() {
    run_ok("unsize-raw-view", UNSIZE_RAW_ONE_ELEM_SRC, "len=1 v=5\ndone\n");
}

// ----- slice-tag predicate unification (Issue #403) -----

const SIZE_OF_VAL_SLICE_OF_ARRAYS_SRC: &str = r#"
fn main() {
    let a: [[i32; 2]; 3] = [[1, 2], [3, 4], [5, 6]];
    let s: &[[i32; 2]] = &a;
    println!("full {}", std::mem::size_of_val(s));
    println!("part {}", std::mem::size_of_val(&a[..2]));
    let v: Vec<[u16; 4]> = vec![[1, 2, 3, 4], [5, 6, 7, 8]];
    println!("vec {}", std::mem::size_of_val(&v[..]));
    let deep: &[[[i32; 1]; 2]] = &[[[1], [2]], [[3], [4]]];
    println!("deep {}", std::mem::size_of_val(deep));
}
"#;

#[test]
fn run_mode_size_of_val_slice_of_arrays() {
    // The element type of a slice may itself be an array: the
    // semicolon inside `[i32; 2]` sits one bracket level deeper
    // and does not make `[[i32; 2]]` an array tag. `size_of_val`
    // judged slice-ness with a raw `!contains(';')` and refused
    // these as not runtime-supported (expected values verified
    // against a native 1.100.0 build of the same source).
    run_ok(
        "size-of-val-slice-of-arrays",
        SIZE_OF_VAL_SLICE_OF_ARRAYS_SRC,
        "full 24\npart 16\nvec 16\ndeep 16\n",
    );
}

const ALIGN_OF_VAL_SLICE_OF_ARRAYS_SRC: &str = r#"
fn main() {
    let a: [[i32; 2]; 3] = [[1, 2], [3, 4], [5, 6]];
    let s: &[[i32; 2]] = &a;
    println!("full {}", std::mem::align_of_val(s));
    let v: Vec<[u16; 4]> = vec![[1, 2, 3, 4]];
    println!("vec {}", std::mem::align_of_val(&v[..]));
    let w: &[[u8; 3]] = &[[1, 2, 3], [4, 5, 6]];
    println!("bytes {}", std::mem::align_of_val(w));
}
"#;

#[test]
fn run_mode_align_of_val_slice_of_arrays() {
    // The `align_of_val` twin of the size case above: same raw
    // predicate, same misjudgment of a slice of arrays.
    run_ok(
        "align-of-val-slice-of-arrays",
        ALIGN_OF_VAL_SLICE_OF_ARRAYS_SRC,
        "full 4\nvec 2\nbytes 1\n",
    );
}
