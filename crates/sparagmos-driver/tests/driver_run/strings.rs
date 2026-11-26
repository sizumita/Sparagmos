use super::{fixture, run_driver, run_driver_extra, run_driver_with_aux_specs, run_ok, temp_dir};


const BYTE_SEQ_SRC: &str = r#"
fn main() {
    let slice: &[u8] = &[104, 105, 0, 255];
    println!("{:?}", slice);
    let arr: [u8; 3] = [65, 66, 67];
    println!("{:?}", arr);
    println!("{:?}", b"hello");
    println!("{:#?}", &[10u8, 20, 30] as &[u8]);
    let signed: [i8; 4] = [1, -1, 127, -128];
    println!("{:?}", signed);
    println!("{:?}", &signed as &[i8]);
    let chars: [std::ffi::c_char; 3] = [97, -1, 0];
    println!("{:?}", chars);
    let v: Vec<u8> = vec![9, 8, 7];
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_debug_format_byte_sequences() {
    // Byte values reach the Debug formatter as `Value::Bytes`
    // blocks (a byte-slice value, a byte-string constant); the
    // element type in the tag decides the signedness each byte
    // prints with. The expected text is native output.
    let dir = temp_dir("bytesfmt");
    let src = fixture(&dir, "bytes.rs", BYTE_SEQ_SRC);
    let out = run_driver_with_aux_specs(&src, Some("2021"), &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "byte sequences must format; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[104, 105, 0, 255]\n[65, 66, 67]\n[104, 101, 108, 108, 111]\n[\n    10,\n    20,\n    30,\n]\n[1, -1, 127, -128]\n[1, -1, 127, -128]\n[97, -1, 0]\n[9, 8, 7]\n",
        "byte sequence Debug output matches native"
    );
}

const C_STRING_SRC: &str = r#"
use std::ffi::{CStr, CString};
fn main() {
    println!("{:?}", c"hello");
    println!("{:?}", c"");
    println!("{:?}", c"a\tb\nc\"d\\e");
    println!("{:?}", c"\xff\x01\x7f");
    let bytes = b"from bytes\0";
    println!("{:?}", CStr::from_bytes_until_nul(bytes).unwrap());
    let owned = CString::new("owned text").unwrap();
    println!("{:?}", owned);
    println!("{:?}", owned.as_c_str());
    let opt: Option<CString> = Some(CString::new("in option").unwrap());
    println!("{:?}", opt);
    let vec = vec![CString::new("elem0").unwrap(), CString::new(vec![b'x', 0x80]).unwrap()];
    println!("{:?}", vec);
    let tup = (CString::new("in tuple").unwrap(), 42);
    println!("{:?}", tup);
    #[derive(Debug)]
    struct Holder<'a> { name: &'a CStr, n: i32 }
    let h = Holder { name: c"field str", n: 7 };
    println!("{:?}", h);
}
"#;

#[test]
fn run_mode_debug_format_c_strings() {
    // `CStr` / `CString` Debug is the quoted, ASCII-escaped
    // string, whatever shape the value model stored the bytes
    // in (constant block, buffer snapshot, owned aggregate).
    // The expected text is native output.
    let dir = temp_dir("cstrfmt");
    let src = fixture(&dir, "cstr.rs", C_STRING_SRC);
    let out = run_driver_with_aux_specs(&src, Some("2021"), &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "C strings must format; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        r##""hello"
""
"a\tb\nc\"d\\e"
"\xff\x01\x7f"
"from bytes"
"owned text"
"owned text"
Some("in option")
["elem0", "x\x80"]
("in tuple", 42)
Holder { name: "field str", n: 7 }
"##,
        "C string Debug output matches native"
    );
}

const USER_C_STRING_NAMES_SRC: &str = r#"
#[derive(Debug)]
struct CString(String);

#[derive(Debug)]
struct CStr([u8; 3]);

mod ffi {
    #[derive(Debug)]
    pub struct CString(pub Vec<u8>);
}

fn main() {
    println!("{:?}", CString("hello".to_string()));
    println!("{:?}", CStr([104, 105, 0]));
    println!("{:?}", ffi::CString(vec![97, 98]));
    let opt: Option<CString> = Some(CString("nested".to_string()));
    println!("{:?}", opt);
}
"#;

#[test]
fn run_mode_debug_format_user_types_named_like_c_strings() {
    // A user-defined type that merely shares the `CStr` /
    // `CString` base name keeps its own (derived) Debug form:
    // the C string formatting path is restricted to the
    // standard library's qualified type paths. The expected
    // text is native output.
    let dir = temp_dir("usercstr");
    let src = fixture(&dir, "usercstr.rs", USER_C_STRING_NAMES_SRC);
    let out = run_driver_with_aux_specs(&src, Some("2021"), &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "user types named like C strings must format; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "CString(\"hello\")\nCStr([104, 105, 0])\nCString([97, 98])\nSome(CString(\"nested\"))\n",
        "user-defined CStr/CString names keep their derived Debug output"
    );
}

// ----- the String / str surface on the abstract string value ------------
//
// `String` and `str` are the string data itself in the interpreter's
// value model, so the std bodies (which project the real struct's
// buffer/length/capacity fields) cannot run on them; the methods
// are runtime support instead. These fixtures pin the observable
// behaviour against native semantics.

const STRING_LEN_SRC: &str = r#"
fn main() {
    let s = String::from("héllo");
    println!("{}", s.len());
    println!("{}", s.is_empty());
    let e = String::new();
    println!("{}", e.is_empty());
    let r: &str = "héllo";
    println!("{}", r.len());
    println!("{}", r.is_empty());
}
"#;

#[test]
fn run_mode_string_len_is_byte_length() {
    let dir = temp_dir("string-len");
    let src = fixture(&dir, "string_len.rs", STRING_LEN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // `len` is a byte count on both String and str: "héllo" is
    // 6 bytes (é is two), not 5 chars.
    assert_eq!(String::from_utf8_lossy(&out.stdout), "6\nfalse\ntrue\n6\nfalse\n");
}

const STRING_INTO_BYTES_SRC: &str = r#"
fn main() {
    let s = String::from("abc");
    let mut b = s.into_bytes();
    println!("{}", b.len());
    println!("{}", b[1]);
    b.push(100);
    println!("{}", b.len());
    println!("{}", b[3]);
}
"#;

#[test]
fn run_mode_string_into_bytes_yields_a_real_vec() {
    let dir = temp_dir("string-into-bytes");
    let src = fixture(&dir, "string_into_bytes.rs", STRING_INTO_BYTES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The result is a genuine Vec<u8> value: it has a length, its
    // elements index, and it grows through Vec::push.
    assert_eq!(String::from_utf8_lossy(&out.stdout), "3\n98\n4\n100\n");
}

const STRING_PUSH_SRC: &str = r#"
fn main() {
    let mut s = String::new();
    s.push_str("ab");
    s.push('é');
    s.push_str("cd");
    println!("{}", s);
    println!("{}", s.len());
}
"#;

#[test]
fn run_mode_string_push_str_and_push_mutate_in_place() {
    let dir = temp_dir("string-push");
    let src = fixture(&dir, "string_push.rs", STRING_PUSH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "abécd\n6\n");
}

const STRING_VIEWS_SRC: &str = r#"
fn main() {
    let s = String::from("abc");
    let b = s.as_bytes();
    println!("{}", b.len());
    println!("{}", b[0]);
    let r: &str = s.as_str();
    println!("{}", r);
    let t: &str = "abc";
    println!("{}", t.as_bytes()[2]);
}
"#;

#[test]
fn run_mode_string_as_bytes_and_as_str() {
    let dir = temp_dir("string-views");
    let src = fixture(&dir, "string_views.rs", STRING_VIEWS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "3\n97\nabc\n99\n");
}

const STR_TRIM_CHARS_SRC: &str = r#"
fn main() {
    let s = String::from("  hi  ");
    println!("[{}]", s.trim());
    println!("[{}]", s.trim_start());
    println!("[{}]", s.trim_end());
    let mut n = 0;
    for _c in "aébc".chars() { n += 1; }
    println!("{}", n);
    let mut m = 0;
    for _b in "aé".bytes() { m += 1; }
    println!("{}", m);
    let mut it = "éx".chars();
    println!("{}", it.next().unwrap());
}
"#;

#[test]
fn run_mode_str_trim_and_char_byte_iterators() {
    let dir = temp_dir("str-trim-chars");
    let src = fixture(&dir, "str_trim_chars.rs", STR_TRIM_CHARS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // chars() counts/iterates chars (aébc = 4), bytes() iterates
    // the UTF-8 bytes (aé = 3).
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[hi]\n[hi  ]\n[  hi]\n4\n3\né\n"
    );
}

const STRING_CAPACITY_SRC: &str = r#"
fn main() {
    let s = String::from("abc");
    println!("{}", s.capacity());
}
"#;

#[test]
fn run_mode_string_capacity_is_an_explicit_unsupported() {
    // Allocation capacity is not tracked by the abstract string
    // model; the call must fail as unsupported naming the method,
    // not as a projection failure inside the collected std body.
    let dir = temp_dir("string-capacity");
    let src = fixture(&dir, "string_capacity.rs", STRING_CAPACITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("std::string::String::capacity"),
        "the error names the method: {stderr}"
    );
}

const CONST_CLOSURE_FN_PTR_SRC: &str = r#"
const F: fn() -> i32 = || 42;
fn main() { println!("{}", F()); }
"#;

const STATIC_CLOSURE_FN_PTR_SRC: &str = r#"
static F: fn() -> i32 = || 42;
fn main() { println!("{}", F()); }
"#;

const DIRECT_CLOSURE_FN_PTR_SRC: &str = r#"
fn main() {
    let f: fn() -> i32 = || 42;
    println!("{}", f());
}
"#;

const CONST_CLOSURE_FN_PTR_ARGS_SRC: &str = r#"
const F: fn(i32) -> i32 = |x| x + 1;
fn main() { println!("{}", F(41)); }
"#;

const CONST_FN_ITEM_SRC: &str = r#"
fn foo() -> i32 { 42 }
const F: fn() -> i32 = foo;
fn main() { println!("{}", F()); }
"#;

fn run_expect_42(tag: &str, name: &str, src: &str) {
    let dir = temp_dir(tag);
    let path = fixture(&dir, name, src);
    let out = run_driver(&path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "callable constant must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "42\n",
        "output must match the native program"
    );
}

#[test]
fn run_mode_const_coerced_closure_fn_ptr() {
    run_expect_42("const-closure", "const_closure.rs", CONST_CLOSURE_FN_PTR_SRC);
}

#[test]
fn run_mode_static_coerced_closure_fn_ptr() {
    run_expect_42("static-closure", "static_closure.rs", STATIC_CLOSURE_FN_PTR_SRC);
}

#[test]
fn run_mode_direct_closure_fn_ptr_cast() {
    run_expect_42("direct-closure", "direct_closure.rs", DIRECT_CLOSURE_FN_PTR_SRC);
}

#[test]
fn run_mode_const_closure_fn_ptr_with_args() {
    run_expect_42("const-closure-args", "const_closure_args.rs", CONST_CLOSURE_FN_PTR_ARGS_SRC);
}

#[test]
fn run_mode_const_fn_item() {
    run_expect_42("const-fn-item", "const_fn_item.rs", CONST_FN_ITEM_SRC);
}

const BYTES_MATCH_SRC: &str = r#"
fn main() {
    match *b"abc" {
        [b'a', b'b', b'c'] => println!("bytes ok"),
        _ => panic!("no"),
    }
    match *b"abd" {
        [b'a', b'b', b'c'] => panic!("wrong arm"),
        _ => println!("bytes ok 2"),
    }
}
"#;

const BOXED_STR_FROM_COW_SRC: &str = r#"
use std::borrow::Cow;

fn main() {
    // Owned: the box takes the String's text over.
    let b: Box<str> = Box::from(Cow::Owned(String::from("hello owned")));
    println!("owned: [{b}] {} {:?}", b.len(), b);

    // Owned from a variable: the String moves into the Cow
    // and then into the box.
    let s = String::from("var owned");
    let b2: Box<str> = Box::from(Cow::Owned(s));
    println!("var: [{b2}]");

    // `Into` resolves to the same conversion.
    let cow: Cow<str> = Cow::Owned(String::from("into owned"));
    let b3: Box<str> = cow.into();
    println!("into: [{b3}]");

    // Borrowed: the box clones the text; later writes to
    // the source do not show through.
    let mut src = String::from("hello borrowed");
    let b4: Box<str> = Box::from(Cow::Borrowed(src.as_str()));
    src.push_str(" MUTATED");
    println!("borrowed: [{b4}] src: [{src}]");

    // Mutation through the box works on the boxed text.
    let mut b5: Box<str> = Box::from(Cow::Owned(String::from("abc")));
    b5.make_ascii_uppercase();
    println!("upper: [{b5}]");

    // Empty text, both variants.
    let b6: Box<str> = Box::from(Cow::Owned(String::new()));
    let b7: Box<str> = Box::from(Cow::Borrowed(""));
    println!("empty: [{}] [{}] {} {}", b6, b7, b6.len(), b7.len());

    // Slicing and comparison read the boxed text.
    let b8: Box<str> = Box::from(Cow::Owned(String::from("slice me")));
    println!("slice: [{}] eq: {}", &b8[0..5], &*b8 == "slice me");

    // A Cow produced by a lossy conversion converts too.
    let b9: Box<str> = Box::from(String::from_utf8_lossy(&[b'h', b'i', 0xFF]));
    println!("lossy: [{b9}]");

    println!("boxed str from cow ok");
}
"#;

#[test]
fn run_mode_boxed_str_from_cow() {
    let dir = temp_dir("boxed-str-from-cow");
    let src = fixture(&dir, "boxed_str_from_cow.rs", BOXED_STR_FROM_COW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<str> from a Cow<str> must hold the string data, not the Cow aggregate; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "owned: [hello owned] 11 \"hello owned\"\nvar: [var owned]\ninto: [into owned]\nborrowed: [hello borrowed] src: [hello borrowed MUTATED]\nupper: [ABC]\nempty: [] [] 0 0\nslice: [slice] eq: true\nlossy: [hi\u{FFFD}]\nboxed str from cow ok\n"
    );
}

const DEBUG_FORMAT_MUT_VEC_SLICE_SRC: &str = r#"
#[derive(Debug)]
struct Named { x: i32, name: String }
fn main() {
    let mut v = vec![Named { x: 1, name: "p".into() }, Named { x: 2, name: "q".into() }];
    println!("{:?}", &mut v[..]);
}
"#;

#[test]
fn run_mode_debug_format_mut_vec_slice() {
    let dir = temp_dir("debug-format-mut-vec-slice");
    let src = fixture(
        &dir,
        "debug_format_mut_vec_slice.rs",
        DEBUG_FORMAT_MUT_VEC_SLICE_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a sub-slice of a Vec formats as the slice it denotes; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[Named { x: 1, name: \"p\" }, Named { x: 2, name: \"q\" }]\n"
    );
}

const REFCELL_GUARD_DISPLAY_SRC: &str = r#"
use std::cell::RefCell;

fn main() {
    let c = RefCell::new(42i32);
    println!("display-ref: {}", c.borrow());
    {
        let mut m = c.borrow_mut();
        *m += 1;
        println!("display-refmut: {}", m);
    }
    let s = RefCell::new(String::from("hello"));
    println!("str-guard: {}", s.borrow());
    let arr = [RefCell::new(7i32), RefCell::new(8i32)];
    println!("from-array: {}", arr[0].borrow());
}
"#;

#[test]
fn run_mode_refcell_guard_display_formats_pointee() {
    let dir = temp_dir("refcell-guard-display");
    let src = fixture(&dir, "refcell_guard_display.rs", REFCELL_GUARD_DISPLAY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Display for a RefCell guard formats the guarded value; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "display-ref: 42\ndisplay-refmut: 43\nstr-guard: hello\nfrom-array: 7\n"
    );
}

const REFCELL_GUARD_DEBUG_SRC: &str = r#"
use std::cell::RefCell;

fn main() {
    let c = RefCell::new(42i32);
    println!("debug-ref: {:?}", c.borrow());
    let s = RefCell::new(String::from("hello"));
    println!("str-guard-debug: {:?}", s.borrow());
    let v = RefCell::new(vec![1, 2, 3]);
    println!("vec-guard-debug: {:?}", v.borrow());
}
"#;

#[test]
fn run_mode_refcell_guard_debug_formats_pointee() {
    let dir = temp_dir("refcell-guard-debug");
    let src = fixture(&dir, "refcell_guard_debug.rs", REFCELL_GUARD_DEBUG_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Debug for a RefCell guard formats the guarded value, not the guard aggregate; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "debug-ref: 42\nstr-guard-debug: \"hello\"\nvec-guard-debug: [1, 2, 3]\n"
    );
}

const REFCELL_GUARD_MAPPED_SRC: &str = r#"
use std::cell::{Ref, RefCell};

fn main() {
    let c = RefCell::new(42i32);
    let mapped: Ref<'_, i32> = Ref::map(c.borrow(), |v| v);
    println!("mapped: {}", mapped);
    let p = RefCell::new((11i32, 22i32));
    let f = Ref::map(p.borrow(), |v| &v.1);
    println!("field: {}", f);
    let t = c.try_borrow().unwrap();
    println!("try: {}", t);
}
"#;

#[test]
fn run_mode_refcell_guard_mapped_formats_pointee() {
    let dir = temp_dir("refcell-guard-mapped");
    let src = fixture(&dir, "refcell_guard_mapped.rs", REFCELL_GUARD_MAPPED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a guard built by Ref::map's converted body formats its pointee like a builtin-built one; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "mapped: 42\nfield: 22\ntry: 42\n"
    );
}

const REFCELL_GUARD_CUSTOM_SRC: &str = r#"
use std::cell::RefCell;
use std::fmt;

struct Point { x: i32, y: i32 }

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P({},{})", self.x, self.y)
    }
}

impl fmt::Debug for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pt[{},{}]", self.x, self.y)
    }
}

fn main() {
    let c = RefCell::new(42i32);
    println!("width: [{:>6}]", c.borrow());
    println!("width-debug: [{:^8?}]", c.borrow());
    let p = RefCell::new(Point { x: 3, y: 4 });
    println!("custom: {}", p.borrow());
    println!("custom-debug: {:?}", p.borrow());
}
"#;

#[test]
fn run_mode_refcell_guard_format_options_and_custom_pointee() {
    let dir = temp_dir("refcell-guard-custom");
    let src = fixture(&dir, "refcell_guard_custom.rs", REFCELL_GUARD_CUSTOM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "guard formatting forwards the placeholder options and runs the pointee's own impl body; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "width: [    42]\nwidth-debug: [   42   ]\ncustom: P(3,4)\ncustom-debug: Pt[3,4]\n"
    );
}

const RANGE_SLICE_VIEW_SPARSE_SRC: &str = r#"
use std::cell::Cell;

fn main() {
    let mut v: Vec<Cell<i32>> = Vec::new();
    for i in 0..6 {
        v.push(Cell::new(i * 10));
    }
    let view = &v[1..4];
    view[0].set(111);
    view[2].set(333);
    println!("src: {} {} {}", v[1].get(), v[2].get(), v[3].get());
    v[2].set(222);
    println!("view: {} {} {}", view[0].get(), view[1].get(), view[2].get());
}
"#;

#[test]
fn run_mode_range_slice_view_sparse_cell_writes_reach_source() {
    let dir = temp_dir("range-slice-view-sparse");
    let src = fixture(&dir, "range_slice_view_sparse.rs", RANGE_SLICE_VIEW_SPARSE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a shared range slice over sparse-backed storage must alias it, not snapshot it; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "src: 111 20 333\nview: 111 222 333\n"
    );
}

const RANGE_SLICE_VIEW_SPARSE_KINDS_SRC: &str = r#"
use std::cell::Cell;

fn main() {
    let mut v: Vec<Cell<i32>> = Vec::with_capacity(100);
    for i in 0..6 { v.push(Cell::new(i * 10)); }
    let tail = &v[3..];
    tail[0].set(300);
    println!("from: {}", v[3].get());
    let head = &v[..2];
    head[1].set(100);
    println!("to: {}", v[1].get());
    let full = &v[..];
    full[4].set(400);
    println!("full: {}", v[4].get());
    let incl = &v[1..=2];
    incl[1].set(222);
    println!("incl: {}", v[2].get());
    let nested = &v[1..5][1..3];
    nested[0].set(999);
    println!("nested: {}", v[2].get());
    let mut w: Vec<i32> = Vec::new();
    for i in 0..6 { w.push(i * 10); }
    println!("win: {:?}", &w[1..4]);
    println!("winfmt: {:?}", &w[2..]);
    let view = &v[2..5];
    let e = &view[1];
    e.set(333);
    println!("elem: {}", v[3].get());
}
"#;

#[test]
fn run_mode_range_slice_view_sparse_range_kinds_alias_source() {
    let dir = temp_dir("range-slice-view-sparse-kinds");
    let src = fixture(
        &dir,
        "range_slice_view_sparse_kinds.rs",
        RANGE_SLICE_VIEW_SPARSE_KINDS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "every range kind over sparse-backed storage must alias it; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "from: 300\nto: 100\nfull: 400\nincl: 222\nnested: 999\nwin: [10, 20, 30]\nwinfmt: [20, 30, 40, 50]\nelem: 333\n"
    );
}

const RANGE_SLICE_MUT_SPARSE_SRC: &str = r#"
use std::cell::Cell;

fn main() {
    let mut v: Vec<Cell<i32>> = Vec::new();
    for i in 0..6 {
        v.push(Cell::new(i * 10));
    }
    let view = &mut v[1..4];
    view[0].set(111);
    println!("src: {}", v[1].get());
}
"#;

#[test]
fn run_mode_range_slice_mut_sparse_cell_writes_reach_source() {
    let dir = temp_dir("range-slice-mut-sparse");
    let src = fixture(&dir, "range_slice_mut_sparse.rs", RANGE_SLICE_MUT_SPARSE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a mutable range slice over sparse-backed storage must navigate its elements; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "src: 111\n");
}


#[test]
fn run_mode_byte_string_pattern_match() {
    let dir = temp_dir("bytes-match");
    let src = fixture(&dir, "bytes_match.rs", BYTES_MATCH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "byte-string pattern matching must work; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "bytes ok\nbytes ok 2\n"
    );
}

const TRANSMUTE_BOOL_CHAR_SRC: &str = r#"
fn inpb() -> bool { let mut s = 0; let mut i = 0; while i < 41 { s += 1; i += 1; } s == 42 }
fn inpc() -> char { let mut s = 0u32; let mut i = 0; while i < 65 { s += 1; i += 1; } char::from_u32(s).unwrap() }
fn main() {
    let b = inpb();
    let n: u8 = unsafe { std::mem::transmute(b) };
    let b2: bool = unsafe { std::mem::transmute(1u8) };
    let c = inpc();
    let cn: u32 = unsafe { std::mem::transmute(c) };
    let c2: char = unsafe { std::mem::transmute(66u32) };
    println!("{} {} {} {}", n, b2, cn, c2);
}
"#;

#[test]
fn run_mode_transmute_bool_and_char() {
    run_ok("transmute-bool-char", TRANSMUTE_BOOL_CHAR_SRC, "0 true 65 B\n");
}

const TRANSMUTE_BAD_CHAR_SRC: &str = r#"
fn inp() -> u32 { let mut s = 0x10000u32; s *= 17; s }
fn main() {
    let c: char = unsafe { std::mem::transmute(inp()) };
    println!("{}", c);
}
"#;

#[test]
fn run_mode_transmute_invalid_char_is_explicit_unsupported() {
    // Transmuting an invalid bit pattern into `char` is UB in
    // real Rust; the model reports it as an explicit unsupported
    // error instead of inventing a character.
    let dir = temp_dir("transmute-bad-char");
    let src = fixture(&dir, "case.rs", TRANSMUTE_BAD_CHAR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "invalid char transmute must be a named unsupported; stderr={stderr}"
    );
    assert!(
        stderr.contains("transmute to char"),
        "stderr must name the char transmute; stderr={stderr}"
    );
}

const WEAK_COW_DROP_SRC: &str = r#"
enum Void {}
fn main() {
    let _w1 = std::rc::Weak::<Void>::new();
    let _w2 = std::sync::Weak::<Void>::new();
    let c: std::borrow::Cow<str> = std::borrow::Cow::Borrowed("x");
    let _o: std::borrow::Cow<str> = std::borrow::Cow::Owned(String::from("y"));
    drop(c);
    println!("weak-cow ok");
}
"#;

#[test]
fn run_mode_weak_and_cow_str_drop() {
    let dir = temp_dir("weak-cow-drop");
    let src = fixture(&dir, "weak_cow_drop.rs", WEAK_COW_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dropping Weak pointers and Cow<str> values must be accounted; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "weak-cow ok\n");
}

const WRITE_FMT_STRING_SRC: &str = r#"use std::fmt::Write as _;
fn main() {
    let mut s = String::new();
    // Literal-only arguments take the `Arguments::as_str`
    // fast path inside `fmt::write`.
    write!(s, "hello").unwrap();
    assert_eq!(s, "hello");
    // A template with placeholders goes through the general
    // path: the modelled `Arguments` value's fields are
    // projected by the `Write::write_fmt` default body.
    write!(s, " x={} y={:.3}", 42, 1.23456).unwrap();
    assert_eq!(s, "hello x=42 y=1.235");
    writeln!(s, " [{:>4}]", 7).unwrap();
    assert_eq!(s, "hello x=42 y=1.235 [   7]\n");
    // `format_args!` converts to a string through its
    // `Display` / `ToString` impl.
    let t = format_args!("n={}", 3).to_string();
    assert_eq!(t, "n=3");
    println!("write fmt string ok");
}
"#;

#[test]
fn run_mode_write_fmt_to_string() {
    // `write!` on a `String` runs the `fmt::Write::write_fmt`
    // default body over a modelled `fmt::Arguments` value:
    // its fields are projected from the model, and the text is
    // delivered through `core::fmt::write` to the string's own
    // `write_str`.
    let dir = temp_dir("write-fmt-string");
    let src = fixture(&dir, "write_fmt_string.rs", WRITE_FMT_STRING_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "write fmt string ok\n");
}
const DEBUG_ADT_SRC: &str = r#"
#[derive(Debug)]
struct Named { x: i32, name: String }
#[derive(Debug)]
struct Empty {}
#[derive(Debug)]
struct Tup(i32, bool);
#[derive(Debug)]
struct Tup1(i32);
#[derive(Debug)]
struct Unit;
#[derive(Debug)]
enum E { A, B(i32), C(i32, u8), D { f: i32 }, G {} }
fn main() {
    println!("{:?}", Named { x: 3, name: "hi".to_string() });
    println!("{:?}", Empty {});
    println!("{:?}", Tup(1, true));
    println!("{:?}", Tup1(9));
    println!("{:?}", Unit);
    println!("{:?}", E::A);
    println!("{:?}", E::B(5));
    println!("{:?}", E::C(5, 6));
    println!("{:?}", E::D { f: 7 });
    println!("{:?}", E::G {});
}
"#;

#[test]
fn run_mode_debug_format_derived_adts() {
    let dir = temp_dir("debug-adt");
    let src = fixture(&dir, "debug_adt.rs", DEBUG_ADT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "Named { x: 3, name: \"hi\" }\n\
         Empty\n\
         Tup(1, true)\n\
         Tup1(9)\n\
         Unit\n\
         A\n\
         B(5)\n\
         C(5, 6)\n\
         D { f: 7 }\n\
         G\n",
        "stdout={stdout}"
    );
}

const DEBUG_ADT_PRETTY_SRC: &str = r#"
#[derive(Debug)]
struct Named { x: i32, name: String }
#[derive(Debug)]
struct Tup(i32, bool);
#[derive(Debug)]
enum E { D { f: i32 } }
#[derive(Debug)]
struct Outer { inner: Named, list: Vec<Tup> }
fn main() {
    println!("{:#?}", Named { x: 3, name: "hi".to_string() });
    println!("{:#?}", Tup(1, true));
    println!("{:#?}", E::D { f: 7 });
    println!("{:#?}", Outer { inner: Named { x: 1, name: "z".into() }, list: vec![Tup(3, true)] });
    println!("{:#?}", Vec::<i32>::new());
}
"#;

#[test]
fn run_mode_debug_format_pretty_adts() {
    let dir = temp_dir("debug-adt-pretty");
    let src = fixture(&dir, "debug_adt_pretty.rs", DEBUG_ADT_PRETTY_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "Named {\n    x: 3,\n    name: \"hi\",\n}\n\
         Tup(\n    1,\n    true,\n)\n\
         D {\n    f: 7,\n}\n\
         Outer {\n    inner: Named {\n        x: 1,\n        name: \"z\",\n    },\n    list: [\n        Tup(\n            3,\n            true,\n        ),\n    ],\n}\n\
         []\n",
        "stdout={stdout}"
    );
}

const DEBUG_ADT_CONTAINERS_SRC: &str = r#"
#[derive(Debug)]
struct Named { x: i32, name: String }
#[derive(Debug)]
struct Tup(i32, bool);
#[derive(Debug)]
struct Outer { inner: Named, list: Vec<Tup>, opt: Option<i32> }
#[derive(Debug)]
struct Pair<T> { a: T, b: T }
fn main() {
    println!("{:?}", Some(Named { x: 1, name: "a".into() }));
    println!("{:?}", Ok::<i32, String>(3));
    println!("{:?}", Err::<i32, String>("bad".into()));
    println!("{:?}", vec![Tup(1, false), Tup(2, true)]);
    println!("{:?}", [Named { x: 2, name: "b".into() }]);
    println!("{:?}", (Tup(4, true), 8));
    println!("{:?}", Outer { inner: Named { x: 1, name: "z".into() }, list: vec![Tup(3, true)], opt: None });
    let v = vec![Named { x: 1, name: "p".into() }, Named { x: 2, name: "q".into() }];
    println!("{:?}", &v[..]);
    println!("{:?}", Pair { a: 1, b: 2 });
    println!("{:?}", Pair { a: "s", b: "t" });
    println!("{:?}", format!("{:?}", Named { x: 8, name: "f".into() }));
}
"#;

#[test]
fn run_mode_debug_format_adt_in_containers() {
    let dir = temp_dir("debug-adt-containers");
    let src = fixture(&dir, "debug_adt_containers.rs", DEBUG_ADT_CONTAINERS_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "Some(Named { x: 1, name: \"a\" })\n\
         Ok(3)\n\
         Err(\"bad\")\n\
         [Tup(1, false), Tup(2, true)]\n\
         [Named { x: 2, name: \"b\" }]\n\
         (Tup(4, true), 8)\n\
         Outer { inner: Named { x: 1, name: \"z\" }, list: [Tup(3, true)], opt: None }\n\
         [Named { x: 1, name: \"p\" }, Named { x: 2, name: \"q\" }]\n\
         Pair { a: 1, b: 2 }\n\
         Pair { a: \"s\", b: \"t\" }\n\
         \"Named { x: 8, name: \\\"f\\\" }\"\n",
        "stdout={stdout}"
    );
}

const DEBUG_ADT_REFS_SRC: &str = r#"
use std::fmt::Debug;
#[derive(Debug)]
struct Named { x: i32 }
#[derive(Debug)]
struct Tup(i32, bool);
#[derive(Debug)]
enum E { V(i32), W }
fn main() {
    let n = Named { x: 3 };
    println!("{:?}", &n);
    println!("{:?}", Box::new(Tup(7, false)));
    let d: &dyn Debug = &Named { x: 5 };
    println!("{:?}", d);
    let d2: Box<dyn Debug> = Box::new(E::V(3));
    println!("{:?}", d2);
    let d3: &dyn Debug = &E::W;
    println!("{:?}", d3);
}
"#;

#[test]
fn run_mode_debug_format_refs_box_and_dyn() {
    let dir = temp_dir("debug-adt-refs");
    let src = fixture(&dir, "debug_adt_refs.rs", DEBUG_ADT_REFS_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "Named { x: 3 }\n\
         Tup(7, false)\n\
         Named { x: 5 }\n\
         V(3)\n\
         W\n",
        "stdout={stdout}"
    );
}

const DEBUG_RANGES_SRC: &str = r#"
fn main() {
    println!("{:?}", std::ops::Range { start: 1, end: 4 });
    println!("{:?}", 1..=4);
    println!("{:?}", 1..);
    println!("{:?}", ..4);
    println!("{:?}", ..=4);
    println!("{:?}", ..);
    println!("{:?}", 'a'..='z');
    println!("{:#?}", 1..4);
}
"#;

#[test]
fn run_mode_debug_format_ranges() {
    let dir = temp_dir("debug-ranges");
    let src = fixture(&dir, "debug_ranges.rs", DEBUG_RANGES_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "1..4\n1..=4\n1..\n..4\n..=4\n..\n'a'..='z'\n1..4\n",
        "stdout={stdout}"
    );
}

const FORMAT_INNER_INLINED_SRC: &str = r#"
fn foo(_i: i32) -> i32 { 1 }
fn bar(_i: i32) -> i32 { 1 }

fn main() {
    let i = 40 + 2;
    println!("{}", format!("value={i}!"));
    let b = true;
    println!("{}", format!("{b}"));
    let c = 'x';
    println!("{}", format!("{c}"));
    let a = [1, 2, 3];
    println!("{}", format!("{:?}", a));
    let x: fn(i32) -> i32 = foo;
    let y: fn(i32) -> i32 = bar;
    let s1 = format!("{:?}, {:?}", x, y);
    let s2 = format!("{:?}, {:?}", x, y);
    assert_eq!(s1, s2);
    println!("{}", format!("{:>width$}", i, width = 4));
}
"#;

/// At opt-level 2 and above, rustc inlines the `#[inline]`
/// `alloc::fmt::format` into the guest body, and the inlined
/// body calls its nested helper `format_inner` directly. That
/// helper is neither generic nor `#[inline]`, so no MIR body
/// for it is encoded in the distributed alloc rlib; the call
/// is served by the `fmt::format` builtin model, whose
/// observable behaviour (format the `Arguments` into a fresh
/// `String`) is exactly the helper's. Without the model the
/// run stops at an unsupported "Call to
/// std::fmt::format::format_inner (no converted body)"
/// terminator.
///
/// The arguments of an optimized `format_args!` are built by
/// the guest body itself, as raw `core::fmt::rt::Argument`
/// aggregates whose formatter names the impl in one of the
/// forms conversion can state: an instance name, the
/// impl-segment form (`std::array::<impl Debug for [i32; 3]>::fmt`,
/// `std::ptr::<impl Debug for fn(i32) -> i32>::fmt`), or the
/// bare trait-item name (`std::fmt::Display::fmt`, for `bool`
/// and `char`), whose Self type is recovered from the target
/// place's declared type. The pieces below pin each form, a
/// dynamic-width count argument, and deterministic
/// function-pointer `Debug` text.
#[test]
fn run_mode_o3_format_macro_uses_inlined_format_inner() {
    let dir = temp_dir("format-inner-o3");
    let src = fixture(&dir, "case.rs", FORMAT_INNER_INLINED_SRC);
    for opt in ["-Copt-level=2", "-Copt-level=3"] {
        let out = run_driver_extra(&src, &["--rustc-arg", opt]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(0),
            "inlined format_inner must run at {opt}; stderr={stderr}"
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "value=42!\ntrue\nx\n[1, 2, 3]\n  42\n",
            "formatted output must match native at {opt}; stderr={stderr}"
        );
    }
}

const ITER_COLLECT_SRC: &str = r#"
fn main() {
    let v: Vec<i32> = (0..5).collect();
    let cs: Vec<char> = "xyz".chars().collect();
    let s: String = "abcd".chars().collect();
    let direct: Vec<i32> = std::iter::FromIterator::from_iter(0..4);
    println!(
        "v={} {} cs={} {} s={} direct={}",
        v.len(),
        v[0] + v[4],
        cs.len(),
        cs[2],
        s,
        direct.len()
    );
}
"#;

#[test]
fn run_mode_iterator_collect_into_vec_and_string() {
    let dir = temp_dir("iter_collect");
    let src = fixture(&dir, "iter_collect.rs", ITER_COLLECT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "collect runs; stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "v=5 4 cs=3 z s=abcd direct=4\n"
    );
}

const FROM_BOXED_STR_SRC: &str = r#"
use std::rc::Rc;
use std::sync::Arc;

fn main() {
    let boxed = String::from("abc").into_boxed_str();
    let rc: Rc<str> = boxed.into();
    let boxed2 = String::from("xy").into_boxed_str();
    let arc: Arc<str> = boxed2.into();
    let boxed3 = String::from("zz").into_boxed_str();
    let s: String = boxed3.into();
    println!("{rc} {arc} {s} {} {}", rc.len(), Rc::strong_count(&rc));
}
"#;

#[test]
fn run_mode_from_boxed_str_to_counted_and_string() {
    // `From<Box<str>>` for `Rc<str>` / `Arc<str>` / `String`: the
    // source value is the box aggregate `String::into_boxed_str`
    // builds around the string, not the string itself, so the
    // conversion takes the pointee out of the box (releasing its
    // allocation) instead of rejecting the shape. Before the fix
    // all three conversions were "not runtime-supported".
    run_ok(
        "from-boxed-str",
        FROM_BOXED_STR_SRC,
        "abc xy zz 3 1\n",
    );
}

const FROM_BORROWED_STR_SRC: &str = r#"
use std::rc::Rc;
use std::sync::Arc;

fn main() {
    let b = String::from("abc").into_boxed_str();
    let r: Rc<str> = Rc::from(&*b);
    let a: Arc<str> = Arc::from(&*b);
    println!("{r} {a} {}", &*b);
    let s = String::from("xy");
    let r2: Rc<str> = Rc::from(&*s);
    println!("{r2} {s}");
    let r3: Rc<str> = Rc::from(&*r);
    println!("{r3} {}", Rc::strong_count(&r));
    let e = String::new().into_boxed_str();
    let r4: Rc<str> = Rc::from(&*e);
    println!("empty={:?}", &*r4);
}
"#;

#[test]
fn run_mode_from_borrowed_str_to_counted_string() {
    // `From<&str>` for `Rc<str>` / `Arc<str>` where the borrowed
    // string arrives as a reference into string storage (the
    // dereference of a `Box<str>`, a `String` or an `Rc<str>`),
    // not as a string value: the conversion follows the
    // reference to the string data and copies it into the new
    // handle, leaving the borrowed storage alive. Before the
    // fix these failed with "expected string value, found slice
    // reference" (expected output verified against a native
    // 1.100.0 build of the same source).
    run_ok(
        "from-borrowed-str",
        FROM_BORROWED_STR_SRC,
        "abc abc abc\nxy xy\nabc 1\nempty=\"\"\n",
    );
}

const STR_SPLIT_SRC: &str = r#"
fn show(label: &str, v: Vec<&str>) {
    println!("{label}={v:?}");
}

fn main() {
    show("split", "a,,bb,ccc".split(',').collect());
    show("split-trail", "a,b,".split(',').collect());
    show("split-lead", ",a,b".split(',').collect());
    show("split-empty", "".split(',').collect());
    show("split-only", ",".split(',').collect());
    show("rsplit", "a,,bb,ccc".rsplit(',').collect());
    show("term", "a,,bb,ccc".split_terminator(',').collect());
    show("term-trail", "a,b,".split_terminator(',').collect());
    show("term-empty", "".split_terminator(',').collect());
    show("rterm", "a,b,".rsplit_terminator(',').collect());
    show("rterm-notrail", "a,b".rsplit_terminator(',').collect());
    show("incl", "a,,bb,ccc".split_inclusive(',').collect());
    show("incl-trail", "a,b,".split_inclusive(',').collect());
    show("incl-empty", "".split_inclusive(',').collect());
    show("mb", "α,ββ,γ".split(',').collect());
    show("mb-needle", "aαbααc".split('α').collect());
    show("mb-incl", "α,β".split_inclusive(',').collect());
    let mut it = "a,b,c,d".split(',');
    println!("i1={:?}", it.next());
    println!("i2={:?}", it.next_back());
    println!("i3={:?}", it.next());
    println!("i4={:?}", it.next_back());
    println!("i5={:?}", it.next());
    println!("i6={:?}", it.next_back());
    let mut it = "a,b,".split_terminator(',');
    println!("t1={:?}", it.next_back());
    println!("t2={:?}", it.next());
    println!("t3={:?}", it.next_back());
    let mut it = "a,bb,ccc".split_inclusive(',');
    println!("s1={:?}", it.next());
    println!("s2={:?}", it.next_back());
    println!("s3={:?}", it.next());
    println!("s4={:?}", it.next_back());
    println!("s5={:?}", it.next());
    let mut it = "a,b,c".rsplit(',');
    println!("r1={:?}", it.next());
    println!("r2={:?}", it.next_back());
    println!("r3={:?}", it.next());
    println!("r4={:?}", it.next_back());
    println!("cnt={}", "a,,b".split(',').filter(|s| !s.is_empty()).filter_map(|s| Some(s.len())).count());
    println!("sum={}", "1,2,3".split(',').filter_map(|s| s.parse::<i32>().ok()).sum::<i32>());
}
"#;

#[test]
fn run_mode_str_split_family() {
    // The whole `SplitInternal` family over a `char` pattern:
    // both directions, the terminators' trailing-empty rule,
    // inclusive pieces, multi-byte haystacks and needles, and
    // front/back interleaving on one state. Expected output
    // was produced by the native 1.100.0 toolchain.
    let dir = temp_dir("str-split-family");
    let src = fixture(&dir, "str_split_family.rs", STR_SPLIT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the str split family must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        r#"split=["a", "", "bb", "ccc"]
split-trail=["a", "b", ""]
split-lead=["", "a", "b"]
split-empty=[""]
split-only=["", ""]
rsplit=["ccc", "bb", "", "a"]
term=["a", "", "bb", "ccc"]
term-trail=["a", "b"]
term-empty=[]
rterm=["b", "a"]
rterm-notrail=["b", "a"]
incl=["a,", ",", "bb,", "ccc"]
incl-trail=["a,", "b,"]
incl-empty=[]
mb=["α", "ββ", "γ"]
mb-needle=["a", "b", "", "c"]
mb-incl=["α,", "β"]
i1=Some("a")
i2=Some("d")
i3=Some("b")
i4=Some("c")
i5=None
i6=None
t1=Some("b")
t2=Some("a")
t3=None
s1=Some("a,")
s2=Some("ccc")
s3=Some("bb,")
s4=None
s5=None
r1=Some("c")
r2=Some("a")
r3=Some("b")
r4=None
cnt=2
sum=6
"#,
        "every split variant yields core's pieces in core's order"
    );
}

const STR_SPLIT_PRED_SRC: &str = r#"
fn show(label: &str, v: Vec<&str>) {
    println!("{label}={v:?}");
}

fn is_comma(c: char) -> bool {
    c == ','
}

fn main() {
    show("pred-split", "a b  c".split(|c: char| c == ' ').collect());
    show("pred-rsplit", "a b  c".rsplit(|c: char| c == ' ').collect());
    show("pred-term", "a b ".split_terminator(|c: char| c == ' ').collect());
    show("pred-rterm", "a b ".rsplit_terminator(|c: char| c == ' ').collect());
    show("pred-incl", "a b  c".split_inclusive(|c: char| c == ' ').collect());
    show("pred-multi", "\u{3b1},\u{3b2};\u{3b3}".split(|c: char| c == ',' || c == ';').collect());
    let sep = ';';
    show("pred-capture", "a;b;c".split(|c: char| c == sep).collect());
    show("fnptr", "x,y,z".split(is_comma).collect());
    let mut it = "a b c d".split(|c: char| c == ' ');
    println!("p1={:?}", it.next());
    println!("p2={:?}", it.next_back());
    println!("p3={:?}", it.next());
    println!("p4={:?}", it.next_back());
    println!("p5={:?}", it.next_back());
    let mut it = "a,b,".split_terminator(|c: char| c == ',');
    println!("t1={:?}", it.next_back());
    println!("t2={:?}", it.next());
    println!("t3={:?}", it.next_back());
    let mut it = "a b c d".split(|c: char| c == ' ');
    let mut v: Vec<&str> = Vec::new();
    v.push(it.next().unwrap());
    v.push(it.next_back().unwrap());
    println!("v={v:?}");
    println!("eq={} len={}", v[0] == "a", v[0].len());
}
"#;

#[test]
fn run_mode_str_split_predicate_family() {
    // The `Split` family over a `char`-predicate pattern
    // (core's `CharPredicateSearcher`): both directions, the
    // terminators' trailing-empty rule, inclusive pieces,
    // multi-byte haystacks and predicates, a capturing
    // closure and a function-pointer predicate, front/back
    // interleaving on one state, and the yielded pieces as
    // string values (their length and equality). Expected
    // output was produced by the native 1.100.0 toolchain.
    run_ok(
        "str-split-predicate-family",
        STR_SPLIT_PRED_SRC,
        "pred-split=[\"a\", \"b\", \"\", \"c\"]\n\
         pred-rsplit=[\"c\", \"\", \"b\", \"a\"]\n\
         pred-term=[\"a\", \"b\"]\n\
         pred-rterm=[\"b\", \"a\"]\n\
         pred-incl=[\"a \", \"b \", \" \", \"c\"]\n\
         pred-multi=[\"\u{3b1}\", \"\u{3b2}\", \"\u{3b3}\"]\n\
         pred-capture=[\"a\", \"b\", \"c\"]\n\
         fnptr=[\"x\", \"y\", \"z\"]\n\
         p1=Some(\"a\")\n\
         p2=Some(\"d\")\n\
         p3=Some(\"b\")\n\
         p4=Some(\"c\")\n\
         p5=None\n\
         t1=Some(\"b\")\n\
         t2=Some(\"a\")\n\
         t3=None\n\
         v=[\"a\", \"d\"]\n\
         eq=true len=1\n",
    );
}

const WRAPPER_DEFAULT_SRC: &str = r#"
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

fn main() {
    assert_eq!(<f64 as Default>::default(), 0.0);
    assert_eq!(<f32 as Default>::default(), 0.0);
    assert_eq!(<&str as Default>::default(), "");
    let c: Cell<u32> = Default::default();
    assert_eq!(c.get(), 0);
    let r: RefCell<Vec<u32>> = Default::default();
    assert!(r.borrow().is_empty());
    let b: Box<u32> = Default::default();
    assert_eq!(*b, 0);
    let bs: Box<[u32]> = Default::default();
    assert_eq!(bs.len(), 0);
    let rc: Rc<u32> = Default::default();
    assert_eq!(*rc, 0);
    assert_eq!(Rc::strong_count(&rc), 1);
    let arc: Arc<String> = Default::default();
    assert!(arc.is_empty());
    println!("defaults ok");
}
"#;

#[test]
fn run_mode_default_for_float_str_and_wrappers() {
    run_ok("wrapper-default", WRAPPER_DEFAULT_SRC, "defaults ok\n");
}

const DEBUG_WRAPPERS_SRC: &str = r#"
use std::any::Any;
use std::cell::{Cell, RefCell, UnsafeCell};
use std::marker::PhantomData;

fn main() {
    println!("{:?}", Cell::new(0));
    println!("{:?}", UnsafeCell::new(0));
    println!("{:?}", RefCell::new(5));
    println!("{:?}", PhantomData::<&()>);
    let b: Box<dyn Any> = Box::new(1);
    println!("{:?}", b);
}
"#;

#[test]
fn run_mode_debug_wrapper_forms_match_core() {
    // Core's handwritten `Debug` impls for the cell / marker
    // wrappers and for `dyn Any` are fixed functions of the
    // stored value: `Cell` and `RefCell` show the inner value
    // (not the `UnsafeCell` layer), `UnsafeCell` is
    // non-exhaustive, `PhantomData` prints its type parameter,
    // and a `dyn Any` object prints `Any { .. }`.
    run_ok(
        "debug-wrappers",
        DEBUG_WRAPPERS_SRC,
        "Cell { value: 0 }\nUnsafeCell { .. }\nRefCell { value: 5 }\nPhantomData<&()>\nAny { .. }\n",
    );
}

const WRAPPING_FMT_SRC: &str = r#"
use std::num::Wrapping;
#[derive(Debug)]
struct Holder { w: Wrapping<i32> }
fn main() {
    println!("{:?}", Wrapping(5i32));
    println!("{:?}", Wrapping(-5i32));
    println!("{:#?}", Wrapping(5i32));
    println!("{:?}", Wrapping(18446744073709551615u64));
    println!("{:?}", Wrapping(vec![1, 2, 3]));
    println!("{:#?}", Wrapping(vec![1, 2]));
    println!("{:?}", Wrapping("hi"));
    println!("{:?}", Wrapping(Some(3)));
    println!("{:?}", (Wrapping(1), Wrapping(2)));
    println!("{:?}", Holder { w: Wrapping(7) });
    println!("{:#?}", Holder { w: Wrapping(7) });
    println!("{}", Wrapping(5i32));
    println!("{:x}", Wrapping(255u32));
    println!("{:X}", Wrapping(255u32));
    println!("{:o}", Wrapping(8u32));
    println!("{:b}", Wrapping(5u32));
}
"#;

#[test]
fn run_mode_wrapping_fmt_forwards_to_inner() {
    // Core's fmt impls for `std::num::Wrapping<T>` forward to
    // the inner value's own impl: the wrapper's name never
    // appears, for `Debug` (plain and alternate) as for
    // `Display` and the radix traits, at the top level and in
    // a derived `Debug` field.
    run_ok(
        "wrapping-fmt",
        WRAPPING_FMT_SRC,
        "5\n-5\n5\n18446744073709551615\n[1, 2, 3]\n[\n    1,\n    2,\n]\n\"hi\"\nSome(3)\n(1, 2)\nHolder { w: 7 }\nHolder {\n    w: 7,\n}\n5\nff\nFF\n10\n101\n",
    );
}

const SATURATING_FMT_SRC: &str = r#"
use std::num::Saturating;
#[derive(Debug)]
struct Holder { s: Saturating<i32> }
fn main() {
    println!("{:?}", Saturating(5i32));
    println!("{:?}", Saturating(-5i32));
    println!("{:#?}", Saturating(5i32));
    println!("{:?}", Saturating(18446744073709551615u64));
    println!("{:?}", Saturating(vec![1, 2, 3]));
    println!("{:#?}", Saturating(vec![1, 2]));
    println!("{:?}", Saturating("hi"));
    println!("{:?}", Saturating(Some(3)));
    println!("{:?}", (Saturating(1), Saturating(2)));
    println!("{:?}", Holder { s: Saturating(7) });
    println!("{:#?}", Holder { s: Saturating(7) });
    println!("{}", Saturating(5i32));
    println!("{:x}", Saturating(255u32));
    println!("{:X}", Saturating(255u32));
    println!("{:o}", Saturating(8u32));
    println!("{:b}", Saturating(5u32));
    println!("{:>8}", Saturating(42i32));
    println!("{:08}", Saturating(42i32));
    println!("{:+}", Saturating(42i32));
    println!("{:#x}", Saturating(255u32));
    println!("{:?}", Saturating(2.5f64));
    println!("{}", Saturating(2.5f64));
    println!("{:?}", Saturating('c'));
    println!("{}", Saturating('c'));
}
"#;

#[test]
fn run_mode_saturating_fmt_forwards_to_inner() {
    // Core's fmt impls for `std::num::Saturating<T>` forward
    // to the inner value's own impl, exactly like `Wrapping`:
    // the wrapper's name never appears, for `Debug` (plain
    // and alternate) as for `Display` and the radix traits,
    // the formatting options pass through unchanged, at the
    // top level and in a derived `Debug` field.
    run_ok(
        "saturating-fmt",
        SATURATING_FMT_SRC,
        "5\n-5\n5\n18446744073709551615\n[1, 2, 3]\n[\n    1,\n    2,\n]\n\"hi\"\nSome(3)\n(1, 2)\nHolder { s: 7 }\nHolder {\n    s: 7,\n}\n5\nff\nFF\n10\n101\n      42\n00000042\n+42\n0xff\n2.5\n2.5\n'c'\nc\n",
    );
}

const DYN_ANY_DEBUG_SRC: &str = r#"
use std::any::Any;
use std::fmt::Debug;
#[derive(Debug)]
struct Named { x: i32 }
fn main() {
    let b: Box<dyn Any> = Box::new(Named { x: 1 });
    println!("{:?}", b);
    let d: Box<dyn Debug> = Box::new(Named { x: 2 });
    println!("{:?}", d);
}
"#;

#[test]
fn run_mode_dyn_any_debug_is_opaque() {
    // The `Debug` entry of a `dyn Any` vtable is `Any`'s own
    // impl, which prints the opaque `Any { .. }` whatever the
    // concrete type; a `dyn Debug` value formats its concrete
    // type as usual.
    run_ok(
        "dyn-any-debug",
        DYN_ANY_DEBUG_SRC,
        "Any { .. }\nNamed { x: 2 }\n",
    );
}

const DEBUG_FORWARD_TRANSPARENCY_SRC: &str = r#"
use std::cmp::Reverse;
use std::ffi::OsString;
use std::num::Wrapping;
use std::ops::RangeFrom;
use std::path::PathBuf;

fn main() {
    // Non-transparent one-field wrappers over byte strings
    // keep their own `Debug` text: `Reverse` is derived and
    // `RangeFrom` appends its operator.
    println!("{:?}", Reverse(PathBuf::from("hello")));
    println!("{:?}", Reverse(OsString::from("hello")));
    println!("{:?}", RangeFrom { start: PathBuf::from("hello") });
    // Known transparent adapters still forward to the field.
    println!("{:?}", Wrapping(PathBuf::from("hello")));
    println!("{:?}", Wrapping(Reverse(PathBuf::from("hello"))));
    println!("{:?}", Reverse(Wrapping(PathBuf::from("hello"))));
}
"#;

#[test]
fn run_mode_debug_forward_only_transparent_adapters() {
    let dir = temp_dir("debug-forward-transparency");
    let src = fixture(
        &dir,
        "debug_forward_transparency.rs",
        DEBUG_FORWARD_TRANSPARENCY_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Debug forwarding transparency runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "Reverse(\"hello\")\nReverse(\"hello\")\n\"hello\"..\n\"hello\"\nReverse(\"hello\")\nReverse(\"hello\")\n",
        "Debug forwards only through transparent adapters"
    );
}

const FMT_HANDWRITTEN_DEBUG_SRC: &str = r#"
use std::fmt;
struct Foo;
impl fmt::Debug for Foo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        println!("<Foo as Debug>::fmt()");
        Ok(())
    }
}
fn main() {
    println!("{:?}", Foo);
    println!("{:?}", Foo);
}
"#;

#[test]
fn run_mode_handwritten_debug_body_executes() {
    let dir = temp_dir("fmt-handwritten-debug");
    let src = fixture(&dir, "handwritten.rs", FMT_HANDWRITTEN_DEBUG_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "must run; stdout={stdout}");
    assert_eq!(
        stdout, "<Foo as Debug>::fmt()\n\n<Foo as Debug>::fmt()\n\n",
        "the handwritten Debug body itself must produce the output"
    );
}

const FMT_DERIVED_AND_BUILDER_SRC: &str = r#"
use std::fmt;
#[derive(Debug)]
struct Point { x: i32, y: i32 }
#[derive(Debug)]
struct Six { a: i32, b: i32, c: i32, d: i32, e: i32, f: i32 }
struct Manual { x: i32 }
impl fmt::Debug for Manual {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Manual").field("x", &self.x).finish()
    }
}
struct Wrap([i16; 3]);
impl fmt::Debug for Wrap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}
fn main() {
    println!("{:?}", Point { x: 1, y: 2 });
    println!("{:#?}", Point { x: 1, y: 2 });
    println!("{:?}", Six { a: 1, b: 2, c: 3, d: 4, e: 5, f: 6 });
    println!("{:?}", Manual { x: 42 });
    println!("{:?}", Wrap([10, -3, -9]));
}
"#;

#[test]
fn run_mode_derived_debug_finishers_and_builders() {
    let dir = temp_dir("fmt-derived-builder");
    let src = fixture(&dir, "derived_builder.rs", FMT_DERIVED_AND_BUILDER_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "must run; stdout={stdout}");
    assert_eq!(
        stdout,
        "Point { x: 1, y: 2 }\nPoint {\n    x: 1,\n    y: 2,\n}\nSix { a: 1, b: 2, c: 3, d: 4, e: 5, f: 6 }\nManual { x: 42 }\n[10, -3, -9]\n",
        "derived bodies run through the capturing Formatter: finisher \
         builtins for the field-count forms and the fields form, the \
         debug_struct builder chain, and the array Debug list chain"
    );
}

const FMT_HANDWRITTEN_DISPLAY_SRC: &str = r#"
use std::fmt;
struct Thingy { x: i32, y: i32 }
impl fmt::Debug for Thingy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{ x: {:?}, y: {:?} }}", self.x, self.y)
    }
}
struct Poly<T> { x: T }
impl<T: fmt::Debug> fmt::Debug for Poly<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.x)
    }
}
struct Word;
impl fmt::Display for Word {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("word")
    }
}
fn main() {
    println!("{:?}", Thingy { x: 1, y: 2 });
    println!("{:?}", Poly { x: Thingy { x: 1, y: 2 } });
    println!("{}", Word);
}
"#;

#[test]
fn run_mode_handwritten_generic_debug_and_display_bodies() {
    let dir = temp_dir("fmt-handwritten-generic");
    let src = fixture(&dir, "handwritten_generic.rs", FMT_HANDWRITTEN_DISPLAY_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "must run; stdout={stdout}");
    assert_eq!(
        stdout, "{ x: 1, y: 2 }\n{ x: 1, y: 2 }\nword\n",
        "generic handwritten impls resolve through their instance \
         names, and a handwritten Display body runs for the plain \
         placeholder"
    );
}

const IO_ERROR_FMT_SRC: &str = r#"
use std::io;
fn main() {
    for code in [2, 13, 22] {
        let e = io::Error::from_raw_os_error(code);
        println!("{} | {:?} | {:?} | {:?}", e, e, e.kind(), e.raw_os_error());
    }
    let simple: io::Error = io::ErrorKind::NotFound.into();
    println!("{} | {:?}", simple, simple);
    println!("{}", io::ErrorKind::PermissionDenied);
    let fs_err = std::fs::read("/definitely/not/here-lh-test.xyz").unwrap_err();
    println!("{} | {:?}", fs_err, fs_err);
}
"#;

#[test]
fn run_mode_io_error_display_and_debug() {
    // `io::Error` in the bitpacked representation formats the
    // way core's own impls do: an OS error's text carries the
    // platform's message for the code, its `Debug` is the `Os`
    // struct form, a kind-only error displays as the kind's
    // message and debugs as `Kind(..)`, and an `ErrorKind`
    // displays as its message.
    let dir = temp_dir("io-error-fmt");
    let src = fixture(&dir, "io_error_fmt.rs", IO_ERROR_FMT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "No such file or directory (os error 2) | Os { code: 2, kind: NotFound, message: \"No such file or directory\" } | NotFound | Some(2)\n\
         Permission denied (os error 13) | Os { code: 13, kind: PermissionDenied, message: \"Permission denied\" } | PermissionDenied | Some(13)\n\
         Invalid argument (os error 22) | Os { code: 22, kind: InvalidInput, message: \"Invalid argument\" } | InvalidInput | Some(22)\n\
         entity not found | Kind(NotFound)\n\
         permission denied\n\
         No such file or directory (os error 2) | Os { code: 2, kind: NotFound, message: \"No such file or directory\" }\n"
    );
}

const IO_ERROR_SIMPLE_MESSAGE_FMT_SRC: &str = r#"
use std::io::{self, Read, Write};

struct ZeroWriter;
impl io::Write for ZeroWriter {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> { Ok(0) }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

fn main() {
    let data: &[u8] = &[1, 2, 3];
    let mut buf = [0u8; 8];
    let mut slice = data;
    let err = slice.read_exact(&mut buf).unwrap_err();
    println!("{} | {:?} | {:?}", err, err, err.kind());
    let mut slice2 = data;
    let r: io::Result<()> = slice2.read_exact(&mut buf);
    println!("{:?}", r);
    let mut w = ZeroWriter;
    let err2 = w.write_all(b"hello").unwrap_err();
    println!("{} | {:?} | {:?}", err2, err2, err2.kind());
}
"#;

#[test]
fn run_mode_io_error_simple_message_display_and_debug() {
    // A guest-built `SimpleMessage` error (the `read_exact` EOF
    // error, the `write_all` zero error) stores its
    // `&'static SimpleMessage` through the repr's type-erased
    // slot; its `Display` is the message text and its `Debug`
    // is core's `Error { kind, message }` form, including when
    // the error is nested inside a `Result`'s `Debug`.
    let dir = temp_dir("io-error-simple-message-fmt");
    let src = fixture(&dir, "io_error_simple_message.rs", IO_ERROR_SIMPLE_MESSAGE_FMT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "failed to fill whole buffer | Error { kind: UnexpectedEof, message: \"failed to fill whole buffer\" } | UnexpectedEof\n\
         Err(Error { kind: UnexpectedEof, message: \"failed to fill whole buffer\" })\n\
         failed to write whole buffer | Error { kind: WriteZero, message: \"failed to write whole buffer\" } | WriteZero\n"
    );
}

const POINTER_WRAPPER_FMT_SRC: &str = r#"
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;
struct Wrap(u32);
impl fmt::Display for Wrap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "wrap<{}>", self.0) }
}
fn main() {
    println!("{} {:?}", Box::new(10isize), Box::new(String::from("bs")));
    println!("{}", Box::new(Wrap(5)));
    println!("{} {:?}", Rc::new(11), Rc::new(vec![1, 2]));
    println!("{} {:?}", Arc::new(3.25f32), Arc::new(String::from("arcstr")));
    println!("{}", Box::new(Box::new(9)));
}
"#;

#[test]
fn run_mode_pointer_wrapper_display_and_debug() {
    // The standard pointer wrappers forward `Display` and
    // `Debug` to the pointee: a `Box` prints as its contents
    // (through a guest `Display` body when the pointee has
    // one), and `Rc` / `Arc` print as the value behind the
    // handle rather than as the handle's own fields.
    let dir = temp_dir("pointer-wrapper-fmt");
    let src = fixture(&dir, "pointer_wrapper_fmt.rs", POINTER_WRAPPER_FMT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "10 \"bs\"\nwrap<5>\n11 [1, 2]\n3.25 \"arcstr\"\n9\n"
    );
}

const DYN_DISPLAY_SRC: &str = r#"
use std::fmt::{Debug, Display, LowerHex};

trait Greeter: Display + Debug {}

#[derive(Debug)]
struct Bot {
    id: u32,
}
impl Display for Bot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bot#{}", self.id)
    }
}
impl Greeter for Bot {}

#[derive(Debug)]
struct MyErr;
impl Display for MyErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "my error")
    }
}
impl std::error::Error for MyErr {}

fn show(d: &dyn Display) {
    println!("{:>6}", d);
}

fn main() {
    let b: Box<dyn Display> = Box::new(42i32);
    println!("{}", b);
    println!("{:>8}", b);
    let s: Box<dyn Display> = Box::new(String::from("hello"));
    println!("{:.3}", s);
    let r: std::rc::Rc<dyn Display> = std::rc::Rc::new(String::from("rc-text"));
    println!("{}", r);
    let a: std::sync::Arc<dyn Display> = std::sync::Arc::new(123i32);
    println!("{}", a);
    show(&77i32);
    let g: Box<dyn Greeter> = Box::new(Bot { id: 5 });
    println!("{}", g);
    println!("{:?}", g);
    let e: Box<dyn std::error::Error> = Box::new(MyErr);
    println!("{}", e);
    println!("{:?}", e);
    let h: Box<dyn LowerHex> = Box::new(255u32);
    println!("{:x}", &*h);
    println!("{:X}", 255u32);
}
"#;

#[test]
fn run_mode_dyn_display_formats_concrete() {
    // A trait object's fmt traits are the concrete type's (the
    // vtable entries, for the principal trait and its
    // supertraits alike): `Box` / `Rc` / `Arc` / `&dyn` Display
    // with format specifiers, a supertrait object's Display and
    // Debug, a `dyn Error`, and a radix trait through `&dyn`.
    let dir = temp_dir("dyn-display");
    let src = fixture(&dir, "dyn_display.rs", DYN_DISPLAY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dyn Display formatting must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "42\n      42\nhel\nrc-text\n123\n    77\nbot#5\nBot { id: 5 }\nmy error\nMyErr\nff\nFF\n"
    );
}

const FMT_WRAPPER_LIFETIME_SRC: &str = r#"
use std::fmt::{Display, Formatter, Result};

struct Wrap<'a>(&'a dyn Display);
impl<'a> Display for Wrap<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Display::fmt(self.0, f)
    }
}

struct Named<'a>(&'a i32);
impl<'a> Display for Named<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "named:{}", self.0)
    }
}

fn main() {
    let boxed: Box<dyn Display> = Box::new(13i32);
    println!("{}", Wrap(&*boxed));
    let n = 21i32;
    println!("{}", Named(&n));
}
"#;

#[test]
fn run_mode_fmt_wrapper_with_lifetime_formats() {
    // A guest `fmt` body whose `Self` carries a lifetime: the
    // instance name spells the lifetime (`<'_>` / `<'a>`) while
    // the argument's type tag drops it, so the body lookup must
    // normalise lifetime arguments away. The wrapper's UFCS
    // `Display::fmt` call on its `&dyn Display` field is served
    // by the same dyn dispatch.
    let dir = temp_dir("fmt-wrapper-lifetime");
    let src = fixture(&dir, "fmt_wrapper.rs", FMT_WRAPPER_LIFETIME_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "lifetime-carrying fmt wrappers must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "13\nnamed:21\n");
}

const DYN_ONE_FIELD_REF_SRC: &str = r#"
#[derive(Debug)]
struct A {}

#[derive(Debug)]
struct B<'a> {
    ed: &'a mut A,
}

#[derive(Debug)]
struct C<'a> {
    pd: &'a mut B<'a>,
}

fn main() {
    let mut a = A {};
    let mut b = B { ed: &mut a };
    let c = C { pd: &mut b };
    println!("{:?}", c);
    let mut a2 = A {};
    let b2 = B { ed: &mut a2 };
    let d: &dyn std::fmt::Debug = &b2;
    println!("{:?}", d);
    let mut a3 = A {};
    let b3 = B { ed: &mut a3 };
    let v: Vec<Box<dyn std::fmt::Debug>> = vec![Box::new(b3)];
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_dyn_one_field_ref_struct_debug() {
    // A one-field struct whose field is a reference is the
    // payload itself, not a box aggregate: the derived `Debug`
    // bodies of the nested `B` / `C` shapes must run on the
    // struct values, and a `dyn Debug` over such a struct must
    // dispatch to the struct's own `Debug`.
    let dir = temp_dir("dyn-one-field-ref");
    let src = fixture(&dir, "dyn_one_field.rs", DYN_ONE_FIELD_REF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "one-field ref struct Debug must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "C { pd: B { ed: A } }\nB { ed: A }\n[B { ed: A }]\n"
    );
}

const VECDEQUE_DEBUG_SRC: &str = r#"
use std::collections::VecDeque;

#[derive(Debug)]
struct Wrap {
    items: VecDeque<i32>,
}

fn main() {
    let mut d: VecDeque<i32> = VecDeque::with_capacity(4);
    for i in 0..4 { d.push_back(i); }
    d.pop_front();
    d.pop_front();
    d.push_back(4);
    d.push_back(5);
    println!("{:?}", d);
    println!("{:#?}", d);
    let e: VecDeque<i32> = VecDeque::new();
    println!("{:?}", e);
    println!("{:?}", VecDeque::from(vec![7, 8, 9]));
    println!("{:?}", VecDeque::from([10, 11]));
    let mut g: VecDeque<String> = VecDeque::new();
    g.push_back("a".to_string());
    g.push_front("z".to_string());
    println!("{:?}", g);
    let w = Wrap { items: VecDeque::from([3, 1, 4]) };
    println!("{:?}", w);
    println!("{:?}", Box::new(VecDeque::from([7])));
}
"#;

#[test]
fn run_mode_vecdeque_debug_list_format() {
    // A `VecDeque` value is the ring-buffer aggregate (wrapped
    // head, length, buffer); its `Debug` is the `debug_list` of
    // the logical element sequence, so the structural fmt engine
    // renders the sequence, not the aggregate's fields.
    let dir = temp_dir("vecdeque-debug");
    let src = fixture(&dir, "vecdeque_debug.rs", VECDEQUE_DEBUG_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "VecDeque Debug program must exit 0; stderr={stderr}"
    );
    let expected = "\
[2, 3, 4, 5]
[
    2,
    3,
    4,
    5,
]
[]
[7, 8, 9]
[10, 11]
[\"z\", \"a\"]
Wrap { items: [3, 1, 4] }
[7]
";
    assert_eq!(stdout, expected, "VecDeque Debug output must be the list form");
}

const IO_ERROR_CUSTOM_FMT_SRC: &str = r#"
use std::error::Error;
use std::fmt;

#[derive(Debug)]
struct MyErr;
impl fmt::Display for MyErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "my error display")
    }
}
impl Error for MyErr {}

#[derive(Debug)]
struct Inner {
    x: i32,
    name: &'static str,
}
impl fmt::Display for Inner {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "inner display")
    }
}
impl Error for Inner {}

fn main() {
    let e = std::io::Error::new(std::io::ErrorKind::InvalidInput, MyErr);
    println!("plain: {e}");
    println!("debug: {e:?}");
    println!("prec: [{}]", format!("{e:.3}"));
    println!("width: [{}]", format!("{e:>20}"));
    println!("pretty:\n{}", format!("{e:#?}"));
    let e2 = std::io::Error::new(std::io::ErrorKind::Other, Inner { x: 5, name: "n" });
    println!("nested pretty:\n{}", format!("{e2:#?}"));
    println!("nested debug: {e2:?}");
    let v = vec![std::io::Error::new(std::io::ErrorKind::Other, MyErr)];
    println!("in vec pretty:\n{v:#?}");
    println!("done");
}
"#;

#[test]
fn run_mode_io_error_custom_payload_fmt_forwarding() {
    // `io::Error`'s `Display` forwards the formatter to the
    // payload's own `Display`: a payload that ignores precision
    // and width (a plain `write!` of a literal) renders in full
    // under `{e:.3}` and unpadded under `{e:>20}`, exactly as
    // native. Its `Debug` is core's `Custom` `debug_struct`, so
    // under `{:#?}` it takes the multi-line layout with the
    // payload's own multi-line `Debug` indented under its
    // field, while `{}` / `{:?}` keep the plain / compact
    // forms.
    let dir = temp_dir("io-error-custom-fmt");
    let src = fixture(&dir, "io_error_custom_fmt.rs", IO_ERROR_CUSTOM_FMT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "plain: my error display\ndebug: Custom { kind: InvalidInput, error: MyErr }\nprec: [my error display]\nwidth: [my error display]\npretty:\nCustom {\n    kind: InvalidInput,\n    error: MyErr,\n}\nnested pretty:\nCustom {\n    kind: Other,\n    error: Inner {\n        x: 5,\n        name: \"n\",\n    },\n}\nnested debug: Custom { kind: Other, error: Inner { x: 5, name: \"n\" } }\nin vec pretty:\n[\n    Custom {\n        kind: Other,\n        error: MyErr,\n    },\n]\ndone\n"
    );
}

const NON_DATA_POINTER_FMT_SRC: &str = r#"
fn main() {
    println!("{:?}", 1 as *const String);
    println!("{:p}", 1 as *const String);
    println!("{:?}", std::ptr::null::<u8>());
}
"#;

#[test]
fn run_mode_non_data_pointer_formats_as_address() {
    run_ok(
        "non-data-pointer-fmt",
        NON_DATA_POINTER_FMT_SRC,
        "0x1\n0x1\n0x0\n",
    );
}

const NON_DATA_POINTER_FMT_FLAGS_SRC: &str = r#"
fn main() {
    let p = 1usize as *const u8;
    let q = 0xdeadbeefusize as *const u8;
    println!("[{:8p}]", p);
    println!("[{:<8p}]", p);
    println!("[{:^8p}]", p);
    println!("[{:08p}]", p);
    println!("[{:+p}]", p);
    println!("[{:+08p}]", p);
    println!("[{:#p}]", p);
    println!("[{:#?}]", p);
    println!("[{:<#20p}]", p);
    println!("[{:<08p}]", p);
    println!("[{:*>8p}]", p);
    println!("[{:0>8p}]", p);
    println!("[{:#?}]", q);
    println!("[{:08p}]", std::ptr::null::<u8>());
}
"#;

#[test]
fn run_mode_non_data_pointer_format_flags() {
    run_ok(
        "non-data-pointer-fmt-flags",
        NON_DATA_POINTER_FMT_FLAGS_SRC,
        "[     0x1]\n[0x1     ]\n[  0x1   ]\n[0x000001]\n[+0x1]\n[+0x00001]\n[0x0000000000000001]\n[0x0000000000000001]\n[0x000000000000000001]\n[0x000001]\n[*****0x1]\n[000000x1]\n[0x00000000deadbeef]\n[0x000000]\n",
    );
}

const ERROR_DOWNCAST_REBOX_FMT_SRC: &str = r#"
use std::error::Error;
use std::fmt;
#[derive(Debug)]
struct MyErr { code: u32 }
impl fmt::Display for MyErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "my err {}", self.code) }
}
impl Error for MyErr {}
fn main() {
    let boxed: Box<dyn Error> = Box::new(MyErr { code: 7 });
    let b = boxed.downcast::<MyErr>().unwrap();
    let e: Box<dyn Error> = b;
    println!("display: {}", e);
    println!("debug: {:?}", e);
    println!("done");
}
"#;

#[test]
fn run_mode_error_downcast_rebox_formats_as_dyn() {
    // Formatting the trait object a by-value downcast rebuilt:
    // the sized box the downcast produces is an aggregate-built
    // box (its `Unique` at field 0 beside the allocator field),
    // so the rebuilt `Box<dyn Error>`'s data pointer denotes the
    // payload through that aggregate. Its `Display` / `Debug`
    // are the concrete type's, reached through the box aggregate
    // the way dereferencing the same base reaches the payload.
    run_ok(
        "error-downcast-rebox-fmt",
        ERROR_DOWNCAST_REBOX_FMT_SRC,
        "display: my err 7\ndebug: MyErr { code: 7 }\ndone\n",
    );
}

const TO_STRING_DISPLAY_SHAPES_SRC: &str = r#"
use std::error::Error;
use std::fmt;
#[derive(Debug)]
struct MyErr { code: u32 }
impl fmt::Display for MyErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "my err {}", self.code) }
}
impl Error for MyErr {}
fn main() {
    let e: &dyn Error = &MyErr { code: 5 };
    println!("dyn to_string: {}", e.to_string());
    println!("float: {}", 1.5f64.to_string());
    println!("custom: {}", MyErr { code: 3 }.to_string());
    let raw = Box::into_raw(Box::new(MyErr { code: 7 }));
    let b2: Box<MyErr> = unsafe { Box::from_raw(raw) };
    println!("from_raw debug: {:?}", b2);
    println!("from_raw display: {}", b2);
}
"#;

#[test]
fn run_mode_to_string_display_shapes() {
    // `to_string` is the receiver's `Display` formatting: a
    // trait object's is its concrete type's, a float's is its
    // decimal form, a guest type's is its collected body. A
    // `Box::from_raw` product (the aggregate-built box shape)
    // forwards `Debug` / `Display` to its pointee like a built
    // box does.
    run_ok(
        "to-string-display-shapes",
        TO_STRING_DISPLAY_SHAPES_SRC,
        "dyn to_string: my err 5\nfloat: 1.5\ncustom: my err 3\nfrom_raw debug: MyErr { code: 7 }\nfrom_raw display: my err 7\n",
    );
}

const IPV4_DISPLAY_SRC: &str = r#"
use std::net::{IpAddr, Ipv4Addr, SocketAddrV4};
fn main() {
    let a = Ipv4Addr::new(192, 168, 0, 1);
    println!("display: {}", a);
    println!("debug: {:?}", a);
    println!("alt debug: {:#?}", a);
    println!("localhost: {}", Ipv4Addr::LOCALHOST);
    println!("unspecified: {:?}", Ipv4Addr::UNSPECIFIED);
    println!("broadcast: {}", Ipv4Addr::BROADCAST);
    println!("padded: [{:>15}]", a);
    println!("precision: [{:.3}]", a);
    println!("to_string: {}", a.to_string());
    println!("formatted: {}", format!("{}", Ipv4Addr::new(9, 8, 7, 6)));
    println!("ipaddr: {} {:?}", IpAddr::V4(a), IpAddr::V4(a));
    println!("socket: {} {:?}", SocketAddrV4::new(a, 8080), SocketAddrV4::new(a, 8080));
    println!("predicates: {} {}", a.is_private(), Ipv4Addr::LOCALHOST.is_loopback());
    println!("octets: {:?}", a.octets());
}
"#;

#[test]
fn run_mode_ipv4_addr_display_and_debug() {
    // `Ipv4Addr`'s `Display` is the dotted-decimal text and
    // core defines its `Debug` as the same text (not the
    // derived struct form); width / precision go through
    // `Formatter::pad`. The address constants materialise in
    // the same shape, and `IpAddr::V4` / `SocketAddrV4`
    // forward to it.
    run_ok(
        "ipv4-display",
        IPV4_DISPLAY_SRC,
        "display: 192.168.0.1\ndebug: 192.168.0.1\nalt debug: 192.168.0.1\nlocalhost: 127.0.0.1\nunspecified: 0.0.0.0\nbroadcast: 255.255.255.255\npadded: [    192.168.0.1]\nprecision: [192]\nto_string: 192.168.0.1\nformatted: 9.8.7.6\nipaddr: 192.168.0.1 192.168.0.1\nsocket: 192.168.0.1:8080 192.168.0.1:8080\npredicates: true true\noctets: [192, 168, 0, 1]\n",
    );
}

const BUFFER_FIELD_DYN_DEBUG_SRC: &str = r#"
#[derive(Debug)]
struct W { b: Box<[u8]> }
#[derive(Debug)]
struct S { w: W, n: u32 }
#[derive(Debug)]
struct T(W, u32);
use std::fmt::Debug;
fn show(d: &dyn Debug) { println!("dyn={:?}", d); }
fn main() {
    let s = S { w: W { b: vec![10u8, 20, 30].into_boxed_slice() }, n: 1 };
    println!("{:?}", s);
    let t = T(W { b: vec![40u8, 50].into_boxed_slice() }, 2);
    println!("{:?}", t);
    show(&s.w);
    let rc: std::rc::Rc<dyn Debug> = std::rc::Rc::new(vec![7u8, 8]);
    println!("rc={:?}", rc);
    let bx: Box<dyn Debug> = Box::new(String::from("boxed"));
    println!("box={:?}", bx);
}
"#;

#[test]
fn run_mode_buffer_shaped_field_dyn_debug_reads_whole_value() {
    // A one-field aggregate over a one-field box aggregate over
    // an allocation pointer (a wrapper around a boxed slice,
    // the shape a `CString` field also has) shares the value
    // shape of an `Rc` handle. Shaping such a field's trait
    // object as a handle would read a byte inside the buffer
    // instead of the field value; the handle reading is
    // licensed by the allocation's contents (an `RcInner`
    // aggregate), so the field formats as the whole value, and
    // genuine `Rc` / `Box` trait objects keep formatting their
    // pointees.
    run_ok(
        "buffer-field-dyn-debug",
        BUFFER_FIELD_DYN_DEBUG_SRC,
        "S { w: W { b: [10, 20, 30] }, n: 1 }\nT(W { b: [40, 50] }, 2)\ndyn=W { b: [10, 20, 30] }\nrc=[7, 8]\nbox=\"boxed\"\n",
    );
}

const CSTRING_FIELD_DYN_DEBUG_SRC: &str = r#"
use std::ffi::CString;
#[derive(Debug)]
struct S { name: CString, n: u32 }
#[derive(Debug)]
struct T(CString, u32);
fn main() {
    let s = S { name: CString::new("hello").unwrap(), n: 42 };
    let via_struct = format!("{:?}", s);
    let direct = format!("{:?}", s.name);
    println!("field_is_value={}", via_struct.contains(&direct));
    println!("struct_n={}", via_struct.contains("n: 42"));
    let t = T(CString::new("world").unwrap(), 7);
    let via_tuple = format!("{:?}", t);
    let direct_t = format!("{:?}", t.0);
    println!("tuple_field_is_value={}", via_tuple.contains(&direct_t));
}
"#;

#[test]
fn run_mode_cstring_field_dyn_debug_reads_whole_value() {
    // The invariant under test is that the field renders as
    // the `CString` value itself (the same text as formatting
    // the value directly), never as a single byte of its
    // buffer; the assertion is phrased over the direct
    // rendering so it does not depend on how `CString`'s own
    // `Debug` text is produced.
    run_ok(
        "cstring-field-dyn-debug",
        CSTRING_FIELD_DYN_DEBUG_SRC,
        "field_is_value=true\nstruct_n=true\ntuple_field_is_value=true\n",
    );
}

const HANDLE_DYN_DEBUG_SRC: &str = r#"
use std::fmt::Debug;
fn show(d: &dyn Debug) { println!("{:?}", d); }
#[derive(Debug)]
struct Holder { rc: std::rc::Rc<String>, bx: Box<Vec<u8>> }
fn main() {
    let rc = std::rc::Rc::new(String::from("rcstr"));
    show(&rc);
    let rc2: std::rc::Rc<dyn Debug> = std::rc::Rc::new(42u32);
    println!("{:?}", rc2);
    let arc: std::sync::Arc<dyn Debug> = std::sync::Arc::new(vec![1u8, 2, 3]);
    println!("{:?}", arc);
    let b: Box<dyn Debug> = Box::new(String::from("bx"));
    println!("{:?}", b);
    let h = Holder { rc: std::rc::Rc::new("held".to_string()), bx: Box::new(vec![9u8]) };
    println!("{:?}", h);
}
"#;

#[test]
fn run_mode_handle_dyn_debug_formats_pointee() {
    // The genuine reference-counted handles: their trait
    // objects and derived-`Debug` fields format the pointee
    // the handle names in its inner allocation.
    run_ok(
        "handle-dyn-debug",
        HANDLE_DYN_DEBUG_SRC,
        "\"rcstr\"\n42\n[1, 2, 3]\n\"bx\"\nHolder { rc: \"held\", bx: [9] }\n",
    );
}
const FMT_FORWARD_INT_DEBUG_SRC: &str = r#"
use std::fmt;

struct SlowFmt(u32);
impl fmt::Debug for SlowFmt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

struct Signed(i32);
impl fmt::Debug for Signed {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

fn main() {
    println!("{:?}{:?}", SlowFmt(7), SlowFmt(7));
    println!("{:?}", Signed(-7));
    println!("{:x?} {:X?}", SlowFmt(255), SlowFmt(255));
    println!("[{:>5?}][{:05?}]", SlowFmt(42), SlowFmt(42));
}
"#;

#[test]
fn run_mode_fmt_forward_integer_debug() {
    // A guest `Debug` impl forwarding to an integer field's
    // `Debug` (`self.0.fmt(f)`, the `atomic-print` shape): the
    // collected body of core's integer `Debug` impl is keyed
    // under the module-qualified impl name
    // (`core::fmt::num::<impl Debug for u32>::fmt`) and reads
    // the `Formatter`'s flag word directly, which the value
    // model does not reproduce — the call must be served by
    // the structural fmt engine, like an integer formatted as
    // an argument. The `{:x?}` / `{:X?}` lines are the flag
    // word's observable effect. Expected text is native output.
    run_ok(
        "fmt-fwd-int-debug",
        FMT_FORWARD_INT_DEBUG_SRC,
        "77\n-7\nff FF\n[   42][00042]\n",
    );
}

const FMT_FORWARD_INT_DEBUG_TYPES_SRC: &str = r#"
use std::fmt;

struct Byte(u8);
impl fmt::Debug for Byte {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

struct Wide(i64);
impl fmt::Debug for Wide {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

struct Inner(u32);
impl fmt::Debug for Inner {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

struct Outer(Inner);
impl fmt::Debug for Outer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

fn main() {
    println!("{:?} {:?} {:?}", Byte(3), Wide(-9), Outer(Inner(11)));
}
"#;

#[test]
fn run_mode_fmt_forward_integer_debug_types() {
    // The seam is in the integer `Debug` impl family as a
    // whole (every integer type's body in `core::fmt::num`
    // dispatches on the flag word), and it applies at any
    // forwarding depth: `Outer` forwards to the guest `Inner`,
    // whose own forwarding to its `u32` field hits the same
    // foreign body. Expected text is native output.
    run_ok(
        "fmt-fwd-int-debug-types",
        FMT_FORWARD_INT_DEBUG_TYPES_SRC,
        "3 -9 11\n",
    );
}

const STR_LINES_ZST_FN_SRC: &str = r#"
fn main() {
    let text = "alpha\r\nbeta\ngamma\r\n";
    let mut total = 0;
    for line in text.lines() {
        total += line.len();
        println!("{line}");
    }
    println!("total={total} count={}", text.lines().count());
    let mut it = "x\ny".lines();
    println!("{:?}", it.next());
    println!("{:?}", it.next_back());
    println!("{:?}", it.next());
}
"#;

#[test]
fn run_mode_str_lines_zst_fn() {
    // `str::lines` is a `Map` over the zero-sized `LinesMap`
    // callable (an `impl_fn_for_zst!` Fn type in core): its
    // value is an untyped empty ADT and no collected body calls
    // its `Fn::call` body, so the run needs the driver's ZST-Fn
    // demand and the interpreter's type-directed dispatch for
    // empty-ADT callables. The CRLF cases pin the body's own
    // suffix stripping (not a reimplementation): a line ending
    // in `\r\n` loses both bytes.
    run_ok(
        "str-lines-zst-fn",
        STR_LINES_ZST_FN_SRC,
        "alpha\nbeta\ngamma\ntotal=14 count=3\nSome(\"x\")\nSome(\"y\")\nNone\n",
    );
}

const STR_LINES_ZST_FN_COMPOSED_SRC: &str = r#"
fn main() {
    let kept: Vec<String> = "one\ntwo\nthree\nfour"
        .lines()
        .filter(|l| l.len() > 3)
        .map(|l| l.to_uppercase())
        .collect();
    println!("{kept:?}");
    let n = "a\nb\nc".lines().map(|l| l.len()).sum::<usize>();
    println!("sum={n}");
}
"#;

#[test]
fn run_mode_str_lines_zst_fn_composed() {
    // The ZST-Fn dispatch under adapter composition: `Lines`
    // stepped as the inner iterator of `Filter` / `Map`
    // adapters whose own callables are guest closures, and as
    // the source of the `collect` / `sum` consumers.
    run_ok(
        "str-lines-zst-fn-composed",
        STR_LINES_ZST_FN_COMPOSED_SRC,
        "[\"THREE\", \"FOUR\"]\nsum=3\n",
    );
}
const CHAR_ESCAPE_SRC: &str = r#"
fn main() {
    println!("debug-lit={}", 'a'.escape_debug().collect::<String>());
    println!("debug-lit2={}", 'Z'.escape_debug().collect::<String>());
    println!("debug-nl={}", '\n'.escape_debug().collect::<String>());
    println!("debug-tab={}", '\t'.escape_debug().collect::<String>());
    println!("debug-cr={}", '\r'.escape_debug().collect::<String>());
    println!("debug-nul={}", '\0'.escape_debug().collect::<String>());
    println!("debug-bs={}", '\\'.escape_debug().collect::<String>());
    println!("debug-dq={}", '"'.escape_debug().collect::<String>());
    println!("debug-sq={}", '\''.escape_debug().collect::<String>());
    println!("default-nl={}", '\n'.escape_default().collect::<String>());
    println!("default-bs={}", '\\'.escape_default().collect::<String>());
    println!("default-lit={}", 'a'.escape_default().collect::<String>());
    println!("default-del={}", '\u{7f}'.escape_default().collect::<String>());
    println!("default-01={}", '\u{1}'.escape_default().collect::<String>());
    println!("unicode-a={}", 'a'.escape_unicode().collect::<String>());
    println!("unicode-e9={}", '\u{e9}'.escape_unicode().collect::<String>());
    println!("default-e9={}", '\u{e9}'.escape_default().collect::<String>());
    println!("unicode-heart={}", '\u{2764}'.escape_unicode().collect::<String>());
    let mut it = 'q'.escape_debug();
    println!("step1={:?}", it.next());
    println!("step2={:?}", it.next());
    let mut it = '\n'.escape_default();
    println!("step3={:?}", it.next());
    println!("step4={:?}", it.next());
    println!("step5={:?}", it.next());
}
"#;

#[test]
fn run_mode_char_escape_iterators() {
    // The `char` escape iterators (`EscapeDebug` /
    // `EscapeDefault` / `EscapeUnicode`) stepped through
    // core's `EscapeIterInner` buffer state machine: the
    // literal form (a printable char stored as the union's
    // literal payload), the backslash forms, the `\xNN` /
    // `\u{...}` buffer forms, and manual stepping to
    // exhaustion. Expected output was produced by the
    // native 1.100.0 toolchain.
    run_ok(
        "char-escape-iterators",
        CHAR_ESCAPE_SRC,
        "debug-lit=a\n\
         debug-lit2=Z\n\
         debug-nl=\\n\n\
         debug-tab=\\t\n\
         debug-cr=\\r\n\
         debug-nul=\\0\n\
         debug-bs=\\\\\n\
         debug-dq=\\\"\n\
         debug-sq=\\'\n\
         default-nl=\\n\n\
         default-bs=\\\\\n\
         default-lit=a\n\
         default-del=\\u{7f}\n\
         default-01=\\u{1}\n\
         unicode-a=\\u{61}\n\
         unicode-e9=\\u{e9}\n\
         default-e9=\\u{e9}\n\
         unicode-heart=\\u{2764}\n\
         step1=Some('q')\n\
         step2=None\n\
         step3=Some('\\\\')\n\
         step4=Some('n')\n\
         step5=None\n",
    );
}

const CHAR_ESCAPE_CONSUMERS_SRC: &str = r#"
fn main() {
    let it = '\n'.escape_debug();
    println!("hint={:?} len={}", it.size_hint(), it.len());
    println!("count={}", '\n'.escape_debug().count());
    println!("nth={:?}", 'a'.escape_unicode().nth(3));
    println!("last={:?}", 'a'.escape_default().last());
    println!("last2={:?}", '\n'.escape_unicode().last());
    let mut it = 'x'.escape_unicode();
    println!("first={:?}", it.next());
    println!("hint-after={:?} len-after={}", it.size_hint(), it.len());
    println!("rest={}", it.collect::<String>());
    let upper: String = 'a'.escape_unicode().map(|c| c.to_ascii_uppercase()).collect();
    println!("upper={}", upper);
    let taken: String = '\n'.escape_debug().take(1).collect();
    println!("taken={}", taken);
    let vec: Vec<char> = '\\'.escape_default().collect();
    println!("vec={:?}", vec);
    let mut total = 0usize;
    for c in '\u{7f}'.escape_default() {
        total += c.len_utf8();
    }
    println!("total={}", total);
}
"#;

#[test]
fn run_mode_char_escape_iterator_consumers() {
    // The modelled consumers and adapters over the escape
    // iterators: exact size accounting (`size_hint` /
    // `len`) before and after a step, `count`, `nth`,
    // `last`, `Map` / `Take` composition, a by-value
    // `Vec<char>` collect and a `for` loop. Expected output
    // was produced by the native 1.100.0 toolchain.
    run_ok(
        "char-escape-iterator-consumers",
        CHAR_ESCAPE_CONSUMERS_SRC,
        "hint=(2, Some(2)) len=2\n\
         count=2\n\
         nth=Some('6')\n\
         last=Some('a')\n\
         last2=Some('}')\n\
         first=Some('\\\\')\n\
         hint-after=(5, Some(5)) len-after=5\n\
         rest=u{78}\n\
         upper=\\U{61}\n\
         taken=\\\n\
         vec=['\\\\', '\\\\']\n\
         total=6\n",
    );
}
