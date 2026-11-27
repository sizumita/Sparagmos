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

const COW_SLICE_DEREF_SRC: &str = r#"
use std::borrow::Cow;

fn total_len(c: &Cow<'_, [&str]>) -> usize {
    c.len()
}

fn main() {
    let owned: Cow<'_, [&str]> = Cow::Owned(vec!["alpha\n", "beta\n", "gamma\n"]);
    println!("owned len: {}", total_len(&owned));
    let s: &[&str] = &owned;
    println!("first: {}", s[0]);
    println!("joined: {}", s.join("|"));
    let mut count = 0;
    for line in &*owned {
        count += 1;
        println!("line {count}: {line:?} eq_one: {}", *line == "alpha\n");
    }
    let borrowed: Cow<'_, [&str]> = Cow::Borrowed(&["x\n", "y\n"]);
    println!("borrowed len: {}", total_len(&borrowed));
    let text: Cow<'_, str> = Cow::Owned(String::from("hello"));
    println!("text len: {} slice: {} starts: {}", text.len(), &text[1..3], text.starts_with("he"));
    let borrowed_text: Cow<'_, str> = Cow::Borrowed("world");
    println!("borrowed text len: {}", borrowed_text.len());
    let bytes: Cow<'_, [u8]> = Cow::Owned(vec![10u8, 20, 30]);
    println!("bytes len: {}", bytes.len());
    let bs: &[u8] = &bytes;
    println!("bytes sum: {} second: {}", bs.iter().map(|b| *b as u32).sum::<u32>(), bs[1]);
    println!("cow slice deref ok");
}
"#;

#[test]
fn run_mode_cow_slice_deref() {
    // `Cow`'s `Deref` builtin decided the string case by
    // substring: a `Cow<[&str]>` (the shape `similar`'s line
    // tokenizer hands to its diff) was read as a string target
    // and the owned `Vec<&str>` went to the string readers,
    // which refused it. The borrowed-type parameter must be
    // exactly `str` for the string case; a slice target derefs
    // to the fat reference over the owned vector.
    let dir = temp_dir("cow-slice-deref");
    let src = fixture(&dir, "cow_slice_deref.rs", COW_SLICE_DEREF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cow slice deref must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "owned len: 3\nfirst: alpha\n\njoined: alpha\n|beta\n|gamma\n\nline 1: \"alpha\\n\" eq_one: true\nline 2: \"beta\\n\" eq_one: false\nline 3: \"gamma\\n\" eq_one: false\nborrowed len: 2\ntext len: 5 slice: el starts: true\nborrowed text len: 5\nbytes len: 3\nbytes sum: 60 second: 20\ncow slice deref ok\n"
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

const DEBUG_FORMAT_DEREF_SLICE_SRC: &str = r#"
use std::borrow::Cow;

#[derive(Debug)]
struct Wrapper<'a> { s: &'a [u8], n: i32 }

fn main() {
    let v = vec![1u8, 2, 3];
    let s: &[u8] = &v;
    println!("vec: {:?}", s);
    println!("vec-alt: {:#?}", s);
    println!("deref-expr: {:?}", &*v);
    let mut mv = vec![4u8, 5];
    let ms: &mut [u8] = &mut mv;
    println!("mut: {:?}", ms);
    let owned: Cow<'_, [u8]> = Cow::Owned(vec![10u8, 11, 12]);
    println!("cow-owned: {:?}", &*owned);
    let borrowed: Cow<'_, [u8]> = Cow::Borrowed(&[7u8, 8, 9]);
    println!("cow-borrowed: {:?}", &*borrowed);
    let vi = vec![-1i8, 2, -3];
    println!("i8: {:?}", &*vi);
    let v16 = vec![1000u16, 2000, 3000];
    println!("u16: {:?}", &*v16);
    let mut sparse = Vec::with_capacity(4);
    sparse.push(4u8);
    sparse.push(5);
    sparse.push(6);
    println!("sparse: {:?}", &*sparse);
    let w = Wrapper { s: &v, n: 7 };
    println!("nested: {:?}", w);
    let cow_strs: Cow<'_, [&str]> = Cow::Owned(vec!["a\n", "b\n"]);
    println!("cow-strs: {:?}", &*cow_strs);
    println!("debug deref slice ok");
}
"#;

#[test]
fn run_mode_debug_format_dereferenced_slice() {
    // A slice produced by a `Deref` (a `Vec`'s, an owned
    // `Cow`'s) is a fat pointer denoting the container itself,
    // so resolving it hands the sequence formatter the
    // container's aggregate value, which it used to refuse
    // ("expected tuple or byte value, found ADT value"). The
    // sequence formatter now reads such a value's elements
    // through the slice element readers, as it does for a fat
    // pointer value.
    let dir = temp_dir("debug-format-deref-slice");
    let src = fixture(
        &dir,
        "debug_format_deref_slice.rs",
        DEBUG_FORMAT_DEREF_SLICE_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Debug of a dereferenced slice must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "vec: [1, 2, 3]\nvec-alt: [\n    1,\n    2,\n    3,\n]\nderef-expr: [1, 2, 3]\nmut: [4, 5]\ncow-owned: [10, 11, 12]\ncow-borrowed: [7, 8, 9]\ni8: [-1, 2, -3]\nu16: [1000, 2000, 3000]\nsparse: [4, 5, 6]\nnested: Wrapper { s: [1, 2, 3], n: 7 }\ncow-strs: [\"a\\n\", \"b\\n\"]\ndebug deref slice ok\n"
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

const PROJECTED_FIELD_DISPLAY_SRC: &str = r#"
struct Firework { strength: i32 }
impl Drop for Firework {
    fn drop(&mut self) { println!("BOOM times {}!!!", self.strength); }
}
struct Letter { mark: char }
struct Inner { x: i32 }
struct Outer { inner: Inner }
fn show(f: &Firework) { println!("shown {}", f.strength); }
fn main() {
    let f = Firework { strength: 100 };
    show(&f);
    println!("direct {}", f.strength);
    let l = Letter { mark: 'z' };
    println!("mark {}", l.mark);
    let o = Outer { inner: Inner { x: 7 } };
    println!("nested {}", o.inner.x);
    drop(f);
}
"#;

/// A raw fmt argument (see
/// `run_mode_o3_format_macro_uses_inlined_format_inner`) whose
/// erased reference denotes a *projected* place — a struct
/// field read straight from its local, through a reference
/// parameter, or inside a `Drop` impl — names the bare
/// trait-item formatter, and the Self type is the local's
/// declared type walked over the projection: the field's own
/// type, stated exactly (a `char` field's tag comes from the
/// field's recorded debug shape, the IR type rendering `char`
/// as `u32`). Without the walk the run stops at "Display
/// formatting for an ADT value (its impl lives in unconverted
/// code)", formatting the whole struct aggregate instead of
/// the field (the coverage corpus' `drop_trait` shape).
#[test]
fn run_mode_o2_projected_field_display_argument() {
    let dir = temp_dir("projected-field-display");
    let src = fixture(&dir, "case.rs", PROJECTED_FIELD_DISPLAY_SRC);
    for opt in ["-Copt-level=2", "-Copt-level=3"] {
        let out = run_driver_extra(&src, &["--rustc-arg", opt]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(0),
            "projected field arguments must run at {opt}; stderr={stderr}"
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "shown 100\ndirect 100\nmark z\nnested 7\nBOOM times 100!!!\n",
            "formatted output must match native at {opt}; stderr={stderr}"
        );
    }
}

const GENERIC_FIELD_DISPLAY_SRC: &str = r#"
struct Firework<T: Copy + std::fmt::Display> { strength: T }
impl<T: Copy + std::fmt::Display> Drop for Firework<T> {
    fn drop(&mut self) { println!("BOOM times {}!!!", self.strength); }
}
fn main() {
    let _a = Firework { strength: 2 };
    let _b = Firework { strength: 300.3f64 };
}
"#;

/// The projected-field recovery for a field whose type is the
/// ADT's generic parameter: the field-type table records no
/// type for it, so the Self type comes from the debug shape
/// recorded for the base local's exact instance (its
/// variable-table entry names `Firework<f64>` /
/// `Firework<i32>`), which states the field's type in that
/// instance. Without it the run stops at the same
/// unconverted-impl Display error (the coverage corpus'
/// `generics` shape).
#[test]
fn run_mode_o2_generic_field_display_argument() {
    let dir = temp_dir("generic-field-display");
    let src = fixture(&dir, "case.rs", GENERIC_FIELD_DISPLAY_SRC);
    for opt in ["-Copt-level=2", "-Copt-level=3"] {
        let out = run_driver_extra(&src, &["--rustc-arg", opt]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(0),
            "generic field arguments must run at {opt}; stderr={stderr}"
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "BOOM times 300.3!!!\nBOOM times 2!!!\n",
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

const STR_SPLIT_CHAR_SET_SRC: &str = r#"fn show(label: &str, v: Vec<&str>) {
    println!("{label}={v:?}");
}

fn main() {
    show("split", "a,b;c".split(&[',', ';'][..]).collect());
    show("rsplit", "a,b;c".rsplit(&[',', ';'][..]).collect());
    show("term", "a,b;c,".split_terminator(&[',', ';'][..]).collect());
    show("rterm", "a,b;c,".rsplit_terminator(&[',', ';'][..]).collect());
    show("incl", "a,b;c".split_inclusive(&[',', ';'][..]).collect());
    show("arr", "a,b;c".split([',', ';']).collect());
    show("refarr", "a,b;c".split(&[',', ';']).collect());
    let arr = [',', ';', 'x'];
    show("subslice", "a,b;c".split(&arr[..2]).collect());
    let v: Vec<char> = vec![',', ';'];
    show("vec", "a,b;c".split(&v[..]).collect());
    show("mb", "aαbβc".split(&['α', 'β'][..]).collect());
    show("empty-set", "a,b".split(&[][..]).collect());
    let mut it = "a,b;c,d".split(&[',', ';'][..]);
    println!("i1={:?}", it.next());
    println!("i2={:?}", it.next_back());
    println!("i3={:?}", it.next());
    println!("i4={:?}", it.next_back());
    println!("i5={:?}", it.next());
    let m: Vec<&str> = "a,b;c".matches(&[',', ';'][..]).collect();
    println!("matches={m:?}");
    let mi: Vec<(usize, &str)> = "a,b;c".match_indices(&[',', ';'][..]).collect();
    println!("indices={mi:?}");
    let ma: Vec<&str> = "a,b;c".matches([',', ';']).collect();
    println!("matches-arr={ma:?}");
    let mr: Vec<&str> = "a,b;c".matches(&[',', ';']).collect();
    println!("matches-refarr={mr:?}");
    println!("cnt={}", "a,b;c".split(&[',', ';'][..]).count());
}
"#;

#[test]
fn run_mode_str_split_char_set_family() {
    // The `Split` and `Matches` families over the char-set
    // forms of core's `MultiCharEqSearcher`: a `&[char]`
    // pattern's slice, a `[char; N]` pattern's chars by
    // value, a `&[char; N]` pattern's reference, a sub-slice
    // of a larger array, a `Vec`'s slice, multi-byte set
    // chars, the empty set, front/back interleaving on one
    // state, and the matches family's three set forms.
    // Expected output was produced by the native 1.100.0
    // toolchain.
    run_ok(
        "str-split-char-set-family",
        STR_SPLIT_CHAR_SET_SRC,
        "split=[\"a\", \"b\", \"c\"]\n\
         rsplit=[\"c\", \"b\", \"a\"]\n\
         term=[\"a\", \"b\", \"c\"]\n\
         rterm=[\"c\", \"b\", \"a\"]\n\
         incl=[\"a,\", \"b;\", \"c\"]\n\
         arr=[\"a\", \"b\", \"c\"]\n\
         refarr=[\"a\", \"b\", \"c\"]\n\
         subslice=[\"a\", \"b\", \"c\"]\n\
         vec=[\"a\", \"b\", \"c\"]\n\
         mb=[\"a\", \"b\", \"c\"]\n\
         empty-set=[\"a,b\"]\n\
         i1=Some(\"a\")\n\
         i2=Some(\"d\")\n\
         i3=Some(\"b\")\n\
         i4=Some(\"c\")\n\
         i5=None\n\
         matches=[\",\", \";\"]\n\
         indices=[(1, \",\"), (3, \";\")]\n\
         matches-arr=[\",\", \";\"]\n\
         matches-refarr=[\",\", \";\"]\n\
         cnt=3\n",
    );
}

const STR_MATCHES_SRC: &str = r#"fn main() {
    let v: Vec<(usize, &str)> = "αXβXγ".match_indices('X').collect();
    println!("mb={v:?}");
    let v2: Vec<(usize, &str)> = "αβγβδ".match_indices('β').collect();
    println!("mb-needle={v2:?}");
    let m: Vec<&str> = "aXbXc".matches('X').collect();
    println!("m={m:?}");
    let mut it = "a.b.c.d".match_indices('.');
    println!("i1={:?}", it.next());
    println!("i2={:?}", it.next_back());
    println!("i3={:?}", it.next());
    println!("i4={:?}", it.next_back());
    println!("i5={:?}", it.next());
    let w: Vec<&str> = "x=y=z".matches('=').rev().collect();
    println!("rev={w:?}");
    println!("none={}", "abc".matches('z').count());
    println!("empty={}", "".matches('z').count());
    let e: Vec<(usize, &str)> = "XaX".match_indices('X').collect();
    println!("ends={e:?}");
    let total: usize = "aXbXcX".match_indices('X').map(|(i, _)| i).sum();
    println!("sum={total}");
    println!("last={:?}", "aXbXc".matches('X').last());
    println!("cnt={}", "aXbXc".matches('X').count());
    let mut it2 = "a b c".match_indices(' ');
    let p = it2.next().unwrap();
    println!("eq={} len={} disp=[{}]", p.1 == " ", p.1.len(), p.1);
}
"#;

#[test]
fn run_mode_str_matches_family() {
    // The `Matches` / `MatchIndices` family over a `char`
    // pattern (core's `CharSearcher`): both directions and
    // their interleaving on one state, multi-byte haystacks
    // and needles with exact byte indices, the consumers,
    // and the yielded pieces as string values (equality,
    // length, Display). Expected output was produced by the
    // native 1.100.0 toolchain.
    run_ok(
        "str-matches-family",
        STR_MATCHES_SRC,
        "mb=[(2, \"X\"), (5, \"X\")]\n\
         mb-needle=[(2, \"β\"), (6, \"β\")]\n\
         m=[\"X\", \"X\"]\n\
         i1=Some((1, \".\"))\n\
         i2=Some((5, \".\"))\n\
         i3=Some((3, \".\"))\n\
         i4=None\n\
         i5=None\n\
         rev=[\"=\", \"=\"]\n\
         none=0\n\
         empty=0\n\
         ends=[(0, \"X\"), (2, \"X\")]\n\
         sum=9\n\
         last=Some(\"X\")\n\
         cnt=2\n\
         eq=true len=1 disp=[ ]\n"
    );
}

const STR_MATCHES_EQ_SRC: &str = r#"fn is_digit(c: char) -> bool {
    c.is_ascii_digit()
}

fn main() {
    let target = 'l';
    let p: Vec<(usize, &str)> = "hello world"
        .match_indices(|c: char| c == target)
        .collect();
    println!("pred={p:?}");
    let mut pit = "hello".match_indices(|c: char| c == 'l');
    println!("b1={:?}", pit.next_back());
    println!("f1={:?}", pit.next());
    println!("b2={:?}", pit.next_back());
    let pm: Vec<&str> = "hello".matches(|c: char| c == 'l').collect();
    println!("pred-m={pm:?}");
    let d: Vec<&str> = "a1b22c".matches(is_digit).collect();
    println!("fnptr={d:?}");
    let s: Vec<(usize, &str)> = "a1b2c3".match_indices(['1', '2', '3']).collect();
    println!("set={s:?}");
    let set: &[char] = &['a', 'b'];
    let mut sit = "abcba".match_indices(set);
    println!("s1={:?}", sit.next_back());
    println!("s2={:?}", sit.next());
    println!("s3={:?}", sit.next());
    println!("s4={:?}", sit.next_back());
    let sm: Vec<&str> = "xaybz".matches(['a', 'b']).collect();
    println!("set-m={sm:?}");
    println!("setcnt={}", "a1b2".matches(['1', '2']).count());
}
"#;

#[test]
fn run_mode_str_matches_eq_family() {
    // The `Matches` / `MatchIndices` family over the
    // `CharEqSearcher` patterns: a capturing closure and a
    // function-pointer predicate, and char sets as a
    // `[char; N]` pattern and a `&[char]` slice, in both
    // directions. Expected output was produced by the
    // native 1.100.0 toolchain.
    run_ok(
        "str-matches-eq-family",
        STR_MATCHES_EQ_SRC,
        "pred=[(2, \"l\"), (3, \"l\"), (9, \"l\")]\n\
         b1=Some((3, \"l\"))\n\
         f1=Some((2, \"l\"))\n\
         b2=None\n\
         pred-m=[\"l\", \"l\"]\n\
         fnptr=[\"1\", \"2\", \"2\"]\n\
         set=[(1, \"1\"), (3, \"2\"), (5, \"3\")]\n\
         s1=Some((4, \"a\"))\n\
         s2=Some((0, \"a\"))\n\
         s3=Some((1, \"b\"))\n\
         s4=Some((3, \"b\"))\n\
         set-m=[\"a\", \"b\"]\n\
         setcnt=2\n"
    );
}

const STR_SPLIT_WHITESPACE_SRC: &str = r#"
fn show(label: &str, v: Vec<&str>) {
    println!("{label}={v:?}");
}

fn main() {
    show("basic", " a b  c ".split_whitespace().collect());
    show("empty", "".split_whitespace().collect());
    show("ws-only", " \t\n\u{2003} ".split_whitespace().collect());
    show("single", "hello".split_whitespace().collect());
    show("unicode", "a\u{2003}b\u{00a0}c".split_whitespace().collect());
    let mut it = "a b  c d".split_whitespace();
    println!("f1={:?}", it.next());
    println!("b1={:?}", it.next_back());
    println!("f2={:?}", it.next());
    println!("b2={:?}", it.next_back());
    println!("b3={:?}", it.next_back());
    println!("cnt={}", "  one\ttwo\nthree  ".split_whitespace().count());
    show("rev", "p q r".split_whitespace().rev().collect());
    let lens: Vec<usize> = "aa b ccc".split_whitespace().map(|s| s.len()).collect();
    println!("lens={lens:?}");
}
"#;

#[test]
fn run_mode_str_split_whitespace() {
    // `str::split_whitespace`: core defines the iterator as
    // a one-field wrapper over
    // `Filter<Split<IsWhitespace>, IsNotEmpty>`, and both
    // callables are zero-sized `Fn` types (the
    // `impl_fn_for_zst!` family), so the step delegates to
    // the wrapped `Filter` under its tag and both predicates
    // are applied through the type-directed dispatch -- the
    // `IsWhitespace` pattern inside the split's predicate
    // searcher and the `IsNotEmpty` filter predicate.
    // Expected output was produced by the native 1.100.0
    // toolchain.
    run_ok(
        "str-split-whitespace",
        STR_SPLIT_WHITESPACE_SRC,
        "basic=[\"a\", \"b\", \"c\"]\n\
         empty=[]\n\
         ws-only=[]\n\
         single=[\"hello\"]\n\
         unicode=[\"a\", \"b\", \"c\"]\n\
         f1=Some(\"a\")\n\
         b1=Some(\"d\")\n\
         f2=Some(\"b\")\n\
         b2=Some(\"c\")\n\
         b3=None\n\
         cnt=3\n\
         rev=[\"r\", \"q\", \"p\"]\n\
         lens=[2, 1, 3]\n",
    );
}

const SLICE_SPLIT_SRC: &str = r#"
fn main() {
    // The vertical tab is not ASCII whitespace, so it
    // stays inside the piece (the shape of
    // tests/ui/whitespace/ascii_whitespace_excludes_vertical_tab.rs).
    let parts: Vec<&str> = "a\u{b}b".split_ascii_whitespace().collect();
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0], "a\u{b}b");
    let parts2: Vec<&str> = "a b".split_ascii_whitespace().collect();
    assert_eq!(parts2.len(), 2);
    assert_eq!(parts2[0], "a");
    assert_eq!(parts2[1], "b");
    println!("cnt={}", "  aa\tb\n ccc ".split_ascii_whitespace().count());
    println!("empty={}", "".split_ascii_whitespace().count());
    println!("ws={}", " \t\n".split_ascii_whitespace().count());
    let mut it = "a b c".split_ascii_whitespace();
    assert_eq!(it.next_back(), Some("c"));
    assert_eq!(it.next(), Some("a"));
    assert_eq!(it.next_back(), Some("b"));
    assert_eq!(it.next(), None);
    println!("saw ok");
    let data = [1u32, 0, 2, 3, 0, 4];
    let parts: Vec<&[u32]> = data.split(|x| *x == 0).collect();
    println!("split={parts:?}");
    fn is_zero(x: &u32) -> bool { *x == 0 }
    let parts4: Vec<&[u32]> = data.split(is_zero).collect();
    println!("fnptr={parts4:?}");
    let incl: Vec<&[u32]> = data.split_inclusive(|x| *x == 0).collect();
    println!("incl={incl:?}");
    let rs: Vec<&[u32]> = data.rsplit(|x| *x == 0).collect();
    println!("rsplit={rs:?}");
    let mut sp = data.split(|x| *x == 0);
    println!("nb1={:?}", sp.next_back());
    println!("nf1={:?}", sp.next());
    println!("nb2={:?}", sp.next_back());
    println!("nf2={:?}", sp.next());
    let t: Vec<&[u8]> = b"a;".split(|b| *b == b';').collect();
    println!("trail={}", t.len());
    println!("lead={}", b";a".split(|b| *b == b';').count());
}
"#;

#[test]
fn run_mode_slice_split_family() {
    // The `slice::Split` step machine and the
    // `SplitAsciiWhitespace` delegations: core's
    // `str::SplitAsciiWhitespace` is a one-field wrapper
    // over `Map<Filter<slice::Split<u8, IsAsciiWhitespace>,
    // BytesIsNotEmpty>, UnsafeBytesToStr>`, so its steps
    // delegate to the wrapped `Map` under that tag, and
    // the `slice::Split` state ([remaining, predicate,
    // finished]) is scanned with the predicate applied
    // through the modelled call path - a capturing closure
    // and a function pointer here. `SplitInclusive` ends
    // each piece just past the separator, and `RSplit`
    // forwards its front step to the wrapped state's back
    // step. Expected output was produced by the native
    // 1.100.0 toolchain.
    run_ok(
        "slice-split-family",
        SLICE_SPLIT_SRC,
        "cnt=3\n\
         empty=0\n\
         ws=0\n\
         saw ok\n\
         split=[[1], [2, 3], [4]]\n\
         fnptr=[[1], [2, 3], [4]]\n\
         incl=[[1, 0], [2, 3, 0], [4]]\n\
         rsplit=[[4], [2, 3], [1]]\n\
         nb1=Some([4])\n\
         nf1=Some([1])\n\
         nb2=Some([2, 3])\n\
         nf2=None\n\
         trail=2\n\
         lead=2\n",
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

const STR_LINES_ADAPTER_INNER_SRC: &str = r#"
fn main() {
    let out = String::from("{\"a\":1}\n{\"b\":2}\n");
    for (line, n) in out.lines().zip(1..) {
        println!("{n}: {line}");
    }
    let pairs: Vec<(&str, i32)> = "x\ny".lines().zip(1..).collect();
    println!("{pairs:?}");
    for (i, l) in "p\nq".lines().enumerate() {
        println!("{i}={l}");
    }
    for s in ["", "\n", "a\r\nb\r\n", "a\nb", "a\n\nb\n"] {
        let v: Vec<(&str, usize)> = s.lines().zip(0..).collect();
        println!("{v:?}");
    }
}
"#;

#[test]
fn run_mode_str_lines_adapter_inner() {
    // `Lines` stepped as the inner iterator of an adapter:
    // `Zip` and `Enumerate` step their inner iterators
    // through the state-level entry (`next_on_state`), which
    // had no routing for the one-field wrapper iterators, so
    // a `Lines` under an adapter hit the unsupported verdict
    // even though the in-place and back steps already routed
    // it. The first loop is the run-make recipe shape
    // (libtest-json zips the lines of captured output with
    // line numbers); the edge cases pin `Lines` semantics
    // through the adapter path (empty input yields nothing,
    // a lone newline yields one empty line, CRLF endings
    // lose both bytes, a missing trailing newline still
    // yields the last line, consecutive newlines yield the
    // empty line between them).
    run_ok(
        "str-lines-adapter-inner",
        STR_LINES_ADAPTER_INNER_SRC,
        "1: {\"a\":1}\n2: {\"b\":2}\n[(\"x\", 1), (\"y\", 2)]\n0=p\n1=q\n[]\n[(\"\", 0)]\n[(\"a\", 0), (\"b\", 1)]\n[(\"a\", 0), (\"b\", 1)]\n[(\"a\", 0), (\"\", 1), (\"b\", 2)]\n",
    );
}

const STR_SPLIT_WHITESPACE_ADAPTER_INNER_SRC: &str = r#"
fn main() {
    let sw: Vec<(&str, usize)> = "  a  b\tc ".split_whitespace().zip(0..).collect();
    println!("{sw:?}");
    let saw: Vec<(&str, usize)> = "  a  b\tc ".split_ascii_whitespace().zip(0..).collect();
    println!("{saw:?}");
}
"#;

#[test]
fn run_mode_str_split_whitespace_adapter_inner() {
    // The sibling one-field wrappers under the same
    // state-level step: `SplitWhitespace` and
    // `SplitAsciiWhitespace` zipped with a counter reach
    // their wrapped `Filter` / `Map` states through the
    // same `next_on_state` routing as `Lines`.
    run_ok(
        "str-split-whitespace-adapter-inner",
        STR_SPLIT_WHITESPACE_ADAPTER_INNER_SRC,
        "[(\"a\", 0), (\"b\", 1), (\"c\", 2)]\n[(\"a\", 0), (\"b\", 1), (\"c\", 2)]\n",
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

const CHAR_ESCAPE_DISPLAY_SRC: &str = r#"
fn main() {
    println!("dbg-nl={}", '\n'.escape_debug());
    println!("dbg-lit={}", 'a'.escape_debug());
    println!("dbg-emoji={}", '\u{1f600}'.escape_debug().to_string());
    println!("dft-nl={}", '\n'.escape_default());
    println!("uni-e9={}", '\u{e9}'.escape_unicode());
    println!("dbg-301={}", '\u{301}'.escape_debug());
    let mut it = '\n'.escape_debug();
    let _ = it.next();
    println!("dbg-rest={}", it);
}
"#;

#[test]
fn run_mode_char_escape_display() {
    // `Display` of the `char` escape iterators: core defines
    // the impls as draining the iterator into the formatter
    // (the `EscapeDebug` / `EscapeDefault` / `EscapeUnicode`
    // impls forward to the inner buffer state's own
    // drain-shaped `Display`), and the formatter follows
    // that definition over the modelled step — including a
    // partially consumed iterator, whose `Display` yields
    // only the remaining chars. The `dbg-301` line also
    // constructs its iterator through the unicode property
    // tables (see `run_mode_char_escape_non_ascii_construction`).
    // Expected output was produced by the native 1.100.0
    // toolchain.
    run_ok(
        "char-escape-display",
        CHAR_ESCAPE_DISPLAY_SRC,
        "dbg-nl=\\n\n\
         dbg-lit=a\n\
         dbg-emoji=\u{1f600}\n\
         dft-nl=\\n\n\
         uni-e9=\\u{e9}\n\
         dbg-301=\\u{301}\n\
         dbg-rest=n\n",
    );
}

const CHAR_ESCAPE_NON_ASCII_SRC: &str = r#"
fn main() {
    println!("{:?}", '\u{e9}'.escape_debug().collect::<Vec<char>>());
    println!("{:?}", '\u{301}'.escape_debug().collect::<Vec<char>>());
    println!("{:?}", '\u{600}'.escape_debug().collect::<Vec<char>>());
    println!("{:?}", '\u{ad}'.escape_debug().collect::<Vec<char>>());
    println!("{:?}", '\u{378}'.escape_debug().collect::<Vec<char>>());
    println!("{}", '\u{e000}'.escape_debug().count());
    println!("{}", '\u{7f}'.escape_debug().count());
    println!("{}", '\u{e9}'.is_alphabetic());
    println!("{}", '\u{301}'.is_alphabetic());
}
"#;

#[test]
fn run_mode_char_escape_non_ascii_construction() {
    // Constructing `escape_debug` for a non-ASCII char: the
    // decision path reads core's generated unicode property
    // tables through each property module's `lookup_slow`,
    // which has no MIR body in the shipped core. The
    // interpreter models `lookup_slow` by replaying core's
    // skiplist search over the module's two table statics,
    // whose seeds the driver records with the program (no
    // converted body ever references them). The cases cover
    // the grapheme-extend (U+0301), format-control and
    // default-ignorable (U+0600, U+00AD), unassigned
    // (U+0378) and private-use (U+E000) branches, plus the
    // alphabetic property read through the same model.
    // Dropping the constructed values exercises the union
    // inside the iterator state, whose fields carry no drop
    // glue. Expected output was produced by the native
    // 1.100.0 toolchain.
    run_ok(
        "char-escape-non-ascii",
        CHAR_ESCAPE_NON_ASCII_SRC,
        "['\u{e9}']\n\
         ['\\\\', 'u', '{', '3', '0', '1', '}']\n\
         ['\\\\', 'u', '{', '6', '0', '0', '}']\n\
         ['\\\\', 'u', '{', 'a', 'd', '}']\n\
         ['\\\\', 'u', '{', '3', '7', '8', '}']\n\
         8\n\
         6\n\
         true\n\
         false\n",
    );
}

const STR_ESCAPE_ITER_SRC: &str = r#"
fn main() {
    let mut n = 0;
    for _c in "a\nb".escape_debug() {
        n += 1;
    }
    println!("count-loop={n}");
    let mut s = String::new();
    for c in "x\u{1b}y".escape_default() {
        s.push(c);
    }
    println!("collected={s}");
    println!("count={}", "\u{10ffff}".escape_unicode().count());
    println!("nth={:?}", "ab\nc".escape_default().nth(2));
    let upper: String = "a\nb".escape_default().map(|c| c.to_ascii_uppercase()).collect();
    println!("upper={upper}");
    let taken: String = "hello\nworld".escape_unicode().take(7).collect();
    println!("taken={taken}");
    let mut done = "x".escape_default();
    println!("fused={:?} {:?} {:?}", done.next(), done.next(), done.next());
    let orig = "q\tz".escape_default();
    println!("clone={} orig={}", orig.clone().collect::<String>(), orig.collect::<String>());
    let mut m = 0;
    "a\rb".escape_unicode().for_each(|_| m += 1);
    println!("for-each={m}");
}
"#;

const STR_SEARCHER_SRC: &str = r#"fn main() {
    // TwoWay payload: multi-byte needles, incl. a periodic one.
    let t = "the quick the";
    println!("tw={:?}", t.matches("the").collect::<Vec<_>>());
    println!("twi={:?}", t.match_indices("the").collect::<Vec<_>>());
    println!("per={:?}", "abcabcabc".matches("abcab").collect::<Vec<_>>());
    println!("ovl={:?}", "aaaa".matches("aa").collect::<Vec<_>>());
    // Byte payload: a one-byte needle.
    println!("byte={:?}", "aXbXc".matches("X").collect::<Vec<_>>());
    println!("bytei={:?}", "aXbXc".match_indices("X").collect::<Vec<_>>());
    // Empty payload: a match at every char boundary.
    println!("emp={:?}", "ab".matches("").collect::<Vec<_>>());
    println!("empi={:?}", "a\u{306}b".match_indices("").collect::<Vec<_>>());
    println!("emp0={:?}", "".matches("").collect::<Vec<_>>());
    // Split family over &str patterns.
    println!("sp={:?}", "one::two::three".split("::").collect::<Vec<_>>());
    println!("rsp={:?}", "one::two::three".rsplit("::").collect::<Vec<_>>());
    println!("spt={:?}", "a--b--c--".split_terminator("--").collect::<Vec<_>>());
    println!("rspt={:?}", "a--b--c--".rsplit_terminator("--").collect::<Vec<_>>());
    println!("spi={:?}", "a--b--c--".split_inclusive("--").collect::<Vec<_>>());
    println!("spb={:?}", "aXbXcXd".split("X").collect::<Vec<_>>());
    println!("spe={:?}", "ab".split("").collect::<Vec<_>>());
    println!("sped={:?}", "xxmidxx".split("xx").collect::<Vec<_>>());
    println!("spn={:?}", "ab".split("abcdef").collect::<Vec<_>>());
    println!("spp={:?}", "mississippi".split("ss").collect::<Vec<_>>());
    // The converted Searcher bodies over the built state.
    let s = "hello world, hello two-way";
    println!("c={} f={:?} rf={:?} sw={} ew={}",
        s.contains("two-way"), s.find("world"), s.rfind("hello"),
        s.starts_with("hello"), s.ends_with("two"));
    println!("c1={} f1={:?}", s.contains("o"), s.find("o"));
}
"#;

#[test]
fn run_mode_str_escape_iterators() {
    // Direct iteration and the modelled consumers over
    // the `str` escape iterators (`EscapeDebug` /
    // `EscapeDefault` / `EscapeUnicode`): core defines
    // them as one-field wrappers over a `FlatMap` of the
    // string's `Chars` through the matching zero-sized
    // `char` escape callable, and the step delegates to
    // that composition — a `for` loop, `count`, `nth`,
    // `Map` / `Take` composition, `for_each`, a fused
    // exhaustion and a mid-iteration `Clone`. Expected
    // output was produced by the native 1.100.0 toolchain.
    run_ok(
        "str-escape-iterators",
        STR_ESCAPE_ITER_SRC,
        "count-loop=4\n\
         collected=x\\u{1b}y\n\
         count=10\n\
         nth=Some('\\\\')\n\
         upper=A\\NB\n\
         taken=\\u{68}\\\n\
         fused=Some('x') None None\n\
         clone=q\\tz orig=q\\tz\n\
         for-each=17\n",
    );
}

const STR_ESCAPE_DISPLAY_SRC: &str = r#"
fn main() {
    println!("dbg={}", "a\nb\t\\é".escape_debug());
    println!("dft={}", "a\nb\t\\é".escape_default());
    println!("uni={}", "a\nb\t\\é".escape_unicode());
    println!("empty=[{}]", "".escape_debug());
    let mut it = "a\nb".escape_debug();
    let first = it.next().unwrap();
    println!("rest={first} {}", it);
    println!("width=[{:>8}]", "ab".escape_default());
}
"#;

#[test]
fn run_mode_str_escape_display() {
    // `Display` of the `str` escape iterators: core
    // defines the impls as `clone().try_for_each(write_char)`,
    // and the formatter follows that definition over the
    // modelled step — including a partially consumed
    // iterator, whose `Display` yields only the remaining
    // chars. The impls write char by char and never pad
    // or truncate, so the `width` line's spec leaves the
    // text untouched. Expected output was produced by
    // the native 1.100.0 toolchain.
    run_ok(
        "str-escape-display",
        STR_ESCAPE_DISPLAY_SRC,
        "dbg=a\\nb\\t\\\\\u{e9}\n\
         dft=a\\nb\\t\\\\\\u{e9}\n\
         uni=\\u{61}\\u{a}\\u{62}\\u{9}\\u{5c}\\u{e9}\n\
         empty=[]\n\
         rest=a \\nb\n\
         width=[ab]\n",
    );
}

const STR_ESCAPE_SIZE_HINT_SRC: &str = r#"
fn main() {
    // Fresh iterators (no char pulled yet).
    println!("fresh-debug={:?}", "ab".escape_debug().size_hint());
    println!("fresh-default={:?}", "ab".escape_default().size_hint());
    println!("fresh-unicode={:?}", "ab".escape_unicode().size_hint());
    println!("fresh-empty={:?}", "".escape_debug().size_hint());
    println!("fresh-nl={:?}", "\n".escape_debug().size_hint());
    println!("fresh-wide={:?}", "\u{10ffff}x".escape_unicode().size_hint());

    // After one next(): the first char's escape is open.
    let mut it = "\n".escape_debug();
    println!("nl-first={:?}", it.next());
    println!("nl-after1={:?}", it.size_hint());
    println!("nl-second={:?}", it.next());
    println!("nl-done={:?}", it.size_hint());
    println!("nl-third={:?}", it.next());
    println!("nl-after-end={:?}", it.size_hint());

    // Plain char first (escape length 1), more chars behind.
    let mut it2 = "ab".escape_debug();
    println!("ab-first={:?}", it2.next());
    println!("ab-after1={:?}", it2.size_hint());
    println!("ab-second={:?}", it2.next());
    println!("ab-done={:?}", it2.size_hint());

    // Multi-char escape in the middle: front open, chars remain.
    let mut it3 = "\u{1}z".escape_default();
    println!("ctl-first={:?}", it3.next());
    println!("ctl-mid={:?}", it3.size_hint());
    println!("ctl-rest={}", it3.clone().collect::<String>());
    println!("ctl-mid2={:?}", it3.size_hint());

    // Escape unicode of a max char: 10 output chars.
    let mut it4 = "\u{10ffff}".escape_unicode();
    println!("max-first={:?}", it4.next());
    println!("max-mid={:?}", it4.size_hint());
    let mut n = 0;
    while it4.next().is_some() {
        n += 1;
    }
    println!("max-rest-count={n}");
    println!("max-done={:?}", it4.size_hint());

    // Default escape of DEL (4 chars: \u{7f}).
    let mut it5 = "\u{7f}".escape_default();
    println!("del-first={:?}", it5.next());
    println!("del-mid={:?}", it5.size_hint());

    // Debug escape of a quote (2 chars) with a trailing char.
    let mut it6 = "\"a".escape_debug();
    println!("dq-first={:?}", it6.next());
    println!("dq-mid={:?}", it6.size_hint());
}
"#;

#[test]
fn run_mode_str_escape_size_hint() {
    // `size_hint` of the `str` escape iterators:
    // core's impls forward to the wrapped `FlatMap`,
    // whose `FlattenCompat::size_hint` the builtin
    // replays on the modelled state -- the open inner
    // escape's exact remaining count is the lower
    // bound while chars remain, and the hint turns
    // exact once the `Chars` is spent. Expected
    // output was produced by the native 1.100.0
    // toolchain.
    run_ok(
        "str-escape-size-hint",
        STR_ESCAPE_SIZE_HINT_SRC,
        "fresh-debug=(0, None)\n\
         fresh-default=(0, None)\n\
         fresh-unicode=(0, None)\n\
         fresh-empty=(0, Some(0))\n\
         fresh-nl=(0, None)\n\
         fresh-wide=(0, None)\n\
         nl-first=Some('\\\\')\n\
         nl-after1=(1, Some(1))\n\
         nl-second=Some('n')\n\
         nl-done=(0, Some(0))\n\
         nl-third=None\n\
         nl-after-end=(0, Some(0))\n\
         ab-first=Some('a')\n\
         ab-after1=(0, None)\n\
         ab-second=Some('b')\n\
         ab-done=(0, Some(0))\n\
         ctl-first=Some('\\\\')\n\
         ctl-mid=(4, None)\n\
         ctl-rest=u{1}z\n\
         ctl-mid2=(4, None)\n\
         max-first=Some('\\\\')\n\
         max-mid=(9, Some(9))\n\
         max-rest-count=9\n\
         max-done=(0, Some(0))\n\
         del-first=Some('\\\\')\n\
         del-mid=(5, Some(5))\n\
         dq-first=Some('\\\\')\n\
         dq-mid=(1, None)\n",
    );
}

#[test]
fn run_mode_str_searcher_pattern_family() {
    // The `Matches` / `MatchIndices` and split families over
    // an `&str` pattern: the pattern's `StrSearcher`
    // constructor has no MIR body in core's rlib (it is not
    // `#[inline]`), so the `StrSearcherNew` builtin builds
    // the state -- all three payloads (the two-way searcher
    // for a multi-byte needle, the byte searcher for a
    // one-byte needle, the empty searcher for an empty
    // needle) -- and the modelled steps plus the converted
    // `Searcher` bodies (`contains` / `find` / `rfind` /
    // `starts_with` / `ends_with`) read it. Expected output
    // was produced by the native 1.100.0 toolchain.
    run_ok(
        "str-searcher-pattern-family",
        STR_SEARCHER_SRC,
        "tw=[\"the\", \"the\"]\n\
         twi=[(0, \"the\"), (10, \"the\")]\n\
         per=[\"abcab\"]\n\
         ovl=[\"aa\", \"aa\"]\n\
         byte=[\"X\", \"X\"]\n\
         bytei=[(1, \"X\"), (3, \"X\")]\n\
         emp=[\"\", \"\", \"\"]\n\
         empi=[(0, \"\"), (1, \"\"), (3, \"\"), (4, \"\")]\n\
         emp0=[\"\"]\n\
         sp=[\"one\", \"two\", \"three\"]\n\
         rsp=[\"three\", \"two\", \"one\"]\n\
         spt=[\"a\", \"b\", \"c\"]\n\
         rspt=[\"c\", \"b\", \"a\"]\n\
         spi=[\"a--\", \"b--\", \"c--\"]\n\
         spb=[\"a\", \"b\", \"c\", \"d\"]\n\
         spe=[\"\", \"a\", \"b\", \"\"]\n\
         sped=[\"\", \"mid\", \"\"]\n\
         spn=[\"ab\"]\n\
         spp=[\"mi\", \"i\", \"ippi\"]\n\
         c=true f=Some(6) rf=Some(13) sw=true ew=false\n\
         c1=true f1=Some(4)\n"
    );
}

const STR_SPLITN_SRC: &str = r#"fn main() {
    // char pattern: truncation, remainder-as-last-piece.
    println!("a={:?}", "a:b:c:d".splitn(2, ':').collect::<Vec<_>>());
    println!("b={:?}", "a:b:c:d".rsplitn(2, ':').collect::<Vec<_>>());
    println!("c={:?}", "a:b:c:d".splitn(0, ':').collect::<Vec<_>>());
    println!("d={:?}", "a:b:c".splitn(1, ':').collect::<Vec<_>>());
    println!("e={:?}", "a:b:c".rsplitn(1, ':').collect::<Vec<_>>());
    println!("f={:?}", "a:b:c".splitn(5, ':').collect::<Vec<_>>());
    // Empty haystack, no delimiter, trailing delimiter.
    println!("g={:?}", "".splitn(2, ':').collect::<Vec<_>>());
    println!("h={:?}", "".rsplitn(3, ':').collect::<Vec<_>>());
    println!("i={:?}", "abc".splitn(2, ':').collect::<Vec<_>>());
    println!("j={:?}", "a:b:".splitn(3, ':').collect::<Vec<_>>());
    println!("k={:?}", "a:b:".rsplitn(3, ':').collect::<Vec<_>>());
    println!("l={:?}", "::".splitn(2, ':').collect::<Vec<_>>());
    println!("m={:?}", "::".rsplitn(2, ':').collect::<Vec<_>>());
    // Manual next: the count spends down to the get_end piece.
    let mut it = "x:y:z".splitn(2, ':');
    println!("n={:?} {:?} {:?}", it.next(), it.next(), it.next());
    let mut it2 = "x:y:z".rsplitn(2, ':');
    println!("o={:?} {:?} {:?}", it2.next(), it2.next(), it2.next());
    // &str pattern: multi-byte, one-byte, and empty needles.
    println!("p={:?}", "a::b::c".splitn(2, "::").collect::<Vec<_>>());
    println!("q={:?}", "a::b::c".rsplitn(2, "::").collect::<Vec<_>>());
    println!("r={:?}", "aXbXc".splitn(2, "X").collect::<Vec<_>>());
    println!("s={:?}", "abc".splitn(2, "").collect::<Vec<_>>());
    println!("t={:?}", "abc".rsplitn(2, "").collect::<Vec<_>>());
    // Predicate pattern.
    println!("u={:?}", "a1b2c".splitn(2, |c: char| c.is_ascii_digit()).collect::<Vec<_>>());
    println!("v={:?}", "a1b2c".rsplitn(2, |c: char| c.is_ascii_digit()).collect::<Vec<_>>());
    // Consumers and a mid-iteration clone.
    println!("w={:?} {:?}", "a:b:c:d".splitn(4, ':').count(), "a:b:c:d".rsplitn(4, ':').nth(1));
    let mut it3 = "p:q:r".splitn(3, ':');
    let _ = it3.next();
    let cl = it3.clone();
    println!("x={:?} {:?}", it3.collect::<Vec<_>>(), cl.collect::<Vec<_>>());
}
"#;

#[test]
fn run_mode_str_splitn_family() {
    // The `SplitN` / `RSplitN` step: core's `SplitNInternal`
    // count logic over the wrapped split state -- a count of
    // zero yields nothing, a count of one yields the wrapped
    // iterator's `get_end` (the whole remainder), and larger
    // counts spend one wrapped step each, front for `SplitN`
    // and back for `RSplitN`, whose `Iterator::next` is the
    // internal `next_back`. The wrapped state is the split
    // family's own, so the char, `&str` (all three searcher
    // payloads), and predicate patterns all step. Expected
    // output was produced by the native 1.100.0 toolchain.
    run_ok(
        "str-splitn-family",
        STR_SPLITN_SRC,
        "a=[\"a\", \"b:c:d\"]\n\
         b=[\"d\", \"a:b:c\"]\n\
         c=[]\n\
         d=[\"a:b:c\"]\n\
         e=[\"a:b:c\"]\n\
         f=[\"a\", \"b\", \"c\"]\n\
         g=[\"\"]\n\
         h=[\"\"]\n\
         i=[\"abc\"]\n\
         j=[\"a\", \"b\", \"\"]\n\
         k=[\"\", \"b\", \"a\"]\n\
         l=[\"\", \":\"]\n\
         m=[\"\", \":\"]\n\
         n=Some(\"x\") Some(\"y:z\") None\n\
         o=Some(\"z\") Some(\"x:y\") None\n\
         p=[\"a\", \"b::c\"]\n\
         q=[\"c\", \"a::b\"]\n\
         r=[\"a\", \"bXc\"]\n\
         s=[\"\", \"abc\"]\n\
         t=[\"\", \"abc\"]\n\
         u=[\"a\", \"b2c\"]\n\
         v=[\"c\", \"a1b\"]\n\
         w=4 Some(\"c\")\n\
         x=[\"q\", \"r\"] [\"q\", \"r\"]\n",
    );
}

const STRING_FROM_MUT_STR_SRC: &str = r#"
fn via_ref(s: &mut str) -> String { s.into() }
fn main() {
    let mut buf = [0u8; 4];
    let s: String = '\u{1F603}'.encode_utf8(&mut buf).into();
    println!("{} {}", s, s.len());
    let mut owned = String::from("xy");
    let t = via_ref(&mut owned);
    println!("{t}");
}
"#;

#[test]
fn run_mode_string_from_mut_str() {
    // `String: From<&mut str>` copies the text; a `str`
    // place over byte storage (the `encode_utf8` window
    // over a byte array) does not resolve to the text
    // value, so the conversion reads its bytes as a byte
    // slice.
    let dir = temp_dir("string-from-mut-str");
    let src = fixture(&dir, "stringfrommut.rs", STRING_FROM_MUT_STR_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "\u{1F603} 4\nxy\n", "stdout={stdout}");
}

const DERIVED_DEBUG_CONST_PARAMS_SRC: &str = r#"
#[derive(Debug)]
struct S1<A> { a: A }
#[derive(Debug)]
struct S3<'a, const C: usize, T, U, const B: bool> {
    t: T,
    pd: std::marker::PhantomData<&'a [(usize, U); C]>,
}
fn main() {
    let x: S3<'static, 123, S1<usize>, (), true> = S3 {
        t: S1 { a: 3usize },
        pd: std::marker::PhantomData::<&[(usize, ()); 123]>,
    };
    println!("{x:?}");
}
"#;

#[test]
fn run_mode_debug_format_derived_const_params() {
    // A derived `Debug` over an ADT whose generic parameters
    // mix kinds: a parameter index counts lifetimes and
    // consts too, so the recorded field tags must substitute
    // the instance's type arguments at their own positions,
    // and an array length that is a const parameter must
    // resolve to the instance's value. Either loss leaves
    // the `PhantomData` field's tag naming no recorded
    // shape, and formatting stops.
    let dir = temp_dir("derived-debug-const-params");
    let src = fixture(&dir, "deriveddebug.rs", DERIVED_DEBUG_CONST_PARAMS_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "S3 { t: S1 { a: 3 }, pd: PhantomData<&[(usize, ()); 123]> }\n",
        "stdout={stdout}"
    );
}

const STR_SLICE_END_INSIDE_CHAR_SRC: &str = r#"
fn main() {
    let s = "héllo";
    println!("{}", &s[1..2]);
}
"#;

#[test]
fn run_mode_str_slice_end_inside_char_panics_with_core_message() {
    let dir = temp_dir("str-slice-boundary");
    let src = fixture(&dir, "strslice.rs", STR_SLICE_END_INSIDE_CHAR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stderr={stderr}");
    assert!(
        stderr.contains(
            "end byte index 2 is not a char boundary; it is inside 'é' (bytes 1..3) of string"
        ),
        "stderr={stderr}"
    );
}

const STR_SLICE_END_OUT_OF_BOUNDS_SRC: &str = r#"
fn main() {
    let s = "hello";
    println!("{}", &s[3..10]);
}
"#;

#[test]
fn run_mode_str_slice_end_out_of_bounds_panics_with_core_message() {
    let dir = temp_dir("str-slice-oob");
    let src = fixture(&dir, "strslice_oob.rs", STR_SLICE_END_OUT_OF_BOUNDS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stderr={stderr}");
    assert!(
        stderr.contains("end byte index 10 is out of bounds for string of length 5"),
        "stderr={stderr}"
    );
}

const STR_SLICE_UNICODE_SRC: &str = r#"
fn main() {
    let s = "héllo";
    println!("{}", &s[0..1]);
    println!("{}", &s[1..3]);
}
"#;

#[test]
fn run_mode_str_slice_on_char_boundaries() {
    run_ok("str-slice-unicode", STR_SLICE_UNICODE_SRC, "h\né\n");
}


const FROM_UTF8_UNCHECKED_SRC: &str = r#"
fn main() {
    let data: &[u8] = b"ab";
    let s = unsafe { std::str::from_utf8_unchecked(data) };
    println!("{:?}", s);
    println!("{}", s.len());
    let mut it = data.windows(1);
    let piece = it.next().unwrap();
    let p = unsafe { std::str::from_utf8_unchecked(piece) };
    println!("{:?} {}", p, p.len());
    let v: Vec<u8> = vec![0xE3, 0x81, 0x82];
    let m = unsafe { std::str::from_utf8_unchecked(&v) };
    println!("{:?} {} {}", m, m.len(), m.chars().count());
    for piece in "a b".split_ascii_whitespace() {
        println!("{:?} {}", piece, piece.len());
    }
}
"#;

#[test]
fn run_mode_from_utf8_unchecked_denominates_str() {
    // `from_utf8_unchecked` is a transmute from a byte
    // slice to a `str`: the result must be denominated as
    // the string it denotes, so `len`, `chars` and the
    // Debug rendering read it as text -- over a plain
    // slice, a `windows` piece, a `Vec` buffer and the
    // `split_ascii_whitespace` pieces alike. (The checked
    // `from_utf8` always produced the string value; the
    // unchecked form kept the byte form and the string
    // consumers refused it.)
    let dir = temp_dir("from-utf8-unchecked");
    let src = fixture(&dir, "utf8unchecked.rs", FROM_UTF8_UNCHECKED_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "\"ab\"\n2\n\"a\" 1\n\"あ\" 3 1\n\"a\" 1\n\"b\" 1\n",
        "stdout={stdout}"
    );
}

const STR_BYTE_SLICE_CAST_SRC: &str = r#"
fn main() {
    let data: &[u8] = b"ab";
    let s: &str = unsafe { std::mem::transmute(data) };
    println!("{:?} {}", s, s.len());
    let b: &[u8] = unsafe { std::mem::transmute(s) };
    println!("{} {} {}", b.len(), b[0], b[1]);
    let s2: &str = unsafe { std::mem::transmute(b) };
    println!("{:?} {}", s2, s2.len());
}
"#;

#[test]
fn run_mode_str_byte_slice_casts_denominate_both_ways() {
    // The transmute crossing denominates in both
    // directions: a byte slice transmuted to a `str`
    // reads as the string, and the round trip back
    // through a transmute reads the string value as the
    // byte sequence it denotes. (The raw-pointer cast
    // form -- `as *const str` over a byte slice -- is
    // deliberately not denominated at the cast: a fat
    // raw pointer's length metadata is reinterpreted by
    // later casts, as cast-rfc0401's `*mut str` ->
    // `*const [u16]` crossing exercises, so the pointer
    // keeps its identity there.)
    let dir = temp_dir("str-byte-slice-casts");
    let src = fixture(&dir, "strbyteslicecasts.rs", STR_BYTE_SLICE_CAST_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "\"ab\" 2\n2 97 98\n\"ab\" 2\n",
        "stdout={stdout}"
    );
}

const OS_STRING_PUSH_ENCODE_UTF8_SRC: &str = r#"
use std::ffi::OsString;

fn main() {
    let mut char_buf = [0u8; 4];
    let mut one = OsString::new();
    one.push('x'.encode_utf8(&mut char_buf));
    println!("{:?}", one.into_encoded_bytes());

    let mut two = OsString::new();
    two.push('é'.encode_utf8(&mut char_buf));
    println!("{:?}", two.into_encoded_bytes());
}
"#;

#[test]
fn run_mode_os_string_push_encode_utf8_window() {
    // `OsString::push` views its argument as an `&OsStr`; for
    // a `char::encode_utf8` result the receiver of that view
    // is a fat pointer whose base denotes one element of the
    // char buffer and whose length is the encoded length.
    // The view denominates the window's bytes (one byte for
    // an ASCII char, two for `é`), not the buffer's capacity
    // and not the single denoted element.
    run_ok(
        "os-string-push-encode-utf8",
        OS_STRING_PUSH_ENCODE_UTF8_SRC,
        "[120]\n[195, 169]\n",
    );
}

const OS_STRING_TMPNAME_WINDOW_SRC: &str = r#"
use std::ffi::OsString;

fn main() {
    // The shape tempfile's `tmpname` builds: a
    // capacity-reserved `OsString`, a plain prefix, then one
    // `encode_utf8` push per name char.
    let mut name = OsString::with_capacity(16);
    name.push("tmp");
    for c in ['a', 'é', 'z', 'ß'] {
        let mut char_buf = [0u8; 4];
        name.push(c.encode_utf8(&mut char_buf));
    }
    assert_eq!(name.into_encoded_bytes(), "tmpaézß".as_bytes());
    println!("ok");
}
"#;

#[test]
fn run_mode_os_string_tmpname_encode_utf8_loop() {
    run_ok(
        "os-string-tmpname-window",
        OS_STRING_TMPNAME_WINDOW_SRC,
        "ok\n",
    );
}

const STR_WINDOW_VIEWS_SRC: &str = r#"
fn main() {
    let mut buf = [0u8; 4];
    let w = 'é'.encode_utf8(&mut buf);
    let b: &[u8] = w.as_ref();
    println!("{:?}", b);
    let w2: &str = w.as_ref();
    println!("{:?}", w2);
}
"#;

#[test]
fn run_mode_str_window_as_ref_views() {
    // The same window viewed as a byte slice and as a `str`
    // itself: both denominate the fat pointer's length.
    run_ok(
        "str-window-as-ref-views",
        STR_WINDOW_VIEWS_SRC,
        "[195, 169]\n\"é\"\n",
    );
}

const STR_WINDOW_TOO_LONG_SRC: &str = r#"
use std::ffi::OsString;

fn main() {
    let buf = [0u8; 4];
    // A view claiming eight bytes over a four-byte buffer:
    // the model must refuse it, not read past the storage.
    let s = unsafe {
        std::str::from_utf8_unchecked(std::slice::from_raw_parts(buf.as_ptr(), 8))
    };
    let mut name = OsString::new();
    name.push(s);
    println!("{name:?}");
}
"#;

#[test]
fn run_mode_str_window_longer_than_buffer_refused() {
    let dir = temp_dir("str-window-too-long");
    let src = fixture(&dir, "strwindowtoolong.rs", STR_WINDOW_TOO_LONG_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an over-long window view is refused; stderr={stderr}"
    );
    assert!(
        stderr.contains("expected string data"),
        "the refusal is the string view's; stderr={stderr}"
    );
}

const OS_STRING_REAL_AGGREGATE_DEREF_SRC: &str = r#"
use std::ffi::{OsStr, OsString};

fn main() {
    // An `OsString` built by `with_capacity` plus `push` is
    // stored in its real aggregate form (its bytes behind the
    // buffer's raw pointer); dereferencing it to `&OsStr`
    // reads the bytes through that buffer.
    let mut s = OsString::with_capacity(16);
    s.push("hello");
    let r: &OsStr = &s;
    println!("len={}", r.len());
    println!("empty={}", s.is_empty());
    let empty = OsString::with_capacity(4);
    println!("len0={}", empty.len());
}
"#;

#[test]
fn run_mode_os_string_real_aggregate_deref() {
    run_ok(
        "os-string-real-aggregate-deref",
        OS_STRING_REAL_AGGREGATE_DEREF_SRC,
        "len=5\nempty=false\nlen0=0\n",
    );
}

const OS_STRING_REAL_AGGREGATE_EQ_SRC: &str = r#"
use std::ffi::OsString;

fn main() {
    // The real aggregate form and the element form of the
    // same bytes are the same `OsString`: equality reads the
    // bytes each side's buffer holds, not the storage
    // arrangement (a clone owns a fresh allocation).
    let mut a = OsString::with_capacity(16);
    a.push("same");
    let b = OsString::from("same");
    println!("eq={}", a == b);
    println!("eq_self={}", a == a.clone());
    let c = OsString::from("diff");
    println!("ne={}", a != c);
    let empty = OsString::with_capacity(4);
    println!("eq_empty={}", empty == OsString::new());
}
"#;

#[test]
fn run_mode_os_string_real_aggregate_equality() {
    run_ok(
        "os-string-real-aggregate-equality",
        OS_STRING_REAL_AGGREGATE_EQ_SRC,
        "eq=true\neq_self=true\nne=true\neq_empty=true\n",
    );
}

const OS_STRING_REAL_AGGREGATE_FMT_SRC: &str = r#"
use std::ffi::OsString;
use std::path::{Path, PathBuf};

fn main() {
    // `Debug` of the real aggregate form is the quoted
    // content, and the value hands its bytes to the
    // `PathBuf` conversion and the lossy conversion.
    let mut a = OsString::with_capacity(16);
    a.push("ab");
    println!("{:?}", a);
    println!("{}", a.to_string_lossy());
    let p = PathBuf::from(a);
    println!("{:?}", p);
    println!("{}", p.display());
    let r: &Path = &p;
    println!("{:?}", r);
    println!("eq={}", p == PathBuf::from("ab"));
}
"#;

#[test]
fn run_mode_os_string_real_aggregate_fmt_and_pathbuf() {
    run_ok(
        "os-string-real-aggregate-fmt",
        OS_STRING_REAL_AGGREGATE_FMT_SRC,
        "\"ab\"\nab\n\"ab\"\nab\n\"ab\"\neq=true\n",
    );
}

const OS_STRING_FROM_PATHBUF_SRC: &str = r#"
use std::ffi::OsString;
use std::path::PathBuf;

fn main() {
    // `OsString: From<PathBuf>` hands the path's bytes to
    // the OS string, whichever shape the path value was
    // built in: pushed, joined, converted from a `String`,
    // wrapped around a real aggregate `OsString`, or read
    // back out of a vector of paths.
    let mut p = PathBuf::from("/tmp");
    p.push("alpha");
    p.push("beta.txt");
    let o = OsString::from(p);
    println!("{:?} len={} eq={}", o, o.len(), o == OsString::from("/tmp/alpha/beta.txt"));
    let j = PathBuf::from("/base").join("x").join("y");
    let oj: OsString = j.into();
    println!("{:?} len={}", oj, oj.len());
    let s = PathBuf::from(String::from("hello/world"));
    let os = OsString::from(s);
    println!("{:?} len={}", os, os.len());
    let mut a = OsString::with_capacity(16);
    a.push("real/path");
    let oa = OsString::from(PathBuf::from(a));
    println!("{:?} len={}", oa, oa.len());
    let v: Vec<PathBuf> = vec![PathBuf::from("one"), PathBuf::from("two/three")];
    for p in v {
        let o = OsString::from(p);
        println!("{:?} len={}", o, o.len());
    }
}
"#;

#[test]
fn run_mode_os_string_from_pathbuf() {
    run_ok(
        "os-string-from-pathbuf",
        OS_STRING_FROM_PATHBUF_SRC,
        "\"/tmp/alpha/beta.txt\" len=19 eq=true\n\"/base/x/y\" len=9\n\"hello/world\" len=11\n\"real/path\" len=9\n\"one\" len=3\n\"two/three\" len=9\n",
    );
}

const OS_STRING_FROM_REF_PATH_STRING_SRC: &str = r#"
use std::ffi::OsString;
use std::path::{Path, PathBuf};

fn main() {
    // The borrowed conversions clone: `From<&Path>` (a
    // path literal and a buffer's `as_path`) and
    // `From<&String>` (the source string is untouched).
    let o = OsString::from(Path::new("/a/b"));
    println!("{:?} len={}", o, o.len());
    let pb = PathBuf::from("/c/d");
    let o2 = OsString::from(pb.as_path());
    println!("{:?} len={} eq={}", o2, o2.len(), o2 == OsString::from("/c/d"));
    let s = String::from("refstr");
    let o3 = OsString::from(&s);
    println!("{:?} len={} eq={}", o3, o3.len(), o3 == OsString::from("refstr"));
    println!("src={}", s);
}
"#;

#[test]
fn run_mode_os_string_from_ref_path_and_string() {
    run_ok(
        "os-string-from-ref-path-string",
        OS_STRING_FROM_REF_PATH_STRING_SRC,
        "\"/a/b\" len=4\n\"/c/d\" len=4 eq=true\n\"refstr\" len=6 eq=true\nsrc=refstr\n",
    );
}

const OS_STRING_FROM_COW_SRC: &str = r#"
use std::borrow::Cow;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

fn main() {
    // `From<Cow<'_, OsStr>>` for `OsString` and
    // `From<Cow<'_, Path>>` for `PathBuf` are `into_owned`:
    // the borrowed variant clones the bytes, the owned
    // variant hands its buffer over.
    let borrowed: Cow<'_, OsStr> = Cow::Borrowed(OsStr::new("cowb"));
    let o = OsString::from(borrowed);
    println!("{:?} len={}", o, o.len());
    let owned: Cow<'_, OsStr> = Cow::Owned(OsString::from("cowo"));
    let o2 = OsString::from(owned);
    println!("{:?} len={}", o2, o2.len());
    let bp: Cow<'_, Path> = Cow::Borrowed(Path::new("cpb"));
    println!("{:?}", PathBuf::from(bp));
    let op: Cow<'_, Path> = Cow::Owned(PathBuf::from("cpo"));
    println!("{:?}", PathBuf::from(op));
}
"#;

#[test]
fn run_mode_os_string_from_cow() {
    run_ok(
        "os-string-from-cow",
        OS_STRING_FROM_COW_SRC,
        "\"cowb\" len=4\n\"cowo\" len=4\n\"cpb\"\n\"cpo\"\n",
    );
}

const OS_STRING_INTO_BOXED_OS_STR_SRC: &str = r#"
use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::{OsStrExt, OsStringExt};

fn check(tag: &str, os: OsString) {
    // `into_boxed_os_str` hands the bytes to a fresh box;
    // every `OsStr` consumer reads the same bytes back,
    // whichever form the `OsString` was built in.
    let b = os.into_boxed_os_str();
    println!("{} len={} empty={} bytes={:?} lossy={} eq={} dbg={:?}",
        tag, b.len(), b.is_empty(), b.as_bytes(), b.to_string_lossy(),
        &*b == OsStr::new("hello"), b);
}

fn main() {
    check("from", OsString::from("hello"));
    let mut os = OsString::with_capacity(16);
    os.push("hel");
    os.push("lo");
    check("push", os);
    check("vec", OsString::from_vec(b"hello".to_vec()));
}
"#;

#[test]
fn run_mode_os_string_into_boxed_os_str() {
    run_ok(
        "os-string-into-boxed-os-str",
        OS_STRING_INTO_BOXED_OS_STR_SRC,
        "from len=5 empty=false bytes=[104, 101, 108, 108, 111] lossy=hello eq=true dbg=\"hello\"\npush len=5 empty=false bytes=[104, 101, 108, 108, 111] lossy=hello eq=true dbg=\"hello\"\nvec len=5 empty=false bytes=[104, 101, 108, 108, 111] lossy=hello eq=true dbg=\"hello\"\n",
    );
}

const OS_STRING_INTO_BOXED_OS_STR_EDGE_SRC: &str = r#"
use std::ffi::OsString;
use std::os::unix::ffi::{OsStrExt, OsStringExt};

fn main() {
    // The empty string boxes to an empty `OsStr`, non-UTF-8
    // bytes survive the round trip unaltered, and the box
    // converts back into the `OsString` it came from.
    let empty = OsString::new().into_boxed_os_str();
    println!("empty len={} empty={}", empty.len(), empty.is_empty());
    let raw = OsString::from_vec(vec![0xff, b'a', 0x80]).into_boxed_os_str();
    println!("nonutf8 bytes={:?}", raw.as_bytes());
    let back: OsString = OsString::from(OsString::from("hello").into_boxed_os_str());
    println!("back={:?}", back);
}
"#;

#[test]
fn run_mode_os_string_into_boxed_os_str_edges() {
    run_ok(
        "os-string-into-boxed-os-str-edges",
        OS_STRING_INTO_BOXED_OS_STR_EDGE_SRC,
        "empty len=0 empty=true\nnonutf8 bytes=[255, 97, 128]\nback=\"hello\"\n",
    );
}

const BOXED_OS_STR_VEC_SORT_SRC: &str = r#"
use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStrExt;

fn show(tag: &str, v: &[Box<OsStr>]) {
    for (i, e) in v.iter().enumerate() {
        println!("{} {} len={} bytes={:?} lossy={}", tag, i, e.len(), e.as_bytes(), e.to_string_lossy());
    }
}

fn main() {
    let mut v: Vec<Box<OsStr>> = vec![
        OsString::from("pear").into_boxed_os_str(),
        OsString::from("banana").into_boxed_os_str(),
        OsString::from("kiwi").into_boxed_os_str(),
        OsString::from("apple").into_boxed_os_str(),
    ];
    v.sort();
    show("sorted", &v);
    println!("ord_ok={}", v.windows(2).all(|w| w[0] <= w[1]));
    let needle: Box<OsStr> = OsString::from("banana").into_boxed_os_str();
    println!("eq0={} eq1={}", v[0] == needle, v[1] == needle);
    let mut u: Vec<Box<OsStr>> = vec![
        OsString::from("pear").into_boxed_os_str(),
        OsString::from("banana").into_boxed_os_str(),
        OsString::from("kiwi").into_boxed_os_str(),
    ];
    u.sort_unstable();
    show("unstable", &u);
    u.sort_by(|x, y| x.cmp(y));
    show("by_box", &u);
}
"#;

#[test]
fn run_mode_boxed_os_str_vec_sort_orders() {
    // Sorting a `Vec<Box<OsStr>>` compares through the
    // generic sort machinery, which hands the comparison
    // one side as the box aggregate a slot holds and the
    // other as the bare pointee snapshot it extracted: the
    // ordering is the pointees' byte order either way.
    // Comparisons after the sort meet the same pair of
    // forms in the other order (an element the rotation
    // moved keeps the snapshot form in its slot), for
    // equality as for ordering. The sorted order is read
    // back through the byte readers, which accept every
    // form.
    run_ok(
        "boxed-os-str-vec-sort",
        BOXED_OS_STR_VEC_SORT_SRC,
        "sorted 0 len=5 bytes=[97, 112, 112, 108, 101] lossy=apple\nsorted 1 len=6 bytes=[98, 97, 110, 97, 110, 97] lossy=banana\nsorted 2 len=4 bytes=[107, 105, 119, 105] lossy=kiwi\nsorted 3 len=4 bytes=[112, 101, 97, 114] lossy=pear\nord_ok=true\neq0=false eq1=true\nunstable 0 len=6 bytes=[98, 97, 110, 97, 110, 97] lossy=banana\nunstable 1 len=4 bytes=[107, 105, 119, 105] lossy=kiwi\nunstable 2 len=4 bytes=[112, 101, 97, 114] lossy=pear\nby_box 0 len=6 bytes=[98, 97, 110, 97, 110, 97] lossy=banana\nby_box 1 len=4 bytes=[107, 105, 119, 105] lossy=kiwi\nby_box 2 len=4 bytes=[112, 101, 97, 114] lossy=pear\n",
    );
}

const BOXED_SCALAR_VEC_SORT_SRC: &str = r#"
fn main() {
    let mut v: Vec<Box<u32>> = vec![Box::new(30), Box::new(10), Box::new(20)];
    v.sort();
    println!("sorted={:?}", v);
    let a = Box::new(7u32);
    let b = Box::new(9u32);
    println!("cmp={:?} eq={} lt={}", a.cmp(&b), a == b, a < b);
}
"#;

#[test]
fn run_mode_boxed_scalar_vec_sort() {
    // The scalar-pointee form of the same presentation: a
    // sort over `Vec<Box<u32>>` hands the comparison the
    // stored box against the bare integer snapshot.
    run_ok(
        "boxed-scalar-vec-sort",
        BOXED_SCALAR_VEC_SORT_SRC,
        "sorted=[10, 20, 30]\ncmp=Less eq=false lt=true\n",
    );
}

const BOXED_SLICE_CMP_SORT_SRC: &str = r#"
fn main() {
    let a: Box<[u8]> = vec![2u8, 2].into_boxed_slice();
    let b: Box<[u8]> = vec![2u8, 3].into_boxed_slice();
    let c: Box<[u8]> = vec![2u8, 2].into_boxed_slice();
    println!("cmp={:?} eq={} lt={}", a.cmp(&b), a == c, a < b);
    let mut v: Vec<Box<[u8]>> = vec![vec![3u8, 1].into_boxed_slice(), vec![1u8, 9].into_boxed_slice(), vec![3u8, 0].into_boxed_slice()];
    v.sort();
    for (i, e) in v.iter().enumerate() {
        println!("s{} len={} elems={:?}", i, e.len(), &**e);
    }
}
"#;

#[test]
fn run_mode_boxed_slice_cmp_and_sort() {
    // A slice pointee orders element-wise, in the direct
    // (tagged) comparisons and in a sort's generic ones.
    run_ok(
        "boxed-slice-cmp-and-sort",
        BOXED_SLICE_CMP_SORT_SRC,
        "cmp=Less eq=true lt=true\ns0 len=2 elems=[1, 9]\ns1 len=2 elems=[3, 0]\ns2 len=2 elems=[3, 1]\n",
    );
}

const BOXED_OS_STR_VEC_DEBUG_SRC: &str = r#"
use std::ffi::OsString;

fn main() {
    let mut v: Vec<Box<std::ffi::OsStr>> = vec![
        OsString::from("pear").into_boxed_os_str(),
        OsString::from("banana").into_boxed_os_str(),
        OsString::from("kiwi").into_boxed_os_str(),
    ];
    println!("pre={:?}", v);
    v.sort();
    println!("sorted={:?}", v);
    println!("pretty={:#?}", v);
    let mut w: Vec<Box<std::ffi::OsStr>> = vec![
        OsString::from("pear").into_boxed_os_str(),
        OsString::from("banana").into_boxed_os_str(),
    ];
    w.swap(0, 1);
    println!("swapped={:?}", w);
}
"#;

#[test]
fn run_mode_boxed_os_str_vec_debug_after_move() {
    // A move through the sort machinery (or a `swap`) leaves
    // the extracted pointee snapshot in the element slot in
    // place of the box aggregate: formatting the vector then
    // hands the formatter the bare byte value under the
    // `Box<OsStr>` tag. The formatter treats a value without
    // the wrapper's stored shape as the pointee itself in
    // storage form and renders it under the element tag, so
    // the moved elements debug as the OS strings they are.
    run_ok(
        "boxed-os-str-vec-debug-after-move",
        BOXED_OS_STR_VEC_DEBUG_SRC,
        "pre=[\"pear\", \"banana\", \"kiwi\"]\nsorted=[\"banana\", \"kiwi\", \"pear\"]\npretty=[\n    \"banana\",\n    \"kiwi\",\n    \"pear\",\n]\nswapped=[\"banana\", \"pear\"]\n",
    );
}

const BOXED_SLICE_VEC_DEBUG_SRC: &str = r#"
fn main() {
    let mut v: Vec<Box<[u8]>> = vec![
        vec![1u8, 9].into_boxed_slice(),
        vec![0u8, 5].into_boxed_slice(),
        vec![1u8, 2].into_boxed_slice(),
    ];
    println!("pre={:?}", v);
    v.sort();
    println!("sorted={:?}", v);
}
"#;

#[test]
fn run_mode_boxed_slice_vec_debug_after_sort() {
    // The slice-pointee form of the same storage-form
    // presentation: an element the sort moved sits in its
    // slot as the bare tuple of its elements under the
    // `Box<[u8]>` tag, and debugs as the slice it denotes,
    // not as a tuple.
    run_ok(
        "boxed-slice-vec-debug-after-sort",
        BOXED_SLICE_VEC_DEBUG_SRC,
        "pre=[[1, 9], [0, 5], [1, 2]]\nsorted=[[0, 5], [1, 2], [1, 9]]\n",
    );
}

const BOXED_OS_STR_NONUTF8_DEBUG_SRC: &str = r#"
use std::ffi::OsString;
use std::os::unix::ffi::OsStringExt;

fn main() {
    let mut v: Vec<Box<std::ffi::OsStr>> = vec![
        OsString::from_vec(vec![b'z', 0xff, b'a']).into_boxed_os_str(),
        OsString::from_vec(vec![b'a', 0x80]).into_boxed_os_str(),
    ];
    v.sort();
    println!("sorted={:?}", v);
}
"#;

#[test]
fn run_mode_boxed_os_str_nonutf8_debug_after_sort() {
    // The storage-form rendering is the host `OsStr`
    // `Debug`: non-UTF-8 bytes of a moved element escape
    // exactly as an unmoved element's do.
    run_ok(
        "boxed-os-str-nonutf8-debug-after-sort",
        BOXED_OS_STR_NONUTF8_DEBUG_SRC,
        "sorted=[\"a\\x80\", \"z\\xFFa\"]\n",
    );
}

const SORT_STRING_ELEMENTS_SRC: &str = r#"
fn main() {
    // Sorting more than twenty `String` elements takes the
    // quicksort path, whose element rotation inside the
    // converted sort bodies hands the comparisons one side as
    // the one-field `String` aggregate around string data and
    // the other as the data itself: the comparison reads both
    // as the strings they denote. (Thirty-two elements stay
    // below the partition-closure demand threshold, so the
    // comparison is the first stop on this path.)
    let mut v: Vec<String> = (0..32).map(|i| format!("s{:03}", (i * 37) % 32)).collect();
    v.sort_unstable();
    println!("first={} last={}", v[0], v[31]);
    let mut w: Vec<String> = (0..32).map(|i| format!("s{:03}", (i * 37) % 32)).collect();
    w.sort();
    println!("stable first={} last={}", w[0], w[31]);
}
"#;

const OS_STRING_REAL_AGGREGATE_ORD_SRC: &str = r#"
use std::cmp::Ordering;
use std::ffi::OsString;

fn agg(s: &str) -> OsString {
    let mut o = OsString::with_capacity(32);
    o.push(s);
    o
}

fn ord_name(o: Ordering) -> &'static str {
    match o {
        Ordering::Less => "Less",
        Ordering::Equal => "Equal",
        Ordering::Greater => "Greater",
    }
}

fn main() {
    // Ordering reads the same buffer bytes equality does:
    // the real aggregate form (built by `with_capacity`
    // plus `push`) orders against the element form and
    // against a clone by content, in both operand orders.
    let a_agg = agg("abc");
    let b_agg = agg("abd");
    let a_from = OsString::from("abc");
    let b_from = OsString::from("abd");
    let a_clone = a_agg.clone();
    println!("lt agg<from: {}", a_agg < b_from);
    println!("lt from<agg: {}", a_from < b_agg);
    println!("lt agg<agg: {}", a_agg < b_agg);
    println!("gt agg>from: {}", b_agg > a_from);
    println!("cmp agg/from: {}", ord_name(a_agg.cmp(&a_from)));
    println!("cmp agg/agg: {}", ord_name(a_agg.cmp(&b_agg)));
    println!("cmp from/agg: {}", ord_name(b_from.cmp(&a_agg)));
    println!("le agg<=clone: {}", a_agg <= a_clone);
    println!("ge from>=agg: {}", b_from >= b_agg);
}
"#;

#[test]
fn run_mode_sort_string_elements_compare() {
    run_ok(
        "sort-string-elements-compare",
        SORT_STRING_ELEMENTS_SRC,
        "first=s000 last=s031\nstable first=s000 last=s031\n",
    );
}

const SORT_STRING_COMPARATOR_READS_SRC: &str = r#"
fn main() {
    // The comparator receives the elements in the same
    // wrapped form: reading one as a string (its length, its
    // equality, its order) goes through the shared string
    // readers, which denominate the wrapper as the string it
    // holds.
    let mut v: Vec<String> = (0..32).map(|i| format!("s{:03}", (i * 37) % 32)).collect();
    v.sort_unstable_by(|a, b| {
        a.len().cmp(&b.len()).then_with(|| {
            if a == b {
                std::cmp::Ordering::Equal
            } else {
                a.cmp(b)
            }
        })
    });
    println!("first={} last={}", v[0], v[31]);
}
"#;

#[test]
fn run_mode_sort_string_comparator_reads_elements() {
    run_ok(
        "sort-string-comparator-reads",
        SORT_STRING_COMPARATOR_READS_SRC,
        "first=s000 last=s031\n",
    );
}

#[test]
fn run_mode_os_string_real_aggregate_ordering() {
    run_ok(
        "os-string-real-aggregate-ordering",
        OS_STRING_REAL_AGGREGATE_ORD_SRC,
        "lt agg<from: true\nlt from<agg: true\nlt agg<agg: true\ngt agg>from: true\ncmp agg/from: Equal\ncmp agg/agg: Less\ncmp from/agg: Greater\nle agg<=clone: true\nge from>=agg: true\n",
    );
}

const OS_STRING_COPY_SHIFT_SRC: &str = r#"
use std::ffi::OsString;

fn main() {
    // A raw-pointer copy of the real-aggregate form (built by
    // `with_capacity` plus `push`) over a sibling slot of the
    // same buffer: the copied element keeps the element's own
    // form, so equality, length and comparison still read it.
    let mut a = OsString::with_capacity(32);
    a.push("hello-world");
    let b = OsString::from("hello-world");
    let mut v: Vec<OsString> = vec![a, b];
    unsafe {
        let base = v.as_mut_ptr();
        std::ptr::copy_nonoverlapping(base, base.add(1), 1);
    }
    let expect = OsString::from("hello-world");
    println!("eq1: {}", v[1] == expect);
    println!("len1: {}", v[1].len());
    println!("cmp1: {:?}", v[1].cmp(&expect));
    println!("eq0: {}", v[0] == expect);
    println!("lossy1: {}", v[1].to_string_lossy());
    std::mem::forget(v);
}
"#;

#[test]
fn run_mode_os_string_copy_nonoverlapping_shift() {
    run_ok(
        "os-string-copy-nonoverlapping-shift",
        OS_STRING_COPY_SHIFT_SRC,
        "eq1: true\nlen1: 11\ncmp1: Equal\neq0: true\nlossy1: hello-world\n",
    );
}

const OS_STRING_COPY_SPARE_SRC: &str = r#"
use std::ffi::OsString;

fn main() {
    // The same copy into a spare-capacity slot (never written
    // before): the element lands whole there too.
    let mut a = OsString::with_capacity(32);
    a.push("hello-world");
    let mut v: Vec<OsString> = Vec::with_capacity(4);
    v.push(OsString::from("zzz"));
    unsafe {
        let dst = v.as_mut_ptr().add(1);
        std::ptr::copy_nonoverlapping(&a, dst, 1);
        v.set_len(2);
    }
    let expect = OsString::from("hello-world");
    println!("eq1: {}", v[1] == expect);
    println!("len1: {}", v[1].len());
    println!("cmp1: {:?}", v[1].cmp(&expect));
    println!("lossy1: {}", v[1].to_string_lossy());
    std::mem::forget(a);
}
"#;

#[test]
fn run_mode_os_string_copy_into_spare_capacity() {
    run_ok(
        "os-string-copy-into-spare-capacity",
        OS_STRING_COPY_SPARE_SRC,
        "eq1: true\nlen1: 11\ncmp1: Equal\nlossy1: hello-world\n",
    );
}

const OS_STRING_COPY_MULTI_SRC: &str = r#"
use std::ffi::OsString;

fn agg(s: &str) -> OsString {
    let mut o = OsString::with_capacity(32);
    o.push(s);
    o
}

fn main() {
    // A multi-element copy shifts both elements with their
    // forms intact.
    let mut v: Vec<OsString> = vec![agg("aaa"), OsString::from("bbb"), agg("ccc")];
    unsafe {
        let base = v.as_mut_ptr();
        std::ptr::copy_nonoverlapping(base, base.add(1), 2);
    }
    let ea = OsString::from("aaa");
    let eb = OsString::from("bbb");
    println!("eq1: {}", v[1] == ea);
    println!("len1: {}", v[1].len());
    println!("eq2: {}", v[2] == eb);
    println!("len2: {}", v[2].len());
    std::mem::forget(v);
}
"#;

#[test]
fn run_mode_os_string_copy_multi_element() {
    run_ok(
        "os-string-copy-multi-element",
        OS_STRING_COPY_MULTI_SRC,
        "eq1: true\nlen1: 3\neq2: true\nlen2: 3\n",
    );
}

const OS_STRING_PTR_COPY_SRC: &str = r#"
use std::ffi::OsString;

fn agg(s: &str) -> OsString {
    let mut o = OsString::with_capacity(32);
    o.push(s);
    o
}

fn main() {
    // The overlapping copy (the sort shift's form) keeps the
    // shifted element's form too.
    let mut v: Vec<OsString> = vec![agg("hello-world"), OsString::from("hello-world")];
    unsafe {
        let base = v.as_mut_ptr();
        std::ptr::copy(base, base.add(1), 1);
    }
    let expect = OsString::from("hello-world");
    println!("eq1: {}", v[1] == expect);
    println!("len1: {}", v[1].len());
    println!("lossy1: {}", v[1].to_string_lossy());
    std::mem::forget(v);
}
"#;

#[test]
fn run_mode_os_string_ptr_copy_shift() {
    run_ok(
        "os-string-ptr-copy-shift",
        OS_STRING_PTR_COPY_SRC,
        "eq1: true\nlen1: 11\nlossy1: hello-world\n",
    );
}

const OS_STRING_SORT_MIXED_SRC: &str = r#"
use std::ffi::OsString;

fn agg(s: &str) -> OsString {
    let mut o = OsString::with_capacity(32);
    o.push(s);
    o
}

fn main() {
    // Sorting a vector that mixes the real-aggregate form
    // with the from-literal form: the shifts inside the sort
    // move whole elements, so every comparison after a shift
    // still reads an intact element.
    let mut v: Vec<OsString> = vec![agg("bbb"), OsString::from("ddd"), agg("aaa"), OsString::from("ccc")];
    v.sort();
    for s in &v {
        print!("{} ", s.to_string_lossy());
    }
    println!();
    let expect: Vec<OsString> = vec![OsString::from("aaa"), OsString::from("bbb"), OsString::from("ccc"), OsString::from("ddd")];
    println!("sorted_eq: {}", v == expect);
}
"#;

#[test]
fn run_mode_os_string_vec_sort_mixed_forms() {
    run_ok(
        "os-string-vec-sort-mixed-forms",
        OS_STRING_SORT_MIXED_SRC,
        "aaa bbb ccc ddd \nsorted_eq: true\n",
    );
}

const OS_STRING_COPY_WRAPPER_SRC: &str = r#"
use std::ffi::OsString;

fn agg(s: &str) -> OsString {
    let mut o = OsString::with_capacity(32);
    o.push(s);
    o
}

#[repr(transparent)]
struct W(OsString);

fn main() {
    // The genuine wrapper source: the pointer names a
    // transparent wrapper place, and the element is its only
    // field -- the copy moves the field, with the element's
    // own layers intact below the wrapper's.
    let w = W(agg("hello-world"));
    let mut v: Vec<OsString> = vec![OsString::from("zzz")];
    unsafe {
        let src = &w as *const W as *const OsString;
        let dst = v.as_mut_ptr();
        std::ptr::copy_nonoverlapping(src, dst, 1);
    }
    let expect = OsString::from("hello-world");
    println!("eq0: {}", v[0] == expect);
    println!("len0: {}", v[0].len());
    println!("lossy0: {}", v[0].to_string_lossy());
    std::mem::forget(v);
    std::mem::forget(w);
}
"#;

#[test]
fn run_mode_os_string_copy_from_wrapper_place() {
    run_ok(
        "os-string-copy-from-wrapper-place",
        OS_STRING_COPY_WRAPPER_SRC,
        "eq0: true\nlen0: 11\nlossy0: hello-world\n",
    );
}

const STRING_DRAIN_NEXT_BACK_STORAGE_SRC: &str = r#"
fn main() {
    let mut j = ["a", "b", "c"].join("/");
    let mut dj = j.drain(0..1);
    println!("{:?}", dj.next_back());
    println!("{:?}", dj.next_back());
    std::mem::forget(dj);
    println!("[{}]", j);
    let mut c = ["a/", "b/", "c"].concat();
    let mut dc = c.drain(0..1);
    println!("{:?}", dc.next_back());
    println!("{:?}", dc.next_back());
    std::mem::forget(dc);
    println!("[{}]", c);
}
"#;

#[test]
fn run_mode_string_drain_next_back_storage_backed() {
    // A drain over a join/concat-built string holds a `Chars`
    // state whose byte pointers denote the string's own
    // storage. The back step is served by the modelled state
    // machine (as the front step is), which snapshots the
    // remaining bytes through the pointers before stepping;
    // the collected `Drain::next_back` body forwarded the
    // stored storage-backed state to the `Chars` back step
    // directly, and its snapshot reader only reads value
    // snapshots. The drained range is the first byte, the one
    // element whose stored form is plain, so the step is
    // judged on the pointer form alone. The drain values are
    // forgotten: the drop-side removal over a join/concat
    // string is a separate consumer.
    run_ok(
        "string-drain-next-back-storage-backed",
        STRING_DRAIN_NEXT_BACK_STORAGE_SRC,
        "Some('a')\nNone\n[a/b/c]\nSome('a')\nNone\n[a/b/c]\n",
    );
}

const STRING_DRAIN_NEXT_BACK_VALUE_SRC: &str = r#"
fn main() {
    let mut m = String::new();
    m.push_str("a/b/c");
    {
        let mut d = m.drain(1..4);
        println!("{:?}", d.next_back());
    }
    println!("[{}]", m);
}
"#;

#[test]
fn run_mode_string_drain_next_back_value_snapshot() {
    // The same back step over a push-built string, whose
    // drain state already denotes a value snapshot: the
    // modelled route serves it identically, and the drain's
    // drop still removes the stepped range.
    run_ok(
        "string-drain-next-back-value-snapshot",
        STRING_DRAIN_NEXT_BACK_VALUE_SRC,
        "Some('/')\n[ac]\n",
    );
}

const JOIN_BUILT_MUTATE_SRC: &str = r#"
use std::cmp::Ordering;
use std::path::PathBuf;
fn main() {
    let mut s = ["a", "b"].join("/");
    s.push('x');
    println!("{}", s);
    s.push_str("yz");
    println!("{}", s);
    let mut c = ["a", "b"].concat();
    c.push('x');
    println!("{}", c);
    assert_eq!(s, "a/bxyz");
    assert_eq!(s.len(), 6);
    assert_eq!(s.cmp(&String::from("a/bxyz")), Ordering::Equal);
    let p = PathBuf::from(s.clone());
    println!("{:?}", p);
    let mut i = ["he", "llo"].join("");
    i.insert(0, 'X');
    i.insert_str(6, "!");
    println!("{}", i);
    let mut r = ["a", "b", "c"].join("/");
    r.replace_range(0..1, "z");
    let ch = r.remove(4);
    println!("{} {}", r, ch);
    let mut a = ["a", "b"].join("/");
    a += "zz";
    println!("{}", a);
}
"#;

const COW_FROM_OWNED_SRC: &str = r#"use std::borrow::Cow;

fn main() {
    let c: Cow<'_, str> = Cow::from(String::from("from string"));
    println!("deref={}", &*c);
    let v: Cow<'_, [u8]> = Cow::from(vec![1u8, 2, 3]);
    println!("vlen={}", v.len());
}
"#;

#[test]
fn run_mode_join_built_string_mutations() {
    // A `join` / `concat` result stored in a place is the
    // vector's element form (the elements past the first keep
    // the `MaybeUninit` union wrapper), which the mutation
    // path's place read must denominate like the operand
    // readers do; the write-back is the abstract form, so the
    // downstream readers (equality, ordering, `PathBuf::from`)
    // see the mutated text. The expected text is native output.
    run_ok(
        "join-built-mutate",
        JOIN_BUILT_MUTATE_SRC,
        "a/bx\na/bxyz\nabx\n\"a/bxyz\"\nXhello!\nz/b/ c\na/bzz\n",
    );
}

const JOIN_BUILT_TEXT_AT_SRC: &str = r#"
fn main() {
    let mut l = ["a", "b", "c"].join("/");
    l.retain(|ch| ch != 'b');
    println!("{}", l);
    let mut m = ["a", "b", "c"].join("/");
    {
        let _dr = m.drain(0..2);
    }
    println!("{}", m);
    let mut t = ["a", "b", "c"].join("/");
    t.truncate(3);
    let p = t.pop();
    println!("{} {:?}", t, p);
    let tail = t.split_off(2);
    println!("{} {}", t, tail);
}
"#;

#[test]
fn run_mode_join_built_string_retain_and_drain_drop() {
    // The read-only place read behind `retain` (and the
    // removal a dropped `Drain` performs through the mutation
    // path) shares the mutation read's composition, so a
    // stored `join` result serves both. The expected text is
    // native output.
    run_ok(
        "join-built-text-at",
        JOIN_BUILT_TEXT_AT_SRC,
        "a//c\nb/c\na/ Some('b')\na/ \n",
    );
}

const VEC_FROM_JOIN_SRC: &str = r#"
fn main() {
    // `Vec<u8>: From<String>` is `String::into_bytes`: the
    // bytes of a join-built (real aggregate) string as the
    // vector's elements, consumable downstream like any
    // other byte vector.
    let v: Vec<u8> = Vec::from(["ab", "cd"].join("-"));
    println!("{}|{:?}", v.len(), v);
    let w: Vec<u8> = ["a", "b"].join("/").into();
    println!("{:?}", w);
    let back = String::from_utf8(v).unwrap();
    println!("{}", back);
    println!("vec-from-join-ok");
}
"#;

#[test]
fn run_mode_vec_u8_from_join_built_string() {
    run_ok(
        "vec-from-join",
        VEC_FROM_JOIN_SRC,
        "5|[97, 98, 45, 99, 100]\n[97, 47, 98]\nab-cd\nvec-from-join-ok\n",
    );
}

#[test]
fn run_mode_cow_from_owned_sources() {
    // `Cow::from(String)` / `Cow::from(Vec<T>)` (and the
    // matching `Into`): the generic From/Into handler had
    // arms whose source was a `Cow` but none whose target
    // was one, so every construction of a `Cow` through
    // `From` stopped as unsupported. The owned source
    // wraps as the owned variant.
    let dir = temp_dir("cow-from-owned");
    let src = fixture(&dir, "cow_from_owned.rs", COW_FROM_OWNED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cow from owned sources must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "deref=from string\nvlen=3\n"
    );
}

const OS_STR_DETACHED_DEREF_SRC: &str = r#"
use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStringExt;

fn main() {
    // The value `OsString::deref` produces for non-UTF-8
    // bytes is a detached fat pointer over the byte tuple:
    // the byte readers (length, encoded bytes, equality)
    // read it directly, and the consumers that reborrow it
    // (`to_str` through `from_utf8` over
    // `as_encoded_bytes`, iteration over the encoded bytes)
    // read the same bytes through the reborrowed pointer.
    let os = OsString::from_vec(vec![b'a', 0xFF, b'b']);
    let s: &OsStr = &os;
    println!("len={}", s.len());
    println!("empty={}", s.is_empty());
    println!("enc={:?}", s.as_encoded_bytes());
    println!("eq_self={}", s == s);
    println!("to_str={:?}", s.to_str());
    println!("lossy={}", s.to_string_lossy());
    println!("debug={:?}", s);
    println!("utf8={:?}", std::str::from_utf8(s.as_encoded_bytes()).map(|x| x.len()));
    let mut sum = 0u32;
    for x in s.as_encoded_bytes() {
        sum += *x as u32;
    }
    println!("sum={}", sum);
}
"#;

#[test]
fn run_mode_os_str_detached_deref_consumers() {
    run_ok(
        "os-str-detached-deref-consumers",
        OS_STR_DETACHED_DEREF_SRC,
        "len=3\nempty=false\nenc=[97, 255, 98]\neq_self=true\nto_str=None\nlossy=a\u{FFFD}b\ndebug=\"a\\xFFb\"\nutf8=Err(Utf8Error { valid_up_to: 1, error_len: Some(1) })\nsum=450\n",
    );
}

const OS_STR_DETACHED_DEREF_ORDER_SRC: &str = r#"
use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStringExt;
use std::path::PathBuf;

fn main() {
    // Ordering between two detached derefs (the operands
    // reach the comparison as references to the stored fat
    // pointers), sorting a vector of them, their `Debug`
    // inside composite forms, and the same consumers over
    // the detached values `parent` / `file_name` produce.
    let a = OsString::from_vec(vec![b'a', 0xFF, b'b']);
    let b = OsString::from_vec(vec![b'a', 0xFF, b'c']);
    let sa: &OsStr = &a;
    let sb: &OsStr = &b;
    println!("eq={}", sa == sb);
    println!("lt={}", sa < sb);
    println!("gt={}", sb > sa);
    println!("cmp={:?}", sa.cmp(sb));
    println!("pcmp={:?}", sa.partial_cmp(sb));
    let mut v: Vec<&OsStr> = vec![sb, sa];
    v.sort();
    println!("sorted={:?}", v);
    println!("tuple={:?}", (sa, 1));
    let mut pb = PathBuf::from(OsString::from_vec(vec![b'd', 0xFF]));
    pb.push(OsString::from_vec(vec![b'a', 0xFF, b'b']));
    let parent = pb.parent().unwrap();
    println!("parent_tostr={:?}", parent.to_str());
    println!("parent_debug={:?}", parent);
    let fname = pb.file_name().unwrap();
    println!("fname_tostr={:?}", fname.to_str());
    println!("fname_debug={:?}", fname);
    println!("fname_lt={}", fname < parent.as_os_str());
}
"#;

const BOXED_STR_SORT_SRC: &str = r#"
fn main() {
    // A two-element sort is the smallest shape that reaches
    // smallsort's `insert_tail`: it prices the neighbouring
    // elements' copy ranges, and the fat element's stride is
    // the element type's size -- sixteen, not the thin
    // pointer's eight.
    let mut v: Vec<Box<str>> = vec!["pear".into(), "banana".into()];
    v.sort();
    println!("two={:?}", v);
    let mut sorted: Vec<Box<str>> = vec!["apple".into(), "banana".into()];
    sorted.sort();
    println!("sorted={:?}", sorted);
    let a: Box<str> = "pear".into();
    let b: Box<str> = "banana".into();
    println!("cmp={:?} eq={}", a.cmp(&b), a == b);
}
"#;

#[test]
fn run_mode_os_str_detached_deref_ordering() {
    run_ok(
        "os-str-detached-deref-ordering",
        OS_STR_DETACHED_DEREF_ORDER_SRC,
        "eq=false\nlt=true\ngt=true\ncmp=Less\npcmp=Some(Less)\nsorted=[\"a\\xFFb\", \"a\\xFFc\"]\ntuple=(\"a\\xFFb\", 1)\nparent_tostr=None\nparent_debug=\"d\\xFF\"\nfname_tostr=None\nfname_debug=\"a\\xFFb\"\nfname_lt=true\n",
    );
}

const OS_STR_BARE_FORM_ORD_SRC: &str = r#"
use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStringExt;

fn agg(s: &str) -> OsString {
    let mut o = OsString::with_capacity(16);
    o.push(s);
    o
}

fn main() {
    // A boxed `OsStr`'s pointee is the bare byte string;
    // ordering it against a literal `OsStr` (the text
    // form) and against `OsString` values in their
    // element and real-aggregate forms reads the same
    // bytes equality does, in both operand orders.
    let boxed = OsString::from_vec(b"abc".to_vec()).into_boxed_os_str();
    let lit = OsStr::new("abd");
    println!("lit lt={} pc={:?}", &*boxed < lit, (*boxed).partial_cmp(lit));
    println!("lit gt={} pc={:?}", lit > &*boxed, lit.partial_cmp(&*boxed));
    let from = OsString::from("abd");
    println!("from lt={} pc={:?}", &*boxed < from, (*boxed).partial_cmp(&from));
    println!("from gt={} pc={:?}", from > &*boxed, from.partial_cmp(&*boxed));
    let a = agg("abd");
    println!("agg lt={} pc={:?}", &*boxed < a, (*boxed).partial_cmp(&a));
    println!("agg gt={} pc={:?}", a > &*boxed, a.partial_cmp(&*boxed));
    let same = OsString::from("abc");
    println!("same pc={:?} lt={} le={}", (*boxed).partial_cmp(&same), &*boxed < same, &*boxed <= same);
    // The text-form box (its bytes came from a string)
    // orders the same way.
    let text_boxed = OsString::from("abc").into_boxed_os_str();
    println!("text lt={} pc={:?}", &*text_boxed < lit, (*text_boxed).partial_cmp(lit));
    // Non-UTF-8 bytes order by their raw bytes, against
    // another box and against a literal.
    let raw = OsString::from_vec(vec![0xff, b'a']).into_boxed_os_str();
    let raw2 = OsString::from_vec(vec![0xff, b'b']).into_boxed_os_str();
    println!("raw lt={} pc={:?}", raw < raw2, raw.partial_cmp(&raw2));
    println!("raw-lit pc={:?}", (*raw).partial_cmp(OsStr::new("a")));
}
"#;

#[test]
fn run_mode_os_str_bare_form_ordering() {
    run_ok(
        "os-str-bare-form-ordering",
        OS_STR_BARE_FORM_ORD_SRC,
        "lit lt=true pc=Some(Less)\nlit gt=true pc=Some(Greater)\nfrom lt=true pc=Some(Less)\nfrom gt=true pc=Some(Greater)\nagg lt=true pc=Some(Less)\nagg gt=true pc=Some(Greater)\nsame pc=Some(Equal) lt=false le=true\ntext lt=true pc=Some(Less)\nraw lt=true pc=Some(Less)\nraw-lit pc=Some(Greater)\n",
    );
}

const FROM_OS_STR_TO_COUNTED_SRC: &str = r#"
use std::ffi::{OsStr, OsString};
use std::rc::Rc;
use std::sync::Arc;
fn main() {
    let r: Rc<OsStr> = Rc::from(OsStr::new("borrowed"));
    println!("rc-ref: {:?} len={}", r, r.len());
    let r2: Rc<OsStr> = Rc::from(OsString::from("owned"));
    println!("rc-owned: {:?}", r2);
    let b: Box<OsStr> = OsString::from("boxed").into_boxed_os_str();
    let r3: Rc<OsStr> = Rc::from(b);
    println!("rc-box: {:?} strong={}", r3, Rc::strong_count(&r3));
    let a: Arc<OsStr> = Arc::from(OsStr::new("arc-ref"));
    let a2: Arc<OsStr> = Arc::from(OsString::from("arc-owned"));
    let ab: Box<OsStr> = OsString::from("arc-box").into_boxed_os_str();
    let a3: Arc<OsStr> = Arc::from(ab);
    println!("arc: {:?} {:?} {:?}", a, a2, a3);
    let v: Vec<Rc<OsStr>> = vec![Rc::from(OsStr::new("x")), Rc::from(OsString::from("yy"))];
    println!("vec: {:?}", v);
    println!("eq={}", r == Rc::from(OsStr::new("borrowed")));
}
"#;

#[test]
fn run_mode_from_os_str_to_counted() {
    // `From<&OsStr>` / `From<OsString>` / `From<Box<OsStr>>`
    // for `Rc<OsStr>` / `Arc<OsStr>`: the borrowed and owned
    // forms clone the byte string into a fresh counted handle;
    // the boxed form moves the pointee out of the box. Before
    // the fix all three stopped in the generic `From` handler
    // as not runtime-supported, so a `Vec<Rc<OsStr>>` could not
    // even be constructed (expected output verified against a
    // native 1.100.0 build of the same source).
    run_ok(
        "from-os-str-counted",
        FROM_OS_STR_TO_COUNTED_SRC,
        "rc-ref: \"borrowed\" len=8\nrc-owned: \"owned\"\nrc-box: \"boxed\" strong=1\narc: \"arc-ref\" \"arc-owned\" \"arc-box\"\nvec: [\"x\", \"yy\"]\neq=true\n",
    );
}

const FROM_PATH_TO_COUNTED_SRC: &str = r#"
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
fn main() {
    let r: Rc<Path> = Rc::from(Path::new("a/b"));
    println!("rc-ref: {:?} name={:?}", r, r.file_name());
    let r2: Rc<Path> = Rc::from(PathBuf::from("c/d"));
    println!("rc-owned: {:?}", r2);
    let b: Box<Path> = PathBuf::from("e/f").into_boxed_path();
    let r3: Rc<Path> = Rc::from(b);
    println!("rc-box: {:?} disp={}", r3, r3.display());
    let a: Arc<Path> = Arc::from(Path::new("g/h"));
    let a2: Arc<Path> = Arc::from(PathBuf::from("i"));
    println!("arc: {:?} {:?}", a, a2);
}
"#;

#[test]
fn run_mode_from_path_to_counted() {
    // `From<&Path>` / `From<PathBuf>` / `From<Box<Path>>` for
    // `Rc<Path>` / `Arc<Path>`, the path twin of the `OsStr`
    // conversions above.
    run_ok(
        "from-path-counted",
        FROM_PATH_TO_COUNTED_SRC,
        "rc-ref: \"a/b\" name=Some(\"b\")\nrc-owned: \"c/d\"\nrc-box: \"e/f\" disp=e/f\narc: \"g/h\" \"i\"\n",
    );
}

const FROM_C_STR_TO_COUNTED_SRC: &str = r#"
use std::ffi::{CStr, CString};
use std::rc::Rc;
use std::sync::Arc;
fn main() {
    let r: Rc<CStr> = Rc::from(CStr::from_bytes_with_nul(b"hello\0").unwrap());
    println!("rc-ref: {:?} bytes={:?}", r, r.to_bytes());
    let r2: Rc<CStr> = Rc::from(CString::new("owned").unwrap());
    println!("rc-owned: {:?}", r2);
    let b: Box<CStr> = Box::from(CStr::from_bytes_with_nul(b"boxed\0").unwrap());
    let r3: Rc<CStr> = Rc::from(b);
    println!("rc-box: {:?} strong={}", r3, Rc::strong_count(&r3));
    let a: Arc<CStr> = Arc::from(CStr::from_bytes_with_nul(b"arc-ref\0").unwrap());
    let a2: Arc<CStr> = Arc::from(CString::new("arc-owned").unwrap());
    println!("arc: {:?} {:?}", a, a2);
}
"#;

#[test]
fn run_mode_from_c_str_to_counted() {
    // `From<&CStr>` / `From<CString>` / `From<Box<CStr>>` for
    // `Rc<CStr>` / `Arc<CStr>`: the handle stores the byte
    // sequence, terminator included, as a `CString` buffer
    // does.
    run_ok(
        "from-c-str-counted",
        FROM_C_STR_TO_COUNTED_SRC,
        "rc-ref: \"hello\" bytes=[104, 101, 108, 108, 111]\nrc-owned: \"owned\"\nrc-box: \"boxed\" strong=1\narc: \"arc-ref\" \"arc-owned\"\n",
    );
}

const FROM_BOX_TO_COUNTED_SRC: &str = r#"
use std::rc::Rc;
use std::sync::Arc;
fn main() {
    let r: Rc<u32> = Rc::from(Box::new(42u32));
    println!("rc-box-u32: {} strong={}", r, Rc::strong_count(&r));
    let a: Arc<String> = Arc::from(Box::new(String::from("text")));
    println!("arc-box-string: {}", a);
}
"#;

#[test]
fn run_mode_from_box_to_counted() {
    // The generic `From<Box<T>>` for `Rc<T>` / `Arc<T>`: the
    // box pointee moves into a fresh counted handle and the
    // box allocation is released.
    run_ok(
        "from-box-counted",
        FROM_BOX_TO_COUNTED_SRC,
        "rc-box-u32: 42 strong=1\narc-box-string: text\n",
    );
}

const FROM_COW_TO_COUNTED_SRC: &str = r#"
use std::borrow::Cow;
use std::ffi::{CStr, OsStr, OsString};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
fn main() {
    let a: Rc<str> = Rc::from(Cow::Borrowed("x"));
    let a2: Rc<str> = Rc::from(Cow::Owned::<str>(String::from("owned-str")));
    let b: Rc<[u8]> = Rc::from(Cow::Borrowed(&[1u8, 2][..]));
    let b2: Arc<[u8]> = Arc::from(Cow::Owned::<[u8]>(vec![3u8, 4]));
    let c: Rc<OsStr> = Rc::from(Cow::Borrowed(OsStr::new("y")));
    let c2: Arc<OsStr> = Arc::from(Cow::Owned::<OsStr>(OsString::from("owned-os")));
    let d: Rc<Path> = Rc::from(Cow::Borrowed(Path::new("z")));
    let e: Rc<CStr> = Rc::from(Cow::Borrowed(CStr::from_bytes_with_nul(b"w\0").unwrap()));
    println!("{:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}", a, a2, b, b2, c, c2, d, e);
}
"#;

#[test]
fn run_mode_from_cow_to_counted() {
    // The Cow forms of the counted conversions: the borrowed
    // variant clones the borrowed data into the handle, the
    // owned variant hands its owned form over.
    run_ok(
        "from-cow-counted",
        FROM_COW_TO_COUNTED_SRC,
        "\"x\" \"owned-str\" [1, 2] [3, 4] \"y\" \"owned-os\" \"z\" \"w\"\n",
    );
}

const OS_STRING_REAL_AGGREGATE_AS_REF_SRC: &str = r#"
use std::ffi::{OsStr, OsString};

fn main() {
    let mut s = OsString::with_capacity(16);
    s.push("hello");
    s.push("-world");
    let r: &OsStr = s.as_ref();
    println!("len={}", r.len());
    println!("lossy={}", r.to_string_lossy());
}
"#;

#[test]
fn run_mode_os_string_real_aggregate_as_ref() {
    // An `OsString` built by `with_capacity` plus `push` is
    // stored in its real aggregate form (its bytes behind the
    // buffer's raw pointer); viewing it as an `&OsStr`
    // through `AsRef` reads the bytes through that buffer,
    // like the deref, equality and format consumers do.
    run_ok(
        "os-string-real-aggregate-as-ref",
        OS_STRING_REAL_AGGREGATE_AS_REF_SRC,
        "len=11\nlossy=hello-world\n",
    );
}

const STRING_JOIN_AS_REF_SRC: &str = r#"
fn main() {
    let s = ["foo", "bar", "baz"].join("/");
    let r: &str = s.as_ref();
    println!("joined={} len={}", r, r.len());
    let b: &[u8] = s.as_ref();
    println!("bytes={:?}", b);
}
"#;

#[test]
fn run_mode_string_join_as_ref_views() {
    // A `String` built by `slice::join` is stored as the
    // element form whose elements past the first copied
    // slice keep the `MaybeUninit` union form; viewing it
    // as an `&str` (or as a byte slice) through `AsRef`
    // denominates that form as the string it holds.
    run_ok(
        "string-join-as-ref-views",
        STRING_JOIN_AS_REF_SRC,
        "joined=foo/bar/baz len=11\nbytes=[102, 111, 111, 47, 98, 97, 114, 47, 98, 97, 122]\n",
    );
}


const OS_STRING_REAL_AGGREGATE_CLONE_SRC: &str = r#"
use std::ffi::OsString;

fn agg() -> OsString {
    let mut o = OsString::with_capacity(16);
    o.push("banana");
    o
}

fn main() {
    let v: Vec<OsString> = vec![agg()];
    let c = v.clone();
    println!("clone={:?}", c);
    let boxed: Vec<Box<std::ffi::OsStr>> =
        c.into_iter().map(|o| o.into_boxed_os_str()).collect();
    println!("clone-box={:?}", boxed);
    let both: Vec<OsString> = vec![OsString::from("pear"), agg()];
    let bc = both.clone();
    println!("mixed-clone={:?}", bc);
}
"#;

#[test]
fn run_mode_os_string_real_aggregate_clone() {
    // An `OsString` built by `with_capacity` + `push` is in
    // its real-aggregate form (the buffer's `Vec` is a
    // `RawVec` + length pair, not an element list): cloning a
    // vector of them treated that pair as elements, and the
    // shape-only clone read the buffer's `NonNull` as a built
    // box, so the clone kept only the first byte
    // ("b\0\0\0\0\0"). The clone now translates the real
    // form to the modelled element form first. The expected
    // text is native output.
    run_ok(
        "os-string-real-agg-clone",
        OS_STRING_REAL_AGGREGATE_CLONE_SRC,
        "clone=[\"banana\"]\nclone-box=[\"banana\"]\nmixed-clone=[\"pear\", \"banana\"]\n",
    );
}

const OS_STRING_REAL_AGGREGATE_DRAIN_SRC: &str = r#"
use std::ffi::OsString;

fn agg() -> OsString {
    let mut o = OsString::with_capacity(16);
    o.push("banana");
    o
}

fn main() {
    let mut v: Vec<OsString> = vec![agg()];
    let mut d = v.drain(..);
    let item = d.next().unwrap();
    println!("len={}", item.len());
    println!("boxed={:?}", item.into_boxed_os_str());
    drop(d);
    let mut v2: Vec<OsString> = vec![agg(), OsString::from("pear")];
    let drained: Vec<OsString> = v2.drain(..).collect();
    println!("drained={:?}", drained);
    println!("rest={:?}", v2);
}
"#;

#[test]
fn run_mode_os_string_real_aggregate_drain() {
    // The converted `Vec::drain` construction builds its
    // iterator state over the vector's first-element pointer;
    // taking that pointer of the range slice (itself a
    // first-element pointer) appended a second index step, so
    // the drain's element read projected into the element
    // value and yielded its first field (the `Buf`) in place
    // of the `OsString` -- consumers then refused the ADT
    // value. The first-element judgment now keeps an ADT
    // element at an element position as an element. The
    // expected text is native output.
    run_ok(
        "os-string-real-agg-drain",
        OS_STRING_REAL_AGGREGATE_DRAIN_SRC,
        "len=6\nboxed=\"banana\"\ndrained=[\"banana\", \"pear\"]\nrest=[]\n",
    );
}

#[test]
fn run_mode_boxed_str_vec_sort() {
    run_ok(
        "boxed-str-vec-sort",
        BOXED_STR_SORT_SRC,
        "two=[\"banana\", \"pear\"]\nsorted=[\"apple\", \"banana\"]\ncmp=Greater eq=false\n",
    );
}

const BOXED_STR_MOVEMENT_SRC: &str = r#"
fn main() {
    // Element movement over fat-box elements that no
    // comparison decides: reversal, rotation, swaps, and the
    // shifting `remove` all move the sixteen-byte elements
    // through the copy machinery.
    let mut w: Vec<Box<str>> = vec!["a".into(), "b".into(), "c".into(), "d".into()];
    w.reverse();
    println!("rev={:?}", w);
    w.rotate_left(1);
    println!("rotl={:?}", w);
    w.swap(0, 3);
    println!("swap={:?}", w);
    let r = w.remove(1);
    println!("removed={} w={:?}", r, w);
    let x: Box<str> = "hello".into();
    let y = x.clone();
    println!("clone={} eq={}", y, x == y);
}
"#;

#[test]
fn run_mode_boxed_str_vec_movement() {
    run_ok(
        "boxed-str-vec-movement",
        BOXED_STR_MOVEMENT_SRC,
        "rev=[\"d\", \"c\", \"b\", \"a\"]\nrotl=[\"c\", \"b\", \"a\", \"d\"]\nswap=[\"d\", \"b\", \"a\", \"c\"]\nremoved=b w=[\"d\", \"a\", \"c\"]\nclone=hello eq=true\n",
    );
}

const BOXED_STR_SWAP_THEN_SORT_SRC: &str = r#"
fn main() {
    // A `ptr::swap` exchange moves each element out of its
    // slot through a wrapper's field (the swap temporary, and
    // the sort's own `ManuallyDrop` hole): the element must
    // arrive whole, or the sort's element access refuses the
    // slot the exchange left behind.
    let mut v: Vec<Box<str>> = vec!["pear".into(), "kiwi".into(), "apple".into(), "plum".into()];
    v.swap(0, 3);
    v.sort();
    println!("swap03={:?}", v);
    let mut v: Vec<Box<str>> = vec!["pear".into(), "kiwi".into(), "apple".into(), "plum".into()];
    v.swap(1, 3);
    v.sort_unstable();
    println!("swap13u={:?}", v);
    let mut v: Vec<Box<str>> = vec!["pear".into(), "kiwi".into(), "apple".into(), "plum".into()];
    v.swap(0, 3);
    v.swap(0, 3);
    v.sort();
    println!("ident={:?}", v);
    let mut v: Vec<Box<str>> = vec!["date".into(), "pear".into(), "kiwi".into(), "apple".into(), "plum".into()];
    v.swap(0, 3);
    v.sort();
    println!("five={:?}", v);
    let mut v: Vec<Box<str>> = Vec::new();
    v.push("pear".into());
    v.push("kiwi".into());
    v.push("apple".into());
    v.push("plum".into());
    v.swap(0, 3);
    let mut c = v.clone();
    c.sort();
    println!("push_clone={:?}", c);
}
"#;

#[test]
fn run_mode_boxed_str_vec_swap_then_sort() {
    run_ok(
        "boxed-str-vec-swap-then-sort",
        BOXED_STR_SWAP_THEN_SORT_SRC,
        "swap03=[\"apple\", \"kiwi\", \"pear\", \"plum\"]\nswap13u=[\"apple\", \"kiwi\", \"pear\", \"plum\"]\nident=[\"apple\", \"kiwi\", \"pear\", \"plum\"]\nfive=[\"apple\", \"date\", \"kiwi\", \"pear\", \"plum\"]\npush_clone=[\"apple\", \"kiwi\", \"pear\", \"plum\"]\n",
    );
}

const BOXED_STR_SWAP_SIBLINGS_SRC: &str = r#"
fn main() {
    // The other slot surgery after an exchange: the shifted
    // and removed elements cross the same wrapper-field
    // positions.
    let mut v: Vec<Box<str>> = vec!["pear".into(), "kiwi".into(), "apple".into(), "plum".into()];
    v.swap(0, 3);
    let r = v.swap_remove(1);
    println!("swap_remove={} {:?}", r, v);
    let mut v: Vec<Box<str>> = vec!["pear".into(), "kiwi".into(), "apple".into(), "plum".into()];
    v.swap(0, 3);
    v.insert(1, "fig".into());
    println!("insert={:?}", v);
    let mut v: Vec<Box<str>> = vec!["pear".into(), "kiwi".into(), "apple".into(), "plum".into()];
    v.swap(0, 3);
    let r = v.remove(0);
    println!("remove={} {:?}", r, v);
    let mut v: Vec<Box<str>> = vec!["pear".into(), "kiwi".into(), "apple".into(), "plum".into()];
    v.swap(0, 3);
    v.sort_by_key(|s| s.len());
    println!("by_key={:?}", v);
}
"#;

#[test]
fn run_mode_boxed_str_vec_swap_siblings() {
    run_ok(
        "boxed-str-vec-swap-siblings",
        BOXED_STR_SWAP_SIBLINGS_SRC,
        "swap_remove=kiwi [\"plum\", \"pear\", \"apple\"]\ninsert=[\"plum\", \"fig\", \"kiwi\", \"apple\", \"pear\"]\nremove=plum [\"kiwi\", \"apple\", \"pear\"]\nby_key=[\"plum\", \"kiwi\", \"pear\", \"apple\"]\n",
    );
}

const BOXED_STR_SMART_POINTER_SORT_SRC: &str = r#"
use std::rc::Rc;
use std::sync::Arc;

fn main() {
    // The other fat smart pointers over `str`: their element
    // stride is two words as well.
    let mut rc: Vec<Rc<str>> = vec!["pear".into(), "banana".into()];
    rc.sort();
    println!("rc={:?}", rc);
    let mut arc: Vec<Arc<str>> = vec!["pear".into(), "banana".into()];
    arc.sort();
    println!("arc={:?}", arc);
}
"#;

#[test]
fn run_mode_smart_pointer_str_vec_sort() {
    run_ok(
        "smart-pointer-str-vec-sort",
        BOXED_STR_SMART_POINTER_SORT_SRC,
        "rc=[\"banana\", \"pear\"]\narc=[\"banana\", \"pear\"]\n",
    );
}

const BOXED_STR_SORT_MIXED_SRC: &str = r#"
fn main() {
    // Sorts of three or more `Box<str>` elements: the
    // sort's rotations hand the comparison the mixed
    // presentation -- one side the stored box aggregate,
    // the other the bare `str` snapshot a temporary holds
    // -- which the box-pointee comparison machinery
    // decides through the pointees.
    let mut v: Vec<Box<str>> = vec!["ape".into(), "bee".into(), "cat".into()];
    v.sort();
    println!("p1={:?}", v);
    let mut v: Vec<Box<str>> = vec!["ape".into(), "cat".into(), "bee".into()];
    v.sort();
    println!("p2={:?}", v);
    let mut v: Vec<Box<str>> = vec!["bee".into(), "ape".into(), "cat".into()];
    v.sort();
    println!("p3={:?}", v);
    let mut v: Vec<Box<str>> = vec!["bee".into(), "cat".into(), "ape".into()];
    v.sort();
    println!("p4={:?}", v);
    let mut v: Vec<Box<str>> = vec!["cat".into(), "ape".into(), "bee".into()];
    v.sort();
    println!("p5={:?}", v);
    let mut v: Vec<Box<str>> = vec!["cat".into(), "bee".into(), "ape".into()];
    v.sort();
    println!("p6={:?}", v);
    let mut v: Vec<Box<str>> = vec!["pear".into(), "kiwi".into(), "apple".into(), "plum".into(), "fig".into()];
    v.sort();
    println!("five={:?}", v);
    let mut v: Vec<Box<str>> = vec!["kiwi".into(), "kiwi".into(), "apple".into(), "kiwi".into(), "apple".into()];
    v.sort();
    println!("dup={:?}", v);
    let mut v: Vec<Box<str>> = vec!["pear".into(), "banana".into(), "kiwi".into(), "apple".into(), "plum".into(), "fig".into()];
    v.sort_unstable();
    println!("unstable={:?}", v);
    let mut v: Vec<Box<str>> = vec!["one".into(), "two".into(), "three".into(), "four".into(), "five".into(), "six".into(), "seven".into(), "eight".into()];
    v.sort_unstable();
    println!("unstable8={:?}", v);
    let mut v: Vec<Box<str>> = vec!["pear".into(), "kiwi".into(), "apple".into(), "plum".into()];
    v.sort_by(|a, b| a.cmp(b));
    println!("by-cmp={:?}", v);
    let mut v: Vec<Box<str>> = vec!["pear".into(), "kiwi".into(), "apple".into(), "plum".into()];
    v.sort_by(|a, b| b.cmp(a));
    println!("by-rev={:?}", v);
    let mut v: Vec<Box<str>> = vec!["ccc".into(), "a".into(), "bb".into(), "dddd".into(), "e".into()];
    v.sort_by_key(|s| s.len());
    println!("by-len={:?}", v);
}
"#;

#[test]
fn run_mode_boxed_str_vec_sort_mixed_pairs() {
    run_ok(
        "boxed-str-vec-sort-mixed-pairs",
        BOXED_STR_SORT_MIXED_SRC,
        "p1=[\"ape\", \"bee\", \"cat\"]\np2=[\"ape\", \"bee\", \"cat\"]\np3=[\"ape\", \"bee\", \"cat\"]\np4=[\"ape\", \"bee\", \"cat\"]\np5=[\"ape\", \"bee\", \"cat\"]\np6=[\"ape\", \"bee\", \"cat\"]\nfive=[\"apple\", \"fig\", \"kiwi\", \"pear\", \"plum\"]\ndup=[\"apple\", \"apple\", \"kiwi\", \"kiwi\", \"kiwi\"]\nunstable=[\"apple\", \"banana\", \"fig\", \"kiwi\", \"pear\", \"plum\"]\nunstable8=[\"eight\", \"five\", \"four\", \"one\", \"seven\", \"six\", \"three\", \"two\"]\nby-cmp=[\"apple\", \"kiwi\", \"pear\", \"plum\"]\nby-rev=[\"plum\", \"pear\", \"kiwi\", \"apple\"]\nby-len=[\"a\", \"e\", \"bb\", \"ccc\", \"dddd\"]\n",
    );
}

const BOXED_STR_SORT_AFTER_MOVEMENT_SRC: &str = r#"
fn main() {
    // Sorts fed by movement shapes whose elements no
    // comparison has touched yet, and whole-vector
    // comparisons of the sorted results: the elements
    // arrive at the sort in the forms reversal and
    // rotation left them in.
    let mut v: Vec<Box<str>> = vec!["pear".into(), "kiwi".into(), "apple".into(), "plum".into()];
    v.reverse();
    println!("rev={:?}", v);
    v.sort();
    println!("rev-sorted={:?}", v);
    let mut w: Vec<Box<str>> = vec!["dog".into(), "ape".into(), "cat".into(), "bee".into()];
    w.rotate_left(1);
    println!("rotl={:?}", w);
    w.sort();
    println!("rotl-sorted={:?}", w);
    let mut x: Vec<Box<str>> = vec!["pear".into(), "kiwi".into(), "apple".into(), "plum".into(), "fig".into()];
    let mut y: Vec<Box<str>> = vec!["pear".into(), "kiwi".into(), "apple".into(), "plum".into(), "fig".into()];
    x.sort();
    y.sort();
    println!("eq={} pcmp={:?} slice-eq={}", x == y, x.partial_cmp(&y), x.as_slice() == y.as_slice());
    let a: Box<str> = "same".into();
    let b: Box<str> = "same".into();
    println!("box-eq={} box-lt={}", a == b, a < b);
}
"#;

#[test]
fn run_mode_boxed_str_vec_sort_after_movement() {
    run_ok(
        "boxed-str-vec-sort-after-movement",
        BOXED_STR_SORT_AFTER_MOVEMENT_SRC,
        "rev=[\"plum\", \"apple\", \"kiwi\", \"pear\"]\nrev-sorted=[\"apple\", \"kiwi\", \"pear\", \"plum\"]\nrotl=[\"ape\", \"cat\", \"bee\", \"dog\"]\nrotl-sorted=[\"ape\", \"bee\", \"cat\", \"dog\"]\neq=true pcmp=Some(Equal) slice-eq=true\nbox-eq=true box-lt=false\n",
    );
}
