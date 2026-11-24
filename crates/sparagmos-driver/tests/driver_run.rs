//! Run-mode smoke tests: build small fixture programs in a temp dir,
//! execute them with `sparagmos-driver run`, and check the exit-code
//! contract (0 = normal termination, 101 = guest panic, 2 = could
//! not convert/execute, with the reason on stderr). The test only
//! spawns the binary; the driver itself must have been built with
//! the pinned 1.100.0 toolchain (see the crate README).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture(dir: &Path, name: &str, src: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, src).unwrap();
    path
}

fn run_driver(src: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .output()
        .expect("failed to run sparagmos-driver")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("sparagmos-driver-run-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const PANIC_SRC: &str = r#"
fn boom(x: i32) -> i32 { x + x }
fn main() { let _ = boom(2_000_000_000); }
"#;

const PRINTLN_SRC: &str = r#"
fn main() { println!("hi"); }
"#;

const SILENT_SRC: &str = r#"
fn main() {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < 100 { s += i; i += 1; }
    let _ = s;
}
"#;

#[test]
fn run_mode_guest_panic_exits_101() {
    let dir = temp_dir("panic");
    let src = fixture(&dir, "panic.rs", PANIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "guest panic must exit 101; stderr={stderr}"
    );
    assert!(
        stderr.contains("panicked"),
        "panic message goes to stderr: {stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "run mode keeps stdout clean: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn summary_mode_tail_call_body_is_skipped_with_named_reason() {
    // In summary (convert) mode the same containment reports the
    // tail-call body as skipped with its named reason, and the
    // rest of the crate still converts: no conversion panic is
    // reported anywhere.
    let dir = temp_dir("tailcall-summary");
    let src = fixture(&dir, "tailcall.rs", TAIL_CALL_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "summary mode succeeds with a named skip; stderr={stderr}"
    );
    let line = stdout
        .lines()
        .find(|l| l.contains("countdown"))
        .unwrap_or_else(|| panic!("summary lists countdown: {stdout}"));
    assert!(
        line.contains("body contains a tail call"),
        "the skip reason names the tail call: {line}"
    );
    assert!(
        !stdout.contains("panicked") && !stderr.contains("panicked"),
        "no conversion panic is reported: stdout={stdout} stderr={stderr}"
    );
}

#[test]
fn run_mode_println_prints_to_stdout() {
    let dir = temp_dir("println");
    let src = fixture(&dir, "println.rs", PRINTLN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "println! runs through the interpreter's runtime print support; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hi\n",
        "the guest's stdout is the driver's stdout"
    );
}

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

#[test]
fn run_mode_silent_program_exits_zero() {
    // A plain computation-only main converts and executes to the end.
    // This used to be blocked in the interpreter: real MIR never
    // writes the return place `_0` of a unit-returning function, and
    // `Return` read it as uninitialized (exit 2, reason naming `_0`).
    // The interpreter now reads an unwritten unit `_0` as `()`, so
    // the program terminates normally.
    let dir = temp_dir("silent");
    let src = fixture(&dir, "silent.rs", SILENT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "silent program must terminate normally; stderr={stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "run mode keeps stdout clean: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

// ----- by-reference constants (Lane R) ---------------------------------------

const PROMOTED_REF_SRC: &str = r#"
fn main() {
    let x = 10;
    let y = x;
    assert_eq!(y, 10);
    let z: u8 = 255;
    assert_eq!(z, 255);
}
"#;

const CONST_SHAPES_SRC: &str = r#"
struct S { a: i32, b: u8 }
const S1: S = S { a: -2, b: 7 };
const ARR: [i32; 3] = [10, 20, 30];
const R: &i32 = &ARR[1];
static A: usize = 42;
static B: usize = A;
const NAMES: [&str; 2] = ["x", "yy"];
const OPT: Option<&i32> = Some(&5);
fn main() {
    println!("{} {} {} {} {} {}", S1.a, S1.b, *R, B, NAMES[1],
             match OPT { Some(r) => *r, None => 0 });
}
"#;

#[test]
fn run_mode_promoted_ref_constant() {
    // `assert_eq!` against a literal builds a promoted `&i32`
    // constant; it must execute, not stop at the constant.
    let dir = temp_dir("promoted");
    let src = fixture(&dir, "promoted.rs", PROMOTED_REF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "promoted reference constants must run; stderr={stderr}"
    );
}

/// Build a guest source holding a 5,000-element byte table
/// (past the driver's element-wise constant bound of 4,096):
/// `decl` declares the table (with `ELEMS` as the element list)
/// and a `read_at` accessor over it; `main` sums the table and
/// prints the sum plus two spot reads.
fn large_byte_table_src(decl: &str) -> String {
    let elems: Vec<String> = (0..5000).map(|i| (i % 251).to_string()).collect();
    let mut src = decl.replace("ELEMS", &elems.join(", "));
    src.push_str("fn main() { let mut sum: u32 = 0;");
    src.push_str(" for i in 0..5000 { sum += read_at(i) as u32; }");
    src.push_str(" println!(\"{} {} {}\", sum, read_at(0), read_at(4999)); }");
    src
}

#[test]
fn run_mode_large_byte_table_static() {
    // A static byte table past the element-wise constant bound
    // materialises as the raw bytes and reads back exactly.
    let src = large_byte_table_src(
        "static TABLE: [u8; 5000] = [ELEMS]; fn read_at(i: usize) -> u8 { TABLE[i] }",
    );
    let dir = temp_dir("large-byte-table");
    let src = fixture(&dir, "large_byte_table.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "large byte table must run; stderr={stderr}"
    );
    let expect_sum: u32 = (0..5000).map(|i| i % 251).sum();
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("{expect_sum} 0 {}\n", 4999 % 251),
        "decoded table values must match the source program"
    );
}

#[test]
fn run_mode_large_byte_table_wrapped_static() {
    // The same table inside an alignment-wrapper newtype -- the
    // shape of the `unicode_ident` XID tables, whose reference
    // constants reached the converter as `ByRef` identities
    // before byte arrays past the bound materialised.
    let src = large_byte_table_src(
        "#[repr(C, align(64))] struct Align64<T>(T); static LEAF: Align64<[u8; 5000]> = Align64([ELEMS]); fn read_at(i: usize) -> u8 { LEAF.0[i] }",
    );
    let dir = temp_dir("large-byte-table-wrapped");
    let src = fixture(&dir, "large_byte_table_wrapped.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "wrapped large byte table must run; stderr={stderr}"
    );
    let expect_sum: u32 = (0..5000).map(|i| i % 251).sum();
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("{expect_sum} 0 {}\n", 4999 % 251),
        "decoded table values must match the source program"
    );
}

#[test]
fn run_mode_materialised_constant_shapes() {
    // Struct constants, a reference into the middle of an array
    // constant, statics read by value, a `[&str]` constant and a
    // niche-encoded `Option<&i32>` constant, all printed.
    let dir = temp_dir("const-shapes");
    let src = fixture(&dir, "const_shapes.rs", CONST_SHAPES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "materialised constants must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "-2 7 20 42 yy 5\n",
        "decoded constant values must match the source program"
    );
}

const TAIL_CALL_SRC: &str = r#"
#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]
fn countdown(n: u32) -> u32 {
    if n == 0 { 0 } else { become countdown(n - 1) }
}
fn main() { let _ = countdown(3); }
"#;

const ASYNC_CLOSURE_SRC: &str = r#"
fn main() {
    let f = async move |x: i32| x + 1;
    let _ = &f;
}
"#;

const ASYNC_CLOSURE_CALL_SRC: &str = r#"
fn main() {
    let f = async move |x: i32| x + 1;
    let _fut = f(1);
}
"#;

const COMPLETED_FUTURE_DROP_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

fn poll_once<F: Future>(fut: F) -> Poll<F::Output> {
    let mut fut = Box::pin(fut);
    let mut cx = Context::from_waker(Waker::noop());
    Pin::as_mut(&mut fut).poll(&mut cx)
}

struct Guard(&'static str);
impl Drop for Guard {
    fn drop(&mut self) {
        println!("dropped {}", self.0);
    }
}

fn main() {
    {
        let g = Guard("block-capture");
        let fut = async move {
            let _ = &g;
            5
        };
        drop(fut);
        println!("block future dropped");
    }
    let g = Guard("block-done");
    let fut = async move {
        let _ = &g;
        6
    };
    match poll_once(fut) {
        Poll::Ready(v) => println!("ready: {}", v),
        Poll::Pending => println!("pending"),
    }
    println!("end of main");
}
"#;

const LOG_SYNTAX_SRC: &str = r#"
#![feature(log_syntax)]
fn main() {
    log_syntax!("compile-time marker");
    println!("guest marker");
}
"#;


#[test]
fn run_mode_tail_call_body_is_skipped_and_call_exits_two() {
    // Converting a `become` (explicit tail call) makes rustc_public
    // panic (`unimplemented!()` in Stable MIR's terminator
    // conversion). The driver contains that panic at the compiler
    // boundary and skips the body with the named tail-call reason,
    // so conversion completes; the call in `main` then reports the
    // unconverted body under the usual run-mode contract — exit 2 —
    // instead of the whole run dying with the panic (exit 101 and
    // a default panic dump on stderr).
    let dir = temp_dir("tailcall");
    let src = fixture(&dir, "tailcall.rs", TAIL_CALL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "calling an unconverted body is unsupported (exit 2), not a process panic; stderr={stderr}"
    );
    assert!(
        stderr.contains("no converted body"),
        "the reason names the unconverted body: {stderr}"
    );
    assert!(
        stderr.contains("tailcall::countdown"),
        "the reason names the skipped function: {stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "the conversion panic is contained, not reported: {stderr}"
    );
    assert!(
        !stderr.contains("RUST_BACKTRACE"),
        "the default panic dump is suppressed, the message is reported as the reason: {stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "run mode keeps stdout clean: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}


#[test]
fn run_mode_async_closure_call_runs() {
    // Calling an async closure goes through the `AsyncFn` trait
    // dispatch: the call creates the closure's coroutine (the
    // value the compiler's shim body would construct) and the
    // program runs to completion - the created future here is
    // simply never polled. (Before Lane MJ this call was the
    // unsupported case this test pinned; the end-to-end behaviour
    // of the dispatch is covered by the driver_async_closure
    // tests.)
    let dir = temp_dir("asyncclosure");
    let src = fixture(&dir, "asyncclosure.rs", ASYNC_CLOSURE_CALL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "calling an async closure runs; stderr={stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "the program prints nothing: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}


#[test]
fn run_mode_completed_future_drops_capture_once() {
    // A completed async block's capture is dropped by the
    // coroutine body itself when it completes (the body's own
    // `Drop` terminator on the state field), and dropping the
    // completed future afterwards must not drop it a second
    // time: the body's drop leaves the state cell dead, and the
    // future's glue finds nothing left there. An unpolled future
    // still drops its capture exactly once, through its glue.
    let dir = temp_dir("completed_future_drop");
    let src = fixture(&dir, "completed_future_drop.rs", COMPLETED_FUTURE_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program runs to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "dropped block-capture\nblock future dropped\ndropped block-done\nready: 6\nend of main\n",
        "each capture drops exactly once"
    );
}


#[test]
fn run_mode_uncalled_async_closure_runs() {
    // Defining an async closure and borrowing the value never
    // calls it: the closure value is a zero-sized value whose
    // creation the converted body represents without the
    // closure's type (which Stable MIR cannot hand over), so
    // the program runs to completion.
    let dir = temp_dir("asyncclosure_uncalled");
    let src = fixture(&dir, "asyncclosure.rs", ASYNC_CLOSURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an uncalled async closure runs; stderr={stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "run mode keeps stdout clean: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}


#[test]
fn run_mode_log_syntax_output_stays_off_stdout() {
    // `log_syntax!` prints to the *compiler's* stdout at compile
    // time. The compiler runs in-process, so without redirection
    // that output would land in the driver's stdout — which run
    // mode reserves for the guest program's output. It must appear
    // on stderr (compiler-side diagnostics) instead, leaving stdout
    // byte-identical to the guest's own output.
    let dir = temp_dir("logsyntax");
    let src = fixture(&dir, "logsyntax.rs", LOG_SYNTAX_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program itself runs fine; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "guest marker\n",
        "stdout carries the guest's output alone"
    );
    assert!(
        stderr.contains("compile-time marker"),
        "the log_syntax output is redirected to stderr: {stderr}"
    );
}

const EXPLICIT_DISCRIMINANT_SRC: &str = r#"
#[repr(i32)]
enum E { A = 3, B = 7, C = -2 }
#[repr(i32)]
enum P { X(i32) = -5, Y(u32) = 42, Z = 0 }
#[repr(u8)]
enum U { Lo = 3, Hi = 200 }
fn name(e: E) -> i32 { match e { E::A => 100, E::B => 200, E::C => 300 } }
fn p_val(p: P) -> i32 { match p { P::X(v) => v, P::Y(v) => v as i32 + 1, P::Z => -1 } }
fn u_val(u: U) -> u32 { match u { U::Lo => 1, U::Hi => 2 } }
const PC: P = P::Y(9);
fn main() {
    println!("{}", name(E::A));
    println!("{}", name(E::B));
    println!("{}", name(E::C));
    println!("{}", p_val(P::X(-7)));
    println!("{}", p_val(PC));
    println!("{}", u_val(U::Hi));
    println!("{}", E::C as i32);
}
"#;

#[test]
fn run_mode_explicit_discriminant_match() {
    // Enums with explicit discriminants: `Discriminant` must yield
    // the discriminant values (3, 7, -2, ...), not the variant
    // indices, or every match falls through. The expected output
    // is what the native build of the same program prints.
    let dir = temp_dir("explicit-discriminant");
    let src = fixture(&dir, "explicit_discriminant.rs", EXPLICIT_DISCRIMINANT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "explicit-discriminant enums must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "100\n200\n300\n-7\n10\n2\n-2\n",
        "matches on explicit discriminants must pick the native branches"
    );
}

const NEGATIVE_INT_MATCH_SRC: &str = r#"
fn f(x: i32) -> i32 { match x { -2 => 300, 3 => 100, _ => 0 } }
fn main() { println!("{}", f(-2)); println!("{}", f(3)); println!("{}", f(9)); }
"#;

#[test]
fn run_mode_negative_int_match() {
    // SwitchInt branch keys are stored as zero-extended bit
    // patterns; a signed match must still hit its negative arms.
    let dir = temp_dir("negative-int-match");
    let src = fixture(&dir, "negative_int_match.rs", NEGATIVE_INT_MATCH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "signed int match must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "300\n100\n0\n",
        "the -2 arm must be taken for -2"
    );
}

// ----- auxiliary crates (--aux) ----------------------------------------------

const AUX_BASE_SRC: &str = r#"
pub fn base_val() -> i32 { 40 }
"#;

const AUX_BROKEN_SRC: &str = r#"
pub fn broken( -> i32 { 1 }
"#;

const AUX_LIB_SRC: &str = r#"
fn secret(x: i32) -> i32 { x * 3 }
pub fn add(a: i32, b: i32) -> i32 { a + b }
pub fn triple(x: i32) -> i32 { secret(x) }
pub fn generic_id<T>(x: T) -> T { x }
"#;

const AUX_MAIN_SRC: &str = r#"
extern crate auxlib;
fn main() {
    println!("add={}", auxlib::add(2, 3));
    println!("triple={}", auxlib::triple(7));
}
"#;

const AUX_MID_SRC: &str = r#"
extern crate base;
pub fn with_base() -> i32 { base::base_val() + 2 }
"#;

const AUX_NESTED_MAIN_SRC: &str = r#"
extern crate mid;
fn main() { println!("val={}", mid::with_base()); }
"#;

fn run_driver_with_aux(src: &Path, aux: &[(&str, &Path)]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
    cmd.arg("run").arg(src);
    for (name, path) in aux {
        cmd.arg("--aux").arg(format!("{}={}", name, path.display()));
    }
    cmd.output().expect("failed to run sparagmos-driver")
}

#[test]
fn run_mode_aux_compile_failure_exits_two() {
    let dir = temp_dir("aux-broken");
    let aux = fixture(&dir, "auxlib.rs", AUX_BROKEN_SRC);
    let src = fixture(&dir, "auxmain.rs", AUX_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("auxlib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an uncompilable auxiliary crate is a conversion failure; stderr={stderr}"
    );
    assert!(
        stderr.contains("auxiliary crate `auxlib` failed to compile"),
        "the reason names the auxiliary crate: {stderr}"
    );
}

#[test]
fn run_mode_aux_crate_functions_execute() {
    // Functions of an auxiliary crate -- including its private
    // helpers -- are converted into the same program and execute in
    // the interpreter; the crate's generic functions are skipped
    // without affecting the run.
    let dir = temp_dir("aux-basic");
    let aux = fixture(&dir, "auxlib.rs", AUX_LIB_SRC);
    let src = fixture(&dir, "auxmain.rs", AUX_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("auxlib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "aux calls must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "add=5\ntriple=21\n",
        "aux function results reach the guest's stdout"
    );
}

const AUX_EXPORT_SRC: &str = r#"
#[no_mangle]
pub extern "C" fn aux_export_add(a: i32, b: i32) -> i32 { a + b + 100 }

#[export_name = "renamed_export"]
pub extern "C" fn some_fn(x: i32) -> i32 { x * 3 }
"#;

const AUX_EXPORT_MAIN_SRC: &str = r#"
extern crate auxlib;

extern "C" {
    fn aux_export_add(a: i32, b: i32) -> i32;
    #[link_name = "renamed_export"]
    fn renamed(x: i32) -> i32;
}

fn main() {
    unsafe {
        println!("sum={}", aux_export_add(1, 2));
        println!("ren={}", renamed(7));
    }
}
"#;

#[test]
fn run_mode_extern_decl_binds_aux_export_by_link_symbol() {
    // An extern declaration and the auxiliary crate's
    // `#[no_mangle]` / `#[export_name]` definition share no
    // item identity, but the linker binds them by symbol; the
    // run binds them the same way and executes the definition.
    let dir = temp_dir("aux-export");
    let aux = fixture(&dir, "auxlib.rs", AUX_EXPORT_SRC);
    let src = fixture(&dir, "auxmain.rs", AUX_EXPORT_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("auxlib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "extern declarations must bind to the aux exports; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "sum=103\nren=21\n",
        "the exported definitions execute through the declarations"
    );
}

const LOCAL_EXPORT_SRC: &str = r#"
mod defs {
    #[no_mangle]
    pub extern "C" fn local_export(x: i32) -> i32 { x + 40 }
}

extern "C" {
    fn local_export(x: i32) -> i32;
}

fn main() {
    unsafe {
        println!("loc={}", local_export(2));
    }
}
"#;

#[test]
fn run_mode_extern_decl_binds_local_export_by_link_symbol() {
    // The same binding within one crate: the declaration names
    // a foreign item, the definition is an ordinary function in
    // another module; only the link symbol connects them.
    let dir = temp_dir("local-export");
    let src = fixture(&dir, "localexport.rs", LOCAL_EXPORT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the local export must serve the declaration; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "loc=42\n",
        "the local definition executes through the declaration"
    );
}

const LIBC_EXTERN_SRC: &str = r#"
use std::ffi::{c_int, c_void};

extern "C" {
    fn free(p: *mut c_void);
    fn lgamma_r(x: f64, signp: *mut c_int) -> f64;
}

fn main() {
    unsafe {
        free(std::ptr::null_mut());
        println!("free-null-ok");
        let mut sign: c_int = 0;
        let v = lgamma_r(5.0, &mut sign);
        println!("lg(5)={v:.10} sign={sign}");
        let v2 = lgamma_r(-0.5, &mut sign);
        println!("lg(-0.5)={v2:.10} sign={sign}");
    }
}
"#;

#[test]
fn run_mode_libc_extern_free_and_lgamma_r() {
    // Body-less C library declarations with self-contained
    // semantics: `free(NULL)` is a no-op, and `lgamma_r` is the
    // host libm function, sign write-back included.
    let dir = temp_dir("libc-extern");
    let src = fixture(&dir, "libcextern.rs", LIBC_EXTERN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the libc models must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "free-null-ok\nlg(5)=3.1780538303 sign=1\nlg(-0.5)=1.2655121235 sign=-1\n",
        "host libm results and signs reach the guest"
    );
}


const AUX_PM_SRC: &str = r#"
extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn answer(_input: TokenStream) -> TokenStream {
    "42".parse().unwrap()
}
"#;

const AUX_PM_MAIN_SRC: &str = r#"
extern crate pmaux;
fn main() { println!("answer={}", pmaux::answer!()); }
"#;

// `async` is an identifier only in edition 2015.
const AUX_ED2015_SRC: &str = r#"
pub fn async() -> i32 { 11 }
"#;

// `gen` is a reserved keyword only in edition 2024.
const AUX_ED2021_SRC: &str = r#"
pub fn gen() -> i32 { 7 }
"#;

const AUX_EDITIONS_MAIN_SRC: &str = r#"
extern crate auxa;
extern crate auxb;
fn main() {
    println!("a={}", auxa::r#async());
    println!("b={}", auxb::r#gen());
}
"#;

fn run_driver_with_aux_specs(
    src: &Path,
    edition: Option<&str>,
    specs: &[String],
) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
    cmd.arg("run").arg(src);
    if let Some(ed) = edition {
        cmd.arg("--edition").arg(ed);
    }
    for spec in specs {
        cmd.arg("--aux").arg(spec);
    }
    cmd.output().expect("failed to run sparagmos-driver")
}

fn aux_spec(name: &str, path: &Path, attrs: &str) -> String {
    format!("{}={}{}", name, path.display(), attrs)
}

#[test]
fn run_mode_aux_proc_macro_crate_type_attribute() {
    // A proc-macro auxiliary crate whose source declares no crate
    // type (the shape compiletest's `//@ proc-macro:` produces):
    // the `;crate-type=proc-macro` attribute builds it as a proc
    // macro, and the main crate's macro invocation runs inside the
    // compiler before the guest executes.
    let dir = temp_dir("aux-proc-macro");
    let pm = fixture(&dir, "pmaux.rs", AUX_PM_SRC);
    let src = fixture(&dir, "pmmain.rs", AUX_PM_MAIN_SRC);
    let out = run_driver_with_aux_specs(
        &src,
        None,
        &[aux_spec("pmaux", &pm, ";crate-type=proc-macro")],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "proc-macro aux must build and expand; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "answer=42\n");
}

#[test]
fn run_mode_aux_proc_macro_without_attribute_fails() {
    // The same source without the attribute is built as an rlib,
    // which rustc rejects for `#[proc_macro]` items: a conversion
    // failure (exit 2), not a silent mis-build.
    let dir = temp_dir("aux-proc-macro-plain");
    let pm = fixture(&dir, "pmaux.rs", AUX_PM_SRC);
    let src = fixture(&dir, "pmmain.rs", AUX_PM_MAIN_SRC);
    let out = run_driver_with_aux_specs(&src, None, &[aux_spec("pmaux", &pm, "")]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("auxiliary crate `pmaux` failed to compile"),
        "the reason names the auxiliary crate: {stderr}"
    );
}

#[test]
fn run_mode_aux_per_crate_editions() {
    // Two auxiliary crates with different editions, both different
    // from the main crate's edition 2024: each `;edition=` wins for
    // its own crate (compiletest builds each auxiliary with the
    // edition the auxiliary file itself declares).
    let dir = temp_dir("aux-editions");
    let auxa = fixture(&dir, "auxa.rs", AUX_ED2015_SRC);
    let auxb = fixture(&dir, "auxb.rs", AUX_ED2021_SRC);
    let src = fixture(&dir, "edmain.rs", AUX_EDITIONS_MAIN_SRC);
    let out = run_driver_with_aux_specs(
        &src,
        Some("2024"),
        &[
            aux_spec("auxa", &auxa, ";edition=2015"),
            aux_spec("auxb", &auxb, ";edition=2021"),
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "per-aux editions must apply; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "a=11\nb=7\n");
}

#[test]
fn run_mode_aux_without_edition_attribute_uses_main_edition() {
    // Without `;edition=`, the auxiliary crate is built with the
    // main edition (2024), under which `gen` is reserved: the
    // build fails as a conversion failure.
    let dir = temp_dir("aux-edition-default");
    let auxb = fixture(&dir, "auxb.rs", AUX_ED2021_SRC);
    let src = fixture(&dir, "edmain.rs", AUX_EDITIONS_MAIN_SRC);
    let out = run_driver_with_aux_specs(
        &src,
        Some("2024"),
        &[aux_spec("auxb", &auxb, "")],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("auxiliary crate `auxb` failed to compile"),
        "the reason names the auxiliary crate: {stderr}"
    );
}

#[test]
fn run_mode_aux_namespaced_binding_name() {
    // Lane CI: an `--aux` spec name is an `--extern` binding
    // string, not a crate name: `my_api::utils` is a namespaced
    // binding (compiletest's open-ns tests, enabled by the test's
    // own `-Z namespaced-crates` compile-flag). The crate builds
    // under its file stem and binds under the namespaced name;
    // passing the binding to `--crate-name` made rustc reject
    // the build ("invalid character ':' in crate name").
    let dir = temp_dir("aux-namespaced");
    let aux = fixture(
        &dir,
        "open-ns-my_api_utils.rs",
        "pub fn val() -> i32 { 7 }\n",
    );
    let src = fixture(
        &dir,
        "nsmain2.rs",
        "//@ aux-crate:my_api::utils=open-ns-my_api_utils.rs\n\
         fn main() { println!(\"v={}\", my_api::utils::val()); }\n",
    );
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("-Z")
        .arg("--rustc-arg")
        .arg("namespaced-crates")
        .arg("--aux")
        .arg(aux_spec("my_api::utils", &aux, ""))
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "namespaced aux binding must build and run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "v=7\n");
}

#[test]
fn run_mode_aux_commented_crate_name_is_not_a_declaration() {
    // Lane CI: `// #![crate_name = "my_api::core"]` at the top of
    // an auxiliary source is a comment, not a declaration. Reading
    // it as one made the driver search for an artifact under the
    // namespaced name and report "produced no linkable artifact"
    // for a build that had in fact succeeded under the file stem.
    let dir = temp_dir("aux-commented-name");
    let aux = fixture(
        &dir,
        "thing-core.rs",
        "// #![crate_name = \"my_api::core\"]\n\
         pub fn val() -> i32 { 9 }\n",
    );
    let src = fixture(
        &dir,
        "cmain.rs",
        "extern crate thing_core;\n\
         fn main() { println!(\"v={}\", thing_core::val()); }\n",
    );
    let out = run_driver_with_aux_specs(
        &src,
        None,
        &[aux_spec("thing_core", &aux, "")],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "commented crate_name must be ignored; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "v=9\n");
}

#[test]
fn run_mode_aux_build_name_binding() {
    // Lane CI round 2: a consumer can declare an auxiliary crate
    // by the name the crate's own source declares
    // (`#![crate_name = "buildname"]`), not only by the `--aux`
    // spec name (the svh crates and issue-6919 shape). Binding
    // only the spec name left the `extern crate buildname;` item
    // unresolved (E0463).
    let dir = temp_dir("aux-build-name");
    let aux = fixture(
        &dir,
        "iss-x.rs",
        "#![crate_name = \"buildname\"]\npub fn val() -> i32 { 33 }\n",
    );
    let src = fixture(
        &dir,
        "bmain.rs",
        "extern crate buildname;\n\
         fn main() { println!(\"v={}\", buildname::val()); }\n",
    );
    let out = run_driver_with_aux(&src, &[("iss_x", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "build-name declaration must bind; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "v=33\n");
}

#[test]
fn run_mode_aux_attribute_prefixed_extern_crate() {
    // Lane CI round 2: an auxiliary source can declare its
    // dependency with an attribute-prefixed item
    // (`#[macro_use] extern crate base;` — the no_link-crate
    // shape). The declaration scan skipped lines not starting
    // with `extern crate`, so the dependency was never bound
    // and the auxiliary build failed with E0463.
    let dir = temp_dir("aux-attr-extern");
    let base = fixture(&dir, "base.rs", "pub fn base_val() -> i32 { 40 }\n");
    let mid = fixture(
        &dir,
        "mid.rs",
        "#[macro_use] extern crate base;\n\
         pub fn with_base() -> i32 { base::base_val() + 2 }\n",
    );
    let src = fixture(
        &dir,
        "amain.rs",
        "extern crate mid;\n\
         fn main() { println!(\"val={}\", mid::with_base()); }\n",
    );
    let out = run_driver_with_aux(&src, &[("base", &base), ("mid", &mid)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "attribute-prefixed extern crate must declare the dependency; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "val=42\n");
}

#[test]
fn run_mode_aux_extra_filename_artifact() {
    // Lane CI round 2: an auxiliary source whose own
    // compile-flags carry `-C extra-filename=-1` (the
    // crateresolve shape) produces `lib<name>-1.rlib`; the
    // artifact search only recognised `lib<name>.rlib` and
    // reported the successful build as producing no linkable
    // artifact.
    let dir = temp_dir("aux-extra-filename");
    let aux = fixture(
        &dir,
        "xflag.rs",
        "//@ compile-flags: -C extra-filename=-1\n\
         pub fn val() -> i32 { 5 }\n",
    );
    let src = fixture(
        &dir,
        "xmain.rs",
        "extern crate xflag;\n\
         fn main() { println!(\"v={}\", xflag::val()); }\n",
    );
    let out = run_driver_with_aux(&src, &[("xflag", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "extra-filename artifact must be found; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "v=5\n");
}

#[test]
fn run_mode_aux_locator_extern_name() {
    // Lane CI round 2: a caller can declare an auxiliary crate
    // with a locator `--extern` flag (no path) in the forwarded
    // rustc arguments — tests/ui/resolve/extern-prelude.rs
    // passes `--extern Vec` for an auxiliary crate declared
    // `#![crate_name = "Vec"]` and refers to it through the
    // extern prelude with no `extern crate` item. The name
    // counts as a declaration, so the driver binds the built
    // crate under it.
    let dir = temp_dir("aux-locator");
    let aux = fixture(
        &dir,
        "vec-aux.rs",
        "#![crate_name = \"Vec\"]\npub fn new(x: i32) -> i32 { x + 1 }\n",
    );
    let src = fixture(
        &dir,
        "lmain.rs",
        "fn main() { println!(\"v={}\", Vec::new(41)); }\n",
    );
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("--extern")
        .arg("--rustc-arg")
        .arg("Vec")
        .arg("--aux")
        .arg(aux_spec("vec_aux", &aux, ""))
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "locator --extern name must bind the built crate; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "v=42\n");
}

#[test]
fn run_mode_aux_modified_binding_stays_out_of_prelude() {
    // Lane CI round 2: an `aux-crate:` binding with modifiers
    // (`priv,noprelude:somedep`) is the binding itself — the
    // driver must not ALSO bind the crate under its unmodified
    // build name, or the crate lands in the extern prelude the
    // `noprelude` modifier exists to keep it out of, and
    // tests/ui/extern-flag/multiple-opts.rs stops failing the
    // way it must. The `extern crate somedep;` item inside the
    // module still resolves through the modified binding; the
    // prelude use in `main` cannot, so compilation fails
    // (run-mode exit 2).
    let dir = temp_dir("aux-modified-binding");
    let aux = fixture(
        &dir,
        "somedep.rs",
        "pub struct S;\npub fn somefun() {}\n",
    );
    let src = fixture(
        &dir,
        "momain.rs",
        "//@ aux-crate:priv,noprelude:somedep=somedep.rs\n\
         #![warn(exported_private_dependencies)]\n\
         pub mod m {\n\
             extern crate somedep;\n\
             pub struct PublicType { pub field: somedep::S }\n\
         }\n\
         fn main() { somedep::somefun(); }\n",
    );
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("-Z")
        .arg("--rustc-arg")
        .arg("unstable-options")
        .arg("--aux")
        .arg(aux_spec("priv,noprelude:somedep", &aux, ""))
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a noprelude binding must keep the crate out of the prelude; stderr={stderr}"
    );
    assert!(
        stderr.contains("cannot find"),
        "the failure is the unresolved prelude use: {stderr}"
    );
}

#[test]
fn run_mode_nested_aux_crates_execute() {
    // Auxiliary crates are passed in dependency order: `mid` uses
    // `base`, and the main crate uses `mid`.
    let dir = temp_dir("aux-nested");
    let base = fixture(&dir, "base.rs", AUX_BASE_SRC);
    let mid = fixture(&dir, "mid.rs", AUX_MID_SRC);
    let src = fixture(&dir, "nestedmain.rs", AUX_NESTED_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("base", &base), ("mid", &mid)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested aux calls must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "val=42\n");
}

// ----- closures (Lane AU) ----------------------------------------------------

fn summarize_driver(src: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg(src)
        .output()
        .expect("failed to run sparagmos-driver")
}

const CLOSURE_CAPTURE_SRC: &str = r#"
fn main() {
    let base = 40;
    let add = |x: i32| x + base;
    println!("{}", add(2));
}
"#;

const CLOSURE_NO_CAPTURE_SRC: &str = r#"
fn main() {
    let twice = |x: i32| x * 2;
    println!("{}", twice(21));
}
"#;

const CLOSURE_FNMUT_SRC: &str = r#"
fn main() {
    let mut count = 0;
    let mut inc = || { count += 1; count };
    println!("{}", inc());
    println!("{}", inc());
}
"#;

const CLOSURE_IN_STRUCT_SRC: &str = r#"
struct Holder<F: Fn(i32) -> i32> { f: F }
fn main() {
    let k = 5;
    let h = Holder { f: |x: i32| x + k };
    println!("{}", (h.f)(7));
}
"#;

const CLOSURE_FNONCE_MOVE_OUT_SRC: &str = r#"
fn main() {
    let s = String::from("abc");
    let f = move || -> String { s };
    let out = f();
    println!("{}", out);
}
"#;

const CLOSURE_FNONCE_IN_STRUCT_SRC: &str = r#"
struct Once<F: FnOnce() -> String> { f: F }
fn main() {
    let s = String::from("xyz");
    let o = Once { f: move || s };
    let out = (o.f)();
    println!("{}", out);
}
"#;

const CLOSURE_RETURNED_SRC: &str = r#"
fn make_adder(n: i32) -> impl Fn(i32) -> i32 {
    move |x| x + n
}
fn main() {
    let add = make_adder(30);
    println!("{}", add(12));
}
"#;

const CLOSURE_NESTED_SRC: &str = r#"
fn main() {
    let outer = |x: i32| {
        let inner = |y: i32| y + 1;
        inner(x) * 2
    };
    println!("{}", outer(10));
}
"#;

#[test]
fn run_mode_closure_with_capture() {
    let dir = temp_dir("closure-capture");
    let src = fixture(&dir, "closure_capture.rs", CLOSURE_CAPTURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure capturing a local must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_closure_without_capture() {
    let dir = temp_dir("closure-no-capture");
    let src = fixture(&dir, "closure_no_capture.rs", CLOSURE_NO_CAPTURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a captureless closure must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_fnmut_closure_keeps_state() {
    let dir = temp_dir("closure-fnmut");
    let src = fixture(&dir, "closure_fnmut.rs", CLOSURE_FNMUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an FnMut closure must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "1\n2\n",
        "the captured counter must persist across calls"
    );
}

#[test]
fn run_mode_closure_stored_in_struct() {
    let dir = temp_dir("closure-in-struct");
    let src = fixture(&dir, "closure_in_struct.rs", CLOSURE_IN_STRUCT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure stored in a struct field must be callable; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "12\n");
}

#[test]
fn run_mode_fnonce_closure_moves_capture_out() {
    let dir = temp_dir("closure-fnonce-move-out");
    let src = fixture(&dir, "closure_fnonce_move_out.rs", CLOSURE_FNONCE_MOVE_OUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an FnOnce closure moving its capture out must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "abc\n");
}

#[test]
fn run_mode_fnonce_closure_in_struct_field() {
    let dir = temp_dir("closure-fnonce-in-struct");
    let src = fixture(
        &dir,
        "closure_fnonce_in_struct.rs",
        CLOSURE_FNONCE_IN_STRUCT_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an FnOnce closure stored in a struct field must be callable; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "xyz\n");
}

#[test]
fn run_mode_closure_returned_from_function() {
    let dir = temp_dir("closure-returned");
    let src = fixture(&dir, "closure_returned.rs", CLOSURE_RETURNED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure returned from a function must be callable; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_nested_closures() {
    let dir = temp_dir("closure-nested");
    let src = fixture(&dir, "closure_nested.rs", CLOSURE_NESTED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested closures must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "22\n");
}

const CLOSURE_GENERIC_CONTEXT_SRC: &str = r#"
fn wrap<T: Copy>(v: T) -> T {
    let id = |x: T| x;
    id(v)
}
fn main() { println!("{}", wrap::<i32>(7)); }
"#;

const CLOSURE_FNONCE_SRC: &str = r#"
struct Bag(i32);
fn main() {
    let bag = Bag(41);
    let f = move || bag;
    let _b = f();
}
"#;

#[test]
fn summary_mode_generic_context_closure_is_converted_when_demanded() {
    // A closure inside a generic function has no instance as an
    // item, but the monomorphic instance of the enclosing function
    // (`wrap::<i32>` here) mentions it with concrete arguments, and
    // the closure collection converts the demanded instance.
    let dir = temp_dir("closure-generic");
    let src = fixture(&dir, "closure_generic.rs", CLOSURE_GENERIC_CONTEXT_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("{closure#0}"))
        .expect("summary lists the closure item");
    assert!(
        line.contains("[converted]"),
        "the demanded generic-context closure instance is converted: {line}"
    );
}

const CLOSURE_GENERIC_UNDEMANDED_SRC: &str = r#"
fn wrap<T: Copy>(v: T) -> T {
    let id = |x: T| x;
    id(v)
}
fn main() { println!("hi"); }
"#;

#[test]
fn summary_mode_undemanded_generic_context_closure_is_not_demanded() {
    // The same closure with no monomorphic caller anywhere is
    // accounted not demanded, with the reason naming the generic
    // context: nothing reaches it, so it is not part of the
    // program, but the record stays (it is not counted converted,
    // and it does not count as a conversion failure either).
    let dir = temp_dir("closure-generic-undemanded");
    let src = fixture(
        &dir,
        "closure_generic_undemanded.rs",
        CLOSURE_GENERIC_UNDEMANDED_SRC,
    );
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("{closure#0}"))
        .expect("summary lists the closure item");
    assert!(
        line.contains("NOT DEMANDED") && line.contains("generic context"),
        "the undemanded closure is not demanded, with the generic-context reason: {line}"
    );
}

const COROUTINE_DEMANDED_SRC: &str = r#"
fn main() {
    let _a = async { 1 };
    let _b = async move { 2 };
}
"#;

#[test]
fn summary_mode_demanded_async_block_is_converted() {
    // A coroutine item (an async block) has no monomorphic
    // instance as an item, but the converted body that creates it
    // mentions it with concrete arguments, and the coroutine
    // collection converts the demanded instance.
    let dir = temp_dir("coroutine-demanded");
    let src = fixture(&dir, "coroutine_demanded.rs", COROUTINE_DEMANDED_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    for name in ["{closure#0}", "{closure#1}"] {
        let line = stdout
            .lines()
            .find(|l| l.contains(name))
            .expect("summary lists the coroutine item");
        assert!(
            line.contains("[converted]"),
            "the demanded coroutine instance is converted: {line}"
        );
    }
}

const COROUTINE_UNDEMANDED_SRC: &str = r#"
fn wrap<T: Copy>(v: T) -> T {
    let _f = async move { v };
    v
}
fn main() { println!("hi"); }
"#;

#[test]
fn summary_mode_undemanded_coroutine_is_not_demanded() {
    // A coroutine inside a generic function that is never
    // instantiated has no concrete demand anywhere and is
    // accounted not demanded: an undemanded generic is never
    // counted converted, and its absence is not a conversion
    // failure either.
    let dir = temp_dir("coroutine-undemanded");
    let src = fixture(&dir, "coroutine_undemanded.rs", COROUTINE_UNDEMANDED_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("{closure#0}"))
        .expect("summary lists the coroutine item");
    assert!(
        line.contains("NOT DEMANDED") && line.contains("no monomorphic instance"),
        "the undemanded coroutine is not demanded: {line}"
    );
}

const COROUTINE_ONLY_CALLER_SRC: &str = r#"
fn helper<T: Copy>(v: T) -> T { v }
fn main() {
    let _f = async { helper::<i32>(3) };
}
"#;

#[test]
fn summary_mode_generic_fn_called_only_from_coroutine_body_is_converted() {
    // The only call to the generic function lives in the
    // coroutine's body. Converting the coroutine must feed the
    // instance demand too: every converted body is scanned by
    // every demand pass.
    let dir = temp_dir("coroutine-only-caller");
    let src = fixture(&dir, "coroutine_only_caller.rs", COROUTINE_ONLY_CALLER_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("helper"))
        .expect("summary lists the generic function item");
    assert!(
        line.contains("[converted]"),
        "the generic function demanded only by the coroutine body is converted: {line}"
    );
}

#[test]
fn run_mode_generic_context_closure_runs() {
    let dir = temp_dir("closure-generic-run");
    let src = fixture(&dir, "closure_generic.rs", CLOSURE_GENERIC_CONTEXT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure inside a generic function must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "7\n");
}

const CLOSURE_GENERIC_TWO_TYPES_SRC: &str = r#"
fn twice<T: Copy + std::ops::Add<Output = T>>(v: T) -> T {
    let f = |x: T| x + x;
    f(v)
}
fn main() {
    println!("{}", twice::<i32>(21));
    println!("{}", twice::<f64>(1.5));
}
"#;

#[test]
fn run_mode_generic_closure_two_instantiations() {
    // Two monomorphic instances of one closure definition coexist
    // in the program and each call reaches its own body.
    let dir = temp_dir("closure-generic-two-types");
    let src = fixture(
        &dir,
        "closure_generic_two_types.rs",
        CLOSURE_GENERIC_TWO_TYPES_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "both instantiations of the generic-context closure must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n3\n");
}

const GENERIC_HOF_FNONCE_SRC: &str = r#"
fn apply<F: FnOnce() -> i32>(f: F) -> i32 { f() }
fn main() {
    let base = 40;
    let r = apply(|| base + 2);
    println!("{r}");
}
"#;

#[test]
fn run_mode_generic_hof_fnonce_with_closure() {
    // A generic higher-order function applied to a closure: the
    // `FnOnce::call_once` call inside the generic body resolves to
    // the closure instance the caller monomorphized it with.
    let dir = temp_dir("generic-hof-fnonce");
    let src = fixture(&dir, "generic_hof_fnonce.rs", GENERIC_HOF_FNONCE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a generic HOF applied to a closure must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn summary_mode_fnonce_closure_body_is_converted() {
    // Conversion covers FnOnce closures too: the body is collected
    // even though the interpreter's by-value environment passing is
    // a separate concern.
    let dir = temp_dir("closure-fnonce");
    let src = fixture(&dir, "closure_fnonce.rs", CLOSURE_FNONCE_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("{closure#0}"))
        .expect("summary lists the closure item");
    assert!(
        line.contains("[converted]"),
        "the FnOnce closure body is converted: {line}"
    );
}

// ----- explicit mem::drop (Issue #61) ------------------------------------------

const MEM_DROP_SRC: &str = r#"
struct D(i32);
impl Drop for D {
    fn drop(&mut self) { println!("drop {}", self.0); }
}
fn main() {
    let a = D(1);
    let b = D(2);
    std::mem::drop(a);
    println!("middle");
}
"#;

const CONST_BOUND_MATCH_SRC: &str = r#"
struct Conn;
impl Drop for Conn {
    fn drop(&mut self) { println!("dropped"); }
}
struct Wrap(Conn);
impl ::std::ops::Deref for Wrap {
    type Target = Conn;
    fn deref(&self) -> &Conn { &self.0 }
}
fn inner() {
    let conn = &*match Some(Wrap(Conn)) {
        Some(val) => val,
        None => return,
    };
    let _ = conn;
    println!("inner end");
}
fn main() {
    inner();
    println!("main end");
}
"#;

#[test]
fn run_mode_const_bound_match_binding_is_initialized() {
    // The scrutinee is a constant aggregate, so optimized MIR
    // const-propagates the by-value match binding away: the
    // binding local has no assignment anywhere in the body and
    // its value survives only as a constant debug-info entry.
    // The driver restores the definition at conversion; without
    // it the local is met uninitialized and its scope-end Drop
    // fails with "unsupported terminator: Drop"
    // (tests/ui/mir/mir_early_return_scope.rs).
    let dir = temp_dir("constbound");
    let src = fixture(&dir, "constbound.rs", CONST_BOUND_MATCH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "const-bound match binding program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "inner end\ndropped\nmain end\n",
        "the binding's destructor runs at scope end, after the body"
    );
}

#[test]
fn run_mode_mem_drop_runs_destructors() {
    // An explicit std::mem::drop runs the same destructor as a
    // scope-end drop: the explicitly dropped value's destructor
    // runs at the call, and the remaining local drops afterwards,
    // at scope end — the native interleaving, byte for byte. (The
    // container cases are covered at the interpreter level in
    // memdrop_glue_tests; a Vec of droppable elements additionally
    // needs the runtime allocation path.)
    let dir = temp_dir("memdrop");
    let src = fixture(&dir, "memdrop.rs", MEM_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "mem::drop program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "drop 1\nmiddle\ndrop 2\n",
        "explicit and scope-end drops interleave exactly as natively"
    );
}

// ----- Box pointee move-out (Lane MQ) ----------------------------------------

const BOX_MOVE_OUT_ZST_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
fn main() {
    let b = Box::new(DropMe);
    let _d = *b;
    println!("done");
}
"#;

#[test]
fn run_mode_box_pointee_move_out_drops_once_zst() {
    // Moving the pointee out of a box leaves the box an empty
    // shell: the moved value's destructor runs once (at its own
    // scope end), and the box's own drop — which drop elaboration
    // emits as the `Drop::drop` method call — only releases the
    // allocation. The pointee is a ZST here, so the move itself
    // leaves no statement in the converted body at all.
    let dir = temp_dir("boxmovezst");
    let src = fixture(&dir, "boxmovezst.rs", BOX_MOVE_OUT_ZST_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "box move-out program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "done\ndrop\n",
        "the moved-out pointee drops exactly once"
    );
}

const BOX_MOVE_OUT_SCALAR_SRC: &str = r#"
struct DropMe(u32);
impl Drop for DropMe { fn drop(&mut self) { println!("drop {}", self.0); } }
fn main() {
    let b = Box::new(DropMe(7));
    let _d = *b;
    println!("done");
}
"#;

#[test]
fn run_mode_box_pointee_move_out_drops_once_scalar() {
    // The non-ZST form of the move-out: the pointee load is a
    // shaped load converted from a move, which clears the
    // pointee's slot in the box's allocation, and the box's
    // method-call drop is the shell step only.
    let dir = temp_dir("boxmovescalar");
    let src = fixture(&dir, "boxmovescalar.rs", BOX_MOVE_OUT_SCALAR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "box move-out program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "done\ndrop 7\n",
        "the moved-out pointee drops exactly once, with its value intact"
    );
}

const BOX_MOVE_OUT_SCOPE_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
fn main() {
    let _d;
    {
        let b = Box::new(DropMe);
        _d = *b;
    }
    println!("mid");
}
"#;

#[test]
fn run_mode_box_pointee_move_out_scope_order() {
    // The emptied box drops at the inner scope's end, silently;
    // the moved-out value drops at the outer scope's end.
    let dir = temp_dir("boxmovescope");
    let src = fixture(&dir, "boxmovescope.rs", BOX_MOVE_OUT_SCOPE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "box move-out program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "mid\ndrop\n",
        "the empty box drops silently before the moved value drops"
    );
}

const BOX_PARTIAL_MOVE_OUT_SRC: &str = r#"
struct A(u32);
impl Drop for A { fn drop(&mut self) { println!("drop A {}", self.0); } }
struct B(u32);
impl Drop for B { fn drop(&mut self) { println!("drop B {}", self.0); } }
struct P { a: A, b: B }
fn main() {
    let bx = Box::new(P { a: A(1), b: B(2) });
    let _a = bx.a;
    println!("end");
}
"#;

#[test]
fn run_mode_box_pointee_partial_move_out_drops_rest_once() {
    // A field moved out of the pointee drops at its own scope
    // end; the field that stayed is dropped by the elaborated
    // field drop, and the box's method-call drop is the shell
    // step only — neither field's destructor runs twice.
    let dir = temp_dir("boxpartialmove");
    let src = fixture(&dir, "boxpartialmove.rs", BOX_PARTIAL_MOVE_OUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "partial move-out program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "end\ndrop A 1\ndrop B 2\n",
        "each field drops exactly once"
    );
}

const BOX_NO_MOVE_OUT_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
fn main() {
    let _b = Box::new(DropMe);
    println!("done");
}
"#;

#[test]
fn run_mode_box_drop_without_move_out_still_drops_pointee() {
    // The plain scope-end drop of a box (the `Drop` terminator
    // form, no move-out anywhere) still runs the pointee's
    // destructor exactly once: the shell-only glue applies to
    // the method-call form only.
    let dir = temp_dir("boxnomove");
    let src = fixture(&dir, "boxnomove.rs", BOX_NO_MOVE_OUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "plain box program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "done\ndrop\n",
        "an unmoved box still drops its pointee"
    );
}


// ----- Box<dyn Fn> drop timing (Lane EV) --------------------------------------

const BOX_DYN_FN_WHOLE_CAPTURE_SRC: &str = r#"
struct M(i32, i32);
impl M { fn get(&self) -> i32 { self.0 } }
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let m = M(7, 8);
    let f: Box<dyn Fn(i32) -> i32> = Box::new(move |x| x + m.get());
    println!("call {}", f(1));
    drop(f);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_fn_drop_runs_captured_dtor_at_drop() {
    // The closure owns the captured `M` (the body uses the whole
    // place through a method call, so edition-2021 disjoint capture
    // still captures all of `m`): dropping the box at `drop(f)`
    // runs the capture's destructor there, before the next
    // statement — the native interleaving, byte for byte.
    let dir = temp_dir("box-dyn-fn-drop");
    let src = fixture(&dir, "box_dyn_fn_drop.rs", BOX_DYN_FN_WHOLE_CAPTURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<dyn Fn> drop program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "call 8\ndtor 7\nafter drop\n",
        "the captured value's destructor runs at the explicit drop"
    );
}

const BOX_DYN_FN_FIELD_CAPTURE_SRC: &str = r#"
struct M(i32, i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let m = M(7, 8);
    let f: Box<dyn Fn(i32) -> i32> = Box::new(move |x| x + m.0);
    println!("call {}", f(1));
    drop(f);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_fn_field_capture_dtor_stays_with_captured_place() {
    // The same source means different captures in different
    // editions, and the destructor follows the capture: in edition
    // 2021 the closure captures only the place `m.0` (an `i32`
    // copy), so `M` itself stays in `main` and its destructor runs
    // at scope end, after the explicit drop of the box — exactly
    // the native 2021 output. In edition 2015 the closure captures
    // all of `m`, so the destructor runs at `drop(f)`. Both
    // interleavings are pinned so a change in capture or drop
    // semantics cannot silently move the destructor.
    let dir = temp_dir("box-dyn-fn-field-capture");
    let src = fixture(
        &dir,
        "box_dyn_fn_field_capture.rs",
        BOX_DYN_FN_FIELD_CAPTURE_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "field-capture program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "call 8\nafter drop\ndtor 7\n",
        "edition 2021: only m.0 was captured, so M's destructor stays at scope end"
    );
    let out = run_driver_with_aux_specs(&src, Some("2015"), &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "field-capture program must terminate normally in edition 2015; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "call 8\ndtor 7\nafter drop\n",
        "edition 2015: the whole M was captured, so its destructor runs at drop(f)"
    );
}

const BOX_DYN_TRAIT_PAYLOAD_SRC: &str = r#"
trait G { fn get(&self) -> i32; }
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
struct S { m: M }
impl G for S { fn get(&self) -> i32 { self.m.0 } }
fn main() {
    let b: Box<dyn G> = Box::new(S { m: M(9) });
    println!("get {}", b.get());
    drop(b);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_trait_drop_runs_payload_dtor_at_drop() {
    // A `Box<dyn Trait>` whose payload is a plain ADT: the
    // payload's field destructor runs at the explicit drop, as
    // with the closure case.
    let dir = temp_dir("box-dyn-trait-drop");
    let src = fixture(&dir, "box_dyn_trait_drop.rs", BOX_DYN_TRAIT_PAYLOAD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<dyn Trait> drop program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "get 9\ndtor 9\nafter drop\n",
        "the payload's destructor runs at the explicit drop"
    );
}

const BOX_DYN_FNONCE_SRC: &str = r#"
struct M(i32, i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let m = M(7, 8);
    let f: Box<dyn FnOnce(i32) -> i32> = Box::new(move |x| { let mm = m; x + mm.0 });
    println!("made");
    drop(f);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_fnonce_uncalled_drop_runs_captured_dtor() {
    // An uncalled `Box<dyn FnOnce>` still owns its capture:
    // dropping the box runs the captured value's destructor at
    // the drop, even though the closure body (which would have
    // moved the capture out) never ran.
    let dir = temp_dir("box-dyn-fnonce-drop");
    let src = fixture(&dir, "box_dyn_fnonce_drop.rs", BOX_DYN_FNONCE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<dyn FnOnce> drop program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "made\ndtor 7\nafter drop\n",
        "the uncalled FnOnce capture's destructor runs at the explicit drop"
    );
}

const BOX_DYN_FN_INNER_SCOPE_SRC: &str = r#"
struct M(i32, i32);
impl M { fn get(&self) -> i32 { self.0 } }
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let m = M(7, 8);
    let f: Box<dyn Fn(i32) -> i32> = Box::new(move |x| x + m.get());
    println!("call {}", f(1));
    { let _g = f; println!("in inner"); }
    println!("after inner");
}
"#;

#[test]
fn run_mode_box_dyn_fn_moved_into_inner_scope_drops_at_inner_end() {
    // Moving the box into an inner scope moves the capture's
    // destructor with it: it runs at the inner scope's end (the
    // `Drop` terminator for the moved-to local), not at the
    // outer function's end.
    let dir = temp_dir("box-dyn-fn-inner-scope");
    let src = fixture(&dir, "box_dyn_fn_inner_scope.rs", BOX_DYN_FN_INNER_SCOPE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "inner-scope program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "call 8\nin inner\ndtor 7\nafter inner\n",
        "the destructor runs at the inner scope's end"
    );
}

const SIZED_CLOSURE_DROP_SRC: &str = r#"
struct M(i32, i32);
impl M { fn get(&self) -> i32 { self.0 } }
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let m = M(7, 8);
    let c = move |x| x + m.get();
    println!("call {}", c(1));
    drop(c);
    println!("after drop c");
}
"#;

#[test]
fn run_mode_sized_closure_drop_runs_captured_dtor_at_drop() {
    // The sized counterpart of the `Box<dyn Fn>` case: an explicit
    // drop of the closure value itself runs its capture's
    // destructor at the drop.
    let dir = temp_dir("sized-closure-drop");
    let src = fixture(&dir, "sized_closure_drop.rs", SIZED_CLOSURE_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "sized closure drop program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "call 8\ndtor 7\nafter drop c\n",
        "the sized closure's captured destructor runs at the explicit drop"
    );
}

// ----- generic function definitions (Lane BA) --------------------------------

const GENERIC_CALLS_SRC: &str = r#"
fn id<T>(x: T) -> T { x }
fn first<T>(v: &[T]) -> &T { &v[0] }
struct Wrap<T>(T);
impl<T> Wrap<T> {
    fn get(&self) -> &T { &self.0 }
}
fn main() {
    let a = id::<u32>(3);
    let b = id::<i64>(-4);
    let w = Wrap::<u32>(9);
    let arr = [10u32, 20, 30];
    let c = *first(&arr);
    println!("{}", a as i64 + b + *w.get() as i64 + c as i64);
}
"#;

#[test]
fn run_mode_generic_function_instances_execute() {
    // Generic definitions have no instance as items; the instances
    // their call sites demand are resolved and converted, so a
    // program calling generic functions runs end to end.
    let dir = temp_dir("generic-calls");
    let src = fixture(&dir, "generic_calls.rs", GENERIC_CALLS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "generic calls must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "18\n");
}

const GENERIC_UNUSED_SRC: &str = r#"
fn used<T>(x: T) -> T { x }
fn unused<T: Clone>(x: T) -> T { x.clone() }
fn main() {
    let _ = used::<u32>(1);
}
"#;

#[test]
fn summary_mode_demanded_generic_instances_are_converted() {
    // The demanded definition is reported converted; a definition
    // nothing demands keeps its no-instance record as not
    // demanded: it is not silently counted as converted, and it
    // does not make the target crate's conversion incomplete.
    let dir = temp_dir("generic-unused");
    let src = fixture(&dir, "generic_unused.rs", GENERIC_UNUSED_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let used = stdout
        .lines()
        .find(|l| l.starts_with("fn generic_unused::used:"))
        .expect("summary lists the used generic definition");
    assert!(
        used.contains("[converted]"),
        "the demanded generic definition is converted: {used}"
    );
    let unused = stdout
        .lines()
        .find(|l| l.starts_with("fn generic_unused::unused:"))
        .expect("summary lists the unused generic definition");
    assert!(
        unused.contains("NOT DEMANDED (no monomorphic instance available (kind=Fn)"),
        "the undemanded generic definition is explicitly not demanded: {unused}"
    );
    assert!(
        stdout.contains("target-crate converted: 2, skipped: 0"),
        "the target crate is complete with the undemanded definition aside: {stdout}"
    );
    assert!(
        stdout.contains("target-crate not demanded: 1"),
        "the undemanded definition is counted separately: {stdout}"
    );
}

const GENERIC_CONST_SRC: &str = r#"
struct Foo<const N: usize>;
trait Bar { const VAL: usize; }
impl<const N: usize> Bar for Foo<N> { const VAL: usize = N; }
fn arr<const N: usize>() -> [u8; N] { [0; N] }
fn main() { let _ = <Foo<3> as Bar>::VAL; let _ = arr::<2>(); }
"#;

#[test]
fn summary_mode_generic_const_definition_is_not_demanded() {
    // A generic constant definition (here an associated constant
    // of a generic impl) has no instance as an item, and its own
    // body still mentions its const parameter, so converting the
    // definition body fails on `ConstantKind::Param`. That is the
    // generic definition itself, not a monomorphic body: the
    // monomorphized bodies that use the constant carry its
    // evaluated value, so the definition is accounted not
    // demanded — it does not make the target crate's conversion
    // incomplete, and it is not silently counted as converted.
    let dir = temp_dir("generic-const");
    let src = fixture(&dir, "generic_const.rs", GENERIC_CONST_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let val = stdout
        .lines()
        .find(|l| l.starts_with("fn <generic_const::Foo<N> as generic_const::Bar>::VAL:"))
        .expect("summary lists the generic associated constant");
    assert!(
        val.contains("NOT DEMANDED") && val.contains("generic constant definition"),
        "the generic constant definition is explicitly not demanded: {val}"
    );
    assert!(
        stdout.contains("target-crate converted: 4, skipped: 0"),
        "the target crate is complete with the generic constant aside: {stdout}"
    );
    assert!(
        stdout.contains("target-crate not demanded: 1"),
        "the generic constant is counted separately: {stdout}"
    );
}

const TAIL_CALL_GENERIC_SRC: &str = r#"
#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]
fn countdown<T: Copy>(n: u32, m: T) -> u32 {
    if n == 0 { 0 } else { become countdown(n - 1, m) }
}
fn main() { let _ = countdown::<u32>(3, 0); }
"#;

#[test]
fn summary_mode_demanded_generic_with_unconvertible_body_stays_skipped() {
    // Demand does not excuse a body that cannot be converted: the
    // generic definition is demanded concretely from `main`, its
    // instance fails conversion with the named tail-call reason,
    // and the definition stays skipped — the target crate stays
    // incomplete for that named reason alone.
    let dir = temp_dir("generic-tailcall");
    let src = fixture(&dir, "generic_tailcall.rs", TAIL_CALL_GENERIC_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.starts_with("fn generic_tailcall::countdown:"))
        .expect("summary lists the generic definition");
    assert!(
        line.contains("SKIPPED") && line.contains("tail call"),
        "the demanded-but-unconvertible definition stays skipped with its reason: {line}"
    );
    assert!(
        stdout.contains("target-crate converted: 1, skipped: 1"),
        "the target crate stays incomplete for the demanded failure: {stdout}"
    );
}

const GENERIC_DROP_CREDIT_SRC: &str = r#"
struct Guard<T> { _v: T }
impl<T> Drop for Guard<T> { fn drop(&mut self) {} }
struct Holder { _g: Guard<u32> }
fn main() { let _h = Holder { _g: Guard { _v: 1 } }; }
"#;

#[test]
fn summary_mode_generic_drop_impl_is_credited_from_drop_demand() {
    // A generic `Drop` impl is demanded only through drop glue
    // (never a call edge): the drop demand converts the concrete
    // instance, and that conversion is credited to the impl
    // definition, whose own entry is reported converted instead
    // of standing as an unmet no-instance record. (That the
    // demanded destructor body actually runs is pinned by the
    // Lane BS run-mode tests below.)
    let dir = temp_dir("generic-drop-credit");
    let src = fixture(&dir, "generic_drop_credit.rs", GENERIC_DROP_CREDIT_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.starts_with("fn <generic_drop_credit::Guard<T> as std::ops::Drop>::drop:"))
        .expect("summary lists the generic Drop impl");
    assert!(
        line.contains("[converted]"),
        "the drop-demanded generic Drop impl is converted: {line}"
    );
    assert!(
        stdout.contains("target-crate converted: 2, skipped: 0"),
        "no target-crate function stays skipped: {stdout}"
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

// ----- Vec allocation paths (Lane BO) -----

const VEC_DROPPABLE_SRC: &str = r#"
struct D(i32);
impl Drop for D {
    fn drop(&mut self) {
        println!("drop {}", self.0);
    }
}
fn main() {
    let v = vec![D(3), D(4)];
    println!("len {}", v.len());
    drop(v);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_macro_of_droppables() {
    // The `vec![...]` list form builds the vector through
    // `Box::new_uninit` + `box_assume_init_into_vec_unsafe`, whose
    // real bodies bottom out in the runtime allocator
    // (`Global::alloc_impl_runtime` has no body). The interpreter
    // models those entry points, so a Vec of droppable elements
    // runs end to end and the element destructors run in order,
    // byte for byte as natively.
    let dir = temp_dir("vecdroppable");
    let src = fixture(&dir, "vecdroppable.rs", VEC_DROPPABLE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec! of droppables must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 2\ndrop 3\ndrop 4\ndone\n",
        "vec! elements drop in order, exactly as natively"
    );
}

const VEC_ZST_ENUM_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) {} }
enum CS<Y, R> { Yielded(Y), Complete(R) }
enum Never {}
fn read_it(v: Vec<CS<Never, DropMe>>) {
    match &v[0] { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
    match &v[1] { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
    println!("done");
}
fn main() {
    read_it(vec![CS::Complete(DropMe), CS::Complete(DropMe)]);
}
"#;

#[test]
fn run_mode_vec_of_zero_sized_enum_elements_read_by_index() {
    // A `Vec` of a zero-sized enum (one variant uninhabited, the
    // other's payload zero-sized) built by `vec!`: MIR elides the
    // element store entirely, so the elements reach execution as
    // uninitialized slots, and an index read of an element denotes
    // the element type's single value (the canonical-value rescue
    // follows the projection's index step). Both elements read
    // back as the `Complete` variant, exactly as natively.
    let dir = temp_dir("veczstenum");
    let src = fixture(&dir, "veczstenum.rs", VEC_ZST_ENUM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec! of zero-sized enum elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\nc\ndone\n",
        "zero-sized enum elements read by index denote their single value"
    );
}

const VEC_ZST_ENUM_GENERIC_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) {} }
enum CS<Y, R> { Yielded(Y), Complete(R) }
trait Tr { type Out; }
struct S;
impl Tr for S { type Out = DropMe; }
enum Never {}
fn read_it<T: Tr>(v: Vec<CS<Never, T::Out>>) {
    match &v[0] { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
    println!("done");
}
fn main() {
    read_it::<S>(vec![CS::Complete(DropMe)]);
}
"#;

#[test]
fn run_mode_vec_of_zero_sized_enum_elements_generic_projection() {
    // The same element read with the payload type arriving
    // through a trait projection (`T::Out`): the element type is
    // still single-valued after instantiation, so the index read
    // is rescued the same way.
    let dir = temp_dir("veczstenumgeneric");
    let src = fixture(&dir, "veczstenumgeneric.rs", VEC_ZST_ENUM_GENERIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "generic vec! of zero-sized enum elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\ndone\n",
        "projected zero-sized enum elements read by index denote their single value"
    );
}

const VEC_GROWTH_SRC: &str = r#"
struct D(i32);
impl Drop for D {
    fn drop(&mut self) {
        println!("drop {}", self.0);
    }
}
fn main() {
    let mut v: Vec<D> = Vec::with_capacity(4);
    v.reserve(8);
    v.push(D(1));
    v.push(D(2));
    println!("len {}", v.len());
}
"#;

#[test]
fn run_mode_vec_with_capacity_push_and_reserve() {
    // `with_capacity` and `reserve` are allocation entry points:
    // the abstract Vec model does not represent capacity, so both
    // are modelled (an empty vector / a no-op) and pushes onto the
    // reserved vector behave as natively, including the scope-end
    // drop of the elements.
    let dir = temp_dir("vecgrowth");
    let src = fixture(&dir, "vecgrowth.rs", VEC_GROWTH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "with_capacity/reserve/push must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 2\ndrop 1\ndrop 2\n",
        "reserved Vec pushes and drops exactly as natively"
    );
}


const VEC_RESIZE_GROW_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3];
    v.resize(6, 9);
    println!("{:?} len={} cap={}", v, v.len(), v.capacity());
    let mut w: Vec<i32> = Vec::new();
    w.resize(4, 7);
    println!("{:?} len={} cap={}", w, w.len(), w.capacity());
}
"#;

#[test]
fn run_mode_vec_resize_grows_with_fill() {
    // `Vec::resize` growth runs through `extend_with`, whose
    // `SetLenOnDrop` guard holds a `&mut` to the vector's length
    // slot and writes the filled length back through it on drop:
    // the write must land on the vector itself (it used to fail
    // with "expected mutable place, found constant aggregate"
    // because the length-slot reference denoted a detached value).
    let dir = temp_dir("vecresizegrow");
    let src = fixture(&dir, "vecresizegrow.rs", VEC_RESIZE_GROW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec resize growth must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 2, 3, 9, 9, 9] len=6 cap=6\n[7, 7, 7, 7] len=4 cap=4\n",
        "grown vectors hold the fill values with native len/cap"
    );
}

const VEC_RESIZE_REGROW_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3, 4, 5];
    v.resize(2, 0);
    println!("{:?} len={} cap={}", v, v.len(), v.capacity());
    v.resize(3, 42);
    println!("{:?} len={} cap={}", v, v.len(), v.capacity());
}
"#;

#[test]
fn run_mode_vec_resize_regrows_within_capacity() {
    // Shrinking then growing back inside the reserved capacity
    // takes the same length-slot write path with no reservation
    // in between: the regrown slot is filled, the length and the
    // (unchanged) capacity answer natively.
    let dir = temp_dir("vecresizeregrow");
    let src = fixture(&dir, "vecresizeregrow.rs", VEC_RESIZE_REGROW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "shrink-then-regrow must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 2] len=2 cap=5\n[1, 2, 42] len=3 cap=5\n",
        "regrowth within capacity fills and reports natively"
    );
}

const VEC_RESIZE_PAYLOAD_SRC: &str = r#"
#[derive(Debug, Clone)]
struct P {
    a: i32,
    b: i32,
}
fn main() {
    let mut s = vec!["a".to_string()];
    s.resize(4, "x".to_string());
    println!("{:?} len={}", s, s.len());
    let mut p = vec![P { a: 1, b: 2 }];
    p.resize(3, P { a: 7, b: 8 });
    println!("{:?} len={}", p, p.len());
}
"#;

#[test]
fn run_mode_vec_resize_grows_clone_payloads() {
    // The fill value is cloned per new slot: heap (`String`) and
    // aggregate (struct) payloads grow through the same path and
    // read back exactly.
    let dir = temp_dir("vecresizepayload");
    let src = fixture(&dir, "vecresizepayload.rs", VEC_RESIZE_PAYLOAD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "resize of payload vectors must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[\"a\", \"x\", \"x\", \"x\"] len=4\n[P { a: 1, b: 2 }, P { a: 7, b: 8 }, P { a: 7, b: 8 }] len=3\n",
        "cloned fill payloads read back exactly"
    );
}

const VEC_RESIZE_PANIC_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};
static CLONES: AtomicUsize = AtomicUsize::new(0);
#[derive(Debug)]
struct Bomb(i32);
impl Clone for Bomb {
    fn clone(&self) -> Self {
        let n = CLONES.fetch_add(1, Ordering::SeqCst);
        if n >= 1 {
            panic!("clone boom");
        }
        Bomb(self.0)
    }
}
impl Drop for Bomb {
    fn drop(&mut self) {
        println!("drop {}", self.0);
    }
}
fn main() {
    let mut v = vec![Bomb(1)];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        v.resize(5, Bomb(9));
    }));
    println!("caught={} len={}", r.is_err(), v.len());
    println!(
        "vals: {}",
        v.iter().map(|b| b.0.to_string()).collect::<Vec<_>>().join(",")
    );
}
"#;

#[test]
fn run_mode_vec_resize_clone_panic_sets_partial_len() {
    // When a fill clone panics mid-growth, the `SetLenOnDrop`
    // guard still runs during the unwind and publishes the
    // partial length: the elements written so far stay in the
    // vector (and drop at scope end), the unwritten slots do
    // not appear.
    let dir = temp_dir("vecresizepanic");
    let src = fixture(&dir, "vecresizepanic.rs", VEC_RESIZE_PANIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a caught clone panic must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "drop 9\ncaught=true len=2\nvals: 1,9\ndrop 1\ndrop 9\n",
        "the guard publishes the partial length on unwind"
    );
}

const VEC_RESIZE_SPARE_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::with_capacity(100);
    v.push(1);
    v.push(2);
    v.resize(50, 7);
    println!("len={} cap={} sum={}", v.len(), v.capacity(), v.iter().sum::<i32>());
    v.resize(10, 0);
    println!("len={} cap={}", v.len(), v.capacity());
    v.resize(60, 3);
    println!("len={} cap={} sum={}", v.len(), v.capacity(), v.iter().sum::<i32>());
}
"#;

#[test]
fn run_mode_vec_resize_uses_spare_capacity() {
    // Growth into a large spare capacity, a shrink, and a
    // regrowth past the previous length all stay inside the
    // original allocation: capacity never moves and the sums
    // cover exactly the declared elements.
    let dir = temp_dir("vecresizespare");
    let src = fixture(&dir, "vecresizespare.rs", VEC_RESIZE_SPARE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "resize within spare capacity must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len=50 cap=100 sum=339\nlen=10 cap=100\nlen=60 cap=100 sum=209\n",
        "spare-capacity growth/shrink/regrowth answers natively"
    );
}

const VEC_HUGE_CAPACITY_SRC: &str = r#"
fn main() {
    let n: usize = (1 << 22) + 5;
    let v: Vec<std::mem::MaybeUninit<u8>> = Vec::with_capacity(n);
    println!("len {} cap {}", v.len(), v.capacity());
    println!("done");
}
"#;

#[test]
fn run_mode_vec_with_capacity_past_materialisation_limit() {
    // A `with_capacity` request past the materialisation limit
    // stores no elements, but the requested capacity is the
    // vector's storage and stays in the value's account: the
    // reported capacity is the requested one, as natively —
    // not the zero stored elements.
    let dir = temp_dir("vechugecapacity");
    let src = fixture(&dir, "vechugecapacity.rs", VEC_HUGE_CAPACITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "huge with_capacity must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 0 cap 4194309\ndone\n",
        "the requested capacity past the limit is reported as natively"
    );
}

const VEC_HUGE_SET_LEN_POP_SRC: &str = r#"
fn main() {
    let n: usize = (1 << 22) + 5;
    let mut v: Vec<std::mem::MaybeUninit<u8>> = Vec::with_capacity(n);
    unsafe {
        v.set_len(n);
        match v.pop() {
            Some(b) => {
                std::mem::forget(b);
                println!("popped len {}", v.len());
            }
            None => println!("none"),
        }
    }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_huge_set_len_pop() {
    // `Vec::pop` decrements the length and then asserts the
    // new length is below the capacity. On a vector grown by
    // `set_len` to the requested capacity, the accounted
    // capacity is the full request, so the assertion holds as
    // natively; a capacity collapsed onto the declared length
    // would make the decremented length equal it and trip the
    // assertion.
    let dir = temp_dir("vechugesetlenpop");
    let src = fixture(&dir, "vechugesetlenpop.rs", VEC_HUGE_SET_LEN_POP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "pop on a huge set_len-grown vector must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "popped len 4194308\ndone\n",
        "pop returns the tail slot and the decremented length, as natively"
    );
}

const STRING_WITH_CAPACITY_SRC: &str = r#"
fn main() {
    let mut s = String::with_capacity(16);
    s.push_str("hi");
    println!("{} {}", s, s.len());
}
"#;

#[test]
fn run_mode_string_with_capacity_push_str_and_len() {
    // `String::with_capacity` is a direct `RawVecInner` caller:
    // its collected body inlines down to
    // `RawVecInner::with_capacity_in`, which has no converted
    // body, so the constructor is modelled as the empty string
    // (the requested capacity is not represented, as for
    // `Vec::with_capacity`).
    let dir = temp_dir("stringwithcapacity");
    let src = fixture(&dir, "stringwithcapacity.rs", STRING_WITH_CAPACITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "String::with_capacity must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hi 2\n");
}

const VEC_TRY_RESERVE_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3];
    println!("small {}", v.try_reserve(10).is_ok());
    println!("huge {}", v.try_reserve(usize::MAX).is_ok());
    v.push(4);
    println!("len {}", v.len());
}
"#;

#[test]
fn run_mode_vec_try_reserve_verdicts() {
    // Allocation never fails in the abstract model, so the only
    // error `try_reserve` can produce is capacity overflow:
    // 3 + 10 elements fit easily (Ok), while `usize::MAX` more
    // elements of 4 bytes exceed the `isize::MAX`-byte limit
    // (Err), exactly as natively.
    let dir = temp_dir("vectryreserve");
    let src = fixture(&dir, "vectryreserve.rs", VEC_TRY_RESERVE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::try_reserve must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "small true\nhuge false\nlen 4\n"
    );
}

const VEC_SHRINK_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3];
    v.shrink_to_fit();
    v.shrink_to(2);
    println!("len {}", v.len());
    v.push(4);
    println!("len {}", v.len());
}
"#;

#[test]
fn run_mode_vec_shrink_to_fit_and_shrink_to() {
    // Shrinking moves only the capacity, which the abstract Vec
    // model does not represent: length, elements and later
    // pushes are exactly as natively.
    let dir = temp_dir("vecshrink");
    let src = fixture(&dir, "vecshrink.rs", VEC_SHRINK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::shrink_to_fit/shrink_to must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len 3\nlen 4\n");
}

const VEC_REMOVE_ADT_SRC: &str = r#"
struct E(i32);
impl Drop for E {
    fn drop(&mut self) {
        println!("drop E({})", self.0);
    }
}
fn main() {
    // `Vec::remove` shifts the tail with one `ptr::copy` of ADT
    // elements: every shifted element must land whole, in order.
    let mut v: Vec<E> = (0..6).map(E).collect();
    let x = v.remove(1);
    println!("removed {:?}", x.0);
    std::mem::forget(x);
    println!("v={:?}", v.iter().map(|e| e.0).collect::<Vec<_>>());
    // A shorter tail (a two-element shift) and a two-field
    // element take the same copy.
    let mut w: Vec<E> = (0..4).map(E).collect();
    let y = w.remove(1);
    std::mem::forget(y);
    println!("w={:?}", w.iter().map(|e| e.0).collect::<Vec<_>>());
}
"#;

#[test]
fn run_mode_vec_remove_shifts_adt_elements_whole() {
    let dir = temp_dir("vec-remove-adt");
    let src = fixture(&dir, "vec_remove_adt.rs", VEC_REMOVE_ADT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::remove over ADT elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "removed 1\nv=[0, 2, 3, 4, 5]\nw=[0, 2, 3]\ndrop E(0)\ndrop E(2)\ndrop E(3)\ndrop E(0)\ndrop E(2)\ndrop E(3)\ndrop E(4)\ndrop E(5)\n"
    );
}

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

const REF_VEC_ITER_SRC: &str = r#"
fn main() {
    let v = vec![10, 20, 30];
    let mut sum = 0;
    for x in &v {
        sum += *x;
    }
    println!("{}", sum);
}
"#;

const BTREE_BORROWED_ITER_SRC: &str = r#"
use std::collections::{BTreeMap, BTreeSet};
fn main() {
    let s: BTreeSet<i32> = BTreeSet::new();
    let m: BTreeMap<i32, i32> = BTreeMap::new();
    let mut n = 0;
    for _x in &s {
        n += 1;
    }
    for (_k, _v) in &m {
        n += 1;
    }
    for _x in s.iter() {
        n += 1;
    }
    for _kv in m.iter() {
        n += 1;
    }
    for _k in m.keys() {
        n += 1;
    }
    for _v in m.values() {
        n += 1;
    }
    println!("count {}", n);
    println!("hint {:?}", s.iter().size_hint());
    let mut it = m.iter();
    println!("back none {}", it.next_back().is_none());
}
"#;

#[test]
fn run_mode_btree_borrowed_iteration() {
    // Borrowed iteration over the B-tree collections:
    // `IntoIterator` for `&BTreeSet` / `&BTreeMap` and the
    // `iter` / `keys` / `values` methods yield iterators over an
    // element snapshot walked from the tree's node storage. An
    // empty tree's snapshot is empty, so every form yields
    // nothing; a non-empty tree cannot be built yet (insertion
    // itself is blocked by the raw-pointer write path), so the
    // empty forms are the reachable surface.
    let dir = temp_dir("btreeborrowediter");
    let src = fixture(&dir, "btreeborrowediter.rs", BTREE_BORROWED_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating &BTreeSet / &BTreeMap must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "count 0\nhint (0, Some(0))\nback none true\n"
    );
}

const BTREE_OWNED_ITER_SRC: &str = r#"
use std::collections::{BTreeMap, BTreeSet};
fn main() {
    let s: BTreeSet<i32> = BTreeSet::new();
    let mut n = 0;
    for _x in s {
        n += 1;
    }
    let m: BTreeMap<i32, i32> = BTreeMap::new();
    for _kv in m {
        n += 1;
    }
    println!("owned {}", n);
}
"#;

#[test]
fn run_mode_btree_owned_iteration() {
    // By-value iteration consumes the collection into the
    // owning-iterator snapshot state; dropping the partially or
    // fully consumed iterator afterwards drops exactly the
    // unyielded snapshot elements (the drop plan entry for the
    // btree `IntoIter` types).
    let dir = temp_dir("btreeownediter");
    let src = fixture(&dir, "btreeownediter.rs", BTREE_OWNED_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a BTreeSet / BTreeMap by value must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "owned 0\n");
}

#[test]
fn run_mode_ref_vec_iteration() {
    // `for x in &v` goes through `IntoIterator::into_iter` for
    // `&Vec<T>`, which yields a `slice::Iter` over the elements;
    // its items are `&T`, dereferenced by the loop body.
    let dir = temp_dir("refveciter");
    let src = fixture(&dir, "refveciter.rs", REF_VEC_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating &Vec must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "60\n");
}

const VEC_REPEAT_SRC: &str = r#"
fn main() {
    let v = vec![7; 4];
    println!("len {} sum {}", v.len(), v[0] + v[3]);
    let w = vec![String::from("ab"); 2];
    println!("{} {}", w.len(), w[1]);
}
"#;

#[test]
fn run_mode_vec_repeat_form() {
    // The `vec![x; n]` repeat form goes through
    // `std::vec::from_elem`, another allocation entry point the
    // interpreter models.
    let dir = temp_dir("vecrepeat");
    let src = fixture(&dir, "vecrepeat.rs", VEC_REPEAT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec![x; n] must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 4 sum 14\n2 ab\n",
        "repeat-form Vec behaves exactly as natively"
    );
}

// ----- trait-declared method keys (Lane BV, Issue #64) -----

const TRAIT_CALL_SRC: &str = r#"
trait Greeter { fn greet(&self) -> u32; }
struct S;
impl Greeter for S { fn greet(&self) -> u32 { 42 } }
fn main() {
    let s = S;
    println!("{} {}", s.greet(), <S as Greeter>::greet(&s));
}
"#;

#[test]
fn run_mode_trait_method_call_executes() {
    // A call to a trait-declared method names the trait
    // declaration in its FnDef target, while the item loop
    // converts the impl method under the impl's own key; the
    // driver registers the impl body under the trait key too, so
    // the call executes (Issue #64).
    let dir = temp_dir("trait-call");
    let src = fixture(&dir, "trait_call.rs", TRAIT_CALL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "trait method calls must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42 42\n");
}

const TRAIT_TWO_TYPES_SRC: &str = r#"
trait Get { fn get(&self) -> u32; }
struct Wrap<T>(T);
impl Get for Wrap<u32> { fn get(&self) -> u32 { self.0 + 1 } }
impl Get for Wrap<u16> { fn get(&self) -> u32 { self.0 as u32 + 100 } }
trait Tag { fn tag(&self) -> u32; }
struct Pair<T>(T, T);
impl<T> Tag for Pair<T> { fn tag(&self) -> u32 { 7 } }
fn main() {
    let a = Wrap(3u32).get();
    let b = Wrap(4u16).get();
    let c = Pair(1u32, 2u32).tag();
    let d = Pair(true, false).tag();
    println!("{} {} {} {}", a, b, c, d);
}
"#;

#[test]
fn run_mode_trait_method_two_type_arguments() {
    // The same trait method demanded at two different type
    // arguments resolves to the right impl body for each: two
    // concrete impls of one generic struct, and one generic impl
    // instantiated at two types.
    let dir = temp_dir("trait-two-types");
    let src = fixture(&dir, "trait_two_types.rs", TRAIT_TWO_TYPES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "trait calls at two type arguments must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "4 104 7 7\n");
}

const TRAIT_DEFAULT_SRC: &str = r#"
trait D {
    fn val(&self) -> u32 { 10 }
    fn extra(&self) -> u32 { self.val() + 1 }
}
struct X;
impl D for X {}
struct Y;
impl D for Y { fn val(&self) -> u32 { 20 } }
fn main() {
    println!("{} {} {} {}", X.val(), X.extra(), Y.val(), Y.extra());
}
"#;

const DYN_DEFAULT_SRC: &str = r#"
trait Foo {
    fn a(&self) -> i32 { 100 }
    fn b(&self) -> i32 { self.a() + 1 }
}
impl Foo for i32 {}
trait Greeter {
    fn name(&self) -> i32;
    fn greet(&self) -> i32 { self.name() + 1 }
}
struct S;
impl Greeter for S {
    fn name(&self) -> i32 { 41 }
}
fn main() {
    let f: &dyn Foo = &1i32;
    assert_eq!(f.a(), 100);
    assert_eq!(f.b(), 101);
    let g: Box<dyn Foo> = Box::new(2i32);
    assert_eq!(g.a(), 100);
    assert_eq!(g.b(), 101);
    let h: &dyn Greeter = &S;
    assert_eq!(h.greet(), 42);
    let h2: Box<dyn Greeter> = Box::new(S);
    assert_eq!(h2.greet(), 42);
    println!("dyn-default ok");
}
"#;

#[test]
fn run_mode_dyn_trait_default_method() {
    // A trait default method called through a `dyn` receiver: the
    // dispatch target is the trait's own body instantiated with
    // the concrete `Self`, demanded from the vtable the unsizing
    // cast forms (Lane EU). Both `&dyn` and `Box<dyn>` receivers,
    // a default calling another default, and a default calling a
    // required method the impl provides.
    let dir = temp_dir("dyn-default");
    let src = fixture(&dir, "dyn_default.rs", DYN_DEFAULT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dyn default methods must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-default ok\n");
}

#[test]
fn summary_mode_dyn_trait_default_method_is_converted() {
    // The demanded vtable instance also settles the trait
    // method's own accounting: it no longer stands as a skipped
    // "dynamic dispatch target (virtual instance)".
    let dir = temp_dir("dyn-default-summary");
    let src = fixture(&dir, "dyn_default.rs", DYN_DEFAULT_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    for name in ["::Foo::a:", "::Foo::b:", "::Greeter::greet:"] {
        let line = stdout
            .lines()
            .find(|l| l.contains(name))
            .unwrap_or_else(|| panic!("summary lists {name}: {stdout}"));
        assert!(
            line.contains("[converted]"),
            "the trait default method {name} is converted: {line}"
        );
    }
    assert!(
        !stdout.contains("target-crate unsupported:"),
        "no target-crate function stays skipped: {stdout}"
    );
}

const DYN_DEFAULT_OVERRIDE_SRC: &str = r#"
trait D {
    fn val(&self) -> u32 { 10 }
    fn extra(&self) -> u32 { self.val() + 1 }
}
struct X;
impl D for X {}
struct Y;
impl D for Y {
    fn val(&self) -> u32 { 20 }
}
fn main() {
    let x: &dyn D = &X;
    let y: &dyn D = &Y;
    println!("{} {} {} {}", x.val(), x.extra(), y.val(), y.extra());
    let bx: Box<dyn D> = Box::new(X);
    let by: Box<dyn D> = Box::new(Y);
    println!("{} {} {} {}", bx.val(), bx.extra(), by.val(), by.extra());
}
"#;

#[test]
fn run_mode_dyn_trait_default_method_override_wins() {
    // An impl that overrides the method is what a dyn call must
    // reach — never the default — and the default `extra` body
    // must dispatch its inner `self.val()` to the override for
    // `Y` and to the default for `X`. This also pins the dispatch
    // precedence: the statically demanded default instance of
    // `val` (from `extra`'s inner call) sits in the program under
    // the plain trait name, and a dyn call must not be served by
    // it directly.
    let dir = temp_dir("dyn-default-override");
    let src = fixture(&dir, "dyn_default_override.rs", DYN_DEFAULT_OVERRIDE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dyn default/override dispatch must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "10 11 20 21\n10 11 20 21\n"
    );
}

const DYN_DEFAULT_SUPER_SRC: &str = r#"
trait Super {
    fn s(&self) -> i32 { 1 }
    fn s2(&self) -> i32;
}
trait Sub: Super {
    fn b(&self) -> i32 { 2 }
}
struct S;
impl Super for S {
    fn s2(&self) -> i32 { 3 }
}
impl Sub for S {}
fn main() {
    let d: &dyn Sub = &S;
    println!("{} {} {}", d.b(), d.s(), d.s2());
    let u: &dyn Super = d;
    println!("{} {}", u.s(), u.s2());
}
"#;

#[test]
fn run_mode_dyn_trait_default_method_supertrait_upcast() {
    // Supertrait default methods are vtable entries of the
    // supertrait vtable (`TraitVPtr`): calling them through the
    // subtrait object and through an upcast `&dyn Super` both
    // reach the default body.
    let dir = temp_dir("dyn-default-super");
    let src = fixture(&dir, "dyn_default_super.rs", DYN_DEFAULT_SUPER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "supertrait dyn defaults must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "2 1 3\n1 3\n");
}

const DYN_DEFAULT_GENERIC_TRAIT_SRC: &str = r#"
trait G<T> {
    fn get(&self, t: T) -> T { t }
}
struct S;
impl G<i32> for S {}
fn main() {
    let d: &dyn G<i32> = &S;
    println!("{}", d.get(41) + 1);
}
"#;

#[test]
fn run_mode_dyn_trait_default_method_generic_trait() {
    // A trait with generic arguments: the instance name carries
    // them (`<S as G<i32>>::get`) while the dyn call name does
    // not, so the vtable lookup resolves by its unique scan.
    let dir = temp_dir("dyn-default-generic-trait");
    let src = fixture(&dir, "dyn_default_generic.rs", DYN_DEFAULT_GENERIC_TRAIT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "generic-trait dyn defaults must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_trait_default_method_body() {
    // A trait default method keeps its own body: X uses both
    // defaults, Y overrides `val`, and the default `extra` body
    // dispatches its inner `self.val()` call to Y's override.
    let dir = temp_dir("trait-default");
    let src = fixture(&dir, "trait_default.rs", TRAIT_DEFAULT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "trait default methods must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "10 11 20 21\n");
}

// ----- anonymous constants (Lane BW) -----

const ANON_CONST_GENERIC_SRC: &str = r#"
fn f<T>(x: T) -> usize {
    let a = [0u32; 1 + 2];
    a.len() + std::mem::size_of::<T>()
}
fn main() {
    println!("{}", f::<u8>(0));
}
"#;

#[test]
fn summary_mode_generic_context_anon_consts_are_converted() {
    // An anonymous constant inside a generic function requires
    // monomorphization as an item, so it has no instance; its own
    // MIR body is converted instead, and the target crate reports
    // no skipped functions.
    let dir = temp_dir("anon-const-generic");
    let src = fixture(&dir, "anon_const_generic.rs", ANON_CONST_GENERIC_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let konst = stdout
        .lines()
        .find(|l| l.contains("::{constant#0}:"))
        .expect("summary lists the anonymous constant");
    assert!(
        konst.contains("[converted]"),
        "the generic-context anonymous constant is converted: {konst}"
    );
    assert!(
        stdout.contains("target-crate converted: 3, skipped: 0"),
        "no target-crate function is skipped: {stdout}"
    );
}

#[test]
fn run_mode_generic_context_anon_const_executes() {
    let dir = temp_dir("anon-const-generic-run");
    let src = fixture(&dir, "anon_const_generic.rs", ANON_CONST_GENERIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "4\n");
}

const TRIVIAL_CONST_SRC: &str = r#"
const N: usize = 3;
static S: [u32; 3] = [4; 1 + 2];
fn main() {
    let a = [0u32; N];
    println!("{}", a.len() + S.len());
}
"#;

#[test]
fn summary_mode_trivial_consts_are_converted_without_bodies() {
    // Trivial constants (a literal, a path) have no MIR body
    // anywhere by design; they are accounted as converted with no
    // body instead of being reported as missing bodies, while the
    // non-trivial constant of the static initializer keeps its
    // real body.
    let dir = temp_dir("trivial-consts");
    let src = fixture(&dir, "trivial_consts.rs", TRIVIAL_CONST_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let named = stdout
        .lines()
        .find(|l| l.starts_with("fn trivial_consts::N:"))
        .expect("summary lists the named constant");
    assert!(
        named.contains("blocks=0") && named.contains("[converted]"),
        "the trivial named constant is converted with no body: {named}"
    );
    let trivial = stdout
        .lines()
        .find(|l| l.contains("::S::{constant#0}:"))
        .expect("summary lists the static type-length constant");
    assert!(
        trivial.contains("blocks=0") && trivial.contains("[converted]"),
        "the trivial type-length constant is converted with no body: {trivial}"
    );
    let real = stdout
        .lines()
        .find(|l| l.contains("::S::{constant#1}:"))
        .expect("summary lists the static initializer constant");
    assert!(
        real.contains("[converted]") && !real.contains("blocks=0"),
        "the non-trivial constant keeps its converted body: {real}"
    );
    assert!(
        stdout.contains("target-crate skipped: 0")
            || stdout.contains("skipped: 0"),
        "no target-crate function is skipped: {stdout}"
    );
}

// ----- external-demand instances (Lane BX) -----

const EXTERNAL_DEMAND_SRC: &str = r#"
struct RepeatN<T> { v: T, n: usize }
impl<T: Copy> Iterator for RepeatN<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.n == 0 { None } else { self.n -= 1; Some(self.v) }
    }
}

struct Total(i32);
impl std::iter::Sum<i32> for Total {
    fn sum<I: Iterator<Item = i32>>(mut iter: I) -> Self {
        let mut acc = 0i32;
        while let Some(x) = iter.next() {
            acc += x;
        }
        Total(acc)
    }
}

fn unused<T>(x: T) -> T { x }

fn main() {
    let t: Total = RepeatN { v: 7i32, n: 3 }.sum();
    println!("{}", t.0);
}
"#;

#[test]
fn run_mode_generic_instance_demanded_through_foreign_body_executes() {
    // Neither generic definition is named by a local body:
    // `Total::sum` (generic over the iterator type) is demanded
    // only from the collected body of `Iterator::sum`, and
    // `RepeatN::next` (a method of a generic impl) only from
    // `Total::sum`'s instance body. Both instances are resolved
    // and converted, so the program runs end to end.
    let dir = temp_dir("external-demand");
    let src = fixture(&dir, "external_demand.rs", EXTERNAL_DEMAND_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "externally demanded generic instances must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "21\n");
}

#[test]
fn summary_mode_externally_demanded_instances_are_converted() {
    // The definitions demanded through foreign bodies are
    // reported converted; a definition nothing demands keeps its
    // no-instance record.
    let dir = temp_dir("external-demand-summary");
    let src = fixture(&dir, "external_demand.rs", EXTERNAL_DEMAND_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let sum = stdout
        .lines()
        .find(|l| l.contains("::sum:") && l.contains("Sum<i32>"))
        .expect("summary lists the Sum impl method");
    assert!(
        sum.contains("[converted]"),
        "the externally demanded generic method is converted: {sum}"
    );
    let next = stdout
        .lines()
        .find(|l| l.contains("RepeatN") && l.contains(">::next:"))
        .expect("summary lists the Iterator impl method");
    assert!(
        next.contains("[converted]"),
        "the transitively demanded generic method is converted: {next}"
    );
    let unused = stdout
        .lines()
        .find(|l| l.starts_with("fn external_demand::unused:"))
        .expect("summary lists the unused generic definition");
    assert!(
        unused.contains("NOT DEMANDED (no monomorphic instance available (kind=Fn)"),
        "the undemanded generic definition is explicitly not demanded: {unused}"
    );
}

// ----- dst-index impl resolution (Lane BY) -----

const DST_INDEX_SRC: &str = r#"
use std::fmt::Debug;
use std::ops::Index;

struct S;

impl Index<usize> for S {
    type Output = str;

    fn index(&self, _: usize) -> &str {
        "hello"
    }
}

struct T;

impl Index<usize> for T {
    type Output = dyn Debug + 'static;

    fn index(&self, _: usize) -> &(dyn Debug + 'static) {
        static X: usize = 42;
        &X as &(dyn Debug + 'static)
    }
}

fn main() {
    assert_eq!(&S[0], "hello");
    let _ = &T[0];
    println!("dst index ok");
}
"#;

#[test]
fn run_mode_index_impls_with_dst_outputs_execute() {
    // The shape of tests/ui/dst/dst-index-success.rs (Issue #65):
    // the Index calls name the trait item, whose name-based builtin
    // model indexed the receiver struct itself and panicked with a
    // bogus out-of-bounds message; the collected impl bodies must
    // run instead, for both the `str` and the `dyn Debug` outputs.
    let dir = temp_dir("dstindex");
    let src = fixture(&dir, "dstindex.rs", DST_INDEX_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dst index program must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dst index ok\n");
}

// ----- Drop demand pass (Lane BS) -----

const GENERIC_DROP_SRC: &str = r#"
struct G<T> { x: T }
impl<T> Drop for G<T> {
    fn drop(&mut self) { println!("dropped"); }
}
fn main() {
    let _g = G { x: 1i32 };
    println!("body");
}
"#;

#[test]
fn run_mode_generic_drop_impl_runs() {
    // A generic `Drop` impl has no monomorphic instance among the
    // crate's items and drop glue is not a call edge, so the driver
    // must demand the instantiation the program drops (Lane BS).
    // Before that pass existed this exited 2 with
    // "unsupported terminator: Drop".
    let dir = temp_dir("generic-drop");
    let src = fixture(&dir, "generic_drop.rs", GENERIC_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "generic Drop impl must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "body\ndropped\n",
        "the destructor's output must follow the body's"
    );
}

const TWO_INSTANCE_DROP_SRC: &str = r#"
struct G<T> { x: T }
impl<T> Drop for G<T> {
    fn drop(&mut self) { println!("dropped"); }
}
fn main() {
    let _a = G { x: 1i32 };
    let _b = G { x: true };
    println!("body");
}
"#;

#[test]
fn run_mode_two_instantiations_of_one_generic_drop_impl() {
    // Two monomorphizations of one generic `Drop` impl in a single
    // program: each drop must find its own instantiation's body.
    let dir = temp_dir("two-instance-drop");
    let src = fixture(&dir, "two_instance_drop.rs", TWO_INSTANCE_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "both instantiations must drop; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "body\ndropped\ndropped\n",
        "both destructors must run"
    );
}

const LIFETIME_DROP_SRC: &str = r#"
struct S<'a> { p: &'a str }
impl<'a> Drop for S<'a> {
    fn drop(&mut self) { println!("dropped {}", self.p); }
}
fn main() {
    let _s = S { p: "x" };
    println!("body");
}
"#;

#[test]
fn run_mode_lifetime_parameterized_drop_impl_runs() {
    // A lifetime-only `Drop` impl is converted by the item loop
    // under its declared spelling (`<S<'a> as ...>::drop`); the
    // drop-demand pass resolves the same instance with the erased
    // spelling (`<S<'_> as ...>::drop`). Registering both left the
    // interpreter's destructor lookup with two candidates, which it
    // treats as ambiguous, and the drop came out as
    // "unsupported terminator: Drop" (the Lane BS demand-pass
    // regression: drop-scope-exit, issue-2735-2 and seven more).
    // The demand pass now skips instances that are already
    // converted, by DefId.
    let dir = temp_dir("lifetime-drop");
    let src = fixture(&dir, "lifetime_drop.rs", LIFETIME_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "lifetime-parameterized Drop impl must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "body\ndropped x\n",
        "the destructor's output must follow the body's"
    );
}

// ----- RefCell model (Lane CC) -----

const REFCELL_SRC: &str = r#"
use std::cell::RefCell;
fn main() {
    let mut c = RefCell::new(40);
    {
        let a = c.borrow();
        let b = c.borrow();
        println!("{}", *a + *b);
    }
    *c.borrow_mut() += 2;
    let v = *c.borrow();
    println!("{}", v);
    let old = c.replace(7);
    println!("{}", old);
    {
        let _m = c.borrow_mut();
        println!("{}", c.try_borrow().is_err());
        println!("{}", c.try_borrow_mut().is_err());
    }
    println!("{}", c.try_borrow().is_ok());
    *c.get_mut() = 9;
    let w = *c.borrow();
    println!("{}", w);
}
"#;

#[test]
fn run_mode_refcell_borrow_discipline() {
    let dir = temp_dir("refcell");
    let src = fixture(&dir, "refcell.rs", REFCELL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "RefCell program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "80\n42\n42\ntrue\ntrue\ntrue\n9\n",
        "shared borrows coexist, guards release on drop, try_borrow \
         reports conflicts, replace/get_mut go through the cell"
    );
}

const REFCELL_PANIC_SRC: &str = r#"
use std::cell::RefCell;
fn main() {
    let c = RefCell::new(1);
    let _a = c.borrow();
    let _b = c.borrow_mut();
}
"#;

#[test]
fn run_mode_refcell_borrow_mut_conflict_panics() {
    let dir = temp_dir("refcell-panic");
    let src = fixture(&dir, "refcell_panic.rs", REFCELL_PANIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "borrow_mut while borrowed is a guest panic; stderr={stderr}"
    );
    assert!(
        stderr.contains("RefCell already borrowed"),
        "the panic carries Rust 1.100.0's exact message: {stderr}"
    );
}

const REFCELL_MUT_PANIC_SRC: &str = r#"
use std::cell::RefCell;
fn main() {
    let c = RefCell::new(1);
    let _m = c.borrow_mut();
    let _a = c.borrow();
}
"#;

#[test]
fn run_mode_refcell_borrow_while_mutable_panics() {
    let dir = temp_dir("refcell-mut-panic");
    let src = fixture(&dir, "refcell_mut_panic.rs", REFCELL_MUT_PANIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "borrow while mutably borrowed is a guest panic; stderr={stderr}"
    );
    assert!(
        stderr.contains("RefCell already mutably borrowed"),
        "the panic carries Rust 1.100.0's exact message: {stderr}"
    );
}

const REFCELL_DROP_SRC: &str = r#"
use std::cell::RefCell;
struct D(i32);
impl Drop for D {
    fn drop(&mut self) {
        println!("dropping {}", self.0);
    }
}
fn main() {
    let c = RefCell::new(D(5));
    println!("borrowed {}", c.borrow().0);
    drop(c);
    println!("done");
}
"#;

#[test]
fn run_mode_refcell_drop_runs_inner_destructor() {
    let dir = temp_dir("refcell-drop");
    let src = fixture(&dir, "refcell_drop.rs", REFCELL_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "RefCell<D> program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "borrowed 5\ndropping 5\ndone\n",
        "dropping the cell runs the wrapped value's destructor"
    );
}

// ----- generic constructors (Lane CF) -----
//
// A generic tuple-struct / tuple-variant constructor has no
// monomorphic instance as an item. Its direct uses are aggregate
// rvalues in the callers, so nothing is left unconverted; used as
// a value, the instance its FnDef constant demands is converted
// (the synthetic one-block body that aggregates the fields) and
// the call through the value runs it.

const CTOR_VALUE_SRC: &str = r#"
struct Pair<T>(T, i32);
fn main() {
    let f = Pair;
    let p = f(3, 4);
    println!("{}", p.0 + p.1);
}
"#;

#[test]
fn run_mode_generic_ctor_as_value_executes() {
    let dir = temp_dir("ctor-value");
    let src = fixture(&dir, "ctor_value.rs", CTOR_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a generic constructor used as a value must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "7\n");
}

const CTOR_HOF_SRC: &str = r#"
struct Wrap<T>(T);
fn apply<F: Fn(i32) -> Wrap<i32>>(f: F) -> i32 { f(41).0 + 1 }
fn main() {
    println!("{}", apply(Wrap));
}
"#;

#[test]
fn run_mode_generic_ctor_passed_to_hof_executes() {
    let dir = temp_dir("ctor-hof");
    let src = fixture(&dir, "ctor_hof.rs", CTOR_HOF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a generic constructor passed as a callback must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const CTOR_ENUM_VALUE_SRC: &str = r#"
enum E<T> { V(T, i32), W }
fn main() {
    let f = E::V;
    let e = f(5, 6);
    match e { E::V(a, b) => println!("{}", a + b), E::W => println!("w") }
}
"#;

#[test]
fn run_mode_generic_enum_variant_ctor_as_value_executes() {
    let dir = temp_dir("ctor-enum-value");
    let src = fixture(&dir, "ctor_enum_value.rs", CTOR_ENUM_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a generic variant constructor used as a value must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "11\n");
}

const CTOR_TWO_INSTANCES_SRC: &str = r#"
struct G<T>(T, i32);
fn main() {
    let fi = G::<i32>;
    let fs = G::<&str>;
    let a = fi(1, 2);
    let b = fs("x", 3);
    println!("{} {} {}", a.0 + a.1, b.0, b.1);
}
"#;

#[test]
fn run_mode_generic_ctor_two_instances_as_values_execute() {
    let dir = temp_dir("ctor-two-instances");
    let src = fixture(&dir, "ctor_two_instances.rs", CTOR_TWO_INSTANCES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "two instances of one generic constructor must both execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "3 x 3\n");
}

const CTOR_DIRECT_ONLY_SRC: &str = r#"
struct G<T>(T, i32);
fn main() {
    let g = G(10, 2);
    println!("{}", g.0 + g.1);
}
"#;

#[test]
fn summary_mode_generic_ctor_is_accounted_converted() {
    // Used directly (aggregate rvalues in main) or as a value
    // (demanded instance converted), a generic constructor is
    // reported converted — not as a missing monomorphic instance.
    let dir = temp_dir("ctor-direct-only");
    let src = fixture(&dir, "ctor_direct_only.rs", CTOR_DIRECT_ONLY_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let g = stdout
        .lines()
        .find(|l| l.starts_with("fn ctor_direct_only::G:"))
        .expect("summary lists the generic constructor");
    assert!(
        g.contains("[converted]"),
        "an undemanded generic constructor is converted without a body: {g}"
    );

    let dir = temp_dir("ctor-value-summary");
    let src = fixture(&dir, "ctor_value.rs", CTOR_VALUE_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let pair = stdout
        .lines()
        .find(|l| l.starts_with("fn ctor_value::Pair:"))
        .expect("summary lists the generic constructor");
    assert!(
        pair.contains("[converted]"),
        "a demanded generic constructor is converted: {pair}"
    );
    assert!(
        !stdout.contains("kind=Ctor"),
        "no constructor is left as a missing instance: {stdout}"
    );
}

// ----- monomorphization validity checks (Lane CA) ---------------------------

const SIMD_SCALAR_SRC: &str = r#"
#![feature(core_intrinsics)]
use std::intrinsics::simd::simd_add;
fn main() { unsafe { let _ = simd_add(0, 1); } }
"#;

const SIMD_VECTOR_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_add;
#[repr(simd)]
#[derive(Copy, Clone)]
struct I32x4([i32; 4]);
fn main() { unsafe { let _ = simd_add(I32x4([1, 2, 3, 4]), I32x4([5, 6, 7, 8])); } }
"#;

const SIMD_TOO_MANY_LANES_SRC: &str = r#"
#![feature(portable_simd)]
use std::simd::Simd;
fn main() { let _x = Simd::<u16, 54321>::splat(0); }
"#;

#[test]
fn convert_mode_simd_intrinsic_with_scalar_args_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-scalar");
    let src = fixture(&dir, "simd_scalar.rs", SIMD_SCALAR_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "an invalid SIMD monomorphization must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("simd_add")
            && stderr.contains("expected SIMD input type"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_intrinsic_with_vector_args_converts() {
    let dir = temp_dir("mono-simd-vector");
    let src = fixture(&dir, "simd_vector.rs", SIMD_VECTOR_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a valid SIMD monomorphization must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_type_with_too_many_lanes_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-lanes");
    let src = fixture(&dir, "simd_lanes.rs", SIMD_TOO_MANY_LANES_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a SIMD type over the lane limit must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("invalid number of lanes"),
        "the failure must name the reason; stderr={stderr}"
    );
}

// ----- ADT layout table (Lane CG) -----

const LAYOUT_SRC: &str = r#"
#[repr(C)]
struct Padded { a: u8, b: u32, c: u16 }
#[repr(packed)]
struct Packed { a: u8, b: u32 }
enum E { A, B(u16), C(u64) }
struct Outer { x: u8, inner: Padded }
struct Pair<A, B> { a: A, b: B }
fn main() {
    println!("padded {} {}", std::mem::size_of::<Padded>(), std::mem::align_of::<Padded>());
    println!("packed {} {}", std::mem::size_of::<Packed>(), std::mem::align_of::<Packed>());
    println!("enum {} {}", std::mem::size_of::<E>(), std::mem::align_of::<E>());
    println!("outer {} {}", std::mem::size_of::<Outer>(), std::mem::align_of::<Outer>());
    println!("pair {} {}", std::mem::size_of::<Pair<u8, u64>>(), std::mem::align_of::<Pair<u8, u64>>());
    let v = Padded { a: 1, b: 2, c: 3 };
    println!("val {}", std::mem::size_of_val(&v));
}
"#;

const PACKED_DERIVE_SRC: &str = r#"
use std::mem;

#[derive(Copy, Clone)]
struct Aligned(usize);

#[inline(never)]
fn check_align(ptr: *const Aligned) {
    assert_eq!(ptr as usize % mem::align_of::<Aligned>(), 0);
}

impl PartialEq for Aligned {
    fn eq(&self, other: &Self) -> bool {
        check_align(self);
        check_align(other);
        self.0 == other.0
    }
}

#[repr(packed)]
#[derive(Copy, Clone, PartialEq)]
struct Packed(Aligned, Aligned);

#[derive(PartialEq)]
#[repr(C)]
struct Dealigned<T>(u8, T);

fn main() {
    let d1 = Dealigned(0, Packed(Aligned(1), Aligned(2)));
    let ck = d1 == d1;
    assert!(ck);
    println!("eq-ok");
}
"#;

#[test]
fn run_mode_size_of_align_of_user_adts() {
    // The values are the compiler's own layouts (verified against a
    // native 1.100.0 build of the same source): the driver records
    // them per monomorphic ADT instance and the mem builtins answer
    // from that table.
    let dir = temp_dir("adt-layout");
    let src = fixture(&dir, "layout.rs", LAYOUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "size_of/align_of of user ADTs run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "padded 12 4\npacked 5 1\nenum 16 8\nouter 16 4\npair 16 8\nval 12\n",
        "layouts match the native ones"
    );
}

// ----- coroutine/closure layouts (Lane HN, ex-DW) -----

const COROUTINE_CLOSURE_SIZE_SRC: &str = r#"
async fn f(x: u8) -> u8 { x + 1 }
fn main() {
    let fut = f(3);
    println!("fut {}", std::mem::size_of_val(&fut));
    let a = async { 1u8 };
    let b = async { 2u16 };
    println!("blocks {} {}", std::mem::size_of_val(&a), std::mem::size_of_val(&b));
    let arr = [1u64, 2u64];
    let big = async move { arr[0] };
    println!("big {}", std::mem::size_of_val(&big));
    let x = 5u32;
    let c1 = move || x;
    let c2 = move || x as u64;
    let y = 7u64;
    let c3 = move || y;
    println!("closures {} {} {}", std::mem::size_of_val(&c1), std::mem::size_of_val(&c2), std::mem::size_of_val(&c3));
    println!("tysize {}", std::mem::size_of_val(&c1) + std::mem::size_of::<u32>());
    println!("call {}", c1());
}
"#;

#[test]
fn run_mode_size_of_val_coroutine_and_closure() {
    // The values are the compiler's own layouts (verified against a
    // native 1.100.0 build of the same source): the driver records
    // closure and coroutine layouts per monomorphic type, keyed by
    // the definition name plus generic arguments, and the mem
    // builtins answer from that table. `big` (24) differs from the
    // two unit-capture blocks (1, 1), so distinct coroutine types in
    // one function do not share an entry; likewise the u64-capturing
    // closure (8) differs from the two u32-capturing ones (4, 4), so
    // distinct closure types do not share one either.
    let dir = temp_dir("coroutine-closure-size");
    let src = fixture(&dir, "sizes.rs", COROUTINE_CLOSURE_SIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "size_of_val of coroutine/closure values runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "fut 2\nblocks 1 1\nbig 24\nclosures 4 4 8\ntysize 8\ncall 5\n",
        "sizes match the native ones"
    );
}

#[test]
fn run_mode_derived_eq_on_packed_struct_uses_align_of() {
    let dir = temp_dir("packed-derive");
    let src = fixture(&dir, "packed_derive.rs", PACKED_DERIVE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "derived eq over a packed struct runs (align_of::<Aligned> resolves); stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "eq-ok\n");
}

// ----- type-mismatch bucket fixes (Lane CD) -----

const BLANKET_FN_SRC: &str = r#"
fn a<F: Fn() -> i32>(f: F) -> i32 { f() }
fn b(f: &dyn Fn() -> i32) -> i32 { a(f) }
fn c<F: Fn() -> i32>(f: &F) -> i32 { a(f) }
fn main() {
    assert_eq!(b(&|| 22), 22);
    assert_eq!(c(&|| 23), 23);
    println!("blanket fn ok");
}
"#;

const BOX_UNSIZE_SRC: &str = r#"
fn main() {
    let xs: Box<[()]> = Box::<[(); 0]>::new([]);
    assert_eq!(xs.len(), 0);
    let ys: Box<[i32]> = Box::new([1, 2, 3]);
    assert_eq!(ys.len(), 3);
    assert_eq!(&*ys, &[1, 2, 3]);
    println!("unsize ok");
}
"#;

const BOXED_SLICE_FROM_VEC_SRC: &str = r#"
#[derive(Debug)]
struct P {
    x: i32,
}

fn main() {
    // `From<Vec<T>> for Box<[T]>` is `Vec::into_boxed_slice`:
    // the pointee is the element sequence, so every slice
    // consumer of the box sees the elements.
    let b: Box<[i32]> = Box::from(vec![4, 5, 6]);
    println!("{:?}", b);
    println!("{}", b.len());
    println!("{}", b[0] + b[2]);
    for x in b.iter() {
        println!("{}", x);
    }
    // `Into` resolves to the same conversion.
    let c: Box<[i32]> = vec![7, 8].into();
    println!("{:?}", c);
    // Clone, comparison and pretty Debug see the elements too.
    let d = b.clone();
    println!("{} {}", b == d, b < c);
    println!("{:#?}", b);
    // ADT elements keep their derived Debug.
    let e: Box<[P]> = Box::from(vec![P { x: 1 }, P { x: 2 }]);
    println!("{:?}", e);
    // `into_vec` hands back a working vector.
    let mut v = b.into_vec();
    v.push(9);
    v.sort();
    println!("{:?}", v);
    // `From<Cow<'_, [T]>>` hands over the Cow's contents: the
    // owned vector's elements, or a clone of the borrowed
    // slice's elements.
    let cow: std::borrow::Cow<[i32]> = std::borrow::Cow::Owned(vec![11, 12]);
    let f: Box<[i32]> = Box::from(cow);
    println!("{:?}", f);
    let v2 = vec![10, 20, 30, 40];
    let cow2: std::borrow::Cow<[i32]> = std::borrow::Cow::Borrowed(&v2[1..3]);
    let g: Box<[i32]> = Box::from(cow2);
    println!("{:?}", g);
    println!("boxed from vec ok");
}
"#;

const BOXED_SLICE_FROM_BORROWED_COW_SRC: &str = r#"
use std::borrow::Cow;

fn main() {
    // `From<Cow<'_, [T]>> for Box<[T]>` clones the borrowed
    // slice's elements into the new allocation: mutating the
    // source afterwards must not show through the box.
    let mut src = [1, 2, 3];
    let b: Box<[i32]> = Box::from(Cow::Borrowed(&src[..]));
    src[0] = 99;
    println!("array: {} {:?} {}", b[0], b, b.len());

    let mut v = vec![10, 20, 30];
    let c: Box<[i32]> = Box::from(Cow::Borrowed(&v[..]));
    v[1] = 77;
    println!("vec: {} {:?}", c[1], c);

    let mut w = vec![5, 6, 7, 8];
    let d: Box<[i32]> = Box::from(Cow::Borrowed(&w[1..3]));
    w[2] = 66;
    println!("sub: {} {:?}", d[1], d);
}
"#;

const BOXED_SLICE_COLLECT_SRC: &str = r#"
fn main() {
    let b: Box<[i32]> = (0..5).collect();
    println!("{:?}", b);
    let s: Box<[String]> = ["a", "bb"].iter().map(|x| x.to_string()).collect();
    println!("{:?}", s);
    let e: Box<[i32]> = std::iter::empty().collect();
    println!("{:?} {}", e, e.len());
    println!("boxed collect ok");
}
"#;

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

const FLOAT_EQ_SRC: &str = r#"
#[derive(PartialEq, Debug)]
struct P { x: f64, y: i32 }
#[derive(PartialEq, Debug)]
struct Pair(u32, u32);
fn main() {
    let a = P { x: 1.5, y: 2 };
    let b = P { x: 1.5, y: 2 };
    assert!(a == b);
    let nan = P { x: f64::NAN, y: 0 };
    assert!(nan != nan);
    assert!(Pair(3, 4) == Pair(3, 4));
    assert!(Pair(3, 4) != Pair(3, 5));
    let t1 = (1.25f32, 7u8);
    let t2 = (1.25f32, 7u8);
    assert!(t1 == t2);
    println!("float eq ok");
}
"#;

const PLACE_SLICE_META_SRC: &str = r#"
static C: &[i32] = &[10, 20, 30];
fn main() {
    assert_eq!(C.len(), 3);
    assert_eq!(C[1], 20);
    let ys: Box<[i32]> = Box::new([1, 2, 3]);
    let s: &[i32] = &*ys;
    assert_eq!(s.len(), 3);
    assert_eq!(s[1], 2);
    println!("meta ok");
}
"#;

const PTR_EQ_SRC: &str = r#"
fn f() {}
fn g() {}
fn main() {
    let x = 5;
    let p = &x as *const i32;
    let q = &x as *const i32;
    assert!(p == q);
    assert!(!p.is_null());
    let fp: fn() = f;
    let fq: fn() = f;
    let fg: fn() = g;
    assert!(fp == fq);
    assert!(fp != fg);
    let a = &x;
    let b = &x;
    assert!(std::ptr::eq(a, b));
    println!("ptr eq ok");
}
"#;

const RANGE_INDEX_SRC: &str = r#"
fn main() {
    let a = [1, 2, 3, 4];
    assert_eq!(&a[1..3], &[2, 3]);
    assert_eq!(&a[..2], &[1, 2]);
    assert_eq!(&a[2..], &[3, 4]);
    assert_eq!(&a[..], &[1, 2, 3, 4]);
    assert_eq!(&a[1..=2], &[2, 3]);
    let st = "héllo";
    assert_eq!(&st[0..1], "h");
    assert_eq!(&st[1..3], "é");
    println!("ranges ok");
}
"#;

const RANGE_PANIC_SRC: &str = r#"
fn main() {
    let a = [1, 2, 3, 4];
    let _s = &a[2..9];
    println!("unreachable");
}
"#;

#[test]
fn run_mode_blanket_fn_impls_for_references() {
    let dir = temp_dir("blanket-fn");
    let src = fixture(&dir, "blanket_fn.rs", BLANKET_FN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "calling through &F / &dyn Fn blanket impls must work; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "blanket fn ok\n");
}

#[test]
fn run_mode_boxed_slice_unsize() {
    let dir = temp_dir("box-unsize");
    let src = fixture(&dir, "box_unsize.rs", BOX_UNSIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[T; N]> to Box<[T]> unsizing must work; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "unsize ok\n");
}

#[test]
fn run_mode_boxed_slice_from_vec() {
    let dir = temp_dir("boxed-from-vec");
    let src = fixture(&dir, "boxed_from_vec.rs", BOXED_SLICE_FROM_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[T]> from Vec must convert and format like into_boxed_slice; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[4, 5, 6]\n3\n10\n4\n5\n6\n[7, 8]\ntrue true\n[\n    4,\n    5,\n    6,\n]\n[P { x: 1 }, P { x: 2 }]\n[4, 5, 6, 9]\n[11, 12]\n[20, 30]\nboxed from vec ok\n"
    );
}

#[test]
fn run_mode_boxed_slice_from_borrowed_cow_clones_elements() {
    let dir = temp_dir("boxed-from-borrowed-cow");
    let src = fixture(
        &dir,
        "boxed_from_borrowed_cow.rs",
        BOXED_SLICE_FROM_BORROWED_COW_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[T]> from a borrowed Cow must clone the elements; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "array: 1 [1, 2, 3] 3\nvec: 20 [10, 20, 30]\nsub: 7 [6, 7]\n"
    );
}

const BOXED_SLICE_ARRAY_ELEMENTS_SRC: &str = r#"
fn main() {
    // The element type of a boxed slice may itself be an array:
    // the semicolon inside `[i32; 2]` sits one bracket level
    // deeper and does not make the pointee an array type.
    let b: Box<[[i32; 2]]> = vec![[1, 2], [3, 4]].into_iter().collect();
    println!("collect: {:?}", b);

    let v: Vec<[i32; 2]> = vec![[5, 6], [7, 8]];
    let c: Box<[[i32; 2]]> = Box::from(v);
    println!("fromvec: {:?}", c);

    let arr: [[i32; 2]; 2] = [[9, 10], [11, 12]];
    let d: Box<[[i32; 2]]> = Box::from(&arr[..]);
    println!("fromslice: {:?}", d);

    let e: Box<[[i32; 2]]> = Box::from(std::borrow::Cow::Borrowed(&arr[..]));
    println!("fromcow: {:?}", e);

    let f: Box<[[[i32; 1]; 2]]> = vec![[[1], [2]], [[3], [4]]].into_iter().collect();
    println!("deep: {:?}", f);
}
"#;

#[test]
fn run_mode_boxed_slice_array_elements() {
    let dir = temp_dir("boxed-array-elements");
    let src = fixture(
        &dir,
        "boxed_array_elements.rs",
        BOXED_SLICE_ARRAY_ELEMENTS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "boxed slices of array elements must convert and collect; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "collect: [[1, 2], [3, 4]]\nfromvec: [[5, 6], [7, 8]]\nfromslice: [[9, 10], [11, 12]]\nfromcow: [[9, 10], [11, 12]]\ndeep: [[[1], [2]], [[3], [4]]]\n"
    );
}

const BOXED_SLICE_FROM_BORROWED_COW_DEEP_SRC: &str = r#"
use std::borrow::Cow;

fn main() {
    // The borrowed elements themselves own allocations: boxing
    // clones every element (reallocating the inner boxes), so a
    // later write through the source's element -- or dropping it
    // by replacement -- is invisible through the box.
    let mut src: Vec<Box<i32>> = vec![Box::new(1), Box::new(2), Box::new(3)];
    let b: Box<[Box<i32>]> = Box::from(Cow::Borrowed(&src[..]));
    *src[0] = 99;
    src[1] = Box::new(77);
    println!("deepclone: {} {} {:?}", *b[0], *b[1], b);

    let mut sv: Vec<String> = vec!["aa".to_string(), "bb".to_string()];
    let s: Box<[String]> = Box::from(Cow::Borrowed(&sv[..]));
    sv[0].push_str("ZZ");
    println!("strings: {} {:?}", s[0], s);
}
"#;

#[test]
fn run_mode_boxed_slice_from_borrowed_cow_deep_clones_elements() {
    let dir = temp_dir("boxed-from-borrowed-cow-deep");
    let src = fixture(
        &dir,
        "boxed_from_borrowed_cow_deep.rs",
        BOXED_SLICE_FROM_BORROWED_COW_DEEP_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[T]> from a borrowed Cow must deep-clone owning elements; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "deepclone: 1 2 [1, 2, 3]\nstrings: aa [\"aa\", \"bb\"]\n"
    );
}

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

#[test]
fn run_mode_boxed_slice_collect() {
    let dir = temp_dir("boxed-collect");
    let src = fixture(&dir, "boxed_collect.rs", BOXED_SLICE_COLLECT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "collect into Box<[T]> must work; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[0, 1, 2, 3, 4]\n[\"a\", \"bb\"]\n[] 0\nboxed collect ok\n"
    );
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

#[test]
fn run_mode_derived_eq_over_floats_and_tuple_structs() {
    let dir = temp_dir("float-eq");
    let src = fixture(&dir, "float_eq.rs", FLOAT_EQ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "derived equality over float fields must work; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "float eq ok\n");
}

#[test]
fn run_mode_pointer_and_fn_pointer_equality() {
    let dir = temp_dir("ptr-eq");
    let src = fixture(&dir, "ptr_eq.rs", PTR_EQ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "pointer equality must run to completion; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ptr eq ok\n");
}

#[test]
fn run_mode_range_indexing() {
    let dir = temp_dir("range-index");
    let src = fixture(&dir, "range_index.rs", RANGE_INDEX_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "range indexing must run to completion; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ranges ok\n");
}

#[test]
fn run_mode_range_index_out_of_bounds_panics() {
    let dir = temp_dir("range-panic");
    let src = fixture(&dir, "range_panic.rs", RANGE_PANIC_SRC);
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(101),
        "an out-of-bounds slice range is a guest panic (101)"
    );
}

#[test]
fn run_mode_slice_metadata_through_places() {
    let dir = temp_dir("place-slice-meta");
    let src = fixture(&dir, "place_slice_meta.rs", PLACE_SLICE_META_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "PtrMetadata/Len through place references must work; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "meta ok\n");
}

// ----- driver demand round 2 (Lane CP) -----

const EXT_CTOR_HOF_SRC: &str = r#"
fn apply<F: FnOnce(u32) -> Option<u32>>(f: F, v: u32) -> u32 {
    match f(v) { Some(x) => x, None => 0 }
}
fn main() {
    println!("{}", apply(Some, 40) + 2);
}
"#;

const EXT_CTOR_VALUE_SRC: &str = r#"
fn main() {
    let f = Some;
    let x = f(3);
    match x { Some(v) => println!("{}", v + 1), None => println!("none") }
}
"#;

const EXT_FN_VALUE_SRC: &str = r#"
fn main() {
    let f = std::num::Wrapping;
    let w = f(20u32) + std::num::Wrapping(22u32);
    println!("{}", w.0);
    let g = String::new;
    let mut s = g();
    s.push_str("hi");
    println!("{}", s.len());
}
"#;

#[test]
fn run_mode_foreign_ctor_as_value_executes() {
    let dir = temp_dir("ext-ctor-value");
    let src = fixture(&dir, "ext_ctor_value.rs", EXT_CTOR_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a foreign constructor used as a value must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "4\n");
}

#[test]
fn run_mode_foreign_ctor_passed_to_hof_executes() {
    let dir = temp_dir("ext-ctor-hof");
    let src = fixture(&dir, "ext_ctor_hof.rs", EXT_CTOR_HOF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a foreign constructor passed to a higher-order function must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_foreign_fn_and_tuple_struct_ctor_as_values_execute() {
    let dir = temp_dir("ext-fn-value");
    let src = fixture(&dir, "ext_fn_value.rs", EXT_FN_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a foreign plain function and tuple-struct constructor used as values must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n2\n");
}

// ----- derived Ord on enums: discriminant values (Lane CT) -----

const DERIVE_DISCRIMINANT_SRC: &str = r#"
// The shape of tests/ui/derives/eq-ord/derive-partial-ord-discriminant:
// derived PartialOrd must order by discriminant value, not by
// variant declaration order, including negative discriminants and
// 64-bit extremes. Operands are constants, so they also arrive as
// promoted by-reference constants.
#[derive(PartialEq, PartialOrd)]
enum E1 {
    Pos2 = 2,
    Neg1 = -1,
    Pos1 = 1,
}

#[derive(PartialEq, PartialOrd)]
#[repr(i8)]
enum E3 {
    Pos2 = 2,
    Neg1 = -1_i8,
    Pos1 = 1,
}

#[derive(PartialEq, PartialOrd)]
#[repr(i64)]
enum Ei64 {
    Pos2 = 2,
    Neg1 = -1,
    NegMin = 1 << 63,
    PosMax = !(1 << 63),
    Pos1 = 1,
}

fn main() {
    assert!(E1::Pos2 > E1::Pos1);
    assert!(E1::Pos1 > E1::Neg1);
    assert!(E1::Pos2 > E1::Neg1);

    assert!(E3::Pos2 > E3::Pos1);
    assert!(E3::Pos1 > E3::Neg1);
    assert!(E3::Pos2 > E3::Neg1);

    assert!(Ei64::Pos2 > Ei64::Pos1);
    assert!(Ei64::Pos2 > Ei64::Neg1);
    assert!(Ei64::Pos1 > Ei64::Neg1);
    assert!(Ei64::Pos2 > Ei64::NegMin);
    assert!(Ei64::Pos1 > Ei64::NegMin);
    assert!(Ei64::Pos2 < Ei64::PosMax);
    assert!(Ei64::Pos1 < Ei64::PosMax);
}
"#;

const DERIVE_NEGATIVE_EQ_SRC: &str = r#"
// Derived PartialEq and Ordering comparisons where a negative
// discriminant is involved: the constant operand is a promoted
// by-reference constant whose stored tag is the width-truncated
// pattern of the discriminant.
#[derive(PartialEq)]
enum P {
    X = -1,
    Y = 7,
}

fn check() -> u32 {
    let mut n = 0;
    if P::X == P::X { n += 1; }
    if P::X != P::Y { n += 1; }
    let o = core::cmp::Ordering::Less;
    if o == core::cmp::Ordering::Less { n += 1; }
    if o < core::cmp::Ordering::Equal { n += 1; }
    if core::cmp::Ordering::Greater > o { n += 1; }
    n
}

fn main() {
    if check() == 5 { println!("neg eq ok"); } else { println!("neg eq BAD"); }
}
"#;

const DERIVE_PAYLOAD_ORD_SRC: &str = r#"
// Derived PartialOrd/Ord on an enum with payloads: the
// discriminant decides across variants, fields within a variant.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum E {
    A(u8),
    B,
    C(u16),
}

fn check() -> u32 {
    let a1 = E::A(1);
    let a2 = E::A(2);
    let b = E::B;
    let c = E::C(0);
    let mut n = 0;
    if a1 < a2 { n += 1; }
    if a2 < b { n += 1; }
    if b < c { n += 1; }
    if a1 < c { n += 1; }
    if a1 == E::A(1) { n += 1; }
    if a1 != a2 { n += 1; }
    if a1.cmp(&a2) == core::cmp::Ordering::Less { n += 1; }
    if b.cmp(&c) == core::cmp::Ordering::Less { n += 1; }
    if c.cmp(&c) == core::cmp::Ordering::Equal { n += 1; }
    n
}

fn main() {
    if check() == 9 { println!("payload ord ok"); } else { println!("payload ord BAD"); }
}
"#;

#[test]
fn run_mode_derived_eq_negative_discriminant_constants() {
    let dir = temp_dir("derive-neg-eq");
    let src = fixture(&dir, "derive_neg_eq.rs", DERIVE_NEGATIVE_EQ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "negative-discriminant constant comparisons must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "neg eq ok\n",
        "derived equality and Ordering comparisons must hold"
    );
}

#[test]
fn run_mode_derived_ord_payload_enum() {
    let dir = temp_dir("derive-payload-ord");
    let src = fixture(&dir, "derive_payload_ord.rs", DERIVE_PAYLOAD_ORD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "derived Ord on a payload enum must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "payload ord ok\n",
        "derived Ord must order by discriminant, then fields"
    );
}

#[test]
fn run_mode_derived_partial_ord_uses_discriminant_values() {
    let dir = temp_dir("derive-discr");
    let src = fixture(&dir, "derive_discr.rs", DERIVE_DISCRIMINANT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "derived PartialOrd on explicit discriminants must run and hold; stderr={stderr}"
    );
}

// ----- allocated constants: f16, null and fn pointers, uninit (Lane CK) -----

const F16_SRC: &str = r#"#![feature(f16)]
const A: f16 = 1.5;
fn main() {
    let b: f16 = A + 0.25;
    println!("{}", b as f32);
    println!("{}", (A * 2.0) as f64);
}
"#;

const NULLPTR_SRC: &str = r#"fn target() {}
const FN_AS_PTR: *const () = target as *const ();
const NULLP: *const u8 = core::ptr::null();
static WORD: u32 = 9;
const STATIC_PTR: *const u32 = &raw const WORD;
fn main() {
    println!("{:?}", NULLP.is_null());
    println!("{:?}", FN_AS_PTR.is_null());
    println!("{:?}", unsafe { *STATIC_PTR });
}
"#;

#[test]
fn run_mode_f16_constants_execute() {
    let dir = temp_dir("f16const");
    let src = fixture(&dir, "f16const.rs", F16_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "f16 program must run to completion; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "1.75\n3\n");
}

const F16_NAN_SRC: &str = r##"#![feature(f16)]
fn emit(x: f16) {
    println!("{}", x.to_bits());
    println!("{}", x.floor().to_bits());
    println!("{}", x.abs().to_bits());
    println!("{}", (x as f32).to_bits());
    println!("{}", (-x).to_bits());
}
fn main() {
    emit(f16::from_bits(0x7c01));
    emit(f16::from_bits(0x7e01));
    emit(f16::from_bits(0xfc01));
    emit(f16::from_bits(0xfe00));
    emit(f16::from_bits(0x7dff));
    let s = f32::from_bits(0x7f800001);
    println!("{}", s.to_bits());
    println!("{}", (s as f16).to_bits());
}
"##;

#[test]
fn run_mode_f16_nan_transport() {
    // NaN bit patterns survive f16 transport: `from_bits` /
    // `to_bits` and place copies are the identity, the bit
    // operations (`abs`, negation) preserve a signaling NaN, the
    // arithmetic operations (`floor`) quiet it with its payload
    // kept, and the widening cast to `f32` preserves it (the
    // native values below were measured on the pinned toolchain).
    let dir = temp_dir("f16nan");
    let src = fixture(&dir, "f16nan.rs", F16_NAN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "f16 NaN program must run to completion; stderr={stderr}"
    );
    let expected = "\
31745
32257
31745
2139103232
64513
32257
32257
32257
2143297536
65025
64513
65025
31745
4286586880
31745
65024
65024
32256
4290772992
32256
32255
32767
32255
2143281152
65023
2139095041
32256
";
    assert_eq!(String::from_utf8_lossy(&out.stdout), expected);
}

const F16_COPYSIGN_SRC: &str = r##"#![feature(f16)]
fn emit(m: u16, s: u16) {
    println!("{}", f16::from_bits(m).copysign(f16::from_bits(s)).to_bits());
}
fn main() {
    emit(0x3c00, 0x0000);
    emit(0x3c00, 0x8000);
    emit(0x0000, 0x8000);
    emit(0x8000, 0x3c00);
    emit(0x7c01, 0x8000);
    emit(0xfc01, 0x0000);
    emit(0x7e00, 0xbc00);
    emit(0x7c00, 0x8000);
    println!("{}", 1.5f32.copysign(-0.0f32).to_bits());
    println!("{}", f32::from_bits(0x7f800001).copysign(-1.0f32).to_bits());
    println!("{}", (-2.5f32).copysign(0.0f32).to_bits());
    println!("{}", 1.5f64.copysign(-0.0f64).to_bits());
    println!("{}", f64::from_bits(0x7ff0000000000001).copysign(-1.0f64).to_bits());
    println!("{}", (-2.5f64).copysign(0.0f64).to_bits());
}
"##;

#[test]
fn run_mode_f16_copysign() {
    // `copysign` takes the sign from the sign argument and keeps
    // the magnitude's bits exactly — NaN payloads (signaling
    // included) pass through unquieted, signed zeros and
    // infinities follow the sign argument. All widths are served
    // by the float-method builtin; the values below were measured
    // natively on the pinned toolchain.
    let dir = temp_dir("f16copysign");
    let src = fixture(&dir, "f16copysign.rs", F16_COPYSIGN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "copysign program must run to completion; stderr={stderr}"
    );
    let expected = "\
15360
48128
32768
0
64513
31745
65024
64512
3217031168
4286578689
1075838976
13832806255468478464
18442240474082181121
4612811918334230528
";
    assert_eq!(String::from_utf8_lossy(&out.stdout), expected);
}

const F128_COPYSIGN_SRC: &str = r##"#![feature(f128)]
fn emit(m: u128, s: u128) {
    println!("{}", f128::from_bits(m).copysign(f128::from_bits(s)).to_bits());
}
fn main() {
    emit(0x3fff0000000000000000000000000000, 0x00000000000000000000000000000000);
    emit(0x3fff0000000000000000000000000000, 0x80000000000000000000000000000000);
    emit(0x00000000000000000000000000000000, 0x80000000000000000000000000000000);
    emit(0x80000000000000000000000000000000, 0x3fff0000000000000000000000000000);
    emit(0x7fff0000000000000000000000000001, 0x80000000000000000000000000000000);
    emit(0xffff0000000000000000000000000001, 0x00000000000000000000000000000000);
    emit(0x7f800000000000000000000000000000, 0xbfff0000000000000000000000000000);
    emit(0x7fff0000000000000000000000000000, 0x80000000000000000000000000000000);
    println!("{}", 1.5f128.copysign(-0.0f128).to_bits());
    println!("{}", (-2.5f128).copysign(0.0f128).to_bits());
    println!("{}", unsafe { std::mem::transmute::<f128, u128>(2.5f128) });
    println!("{}", unsafe { std::mem::transmute::<u128, f128>(0x40004000000000000000000000000000u128) }.to_bits());
}
"##;

#[test]
fn run_mode_f128_copysign_and_transmute() {
    // `f128::copysign` runs the collected `copysignf128` body,
    // whose `to_bits` / `from_bits` are `Cast Transmute`
    // rvalues between `f128` and `u128`. The scalar-shape
    // reinterpretation covers the 128-bit float shape (an
    // `f128` value is stored as its exact bit pattern), so
    // both directions are exact: NaN payloads, signaling
    // included, pass through unquieted, and signed zeros and
    // infinities follow the sign argument. The values below
    // were measured natively on the pinned toolchain.
    let dir = temp_dir("f128copysign");
    let src = fixture(&dir, "f128copysign.rs", F128_COPYSIGN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "f128 copysign program must run to completion; stderr={stderr}"
    );
    let expected = "\
85065399433376081038215121361612832768
255206582893845312769902425077496938496
170141183460469231731687303715884105728
0
340277174624079928635746076935438991361
170135991163610696904058773219554885633
339617752923046005526922703901628039168
340277174624079928635746076935438991360
255209179042274580183716690325661548544
85071889804449249572750784482024357888
85071889804449249572750784482024357888
85071889804449249572750784482024357888
";
    assert_eq!(String::from_utf8_lossy(&out.stdout), expected);
}

#[test]
fn run_mode_null_and_fn_pointer_constants_execute() {
    let dir = temp_dir("nullptr");
    let src = fixture(&dir, "nullptr.rs", NULLPTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "null/fn pointer constant program must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "true\nfalse\n9\n"
    );
}

// ----- statics storage model (Lane CQ) -----

const DROP_15858_SRC: &str = r#"
#![allow(static_mut_refs)]
static mut DROP_RAN: bool = false;

trait Bar {
    fn do_something(&mut self);
}

struct BarImpl;

impl Bar for BarImpl {
    fn do_something(&mut self) {}
}

struct Foo<B: Bar>(B);

impl<B: Bar> Drop for Foo<B> {
    fn drop(&mut self) {
        unsafe {
            DROP_RAN = true;
        }
    }
}

fn main() {
    {
        let _x: Foo<BarImpl> = Foo(BarImpl);
    }
    unsafe {
        assert_eq!(DROP_RAN, true);
        println!("drop ran");
    }
}
"#;

const STATIC_INDEPENDENT_SRC: &str = r#"
static mut A: i32 = 1;
static mut B: i32 = 2;

fn main() {
    unsafe {
        A += 10;
        println!("a={} b={}", A, B);
    }
}
"#;

const STATIC_INTERIOR_PTR_SRC: &str = r#"
use std::cell::UnsafeCell;

#[repr(transparent)]
struct SyncUnsafeCellLike(UnsafeCell<u32>);
unsafe impl Sync for SyncUnsafeCellLike {}

#[repr(C)]
struct SyncPtr {
    foo: *mut u32,
}
unsafe impl Sync for SyncPtr {}

static mut STATIC: u32 = 42;
static PTR: SyncPtr = SyncPtr { foo: std::ptr::addr_of_mut!(STATIC) };

static WRAPPED: SyncUnsafeCellLike = SyncUnsafeCellLike(UnsafeCell::new(7));
static PTR2: SyncPtr =
    SyncPtr { foo: std::ptr::addr_of!(WRAPPED) as *mut u32 };

fn main() {
    unsafe {
        assert_eq!(*PTR.foo, 42);
        *PTR.foo = 0;
        assert_eq!(*PTR.foo, 0);
        assert_eq!(*PTR2.foo, 7);
        *PTR2.foo = 9;
        assert_eq!(*PTR2.foo, 9);
        println!("done");
    }
}
"#;

const STATIC_INTERIOR_REF_SRC: &str = r#"
struct S { a: usize }

static A: S = S { a: 3 };
static B: &'static usize = &A.a;
static D: [usize; 1] = [1];
static F: &'static usize = &D[0];

fn main() {
    assert_eq!(*B, A.a);
    assert_eq!(*F, D[0]);
    println!("b={} f={}", *B, *F);
}
"#;

const STATIC_INTERIOR_SRC: &str = r#"
use std::cell::RefCell;

static mut CELL: RefCell<i32> = RefCell::new(10);

fn main() {
    unsafe {
        *CELL.borrow_mut() = 5;
        let v = *CELL.borrow();
        *CELL.borrow_mut() = v + 2;
        println!("cell={}", *CELL.borrow());
    }
}
"#;

const STATIC_MUT_COUNTER_SRC: &str = r#"
static mut COUNT: i32 = 0;

fn bump() {
    unsafe { COUNT += 1; }
}

fn main() {
    bump();
    bump();
    bump();
    unsafe { println!("count={}", COUNT); }
}
"#;

#[test]
fn run_mode_generic_drop_trait_bound_15858() {
    let dir = temp_dir("drop-15858");
    let src = fixture(&dir, "drop_15858.rs", DROP_15858_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the 15858 destructor must run and write the static; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "drop ran\n",
        "main observes the static write the destructor made"
    );
}

#[test]
fn run_mode_static_interior_mutability() {
    let dir = temp_dir("static-interior");
    let src = fixture(&dir, "interior.rs", STATIC_INTERIOR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a RefCell living in a static must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "cell=7\n",
        "guard writes to a static are stored back and read by later borrows"
    );
}

#[test]
fn run_mode_static_interior_pointer_writes_through() {
    let dir = temp_dir("static-interior-ptr");
    let src = fixture(&dir, "interior_ptr.rs", STATIC_INTERIOR_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a raw pointer to a static (or into its wrapper) must write the store; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "done\n",
        "writes through interior pointers are observed by later reads"
    );
}

#[test]
fn run_mode_static_interior_reference_reads_the_field() {
    let dir = temp_dir("static-interior-ref");
    let src = fixture(&dir, "interior_ref.rs", STATIC_INTERIOR_REF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a reference into a static must read the field it names; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "b=3 f=1\n",
        "the store seed is the full initializer; the reference carries the projection"
    );
}

#[test]
fn run_mode_static_mut_counter_across_calls() {
    let dir = temp_dir("static-counter");
    let src = fixture(&dir, "counter.rs", STATIC_MUT_COUNTER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a static mut counter must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "count=3\n",
        "writes to a static persist across calls"
    );
}

#[test]
fn run_mode_two_statics_of_same_type_stay_independent() {
    let dir = temp_dir("static-independent");
    let src = fixture(&dir, "independent.rs", STATIC_INDEPENDENT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "two statics must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "a=11 b=2\n",
        "statics are keyed by identity, not by type or initializer"
    );
}

// ----- monomorphization validity, round 2 (Lane CS) -----

const BOUNDED_RECURSION_SRC: &str = r#"
fn fact(n: u64) -> u64 { if n == 0 { 1 } else { n * fact(n - 1) } }
fn down<T: Copy>(x: Option<T>) -> usize { match x { None => 0, Some(_) => down::<T>(x) } }
fn main() { let _ = fact(3); let _ = down::<i32>(Some(1)); }
"#;

const HUGE_PHANTOM_ONLY_SRC: &str = r#"
fn f<T>(_x: std::marker::PhantomData<T>) -> usize { std::mem::size_of::<usize>() }
fn main() { let _ = f::<[u8; usize::MAX]>(std::marker::PhantomData); }
"#;

const LARGE_BUT_LEGAL_SRC: &str = r#"
struct Big([u8; 1_000_000]);
fn main() { let _b = Big([7u8; 1_000_000]); }
"#;

const POINTER_TO_HUGE_SRC: &str = r#"
fn main() {
    let _p: *const [u8; usize::MAX] = core::ptr::null();
}
"#;

const HUGE_ZST_ARRAY_CONST_SRC: &str = r#"
fn main() {
    println!("{}", [(); usize::MAX].len());
}
"#;

const TOO_BIG_LOCAL_SRC: &str = r#"
fn main() { let _a = [0u8; usize::MAX]; }
"#;

const TOO_BIG_SIZE_OF_SRC: &str = r#"
struct S { x: [u8; usize::MAX] }
fn f() -> usize { std::mem::size_of::<S>() }
fn main() { let _ = f(); }
"#;

const UNBOUNDED_INSTANTIATION_SRC: &str = r#"
fn g<T>() { g::<Option<T>>(); }
fn main() { g::<i32>(); }
"#;

/// The shape of tests/ui/recursion/infinite-instantiation-via-
/// size-increasing-tuple.rs: each instantiation demands the
/// next with the type doubled, so the nesting depth grows only
/// linearly while the type expansion grows exponentially.
const SIZE_INCREASING_TUPLE_SRC: &str = r#"
trait Foo {
    fn recurse(&self);
}

impl<T> Foo for T {
    fn recurse(&self) {
        (self, self).recurse();
    }
}

fn main() {
    ().recurse();
}
"#;

/// The shape of tests/ui/recursion/infinite-instantiation-via-
/// nested-closures.rs, reduced: four methods pass the generic
/// parameter along a cycle, and two of the hops substitute a
/// fresh closure type that mentions the previous parameter
/// twice (as the enclosing substitution and as a capture), so
/// the demanded arguments' expansion doubles per cycle while
/// the nesting depth grows only linearly.
const SIZE_INCREASING_CLOSURES_SRC: &str = r#"
#![crate_type = "rlib"]
#[derive(Clone)]
struct A(B);

impl A {
    pub fn matches<F: Fn()>(&self, f: &F) {
        let A(term) = self;
        term.matches(f);
    }
}

#[derive(Clone)]
enum B {
    Variant1,
    Variant2(C),
}

impl B {
    pub fn matches<F: Fn()>(&self, f: &F) {
        match self {
            B::Variant2(factor) => factor.matches(&|| ()),
            _ => unreachable!(""),
        }
    }
}

#[derive(Clone)]
struct C(D);

impl C {
    pub fn matches<F: Fn()>(&self, f: &F) {
        let C(base) = self;
        base.matches(&|| {
            C(base.clone()).matches(f);
        });
    }
}

#[derive(Clone)]
struct D(Box<A>);

impl D {
    pub fn matches<F: Fn()>(&self, f: &F) {
        let D(a) = self;
        a.matches(f);
    }
}

pub fn matches() {
    A(B::Variant1).matches(&|| ());
}
"#;

const UNSIZED_NONFINAL_SRC: &str = r#"
#![feature(trivial_bounds)]
#![expect(trivial_bounds)]
struct TwoStrs(str, str)
where
    str: Sized;
fn main() {}
"#;

#[test]
fn convert_mode_bounded_generic_recursion_converts() {
    let dir = temp_dir("mono-bounded-recursion");
    let src = fixture(&dir, "bounded.rs", BOUNDED_RECURSION_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "plain recursion and same-instance generic recursion are \
         finite demand and must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_huge_type_named_only_through_phantom_data_converts() {
    let dir = temp_dir("mono-huge-phantom");
    let src = fixture(&dir, "huge_phantom.rs", HUGE_PHANTOM_ONLY_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a too-big type that is only named (PhantomData, an ordinary \
         generic argument) is never materialized and must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_large_but_legal_type_converts() {
    let dir = temp_dir("mono-large-legal");
    let src = fixture(&dir, "large_legal.rs", LARGE_BUT_LEGAL_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a large type within the target's bound must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_pointer_to_huge_type_converts() {
    // A pointer to a too-big type is itself an ordinary pointer;
    // merely pointing at the type does not materialize it, so
    // the program converts.
    let dir = temp_dir("mono-pointer-to-huge");
    let src = fixture(&dir, "pointer_to_huge.rs", POINTER_TO_HUGE_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a pointer to a huge type must convert; stderr={stderr}"
    );
}

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

#[test]
fn convert_mode_too_big_local_type_is_a_compile_failure() {
    let dir = temp_dir("mono-too-big-local");
    let src = fixture(&dir, "too_big_local.rs", TOO_BIG_LOCAL_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a local of a type too big for the target must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("too big for the target architecture"),
        "the failure must carry rustc's too-big wording; stderr={stderr}"
    );
}

#[test]
fn convert_mode_too_big_size_of_argument_is_a_compile_failure() {
    let dir = temp_dir("mono-too-big-size-of");
    let src = fixture(&dir, "too_big_size_of.rs", TOO_BIG_SIZE_OF_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "measuring a too-big type with size_of must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("too big for the target architecture"),
        "the failure must carry rustc's too-big wording; stderr={stderr}"
    );
}

#[test]
fn convert_mode_unbounded_self_instantiation_is_a_recursion_limit_failure() {
    let dir = temp_dir("mono-recursion-limit");
    let src = fixture(&dir, "unbounded.rs", UNBOUNDED_INSTANTIATION_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "unbounded self-instantiation must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("reached the recursion limit while instantiating"),
        "the failure must name the recursion limit; stderr={stderr}"
    );
}

#[test]
fn convert_mode_unsized_field_in_non_final_position_converts() {
    // A struct with an unsized field in a non-final position is
    // accepted under trivial bounds but can never be
    // materialized; asking the compiler for its layout creates
    // a delayed internal compiler error instead of returning
    // one. The too-big scan must screen the type out and leave
    // the compilation alone.
    let dir = temp_dir("mono-unsized-nonfinal");
    let src = fixture(&dir, "unsized_nonfinal.rs", UNSIZED_NONFINAL_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a never-materializable unsized-field struct must not fail compilation; stderr={stderr}"
    );
}

#[test]
fn run_mode_unbounded_self_instantiation_is_unsupported() {
    let dir = temp_dir("mono-recursion-limit-run");
    let src = fixture(&dir, "unbounded_run.rs", UNBOUNDED_INSTANTIATION_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "run mode reports a monomorphization failure as unsupported; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("reached the recursion limit while instantiating"),
        "the failure must name the recursion limit; stderr={stderr}"
    );
}

#[test]
fn convert_mode_size_increasing_tuple_is_a_size_limit_failure() {
    // The depth and instance-count caps cannot bind here: the
    // demand path nests the definition once per instantiation
    // while the type doubles, so conversion exhausts memory
    // long before either cap is reached. The expanded-size cap
    // must stop it, as the same recursion-limit failure rustc
    // reports.
    let dir = temp_dir("mono-size-increasing-tuple");
    let src = fixture(&dir, "size_increasing.rs", SIZE_INCREASING_TUPLE_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "size-increasing instantiation must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("reached the recursion limit while instantiating")
            && stderr.contains("(instantiation size limit)"),
        "the failure must name the instantiation size limit; stderr={stderr}"
    );
}

#[test]
fn convert_mode_size_increasing_closures_is_a_size_limit_failure() {
    // The closure-mediated shape of the same disease: the
    // demanded arguments' expansion doubles per cycle through
    // the closure arguments, so the size cap, not the depth
    // cap, is what stops the collection.
    let dir = temp_dir("mono-size-increasing-closures");
    let src = fixture(&dir, "size_increasing_closures.rs", SIZE_INCREASING_CLOSURES_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "closure-mediated size-increasing instantiation must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("reached the recursion limit while instantiating")
            && stderr.contains("(instantiation size limit)"),
        "the failure must name the instantiation size limit; stderr={stderr}"
    );
}

// ----- transparent-wrapper pointers (Lane DP) --------------------------------

const UNSAFECELL_GET_SRC: &str = r#"
use std::cell::UnsafeCell;
fn main() {
    let cell = UnsafeCell::new(5i32);
    unsafe {
        let p = cell.get();
        *p = 7;
        println!("{}", *p);
    }
}
"#;

const MAYBEUNINIT_PTR_SRC: &str = r#"
use std::mem::MaybeUninit;
fn main() {
    let mut slot = MaybeUninit::<i32>::uninit();
    unsafe {
        slot.as_mut_ptr().write(11);
        println!("{}", *slot.as_ptr());
    }
}
"#;

const CELL_AS_PTR_SRC: &str = r#"
use std::cell::Cell;
fn main() {
    let c = Cell::new(4i32);
    unsafe { *c.as_ptr() = 6; }
    println!("{}", c.get());
}
"#;

const REFCELL_COMPOUND_SRC: &str = r#"
use std::cell::RefCell;
fn main() {
    let cell = RefCell::new(1u32);
    *cell.borrow_mut() += 1;
    let v = *cell.borrow();
    println!("{v}");
}
"#;

const REFCELL_NESTED_SRC: &str = r#"
use std::cell::RefCell;
fn main() {
    let cell = RefCell::new(1i32);
    *cell.borrow_mut() = *cell.borrow() + 1;
    println!("{}", cell.borrow());
}
"#;

#[test]
fn run_mode_unsafecell_get_reads_and_writes_the_wrapped_value() {
    // `UnsafeCell::get` returns a pointer to the wrapped value
    // itself (the wrapper is transparent). Writing and reading
    // through it must reach the `i32`, not the `UnsafeCell`
    // aggregate: this used to fail with a type mismatch naming
    // `Adt("std::cell::UnsafeCell")` on both directions.
    let dir = temp_dir("unsafecell-get");
    let src = fixture(&dir, "unsafecell_get.rs", UNSAFECELL_GET_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "UnsafeCell::get program must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "7\n");
}

#[test]
fn run_mode_maybeuninit_pointer_reads_and_writes_the_wrapped_value() {
    let dir = temp_dir("maybeuninit-ptr");
    let src = fixture(&dir, "maybeuninit_ptr.rs", MAYBEUNINIT_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "MaybeUninit pointer program must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "11\n");
}

#[test]
fn run_mode_cell_as_ptr_writes_the_cell_value() {
    // `Cell::as_ptr` is `UnsafeCell::get` on the cell's inner
    // field: the builtin `Cell` value shape and the collected
    // `get` body must agree on the natural
    // `Cell { UnsafeCell { value } }` nesting.
    let dir = temp_dir("cell-as-ptr");
    let src = fixture(&dir, "cell_as_ptr.rs", CELL_AS_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cell::as_ptr program must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "6\n");
}

#[test]
fn run_mode_refcell_compound_assign_through_borrow_mut() {
    let dir = temp_dir("refcell-compound");
    let src = fixture(&dir, "refcell_compound.rs", REFCELL_COMPOUND_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "compound assignment through a RefMut guard must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "2\n");
}

#[test]
fn run_mode_refcell_nested_borrow_panics_like_native() {
    // `*cell.borrow_mut() = *cell.borrow() + 1` panics under real
    // Rust too (the shared borrow is taken while the mutable one
    // is still live): the guest panic is the correct outcome.
    let dir = temp_dir("refcell-nested");
    let src = fixture(&dir, "refcell_nested.rs", REFCELL_NESTED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "nested borrow must panic like native; stderr={stderr}"
    );
    assert!(
        stderr.contains("RefCell already borrowed"),
        "the panic must be the borrow conflict; stderr={stderr}"
    );
}
// ----- catch_unwind (Lane DH) ------------------------------------------------

const CATCH_OK_SRC: &str = r#"
fn main() {
    let r = std::panic::catch_unwind(|| 40 + 2);
    match r {
        Ok(v) => println!("ok {v}"),
        Err(_) => println!("err"),
    }
}
"#;

const CATCH_LITERAL_SRC: &str = r#"
fn main() {
    let r = std::panic::catch_unwind(|| -> i32 { panic!("boom") });
    match r {
        Ok(_) => println!("ok"),
        Err(e) => {
            if let Some(s) = e.downcast_ref::<&str>() {
                println!("payload str: {s}");
            } else {
                println!("payload not str");
            }
            println!("is String: {}", e.is::<String>());
        }
    }
}
"#;

const CATCH_FORMATTED_SRC: &str = r#"
fn main() {
    let mut x = 0usize;
    for i in 0..7 { x += i; }
    let r = std::panic::catch_unwind(|| -> i32 { panic!("fmt {}", x) });
    match r {
        Ok(_) => println!("ok"),
        Err(e) => {
            println!("is str: {}", e.is::<&str>());
            if let Some(s) = e.downcast_ref::<String>() {
                println!("payload String: {s}");
            } else {
                println!("payload not String");
            }
        }
    }
}
"#;

const CATCH_PANIC_ANY_SRC: &str = r#"
fn main() {
    let r = std::panic::catch_unwind(|| -> i32 { std::panic::panic_any(42i32) });
    match r {
        Ok(_) => println!("ok"),
        Err(e) => {
            if let Some(v) = e.downcast_ref::<i32>() {
                println!("payload i32: {v}");
            } else {
                println!("payload not i32");
            }
            match e.downcast::<i32>() {
                Ok(v) => println!("downcast value: {}", *v),
                Err(_) => println!("downcast failed"),
            }
        }
    }
}
"#;

const CATCH_RESUME_SRC: &str = r#"
fn main() {
    let r = std::panic::catch_unwind(|| -> i32 { panic!("again") });
    println!("caught: {}", r.is_err());
    if let Err(e) = r {
        std::panic::resume_unwind(e);
    }
    println!("unreachable");
}
"#;

const CATCH_ASSERT_UNWIND_SAFE_SRC: &str = r#"
use std::panic::{catch_unwind, AssertUnwindSafe};
fn main() {
    let mut state = 10;
    let r = catch_unwind(AssertUnwindSafe(|| {
        state = 99;
        panic!("inner");
    }));
    println!("caught={} state={}", r.is_err(), state);
}
"#;

const CATCH_NESTED_SRC: &str = r#"
use std::panic::catch_unwind;
fn main() {
    let r = catch_unwind(|| {
        let inner = catch_unwind(|| -> i32 { panic!("deep") });
        match inner { Ok(_) => 1, Err(_) => 2 }
    });
    match r {
        Ok(v) => println!("outer ok {v}"),
        Err(_) => println!("outer err"),
    }
    let r2 = catch_unwind(|| {
        let inner = catch_unwind(|| -> i32 { panic!("deep2") });
        if let Err(e) = inner { std::panic::resume_unwind(e); }
        0
    });
    match r2 {
        Ok(_) => println!("r2 ok"),
        Err(e) => {
            if let Some(s) = e.downcast_ref::<&str>() {
                println!("r2 payload: {s}");
            } else {
                println!("r2 payload other");
            }
        }
    }
}
"#;

const CATCH_OVERFLOW_SRC: &str = r#"
fn main() {
    let mut x = 0i32;
    for i in 0..7 { x += i; }
    let r = std::panic::catch_unwind(|| { let a = i32::MAX; a + x });
    match r {
        Ok(v) => println!("ok {v}"),
        Err(e) => {
            if let Some(s) = e.downcast_ref::<&str>() {
                println!("overflow payload: {s}");
            } else {
                println!("overflow payload other");
            }
        }
    }
}
"#;

const CATCH_STATE_SRC: &str = r#"
use std::panic::catch_unwind;
use std::sync::atomic::{AtomicI32, Ordering};
static MARK: AtomicI32 = AtomicI32::new(0);
fn main() {
    MARK.store(5, Ordering::SeqCst);
    let _ = catch_unwind(|| {
        MARK.store(7, Ordering::SeqCst);
        panic!("x");
    });
    println!("mark={}", MARK.load(Ordering::SeqCst));
    println!("after");
}
"#;

fn run_ok(tag: &str, src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let src_path = fixture(&dir, "case.rs", src);
    let out = run_driver(&src_path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "catch_unwind case {tag} must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        expected_stdout,
        "catch_unwind case {tag} stdout mismatch; stderr={stderr}"
    );
}

#[test]
fn run_mode_catch_unwind_ok_path() {
    run_ok("catch-ok", CATCH_OK_SRC, "ok 42\n");
}

#[test]
fn run_mode_catch_unwind_literal_payload_is_str() {
    run_ok(
        "catch-literal",
        CATCH_LITERAL_SRC,
        "payload str: boom\nis String: false\n",
    );
}

#[test]
fn run_mode_catch_unwind_formatted_payload_is_string() {
    run_ok(
        "catch-formatted",
        CATCH_FORMATTED_SRC,
        "is str: false\npayload String: fmt 21\n",
    );
}

#[test]
fn run_mode_catch_unwind_panic_any_payload() {
    run_ok(
        "catch-panic-any",
        CATCH_PANIC_ANY_SRC,
        "payload i32: 42\ndowncast value: 42\n",
    );
}

#[test]
fn run_mode_catch_unwind_assert_unwind_safe() {
    run_ok(
        "catch-aus",
        CATCH_ASSERT_UNWIND_SAFE_SRC,
        "caught=true state=99\n",
    );
}

#[test]
fn run_mode_catch_unwind_nested_and_resume_through() {
    run_ok(
        "catch-nested",
        CATCH_NESTED_SRC,
        "outer ok 2\nr2 payload: deep2\n",
    );
}

#[test]
fn run_mode_catch_unwind_overflow_payload_message() {
    run_ok(
        "catch-overflow",
        CATCH_OVERFLOW_SRC,
        "overflow payload: attempt to add with overflow\n",
    );
}

#[test]
fn run_mode_catch_unwind_state_survives_the_catch() {
    run_ok("catch-state", CATCH_STATE_SRC, "mark=7\nafter\n");
}

// ----- unwind drops (Lane EW) ------------------------------------------------
// A panic unwinding through a frame runs the cleanup blocks MIR
// built for it: the destructors of the locals live at the panic
// point, in reverse declaration order per frame, innermost frame
// first. A destructor that itself panics while unwinding aborts
// the run (a native double panic); a destructor that panics on the
// normal path is an ordinary, catchable panic.

const UNWIND_DROP_SINGLE_SRC: &str = r#"
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn boom() { let _m = M(7); panic!("boom"); }
fn main() {
    let r = std::panic::catch_unwind(boom);
    println!("caught {}", r.is_err());
}
"#;

const UNWIND_DROP_NESTED_SRC: &str = r#"
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn inner() { let _a = M(1); let _b = M(2); panic!("deep"); }
fn middle() { let _c = M(3); inner(); let _d = M(4); }
fn main() {
    let r = std::panic::catch_unwind(middle);
    println!("caught {}", r.is_err());
    let _e = M(5);
    println!("end");
}
"#;

const UNWIND_DROP_PAYLOAD_SRC: &str = r#"
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let r = std::panic::catch_unwind(|| -> i32 {
        let _m = M(3);
        panic!("boom")
    });
    match r {
        Ok(_) => println!("ok"),
        Err(e) => {
            if let Some(s) = e.downcast_ref::<&str>() {
                println!("payload str: {s}");
            } else {
                println!("payload not str");
            }
        }
    }
}
"#;

const UNWIND_DROP_UNCAUGHT_SRC: &str = r#"
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn boom() { let _m = M(7); panic!("boom"); }
fn main() { let _x = M(9); boom(); }
"#;

const UNWIND_DROP_PANICS_SRC: &str = r#"
struct Bad;
impl Drop for Bad { fn drop(&mut self) { panic!("dtor panic"); } }
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn boom() { let _b = Bad; let _m = M(1); panic!("orig"); }
fn main() {
    let r = std::panic::catch_unwind(boom);
    println!("caught {}", r.is_err());
}
"#;

const DROP_PANIC_NORMAL_SRC: &str = r#"
struct Bad;
impl Drop for Bad { fn drop(&mut self) { println!("bad dtor runs"); panic!("dtor panic"); } }
fn main() {
    let r = std::panic::catch_unwind(|| { let _b = Bad; });
    println!("caught {}", r.is_err());
}
"#;

#[test]
fn run_mode_unwind_drops_single_frame() {
    run_ok(
        "unwind-drop-single",
        UNWIND_DROP_SINGLE_SRC,
        "dtor 7\ncaught true\n",
    );
}

#[test]
fn run_mode_unwind_drops_nested_frames_in_order() {
    // Reverse declaration order within a frame (2 before 1), the
    // inner frame before its caller (3), a local declared after
    // the panicking call is never built (no dtor 4), and the
    // normal scope-end drop after the catch is unaffected (5,
    // after "end").
    run_ok(
        "unwind-drop-nested",
        UNWIND_DROP_NESTED_SRC,
        "dtor 2\ndtor 1\ndtor 3\ncaught true\nend\ndtor 5\n",
    );
}

#[test]
fn run_mode_unwind_drops_and_payload_downcast() {
    // The destructor runs during the unwind, before the catch
    // handler observes the payload; the payload itself (and its
    // downcast) is exactly as without drops.
    run_ok(
        "unwind-drop-payload",
        UNWIND_DROP_PAYLOAD_SRC,
        "dtor 3\npayload str: boom\n",
    );
}

#[test]
fn run_mode_uncaught_panic_runs_drops() {
    let dir = temp_dir("unwind-drop-uncaught");
    let src = fixture(&dir, "case.rs", UNWIND_DROP_UNCAUGHT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "uncaught panic must exit 101; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "dtor 7\ndtor 9\n",
        "drops run innermost frame first on an uncaught panic; stderr={stderr}"
    );
}

#[test]
fn run_mode_unwind_drop_panics_aborts_uncaught_by_catch_unwind() {
    // Unwinding drops M(1) first (reverse order), then Bad's
    // destructor panics while unwinding: a double panic. The run
    // aborts (101, abort reason on stderr) and the surrounding
    // catch_unwind does NOT observe it ("caught" never prints),
    // exactly as a native abort is not a catchable panic.
    let dir = temp_dir("unwind-drop-panics");
    let src = fixture(&dir, "case.rs", UNWIND_DROP_PANICS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "a panic in an unwind destructor must abort with 101; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "dtor 1\n",
        "only the first unwind drop runs before the abort; stderr={stderr}"
    );
    assert!(
        stderr.contains("aborted"),
        "the abort must be named on stderr; stderr={stderr}"
    );
}

#[test]
fn run_mode_drop_panic_on_normal_path_is_caught() {
    run_ok(
        "drop-panic-normal",
        DROP_PANIC_NORMAL_SRC,
        "bad dtor runs\ncaught true\n",
    );
}

#[test]
fn run_mode_resume_unwind_repanics_out_of_the_catch() {
    let dir = temp_dir("catch-resume");
    let src = fixture(&dir, "resume.rs", CATCH_RESUME_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "a resumed payload escapes as a guest panic (101); stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "caught: true\n",
        "output before the resume is kept; the line after it never runs"
    );
    assert!(
        stderr.contains("again"),
        "the resumed panic reports the payload message; stderr={stderr}"
    );
}

// ----- Waker construction (Lane DQ) ------------------------------------------

const WAKER_FROM_SRC: &str = r#"
use std::sync::Arc;
use std::task::{Wake, Waker};

struct MyWaker;
impl Wake for MyWaker {
    fn wake(self: Arc<Self>) {}
}

fn main() {
    let waker = Waker::from(Arc::new(MyWaker));
    let waker2 = waker.clone();
    std::mem::forget(waker);
    std::mem::forget(waker2);
    println!("waker ok");
}
"#;

#[test]
fn run_mode_waker_from_arc_constructs_and_clones() {
    // `Waker::from(Arc<T: Wake>)` is interpreter runtime support:
    // the modelled waker carries the Arc value, so constructing
    // it and cloning it (a value copy, exactly how the modelled
    // `Arc` itself clones) run to completion. The wakers are
    // forgotten rather than dropped: dropping one runs the
    // shared-ownership glue the value model does not account
    // for - the same limit as dropping an `Arc` itself - and
    // handing the waker to `Context::from_waker` is blocked by
    // the reference-to-reference transmute in that core body,
    // not by the waker value.
    let dir = temp_dir("wakerfrom");
    let src = fixture(&dir, "wakerfrom.rs", WAKER_FROM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "waker construction and clone must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "waker ok\n",
        "the guest's stdout is the driver's stdout"
    );
}

const TRANSMUTE_FLOAT_INT_SRC: &str = r#"
fn inp32() -> f32 { let mut s = 0.0f32; let mut i = 0; while i < 7 { s += 0.5; i += 1; } s }
fn inp64() -> f64 { let mut s = 0.0f64; let mut i = 0; while i < 7 { s += 1.0; i += 1; } s }
fn main() {
    let f = inp32();
    let bits: u32 = unsafe { std::mem::transmute(f) };
    let back: f32 = unsafe { std::mem::transmute(bits) };
    let g = inp64();
    let gbits: u64 = unsafe { std::mem::transmute(g) };
    let sg: i64 = unsafe { std::mem::transmute(g) };
    println!("{:x} {} {:x} {}", bits, back, gbits, sg);
    println!("{:x} {} {}", f.to_bits(), f32::from_bits(bits), f64::from_bits(gbits));
}
"#;

#[test]
fn run_mode_transmute_float_int_bit_reinterpretation() {
    run_ok(
        "transmute-float-int",
        TRANSMUTE_FLOAT_INT_SRC,
        "40600000 3.5 401c000000000000 4619567317775286272\n40600000 3.5 7\n",
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

const TRANSMUTE_FAT_SLICE_SRC: &str = r#"
struct W(*const [u8]);
fn main() {
    let s: &[u8] = &[10, 20, 30, 40];
    let w: [usize; 2] = unsafe { std::mem::transmute(s) };
    println!("{}", w[1]);
    let back: &[u8] = unsafe { std::mem::transmute(w) };
    println!("{} {}", back.len(), back[2]);
    let lit: &str = "world!";
    let t: (usize, usize) = unsafe { std::mem::transmute(lit) };
    println!("{}", t.1);
    let sback: &str = unsafe { std::mem::transmute(t) };
    println!("{}", sback);
    let s3: &[u8] = &[1, 2, 3];
    let wrap = W(s3 as *const [u8]);
    let words: [usize; 2] = unsafe { std::mem::transmute(wrap) };
    println!("{}", words[1]);
    let w2: W = unsafe { std::mem::transmute(words) };
    unsafe { println!("{} {}", (&*w2.0).len(), (*w2.0)[1]) };
}
"#;

#[test]
fn run_mode_transmute_fat_pointer_words() {
    // A fat pointer transmutes to its two words and back: the
    // metadata word is the length, and constructing the words
    // returns the very pointer that flattened to them (its
    // pointee still reads). A newtype wrapper over a fat
    // pointer and a tuple target take the same shape path.
    run_ok(
        "transmute-fat-slice",
        TRANSMUTE_FAT_SLICE_SRC,
        "4\n4 30\n6\nworld!\n3\n3 2\n",
    );
}

const TRANSMUTE_FAT_DYN_UNION_SRC: &str = r#"
use std::fmt::Debug;
union U {
    p: *const [u8],
    w: [usize; 2],
}
fn main() {
    let x = 42u32;
    let d: &dyn Debug = &x;
    let w: [usize; 2] = unsafe { std::mem::transmute(d) };
    let back: &dyn Debug = unsafe { std::mem::transmute(w) };
    println!("{:?}", back);
    let s: &[u8] = &[9, 8, 7];
    let u = U { p: s as *const [u8] };
    unsafe { println!("{}", u.w[1]) };
}
"#;

#[test]
fn run_mode_transmute_dyn_pointer_and_union_fat_field() {
    // A `dyn` fat pointer's metadata word is a vtable identity
    // only the fat-value table resolves: the round trip
    // through the words returns the original pointer, whose
    // dispatch still works. A union field holding a fat
    // pointer reads back through the other field's words.
    run_ok(
        "transmute-fat-dyn-union",
        TRANSMUTE_FAT_DYN_UNION_SRC,
        "42\n3\n",
    );
}

const TRANSMUTE_WRAPPER_SRC: &str = r#"
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::num::Wrapping;
struct Padded { x: u32, _m: PhantomData<u8> }
fn inp() -> u32 { let mut s = 0u32; let mut i = 0; while i < 41 { s += 1; i += 1; } s }
fn main() {
    let n = inp();
    let w: Wrapping<u32> = unsafe { std::mem::transmute(n) };
    let back: u32 = unsafe { std::mem::transmute(w) };
    let wf: Wrapping<f32> = unsafe { std::mem::transmute(0x3fc00000u32) };
    let p: Padded = unsafe { std::mem::transmute(n + 1) };
    let md: ManuallyDrop<u32> = unsafe { std::mem::transmute(n) };
    let mback: u32 = unsafe { std::mem::transmute(md) };
    println!("{} {} {} {} {}", w.0, back, wf.0, p.x, mback);
}
"#;

#[test]
fn run_mode_transmute_wrapper_adts() {
    run_ok(
        "transmute-wrappers",
        TRANSMUTE_WRAPPER_SRC,
        "41 41 1.5 42 41\n",
    );
}

const TRANSMUTE_BOX_SRC: &str = r#"
fn inp() -> u32 { let mut s = 0u32; let mut i = 0; while i < 41 { s += 1; i += 1; } s }
fn main() {
    let n = inp();
    let p: *mut u32 = unsafe { std::mem::transmute(Box::new(n)) };
    unsafe { *p += 1; }
    let q: *const u32 = unsafe { std::mem::transmute(Box::new(n + 2)) };
    println!("{} {}", unsafe { *p }, unsafe { *q });
}
"#;

#[test]
fn run_mode_transmute_box_to_raw_pointer() {
    run_ok("transmute-box", TRANSMUTE_BOX_SRC, "42 43\n");
}

const TRANSMUTE_RAW_TO_BOX_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};
static DROPS: AtomicUsize = AtomicUsize::new(0);
struct D(u32);
impl Drop for D {
    fn drop(&mut self) { DROPS.fetch_add(1, Ordering::SeqCst); }
}
fn main() {
    let p: *mut D = Box::into_raw(Box::new(D(41)));
    let mut b: Box<D> = unsafe { std::mem::transmute(p) };
    b.0 += 1;
    println!("{}", b.0);
    drop(b);
    println!("drops={}", DROPS.load(Ordering::SeqCst));
    let p2: *mut D = Box::into_raw(Box::new(D(7)));
    let b2: Box<D> = unsafe { std::mem::transmute(p2) };
    let p3: *mut D = unsafe { std::mem::transmute(b2) };
    let b3: Box<D> = unsafe { std::mem::transmute(p3) };
    println!("{}", b3.0);
    drop(b3);
    println!("drops={}", DROPS.load(Ordering::SeqCst));
}
"#;

#[test]
fn run_mode_transmute_raw_pointer_to_box() {
    run_ok(
        "transmute-raw-to-box",
        TRANSMUTE_RAW_TO_BOX_SRC,
        "42\ndrops=1\n7\ndrops=2\n",
    );
}

const TRANSMUTE_RAW_TO_ZST_FIRST_WRAPPER_SRC: &str = r#"
#[repr(transparent)]
struct Inner(*mut u32);
#[repr(C)]
struct Outer((), Inner);
fn main() {
    let mut x: u32 = 42;
    let p = &mut x as *mut u32;
    let outer: Outer = unsafe { std::mem::transmute(p) };
    let q = outer.1.0;
    unsafe { *q += 1; }
    println!("{}", x);
    let outer2: Outer = unsafe { std::mem::transmute(q) };
    println!("{}", unsafe { *outer2.1.0 });
}
"#;

#[test]
fn run_mode_transmute_raw_pointer_to_wrapper_with_leading_zst() {
    // The pointer chain enters the target at field 1 (a
    // zero-sized field sits before the payload layer), so
    // the rebuilt value must hold the pointer at that index
    // with the zero-sized field's value beside it — not in a
    // one-field aggregate at field 0, where reading `.1.0`
    // finds a field that was never constructed.
    run_ok(
        "transmute-raw-to-zst-first-wrapper",
        TRANSMUTE_RAW_TO_ZST_FIRST_WRAPPER_SRC,
        "43\n43\n",
    );
}

const TRANSMUTE_DEEP_ZST_CHAIN_SRC: &str = r#"
#[repr(C)]
struct Inner2((), *mut u32);
#[repr(C)]
struct Outer2(Inner2, ());
fn main() {
    let mut x: u32 = 42;
    let p = &mut x as *mut u32;
    let outer: Outer2 = unsafe { std::mem::transmute(p) };
    let q = outer.0.1;
    unsafe { *q += 1; }
    println!("{}", x);
}
"#;

#[test]
fn run_mode_transmute_deep_zst_chain_is_explicit_unsupported() {
    // The chain enters a deeper layer at a non-zero field;
    // rebuilding it would need an aggregate nested inside
    // an operand, which the IR does not express. The
    // transmute stays an explicit unsupported instead of
    // building a value whose fields were never constructed.
    let dir = temp_dir("transmute-deep-zst-chain");
    let src = fixture(&dir, "case.rs", TRANSMUTE_DEEP_ZST_CHAIN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "deep non-zero chain transmute must be a named unsupported; stderr={stderr}"
    );
    assert!(
        stderr.contains("Cast Transmute"),
        "stderr must name the transmute; stderr={stderr}"
    );
}

const TRANSMUTE_CONSTANTS_SRC: &str = r#"
fn main() {
    let a: u32 = unsafe { std::mem::transmute(1.5f32) };
    let b: f32 = unsafe { std::mem::transmute(0x3fc00000u32) };
    let c: u64 = unsafe { std::mem::transmute(-2.5f64) };
    let d: u8 = unsafe { std::mem::transmute(true) };
    let e: [u8; 4] = unsafe { std::mem::transmute(0x04030201u32) };
    println!("{:x} {} {:x} {} {:?}", a, b, c, d, e);
}
"#;

#[test]
fn run_mode_transmute_constant_operands() {
    run_ok(
        "transmute-constants",
        TRANSMUTE_CONSTANTS_SRC,
        "3fc00000 1.5 c004000000000000 1 [1, 2, 3, 4]\n",
    );
}

const TRANSMUTE_PACKED_STRUCT_SRC: &str = r#"
use std::mem;
#[repr(packed)]
struct S5 { a: u8, b: u32 }
fn main() {
    unsafe {
        let s5 = S5 { a: 1, b: 0xff_00_00_ff };
        let transd: [u8; 5] = mem::transmute(s5);
        assert_eq!(transd, [1, 0xff, 0, 0, 0xff]);
        let q: S5 = mem::transmute([9u8, 8, 7, 6, 5]);
        let qa = q.a;
        let qb = q.b;
        println!("{} {}", qa, qb);
    }
}
"#;

#[test]
fn run_mode_transmute_packed_struct_bytes() {
    // A multi-field packed struct transmuted to its byte
    // array and back: the fields sit at their recorded
    // layout offsets with no padding in between.
    run_ok(
        "transmute-packed-struct",
        TRANSMUTE_PACKED_STRUCT_SRC,
        "9 84281096\n",
    );
}

const TRANSMUTE_NICHE_ENUM_SRC: &str = r#"
#[repr(u16)]
enum Mix { A, B(u16) }
enum Nested { C(Mix), D, E }
struct Bar { a: u16, b: u16 }
fn code(n: &Nested) -> u32 {
    match n {
        Nested::C(Mix::A) => 0,
        Nested::C(Mix::B(x)) => 100 + (*x as u32),
        Nested::D => 2,
        Nested::E => 3,
    }
}
fn main() {
    let a: Nested = unsafe { std::mem::transmute(Bar { a: 0, b: 0 }) };
    let b: Nested = unsafe { std::mem::transmute(Bar { a: 1, b: 100 }) };
    let c: Nested = unsafe { std::mem::transmute(Bar { a: 2, b: 0 }) };
    let d: Nested = unsafe { std::mem::transmute(Bar { a: 3, b: 0 }) };
    let e: Nested = unsafe { std::mem::transmute(0u32) };
    println!("{} {} {} {} {}", code(&a), code(&b), code(&c), code(&d), code(&e));
}
"#;

#[test]
fn run_mode_transmute_niche_enum_from_struct_and_int() {
    // A struct whose first field is a payload enum's tag
    // transmuted to an outer enum whose niche lives inside
    // that payload: the tag selects among the payload's own
    // variants and the outer enum's extra variants, from a
    // struct operand and from a bare integer alike. Under
    // debug assertions the compiler also reads the input's
    // bytes through a subslice to validate the construction.
    run_ok(
        "transmute-niche-enum",
        TRANSMUTE_NICHE_ENUM_SRC,
        "0 200 2 3 0\n",
    );
}

const TRANSMUTE_TAGGED_ENUM_SRC: &str = r#"
#[repr(u16)]
enum Mix { A, B(u16) }
struct Bar { a: u16, b: u16 }
fn code(m: &Mix) -> u32 {
    match m {
        Mix::A => 0,
        Mix::B(x) => 100 + (*x as u32),
    }
}
fn main() {
    let a: Mix = unsafe { std::mem::transmute(Bar { a: 0, b: 0 }) };
    let b: Mix = unsafe { std::mem::transmute(Bar { a: 1, b: 100 }) };
    println!("{} {}", code(&a), code(&b));
}
"#;

#[test]
fn run_mode_transmute_tagged_enum_from_struct() {
    // A struct transmuted to a direct-tag enum: the tag
    // field at its recorded offset selects the variant and
    // the remaining fields land at their layout offsets.
    run_ok(
        "transmute-tagged-enum",
        TRANSMUTE_TAGGED_ENUM_SRC,
        "0 200\n",
    );
}

const TRANSMUTE_OPTION_BIGGER_NICHE_SRC: &str = r#"
#[derive(Debug, PartialEq)]
#[repr(u32)]
enum Foo { A = 2, B }
fn main() {
    let a: Option<Foo> = unsafe { std::mem::transmute::<u32, Option<Foo>>(2) };
    let b: Option<Foo> = unsafe { std::mem::transmute::<u32, Option<Foo>>(3) };
    assert_eq!(a, Some(Foo::A));
    assert_eq!(b, Some(Foo::B));
    println!("ok");
}
"#;

#[test]
fn run_mode_transmute_int_to_option_of_fieldless_enum() {
    // An integer transmuted to `Option` of a fieldless enum
    // whose discriminants do not start at zero: the payload
    // enum's own scalar is the niche, its discriminant
    // patterns select the `Some` variants.
    run_ok(
        "transmute-option-bigger-niche",
        TRANSMUTE_OPTION_BIGGER_NICHE_SRC,
        "ok\n",
    );
}

const TRANSMUTE_WIDE_SRC: &str = r#"
fn main() {
    let big: u128 = unsafe { std::mem::transmute(-1i128) };
    let neg: i8 = unsafe { std::mem::transmute(0xffu8) };
    let s16: i16 = unsafe { std::mem::transmute(0xffffu16) };
    println!("{:x} {} {}", big, neg, s16);
}
"#;

#[test]
fn run_mode_transmute_wide_and_signed() {
    run_ok(
        "transmute-wide",
        TRANSMUTE_WIDE_SRC,
        "ffffffffffffffffffffffffffffffff -1 -1\n",
    );
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

const NONZERO_TRANSMUTE_SRC: &str = r#"
use std::num::NonZero;
fn main() {
    let x: NonZero<u32> = unsafe { std::mem::transmute(42u32) };
    println!("nz {}", x.get());
    let y: u32 = unsafe { std::mem::transmute(x) };
    println!("back {}", y);
    let z: NonZero<i32> = unsafe { std::mem::transmute(-5i32) };
    println!("neg {}", z.get());
    let w: NonZero<u16> = unsafe { std::mem::transmute(7u16) };
    println!("w {}", w.get());
}
"#;

#[test]
fn run_mode_nonzero_transmute_roundtrip() {
    // `NonZero<T>`'s only field is declared as the projection
    // alias `<T as ZeroablePrimitive>::NonZeroInner`; the driver
    // records the compiler-normalized field type of each
    // monomorphic instance, so the transmutes (and `get`, itself
    // a transmute in core) execute.
    run_ok("nonzero-transmute", NONZERO_TRANSMUTE_SRC, "nz 42\nback 42\nneg -5\nw 7\n");
}

const NONZERO_FIELD_SRC: &str = r#"
use std::num::NonZero;
struct Wrap { tag: u8, val: NonZero<u32> }
fn main() {
    let v: NonZero<u32> = unsafe { std::mem::transmute(9u32) };
    let w = Wrap { tag: 3, val: v };
    println!("field {} {}", w.tag, w.val.get());
    let raw: u32 = unsafe { std::mem::transmute(w.val) };
    println!("raw {}", raw);
}
"#;

#[test]
fn run_mode_nonzero_field_in_struct() {
    run_ok("nonzero-field", NONZERO_FIELD_SRC, "field 3 9\nraw 9\n");
}

const NONZERO_LAYOUT_SRC: &str = r#"
use std::num::NonZero;
fn main() {
    println!("{} {}", std::mem::size_of::<Option<NonZero<u32>>>(), std::mem::size_of::<u32>());
    println!("{}", std::mem::size_of::<Option<NonZero<u64>>>());
    println!("{}", std::mem::size_of::<NonZero<u32>>());
}
"#;

#[test]
fn run_mode_nonzero_layout_sizes() {
    run_ok("nonzero-layout", NONZERO_LAYOUT_SRC, "4 4\n8\n4\n");
}

const NONZERO_NEW_SRC: &str = r#"
use std::num::NonZero;
fn main() {
    let x: NonZero<u32> = unsafe { std::mem::transmute(42u32) };
    println!("nz {}", x.get());
    let y: u32 = unsafe { std::mem::transmute(x) };
    println!("back {}", y);
    println!("new {}", NonZero::new(7).unwrap().get());
}
"#;

#[test]
fn run_mode_nonzero_new_constructor() {
    // `NonZero::new` itself is a transmute between the bare
    // integer and `Option<NonZero<T>>` in core: it converts
    // through the enum's niche encoding, which the driver
    // records from the compiler's layout and the interpreter
    // reinterprets through.
    run_ok("nonzero-new", NONZERO_NEW_SRC, "nz 42\nback 42\nnew 7\n");
}

const NONZERO_OPTION_SRC: &str = r#"
use std::num::NonZero;
fn main() {
    println!("size {} {}", std::mem::size_of::<Option<NonZero<u32>>>(), std::mem::size_of::<u32>());
    let a: Option<NonZero<u32>> = NonZero::new(5);
    let bits: u32 = unsafe { std::mem::transmute(a) };
    println!("bits {}", bits);
    let b: Option<NonZero<u32>> = unsafe { std::mem::transmute(0u32) };
    println!("none {}", b.is_none());
}
"#;

#[test]
fn run_mode_option_nonzero_transmutes() {
    run_ok("option-nonzero", NONZERO_OPTION_SRC, "size 4 4\nbits 5\nnone true\n");
}

const NONZERO_FIELD_NEW_SRC: &str = r#"
use std::num::NonZero;
struct Wrap { tag: u8, val: NonZero<u32> }
fn main() {
    let w = Wrap { tag: 3, val: NonZero::new(9).unwrap() };
    println!("field {} {}", w.tag, w.val.get());
    let raw: u32 = unsafe { std::mem::transmute(w.val) };
    println!("raw {}", raw);
}
"#;

#[test]
fn run_mode_nonzero_field_constructed_by_new() {
    run_ok("nonzero-field-new", NONZERO_FIELD_NEW_SRC, "field 3 9\nraw 9\n");
}

const NICHE_GENERAL_SRC: &str = r#"
use std::num::NonZero;
fn main() {
    let nz = NonZero::new(1u64 << 40);
    let bits: u64 = unsafe { std::mem::transmute(nz) };
    println!("bits {}", bits);
    let opt: Option<NonZero<u32>> = unsafe { std::mem::transmute(0u32) };
    println!("zero-none {}", opt.is_none());
    let nz8 = NonZero::new(200u8);
    println!("nz8 {}", nz8.map(|n| n.get()).unwrap_or(0));
    let s: Option<NonZero<u32>> = Some(NonZero::new(77u32).unwrap());
    let sbits: u32 = unsafe { std::mem::transmute(s) };
    println!("sbits {}", sbits);
    let b2: Option<bool> = unsafe { std::mem::transmute(2u8) };
    let back: u8 = unsafe { std::mem::transmute(b2) };
    println!("bback {}", back);
    match NonZero::new(0u32) {
        None => println!("new0 none"),
        Some(n) => println!("new0 some {}", n.get()),
    }
}
"#;

#[test]
fn run_mode_niche_enum_transmutes_general() {
    // The niche encoding is not NonZero-specific: widths
    // other than 32 bits and `Option<bool>` (whose None is
    // the value 2, not 0) follow the same recorded encoding.
    run_ok(
        "niche-general",
        NICHE_GENERAL_SRC,
        "bits 1099511627776\nzero-none true\nnz8 200\nsbits 77\nbback 2\nnew0 none\n",
    );
}

const RC_ARC_DROP_SRC: &str = r#"
use std::rc::Rc;
use std::sync::Arc;
static mut CHECK: u32 = 0;
struct Cat;
impl Drop for Cat {
    fn drop(&mut self) {
        unsafe { CHECK += 1; }
    }
}
fn main() {
    {
        let _r = Rc::new(Cat);
        let _a = Arc::new(Cat);
    }
    unsafe { assert_eq!(CHECK, 2); }
    println!("rc-arc ok");
}
"#;

#[test]
fn run_mode_rc_arc_drop_runs_pointee_destructor() {
    let dir = temp_dir("rc-arc-drop");
    let src = fixture(&dir, "rc_arc_drop.rs", RC_ARC_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dropping an Rc / Arc must run the pointee destructor; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "rc-arc ok\n");
}

const BOX_DYN_DROP_SRC: &str = r#"
static mut CHECK: u32 = 0;
trait Dummy {
    fn probe(&self) -> u32;
}
struct Cat;
impl Dummy for Cat {
    fn probe(&self) -> u32 {
        7
    }
}
impl Drop for Cat {
    fn drop(&mut self) {
        unsafe { CHECK += 1; }
    }
}
struct Holder {
    d: Box<dyn Dummy>,
}
fn main() {
    {
        let b: Box<dyn Dummy> = Box::new(Cat);
        assert_eq!(b.probe(), 7);
        let _h = Holder { d: Box::new(Cat) };
    }
    unsafe { assert_eq!(CHECK, 2); }
    println!("box-dyn ok");
}
"#;

#[test]
fn run_mode_box_dyn_drop_runs_concrete_destructor() {
    let dir = temp_dir("box-dyn-drop");
    let src = fixture(&dir, "box_dyn_drop.rs", BOX_DYN_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dropping a Box<dyn Trait> (directly or as a field) must run the concrete destructor; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "box-dyn ok\n");
}

const RC_ARC_DYN_SRC: &str = r#"
use std::rc::Rc;
use std::sync::Arc;
trait T {
    fn val(&self) -> i32;
    fn d(&self) -> i32 { 7 }
}
struct S(i32);
impl T for S {
    fn val(&self) -> i32 { self.0 }
}
fn main() {
    let r: Rc<dyn T> = Rc::new(S(42));
    let r2 = r.clone();
    let a: Arc<dyn T> = Arc::new(S(43));
    println!("{} {} {} {}", r.val(), r2.val(), r.d(), a.val());
}
"#;

#[test]
fn run_mode_rc_arc_dyn_trait_method_calls() {
    // Method calls through an `Rc<dyn Trait>` / `Arc<dyn Trait>`
    // receiver: the Deref builtin must yield the dyn fat pointer
    // itself (as it does for Box), and dispatch must reach both
    // the impl method (reading the pointee's state) and the
    // trait's default method.
    let dir = temp_dir("rc-arc-dyn");
    let src = fixture(&dir, "rc_arc_dyn.rs", RC_ARC_DYN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "calling trait methods through Rc<dyn Trait> / Arc<dyn Trait> must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42 42 7 43\n");
}

const DYN_DEFAULT_DISPATCH_SRC: &str = r#"
use std::rc::Rc;
trait T {
    fn val(&self) -> i32;
    fn extra(&self) -> i32 { self.val() + 1 }
    fn d(&self) -> i32 { 7 }
}
struct A;
impl T for A { fn val(&self) -> i32 { 10 } }
struct B;
impl T for B {
    fn val(&self) -> i32 { 20 }
    fn d(&self) -> i32 { 8 }
}
fn main() {
    let a: Rc<dyn T> = Rc::new(A);
    let b: Rc<dyn T> = Rc::new(B);
    println!("{} {} {} {}", a.extra(), a.d(), b.extra(), b.d());
    let r: &dyn T = &A;
    println!("{} {}", r.extra(), r.d());
}
"#;

#[test]
fn run_mode_dyn_default_method_dispatch() {
    // A trait method the impl does not override has only the
    // trait's default body: dispatch through a dyn receiver must
    // reach it (monomorphized at the concrete type), its inner
    // self calls must hit the concrete overrides, and an
    // overriding impl method must win over the default.
    let dir = temp_dir("dyn-default-dispatch");
    let src = fixture(&dir, "dyn_default.rs", DYN_DEFAULT_DISPATCH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dispatching to trait default methods through dyn receivers must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "11 7 21 8\n11 7\n");
}

const DYN_EQ_REF_SRC: &str = r#"
trait Foo: PartialEq<i32> + std::fmt::Debug {
    fn a(&self) -> i32 { 100 }
}
impl Foo for i32 {}
fn main() {
    let x: i32 = 1;
    assert_eq!(x.a(), 100);
    let foo: &dyn Foo = &x;
    assert_eq!(*foo, 1);
    assert!(*foo != 2);
    println!("dyn-eq-ref ok");
}
"#;

#[test]
fn run_mode_dyn_ref_partial_eq_compares_pointee() {
    let dir = temp_dir("dyn-eq-ref");
    let src = fixture(&dir, "dyn_eq_ref.rs", DYN_EQ_REF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "comparing through a &dyn trait object with a PartialEq supertrait must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-eq-ref ok\n");
}

const DYN_EQ_BOX_SRC: &str = r#"
trait Foo: PartialEq<i32> + std::fmt::Debug {
    fn a(&self) -> i32 { 100 }
}
impl Foo for i32 {}
fn main() {
    let obj: Box<dyn Foo> = Box::new(1);
    assert_eq!(*obj, 1);
    assert!(*obj != 2);
    println!("dyn-eq-box ok");
}
"#;

#[test]
fn run_mode_box_dyn_partial_eq_compares_pointee() {
    let dir = temp_dir("dyn-eq-box");
    let src = fixture(&dir, "dyn_eq_box.rs", DYN_EQ_BOX_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "comparing through a Box<dyn trait> with a PartialEq supertrait must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-eq-box ok\n");
}

const DYN_EQ_ADT_SRC: &str = r#"
trait Foo: PartialEq<Bar> + std::fmt::Debug {}
#[derive(PartialEq, Debug)]
struct Bar(i32);
impl Foo for Bar {}
fn main() {
    let obj: &dyn Foo = &Bar(7);
    assert_eq!(*obj, Bar(7));
    assert!(*obj != Bar(8));
    println!("dyn-eq-adt ok");
}
"#;

#[test]
fn run_mode_dyn_ref_partial_eq_against_adt() {
    let dir = temp_dir("dyn-eq-adt");
    let src = fixture(&dir, "dyn_eq_adt.rs", DYN_EQ_ADT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "comparing a trait object against an ADT value must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-eq-adt ok\n");
}

const DYN_EQ_DYN_SRC: &str = r#"
trait T: std::fmt::Debug {
    fn val(&self) -> i32;
    fn type_id(&self) -> u32;
}
#[derive(Debug)]
struct A(i32);
#[derive(Debug)]
struct B(i32);
impl T for A {
    fn val(&self) -> i32 { self.0 }
    fn type_id(&self) -> u32 { 1 }
}
impl T for B {
    fn val(&self) -> i32 { self.0 }
    fn type_id(&self) -> u32 { 2 }
}
impl PartialEq for dyn T {
    fn eq(&self, other: &Self) -> bool {
        self.type_id() == other.type_id() && self.val() == other.val()
    }
}
fn main() {
    let a1: &dyn T = &A(1);
    let a2: &dyn T = &A(1);
    let a3: &dyn T = &A(2);
    let b1: &dyn T = &B(1);
    assert!(a1 == a2);
    assert!(a1 != a3);
    assert!(a1 != b1);
    println!("dyn-eq-dyn ok");
}
"#;

#[test]
fn run_mode_dyn_partial_eq_against_dyn() {
    let dir = temp_dir("dyn-eq-dyn");
    let src = fixture(&dir, "dyn_eq_dyn.rs", DYN_EQ_DYN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "comparing two trait objects must run (unequal concrete types are unequal); stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-eq-dyn ok\n");
}

const DYN_EQ_UPCAST_SRC: &str = r#"
trait Super: PartialEq<i32> + std::fmt::Debug {}
trait Sub: Super {}
impl Super for i32 {}
impl Sub for i32 {}
fn main() {
    let sub: &dyn Sub = &1;
    let sup: &dyn Super = sub;
    assert_eq!(*sup, 1);
    assert!(*sup != 2);
    let boxed: Box<dyn Sub> = Box::new(5);
    let boxed_super: Box<dyn Super> = boxed;
    assert_eq!(*boxed_super, 5);
    assert!(*boxed_super != 6);
    println!("dyn-eq-upcast ok");
}
"#;

#[test]
fn run_mode_upcast_dyn_partial_eq_compares_pointee() {
    let dir = temp_dir("dyn-eq-upcast");
    let src = fixture(&dir, "dyn_eq_upcast.rs", DYN_EQ_UPCAST_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "comparing through an upcast trait object must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-eq-upcast ok\n");
}

const FROM_BOX_DYN_ERROR_DROP_SRC: &str = r#"
use std::error::Error;
use std::fmt;
#[derive(Debug)]
struct MyErr { code: i32, other: i32 }
impl fmt::Display for MyErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "err {}", self.code) }
}
impl Error for MyErr {}
impl Drop for MyErr {
    fn drop(&mut self) { println!("dropping MyErr {}", self.code); }
}
fn main() {
    let b: Box<dyn Error> = Box::from(MyErr { code: 7, other: 1 });
    println!("made the box");
    drop(b);
    println!("end of main");
}
"#;

#[test]
fn run_mode_from_box_dyn_error_payload_drop_order() {
    let dir = temp_dir("from-box-dyn-error");
    let src = fixture(&dir, "from_box_dyn_error.rs", FROM_BOX_DYN_ERROR_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a From conversion into Box<dyn Error> must produce a droppable trait object; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "made the box\ndropping MyErr 7\nend of main\n"
    );
}

const INTO_BOX_DYN_ERROR_DROP_SRC: &str = r#"
use std::error::Error;
use std::fmt;
#[derive(Debug)]
struct MyErr { code: i32 }
impl fmt::Display for MyErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "err {}", self.code) }
}
impl Error for MyErr {}
impl Drop for MyErr {
    fn drop(&mut self) { println!("dropping MyErr {}", self.code); }
}
fn main() {
    let b: Box<dyn Error> = MyErr { code: 9 }.into();
    println!("made the box");
    drop(b);
    println!("end of main");
}
"#;

#[test]
fn run_mode_into_box_dyn_error_payload_drop_order() {
    let dir = temp_dir("into-box-dyn-error");
    let src = fixture(&dir, "into_box_dyn_error.rs", INTO_BOX_DYN_ERROR_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an Into conversion into Box<dyn Error> must produce a droppable trait object; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "made the box\ndropping MyErr 9\nend of main\n"
    );
}

const FROM_BOX_DYN_ERROR_ZST_SRC: &str = r#"
use std::error::Error;
use std::fmt;
#[derive(Debug)]
struct Unit;
impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "unit") }
}
impl Error for Unit {}
impl Drop for Unit {
    fn drop(&mut self) { println!("dropping Unit"); }
}
fn main() {
    let b: Box<dyn Error> = Box::from(Unit);
    println!("made the box");
    println!("end of main");
}
"#;

#[test]
fn run_mode_from_box_dyn_error_zst_payload_drop() {
    let dir = temp_dir("from-box-dyn-error-zst");
    let src = fixture(&dir, "from_box_dyn_error_zst.rs", FROM_BOX_DYN_ERROR_ZST_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a From conversion of a zero-sized payload into Box<dyn Error> must stay droppable; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "made the box\nend of main\ndropping Unit\n"
    );
}

const BOX_DYN_ERROR_FROM_RAW_DROP_SRC: &str = r#"
use std::error::Error;
use std::fmt;
#[derive(Debug)]
struct MyErr { code: i32 }
impl fmt::Display for MyErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "err {}", self.code) }
}
impl Error for MyErr {}
impl Drop for MyErr {
    fn drop(&mut self) { println!("dropping MyErr {}", self.code); }
}
fn main() {
    let b: Box<dyn Error> = Box::new(MyErr { code: 13 });
    let raw = Box::into_raw(b);
    let b2 = unsafe { Box::from_raw(raw) };
    println!("rebuilt the box");
    drop(b2);
    println!("end of main");
}
"#;

#[test]
fn run_mode_box_dyn_error_from_raw_drop() {
    // A trait-object box rebuilt by `Box::from_raw` is an
    // aggregate chain around the fat pointer, not the
    // builtin-built surface: its drop plan must descend field
    // 0 of each wrapper layer to the `DynRef` at the chain's
    // bottom and run the payload glue from there.
    let dir = temp_dir("box-dyn-error-from-raw");
    let src = fixture(&dir, "box_dyn_error_from_raw.rs", BOX_DYN_ERROR_FROM_RAW_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dropping a Box<dyn Error> rebuilt by from_raw must run the payload destructor; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "rebuilt the box\ndropping MyErr 13\nend of main\n"
    );
}

const FROM_BOX_DYN_FN_CAPTURE_SRC: &str = r#"
struct S { v: i32, w: i32 }
impl Drop for S {
    fn drop(&mut self) { println!("dropping S {} {}", self.v, self.w); }
}
fn main() {
    let s = S { v: 42, w: 1 };
    let f: Box<dyn Fn(i32) -> i32> = Box::from(move |x: i32| x + s.v + s.w);
    println!("f(1) = {}", f(1));
    println!("end of main");
}
"#;

#[test]
fn run_mode_from_box_dyn_fn_reads_captures_and_drops() {
    let dir = temp_dir("from-box-dyn-fn");
    let src = fixture(&dir, "from_box_dyn_fn.rs", FROM_BOX_DYN_FN_CAPTURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "calling a Box<dyn Fn> built by From must read the captures through the dyn reference; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "f(1) = 44\nend of main\ndropping S 42 1\n"
    );
}

const UNION_DROP_SRC: &str = r#"
use std::mem::ManuallyDrop;
struct S;
static mut CHECK: u8 = 0;
impl Drop for S {
    fn drop(&mut self) {
        unsafe { CHECK += 10; }
    }
}
union U {
    a: ManuallyDrop<S>,
}
impl Drop for U {
    fn drop(&mut self) {
        unsafe { CHECK += 1; }
    }
}
fn main() {
    unsafe {
        let mut u = U { a: ManuallyDrop::new(S) };
        u = U { a: ManuallyDrop::new(S) };
        assert_eq!(CHECK, 1);
        *u.a = S;
        assert_eq!(CHECK, 11);
    }
    println!("union ok");
}
"#;

#[test]
fn run_mode_union_drop_and_field_assignment() {
    let dir = temp_dir("union-drop");
    let src = fixture(&dir, "union_drop.rs", UNION_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a union with a Drop impl must drop on overwrite, and assigning through its field must drop the old field value; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "union ok\n");
}

const MEM_DROP_ZST_UNION_SRC: &str = r#"
use std::mem::ManuallyDrop;
struct S;
static mut CHECK: u8 = 0;
impl Drop for S {
    fn drop(&mut self) {
        unsafe { CHECK += 10; }
    }
}
union U {
    a: ManuallyDrop<S>,
}
impl Drop for U {
    fn drop(&mut self) {
        unsafe { CHECK += 1; }
    }
}
fn main() {
    let u = U { a: ManuallyDrop::new(S) };
    drop(u);
    unsafe { assert_eq!(CHECK, 1); }
    println!("mem-drop ok");
}
"#;

#[test]
fn run_mode_mem_drop_of_zst_union_runs_destructor() {
    let dir = temp_dir("mem-drop-zst-union");
    let src = fixture(&dir, "mem_drop_zst_union.rs", MEM_DROP_ZST_UNION_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an explicit mem::drop of a zero-sized union value must run its destructor; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "mem-drop ok\n");
}

const UNION_REINTERPRET_SRC: &str = r#"
union W {
    a: u32,
    b: f32,
}
union V {
    a: u32,
    b: [u8; 4],
    c: u16,
}
union T {
    a: (u8, u8),
    b: u16,
}
union U<A, B>
where
    A: Copy,
    B: Copy,
{
    a: A,
    b: B,
}
unsafe fn union_transmute<A, B>(a: A) -> B
where
    A: Copy,
    B: Copy,
{
    U { a }.b
}
fn main() {
    unsafe {
        let w = W { a: 0x3F80_0000 };
        assert_eq!(w.b, 1.0);
        let mut w2 = W { b: 2.5 };
        assert_eq!(w2.a, 0x4020_0000);
        w2.a = 0x3F80_0000;
        assert_eq!(w2.b, 1.0);
        let v = V { a: 0x0403_0201 };
        assert_eq!(v.b, [1, 2, 3, 4]);
        assert_eq!(v.c, 0x0201);
        let v2 = V { b: [9, 8, 7, 6] };
        assert_eq!(v2.a, 0x0607_0809);
        let mut m = V { a: 0 };
        m.c = 0xBEEF;
        assert_eq!(m.c, 0xBEEF);
        assert_eq!(m.b, [0xEF, 0xBE, 0, 0]);
        let t = T { a: (1, 1) };
        assert_eq!(t.b, (1 << 8) + 1);
        let g: u16 = union_transmute((1u8, 1u8));
        assert_eq!(g, (1 << 8) + 1);
        let f: f32 = union_transmute(0x3F80_0000u32);
        assert_eq!(f, 1.0);
        println!("union reinterpret ok");
    }
}
"#;

#[test]
fn run_mode_union_field_reads_reinterpret_bits() {
    // Reading a union through a different field than the one
    // written reinterprets the held field's bits (little-endian:
    // narrower fields read the low bytes), matching native runs
    // of this same program.
    let dir = temp_dir("union-reinterpret");
    let src = fixture(&dir, "union_reinterpret.rs", UNION_REINTERPRET_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "union field reads must reinterpret the held field's bits; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "union reinterpret ok\n"
    );
}

const UNION_WRAPPER_FIELDS_SRC: &str = r#"
use std::mem::ManuallyDrop;
union C {
    a: u32,
    b: char,
}
union B {
    a: u8,
    b: bool,
}
union M {
    a: ManuallyDrop<u32>,
    b: f32,
}
fn main() {
    unsafe {
        let c = C { a: 0x41 };
        assert_eq!(c.b, 'A');
        let b = B { a: 1 };
        assert_eq!(b.b, true);
        let m = M { b: 1.5 };
        let md = m.a;
        let m3 = M { a: md };
        assert_eq!(m3.b, 1.5);
        let m2 = M { a: ManuallyDrop::new(0x3F80_0000) };
        assert_eq!(m2.b, 1.0);
        println!("union wrapper fields ok");
    }
}
"#;

#[test]
fn run_mode_union_field_reads_through_wrapper_fields() {
    // `char` / `bool` fields and one-payload wrapper fields
    // (ManuallyDrop) reinterpret through the same shapes.
    let dir = temp_dir("union-wrapper-fields");
    let src = fixture(&dir, "union_wrapper_fields.rs", UNION_WRAPPER_FIELDS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "union reads through char/bool/ManuallyDrop fields must reinterpret; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "union wrapper fields ok\n"
    );
}

const UNION_MULTI_FIELD_STRUCT_SRC: &str = r#"
#[derive(Copy, Clone)]
struct P(u32, u32);
union X {
    a: P,
    b: u64,
}
fn main() {
    unsafe {
        let x = X { b: 0x0000_0005_0000_002a };
        let p = x.a;
        println!("{} {}", p.0, p.1);
    }
}
"#;

#[test]
fn run_mode_union_field_read_multi_field_struct() {
    // A multi-field struct field read from another field's
    // bits: the struct's fields sit at their recorded layout
    // offsets, so the read reinterprets exactly.
    let dir = temp_dir("union-multi-field-struct");
    let src = fixture(&dir, "union_multi_field_struct.rs", UNION_MULTI_FIELD_STRUCT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a union read of a multi-field struct must reinterpret; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42 5\n");
}

const UNION_NO_SHAPE_SRC: &str = r#"
#[derive(Copy, Clone)]
struct Big([u64; 3]);
union X {
    a: Big,
    b: u64,
}
fn main() {
    unsafe {
        let x = X { b: 42 };
        let _p = x.a;
        println!("should not reach");
    }
}
"#;

#[test]
fn run_mode_union_field_read_without_scalar_shape_is_explicit() {
    // A field whose type has no scalar shape (a struct over
    // the 128-bit shape limit) cannot be reinterpreted from
    // another field's bits: the read is a named unsupported
    // error, not a fabricated value.
    let dir = temp_dir("union-no-shape");
    let src = fixture(&dir, "union_no_shape.rs", UNION_NO_SHAPE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a union read with no scalar shape must be a named unsupported; stderr={stderr}"
    );
    assert!(
        stderr.contains("union field read"),
        "stderr must name the union field read; stderr={stderr}"
    );
}

const UNION_NESTED_WRITE_SRC: &str = r#"
union U {
    a: (u16, u16),
    b: u32,
}
fn main() {
    unsafe {
        let mut u = U { b: 0xDEAD_DEAD };
        u.a.0 = 0xBEEF;
        println!("should not reach");
    }
}
"#;

#[test]
fn run_mode_union_write_inside_unheld_field_is_explicit() {
    // Writing inside a field the value does not currently hold
    // would have to preserve the held field's remaining bytes;
    // the model holds one field's value, so the write stays an
    // uninitialized error rather than fabricating the field.
    let dir = temp_dir("union-nested-write");
    let src = fixture(&dir, "union_nested_write.rs", UNION_NESTED_WRITE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a write inside an unheld union field must fail explicitly; stderr={stderr}"
    );
    assert!(
        stderr.contains("uninitialized"),
        "stderr must name the uninitialized field; stderr={stderr}"
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
const TEST_CRATE_WITH_MAIN_SRC: &str = r#"
fn main() {
    println!("USER MAIN RAN");
}

#[test]
fn sample_test() {
    assert_eq!(1 + 1, 2);
}
"#;

#[test]
fn run_mode_test_crate_enters_harness_main_not_user_main() {
    // Lane ET: a `--test` crate has two bodies named
    // `<crate>::main` -- the user's and the generated harness
    // entry. Run mode used to hand the interpreter the bare
    // name, which names no stored body (bodies that share a
    // name are keyed by their `DefId`), and the run failed with
    // `unknown function <crate>::main`; picking the entry by
    // name alone would also have entered the user's `main`,
    // which natively is an ordinary function in a test crate.
    // The driver now enters the entry point rustc designated.
    // The harness's call into the libtest runner is served by
    // the `test_main_env_args` builtin, so the crate's tests
    // run; the user's `main` must not have run.
    let dir = temp_dir("test-crate-main");
    let src = fixture(&dir, "testcrate.rs", TEST_CRATE_WITH_MAIN_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("--test")
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the harness entry must be reached and its tests run; stderr={stderr}"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("test sample_test ... ok"),
        "the libtest runner must report the test; stdout={stdout}"
    );
    assert!(
        !stderr.contains("unknown function"),
        "entry resolution must not lose the entry body; stderr={stderr}"
    );
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("USER MAIN RAN"),
        "the user's main is not the entry of a test crate and must not run"
    );
}

const TEST_CRATE_NO_TESTS_SRC: &str = r#"
fn main() {
    println!("USER MAIN RAN");
}
"#;

#[test]
fn run_mode_test_crate_without_tests_still_enters_harness() {
    // Lane ET: under `--test` with no `#[test]` items the
    // harness main is still the crate's entry (natively it runs
    // an empty test list and calls the libtest runner directly);
    // the user's `main` -- the only `main` in the source -- is
    // not entered.
    let dir = temp_dir("test-crate-no-tests");
    let src = fixture(&dir, "notests.rs", TEST_CRATE_NO_TESTS_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("--test")
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the harness entry must be reached and run its empty test list; stderr={stderr}"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("running 0 tests")
            && stdout.contains("test result: ok. 0 passed; 0 failed"),
        "the libtest runner must report the empty run; stdout={stdout}"
    );
    assert!(
        !stderr.contains("unknown function"),
        "entry resolution must not lose the entry body; stderr={stderr}"
    );
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("USER MAIN RAN"),
        "the user's main is not the entry of a test crate and must not run"
    );
}

// ----- box allocation identity (Lane EP) ------------------------------------

// A built `Box`'s pointee lives in the interpreter's allocation
// store and the box value is the allocation pointer, so a
// reference to the pointee names the allocation — never a place
// in the frame that happened to hold a copy of the box. The
// fixtures below are the rustc suite programs whose run used to
// fail with "reference into finished frame #N" for exactly
// that reason (directives stripped; the feature attributes are
// accepted because the driver is built and run with
// RUSTC_BOOTSTRAP=1, like compiletest).

const FUNC_ARG_REF_PATTERN_SRC: &str = r#"// Test argument patterns where we create refs to the inside of
// boxes. Make sure that we don't free the box as we match the
// pattern.

#![feature(deref_patterns)]

fn getaddr(deref!(ref x): Box<usize>) -> *const usize {
    let addr: *const usize = &*x;
    addr
}

fn checkval(deref!(ref x): Box<usize>) -> usize {
    *x
}

pub fn main() {
    let obj: Box<_> = Box::new(1);
    let objptr: *const usize = &*obj;
    let xptr = getaddr(obj);
    assert_eq!(objptr, xptr);

    let obj = Box::new(22);
    assert_eq!(checkval(obj), 22);
}"#;

#[test]
fn run_mode_box_pointee_ref_pattern_arg() {
    // a by-value `Box` argument matched with a ref pattern: the reference the pattern forms names the box's allocation, so the pointer taken in the callee equals the one taken before the move, and the box's drop at the callee's end does not invalidate it
    let dir = temp_dir("func-arg-ref-pattern");
    let src = fixture(&dir, "func_arg_ref_pattern.rs", FUNC_ARG_REF_PATTERN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}

const UNSAFE_PTR_SRC: &str = r#"// Test that we can use raw ptrs when using `capture_disjoint_fields`.

#![allow(dead_code)]

#[derive(Debug)]
struct S {
    s: String,
    t: String,
}

struct T(*const S);

fn unsafe_imm() {
    let s = "".into();
    let t = "".into();
    let my_speed: Box<S> = Box::new(S { s, t });

    let p : *const S = Box::into_raw(my_speed);
    let t = T(p);

    let c = || unsafe {
        println!("{:?}", (*t.0).s);
    };

    c();
}

fn unsafe_mut() {
    let s = "".into();
    let t = "".into();
    let mut my_speed: Box<S> = Box::new(S { s, t });
    let p : *mut S = &mut *my_speed;

    let c = || {
        let x = unsafe { &mut (*p).s };
        *x = "s".into();
    };
    c();
}

fn main() {
    unsafe_mut();
    unsafe_imm();
}"#;

#[test]
fn run_mode_box_into_raw_pointee_outlives_frame() {
    // `Box::into_raw` leaks the allocation: closures created afterwards dereference the raw pointer into storage no frame owns any more, and must still read (and write) the pointee
    let dir = temp_dir("unsafe-ptr");
    let src = fixture(&dir, "unsafe_ptr.rs", UNSAFE_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}

const DEREF_MUT_CLOSURE_DROP_ORDER_SRC: &str = r#"//! Regression test for https://github.com/rust-lang/rust/issues/16774

#![feature(deref_patterns)]

use std::ops::{Deref, DerefMut};

struct X(Box<isize>);

static mut DESTRUCTOR_RAN: bool = false;

impl Drop for X {
    fn drop(&mut self) {
        unsafe {
            assert!(!DESTRUCTOR_RAN);
            DESTRUCTOR_RAN = true;
        }
    }
}

impl Deref for X {
    type Target = isize;

    fn deref(&self) -> &isize {
        let &X(ref x) = self;
        x
    }
}

impl DerefMut for X {
    fn deref_mut(&mut self) -> &mut isize {
        let &mut X(ref mut x) = self;
        x
    }
}

fn main() {
    {
        let mut test = X(Box::new(5));
        {
            let mut change = || { *test = 10 };
            change();
        }
        assert_eq!(*test, 10);
    }
    assert!(unsafe { DESTRUCTOR_RAN });
}"#;

#[test]
fn run_mode_box_deref_mut_write_through_pattern_copy() {
    // a user `DerefMut` impl whose pattern match copies the `X` value (and its `Box` field) into its own frame: the `&mut` it returns names the box's allocation, so the closure's write through it is observed by the owner's later read, and the drop order is preserved
    let dir = temp_dir("deref-mut-closure-drop-order");
    let src = fixture(&dir, "deref_mut_closure_drop_order.rs", DEREF_MUT_CLOSURE_DROP_ORDER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}

const DST_DEREF_SRC: &str = r#"// Test that a custom deref with a fat pointer return type does not ICE


use std::ops::Deref;

pub struct Arr {
    ptr: Box<[usize]>
}

impl Deref for Arr {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        &*self.ptr
    }
}

pub fn foo(arr: &Arr) {
    assert_eq!(arr.len(), 3);
    let x: &[usize] = &**arr;
    assert_eq!(x[0], 1);
    assert_eq!(x[1], 2);
    assert_eq!(x[2], 3);
}

fn main() {
    let a = Arr { ptr: Box::new([1, 2, 3]) };
    foo(&a);
}"#;

#[test]
fn run_mode_box_dst_deref() {
    // a custom `Deref` returning `&*self.ptr` for a `Box<[usize]>`: the slice reference is formed from a copy of the box inside the `Deref` shim and must name the allocation, not the shim's frame
    let dir = temp_dir("dst-deref");
    let src = fixture(&dir, "dst_deref.rs", DST_DEREF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}

const DST_DEREF_MUT_SRC: &str = r#"// Test that a custom deref with a fat pointer return type does not ICE


use std::ops::{Deref, DerefMut};

pub struct Arr {
    ptr: Box<[usize]>
}

impl Deref for Arr {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        panic!();
    }
}

impl DerefMut for Arr {
    fn deref_mut(&mut self) -> &mut [usize] {
        &mut *self.ptr
    }
}

pub fn foo(arr: &mut Arr) {
    let x: &mut [usize] = &mut **arr;
    assert_eq!(x[0], 1);
    assert_eq!(x[1], 2);
    assert_eq!(x[2], 3);
}

fn main() {
    let mut a = Arr { ptr: Box::new([1, 2, 3]) };
    foo(&mut a);
}"#;

#[test]
fn run_mode_box_dst_deref_mut() {
    // the `DerefMut` counterpart: `&mut *self.ptr` for a `Box<[usize]>` names the allocation, so element reads through it in the caller observe the array
    let dir = temp_dir("dst-deref-mut");
    let src = fixture(&dir, "dst_deref_mut.rs", DST_DEREF_MUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}

const REGIONS_DEPENDENT_ADDR_OF_SRC: &str = r#"// Test lifetimes are linked properly when we create dependent region pointers.
// Issue #3148.
#![feature(deref_patterns)]

struct A {
    value: B
}

struct B {
    v1: isize,
    v2: [isize; 3],
    v3: Vec<isize> ,
    v4: C,
    v5: Box<C>,
    v6: Option<C>
}

#[derive(Copy, Clone)]
struct C {
    f: isize
}

fn get_v1(a: &A) -> &isize {
    // Region inferencer must deduce that &v < L2 < L1
    let foo = &a.value; // L1
    &foo.v1             // L2
}

fn get_v2(a: &A, i: usize) -> &isize {
    let foo = &a.value;
    &foo.v2[i]
}

fn get_v3(a: &A, i: usize) -> &isize {
    let foo = &a.value;
    &foo.v3[i]
}

fn get_v4(a: &A, _i: usize) -> &isize {
    let foo = &a.value;
    &foo.v4.f
}

fn get_v5(a: &A, _i: usize) -> &isize {
    let foo = &a.value;
    &foo.v5.f
}

fn get_v6_a(a: &A, _i: usize) -> &isize {
    match a.value.v6 {
        Some(ref v) => &v.f,
        None => panic!()
    }
}

fn get_v6_b(a: &A, _i: usize) -> &isize {
    match *a {
        A { value: B { v6: Some(ref v), .. } } => &v.f,
        _ => panic!()
    }
}

fn get_v6_c(a: &A, _i: usize) -> &isize {
    match a {
        &A { value: B { v6: Some(ref v), .. } } => &v.f,
        _ => panic!()
    }
}

fn get_v5_ref(a: &A, _i: usize) -> &isize {
    match &a.value {
        &B {v5: C {f: ref v}, ..} => v
    }
}

pub fn main() {
    let a = A {value: B {v1: 22,
                         v2: [23, 24, 25],
                         v3: vec![26, 27, 28],
                         v4: C { f: 29 },
                         v5: Box::new(C { f: 30 }),
                         v6: Some(C { f: 31 })}};

    let p = get_v1(&a);
    assert_eq!(*p, a.value.v1);

    let p = get_v2(&a, 1);
    assert_eq!(*p, a.value.v2[1]);

    let p = get_v3(&a, 1);
    assert_eq!(*p, a.value.v3[1]);

    let p = get_v4(&a, 1);
    assert_eq!(*p, a.value.v4.f);

    let p = get_v5(&a, 1);
    assert_eq!(*p, a.value.v5.f);

    let p = get_v6_a(&a, 1);
    assert_eq!(*p, a.value.v6.unwrap().f);

    let p = get_v6_b(&a, 1);
    assert_eq!(*p, a.value.v6.unwrap().f);

    let p = get_v6_c(&a, 1);
    assert_eq!(*p, a.value.v6.unwrap().f);

    let p = get_v5_ref(&a, 1);
    assert_eq!(*p, a.value.v5.f);
}"#;

#[test]
fn run_mode_box_dependent_addr_of() {
    // `&foo.v5.f` where `v5: Box<C>` is copied out of the borrowed struct into the callee: the returned reference names the box's allocation and stays valid after the callee returns
    let dir = temp_dir("regions-dependent-addr-of");
    let src = fixture(&dir, "regions_dependent_addr_of.rs", REGIONS_DEPENDENT_ADDR_OF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}

const ARBITRARY_SELF_TYPES_SRC: &str = r#"#![feature(arbitrary_self_types)]
#![allow(unused_allocation)]

struct SmartPtr<T: ?Sized>(T);

impl<T: ?Sized> std::ops::Receiver for SmartPtr<T> {
    type Target = T;
}

trait Trait {
    fn trait_method<'a>(self: &'a Box<SmartPtr<Self>>) -> &'a [i32];
}

impl Trait for Vec<i32> {
    fn trait_method<'a>(self: &'a Box<SmartPtr<Self>>) -> &'a [i32] {
        &(**self).0
    }
}

fn main() {
    let v = vec![1, 2, 3];

    assert_eq!(&[1, 2, 3], Box::new(SmartPtr(v)).trait_method());
}"#;

#[test]
fn run_mode_box_arbitrary_self_types_receiver() {
    // `&(**self).0` with `self: &Box<SmartPtr<Self>>`: the box is copied into the method's frame and the returned slice reference names the pointee chain inside the allocation
    let dir = temp_dir("arbitrary-self-types");
    let src = fixture(&dir, "arbitrary_self_types.rs", ARBITRARY_SELF_TYPES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}

const BOX_PTR_IDENTITY_SRC: &str = r#"
fn getaddr(b: Box<usize>) -> *const usize {
    &*b as *const usize
}
fn main() {
    let b = Box::new(3usize);
    let p1 = &*b as *const usize;
    let p2 = getaddr(b);
    assert_eq!(p1, p2);
    println!("ok");
}
"#;

#[test]
fn run_mode_box_pointer_identity_across_move() {
    // Pointer identity of a moved box: the address taken from
    // the box before the move and the address extracted from
    // the moved box inside the callee are the same address,
    // because both name the one allocation.
    let dir = temp_dir("box-ptr-identity");
    let src = fixture(&dir, "box_ptr_identity.rs", BOX_PTR_IDENTITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "pointer identity across a move must hold; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const BOX_INTO_RAW_WRITE_SRC: &str = r#"
fn main() {
    let p = Box::into_raw(Box::new(7));
    println!("{}", unsafe { *p });
    unsafe { *p = 9; }
    println!("{}", unsafe { *p });
}
"#;

#[test]
fn run_mode_box_into_raw_write_and_read_back() {
    // A leaked allocation stays readable and writable through
    // the raw pointer after `into_raw`'s frame is gone.
    let dir = temp_dir("box-into-raw-write");
    let src = fixture(&dir, "box_into_raw_write.rs", BOX_INTO_RAW_WRITE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the leaked pointee must stay accessible; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "7\n9\n");
}

const BOX_DYN_FIELD_PROJ_SRC: &str = r#"
trait G { fn get(&self) -> i32; }
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
struct S { m: M }
impl G for S { fn get(&self) -> i32 { self.m.0 } }
fn main() {
    let b: Box<dyn G> = Box::new(S { m: M(9) });
    println!("get {}", b.get());
    drop(b);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_field_projection_reads_payload_field() {
    // The method body reads `self.m.0`: a multi-step field
    // projection through the box deref, where the pointee lives
    // behind the built box's allocation pointer. The receiver
    // must denote the allocation, not a value holding the
    // pointer, or the projection fails on a reference where an
    // ADT value belongs. The payload's destructor runs at the
    // `drop` point, after the read.
    let dir = temp_dir("box-dyn-field-proj");
    let src = fixture(&dir, "box_dyn_field_proj.rs", BOX_DYN_FIELD_PROJ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a Box<dyn Trait> method reading a nested payload field must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "get 9\ndtor 9\nafter drop\n"
    );
}

const BOX_SIZED_FIELD_RW_SRC: &str = r#"
struct M(i32);
struct S { m: M }
fn main() {
    let mut b = Box::new(S { m: M(9) });
    println!("read {}", b.m.0);
    b.m.0 += 5;
    println!("after write {}", b.m.0);
    (*b).m.0 = 100;
    println!("after deref write {}", b.m.0);
}
"#;

#[test]
fn run_mode_box_sized_field_read_write_through_deref() {
    // Field reads and writes through a sized Box's deref reach
    // the pointee in the allocation store: both the auto-deref
    // assignment and the explicit `(*b)` assignment land.
    let dir = temp_dir("box-sized-field-rw");
    let src = fixture(&dir, "box_sized_field_rw.rs", BOX_SIZED_FIELD_RW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "field reads/writes through a sized Box deref must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "read 9\nafter write 14\nafter deref write 100\n"
    );
}

const BOX_NEW_UNINIT_ASSUME_INIT_SRC: &str = r#"
use std::mem::MaybeUninit;
struct P { x: u32, y: u32 }
fn main() {
    let mut b: Box<MaybeUninit<(u32, u32)>> = Box::new_uninit();
    unsafe { b.as_mut_ptr().write((11, 22)); }
    let mut b = unsafe { b.assume_init() };
    println!("{} {}", b.0, b.1);
    b.0 = 99;
    println!("{} {}", b.0, b.1);
    let mut a: Box<MaybeUninit<[u32; 3]>> = Box::new_uninit();
    unsafe { a.as_mut_ptr().write([7, 8, 9]); }
    let a = unsafe { a.assume_init() };
    println!("{} {} {}", a[0], a[1], a[2]);
    let mut p: Box<MaybeUninit<P>> = Box::new_uninit();
    unsafe { p.as_mut_ptr().write(P { x: 5, y: 6 }); }
    let p = unsafe { p.assume_init() };
    println!("{} {}", p.x, p.y);
}
"#;

#[test]
fn run_mode_box_new_uninit_assume_init_reads_payload() {
    // `Box::new_uninit` + `write` + `assume_init`: the payload
    // was written through the wrapper-entering `as_mut_ptr`
    // pointer, and `assume_init`'s transmute retargets the box
    // without the entry step, so the reads and writes through
    // the retargeted box must still address the payload --
    // tuple fields, array elements and struct fields alike.
    let dir = temp_dir("box-new-uninit-assume-init");
    let src = fixture(
        &dir,
        "box_new_uninit_assume_init.rs",
        BOX_NEW_UNINIT_ASSUME_INIT_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box::new_uninit + assume_init payload access must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "11 22\n99 22\n7 8 9\n5 6\n"
    );
}

const BOX_MAYBEUNINIT_ENUM_SRC: &str = r#"
#[derive(Clone, Copy, PartialEq, Debug)]
enum E { A(u32), B(u16) }
fn main() {
    let mut b = Box::new(std::mem::MaybeUninit::<E>::uninit());
    unsafe {
        b.as_mut_ptr().write(E::B(0x1234));
        assert_eq!(*b.assume_init(), E::B(0x1234));
    }
    let mut c = Box::new(std::mem::MaybeUninit::<E>::uninit());
    unsafe {
        c.as_mut_ptr().write(E::A(7));
        let c2 = c.assume_init();
        match *c2 {
            E::A(x) => assert_eq!(x, 7),
            E::B(_) => panic!("wrong variant"),
        }
        let p = Box::into_raw(c2);
        assert_eq!(*p, E::A(7));
        drop(Box::from_raw(p));
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_box_maybeuninit_enum_assume_init_reads_payload() {
    // `Box<MaybeUninit<E>>` where `E`'s first variant takes one
    // field: the wrapper's field count coincides with that
    // variant's declared count, so the entry judgment at the
    // retargeted dereference must rest on the stored field's
    // fit for the declared field type -- the payload (a complete
    // `E` value) cannot be a `u32`, so the stored enum is the
    // wrapper and the reads address the payload.
    let dir = temp_dir("box-maybeuninit-enum-assume-init");
    let src = fixture(&dir, "box_maybeuninit_enum.rs", BOX_MAYBEUNINIT_ENUM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<MaybeUninit<enum>> assume_init reads must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const BOX_MAYBEUNINIT_ENUM_CONTROLS_SRC: &str = r#"
#[derive(Clone, Copy, PartialEq, Debug)]
enum E { A(u32), B(u16) }
#[derive(Clone, Copy, PartialEq, Debug)]
enum G { A(E), B(u8) }
#[derive(Clone, Copy, PartialEq, Debug)]
enum H { Only(u32) }
fn main() {
    // A genuine `G::A(E::B(..))` value: the stored enum has the
    // wrapper's shape, but its field fits the declared field
    // type, so it stays the position's own value.
    let g = Box::new(G::A(E::B(0x1234)));
    assert_eq!(*g, G::A(E::B(0x1234)));
    // A one-field enum through `MaybeUninit` storage peels, and
    // the same value stored directly does not double-peel.
    let h = Box::new(H::Only(3));
    assert_eq!(*h, H::Only(3));
    let mut h2 = Box::new(std::mem::MaybeUninit::<H>::uninit());
    unsafe {
        h2.as_mut_ptr().write(H::Only(3));
        assert_eq!(*h2.assume_init(), H::Only(3));
    }
    // Reading the box as `MaybeUninit<E>` keeps the union value.
    let mut m = Box::new(std::mem::MaybeUninit::<E>::uninit());
    unsafe {
        m.as_mut_ptr().write(E::B(0x1234));
        let mu: std::mem::MaybeUninit<E> = *m;
        assert_eq!(mu.assume_init(), E::B(0x1234));
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_box_maybeuninit_enum_entry_controls() {
    // The controls around the `MaybeUninit` entry judgment for
    // enum payloads: a nested enum value that merely looks like
    // a wrapper is not entered, a one-field enum payload is
    // entered, and a `MaybeUninit`-typed read is never entered.
    let dir = temp_dir("box-maybeuninit-enum-controls");
    let src = fixture(
        &dir,
        "box_maybeuninit_enum_controls.rs",
        BOX_MAYBEUNINIT_ENUM_CONTROLS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "MaybeUninit entry control cases must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const BOX_DYN_TUPLE_PAYLOAD_SRC: &str = r#"
trait G { fn get(&self) -> i32; }
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
struct T(M, i32);
impl G for T { fn get(&self) -> i32 { self.0.0 + self.1 } }
fn main() {
    let b: Box<dyn G> = Box::new(T(M(30), 12));
    println!("get {}", b.get());
    drop(b);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_tuple_payload_field_projection() {
    // A tuple-struct payload behind a Box<dyn Trait>: the
    // method projects two fields of the tuple, one of them a
    // nested tuple field of a Drop type.
    let dir = temp_dir("box-dyn-tuple-payload");
    let src = fixture(&dir, "box_dyn_tuple_payload.rs", BOX_DYN_TUPLE_PAYLOAD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a Box<dyn Trait> method projecting tuple payload fields must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "get 42\ndtor 30\nafter drop\n"
    );
}

const BOX_DYN_DEEP_FIELD_SRC: &str = r#"
trait G { fn get(&self) -> i32; }
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
struct Inner { m: M }
struct S { inner: Inner }
impl G for S { fn get(&self) -> i32 { self.inner.m.0 } }
fn main() {
    let b: Box<dyn G> = Box::new(S { inner: Inner { m: M(7) } });
    println!("get {}", b.get());
    drop(b);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_deep_field_projection() {
    // Three projection steps past the box deref
    // (`self.inner.m.0`) resolve through the allocation.
    let dir = temp_dir("box-dyn-deep-field");
    let src = fixture(&dir, "box_dyn_deep_field.rs", BOX_DYN_DEEP_FIELD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a Box<dyn Trait> method projecting three fields deep must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "get 7\ndtor 7\nafter drop\n"
    );
}

const BOX_DYN_MUT_WRITE_SRC: &str = r#"
trait G { fn get(&self) -> i32; fn bump(&mut self); }
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
struct S { m: M }
impl G for S {
    fn get(&self) -> i32 { self.m.0 }
    fn bump(&mut self) { self.m.0 += 1; }
}
fn main() {
    let mut b: Box<dyn G> = Box::new(S { m: M(9) });
    b.bump();
    println!("get {}", b.get());
    drop(b);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_mut_method_writes_through_allocation() {
    // A `&mut self` method writes through the same receiver:
    // the write lands in the allocation store, so the later
    // read sees it and the destructor observes the written
    // value.
    let dir = temp_dir("box-dyn-mut-write");
    let src = fixture(&dir, "box_dyn_mut_write.rs", BOX_DYN_MUT_WRITE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a &mut dyn method writing a payload field must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "get 10\ndtor 10\nafter drop\n"
    );
}

const RC_GET_MUT_SRC: &str = r#"
use std::rc::Rc;
fn main() {
    let mut r = Rc::new(41);
    *Rc::get_mut(&mut r).unwrap() += 1;
    println!("{}", *r);
}
"#;

#[test]
fn run_mode_rc_get_mut_write_is_observed_by_the_owner() {
    // `Rc::get_mut` runs its converted std body against the
    // allocation the handle names: the write through the returned
    // `&mut` lands in the shared storage and the owner reads it
    // back.
    let dir = temp_dir("rc-get-mut");
    let src = fixture(&dir, "rc_get_mut.rs", RC_GET_MUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Rc::get_mut on a unique Rc must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const RC_GET_MUT_SHARED_SRC: &str = r#"
use std::rc::Rc;
fn main() {
    let mut r = Rc::new(41);
    println!("unique: {}", Rc::get_mut(&mut r).is_some());
    let r2 = r.clone();
    println!("shared: {}", Rc::get_mut(&mut r).is_some());
    drop(r2);
    println!("unique again: {}", Rc::get_mut(&mut r).is_some());
}
"#;

#[test]
fn run_mode_rc_get_mut_is_none_while_shared() {
    // The strong/weak counts live in the allocation, so the
    // converted `get_mut` body observes clones and drops exactly
    // as natively: `None` while a second handle shares the value.
    let dir = temp_dir("rc-get-mut-shared");
    let src = fixture(&dir, "rc_get_mut_shared.rs", RC_GET_MUT_SHARED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Rc::get_mut must track sharing through clone/drop; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "unique: true\nshared: false\nunique again: true\n"
    );
}

const ARC_GET_MUT_SHARED_SRC: &str = r#"
use std::sync::Arc;
fn main() {
    let mut a = Arc::new(41);
    println!("unique: {}", Arc::get_mut(&mut a).is_some());
    let a2 = a.clone();
    println!("shared: {}", Arc::get_mut(&mut a).is_some());
    drop(a2);
    println!("unique again: {}", Arc::get_mut(&mut a).is_some());
}
"#;

#[test]
fn run_mode_arc_get_mut_is_none_while_shared() {
    let dir = temp_dir("arc-get-mut-shared");
    let src = fixture(&dir, "arc_get_mut_shared.rs", ARC_GET_MUT_SHARED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Arc::get_mut must track sharing through clone/drop; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "unique: true\nshared: false\nunique again: true\n"
    );
}

const ARC_MAKE_MUT_SRC: &str = r#"
use std::sync::Arc;
fn main() {
    let mut a = Arc::new(41);
    *Arc::make_mut(&mut a) += 1;
    println!("{}", *a);
}
"#;

#[test]
fn run_mode_arc_make_mut_on_unique_arc_writes_in_place() {
    // The unique path of the converted `make_mut` body: the
    // layout intrinsics resolve, the uniqueness check reads the
    // counts from the allocation, and the returned `&mut` writes
    // into the same storage the owner reads.
    let dir = temp_dir("arc-make-mut");
    let src = fixture(&dir, "arc_make_mut.rs", ARC_MAKE_MUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Arc::make_mut on a unique Arc must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const RC_MAKE_MUT_SRC: &str = r#"
use std::rc::Rc;
fn main() {
    let mut r = Rc::new(41);
    *Rc::make_mut(&mut r) += 1;
    println!("{}", *r);
}
"#;

#[test]
fn run_mode_rc_make_mut_on_unique_rc_writes_in_place() {
    let dir = temp_dir("rc-make-mut");
    let src = fixture(&dir, "rc_make_mut.rs", RC_MAKE_MUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Rc::make_mut on a unique Rc must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const RC_WEAK_SRC: &str = r#"
use std::rc::Rc;
fn main() {
    let r = Rc::new(41);
    let w = Rc::downgrade(&r);
    println!("{}", *w.upgrade().unwrap());
    println!("{}", Rc::strong_count(&r));
    println!("{}", Rc::weak_count(&r));
    drop(r);
    println!("{}", w.upgrade().is_none());
}
"#;

#[test]
fn run_mode_rc_weak_upgrade_and_counts() {
    // Downgrade bumps the weak count in the allocation; upgrade
    // observes the strong count, so it succeeds while a strong
    // handle lives and returns `None` once the last one drops -
    // the storage outlives the pointee on the weak bundle.
    let dir = temp_dir("rc-weak");
    let src = fixture(&dir, "rc_weak.rs", RC_WEAK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Rc::downgrade / Weak::upgrade must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "41\n1\n1\ntrue\n");
}

const RC_DROP_ONCE_SRC: &str = r#"
use std::rc::Rc;
static mut N: usize = 0;
struct D;
impl Drop for D {
    fn drop(&mut self) {
        unsafe { N += 1; }
    }
}
fn main() {
    let r = Rc::new(D);
    let r2 = r.clone();
    drop(r);
    println!("after drop r: {}", unsafe { N });
    drop(r2);
    println!("after drop r2: {}", unsafe { N });
}
"#;

#[test]
fn run_mode_rc_pointee_glue_runs_once_at_the_last_handle() {
    // Dropping one of two handles only decrements the count in
    // the allocation; the pointee's destructor runs when the
    // last strong handle drops, not once per handle copy.
    let dir = temp_dir("rc-drop-once");
    let src = fixture(&dir, "rc_drop_once.rs", RC_DROP_ONCE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Rc drop glue must be count-driven; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "after drop r: 0\nafter drop r2: 1\n"
    );
}

const ARC_MAKE_MUT_COW_SRC: &str = r#"
use std::sync::Arc;
fn main() {
    let mut a = Arc::new(41);
    let a2 = a.clone();
    *Arc::make_mut(&mut a) += 1;
    println!("{} {}", *a, *a2);
}
"#;

#[test]
fn run_mode_arc_make_mut_shared_stops_at_the_allocator() {
    // The copy-on-write path of `make_mut` allocates the
    // replacement `ArcInner` through the allocator machinery
    // (`UniqueArcUninit`, `__rust_alloc`), which needs a
    // byte-addressable allocation model the interpreter does not
    // have yet. Until it does, the run must stop with an explicit
    // unsupported (exit 2) rather than answering wrongly.
    let dir = temp_dir("arc-make-mut-cow");
    let src = fixture(&dir, "arc_make_mut_cow.rs", ARC_MAKE_MUT_COW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "make_mut's shared path is unsupported until the allocator is modelled; stderr={stderr}"
    );
    assert!(
        stderr.contains("unsupported"),
        "the stop names itself as unsupported: {stderr}"
    );
}

// ----- subslice projections and pattern types (Lane FG) ---------------------

const SUBSLICE_SRC: &str = r#"
fn head_rest(s: &[i32]) -> i32 {
    match s {
        [first, rest @ ..] => first + rest.len() as i32,
        [] => -1,
    }
}
fn tail2(s: &[i32]) -> i32 {
    match s {
        [.., a, b] => a * 10 + b,
        _ => 0,
    }
}
fn mid_sum(s: &[i32]) -> i32 {
    match s {
        [_, m @ .., _] => {
            let mut acc = 0;
            let mut i = 0;
            while i < m.len() { acc += m[i]; i += 1; }
            acc
        }
        _ => 0,
    }
}
fn main() {
    let arr = [1, 2, 3, 4, 5];
    let s = &arr[1..3];
    println!("len {} sum {}", s.len(), s[0] + s[1]);
    let mut arr2 = [10, 20, 30, 40];
    arr2[1..3][0] = 99;
    println!("w {}", arr2[1]);
    let t = &arr[..2];
    println!("head {}", t.len());
    let u = &arr[3..];
    println!("tail {}", u[0] + u[1]);
    let v = [5, 1, 2, 3];
    println!("hr {}", head_rest(&v));
    println!("t2 {}", tail2(&v));
    println!("mid {}", mid_sum(&v));
    println!("hr2 {}", head_rest(&[9, 8]));
    let mut arr3 = [0, 1, 2, 3, 4, 5];
    arr3[2..5][1] = 77;
    println!("nested {}", arr3[3]);
    let w = &arr[1..4][1..];
    println!("subsub {} {}", w.len(), w[0]);
}
"#;

#[test]
fn run_mode_subslice_projections_read_write_and_patterns() {
    // Subslice places (from slice patterns and range indexing)
    // convert and execute: shared range reads, writes through
    // `IndexMut` ranges starting above zero, `@ ..` bindings, and
    // nested sub-slices all alias the base storage.
    let dir = temp_dir("subslice");
    let src = fixture(&dir, "subslice.rs", SUBSLICE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "subslice program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 2 sum 5\nw 99\nhead 2\ntail 9\nhr 8\nt2 23\nmid 3\nhr2 10\nnested 77\nsubsub 2 3\n",
        "subslice stdout mismatch; stderr={stderr}"
    );
}

const SUBSLICE_OOB_SRC: &str = r#"
fn main() {
    let v = [1, 2, 3];
    let s: &[i32] = &v;
    let a = 0usize;
    let b = 9usize;
    let _bad = &s[a..b];
    println!("unreachable");
}
"#;

#[test]
fn run_mode_subslice_out_of_range_panics() {
    // A dynamic range past the end of the slice is a guest panic
    // (core's range-bounds check), exit 101, like native.
    let dir = temp_dir("subslice-oob");
    let src = fixture(&dir, "subslice_oob.rs", SUBSLICE_OOB_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "out-of-range subslice must panic; stderr={stderr}"
    );
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("unreachable"),
        "the program must not continue past the panic"
    );
}

const PATTERN_TYPE_SRC: &str = r#"
#![feature(pattern_types, pattern_type_macro)]
use std::pat::pattern_type;
type Y = pattern_type!(u32 is 1..);
struct Thing(Y);
fn id_thing(t: Thing) -> Thing { t }
fn describe(z: Option<Y>) -> u32 {
    match z { Some(_) => 1, None => 0 }
}
fn main() {
    let a = id_thing(Thing(2));
    let _ = a.0;
    println!("some {}", describe(Some(42)));
    println!("none {}", describe(None));
    println!("size {}", std::mem::size_of::<Y>());
    let arr = [Thing(1), Thing(2)];
    println!("moved {}", describe(Some(arr[0].0)) + describe(Some(arr[1].0)));
}
"#;

#[test]
fn run_mode_pattern_type_values_execute_as_base_type() {
    // Pattern types (`Ty::Pat`) convert as their base type: the
    // values move through calls, arrays and enum construction
    // with the base representation. (The fixture needs the
    // internal feature; the driver test harness runs with
    // RUSTC_BOOTSTRAP=1, like compiletest.)
    let dir = temp_dir("pattern-type");
    let src = fixture(&dir, "pattern_type.rs", PATTERN_TYPE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "pattern-type program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "some 1\nnone 0\nsize 4\nmoved 2\n",
        "pattern-type stdout mismatch; stderr={stderr}"
    );
}

// ----- coroutine drop glue and runtime support (Lane FF) ---------------------

const COROUTINE_SUSPENDED_DROP_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
use std::ops::Coroutine;
use std::pin::Pin;
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let mut g = #[coroutine] || {
        let d = D(1);
        yield;
        println!("resumed {}", d.0);
    };
    let _s = Pin::new(&mut g).resume(());
    println!("suspended");
    drop(g);
    println!("after drop");
}
"#;

#[test]
fn run_mode_suspended_coroutine_drop_runs_live_local_dtor() {
    // Dropping a coroutine suspended at a yield runs the
    // destructors of the locals live in that state, at the drop
    // point - the state the value is in selects the glue.
    let dir = temp_dir("coroutine-suspended-drop");
    let src = fixture(&dir, "coroutine_suspended_drop.rs", COROUTINE_SUSPENDED_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "suspended coroutine drop must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "suspended\ndtor 1\nafter drop\n",
        "the live local's destructor runs when the suspended coroutine is dropped"
    );
}

const COROUTINE_COMPLETED_DROP_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
use std::ops::Coroutine;
use std::pin::Pin;
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let mut g = #[coroutine] || {
        let d = D(1);
        yield;
        println!("resumed {}", d.0);
    };
    let _s = Pin::new(&mut g).resume(());
    let _s2 = Pin::new(&mut g).resume(());
    println!("completed");
    drop(g);
    println!("after drop");
}
"#;

#[test]
fn run_mode_completed_coroutine_drop_runs_dtor_once() {
    // A completed coroutine's locals were already dropped by the
    // coroutine's own completion path; dropping the exhausted
    // value afterwards runs nothing a second time.
    let dir = temp_dir("coroutine-completed-drop");
    let src = fixture(&dir, "coroutine_completed_drop.rs", COROUTINE_COMPLETED_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "completed coroutine drop must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "resumed 1\ndtor 1\ncompleted\nafter drop\n",
        "the destructor runs exactly once, inside the completion"
    );
}

const COROUTINE_UNRESUMED_DROP_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let d = D(9);
    let g = #[coroutine] move || { let dd = d; yield; println!("inner {}", dd.0); };
    println!("created");
    drop(g);
    println!("after drop");
}
"#;

#[test]
fn run_mode_unresumed_coroutine_drop_runs_capture_dtor() {
    // Dropping a coroutine that was never resumed drops the
    // values it captured, at the drop point.
    let dir = temp_dir("coroutine-unresumed-drop");
    let src = fixture(&dir, "coroutine_unresumed_drop.rs", COROUTINE_UNRESUMED_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "unresumed coroutine drop must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "created\ndtor 9\nafter drop\n",
        "the captured value's destructor runs when the unresumed coroutine is dropped"
    );
}

const PIN_BOX_DROP_SRC: &str = r#"
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    { let _p = std::pin::Pin::new(Box::new(D(1))); }
    println!("scope done");
    let p = std::pin::Pin::new(Box::new(D(2)));
    std::mem::drop(p);
    println!("end");
}
"#;

#[test]
fn run_mode_pin_box_drop_and_mem_drop_run_pointee_dtor() {
    // `Pin<P>` has no destructor of its own: dropping it drops
    // the wrapped pointer, so a `Pin<Box<T>>` runs the pointee's
    // glue both at scope end and through `mem::drop`.
    let dir = temp_dir("pin-box-drop");
    let src = fixture(&dir, "pin_box_drop.rs", PIN_BOX_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Pin<Box<T>> drops must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "dtor 1\nscope done\ndtor 2\nend\n",
        "the pointee destructor runs at each Pin drop"
    );
}

const PIN_REF_DEREF_SRC: &str = r#"
fn main() {
    let mut x = 5;
    let mut p = std::pin::Pin::new(&mut x);
    *p = 8;
    println!("{}", *p);
    println!("{}", x);
}
"#;

#[test]
fn run_mode_pin_ref_deref_reads_and_writes_through() {
    // Dereferencing a `Pin<&mut T>` goes through the wrapped
    // reference: reads and writes land on the pointee.
    let dir = temp_dir("pin-ref-deref");
    let src = fixture(&dir, "pin_ref_deref.rs", PIN_REF_DEREF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Pin<&mut T> deref must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "8\n8\n",
        "the write through the Pin is visible on the pointee"
    );
}

const ASSERT_UNWIND_SAFE_DROP_SRC: &str = r#"
use std::panic::AssertUnwindSafe;
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    { let _w = AssertUnwindSafe(D(3)); }
    println!("after");
    let w = AssertUnwindSafe(D(4));
    std::mem::drop(w);
    println!("end");
}
"#;

#[test]
fn run_mode_assert_unwind_safe_drop_runs_held_dtor() {
    // The one-field `AssertUnwindSafe` wrapper's glue is the held
    // value's glue, at scope end and through `mem::drop` alike.
    let dir = temp_dir("assert-unwind-safe-drop");
    let src = fixture(&dir, "assert_unwind_safe_drop.rs", ASSERT_UNWIND_SAFE_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "AssertUnwindSafe drops must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "dtor 3\nafter\ndtor 4\nend\n",
        "the held value's destructor runs at each wrapper drop"
    );
}

const CATCH_UNWIND_RESUME_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
use std::ops::Coroutine;
use std::pin::Pin;
use std::panic;
fn main() {
    let mut foo = #[coroutine] || { yield; };
    let res = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        Pin::new(&mut foo).resume(())
    }));
    println!("err={}", res.is_err());
}
"#;

#[test]
fn run_mode_catch_unwind_resume_coroutine() {
    // The panic machinery's path through a coroutine resume:
    // `AssertUnwindSafe` derefs to the closure, the resume result
    // (`CoroutineState`) drops out of the returned `Result`.
    let dir = temp_dir("catch-unwind-resume");
    let src = fixture(&dir, "catch_unwind_resume.rs", CATCH_UNWIND_RESUME_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "catch_unwind around a resume must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "err=false\n",
        "the resume completed without a panic"
    );
}

const ASYNC_SIZE_OF_VAL_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

struct Nop;
impl Future for Nop {
    type Output = u8;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<u8> { Poll::Ready(3) }
}

async fn one() -> u8 { Nop.await }

fn main() {
    println!("size {}", std::mem::size_of_val(&one()));
    println!("size {}", std::mem::size_of_val(&Nop));
}
"#;

#[test]
fn run_mode_size_of_val_on_async_future_uses_compiler_layout() {
    // `size_of_val` on a future answers with the compiler's own
    // state-machine layout for that coroutine instance (the same
    // values a native build prints), not with a model guess.
    let dir = temp_dir("async-size-of-val");
    let src = fixture(&dir, "async_size_of_val.rs", ASYNC_SIZE_OF_VAL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "size_of_val on a future must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "size 1\nsize 0\n",
        "the future's size is the compiler layout size"
    );
}

fn run_driver_extra(src: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .args(extra)
        .output()
        .expect("failed to run sparagmos-driver")
}

const ENV_ARGS_SRC: &str = r#"
fn main() {
    for a in std::env::args() {
        println!("arg:{a}");
    }
    let mut os = std::env::args_os();
    println!("os-first {}", os.next().is_some());
    println!("count {}", std::env::args().count());
}
"#;

#[test]
fn run_mode_env_args_reach_the_guest() {
    let dir = temp_dir("env-args");
    let src = fixture(&dir, "args.rs", ENV_ARGS_SRC);
    let out = run_driver_extra(&src, &["--guest-arg", "one", "--guest-arg", "two"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 5, "stdout={stdout}");
    // argv[0] is the guest's program path: the input source.
    assert!(lines[0].starts_with("arg:"), "stdout={stdout}");
    assert!(lines[0].ends_with("args.rs"), "stdout={stdout}");
    assert_eq!(lines[1], "arg:one");
    assert_eq!(lines[2], "arg:two");
    assert_eq!(lines[3], "os-first true");
    assert_eq!(lines[4], "count 3");
}

const ENV_ARGS_LEN_SRC: &str = r#"
fn main() {
    println!("len {}", std::env::args().len());
    println!("os-len {}", std::env::args_os().len());
    let mut it = std::env::args();
    println!("first {}", it.next().unwrap());
    println!("len-after {}", it.len());
    let mut os = std::env::args_os();
    println!("os-first {}", os.next().is_some());
    println!("os-len-after {}", os.len());
}
"#;

#[test]
fn run_mode_env_args_len_uses_the_real_wrapper_layout() {
    // `ExactSizeIterator::len` for `Args` / `ArgsOs` is not a
    // builtin: the collected std one-liners navigate the real
    // wrapper layout (`ArgsOs { inner: sys::args::common::Args {
    // inner: vec::IntoIter } }`) field by field and land in the
    // `size_hint` builtin on the innermost iterator. The modelled
    // value must carry all three wrapper layers or the second
    // field step reads the element snapshot as the state.
    let dir = temp_dir("env-args-len");
    let src = fixture(&dir, "args_len.rs", ENV_ARGS_LEN_SRC);
    let out = run_driver_extra(&src, &["--guest-arg", "one", "--guest-arg", "two"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 6, "stdout={stdout}");
    assert_eq!(lines[0], "len 3");
    assert_eq!(lines[1], "os-len 3");
    assert!(lines[2].starts_with("first "), "stdout={stdout}");
    assert!(lines[2].contains("args_len.rs"), "stdout={stdout}");
    assert_eq!(lines[3], "len-after 2");
    assert_eq!(lines[4], "os-first true");
    assert_eq!(lines[5], "os-len-after 2");
}

const BOXED_VIEWS_SRC: &str = r#"fn main() {
    let b: Box<str> = "hi".to_owned().into_boxed_str();
    println!("boxed {}", &*b);
    let v: Box<[i32]> = vec![10, 20, 30].into_boxed_slice();
    println!("slice {} {}", v.len(), v[1]);
    let mut s: Box<str> = "ab".to_owned().into_boxed_str();
    s.make_ascii_uppercase();
    println!("upper {}", &*s);
    let text = String::from("hello");
    let r: &str = text.as_ref();
    println!("asref {}", r);
    let bytes: &[u8] = text.as_ref();
    println!("bytes {}", bytes.len());
    let vec = vec![1u8, 2, 3];
    let vb: &[u8] = vec.as_ref();
    println!("vec {}", vb.len());
    let mut m = String::from("xy");
    let mm: &mut str = m.as_mut();
    mm.make_ascii_uppercase();
    println!("asmut {}", m);
}
"#;

const CALLER_LOCATION_INLINED_SRC: &str = r#"macro_rules! caller_location_from_macro {
    () => (core::panic::Location::caller());
}

fn main() {
    let loc = core::panic::Location::caller();
    assert_eq!(loc.file(), file!());
    assert_eq!(loc.line(), 6);
    assert_eq!(loc.column(), 15);

    let loc2 = caller_location_from_macro!();
    assert_eq!(loc2.file(), file!());
    assert_eq!(loc2.line(), 11);
    assert_eq!(loc2.column(), 16);
}
"#;

const THREAD_JOIN_OPT_SRC: &str = r#"fn main() {
    let h = std::thread::spawn(|| 40 + 2);
    let v = h.join().unwrap();
    assert_eq!(v, 42);
}
"#;

const VEC_CELL_INDEX_SRC: &str = r#"use std::cell::Cell;
fn main() {
    let mut v = vec![Cell::new(1u32), Cell::new(2u32)];
    v[0].set(5);
    v[1].set(v[0].get() + 1);
    assert_eq!(v[0].get(), 5);
    assert_eq!(v[1].get(), 6);
    println!("vec cell ok");
    // The vector is dropped at the end of `main`: the `Drop`
    // terminator for the cell element glue runs as well.
}
"#;

const NESTED_BOX_MOVE_SRC: &str = r#"struct A {
    x: Box<isize>,
    y: isize,
}
struct C {
    x: Box<A>,
    y: isize,
}
fn main() {
    let a: Box<C> = Box::new(C { x: Box::new(A { x: Box::new(0), y: 1 }), y: 2 });
    let _x = a.x.x;
    let _y = a.y;
    println!("nested ok");
}
"#;

#[test]
fn run_mode_nested_box_partial_move_and_drop() {
    // Moving `a.x.x` out of nested boxes and dropping the
    // remainder runs the drop shims, whose places walk the
    // `Box` -> `Unique` -> `NonNull` -> pointer field chain:
    // in the value model the box's one field is the pointer
    // itself, so each field-0 step of that chain yields it
    // unchanged.
    let dir = temp_dir("nested-box-move");
    let src = fixture(&dir, "nested_box.rs", NESTED_BOX_MOVE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "nested ok\n");
}

#[test]
fn run_mode_shared_index_write_through_cell() {
    // `v[i].set(..)` mutates through the *shared* element
    // reference `Index::index` returns: the reference must name
    // the vector's storage (a live place), not a snapshot of the
    // element, or the `Cell` write has no place to land.
    let dir = temp_dir("vec-cell-index");
    let src = fixture(&dir, "vec_cell.rs", VEC_CELL_INDEX_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "vec cell ok\n");
}

const VEC_CELL_DROP_SRC: &str = r#"use std::cell::Cell;
struct D(i32);
impl Drop for D { fn drop(&mut self) { println!("drop D({})", self.0); } }
fn main() {
    let v = vec![Cell::new(D(1)), Cell::new(D(2))];
    drop(v);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_cell_drop_runs_element_destructors() {
    // Dropping a `Vec<Cell<D>>` runs the element drop glue
    // through the cells: each element's destructor observes the
    // value the cell holds.
    let dir = temp_dir("vec-cell-drop");
    let src = fixture(&dir, "vec_cell_drop.rs", VEC_CELL_DROP_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "drop D(1)\ndrop D(2)\ndone\n");
}

const PRINTLN_DYNAMIC_COUNT_SRC: &str = r#"fn main() {
    let w = 6usize;
    println!("{:w$}", 42);
    println!("{:.*}", 2, 3.14159f64);
}
"#;

#[test]
fn run_mode_println_dynamic_width_and_precision() {
    // A dynamic width / precision (`{:w$}`, `{:.*}`) reaches
    // the formatter through `rt::Argument::from_usize`, which
    // current core calls with a *reference* to the count (the
    // format-args capture tuple holds references): the builtin
    // reads through the reference to the count value.
    let dir = temp_dir("println-dynamic-count");
    let src = fixture(&dir, "println_count.rs", PRINTLN_DYNAMIC_COUNT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "    42\n3.14\n");
}

const TRANSMUTE_NONNULL_SRC: &str = r#"use std::ptr::NonNull;
fn main() {
    let p = NonNull::<u8>::dangling();
    let n: usize = unsafe { std::mem::transmute(p) };
    assert!(n != 0);
    let q: NonNull<u8> = unsafe { std::mem::transmute(n) };
    assert_eq!(q, p);
    println!("nonnull roundtrip ok");
}
"#;

#[test]
fn run_mode_transmute_nonnull_integer_roundtrip() {
    // `NonNull::dangling` transmutes a `mem::Alignment` into
    // the pointer wrapper (the discriminant is the address),
    // and user code transmutes the wrapper to `usize` and
    // back: the wrapper <-> integer transmutes carry the
    // payload pointer's synthetic address.
    let dir = temp_dir("transmute-nonnull");
    let src = fixture(&dir, "transmute_nonnull.rs", TRANSMUTE_NONNULL_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "nonnull roundtrip ok\n");
}

const OPTION_NONNULL_NICHE_SRC: &str = r#"use std::ptr::NonNull;
fn main() {
    let x = 7u8;
    let some: Option<NonNull<u8>> = NonNull::new(&x as *const u8 as *mut u8);
    // The niche encoding: the option *is* the pointer word,
    // zero for `None`. (The roundtrip is checked on the bits:
    // comparing an address-only pointer against a place
    // pointer is a separate, older gap.)
    let bits: usize = unsafe { std::mem::transmute(some) };
    assert!(bits != 0);
    let back: Option<NonNull<u8>> = unsafe { std::mem::transmute(bits) };
    assert!(back.is_some());
    let bits2: usize = unsafe { std::mem::transmute(back) };
    assert_eq!(bits2, bits);
    assert_eq!(unsafe { *some.unwrap().as_ptr() }, 7);
    let none: Option<NonNull<u8>> = None;
    let zero: usize = unsafe { std::mem::transmute(none) };
    assert_eq!(zero, 0);
    let back_none: Option<NonNull<u8>> = unsafe { std::mem::transmute(0usize) };
    assert!(back_none.is_none());
    // The same encoding through a raw pointer target type.
    let p: *const u32 = unsafe { std::mem::transmute(some) };
    assert!(!p.is_null());
    println!("option nonnull niche ok");
}
"#;

#[test]
fn run_mode_option_nonnull_niche_transmute() {
    // `Option<NonNull<T>>` uses the pointer's null niche as
    // its encoding, so a transmute between the option and an
    // integer (or a raw pointer) is the identity on the
    // address word, with zero reading back as `None`.
    let dir = temp_dir("option-nonnull-niche");
    let src = fixture(&dir, "option_nonnull_niche.rs", OPTION_NONNULL_NICHE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "option nonnull niche ok\n");
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

#[test]
fn run_mode_thread_join_at_optimized_mir() {
    // Optimizing MIR inlines `JoinHandle::join`: field 0 (the
    // `JoinInner`) is moved out of the handle and
    // `JoinInner::join` is called directly. The modelled spawn
    // result nests the thread-id handle inside the join handle
    // so the move yields a `JoinInner`-shaped value, and the
    // inner join is served by the same builtin as
    // `JoinHandle::join`.
    let dir = temp_dir("thread-join-opt");
    let src = fixture(&dir, "thread_join_opt.rs", THREAD_JOIN_OPT_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-O"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
}

#[test]
fn run_mode_caller_location_after_inlining() {
    // At high MIR optimization levels `Location::caller` is
    // inlined into the calling body, leaving a bare
    // `caller_location` intrinsic call: the stamp must come
    // from that call site's own span (the intrinsic is not
    // itself `#[track_caller]`, so the usual stamping gate
    // skips it).
    let dir = temp_dir("caller-location-inlined");
    let src = fixture(&dir, "caller_location.rs", CALLER_LOCATION_INLINED_SRC);
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "-Zmir-opt-level=4"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "");
}

#[test]
fn run_mode_boxed_conversions_and_ref_views() {
    // `String::into_boxed_str` and `Vec::into_boxed_slice` move
    // the `vec` field out of the value and rebuild a box from
    // raw parts in their collected bodies, which the string /
    // vector models cannot serve; the `AsRef` / `AsMut` bodies
    // for the std containers build their views the same way.
    // The builtins compute the views from the modelled data
    // directly.
    let dir = temp_dir("boxed-views");
    let src = fixture(&dir, "boxed_views.rs", BOXED_VIEWS_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        [
            "boxed hi",
            "slice 3 20",
            "upper AB",
            "asref hello",
            "bytes 5",
            "vec 3",
            "asmut XY",
        ]
    );
}

const VEC_AS_REF_SLICE_SRC: &str = r#"fn main() {
    let v: Vec<i32> = vec![5, 3, 8, 1];
    let s: &[i32] = v.as_ref();
    println!("idx {} {} {}", s[0], s[2], s[3]);
    println!("first {:?}", s.first());
    println!("get {:?}", s.get(2));
    println!("debug {:?}", s);
    println!("sub {:?}", &s[1..3]);
    println!("contains {}", s.contains(&8));
    println!("eq {}", s == [5, 3, 8, 1].as_slice());
    println!("to_vec {:?}", s.to_vec());
    let mut w: Vec<i32> = Vec::with_capacity(16);
    w.push(10);
    w.push(20);
    let ws: &[i32] = w.as_ref();
    println!("sparse {} {} len {}", ws[0], ws[1], ws.len());
    let e: Vec<i32> = Vec::new();
    let es: &[i32] = e.as_ref();
    println!("empty {} {:?}", es.len(), es.first());
    let bv = Box::new(vec![7, 9]);
    let bs: &[i32] = bv.as_ref();
    println!("boxvec {} {}", bs[0], bs[1]);
    let rv = std::rc::Rc::new(vec![4, 6]);
    let rs: &[i32] = rv.as_ref();
    println!("rcvec {} {}", rs[0], rs[1]);
    let vs: Vec<String> = vec!["aa".to_string(), "bbb".to_string()];
    let ss: &[String] = vs.as_ref();
    println!("strings {} {} {}", ss[0], ss[1], ss[1].len());
    let vv: Vec<Vec<i32>> = vec![vec![1, 2], vec![3]];
    let sv: &[Vec<i32>] = vv.as_ref();
    println!("nested {} {}", sv[0][1], sv[1][0]);
}
"#;

#[test]
fn run_mode_vec_as_ref_slice_element_access() {
    // Issue #443: `Vec::as_ref` built its slice view as a fat
    // pointer over the vector aggregate itself, a bare ADT value
    // no dereference can follow: `len` and iteration (which read
    // the fat pointer itself) worked, but element access through
    // the view (`s[0]`, `first`, `get`, sub-slicing, `Debug`)
    // failed with "type mismatch in dereference: expected
    // reference, found ADT value". The view now denotes the
    // element sequence, like every other slice construction.
    let dir = temp_dir("vec-as-ref-slice");
    let src = fixture(&dir, "vec_as_ref_slice.rs", VEC_AS_REF_SLICE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        [
            "idx 5 8 1",
            "first Some(5)",
            "get Some(8)",
            "debug [5, 3, 8, 1]",
            "sub [3, 8]",
            "contains true",
            "eq true",
            "to_vec [5, 3, 8, 1]",
            "sparse 10 20 len 2",
            "empty 0 None",
            "boxvec 7 9",
            "rcvec 4 6",
            "strings aa bbb 3",
            "nested 2 3",
        ]
    );
}

const CALLER_LOCATION_TRACK_CALLER_OPT_SRC: &str = r#"#[inline(never)]
#[track_caller]
fn codegen_caller_loc() -> &'static core::panic::Location<'static> {
    core::panic::Location::caller()
}

macro_rules! caller_location_from_macro {
    () => (codegen_caller_loc());
}

fn main() {
    let loc = codegen_caller_loc();
    assert_eq!(loc.line(), 12);
    assert_eq!(loc.column(), 15);

    let loc2 = caller_location_from_macro!();
    assert_eq!(loc2.line(), 16);
    assert_eq!(loc2.column(), 16);
}
"#;

#[test]
fn run_mode_caller_location_track_caller_at_optimized_mir() {
    // At mir-opt level 4 `Location::caller` is inlined into
    // the `#[track_caller]` body (a bare intrinsic call), and
    // the caller's call carries its function operand as a
    // promoted constant whose span is erased: the stamp for
    // the track-caller call must still name the call site,
    // taken from the call terminator's own span.
    let dir = temp_dir("caller-location-track-caller-opt");
    let src = fixture(
        &dir,
        "caller_location_opt.rs",
        CALLER_LOCATION_TRACK_CALLER_OPT_SRC,
    );
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "-Zmir-opt-level=4"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
}

const CALLER_LOCATION_SRC: &str = r#"
#[track_caller]
fn here() -> u32 {
    std::panic::Location::caller().line()
}
fn main() {
    println!("tracked {}", here());
    println!("direct {}", std::panic::Location::caller().line());
}
"#;

#[test]
fn run_mode_caller_location_reports_the_call_site() {
    let dir = temp_dir("caller-location");
    let src = fixture(&dir, "loc.rs", CALLER_LOCATION_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // `here()` is called on line 7 of the fixture; the direct
    // `Location::caller()` is evaluated on line 8.
    assert_eq!(stdout, "tracked 7\ndirect 8\n", "stdout={stdout}");
}

const CALLER_LOCATION_FNPTR_SRC: &str = r#"
fn ptr_call(f: fn() -> u32) -> u32 { f() }
#[track_caller]
fn tracked() -> u32 { std::panic::Location::caller().line() }
trait Tr { fn m() -> u32; }
impl Tr for () {
    #[track_caller]
    fn m() -> u32 { std::panic::Location::caller().line() }
}
fn main() {
    println!("fn {}", ptr_call(tracked));
    println!("impl {}", ptr_call(<() as Tr>::m));
}
"#;

#[test]
fn run_mode_caller_location_through_fn_pointer_is_the_definition() {
    // A `#[track_caller]` function entered through a function
    // pointer observes its own definition location (the call
    // shim's location), for a free function (line 4) and for a
    // trait method item (the impl method, line 8) alike.
    run_ok(
        "caller-location-fnptr",
        CALLER_LOCATION_FNPTR_SRC,
        "fn 4\nimpl 8\n",
    );
}

const CALLER_LOCATION_DYN_SRC: &str = r#"
trait Tr {
    #[track_caller]
    fn def_tracked(&self) -> (u32, u32) {
        let l = std::panic::Location::caller();
        (l.line(), l.column())
    }
    fn impl_only(&self) -> (u32, u32);
}
impl Tr for () {
    #[track_caller]
    fn impl_only(&self) -> (u32, u32) {
        let l = std::panic::Location::caller();
        (l.line(), l.column())
    }
}
fn main() {
    let t: &dyn Tr = &();
    let (l1, c1) = t.def_tracked();
    println!("def {l1} {c1}");
    let (l2, c2) = t.impl_only();
    println!("impl {l2} {c2}");
}
"#;

#[test]
fn run_mode_caller_location_through_dyn_trait_object() {
    // A trait definition's `#[track_caller]` method called on
    // a trait object observes the call site (line 19, the
    // method name at column 22); an impl-only `#[track_caller]`
    // method observes its definition (the impl method at
    // 12:5).
    run_ok(
        "caller-location-dyn",
        CALLER_LOCATION_DYN_SRC,
        "def 19 22\nimpl 12 5\n",
    );
}

const CALLER_LOCATION_CLOSURE_SRC: &str = r#"
#![feature(stmt_expr_attributes)]
#![feature(closure_track_caller)]
use std::panic::Location;
type Loc = &'static Location<'static>;
#[track_caller]
fn mono_fn<F: Fn() -> Loc>(v: &F) -> Loc { v() }
#[track_caller]
fn mono_once<F: FnOnce() -> Loc>(v: F) -> Loc { v() }
#[track_caller]
fn dyn_mut(v: &mut dyn FnMut() -> Loc) -> Loc { v() }
#[track_caller]
fn dyn_once(v: Box<dyn FnOnce() -> Loc>) -> Loc { v() }
fn main() {
    let mut c = #[track_caller] || Location::caller();
    let d = c();
    println!("direct {} {}", d.line(), d.column());
    let m = mono_fn(&c);
    println!("mono {} {}", m.line(), m.column());
    let dm = dyn_mut(&mut c);
    println!("dynmut-other {}", dm.file() != file!());
    let dn = dyn_once(Box::new(c));
    println!("dynonce-other {}", dn.file() != file!());
    let mo = mono_once(#[track_caller] || Location::caller());
    println!("monoonce {} {}", mo.line(), mo.column());
}
"#;

#[test]
fn run_mode_caller_location_through_closures() {
    // A track-caller closure called directly observes the call
    // site (16:13); through a `#[track_caller]` generic `Fn` /
    // `FnOnce` wrapper the location forwards to the wrapper's
    // call site (18:13 and 24:14); through a `dyn` receiver it
    // observes the `Fn` shim's location in core, not this file.
    run_ok(
        "caller-location-closure",
        CALLER_LOCATION_CLOSURE_SRC,
        "direct 16 13\nmono 18 13\ndynmut-other true\ndynonce-other true\nmonoonce 24 14\n",
    );
}

const PROCESS_EXIT_SRC: &str = r#"
fn main() {
    println!("before");
    std::process::exit(7);
}
"#;

const EXIT_CODE_SRC: &str = r#"
fn main() -> std::process::ExitCode {
    std::process::ExitCode::from(3)
}
"#;

#[test]
fn run_mode_process_exit_and_exit_code() {
    let dir = temp_dir("exit-codes");
    let src = fixture(&dir, "exit.rs", PROCESS_EXIT_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(7), "process::exit(7)");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "before\n");
    let src = fixture(&dir, "exitcode.rs", EXIT_CODE_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(3), "ExitCode::from(3)");
}

const RESULT_OK_SRC: &str = r#"
fn main() -> Result<(), String> {
    println!("hello");
    Ok(())
}
"#;

const RESULT_ERR_SRC: &str = r#"
fn main() -> Result<(), String> {
    Err("boom".to_string())
}
"#;

const RESULT_ERR_IO_SRC: &str = r#"
fn main() -> Result<(), std::io::Error> {
    Err(std::io::Error::from_raw_os_error(2))
}
"#;

#[test]
fn run_mode_termination_result_report() {
    // A `fn main() -> Result<_, _>` entry reports through
    // `Termination`: `Ok` exits 0, `Err` writes `Error: {err:?}`
    // to stderr and exits with `ExitCode::FAILURE` (1).
    let dir = temp_dir("termination-result");
    let src = fixture(&dir, "ok.rs", RESULT_OK_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(0), "Result::Ok");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hello\n");
    let src = fixture(&dir, "err.rs", RESULT_ERR_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Result::Err(String)");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: \"boom\"\n"
    );
    let src = fixture(&dir, "err_io.rs", RESULT_ERR_IO_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Result::Err(io::Error)");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: Os { code: 2, kind: NotFound, \
         message: \"No such file or directory\" }\n"
    );
}

const RESULT_ERR_RANGE_FROM_SRC: &str = r#"
use std::ops::RangeFrom;
use std::path::PathBuf;

fn main() -> Result<(), RangeFrom<PathBuf>> {
    Err(RangeFrom {
        start: PathBuf::from("hello"),
    })
}
"#;

#[test]
fn run_mode_termination_range_from_pathbuf_report() {
    // The report renders the payload through the detached
    // `Debug` path (the payload's own formatting is never
    // demanded by the guest): a `RangeFrom` payload keeps
    // its handwritten form, `"hello"..`, rather than
    // collapsing to the held byte string.
    let dir = temp_dir("termination-range-from");
    let src = fixture(&dir, "err_range_from.rs", RESULT_ERR_RANGE_FROM_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Result::Err(RangeFrom<PathBuf>)");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: \"hello\"..\n"
    );
    assert!(
        out.stdout.is_empty(),
        "the report writes nothing to stdout: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

const RESULT_ERR_REVERSE_I32_SRC: &str = r#"
use std::cmp::Reverse;

fn main() -> Result<(), Reverse<i32>> {
    Err(Reverse(42))
}
"#;

const RESULT_ERR_REVERSE_PATHBUF_SRC: &str = r#"
use std::cmp::Reverse;
use std::path::PathBuf;

fn main() -> Result<(), Reverse<PathBuf>> {
    Err(Reverse(PathBuf::from("hello")))
}
"#;

const RESULT_ERR_WRAPPING_STRING_SRC: &str = r#"
use std::num::Wrapping;

fn main() -> Result<(), Wrapping<String>> {
    Err(Wrapping(String::from("hello")))
}
"#;

const RESULT_ERR_WRAPPING_I32_SRC: &str = r#"
use std::num::Wrapping;

fn main() -> Result<(), Wrapping<i32>> {
    Err(Wrapping(7))
}
"#;

#[test]
fn run_mode_termination_reverse_wrapping_report() {
    // The one-field transparent wrappers `Reverse` / `Wrapping`
    // as a report payload: the rendering matches native, and
    // the payload's drop glue is accounted for by the held
    // value's glue (the wrappers have no `Drop` impl of their
    // own), so the run exits 1 like native instead of failing
    // the drop accounting after rendering. Paired with
    // `run_mode_termination_range_from_pathbuf_report`.
    let dir = temp_dir("termination-reverse-wrapping");
    let src = fixture(&dir, "err_reverse_i32.rs", RESULT_ERR_REVERSE_I32_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Result::Err(Reverse<i32>)");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: Reverse(42)\n"
    );
    let src = fixture(
        &dir,
        "err_reverse_pathbuf.rs",
        RESULT_ERR_REVERSE_PATHBUF_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(1),
        "Result::Err(Reverse<PathBuf>)"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: Reverse(\"hello\")\n"
    );
    let src = fixture(
        &dir,
        "err_wrapping_string.rs",
        RESULT_ERR_WRAPPING_STRING_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(1),
        "Result::Err(Wrapping<String>)"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: \"hello\"\n"
    );
    let src = fixture(&dir, "err_wrapping_i32.rs", RESULT_ERR_WRAPPING_I32_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Result::Err(Wrapping<i32>)");
    assert_eq!(String::from_utf8_lossy(&out.stderr), "Error: 7\n");
    assert!(
        out.stdout.is_empty(),
        "the report writes nothing to stdout: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

const RESULT_OK_EXIT_CODE_SRC: &str = r#"
fn main() -> Result<std::process::ExitCode, String> {
    Ok(std::process::ExitCode::from(3))
}
"#;

const RESULT_NESTED_ERR_SRC: &str = r#"
fn main() -> Result<Result<(), String>, String> {
    Ok(Err("inner".to_string()))
}
"#;

const RESULT_NESTED_OK_SRC: &str = r#"
fn main() -> Result<Result<(), String>, String> {
    println!("deep");
    Ok(Ok(()))
}
"#;

const RESULT_NESTED_EXIT_CODE_SRC: &str = r#"
fn main() -> Result<Result<std::process::ExitCode, String>, String> {
    Ok(Ok(std::process::ExitCode::from(5)))
}
"#;

const RESULT_ERR_RC_SRC: &str = r#"
fn main() -> Result<(), std::rc::Rc<String>> {
    Err(std::rc::Rc::new(String::from("boom")))
}
"#;

#[test]
fn run_mode_termination_result_delegates_and_renders_live() {
    // `Result<T, E>::report` delegates `Ok` to `T::report`: an
    // `ExitCode` payload yields its code (not an unconditional
    // 0), and a nested `Result` recurses -- its own `Err`
    // prints and fails, its own `Ok(ExitCode)` yields the code.
    // An `Err` payload that points into the run's allocations
    // (an `Rc`) renders its actual Debug text, not the value's
    // structural form: the report is rendered by the interpreter
    // that executed the entry, before its state is discarded.
    let dir = temp_dir("termination-delegate");
    let src = fixture(&dir, "ok_exit_code.rs", RESULT_OK_EXIT_CODE_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(3), "Ok(ExitCode::from(3))");
    let src = fixture(&dir, "nested_err.rs", RESULT_NESTED_ERR_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Ok(Err(..))");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: \"inner\"\n"
    );
    let src = fixture(&dir, "nested_ok.rs", RESULT_NESTED_OK_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(0), "Ok(Ok(()))");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "deep\n");
    let src = fixture(&dir, "nested_exit_code.rs", RESULT_NESTED_EXIT_CODE_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(5), "Ok(Ok(ExitCode::from(5)))");
    let src = fixture(&dir, "err_rc.rs", RESULT_ERR_RC_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Err(Rc<String>)");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: \"boom\"\n"
    );
}

const RESULT_ERR_PANICKING_DEBUG_SRC: &str = r#"
use std::fmt;
struct Boom;
impl fmt::Debug for Boom {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!("fmt exploded");
    }
}
fn main() -> Result<(), Boom> {
    Err(Boom)
}
"#;

const RESULT_ERR_EXITING_DEBUG_SRC: &str = r#"
use std::fmt;
struct Exiter;
impl fmt::Debug for Exiter {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        std::process::exit(7);
    }
}
fn main() -> Result<(), Exiter> {
    Err(Exiter)
}
"#;

const RESULT_ERR_TLS_DTOR_ORDER_SRC: &str = r#"
use std::cell::RefCell;
struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        eprintln!("tls dtor ran");
    }
}
thread_local! {
    static GUARD: RefCell<Option<Guard>> = RefCell::new(None);
}
fn main() -> Result<(), String> {
    GUARD.with(|g| *g.borrow_mut() = Some(Guard));
    Err("boom".to_string())
}
"#;

const RESULT_ERR_TLS_READING_DEBUG_SRC: &str = r#"
use std::cell::RefCell;
use std::fmt;
thread_local! {
    static VAL: RefCell<Option<String>> = RefCell::new(None);
}
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = VAL.with(|v| v.borrow().clone());
        write!(f, "val={s:?}")
    }
}
fn main() -> Result<(), E> {
    VAL.with(|v| *v.borrow_mut() = Some("alive".to_string()));
    Err(E)
}
"#;

#[test]
fn run_mode_termination_report_guest_outcomes_and_tls_order() {
    // The report's rendering is guest execution: a panic inside
    // the `Err` payload's `Debug` body is the run's outcome (the
    // guest-panic exit, reported like any guest panic), and a
    // `process::exit` there passes its code through -- neither
    // collapses into the report's own failure exit with a
    // structural stand-in for the payload. The report also
    // applies before the entry thread's TLS destructors run, so
    // its line precedes destructor output in stderr, and a
    // `Debug` body that reads TLS still sees it alive.
    let dir = temp_dir("termination-report-outcomes");
    let src = fixture(&dir, "err_panicking_debug.rs", RESULT_ERR_PANICKING_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "panic in Debug::fmt");
    // The report's `Error: ` prefix is written before the
    // formatting starts (the native order), so it stands even
    // though the formatting panicked: it precedes the panic
    // line the driver prints for the run's outcome.
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: thread 'main' panicked: fmt exploded\n"
    );
    let src = fixture(&dir, "err_exiting_debug.rs", RESULT_ERR_EXITING_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(7), "process::exit in Debug::fmt");
    // Likewise the already-written prefix is the whole stderr
    // when the formatting exits the process.
    assert_eq!(String::from_utf8_lossy(&out.stderr), "Error: ");
    let src = fixture(&dir, "err_tls_dtor_order.rs", RESULT_ERR_TLS_DTOR_ORDER_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Err with TLS destructor");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: \"boom\"\ntls dtor ran\n"
    );
    let src = fixture(&dir, "err_tls_reading_debug.rs", RESULT_ERR_TLS_READING_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Debug::fmt reads TLS");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: val=Some(\"alive\")\n"
    );
}


const RESULT_ERR_DROPPING_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "E") }
}
impl Drop for E {
    fn drop(&mut self) { eprintln!("error dropped"); }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_DROP_EXITS_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "E") }
}
impl Drop for E {
    fn drop(&mut self) {
        eprintln!("error dropped");
        std::process::exit(7);
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_DROP_PANICS_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "E") }
}
impl Drop for E {
    fn drop(&mut self) { panic!("drop panicked"); }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_DROP_TLS_ORDER_SRC: &str = r#"
use std::cell::RefCell;
use std::fmt;
struct Guard;
impl Drop for Guard {
    fn drop(&mut self) { eprintln!("tls dtor ran"); }
}
thread_local! {
    static GUARD: RefCell<Option<Guard>> = RefCell::new(None);
}
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "E") }
}
impl Drop for E {
    fn drop(&mut self) { eprintln!("error dropped"); }
}
fn main() -> Result<(), E> {
    GUARD.with(|g| *g.borrow_mut() = Some(Guard));
    Err(E)
}
"#;

const RESULT_ERR_SIDE_EFFECT_DEBUG_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        eprintln!("side effect");
        write!(f, "E")
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_SIDE_EFFECT_EXIT_DEBUG_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        eprintln!("side effect");
        std::process::exit(5);
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_SIDE_EFFECT_PANIC_DEBUG_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        eprintln!("side effect");
        panic!("fmt panicked");
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

#[test]
fn run_mode_termination_report_drops_payload_and_prefix_order() {
    // The native `Result::report` writes the `Error: ` prefix
    // before it formats the payload, prints the formatted
    // payload, and then consumes and drops the error -- all
    // before the thread's TLS destructors run. So: stderr the
    // payload's `Debug` body writes follows the prefix (and
    // the prefix stands when the formatting exits or panics);
    // the payload's destructor runs after the whole
    // `Error: ...` line, its own guest outcome (an exit, a
    // panic) replaces the report's failure exit, and it still
    // precedes TLS destructor output.
    let dir = temp_dir("termination-report-drop");
    let src = fixture(&dir, "err_dropping.rs", RESULT_ERR_DROPPING_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "dropping Err payload");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: E\nerror dropped\n"
    );
    let src = fixture(&dir, "err_drop_exits.rs", RESULT_ERR_DROP_EXITS_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(7), "process::exit in payload Drop");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: E\nerror dropped\n"
    );
    let src = fixture(&dir, "err_drop_panics.rs", RESULT_ERR_DROP_PANICS_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "panic in payload Drop");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: E\nthread 'main' panicked: drop panicked\n"
    );
    let src = fixture(&dir, "err_drop_tls_order.rs", RESULT_ERR_DROP_TLS_ORDER_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "payload Drop before TLS dtors");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: E\nerror dropped\ntls dtor ran\n"
    );
    let src = fixture(&dir, "err_side_effect_debug.rs", RESULT_ERR_SIDE_EFFECT_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Debug::fmt writes to stderr");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: side effect\nE\n"
    );
    let src = fixture(&dir, "err_side_effect_exit_debug.rs", RESULT_ERR_SIDE_EFFECT_EXIT_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(5), "process::exit in Debug::fmt keeps prefix");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: side effect\n"
    );
    let src = fixture(&dir, "err_side_effect_panic_debug.rs", RESULT_ERR_SIDE_EFFECT_PANIC_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "panic in Debug::fmt keeps prefix");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: side effect\nthread 'main' panicked: fmt panicked\n"
    );
}

const RESULT_ERR_FMT_PANIC_DROP_EXITS_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!("fmt exploded");
    }
}
impl Drop for E {
    fn drop(&mut self) {
        eprintln!("dropped");
        std::process::exit(7);
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_FMT_PANIC_DROP_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!("fmt exploded");
    }
}
impl Drop for E {
    fn drop(&mut self) {
        eprintln!("dropped");
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_FMT_PANIC_DROP_PANICS_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!("fmt exploded");
    }
}
impl Drop for E {
    fn drop(&mut self) {
        panic!("drop exploded");
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_STREAMING_DEBUG_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "first")?;
        eprintln!("side effect");
        write!(f, "last")
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_PARTIAL_PANIC_DEBUG_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "partial")?;
        panic!("fmt exploded");
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_PARTIAL_EXIT_DEBUG_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "partial")?;
        std::process::exit(5);
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_DERIVED_DEBUG_SRC: &str = r#"
#[derive(Debug)]
struct E {
    x: i32,
    name: String,
}
fn main() -> Result<(), E> {
    Err(E {
        x: 42,
        name: "n".to_string(),
    })
}
"#;

const RESULT_ERR_INNER_FORMAT_DEBUG_SRC: &str = r#"
use std::fmt;
struct Inner;
impl fmt::Debug for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "INNER")
    }
}
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = format!("{:?}", Inner);
        eprintln!("composed={s}");
        write!(f, "outer[{s}]")
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

#[test]
fn run_mode_termination_report_unwind_drop_and_streaming() {
    // A panic in the payload's `Debug` body unwinds through the
    // native report, dropping the error as it goes: the drop
    // runs as the panic's cleanup (its own guest outcome -- an
    // exit -- replaces the panic; a second panic aborts), while
    // an abort or a `process::exit` in the body keeps the
    // no-drop behavior. And the report's formatter writes
    // stream to stderr as they happen: they interleave with the
    // body's own stderr writes in native order, and text
    // written before a panic or an exit stands.
    let dir = temp_dir("termination-report-unwind");
    let src = fixture(
        &dir,
        "err_fmt_panic_drop_exits.rs",
        RESULT_ERR_FMT_PANIC_DROP_EXITS_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(7), "drop exits during unwind cleanup");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: dropped\n"
    );
    let src = fixture(&dir, "err_fmt_panic_drop.rs", RESULT_ERR_FMT_PANIC_DROP_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "drop runs during unwind cleanup");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: dropped\nthread 'main' panicked: fmt exploded\n"
    );
    let src = fixture(
        &dir,
        "err_fmt_panic_drop_panics.rs",
        RESULT_ERR_FMT_PANIC_DROP_PANICS_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "second panic aborts");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: thread 'main' aborted: panic `drop exploded` raised in the \
         drop of the Termination report payload while unwinding: \
         panicking while unwinding aborts the run\n"
    );
    let src = fixture(&dir, "err_streaming_debug.rs", RESULT_ERR_STREAMING_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "streaming Debug::fmt");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: firstside effect\nlast\n"
    );
    let src = fixture(
        &dir,
        "err_partial_panic_debug.rs",
        RESULT_ERR_PARTIAL_PANIC_DEBUG_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "partial text stands on panic");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: partialthread 'main' panicked: fmt exploded\n"
    );
    let src = fixture(
        &dir,
        "err_partial_exit_debug.rs",
        RESULT_ERR_PARTIAL_EXIT_DEBUG_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(5), "partial text stands on exit");
    assert_eq!(String::from_utf8_lossy(&out.stderr), "Error: partial");
    let src = fixture(&dir, "err_derived_debug.rs", RESULT_ERR_DERIVED_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "derived Debug payload");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: E { x: 42, name: \"n\" }\n"
    );
    let src = fixture(
        &dir,
        "err_inner_format_debug.rs",
        RESULT_ERR_INNER_FORMAT_DEBUG_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "inner format! composes");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: composed=INNER\nouter[INNER]\n"
    );
}

const RESULT_ERR_FMT_ERROR_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Err(fmt::Error)
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_FMT_ERROR_DROP_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Err(fmt::Error)
    }
}
impl Drop for E {
    fn drop(&mut self) {
        eprintln!("dropped");
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_FMT_ERROR_PARTIAL_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "partial")?;
        Err(fmt::Error)
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_FMT_ERROR_DROP_EXITS_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Err(fmt::Error)
    }
}
impl Drop for E {
    fn drop(&mut self) {
        eprintln!("dropped");
        std::process::exit(7);
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_FMT_ERROR_DROP_PANICS_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Err(fmt::Error)
    }
}
impl Drop for E {
    fn drop(&mut self) {
        panic!("drop exploded");
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

#[test]
fn run_mode_termination_report_fmt_error_panics() {
    // A payload `Debug` body that returns `Err(fmt::Error)`
    // does not make the native report fall back or complete:
    // the report's `write_fmt` to stderr panics on the
    // formatting error (core's message), the panic unwinds
    // through the report dropping the payload as its cleanup
    // (a destructor that exits replaces the panic; a second
    // panic in the drop aborts), text the body streamed before
    // returning the error stands, and the run exits 101.
    let dir = temp_dir("termination-report-fmt-error");
    let src = fixture(&dir, "err_fmt_error.rs", RESULT_ERR_FMT_ERROR_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "fmt error panics the report");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: thread 'main' panicked: a formatting trait implementation \
         returned an error when the underlying stream did not\n"
    );
    let src = fixture(&dir, "err_fmt_error_drop.rs", RESULT_ERR_FMT_ERROR_DROP_SRC);
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(101),
        "drop runs during unwind cleanup"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: dropped\nthread 'main' panicked: a formatting trait \
         implementation returned an error when the underlying stream did not\n"
    );
    let src = fixture(
        &dir,
        "err_fmt_error_partial.rs",
        RESULT_ERR_FMT_ERROR_PARTIAL_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(101),
        "partial text stands on fmt error"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: partialthread 'main' panicked: a formatting trait \
         implementation returned an error when the underlying stream did not\n"
    );
    let src = fixture(
        &dir,
        "err_fmt_error_drop_exits.rs",
        RESULT_ERR_FMT_ERROR_DROP_EXITS_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(7),
        "drop exits during unwind cleanup"
    );
    assert_eq!(String::from_utf8_lossy(&out.stderr), "Error: dropped\n");
    let src = fixture(
        &dir,
        "err_fmt_error_drop_panics.rs",
        RESULT_ERR_FMT_ERROR_DROP_PANICS_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "second panic aborts");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: thread 'main' aborted: panic `drop exploded` raised in the \
         drop of the Termination report payload while unwinding: \
         panicking while unwinding aborts the run\n"
    );
}

const LIBTEST_SRC: &str = r#"
#[test]
fn alpha() { assert_eq!(1 + 1, 2); }
#[test]
#[ignore]
fn gamma() {}
#[test]
#[should_panic]
fn panics() { panic!("expected"); }
"#;

const LIBTEST_FAIL_SRC: &str = r#"
#[test]
fn good() { assert_eq!(2, 2); }
#[test]
fn bad() { assert_eq!(1, 2); }
"#;

#[test]
fn run_mode_libtest_harness_runs_tests() {
    let dir = temp_dir("libtest");
    let src = fixture(&dir, "harness.rs", LIBTEST_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "--test"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr} stdout={stdout}");
    assert!(stdout.contains("running 3 tests"), "stdout={stdout}");
    assert!(stdout.contains("test alpha ... ok"), "stdout={stdout}");
    assert!(
        stdout.contains("test panics - should panic ... ok"),
        "stdout={stdout}"
    );
    assert!(stdout.contains("test gamma ... ignored"), "stdout={stdout}");
    assert!(
        stdout.contains("test result: ok. 2 passed; 0 failed; 1 ignored"),
        "stdout={stdout}"
    );
    let src = fixture(&dir, "harness_fail.rs", LIBTEST_FAIL_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "--test"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(stdout.contains("test bad ... FAILED"), "stdout={stdout}");
    assert!(
        stdout.contains("test result: FAILED. 1 passed; 1 failed"),
        "stdout={stdout}"
    );
}


const WRAPPING_SRC: &str = r#"
fn main() {
    let a = 255u8.wrapping_add(1);
    assert_eq!(a, 0);
    let b = i64::MAX.wrapping_add(1);
    assert_eq!(b, i64::MIN);
    println!("ok");
}
"#;

#[test]
fn run_mode_wrapping_methods_wrap_in_a_checked_session() {
    // The plain arithmetic inside the `wrapping_*` method bodies
    // wraps; only the `Assert` after a checked operation may panic,
    // and these bodies contain none.
    run_ok("wrapping", WRAPPING_SRC, "ok\n");
}

const TRAIT_OVERFLOW_SRC: &str = r#"
use std::ops::*;
fn main() {
    assert_eq!(i8::neg(-0x80), -0x80);
    assert_eq!(u8::add(0xff, 1), 0_u8);
    println!("ok");
}
"#;

#[test]
fn run_mode_overflow_asserts_follow_the_session_setting() {
    let dir = temp_dir("overflow-assert");
    let src = fixture(&dir, "neg.rs", TRAIT_OVERFLOW_SRC);
    // The default session checks overflow: the trait bodies'
    // overflow asserts fire and the guest panics (101).
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "default session must panic");
    // `-C overflow-checks=off`: the same asserts are not evaluated
    // and the wrapped results stand.
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "-C", "--rustc-arg", "overflow-checks=off"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
    // `-C debug-assertions=no` implies overflow checks off as well.
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "-C", "--rustc-arg", "debug-assertions=no"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const LITERAL_CMP_SRC: &str = r#"
fn main() {
    let a: u8 = 0;
    let b: u8 = 255;
    assert!(a < b, "locals failed");
    assert!(0u8 < 255u8, "consts failed");
    println!("ok");
}
"#;

#[test]
fn run_mode_literal_operands_compare_under_their_own_type() {
    // A literal-vs-literal comparison reads each literal under its
    // own recorded type: `255u8` is unsigned 255, not -1.
    run_ok("literal-cmp", LITERAL_CMP_SRC, "ok\n");
}

const ADT_CONST_SRC: &str = r#"
#![feature(test)]
extern crate test;
use test::black_box as b;
fn main() {
    assert_eq!(b(1u128).checked_shl(b(127)), Some(1 << 127));
    println!("ok");
}
"#;

#[test]
fn run_mode_adt_constant_fields_use_their_field_types() {
    // The `Some(1 << 127)` constant's payload decodes under the
    // field's own `u128` type, not by a sign-bit guess.
    run_ok("adt-const", ADT_CONST_SRC, "ok\n");
}

const SHOULD_PANIC_MATCH_SRC: &str = r#"
#[test]
#[should_panic(expected = "hello")]
fn panics_with_hello() {
    panic!("hello world");
}
"#;

const SHOULD_PANIC_MISS_SRC: &str = r#"
#[test]
#[should_panic(expected = "bye")]
fn panics_with_hello() {
    panic!("hello world");
}
"#;

#[test]
fn run_mode_libtest_should_panic_expected_message() {
    let dir = temp_dir("libtest-should-panic");
    let src = fixture(&dir, "match.rs", SHOULD_PANIC_MATCH_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "--test"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout={stdout}");
    let src = fixture(&dir, "miss.rs", SHOULD_PANIC_MISS_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "--test"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(stdout.contains("FAILED"), "stdout={stdout}");
}

const LISTING_SRC: &str = r#"
#[test]
fn beta() {}
#[test]
fn alpha() {}
"#;

#[test]
fn run_mode_libtest_listing_and_nightly_only_options() {
    let dir = temp_dir("libtest-listing");
    let src = fixture(&dir, "listing.rs", LISTING_SRC);
    // `--list` prints the sorted test names and succeeds.
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "--test", "--guest-arg", "--list"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout={stdout}");
    assert!(stdout.contains("alpha: test"), "stdout={stdout}");
    assert!(stdout.contains("beta: test"), "stdout={stdout}");
    assert!(stdout.contains("2 tests, 0 benchmarks"), "stdout={stdout}");
    // `--format json` without `-Z unstable-options` is the nightly
    // error, exit 101.
    let out = run_driver_extra(
        &src,
        &[
            "--rustc-arg",
            "--test",
            "--guest-arg",
            "--format",
            "--guest-arg",
            "json",
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stderr={stderr}");
    assert!(
        stderr.contains("The \"json\" format is only accepted on the nightly compiler"),
        "stderr={stderr}"
    );
    // Any `-Z` option is rejected the same way.
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "--test", "--guest-arg", "-Zunstable-options"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stderr={stderr}");
    assert!(
        stderr.contains("the option `Z` is only accepted on the nightly compiler"),
        "stderr={stderr}"
    );}
const LIBTEST_SP_SRC: &str = r#"
#[test]
#[should_panic(expected = "hello")]
fn wrong_msg() { panic!("bye"); }
#[test]
#[should_panic]
fn no_panic() {}
#[test]
#[ignore = "later"]
fn skipped() {}
"#;

#[test]
fn run_mode_libtest_harness_output_fidelity() {
    let dir = temp_dir("libtest-fidelity");
    let src = fixture(&dir, "sp.rs", LIBTEST_SP_SRC);
    // `--list`: one line per test plus the count footer, and
    // nothing runs.
    let out = run_driver_extra(&src, &["--rustc-arg", "--test", "--guest-arg", "--list"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout={stdout}");
    assert_eq!(
        stdout, "no_panic: test\nskipped: test\nwrong_msg: test\n\n3 tests, 0 benchmarks\n",
        "stdout={stdout}"
    );
    // Should-panic verdicts: a panic whose message misses the
    // expected substring and a test that does not panic both
    // fail, with the harness's notes as their failure sections;
    // the ignored test's line carries its ignore message.
    let out = run_driver_extra(&src, &["--rustc-arg", "--test", "--guest-arg", "--nocapture"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(
        stdout.starts_with("\nrunning 3 tests\n"),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("test no_panic - should panic ... FAILED"),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("note: test did not panic as expected at "),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains(
            "note: panic did not contain expected string\n      panic message: \"bye\"\n expected substring: \"hello\""
        ),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("test skipped ... ignored, later"),
        "stdout={stdout}"
    );
    // Multiple positional filters select the union of matches.
    let out = run_driver_extra(
        &src,
        &[
            "--rustc-arg",
            "--test",
            "--guest-arg",
            "wrong_msg",
            "--guest-arg",
            "no_panic",
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("running 2 tests"), "stdout={stdout}");
    assert!(stdout.contains("1 filtered out"), "stdout={stdout}");

}

const LIBTEST_JSON_SRC: &str = r#"
#[test]
fn a() {
    println!("print from successful test");
}
#[test]
fn b() {
    assert!(false);
}
#[test]
#[should_panic]
fn c() {
    assert!(false);
}
#[test]
#[ignore = "msg"]
fn d() {
    assert!(false);
}
"#;

#[test]
fn run_mode_libtest_json_format_output() {
    // Lane IH: `--format json` together with `-Z
    // unstable-options` is the JSON Lines event stream (the
    // shape tests/run-make/libtest-json snapshots), for both
    // runs and `--list` discovery.
    let dir = temp_dir("libtest-json");
    let src = fixture(&dir, "f.rs", LIBTEST_JSON_SRC);
    let out = run_driver_extra(
        &src,
        &[
            "--rustc-arg",
            "--test",
            "--guest-arg",
            "-Zunstable-options",
            "--guest-arg",
            "--format=json",
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(
        stdout.contains("{ \"type\": \"suite\", \"event\": \"started\", \"test_count\": 4 }"),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("{ \"type\": \"test\", \"name\": \"a\", \"event\": \"ok\" }"),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("\"event\": \"failed\", \"stdout\": \"\\nthread 'b'"),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains(
            "{ \"type\": \"test\", \"name\": \"d\", \"event\": \"ignored\", \"message\": \"msg\" }"
        ),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains(
            "{ \"type\": \"suite\", \"event\": \"failed\", \"passed\": 2, \"failed\": 1, \
             \"ignored\": 1, \"measured\": 0, \"filtered_out\": 0"
        ),
        "stdout={stdout}"
    );
    // `--show-output` adds the captured output to passing
    // tests, including the report of an expected panic.
    let out = run_driver_extra(
        &src,
        &[
            "--rustc-arg",
            "--test",
            "--guest-arg",
            "-Zunstable-options",
            "--guest-arg",
            "--format=json",
            "--guest-arg",
            "--show-output",
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(
        stdout.contains(
            "\"name\": \"a\", \"event\": \"ok\", \"stdout\": \"print from successful test\\n\""
        ),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("\"name\": \"c\", \"event\": \"ok\", \"stdout\": \"\\nthread 'c'"),
        "stdout={stdout}"
    );
    // The discovery listing: descriptor spans ride along.
    let out = run_driver_extra(
        &src,
        &[
            "--rustc-arg",
            "--test",
            "--guest-arg",
            "--list",
            "--guest-arg",
            "--format=json",
            "--guest-arg",
            "-Zunstable-options",
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout={stdout}");
    assert!(
        stdout.contains("{ \"type\": \"suite\", \"event\": \"discovery\" }"),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains(
            "\"event\": \"discovered\", \"name\": \"d\", \"ignore\": true, \
             \"ignore_message\": \"msg\""
        ),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains(
            "{ \"type\": \"suite\", \"event\": \"completed\", \"tests\": 4, \
             \"benchmarks\": 0, \"total\": 4, \"ignored\": 1 }"
        ),
        "stdout={stdout}"
    );
}

#[test]
fn run_mode_libtest_ignored_modes() {
    // Lane IH: `--ignored` runs only the ignored tests (the
    // rest count as filtered out); `--include-ignored` runs
    // them with the rest.
    let dir = temp_dir("libtest-ignored");
    let src = fixture(&dir, "f.rs", LIBTEST_JSON_SRC);
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "--test", "--guest-arg", "--ignored"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(stdout.contains("running 1 test"), "stdout={stdout}");
    assert!(
        stdout.contains("0 passed; 1 failed; 0 ignored"),
        "stdout={stdout}"
    );
    assert!(stdout.contains("3 filtered out"), "stdout={stdout}");
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "--test", "--guest-arg", "--include-ignored"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(stdout.contains("running 4 tests"), "stdout={stdout}");
    assert!(
        stdout.contains("2 passed; 2 failed; 0 ignored"),
        "stdout={stdout}"
    );
}

#[test]
fn run_mode_libtest_invalid_rust_test_threads() {
    let dir = temp_dir("libtest-threads");
    let src = fixture(&dir, "listing.rs", LISTING_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .args(["--rustc-arg", "--test"])
        .env("RUST_TEST_THREADS", "foo")
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stderr={stderr}");
    assert!(
        stderr.contains("should be a positive integer"),
        "stderr={stderr}"
    );
}

const BOX_CLONE_SRC: &str = r#"
fn main() {
    let mut i = Box::new(1);
    let mut j = i.clone();
    *j = 3;
    println!("i={} j={}", *i, *j);
    let p1 = &*i as *const i32 as usize;
    let p2 = &*j as *const i32 as usize;
    println!("same_addr={}", p1 == p2);
    let mut a: Vec<Box<_>> = vec![Box::new(10)];
    let b = a.clone();
    *a[0] = 20;
    println!("a={} b={}", *a[0], *b[0]);
}
"#;

#[test]
fn run_mode_box_clone_owns_its_pointee() {
    // A cloned box (and the boxes inside a cloned vector) own fresh
    // storage: writing through one must not rewrite the other.
    run_ok(
        "box-clone",
        BOX_CLONE_SRC,
        "i=1 j=3\nsame_addr=false\na=20 b=10\n",
    );
}

const TIME_SRC: &str = r#"
fn main() {
    let a = std::time::Instant::now();
    let b = std::time::Instant::now();
    assert!(b >= a);
    let s = std::time::SystemTime::now();
    let d = s.duration_since(std::time::UNIX_EPOCH).unwrap();
    assert!(d.as_secs() > 1_600_000_000);
    println!("time ok");
}
"#;

const TIME_OVERFLOW_SRC: &str = r#"
fn main() {
    let t = std::time::Instant::now();
    let _ = t + std::time::Duration::MAX;
}
"#;

#[test]
fn run_mode_time_now_and_overflow() {
    let dir = temp_dir("time");
    let src = fixture(&dir, "time.rs", TIME_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "time ok\n");
    let src = fixture(&dir, "time_overflow.rs", TIME_OVERFLOW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stderr={stderr}");
    assert!(
        stderr.contains("overflow when adding duration to instant"),
        "stderr={stderr}"
    );
}

const PANIC_HOOK_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};
static FIRED: AtomicUsize = AtomicUsize::new(0);
fn main() {
    std::panic::set_hook(Box::new(|info| {
        FIRED.fetch_add(1, Ordering::SeqCst);
        println!("hook line {}", info.location().unwrap().line());
    }));
    let r = std::panic::catch_unwind(|| panic!("foobar"));
    assert!(r.is_err());
    println!("fired {}", FIRED.load(Ordering::SeqCst));
}
"#;

#[test]
fn run_mode_panic_hook_fires_with_location() {
    let dir = temp_dir("panic-hook");
    let src = fixture(&dir, "hook.rs", PANIC_HOOK_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The `panic!` is on line 9 of the fixture.
    assert_eq!(stdout, "hook line 9\nfired 1\n", "stdout={stdout}");
}
// ----- reference deref identity and DST metadata (Lane GB) -----------------

const COROUTINE_RESUME_ARGS_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn main() {
    let mut gen = #[coroutine] |x: i32| {
        let y = yield x + 1;
        y * 2
    };
    let mut pinned = Pin::new(&mut gen);
    match pinned.as_mut().resume(41) {
        CoroutineState::Yielded(v) => println!("yielded {v}"),
        CoroutineState::Complete(v) => println!("complete {v}"),
    }
    match pinned.as_mut().resume(10) {
        CoroutineState::Yielded(v) => println!("yielded {v}"),
        CoroutineState::Complete(v) => println!("complete {v}"),
    }
}
"#;

#[test]
fn run_mode_coroutine_resume_args_through_pin() {
    // `Pin::as_mut` dereferences the `&mut Coroutine` the pin
    // wraps (`DerefMut` for a mutable reference): the result is
    // the inner reference itself, and the coroutine state machine
    // advances across resumes with the resume argument bound.
    let dir = temp_dir("coroutine-resume-args");
    let src = fixture(&dir, "coroutine_resume_args.rs", COROUTINE_RESUME_ARGS_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "yielded 42\ncomplete 20\n", "stdout={stdout}");
}

const DEREF_ON_REFERENCE_SRC: &str = r#"
use std::ops::{Deref, DerefMut};

fn bump(mut r: &mut i32) {
    *r.deref_mut() += 1;
}

fn read(r: &i32) -> i32 {
    *r.deref()
}

fn via_generic<P: DerefMut<Target = i32>>(mut p: P) {
    *p += 10;
}

fn main() {
    let mut x = 1;
    bump(&mut x);
    println!("x {x}");
    println!("read {}", read(&x));
    via_generic(&mut x);
    println!("x {x}");
}
"#;

#[test]
fn run_mode_deref_and_deref_mut_on_references() {
    // `<&mut T as DerefMut>::deref_mut` returns the inner
    // reference itself (writing through it lands in the original
    // place), and `<&T as Deref>::deref` likewise; generic code
    // over `P: DerefMut` instantiated with a reference reaches
    // the same impls.
    let dir = temp_dir("deref-on-reference");
    let src = fixture(&dir, "deref_on_reference.rs", DEREF_ON_REFERENCE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "x 2\nread 2\nx 12\n", "stdout={stdout}");
}

const DST_STRUCT_SRC: &str = r#"
struct Fat<T: ?Sized> {
    f1: isize,
    f2: &'static str,
    ptr: T,
}

fn foo(x: &Fat<[isize]>) {
    println!("f1 {} f2 {} len {} second {}", x.f1, x.f2, x.ptr.len(), x.ptr[1]);
}

fn main() {
    let f1 = Fat { f1: 5, f2: "some str", ptr: [1, 2, 3] };
    foo(&f1);
    let f3: &Fat<[isize]> = &f1;
    foo(f3);
    let b: Box<[isize]> = Box::new([4, 5, 6, 7]);
    println!("box len {}", b.len());
}
"#;

#[test]
fn run_mode_dst_struct_trailing_slice_metadata() {
    // Unsizing `&Fat<[isize; 3]>` to `&Fat<[isize]>` computes the
    // metadata of the struct value from its trailing field: the
    // slice length, not the unit metadata of a sized value.
    let dir = temp_dir("dst-struct");
    let src = fixture(&dir, "dst_struct.rs", DST_STRUCT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "f1 5 f2 some str len 3 second 2\nf1 5 f2 some str len 3 second 2\nbox len 4\n",
        "stdout={stdout}"
    );
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

// ----- Location::file and the fat-pointer transmutes (Lane GE) --------------

const LOCATION_FILE_SRC: &str = r#"
fn main() {
    let loc = std::panic::Location::caller();
    println!("line {}", loc.line());
    println!("file-ok {}", loc.file() == file!());
}
"#;

#[test]
fn run_mode_location_file_reports_the_source_path() {
    // `Location::file` transmutes the stored `NonNull<str>` to a
    // `*const str`: a fat pointer read out of its transparent
    // wrapper. The conversion generalises wrapper transmutes to
    // fat payloads, so the whole accessor now executes.
    let dir = temp_dir("location-file");
    let src = fixture(&dir, "locfile.rs", LOCATION_FILE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // `Location::caller()` is evaluated on line 3 of the fixture.
    assert_eq!(stdout, "line 3\nfile-ok true\n", "stdout={stdout}");
}

const FAT_PTR_TRANSMUTE_SRC: &str = r#"
fn main() {
    let s = "hello";
    let nn: std::ptr::NonNull<str> = unsafe { std::mem::transmute(s as *const str) };
    let back: &str = unsafe { std::mem::transmute(nn) };
    println!("back {back}");
    let arr = [1u8, 2, 3];
    let nn2: std::ptr::NonNull<[u8]> = unsafe { std::mem::transmute(&arr[..] as *const [u8]) };
    let sl: &[u8] = unsafe { std::mem::transmute(nn2) };
    println!("len {} sum {}", sl.len(), sl[0] + sl[1] + sl[2]);
}
"#;

#[test]
fn run_mode_fat_pointer_wrapper_transmutes_round_trip() {
    let dir = temp_dir("fat-ptr-transmute");
    let src = fixture(&dir, "fatptr.rs", FAT_PTR_TRANSMUTE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "back hello\nlen 3 sum 6\n", "stdout={stdout}");
}

const ENUM_WRAP_TRANSMUTE_SRC: &str = r#"
#[repr(isize)]
#[derive(Clone, Copy)]
enum Align { A1 = 1, A2 = 2, A4 = 4, A8 = 8 }
#[repr(transparent)]
struct Wrap(Align);
fn main() {
    let n: usize = unsafe { std::mem::transmute(Wrap(Align::A8)) };
    println!("n {n}");
    let m: usize = unsafe { std::mem::transmute(Wrap(Align::A2)) };
    println!("m {m}");
}
"#;

#[test]
fn run_mode_transparent_wrapper_over_fieldless_enum_transmutes_to_int() {
    // The `mem::Alignment` shape: a one-field transparent struct
    // over a fieldless enum transmutes to the integer as the
    // field's discriminant. (RawVec's allocation path performs
    // exactly this transmute on the `Vec` collect path.)
    let dir = temp_dir("enum-wrap-transmute");
    let src = fixture(&dir, "alignwrap.rs", ENUM_WRAP_TRANSMUTE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "n 8\nm 2\n", "stdout={stdout}");
}

// ----- Vec collect / size_hint / Rev (Lane GE) -------------------------------

const COLLECT_SRC: &str = r#"
fn main() {
    let v: Vec<i32> = (0..3).collect();
    println!("n {} s {}", v.len(), v[0] + v[1] + v[2]);
    let w: Vec<i32> = (0..3).rev().collect();
    println!("rev {} {}", w[0], w[2]);
    let e: Vec<i32> = (5..5).collect();
    println!("empty {}", e.len());
    let mut x: Vec<i32> = Vec::new();
    x.extend(10..12);
    println!("ext {} {}", x.len(), x[0] + x[1]);
}
"#;

#[test]
fn run_mode_collect_builds_vec_from_iterators() {
    // `Iterator::collect::<Vec<_>>` reaches
    // `SpecFromIterNested::from_iter`, whose collected body has
    // `Vec::with_capacity` inlined and drives `RawVecInner`
    // directly; the entry point is modelled by stepping the
    // iterator (see `Builtin::VecFromIterNested`), and
    // `Vec::extend` reaches the modelled `extend_trusted`.
    let dir = temp_dir("collect");
    let src = fixture(&dir, "collect.rs", COLLECT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "n 3 s 3\nrev 2 0\nempty 0\next 2 21\n",
        "stdout={stdout}"
    );
}

const ADAPTER_INTO_ITER_COLLECT_SRC: &str = r#"
fn main() {
    let v = vec![5, 12, 7, 20];
    let out: Vec<i32> = v.into_iter().map(|x| x + 1).collect();
    println!("map {out:?}");
    let e: Vec<i32> = Vec::new();
    let empty: Vec<i32> = e.into_iter().map(|x| x + 1).collect();
    println!("empty {}", empty.len());
    let n = vec![1, 2, 3];
    let nested: Vec<i32> = n.into_iter().map(|x| x * 2).map(|x| x + 10).collect();
    println!("nested {nested:?}");
    let t = vec![10, 20, 30, 40, 50];
    let taken: Vec<i32> = t.into_iter().take(2).collect();
    println!("take {taken:?}");
    let r = vec![1, 2, 3, 4];
    let rev: Vec<i32> = r.into_iter().rev().collect();
    println!("rev {rev:?}");
    let w = vec!["a".to_string(), "bb".to_string()];
    let strs: Vec<String> = w
        .into_iter()
        .map(|mut s| {
            s.push_str("!");
            s
        })
        .collect();
    println!("strs {strs:?}");
}
"#;

#[test]
fn run_mode_adapter_over_vec_into_iter_collect() {
    // Adapters over `vec::IntoIter` collect through the
    // specialized `SpecFromIter::from_iter`, whose body calls
    // `std::vec::in_place_collect::from_iter_in_place` through a
    // function-pointer constant — a shape the name-based
    // collection never sees, so the raw-allocation body was
    // collected by the const-fn demand pass and ran into the
    // body-less leaf specializations (`SourceIter::as_inner`
    // and friends). The call is served by the
    // `VecFromIterNested` builtin, the entry point's own model
    // (drain the iterator into a fresh `Vec`).
    let dir = temp_dir("adapter-into-iter-collect");
    let src = fixture(&dir, "adaptercollect.rs", ADAPTER_INTO_ITER_COLLECT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "map [6, 13, 8, 21]\nempty 0\nnested [12, 14, 16]\ntake [10, 20]\nrev [4, 3, 2, 1]\nstrs [\"a!\", \"bb!\"]\n",
        "stdout={stdout}"
    );
}

const FILTER_ZIP_INTO_ITER_COLLECT_SRC: &str = r#"
fn main() {
    let v = vec![1, 2, 3, 4, 5, 6];
    let evens: Vec<i32> = v.into_iter().filter(|x| x % 2 == 0).collect();
    println!("filter {evens:?}");
    let a = vec![1, 2, 3];
    let b = vec![10, 20, 30, 40];
    let pairs: Vec<(i32, i32)> = a.into_iter().zip(b.into_iter()).collect();
    println!("zip {pairs:?}");
    let e: Vec<i32> = Vec::new();
    let no_pairs: Vec<(i32, i32)> = e.into_iter().zip(vec![1]).collect();
    println!("zip-empty {}", no_pairs.len());
    let w = vec![String::from("a"), String::from("bb")];
    let cloned: Vec<String> = w.iter().cloned().collect();
    println!("cloned {cloned:?}");
}
"#;

#[test]
fn run_mode_filter_zip_cloned_over_vec_into_iter_collect() {
    // The same specialization path as the Map case, for the
    // adapters the iterator framework models: `Filter` over
    // `vec::IntoIter` delegates to the in-place entry, and
    // `Zip` (no single source allocation to reuse) delegates
    // to the nested entry's collected impl body, which drives
    // `Vec::with_capacity` and the raw-buffer machinery. Both
    // calls are served by the `VecFromIterNested` builtin.
    let dir = temp_dir("filter-zip-into-iter-collect");
    let src = fixture(
        &dir,
        "filterzipcollect.rs",
        FILTER_ZIP_INTO_ITER_COLLECT_SRC,
    );
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "filter [2, 4, 6]\nzip [(1, 10), (2, 20), (3, 30)]\nzip-empty 0\ncloned [\"a\", \"bb\"]\n",
        "stdout={stdout}"
    );
}

const ARGS_COLLECT_SRC: &str = r#"
fn main() {
    let fwd: Vec<String> = std::env::args().collect();
    println!("fwd {}", fwd.len());
    println!("fwd1 {}", fwd[1]);
    let rev: Vec<String> = std::env::args().rev().collect();
    println!("rev0 {}", rev[0]);
    println!("rev-last {}", rev[rev.len() - 1] == fwd[0]);
}
"#;

#[test]
fn run_mode_env_args_collect_and_reverse() {
    // `args().collect()` calls the iterator's `size_hint` first
    // (the modelled `Iterator::size_hint` answers for the env
    // iterators, whose real bodies read the real `vec::IntoIter`
    // layout off the modelled state). `args().rev()` is the
    // `Rev` adapter: its `next` is the inner iterator's back
    // step.
    let dir = temp_dir("args-collect");
    let src = fixture(&dir, "argscollect.rs", ARGS_COLLECT_SRC);
    let out = run_driver_extra(&src, &["--guest-arg", "one", "--guest-arg", "two"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "fwd 3\nfwd1 one\nrev0 two\nrev-last true\n",
        "stdout={stdout}"
    );
}

const SIZE_HINT_SRC: &str = r#"
fn main() {
    let (l, u) = (0..10).size_hint();
    println!("range {l} {}", u.unwrap());
    let (l, u) = (2..=5).size_hint();
    println!("incl {l} {}", u.unwrap());
    let (l, u) = "aé€".chars().size_hint();
    println!("chars {l} {}", u.unwrap());
    let (l, u) = "aé€".bytes().size_hint();
    println!("bytes {l} {}", u.unwrap());
    let (l, u) = ('a'..'z').size_hint();
    println!("charrange {l} {}", u.unwrap());
    let (l, u) = std::env::args().size_hint();
    println!("args {l} {}", u.unwrap());
    let (l, u) = (0..=u64::MAX).size_hint();
    println!("big {l} {}", u.is_none());
}
"#;

#[test]
fn run_mode_size_hint_reports_std_values() {
    // The exact pairs the standard library reports (measured
    // against the pinned toolchain): exact counts for ranges and
    // the owned iterators, the byte-count bounds for `Chars`
    // (lower = remaining bytes / 4 rounded up, upper = remaining
    // bytes), the surrogate gap removed for char ranges, and
    // `(usize::MAX, None)` for a count that overflows `usize`.
    let dir = temp_dir("size-hint");
    let src = fixture(&dir, "sizehint.rs", SIZE_HINT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "range 10 10\nincl 4 4\nchars 2 6\nbytes 6 6\ncharrange 25 25\nargs 1 1\nbig 18446744073709551615 true\n",
        "stdout={stdout}"
    );
}

const STR_ITER_WRAPPER_SRC: &str = r#"
fn main() {
    let b = "aé€b".bytes();
    println!("len {}", b.len());
    let mut b2 = "aé€b".bytes();
    b2.next();
    b2.next_back();
    println!("len {}", b2.len());
    let mut it = "aé€b".chars();
    println!("{:?}", it.next());
    println!("{:?}", it.next_back());
    println!("{:?}", it.next());
    println!("{:?}", it.next_back());
    println!("{:?}", it.next());
    let cs: Vec<char> = "aé€b".chars().rev().collect();
    println!("{cs:?}");
}
"#;

#[test]
fn run_mode_str_iter_states_use_the_real_wrapper_shape() {
    // The `str::Chars` / `str::Bytes` states are the structs'
    // real shapes (Chars { slice::Iter }, Bytes(Copied<slice::
    // Iter>)): `ExactSizeIterator::len` for `Bytes` is a
    // collected core body that reads that layout off the state,
    // and it must agree with the modelled front/back steps at
    // every point.
    let dir = temp_dir("str-iter-wrapper");
    let src = fixture(&dir, "striterwrapper.rs", STR_ITER_WRAPPER_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "len 7\nlen 5\nSome('a')\nSome('b')\nSome('é')\nSome('€')\nNone\n['b', '€', 'é', 'a']\n",
        "stdout={stdout}"
    );
}

const SIZE_HINT_DEFAULT_SRC: &str = r#"
struct Thing<'a>(&'a str);
impl<'a> Iterator for Thing<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<&'a str> {
        None
    }
}
fn main() {
    let n = Thing("test").fuse().filter(|_| true).count();
    println!("count {n}");
    let (l, u) = Thing("test").size_hint();
    println!("hint {l} {}", u.is_none());
    let (l, u) = Thing("test").fuse().size_hint();
    println!("fused {l} {}", u.is_none());
}
"#;

#[test]
fn run_mode_size_hint_unmodelled_iterator_gets_trait_default() {
    // Regression (tests/ui/inference/issue-36053.rs): a user
    // iterator that does not override `size_hint`, reached
    // through adapters (`Fuse`, `Filter::count`). Registering
    // the `Iterator::size_hint` builtin removes the trait's
    // default body from collection, so the builtin itself
    // answers unmodelled iterators with that default,
    // `(0, None)` — the exact native answer here, and a
    // contract-valid one for every iterator.
    let dir = temp_dir("size-hint-default");
    let src = fixture(&dir, "sizehintdefault.rs", SIZE_HINT_DEFAULT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "count 0\nhint 0 true\nfused 0 true\n",
        "stdout={stdout}"
    );
}

const ZST_ITER_LEN_SRC: &str = r#"
use std::iter::ExactSizeIterator;
fn main() {
    let a = [(), (), (), (), ()];
    let mut it = a.iter();
    println!("len {}", it.len());
    println!("slice {}", it.as_slice().len());
    it.next();
    println!("len {}", it.len());
    it.next_back();
    println!("len {}", it.len());
    println!("slice {}", it.as_slice().len());
    let e: [(); 0] = [];
    println!("empty {}", e.iter().len());
    let v = vec![(), (), ()];
    println!("vec {}", v.iter().len());
}
"#;

#[test]
fn run_mode_zst_slice_iter_len_and_as_slice() {
    // Issue #346: for a zero-sized element, core's `len()`
    // body reads the remaining count out of the end
    // pointer's own bits, and `as_slice` builds on the same
    // reading; over the modelled snapshot state that word is
    // the snapshot's address, not a count. The modelled
    // `ExactSizeIterator::len` / `as_slice` read the unified
    // bounds the modelled steps maintain instead, before and
    // after steps from both ends.
    let dir = temp_dir("zst-iter-len");
    let src = fixture(&dir, "zstiterlen.rs", ZST_ITER_LEN_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "len 5\nslice 5\nlen 4\nlen 3\nslice 3\nempty 0\nvec 3\n",
        "stdout={stdout}"
    );
}

const ITER_MUT_LEN_SRC: &str = r#"
use std::iter::ExactSizeIterator;
#[derive(Debug)]
struct Z;
fn main() {
    let mut a = [Z, Z, Z, Z];
    let mut m = a.iter_mut();
    println!("mlen {}", m.len());
    m.next();
    println!("mlen {}", m.len());
    m.next_back();
    println!("mlen {}", m.len());
    let mut b = [1, 2, 3, 4];
    let mut n = b.iter_mut();
    n.next();
    println!("rest {:?}", n.as_slice());
    println!("nlen {}", n.len());
    let rest = n.into_slice();
    rest[0] = 20;
    println!("b {:?}", b);
}
"#;

#[test]
fn run_mode_slice_iter_mut_len_after_steps() {
    // Issue #346, the stepped-state half of the same defect:
    // an `IterMut` state built by core's own constructor
    // pairs a positioned front pointer with an end pointer
    // holding the terminal index, so the collected `len()`
    // body — which reads the end pointer's address — stays
    // at the original length as the front steps. The
    // modelled `len` reads the current bounds; `as_slice` /
    // `into_slice` build the remaining slice from the same
    // reading.
    let dir = temp_dir("iter-mut-len");
    let src = fixture(&dir, "itermutlen.rs", ITER_MUT_LEN_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "mlen 4\nmlen 3\nmlen 2\nrest [2, 3, 4]\nnlen 3\nb [1, 20, 3, 4]\n",
        "stdout={stdout}"
    );
}

const EXACT_SIZE_LEN_OTHERS_SRC: &str = r#"
use std::iter::ExactSizeIterator;
use std::collections::BTreeSet;
struct Count(u32, u32);
impl Iterator for Count {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.0 < self.1 { self.0 += 1; Some(self.0) } else { None }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = (self.1 - self.0) as usize;
        (r, Some(r))
    }
}
impl ExactSizeIterator for Count {}
fn main() {
    println!("range {}", (3..7).len());
    let v = vec![9, 8, 7, 6];
    let mut vi = v.into_iter();
    println!("into0 {}", vi.len());
    vi.next();
    println!("into1 {}", vi.len());
    let mut cu = Count(0, 4);
    println!("user0 {}", cu.len());
    cu.next();
    println!("user1 {}", cu.len());
    let mut set = BTreeSet::new();
    set.insert(5);
    set.insert(6);
    set.insert(7);
    let mut bi = set.iter();
    println!("btree0 {}", bi.len());
    bi.next();
    println!("btree1 {}", bi.len());
}
"#;

#[test]
fn run_mode_exact_size_len_modelled_family_and_bodies() {
    // The modelled `len` serves only the slice-state family
    // (here: the B-tree iterator sharing that state); every
    // other type keeps its collected body — `Range`'s and
    // `vec::IntoIter`'s own overrides, and the trait default
    // (delegating to `size_hint`) for a user iterator.
    // Registering the builtin must not divert any of them.
    let dir = temp_dir("exact-size-len-others");
    let src = fixture(&dir, "exactlenothers.rs", EXACT_SIZE_LEN_OTHERS_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "range 4\ninto0 4\ninto1 3\nuser0 4\nuser1 3\nbtree0 3\nbtree1 2\n",
        "stdout={stdout}"
    );
}

const EXACT_SIZE_LEN_REF_WRAPPER_SRC: &str = r#"
use std::iter::ExactSizeIterator;
fn len_of<I: ExactSizeIterator>(it: I) -> usize { it.len() }
fn main() {
    let a = [10, 20, 30, 40];
    let mut it = a.iter();
    let r = &mut it;
    println!("mut0 {}", ExactSizeIterator::len(&r));
    r.next();
    r.next();
    println!("mut1 {}", ExactSizeIterator::len(&r));
    let mut it2 = a.iter();
    let mut r2 = &mut it2;
    let rr = &mut r2;
    println!("dbl0 {}", ExactSizeIterator::len(&rr));
    let mut it3 = a.iter();
    println!("gen0 {}", len_of(&mut it3));
    it3.next();
    println!("gen1 {}", len_of(&mut it3));
    let z = [(), (), (), (), ()];
    let mut zit = z.iter();
    let zr = &mut zit;
    println!("zst0 {}", ExactSizeIterator::len(&zr));
    zr.next();
    println!("zst1 {}", ExactSizeIterator::len(&zr));
    let mut b = [1, 2, 3];
    let mut mit = b.iter_mut();
    let mr = &mut mit;
    println!("itermut0 {}", ExactSizeIterator::len(&mr));
    mr.next();
    println!("itermut1 {}", ExactSizeIterator::len(&mr));
    let mut it4 = a.iter();
    let br = it4.by_ref();
    println!("byref0 {}", br.len());
    println!("byref1 {}", it4.len());
}
"#;

#[test]
fn run_mode_exact_size_len_reference_wrapper_forwards() {
    // A reference-wrapper `Self` (`&mut I`, the one reference
    // form implementing the trait) is not the modelled
    // builtin's to serve: core's blanket body for `&mut I` is
    // collected and forwards to the inner call, which the
    // builtin then serves with the iterator type itself.
    // Peeling the reference in the modelled predicate
    // preempted that body, and the builtin — dereferencing
    // its receiver only once — found a reference where the
    // iterator state belongs ("expected ADT value, found
    // reference"). The ZST case pins that the forwarded
    // inner call still reaches the modelled builtin.
    let dir = temp_dir("exact-size-len-ref-wrapper");
    let src = fixture(&dir, "exactlenref.rs", EXACT_SIZE_LEN_REF_WRAPPER_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "mut0 4\nmut1 2\ndbl0 4\ngen0 4\ngen1 3\nzst0 5\nzst1 4\nitermut0 3\nitermut1 2\nbyref0 4\nbyref1 4\n",
        "stdout={stdout}"
    );
}

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

const CATCH_UNWIND_O3_DATA_SRC: &str = r#"
#[inline(never)]
fn boom() { std::panic::resume_unwind(Box::new(7i32)); }
fn main() {
    let r = std::panic::catch_unwind(|| 40 + 2);
    let ok1 = matches!(r, Ok(42));
    let r2 = std::panic::catch_unwind(|| -> i32 { boom(); 0 });
    let ok2 = matches!(r2, Err(_));
    std::process::exit(if ok1 && ok2 { 0 } else { 3 });
}
"#;

/// At opt-level 3 `catch_unwind` is inlined into `main`, and its
/// `Data` union local is materialized as an assignment of the
/// uninitialized constant, filled field by field through a raw
/// pointer by `do_call` / `do_catch`, with the zero-sized closure
/// field never stored at all. The run must still complete: the
/// uninitialized assignment is not a type mismatch, the field
/// writes partially initialize the union, and the closure field's
/// single value is synthesized on read.
#[test]
fn run_mode_catch_unwind_o3_data_union() {
    let dir = temp_dir("catch-o3-data");
    let src = fixture(&dir, "case.rs", CATCH_UNWIND_O3_DATA_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "O3 catch_unwind Data flow must run to completion; stderr={stderr}"
    );
}

// ----- Optimized-build transmutes in inlined std shapes (Lane JJ) -----
// Expected outputs are the native results of the same sources.

const FN_PTR_ERASED_SIGNATURE_SRC: &str = r#"
fn f(_x: &u64, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Ok(())
}
fn main() {
    let p: fn(&u64, &mut std::fmt::Formatter<'_>) -> std::fmt::Result = f;
    let _q: unsafe fn(
        std::ptr::NonNull<()>,
        &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result = unsafe { std::mem::transmute(p) };
}
"#;

/// The transmute an inlined `fmt::Argument` construction performs
/// on the display function pointer: a function pointer viewed
/// through an erased signature (safe fn -> unsafe fn over an
/// erased argument). A function pointer's value is the function
/// identity, so the cast carries the value over.
#[test]
fn run_mode_o2_transmute_fn_ptr_erased_signature() {
    let dir = temp_dir("fnptr-erased");
    let src = fixture(&dir, "case.rs", FN_PTR_ERASED_SIGNATURE_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "fn-pointer erased-signature transmute must run; stderr={stderr}"
    );
}

const ARRAY_BY_VALUE_SUM_SRC: &str = r#"
fn main() {
    let a = [7u8; 64];
    let mut s: u64 = 0;
    for x in a {
        s += x as u64;
    }
    assert_eq!(s, 448);
}
"#;

/// At opt-level 2 the by-value array iterator is inlined, and
/// its construction transmutes the backing array
/// `[u8; 64]` -> `[MaybeUninit<u8>; 64]`. Each element takes
/// the wrapper's value representation, and the iterated sum
/// must come out exact.
#[test]
fn run_mode_o2_array_into_iter_sum() {
    let dir = temp_dir("array-into-iter");
    let src = fixture(&dir, "case.rs", ARRAY_BY_VALUE_SUM_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "by-value array iteration under -O must sum exactly; stderr={stderr}"
    );
}

const MAYBE_UNINIT_ARRAY_UNWRAP_SRC: &str = r#"
use std::mem::{transmute, MaybeUninit};
fn main() {
    let a: [MaybeUninit<u8>; 4] = [
        MaybeUninit::new(3),
        MaybeUninit::new(4),
        MaybeUninit::new(5),
        MaybeUninit::new(6),
    ];
    let b: [u8; 4] = unsafe { transmute(a) };
    let mut s: u32 = 0;
    let mut j = 0usize;
    while j < 4 {
        s += b[j] as u32;
        j += 1;
    }
    assert_eq!(s, 18);
}
"#;

/// The reverse element-wrapper crossing: an array of
/// `MaybeUninit` values transmuted back to the plain element
/// array unwraps each element to the payload it holds.
#[test]
fn run_mode_o2_transmute_maybe_uninit_array_unwrap() {
    let dir = temp_dir("mu-array-unwrap");
    let src = fixture(&dir, "case.rs", MAYBE_UNINIT_ARRAY_UNWRAP_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "MaybeUninit array unwrap transmute must run; stderr={stderr}"
    );
}

const UNSAFE_CELL_ARRAY_ROUND_TRIP_SRC: &str = r#"
use std::cell::UnsafeCell;
use std::mem::transmute;
fn main() {
    let a: [u8; 4] = [10, 20, 30, 40];
    let b: [UnsafeCell<u8>; 4] = unsafe { transmute(a) };
    let c: [u8; 4] = unsafe { transmute(b) };
    let mut s: u32 = 0;
    let mut j = 0usize;
    while j < 4 {
        s += c[j] as u32;
        j += 1;
    }
    assert_eq!(s, 100);
}
"#;

/// The `UnsafeCell` twin of the element-wrapper crossing, in
/// both directions in one program.
#[test]
fn run_mode_o2_transmute_unsafe_cell_array_round_trip() {
    let dir = temp_dir("uc-array-roundtrip");
    let src = fixture(&dir, "case.rs", UNSAFE_CELL_ARRAY_ROUND_TRIP_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "UnsafeCell array round-trip transmute must run; stderr={stderr}"
    );
}

// ----- InlineAsm: the evaluable register class (Lane GI) -----
// Expected outputs are the native results of the same sources.

const ASM_MOV_ADD_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let x: u32 = 40;
    let mut y: u32 = 0;
    unsafe {
        asm!("mov {0:e}, {1:e}", "add {0:e}, 2", out(reg) y, in(reg) x);
    }
    let mut z: u64 = 5;
    unsafe {
        asm!("add {}, 3", inout(reg) z);
    }
    println!("y={y} z={z}");
}
"#;

#[test]
fn run_mode_inline_asm_mov_add_inout() {
    run_ok("asm-mov-add", ASM_MOV_ADD_SRC, "y=42 z=8\n");
}

const ASM_SHIFT_LOGIC_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let mut a: i32 = -16;
    let mut b: u32 = 3;
    let mut c: u32 = 0b1100;
    unsafe {
        asm!("sar {0:e}, 2", inout(reg) a);
        asm!("shl {}, 4", inout(reg) b);
        asm!("and {}, 0b1010", inout(reg) c);
        asm!("or {}, 0b0101", inout(reg) c);
        asm!("xor {}, {}", inout(reg) c, in(reg) 0b1111u32);
    }
    println!("{a} {b} {c}");
}
"#;

#[test]
fn run_mode_inline_asm_shifts_and_logic() {
    run_ok("asm-shift-logic", ASM_SHIFT_LOGIC_SRC, "-4 48 2\n");
}

const ASM_SUBREG_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let mut x: u16 = 0x1234;
    unsafe {
        asm!("mov {0:l}, 0xAB", inout(reg) x);
        asm!("mov {0:x}, 0xCD", inout(reg) x);
    }
    println!("{x}");
}
"#;

#[test]
fn run_mode_inline_asm_subregister_modifiers() {
    run_ok("asm-subreg", ASM_SUBREG_SRC, "205\n");
}

const ASM_ZERO_EXTEND_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let mut x: u64 = u64::MAX;
    unsafe {
        asm!("mov {0:e}, 1", inout(reg) x);
    }
    println!("{x}");
}
"#;

#[test]
fn run_mode_inline_asm_32bit_write_zero_extends() {
    run_ok("asm-zero-extend", ASM_ZERO_EXTEND_SRC, "1\n");
}

const ASM_CONST_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let y: u32;
    let mut z: u32 = 10;
    unsafe {
        asm!("mov {}, {}", out(reg) y, const 7);
        asm!("add {}, {}", inout(reg) z, const 5);
    }
    println!("{y} {z}");
}
"#;

#[test]
fn run_mode_inline_asm_const_operands() {
    run_ok("asm-const", ASM_CONST_SRC, "7 15\n");
}

const ASM_UNARY_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let mut a: i32 = 5;
    let mut b: u8 = 0x0f;
    let mut c: u32 = 41;
    unsafe {
        asm!("neg {}", inout(reg) a);
        asm!("not {}", inout(reg_byte) b);
        asm!("inc {}", inout(reg) c);
        asm!("dec {}", inout(reg) c);
        asm!("dec {}", inout(reg) c);
    }
    println!("{a} {b} {c}");
}
"#;

#[test]
fn run_mode_inline_asm_unary_ops() {
    run_ok("asm-unary", ASM_UNARY_SRC, "-5 240 40\n");
}

const ASM_NAKED_SRC: &str = r#"
use std::arch::naked_asm;

#[unsafe(naked)]
pub extern "C" fn naked_ret() {
    naked_asm!("ret");
}

fn main() {
    naked_ret();
    println!("back");
}
"#;

#[test]
fn run_mode_naked_fn_ret() {
    run_ok("asm-naked-ret", ASM_NAKED_SRC, "back\n");
}

const ASM_NAKED_ARGS_SRC: &str = r#"
use std::arch::naked_asm;

#[unsafe(naked)]
pub extern "sysv64" fn identity(x: u64) -> u64 {
    naked_asm!("mov rax, rdi", "ret");
}

#[unsafe(naked)]
pub extern "sysv64" fn add2(a: u64, b: u64) -> u64 {
    naked_asm!("mov rax, rdi", "add rax, rsi", "ret");
}

#[unsafe(naked)]
pub extern "win64" fn identity_win(x: u64) -> u64 {
    naked_asm!("mov rax, rcx", "ret");
}

#[unsafe(naked)]
pub extern "C" fn float_bits(x: f64) -> u64 {
    naked_asm!("movq rax, xmm0", "ret");
}

#[unsafe(naked)]
pub extern "C" fn sink(x: u64) {
    naked_asm!("ret");
}

fn main() {
    sink(9);
    println!(
        "{} {} {} {}",
        identity(42),
        add2(40, 2),
        identity_win(42),
        float_bits(1.5)
    );
}
"#;

#[test]
fn run_mode_naked_fn_seeds_incoming_abi_registers() {
    // A naked function's parameters arrive through the enclosing
    // function's ABI, not through asm operands: the register
    // file is seeded from the frame's argument locals (SysV64
    // integer registers, the Win64 position registers, and the
    // xmm sequence for a float), so the programs read their
    // actual arguments. A bare-`ret` body with a parameter it
    // never reads needs no seeding and still runs.
    run_ok(
        "asm-naked-args",
        ASM_NAKED_ARGS_SRC,
        "42 42 42 4609434218613702656\n",
    );
}

const ASM_NAKED_AGGREGATE_SRC: &str = r#"
use std::arch::naked_asm;

pub struct Pair(pub u64, pub u64);

#[unsafe(naked)]
pub extern "C" fn first(p: Pair) -> u64 {
    naked_asm!("mov rax, rdi", "ret");
}

fn main() {
    println!("{}", first(Pair(42, 1)));
}
"#;

#[test]
fn run_mode_naked_fn_unseedable_parameters_are_a_named_error() {
    // An aggregate parameter is not register-passed in the
    // model: the naked body is a named error rather than a
    // block that reads zeroed registers as if they were the
    // argument.
    let dir = temp_dir("asm-naked-aggregate");
    let src = fixture(&dir, "case.rs", ASM_NAKED_AGGREGATE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an unseedable naked parameter must exit two; stderr={stderr}"
    );
    assert!(
        stderr.contains("incoming ABI registers cannot be seeded"),
        "the reason names the unseedable parameters: {stderr}"
    );
}

const ASM_SYM_LOCAL_SRC: &str = r#"
use std::arch::asm;

pub extern "C" fn add_one(x: u64) -> u64 {
    x + 1
}

mod inner {
    pub extern "C" fn add_two(x: u64) -> u64 {
        x + 2
    }
}

fn main() {
    let out1: u64;
    let out2: u64;
    unsafe {
        asm!(
            "mov rdi, {inp}",
            "call {target}",
            inp = in(reg) 41u64,
            target = sym add_one,
            lateout("rax") out1,
        );
        asm!(
            "mov rdi, {inp}",
            "call {target}",
            inp = in(reg) 41u64,
            target = sym inner::add_two,
            lateout("rax") out2,
        );
    }
    println!("{out1} {out2}");
}
"#;

#[test]
fn run_mode_asm_sym_call_resolves_local_functions() {
    // A `sym` operand denoting a function of the local crate —
    // at the crate root or in a module — resolves by its
    // complete path and the call executes that function.
    run_ok("asm-sym-local", ASM_SYM_LOCAL_SRC, "42 43\n");
}

const ASM_SYM_AUX_LIB_SRC: &str = r#"
pub extern "C" fn foo(x: u64) -> u64 {
    x + 1000
}
"#;

const ASM_SYM_AUX_MAIN_SRC: &str = r#"
extern crate auxlib;

use std::arch::asm;

// A local function whose last name segment collides with the
// external `auxlib::foo` the asm block actually names. It is
// never called from Rust code.
pub extern "C" fn foo(x: u64) -> u64 {
    x + 1
}

fn main() {
    let out: u64;
    unsafe {
        asm!(
            "mov rdi, {inp}",
            "call {target}",
            inp = in(reg) 41u64,
            target = sym auxlib::foo,
            lateout("rax") out,
        );
    }
    println!("{out}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_an_external_function_is_rejected() {
    // The `sym` operand names `auxlib::foo`, a function of
    // another crate: it must not resolve to the unrelated local
    // `foo` whose last name segment happens to match. The call
    // stays outside the evaluable class (exit two) instead of
    // executing the local function (which would print 42; the
    // named function would print 1041).
    let dir = temp_dir("asm-sym-aux");
    let aux = fixture(&dir, "auxlib.rs", ASM_SYM_AUX_LIB_SRC);
    let src = fixture(&dir, "auxmain.rs", ASM_SYM_AUX_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("auxlib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an external asm call target must exit two; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "",
        "the unrelated local function must not execute"
    );
}

#[test]
fn convert_mode_asm_with_generic_const_and_sym_operands_converts() {
    // Regression: recording function ABIs for the naked-body
    // argument seeding queried `Instance::ty` on still-generic
    // instances (here `core::mem::size_of::<T>`, named by a
    // `const` asm operand), which panics with "needs further
    // instantiation"; the panic escaped the ABI query's
    // containment and read as a conversion crash (exit 101) in
    // convert mode.
    let dir = temp_dir("asm-generic-const");
    let src = fixture(
        &dir,
        "generic_const.rs",
        r#"use std::arch::asm;

fn foofoo<const N: usize>() {}

unsafe fn foo<const N: usize>() {
    asm!("/* {0} */", const N);
    asm!("/* {0} */", sym foofoo::<N>);
}

fn barbar<T>() {}

unsafe fn bar<T>() {
    asm!("/* {0} */", const std::mem::size_of::<T>());
    asm!("/* {0} */", sym barbar::<T>);
}

fn main() {
    unsafe {
        foo::<0>();
        bar::<usize>();
    }
}
"#,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "generic asm const/sym operands must convert; stderr={stderr}"
    );
}

const ASM_PUSH_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let x: u32 = 1;
    unsafe {
        asm!("push {}", in(reg) x);
    }
    println!("done");
}
"#;

#[test]
fn run_mode_inline_asm_outside_class_is_named_error() {
    // `push` is outside the evaluable class: conversion carries
    // the block faithfully, and the interpreter refuses it at
    // execution time — exit 2, and the reason names InlineAsm.
    let dir = temp_dir("asm-push");
    let src = fixture(&dir, "case.rs", ASM_PUSH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("unsupported terminator: InlineAsm"),
        "the reason names InlineAsm: {stderr}"
    );
}

const ASM_LATEOUT_ALIAS_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let y: u64;
    unsafe {
        asm!("mov rcx, rax", in("rax") 42u64, lateout("rax") y, out("rcx") _);
    }
    println!("{y}");
}
"#;

#[test]
fn run_mode_inline_asm_lateout_aliased_with_input() {
    // The template never writes rax, but the lateout bound to
    // it still observes the incoming content the aliased input
    // seeded: natively y == 42.
    run_ok("asm-lateout-alias", ASM_LATEOUT_ALIAS_SRC, "42\n");
}

const ASM_CALL_BOOL_ABI_WIDTH_SRC: &str = r#"
use std::arch::asm;

pub extern "sysv64" fn choose(x: bool) -> u64 {
    if x {
        42
    } else {
        43
    }
}

fn main() {
    let out1: u64;
    let out2: u64;
    unsafe {
        asm!(
            "call {target}",
            target = sym choose,
            in("rdi") 0x1_0000_0000u64,
            lateout("rax") out1,
        );
        asm!(
            "call {target}",
            target = sym choose,
            in("rdi") 0x1_0000_0001u64,
            lateout("rax") out2,
        );
    }
    println!("{out1} {out2}");
}
"#;

#[test]
fn run_mode_asm_call_bool_argument_reads_only_the_abi_byte() {
    // A boolean argument occupies one byte of its register: the
    // callee reads only the low byte, and the upper bits are
    // unspecified incoming state. The first call passes a false
    // whose upper bits are nonzero (natively 43); the second a
    // genuine true in the low byte (natively 42).
    run_ok(
        "asm-call-bool-width",
        ASM_CALL_BOOL_ABI_WIDTH_SRC,
        "43 42\n",
    );
}

const ASM_LEA_WRAP_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let y: u64;
    let z: u64;
    unsafe {
        asm!("lea {0}, [ecx + 1]", out(reg) y, in("rcx") 0xffff_ffffu64);
        asm!("lea {0}, [rcx + 1]", out(reg) z, in("rcx") 0xffff_ffffu64);
    }
    println!("{y} {z}");
}
"#;

#[test]
fn run_mode_inline_asm_lea_wraps_at_address_size() {
    // The 32-bit base form wraps the effective address at 32
    // bits before it is zero-extended into the destination (0);
    // the 64-bit base form keeps the full sum (2^32).
    run_ok("asm-lea-wrap", ASM_LEA_WRAP_SRC, "0 4294967296\n");
}

const ASM_NAKED_SIGNED_RET_SRC: &str = r#"
use std::arch::naked_asm;

#[unsafe(naked)]
pub extern "sysv64" fn neg_one64() -> i64 {
    naked_asm!("mov rax, -1", "ret");
}

#[unsafe(naked)]
pub extern "sysv64" fn neg_one32() -> i32 {
    naked_asm!("mov eax, -1", "ret");
}

#[unsafe(naked)]
pub extern "sysv64" fn neg_one16() -> i16 {
    naked_asm!("mov ax, -1", "ret");
}

#[unsafe(naked)]
pub extern "sysv64" fn neg_one8() -> i8 {
    naked_asm!("mov al, -1", "ret");
}

fn main() {
    println!(
        "{} {} {} {}",
        neg_one64(),
        neg_one32(),
        neg_one16(),
        neg_one8()
    );
}
"#;

#[test]
fn run_mode_naked_fn_signed_return_values() {
    // A naked body's return value is the register content read
    // at the return type's shape: a signed type reads the bits
    // as two's complement at its width, so each body above
    // returns -1 — natively `-1 -1 -1 -1`.
    run_ok(
        "asm-naked-signed-ret",
        ASM_NAKED_SIGNED_RET_SRC,
        "-1 -1 -1 -1\n",
    );
}

const ASM_NAKED_RSP_SRC: &str = r#"
use std::arch::naked_asm;

#[unsafe(naked)]
pub extern "sysv64" fn alignment() -> u64 {
    naked_asm!("mov rax, rsp", "and rax, 15", "ret");
}

fn main() {
    println!("{}", alignment());
}
"#;

#[test]
fn run_mode_naked_fn_incoming_rsp_read_is_a_named_error() {
    // The incoming `rsp` is fixed by the caller's frame layout
    // (natively this body returns 8, for the return address on
    // the stack); the register model does not represent it, so
    // reading it is a named error — exit 2 — rather than a
    // computation with a fabricated zero stack pointer.
    let dir = temp_dir("asm-naked-rsp");
    let src = fixture(&dir, "case.rs", ASM_NAKED_RSP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a read of the incoming stack pointer must exit two; stderr={stderr}"
    );
    assert!(
        stderr.contains("stack pointer"),
        "the reason names the incoming stack pointer: {stderr}"
    );
}

const ASM_SYM_NAKED_SRC: &str = r#"
use std::arch::{asm, naked_asm};

#[unsafe(naked)]
pub extern "sysv64" fn read_r10() -> u64 {
    naked_asm!("mov rax, r10", "ret");
}

#[unsafe(naked)]
pub extern "sysv64" fn add_r10(x: u64) -> u64 {
    naked_asm!("mov rax, rdi", "add rax, r10", "ret");
}

fn main() {
    let out1: u64;
    let out2: u64;
    unsafe {
        asm!(
            "call {target}",
            target = sym read_r10,
            in("r10") 42u64,
            lateout("rax") out1,
        );
        asm!(
            "mov rdi, {inp}",
            "call {target}",
            inp = in(reg) 41u64,
            target = sym add_r10,
            in("r10") 100u64,
            lateout("rax") out2,
        );
    }
    println!("{out1} {out2}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_naked_fn_forwards_caller_registers() {
    // A `sym` call into a naked function preserves the calling
    // block's architectural register state: the callee's body
    // reads `r10`, a register no argument seeding supplies,
    // whose value the caller established with an explicit
    // `in("r10")` operand. The second callee also takes an
    // argument: its argument register is seeded from the
    // argument as usual, and the forwarded `r10` composes with
    // it. Natively this prints `42 141`.
    run_ok("asm-sym-naked", ASM_SYM_NAKED_SRC, "42 141\n");
}

const ASM_SYM_NAKED_RSP_SRC: &str = r#"
use std::arch::{asm, naked_asm};

#[unsafe(naked)]
pub extern "sysv64" fn read_rsp() -> u64 {
    naked_asm!("mov rax, rsp", "and rax, 15", "ret");
}

fn main() {
    let out: u64;
    unsafe {
        asm!(
            "call {target}",
            target = sym read_rsp,
            lateout("rax") out,
        );
    }
    println!("{out}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_naked_fn_rsp_read_stays_a_named_error() {
    // Forwarding the calling block's register state does not
    // fabricate an incoming stack pointer for the callee: the
    // caller never established `rsp`, so the naked callee's
    // read of it stays the named error (exit 2), exactly as
    // for a naked function called from Rust code.
    let dir = temp_dir("asm-sym-naked-rsp");
    let src = fixture(&dir, "case.rs", ASM_SYM_NAKED_RSP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a forwarded read of the incoming stack pointer must exit two; stderr={stderr}"
    );
    assert!(
        stderr.contains("stack pointer"),
        "the reason names the incoming stack pointer: {stderr}"
    );
}

const ASM_SYM_NAKED_OUTGOING_SRC: &str = r#"
use std::arch::{asm, naked_asm};

#[unsafe(naked)]
pub extern "sysv64" fn set_r10() -> u64 {
    naked_asm!("mov r10, 42", "mov rax, 1", "ret");
}

fn main() {
    let via_bound: u64;
    let via_step: u64;
    let ret: u64;
    unsafe {
        asm!(
            "call {helper}",
            helper = sym set_r10,
            lateout("rax") ret,
            lateout("r10") via_bound,
        );
        asm!(
            "call {helper}",
            "mov {out}, r10",
            helper = sym set_r10,
            out = lateout(reg) via_step,
        );
    }
    println!("{ret} {via_bound} {via_step}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_naked_fn_propagates_outgoing_registers() {
    // A `sym` call into a naked function hands the callee's
    // outgoing register state back to the calling block: the
    // helper writes `r10`, which is not its return register,
    // and the caller observes 42 both through an output bound
    // to `r10` and through a step that reads `r10` after the
    // call. The declared return value still comes back in
    // `rax`. Natively this prints `1 42 42`.
    run_ok(
        "asm-sym-naked-outgoing",
        ASM_SYM_NAKED_OUTGOING_SRC,
        "1 42 42\n",
    );
}

const ASM_SYM_NAKED_PRESERVE_SRC: &str = r#"
use std::arch::{asm, naked_asm};

#[unsafe(naked)]
pub extern "sysv64" fn five() -> u64 {
    naked_asm!("mov rax, 5", "ret");
}

fn main() {
    let kept: u64;
    let ret: u64;
    unsafe {
        asm!(
            "call {helper}",
            helper = sym five,
            in("r10") 7u64,
            lateout("rax") ret,
            lateout("r10") kept,
        );
    }
    println!("{ret} {kept}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_naked_fn_preserves_untouched_registers() {
    // The outgoing hand-back covers only the registers the
    // callee's body actually changed: a register the body
    // never writes keeps the calling block's own state across
    // the call, exactly as a native `call` preserves whatever
    // the callee does not overwrite. Natively this prints
    // `5 7`.
    run_ok(
        "asm-sym-naked-preserve",
        ASM_SYM_NAKED_PRESERVE_SRC,
        "5 7\n",
    );
}

const ASM_SYM_NAKED_WIDE_ARG_SRC: &str = r#"
use std::arch::{asm, naked_asm};

#[unsafe(naked)]
pub extern "sysv64" fn helper(x: u8) -> u64 {
    naked_asm!("mov rax, rdi", "ret");
}

fn main() {
    let out: u64;
    unsafe {
        asm!(
            "call {target}",
            target = sym helper,
            in("rdi") 0x1_0000_0001u64,
            lateout("rax") out,
        );
    }
    println!("{out}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_naked_fn_keeps_forwarded_argument_bits() {
    // The caller establishes rdi at full width; the callee's
    // parameter is a u8, but its body reads the register itself,
    // whose upper bits are the caller's and observable: argument
    // seeding supplies only the registers the forwarded state
    // did not establish, so it cannot narrow this one. Natively
    // this prints `4294967297`.
    run_ok(
        "asm-sym-naked-wide-arg",
        ASM_SYM_NAKED_WIDE_ARG_SRC,
        "4294967297\n",
    );
}

const ASM_SYM_NAKED_WIDE_RET_SRC: &str = r#"
use std::arch::{asm, naked_asm};

#[unsafe(naked)]
pub extern "sysv64" fn helper() -> u8 {
    naked_asm!("mov rax, 0x100000001", "ret");
}

fn main() {
    let y: u64;
    unsafe {
        asm!(
            "call {target}",
            target = sym helper,
            lateout("rax") y,
        );
    }
    println!("{y}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_naked_fn_keeps_raw_return_register() {
    // The callee's declared return is a u8, but the calling
    // block's output reads rax itself at u64 width: the
    // callee's outgoing register state is the register's
    // content, and the narrowed Rust return value must not be
    // scattered over it. Natively this prints `4294967297`.
    run_ok(
        "asm-sym-naked-wide-ret",
        ASM_SYM_NAKED_WIDE_RET_SRC,
        "4294967297\n",
    );
}

const ASM_SYM_NAKED_BARE_RET_SRC: &str = r#"
use std::arch::{asm, naked_asm};

#[unsafe(naked)]
pub extern "sysv64" fn id_u64() -> u64 {
    naked_asm!("ret");
}

#[unsafe(naked)]
pub extern "sysv64" fn id_u8() -> u8 {
    naked_asm!("ret");
}

fn main() {
    let a: u64;
    let b: u64;
    unsafe {
        asm!(
            "call {target}",
            target = sym id_u64,
            inout("rax") 42u64 => a,
        );
        asm!(
            "call {target}",
            target = sym id_u8,
            inout("rax") 0x1_0000_0001u64 => b,
        );
    }
    println!("{a} {b}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_bare_ret_naked_fn_reads_the_return_register() {
    // A bare-`ret` naked body returns whatever the caller left
    // in the return register: the scalar return shape is read
    // from the register file even though the body has no steps
    // before the `ret`, and the register itself is preserved
    // across the call at full width — the u8 callee's upper
    // bits are the caller's. Natively this prints
    // `42 4294967297`.
    run_ok(
        "asm-sym-naked-bare-ret",
        ASM_SYM_NAKED_BARE_RET_SRC,
        "42 4294967297\n",
    );
}

const ASM_SYM_NAKED_CALLS_ORDINARY_SRC: &str = r#"
use std::arch::{asm, naked_asm};

pub extern "sysv64" fn normal() -> u64 {
    42
}

#[unsafe(naked)]
pub extern "sysv64" fn helper() -> u64 {
    naked_asm!("call {f}", "ret", f = sym normal);
}

fn main() {
    let direct = helper();
    let via: u64;
    unsafe {
        asm!(
            "call {t}",
            t = sym helper,
            lateout("rax") via,
        );
    }
    println!("{direct} {via}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_naked_fn_propagates_ordinary_callee_return() {
    // The naked helper's body obtains its return value by
    // calling an ordinary function: the call establishes the
    // return register, so the helper's outgoing state carries
    // it and a caller entering through an asm `call` observes
    // the same value as a direct call. Natively this prints
    // `42 42`.
    run_ok(
        "asm-sym-naked-calls-ordinary",
        ASM_SYM_NAKED_CALLS_ORDINARY_SRC,
        "42 42\n",
    );
}

#[test]
fn convert_mode_inline_asm_outside_class_converts() {
    // The same out-of-class block converts in full: conversion is
    // faithful for every asm shape, and only execution is gated
    // on the evaluable classes.
    let dir = temp_dir("asm-push-convert");
    let src = fixture(&dir, "case.rs", ASM_PUSH_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg(&src)
        .output()
        .expect("failed to run sparagmos-driver");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "convert succeeds: {stdout} {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("target-crate converted: 1, skipped: 0"),
        "the asm body converts: {stdout}"
    );
}

const THREAD_LOCAL_EAGER_SRC: &str = r#"
use std::cell::Cell;

thread_local! {
    static COUNTER: Cell<u32> = const { Cell::new(7) };
}

fn main() {
    COUNTER.with(|c| {
        println!("first {}", c.get());
        c.set(42);
        println!("second {}", c.get());
    });
}
"#;

#[test]
fn run_mode_thread_local_eager_cell_get_set() {
    let dir = temp_dir("tls-eager");
    let src = fixture(&dir, "tls_eager.rs", THREAD_LOCAL_EAGER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "first 7\nsecond 42\n");
}

const THREAD_LOCAL_LAZY_SRC: &str = r#"
use std::cell::RefCell;

thread_local! {
    static LIST: RefCell<Vec<u32>> = RefCell::new(vec![1, 2]);
    static NAME: RefCell<String> = RefCell::new(String::from("hello"));
}

fn show(list: &[u32]) -> String {
    let mut s = String::new();
    let mut i = 0;
    while i < list.len() {
        if i > 0 { s.push_str(","); }
        s.push_str(&list[i].to_string());
        i += 1;
    }
    s
}

fn main() {
    LIST.with(|list| {
        list.borrow_mut().push(3);
        println!("list {}", show(&list.borrow()));
    });
    NAME.with(|name| {
        let s: String = name.borrow().clone();
        println!("name {s}");
    });
    let t = std::thread::spawn(|| {
        LIST.with(|list| {
            println!("in thread {}", show(&list.borrow()));
            list.borrow_mut().push(9);
            println!("thread own {}", show(&list.borrow()));
        });
    });
    t.join().unwrap();
    LIST.with(|list| {
        println!("back {}", show(&list.borrow()));
    });
}
"#;

#[test]
fn run_mode_thread_local_lazy_refcell_and_thread_separation() {
    // The lazy storage initializes on first access in each thread:
    // the spawned thread starts from the initializer value (not the
    // main thread's mutated one), its own mutation stays its own,
    // and the main thread's value is unchanged after the join.
    let dir = temp_dir("tls-lazy");
    let src = fixture(&dir, "tls_lazy.rs", THREAD_LOCAL_LAZY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "list 1,2,3\nname hello\nin thread 1,2\nthread own 1,2,9\nback 1,2,3\n"
    );
}

const THREAD_LOCAL_INIT_ONCE_SRC: &str = r#"
use std::cell::Cell;

static mut INIT_RUNS: u32 = 0;

thread_local! {
    static VAL: Cell<u32> = {
        unsafe { INIT_RUNS += 1; }
        Cell::new(5)
    };
}

fn main() {
    VAL.with(|v| println!("a {}", v.get()));
    VAL.with(|v| println!("b {}", v.get()));
    let t = std::thread::spawn(|| {
        VAL.with(|v| println!("t {}", v.get()));
        VAL.with(|v| println!("t2 {}", v.get()));
    });
    t.join().unwrap();
    println!("runs {}", unsafe { INIT_RUNS });
}
"#;

#[test]
fn run_mode_thread_local_initializer_runs_once_per_thread() {
    let dir = temp_dir("tls-init-once");
    let src = fixture(&dir, "tls_init_once.rs", THREAD_LOCAL_INIT_ONCE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "a 5\nb 5\nt 5\nt2 5\nruns 2\n"
    );
}

const MANUALLY_DROP_DEREF_SRC: &str = r#"
use std::mem::ManuallyDrop;

struct Counter { n: u32 }
impl Counter {
    fn bump(&mut self) { self.n += 1; }
    fn get(&self) -> u32 { self.n }
}

fn main() {
    let mut md = ManuallyDrop::new(Counter { n: 41 });
    assert_eq!(md.get(), 41);
    assert_eq!(md.n, 41);
    md.bump();
    assert_eq!(md.get(), 42);
    *md = Counter { n: 7 };
    assert_eq!(md.n, 7);
    let mut v = ManuallyDrop::new(vec![1, 2, 3]);
    v.push(4);
    assert_eq!(v.len(), 4);
    assert_eq!(v[3], 4);
    let s = ManuallyDrop::new(String::from("hello"));
    assert_eq!(s.len(), 5);
    assert_eq!(&*s, "hello");
    let x = 2u8;
    let md2 = ManuallyDrop::new(&x);
    assert_eq!(**md2, 2);
    println!("ok");
}
"#;

#[test]
fn run_mode_manually_drop_deref_and_deref_mut() {
    // `ManuallyDrop<T>` derefs to its held value: method calls and
    // field reads go through `Deref`, mutations through `DerefMut`
    // land in the wrapper's own storage (a resolved-value snapshot
    // would silently drop them), and assignment through the
    // dereference replaces the held value.
    let dir = temp_dir("manually-drop-deref");
    let src = fixture(&dir, "manually_drop_deref.rs", MANUALLY_DROP_DEREF_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "ok\n", "stdout={stdout}");
}

const MANUALLY_DROP_UNION_SRC: &str = r#"
use std::mem::ManuallyDrop;

union MaybeItem<T: Iterator> {
    elem: ManuallyDrop<T::Item>,
    none: (),
}

fn main() {
    let v: Vec<u8> = vec![1, 2, 3];
    let mut i = v.iter();
    i.next();
    let mi = MaybeItem::<std::slice::Iter<'_, u8>> {
        elem: ManuallyDrop::new(i.next().unwrap()),
    };
    unsafe {
        assert_eq!(**mi.elem, 2);
    }
    println!("union ok");
}
"#;

#[test]
fn run_mode_manually_drop_in_union_field_deref() {
    // The union-generic shape: a `ManuallyDrop` read out of a
    // union field is dereferenced twice (`**mi.elem`) — the first
    // star is `ManuallyDrop`'s own `Deref`, the second the
    // reference it holds.
    let dir = temp_dir("manually-drop-union");
    let src = fixture(&dir, "manually_drop_union.rs", MANUALLY_DROP_UNION_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "union ok\n", "stdout={stdout}");
}

const ITER_COUNT_SRC: &str = r#"
fn main() {
    let a = [10, 20, 30];
    println!("range={}", (0..5).count());
    println!("slice={}", a.iter().count());
    println!("chars={}", "abcd".chars().count());
}
"#;

#[test]
fn run_mode_iterator_count_over_modelled_iterators() {
    let dir = temp_dir("iter_count");
    let src = fixture(&dir, "iter_count.rs", ITER_COUNT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "count runs; stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "range=5\nslice=3\nchars=4\n"
    );
}

const ITER_FOLD_SRC: &str = r#"
fn main() {
    let summed = (0..5).fold(0, |acc, x| acc + x);
    let a = [1, 2, 3];
    let mut weight = 1;
    let folded = a.iter().fold(0, |acc, x| {
        weight *= 2;
        acc + x * weight
    });
    println!("summed={} folded={} weight={}", summed, folded, weight);
}
"#;

#[test]
fn run_mode_iterator_fold_threads_accumulator_and_captures() {
    // The fold closure is FnMut: its captured `weight` must keep
    // its mutations across items (1*2 + 2*4 + 3*8 = 34, w = 8).
    let dir = temp_dir("iter_fold");
    let src = fixture(&dir, "iter_fold.rs", ITER_FOLD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "fold runs; stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "summed=10 folded=34 weight=8\n"
    );
}

const ITER_FOR_EACH_SRC: &str = r#"
fn main() {
    let mut total = 0;
    (0..5).for_each(|x| total += x);
    let mut seen = String::new();
    "abc".chars().for_each(|c| seen.push(c));
    println!("total={} seen={}", total, seen);
}
"#;

#[test]
fn run_mode_iterator_for_each_mutates_captures() {
    let dir = temp_dir("iter_for_each");
    let src = fixture(&dir, "iter_for_each.rs", ITER_FOR_EACH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "for_each runs; stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "total=10 seen=abc\n");
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

const ITER_COLLECT_UNSUPPORTED_SRC: &str = r#"
fn main() {
    let set: std::collections::HashSet<i32> = (0..3).collect();
    println!("n={}", set.len());
}
"#;

#[test]
fn run_mode_iterator_collect_names_unmodelled_targets() {
    // Targets without a value-model shape fail with the target
    // named, not with an approximation or a conversion error
    // from deep inside the collected FromIterator chain.
    let dir = temp_dir("iter_collect_unsupported");
    let src = fixture(&dir, "iter_collect_unsupported.rs", ITER_COLLECT_UNSUPPORTED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "unmodelled collect target must exit 2; stderr={stderr}"
    );
    assert!(
        stderr.contains("collect into std::collections::HashSet")
            && stderr.contains("is not runtime-supported"),
        "the target is named in the reason; stderr={stderr}"
    );
}

const ARRAY_INTO_ITER_COLLECT_SRC: &str = r#"
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;
fn main() {
    let v: Vec<i32> = [1, 2, 3].into_iter().collect();
    let s: String = ['a', 'b'].into_iter().collect();
    println!("v={v:?} s={s}");
    let b: Box<[i32]> = [4, 5, 6].into_iter().collect();
    let bs: Box<str> = ['h', 'i'].into_iter().collect();
    println!("b={b:?} bs={bs}");
    let r: Rc<[i32]> = [10, 20, 30].into_iter().collect();
    let r2 = r.clone();
    println!("r={} {} {}", r[2], r2.len(), Rc::strong_count(&r));
    let a: Arc<[String]> = ["x".to_string(), "yy".to_string()].into_iter().collect();
    println!("a={} {}", a.len(), a[1]);
    let mut d: VecDeque<i32> = [7, 8].into_iter().collect();
    d.push_back(9);
    let front = d.pop_front();
    println!("d={d:?} f={front:?}");
    let empty: Rc<[i32]> = [].into_iter().collect();
    let from_arr: Rc<[i32]> = Rc::from([3, 1]);
    let from_arc: Arc<[i32]> = Arc::from([2, 4]);
    println!("e={} fa={from_arr:?} fc={from_arc:?}", empty.len());
}
"#;

#[test]
fn run_mode_array_into_iter_collect_into_boxed_and_counted_slices() {
    // `array::IntoIter` is a modelled iterator, so its `collect`
    // reaches the runtime's collect model directly (unlike a
    // slice iterator, whose collected std body can build these
    // targets itself). The boxed and counted slice targets take
    // the shapes the `into_boxed_slice` / counted-slice `From`
    // models build, and the deque the `From`-conversion
    // aggregate; each collected value must then behave like the
    // native one (indexing, clone counts, mutation, drops).
    let dir = temp_dir("array_into_iter_collect");
    let src = fixture(&dir, "case.rs", ARRAY_INTO_ITER_COLLECT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "array IntoIter collect into boxed/counted/deque targets runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "v=[1, 2, 3] s=ab\nb=[4, 5, 6] bs=hi\nr=30 3 2\na=2 yy\nd=[8, 9] f=Some(7)\ne=0 fa=[3, 1] fc=[2, 4]\n"
    );
}

const ARRAY_INTO_ITER_COLLECT_PARTIAL_SRC: &str = r#"
use std::iter::FromIterator;
use std::rc::Rc;
fn main() {
    let b = Box::from_iter([4, 5].into_iter());
    println!("{:?}", b);
    let mut it = [1, 2, 3, 4].into_iter();
    let _ = it.next();
    let r: Rc<[i32]> = it.collect();
    println!("{:?}", r);
    let v = Vec::from_iter(['x', 'y'].into_iter());
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_array_into_iter_collect_partial_and_from_iter() {
    // The `FromIterator::from_iter` entry reaches the same
    // collect model, and a partially consumed array iterator
    // collects only its remaining elements.
    let dir = temp_dir("array_into_iter_collect_partial");
    let src = fixture(&dir, "case.rs", ARRAY_INTO_ITER_COLLECT_PARTIAL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "from_iter and partially consumed array IntoIter collect run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[4, 5]\n[2, 3, 4]\n['x', 'y']\n"
    );
}

// ----- IntoIterator::into_iter runtime support (Lane DT) ---------------------

const VEC_BY_VALUE_SRC: &str = r#"
fn main() {
    let v = vec![1, 2, 3];
    let mut sum = 0;
    for x in v {
        sum += x;
    }
    println!("sum={sum}");
    let mut it = vec![10, 20].into_iter();
    let a = it.next().unwrap_or(-1);
    let b = it.next().unwrap_or(-1);
    let c = it.next().unwrap_or(-1);
    println!("seq={a},{b},{c}");
    let w = vec![String::from("a"), String::from("bb"), String::from("ccc")];
    let mut n = 0;
    for s in w {
        n += s.len();
        break;
    }
    println!("n={n}");
}
"#;

#[test]
fn run_mode_vec_by_value_iteration() {
    let dir = temp_dir("vec-by-value");
    let src = fixture(&dir, "vec_by_value.rs", VEC_BY_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec by-value iteration must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "sum=6\nseq=10,20,-1\nn=1\n",
        "owned elements are yielded in order, then None"
    );
}

const VEC_DROP_ORDER_SRC: &str = r#"
struct Frag(i32);
impl Drop for Frag {
    fn drop(&mut self) { println!("drop {}", self.0); }
}
fn main() {
    let v = vec![Frag(1), Frag(2), Frag(3)];
    for x in v {
        println!("got {}", x.0);
        break;
    }
    println!("end");
    let w = vec![Frag(4), Frag(5)];
    for x in w {
        println!("got {}", x.0);
    }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_into_iter_drop_order() {
    // Each element drops exactly once: yielded ones at the end of
    // their loop-body scope, the rest in order when the iterator
    // itself is dropped.
    let dir = temp_dir("vec-drop-order");
    let src = fixture(&dir, "vec_drop_order.rs", VEC_DROP_ORDER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dropping a partially consumed vec::IntoIter must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "got 1\ndrop 1\ndrop 2\ndrop 3\nend\ngot 4\ndrop 4\ngot 5\ndrop 5\ndone\n",
        "yielded elements drop once, unyielded elements drop once, in order"
    );
}

const OPTION_BY_VALUE_SRC: &str = r#"
fn main() {
    let mut n = 0;
    for x in Some(41) {
        n += x + 1;
    }
    for x in None::<i32> {
        n += 100;
    }
    let o = Some(String::from("hi"));
    for s in o {
        n += s.len() as i32;
    }
    println!("n={n}");
    let mut m = Some(7);
    for x in &m {
        n += x;
    }
    for x in &mut m {
        *x += 1;
    }
    println!("m={}", m.unwrap_or(0));
    let p = Some(9);
    let mut t = 0;
    for x in p.iter() {
        t += x;
    }
    println!("t={t} n={n}");
}
"#;

#[test]
fn run_mode_option_iteration() {
    let dir = temp_dir("option-iteration");
    let src = fixture(&dir, "option_iteration.rs", OPTION_BY_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Option iteration (by value, by reference, .iter()) must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "n=44\nm=8\nt=9 n=51\n",
        "Some yields its payload once, None yields nothing"
    );
}

const MAP_SRC: &str = r#"
fn double(x: i32) -> i32 { x * 2 }
fn main() {
    let mut s = 0;
    for x in (1..4).map(|x| x * 10) {
        s += x;
    }
    println!("s={s}");
    let mut calls = 0;
    let v = vec![1, 2, 3];
    let mut t = 0;
    for x in v.into_iter().map(|x| { calls += 1; x + calls }) {
        t += x;
    }
    println!("t={t} calls={calls}");
    let mut u = 0;
    for x in (5..8).map(double) {
        u += x;
    }
    println!("u={u}");
}
"#;

#[test]
fn run_mode_map_iteration() {
    // The FnMut case pins that capture writes persist across
    // steps (the closure is called through its own storage).
    let dir = temp_dir("map-iteration");
    let src = fixture(&dir, "map_iteration.rs", MAP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Map over Range and vec::IntoIter must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "s=60\nt=12 calls=3\nu=36\n",
        "closure, capturing closure and fn pointer all apply per element"
    );
}

const ADAPTERS_SRC: &str = r#"
use std::iter::{empty, once, repeat};
fn main() {
    let a = [10, 20, 30];
    let mut e = 0;
    for (i, x) in a.iter().enumerate() {
        e += (i as i32) * x;
    }
    println!("e={e}");
    let mut c = 0;
    for x in a.iter().copied() {
        c += x;
    }
    println!("c={c}");
    let mut t = 0;
    for x in (0..100).take(3) {
        t += x;
    }
    println!("t={t}");
    let mut o = 0;
    for x in once(5) {
        o += x;
    }
    for x in empty::<i32>() {
        o += 100;
    }
    println!("o={o}");
    let mut r = 0;
    for x in repeat(2).take(3) {
        r += x;
    }
    println!("r={r}");
    let v = vec![1, 2, 3, 1, 4];
    let mut s = 0;
    for x in v.into_iter().skip_while(|x| *x < 3) {
        s += x;
    }
    println!("s={s}");
    let mut calls = 0;
    let w = vec![5, 6, 7];
    let mut u = 0;
    for x in w.into_iter().skip_while(|x| { calls += 1; *x < 6 }) {
        u += x;
    }
    println!("u={u} calls={calls}");
}
"#;

#[test]
fn run_mode_adapter_iteration() {
    // The SkipWhile capture count pins that the predicate is not
    // consulted again after its first failure.
    let dir = temp_dir("adapter-iteration");
    let src = fixture(&dir, "adapter_iteration.rs", ADAPTERS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Enumerate/Copied/Take/Once/Empty/Repeat/SkipWhile must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "e=80\nc=60\nt=3\no=5\nr=6\ns=8\nu=13 calls=2\n",
        "each adapter yields exactly the elements its semantics prescribe"
    );
}

const FILTER_SRC: &str = r#"
fn main() {
    let v = vec![1, 2, 3, 4, 5, 6];
    let mut sum = 0;
    for x in v.iter().filter(|x| **x % 2 == 0) {
        sum += x;
    }
    println!("sum={sum}");
    let mut calls = 0;
    let mut it = v.iter().filter(|x| { calls += 1; **x % 2 == 0 });
    println!("{:?}", it.next());
    println!("{:?}", it.next());
    println!("{:?}", it.next());
    println!("calls={calls}");
    let kept: Vec<i32> = (0..10).filter(|x| x % 3 == 0).collect();
    println!("{kept:?}");
    let n = (0..20).filter(|x| x % 2 == 1).count();
    println!("n={n}");
    let it2 = v.iter().filter(|x| **x > 1);
    println!("hint={:?}", it2.size_hint());
}
"#;

#[test]
fn run_mode_filter_iteration() {
    // The call count pins that the predicate runs once per
    // pulled element and that its capture writes persist
    // across steps (the predicate is called through its own
    // storage); the size hint is core's `(0, inner upper)`.
    let dir = temp_dir("filter-iteration");
    let src = fixture(&dir, "filter_iteration.rs", FILTER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Filter over slice::Iter and Range must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "sum=12\nSome(2)\nSome(4)\nSome(6)\ncalls=6\n[0, 3, 6, 9]\nn=10\nhint=(0, Some(6))\n",
        "the predicate accepts exactly the even elements, once per pull"
    );
}

const FILTER_BACK_SRC: &str = r#"
fn main() {
    let v = vec![1, 2, 3, 4, 5, 6];
    let mut it = v.iter().filter(|x| **x % 2 == 0);
    println!("{:?}", it.next());
    println!("{:?}", it.next_back());
    println!("{:?}", it.next());
    println!("{:?}", it.next_back());
    println!("{:?}", it.next());
    let mut rsum = 0;
    for x in (0..10).filter(|x| x % 2 == 0).rev() {
        rsum += x;
    }
    println!("revsum={rsum}");
    let out: Vec<i32> = v.iter().filter(|x| **x % 2 == 0).copied().collect();
    println!("{out:?}");
    let out2: Vec<i32> = v.iter().copied().filter(|x| x % 2 == 1).collect();
    println!("{out2:?}");
    let mut seen = 0;
    let stateful: Vec<i32> = (0..10).filter(move |x| {
        if *x % 2 == 0 {
            seen += 1;
            seen % 2 == 0
        } else {
            false
        }
    }).collect();
    println!("{stateful:?}");
}
"#;

#[test]
fn run_mode_filter_next_back_and_composition() {
    // The interleaved steps pin that front and back pulls
    // meet in the middle without re-yielding an element; the
    // stateful predicate pins that a by-value capture's
    // writes persist from one pull to the next.
    let dir = temp_dir("filter-next-back");
    let src = fixture(&dir, "filter_next_back.rs", FILTER_BACK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Filter next_back, Rev and Copied composition must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "Some(2)\nSome(6)\nSome(4)\nNone\nNone\nrevsum=20\n[2, 4, 6]\n[1, 3, 5]\n[2, 6]\n",
        "front and back steps share one predicate-gated sequence"
    );
}

const ITER_MUT_SRC: &str = r#"
fn bump(xs: &mut [i32]) {
    for x in xs {
        *x += 1;
    }
}
fn main() {
    let mut a = [1, 2, 3];
    for x in &mut a {
        *x += 10;
    }
    println!("a={} {} {}", a[0], a[1], a[2]);
    let mut v = vec![1, 2, 3];
    for x in &mut v {
        *x *= 2;
    }
    println!("v={} {} {}", v[0], v[1], v[2]);
    let mut s = [5, 6, 7];
    bump(&mut s);
    println!("s={} {} {}", s[0], s[1], s[2]);
}
"#;

#[test]
fn run_mode_iter_mut() {
    let dir = temp_dir("iter-mut");
    let src = fixture(&dir, "iter_mut.rs", ITER_MUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "IterMut over array, Vec and slice must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "a=11 12 13\nv=2 4 6\ns=6 7 8\n",
        "writes through the yielded references land in the container"
    );
}

const ITER_MUT_METHOD_SRC: &str = r#"
fn main() {
    // The `iter_mut` method builds the iterator's real state;
    // its consumers step element pointers through the
    // container's own storage.
    let mut a = [1, 2, 3, 4];
    let n = a.iter_mut().count();
    println!("count {}", n);
    for x in a.iter_mut() {
        *x *= 10;
    }
    println!("{:?}", a);
    let mut v = vec![5, 6, 7];
    let mut it = v.iter_mut();
    println!("len {} hint {:?}", it.len(), it.size_hint());
    *it.next().unwrap() += 100;
    *it.next_back().unwrap() += 100;
    println!("len {} {:?}", it.len(), v);
    let mut boxed: Box<[i32]> = Box::new([1, 2]);
    for x in boxed.iter_mut() {
        *x += 1;
    }
    println!("{:?}", boxed);
    let mut e: Vec<i32> = Vec::new();
    println!("empty {}", (&mut e).into_iter().count());
    let mut g = [1, 2, 3];
    println!("refcount {}", (&mut g).into_iter().count());
}
"#;

#[test]
fn run_mode_iter_mut_method_forms() {
    let dir = temp_dir("iter-mut-method");
    let src = fixture(&dir, "iter_mut_method.rs", ITER_MUT_METHOD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "slice::IterMut method forms must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "count 4\n[10, 20, 30, 40]\nlen 3 hint (3, Some(3))\nlen 1 [105, 6, 107]\n[2, 3]\nempty 0\nrefcount 3\n",
        "IterMut steps, counts and lengths read the real iterator state"
    );
}

const ITER_MUT_ZST_SRC: &str = r#"
fn main() {
    // A zero-sized element type stores the remaining length in
    // the iterator's end field as a provenance-free address.
    let mut z = [(), (), ()];
    let mut n = 0;
    for _ in z.iter_mut() {
        n += 1;
    }
    println!("zst {}", n);
    let mut w = [(), (), (), ()];
    let mut it = (&mut w).into_iter();
    it.next_back();
    println!("zlen {}", it.len());
    let mut m = 0;
    while it.next().is_some() {
        m += 1;
    }
    println!("zrest {}", m);
}
"#;

#[test]
fn run_mode_iter_mut_zst() {
    let dir = temp_dir("iter-mut-zst");
    let src = fixture(&dir, "iter_mut_zst.rs", ITER_MUT_ZST_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "slice::IterMut over a zero-sized element type must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "zst 3\nzlen 3\nzrest 3\n",
        "the length stored in the end field drives every step"
    );
}

const ITER_MUT_ZST_MIXED_SRC: &str = r#"
fn main() {
    // Regression for the front/back count disagreement: the
    // front step of a storage-backed ZST `IterMut` advances the
    // front pointer's element index while the end field carries
    // the terminal index, so a back step must read the end
    // field as that terminal index, not as a remaining count.
    // Front step then two back steps over a two-element array
    // yield two items total.
    let mut a = [(); 2];
    let mut it = (&mut a).into_iter();
    println!("repro {} {} {} {}",
        it.next().is_some(), it.next_back().is_some(),
        it.next_back().is_some(), it.next().is_some());
    // Back first, then front, on three elements.
    let mut b = [(); 3];
    let mut it = (&mut b).into_iter();
    println!("bf {} {} {} {} {}",
        it.next_back().is_some(), it.next().is_some(),
        it.next_back().is_some(), it.next().is_some(),
        it.next_back().is_some());
    // Alternating drain of five, with the size hint after
    // each step.
    let mut c = [(); 5];
    let mut it = (&mut c).into_iter();
    let mut total = 0;
    println!("hint0 {:?}", it.size_hint());
    if it.next().is_some() { total += 1; }
    println!("hint1 {:?}", it.size_hint());
    if it.next_back().is_some() { total += 1; }
    println!("hint2 {:?}", it.size_hint());
    if it.next_back().is_some() { total += 1; }
    if it.next().is_some() { total += 1; }
    if it.next_back().is_some() { total += 1; }
    println!("drain {} end {} {}", total,
        it.next().is_some(), it.next_back().is_some());
    // The collected `iter_mut` body shape, over a Vec and an
    // array, mixes the same way.
    let mut v = vec![(); 3];
    let mut it = v.iter_mut();
    let mut n = 0;
    if it.next().is_some() { n += 1; }
    while it.next_back().is_some() { n += 1; }
    println!("vec {}", n);
    let mut d = [(); 4];
    let mut it = d.iter_mut();
    let mut n = 0;
    if it.next_back().is_some() { n += 1; }
    while it.next().is_some() { n += 1; }
    println!("arr {}", n);
    // A named zero-sized struct element type.
    #[derive(Clone, Copy)]
    struct Zst;
    let mut e = [Zst; 3];
    let mut it = (&mut e).into_iter();
    let mut n = 0;
    if it.next().is_some() { n += 1; }
    while it.next_back().is_some() { n += 1; }
    println!("named {}", n);
    // The immutable iterator over the same shapes.
    let f = [(); 3];
    let mut it = f.iter();
    let mut n = 0;
    if it.next().is_some() { n += 1; }
    while it.next_back().is_some() { n += 1; }
    println!("iter {}", n);
    // Rev composes the back step.
    let mut g = [(); 3];
    let n = (&mut g).into_iter().rev().count();
    println!("rev {}", n);
}
"#;

#[test]
fn run_mode_iter_mut_zst_front_back_mixed() {
    let dir = temp_dir("iter-mut-zst-mixed");
    let src = fixture(&dir, "iter_mut_zst_mixed.rs", ITER_MUT_ZST_MIXED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "mixed front/back steps of a ZST slice::IterMut must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "repro true true false false\nbf true true true false false\nhint0 (5, Some(5))\nhint1 (4, Some(4))\nhint2 (3, Some(3))\ndrain 5 end false false\nvec 3\narr 4\nnamed 3\niter 3\nrev 3\n",
        "front and back steps agree on the remaining count in every shape"
    );
}

// ----- pointer offset arithmetic (Lane DV) --------------------------------
// `BinaryOp::Offset` (what `ptr::add` / `ptr::offset` compute in
// MIR) and the `arith_offset` intrinsic (what the `wrapping_*`
// family lowers to) move a pointer by whole elements. Element
// positions of an array place are represented exactly; positions
// between elements are carried as byte displacements, through
// which reads and writes stay explicit unsupported errors.

const PTR_ADD_SRC: &str = r#"
fn main() {
    let mut arr = [10i32, 20, 30, 40];
    unsafe {
        let p = arr.as_mut_ptr();
        *p.add(2) = 99;
        let q = p.add(1);
        assert_eq!(*q, 20);
        assert_eq!(*p.add(2), 99);
    }
    assert_eq!(arr[2], 99);
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_add_reads_and_writes_elements() {
    let dir = temp_dir("ptr-add");
    let src = fixture(&dir, "ptr_add.rs", PTR_ADD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "ptr::add element access must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PTR_OFFSET_SRC: &str = r#"
fn main() {
    let arr = [10i32, 20, 30, 40];
    unsafe {
        let p = arr.as_ptr();
        let q = p.offset(1);
        assert_eq!(*q, 20);
        let r = q.offset(-1);
        assert_eq!(*r, 10);
        let z = q.offset(0);
        assert_eq!(*z, 20);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_offset_checked_round_trip() {
    let dir = temp_dir("ptr-offset");
    let src = fixture(&dir, "ptr_offset.rs", PTR_OFFSET_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "checked ptr::offset round trip must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PTR_WRAPPING_SRC: &str = r#"
fn main() {
    let v: i8 = 42;
    let p = &v as *const i8;
    unsafe {
        let q = p.wrapping_sub(1);
        let r = q.wrapping_add(1);
        assert_eq!(*r, 42);
        let s = p.wrapping_offset(0);
        assert_eq!(*s, 42);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_wrapping_offset_round_trip() {
    let dir = temp_dir("ptr-wrapping");
    let src = fixture(&dir, "ptr_wrapping.rs", PTR_WRAPPING_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "wrapping pointer offsets must round-trip; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PTR_BYTE_SRC: &str = r#"
fn main() {
    let mut arr = [10i32, 20, 30, 40];
    unsafe {
        let p = arr.as_mut_ptr();
        let b = (p as *mut u8).byte_add(8);
        assert_eq!(*(b as *const i32), 30);
        let c = (p as *mut u8).byte_offset(4);
        *(c as *mut i32) = 21;
        let d = (p as *const u8).byte_add(1).byte_add(3);
        assert_eq!(*(d as *const i32), 21);
    }
    assert_eq!(arr[1], 21);
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_byte_add_and_byte_offset() {
    let dir = temp_dir("ptr-byte");
    let src = fixture(&dir, "ptr_byte.rs", PTR_BYTE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "byte_add/byte_offset on element boundaries must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PTR_BYTE_DISPLACED_READ_SRC: &str = r#"
fn main() {
    let arr = [10i32, 20, 30, 40];
    unsafe {
        let b = (arr.as_ptr() as *const u8).byte_add(1);
        let x = *(b as *const i32);
        assert_eq!(x, 0x1400_0000);
        let y = *((arr.as_ptr() as *const u8).byte_add(4));
        assert_eq!(y, 20);
        let z = *((arr.as_ptr() as *const u8).byte_add(5));
        assert_eq!(z, 0);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_byte_displaced_reads_match_native() {
    // A pointer displaced by a non-element number of bytes denotes
    // a position between elements: the load/store path reads and
    // writes it through the type-width byte view of the value's
    // storage (Lane LP, #228), composing the very bytes the native
    // program reads — an unaligned i32 load across an element
    // boundary and single bytes inside elements — never a silent
    // wrong value. Positions the byte view cannot judge (padding,
    // outside the storage) remain explicit unsupported errors.
    let dir = temp_dir("ptr-byte-displaced");
    let src = fixture(&dir, "ptr_byte_displaced.rs", PTR_BYTE_DISPLACED_READ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "byte-displaced reads must run with native values; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const ENUM_TAG_BYTE_READ_SRC: &str = r#"
enum E { A(u32), B(u16) }
enum SE { A = -2, B = 3 }
fn main() {
    let a = E::A(0x11223344);
    let b = E::B(0x1234);
    unsafe {
        let pa = &a as *const E as *const u8;
        assert_eq!(*pa, 0);
        assert_eq!(*pa.add(1), 0);
        assert_eq!(*pa.add(4), 0x44);
        let pb = &b as *const E as *const u8;
        assert_eq!(*pb, 1);
        assert_eq!(*pb.add(2), 0x34);
        // A padding byte (offset 2 of A) loads as the model's
        // indeterminate value: carried, never fabricated, and
        // never an error at the read itself.
        let _pad = *pa.add(2);
        let sa = SE::A;
        assert_eq!(*(&sa as *const SE as *const u8), 0xfe);
        let sb = SE::B;
        assert_eq!(*(&sb as *const SE as *const u8), 3);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_enum_tag_byte_reads_match_native() {
    // A byte pointer at an enum's storage reads the tag span as
    // the discriminant's own bytes (Lane MI, #247): the recorded
    // tag layout answers the positions the field rows do not
    // cover, and padding positions answer the indeterminate
    // value instead of the explicit error the byte view raised
    // before the tag layout was recorded.
    let dir = temp_dir("enum-tag-byte-read");
    let src = fixture(&dir, "enum_tag_byte_read.rs", ENUM_TAG_BYTE_READ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "enum tag byte reads must run with native values; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const ENUM_TAG_BYTE_WRITE_SRC: &str = r#"
enum E { A(u32), B(u16) }
enum E2 { A(u32), B(u32) }
fn main() {
    let mut e = E::A(0x11223344);
    unsafe { *(&mut e as *mut E as *mut u8) = 1; }
    assert!(matches!(e, E::B(_)));
    let mut e2 = E::B(0x1234);
    unsafe { *(&mut e2 as *mut E as *mut u8) = 0; }
    assert!(matches!(e2, E::A(_)));
    // Shared payload storage: the bytes survive the flip.
    let mut s = E2::A(0x11223344);
    unsafe { *(&mut s as *mut E2 as *mut u8) = 1; }
    match s {
        E2::B(v) => assert_eq!(v, 0x11223344),
        E2::A(_) => panic!("tag write must flip the variant"),
    }
    // A two-byte store over the tag span.
    let mut w = E::A(0x11223344);
    unsafe { *(&mut w as *mut E as *mut u16) = 1; }
    assert!(matches!(w, E::B(_)));
    println!("ok");
}
"#;

#[test]
fn run_mode_enum_tag_byte_write_rewrites_discriminant() {
    // A byte write into the tag span rewrites the discriminant
    // (Lane MI, #247): when the spliced tag names another
    // variant the value is re-formed as that variant, carrying
    // the storage bytes the two variants share, so a branch on
    // the rewritten value takes the native arm.
    let dir = temp_dir("enum-tag-byte-write");
    let src = fixture(&dir, "enum_tag_byte_write.rs", ENUM_TAG_BYTE_WRITE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "enum tag byte writes must flip the variant; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PTR_BYTE_VEC_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct Point { x: i32, y: i32 }
fn main() {
    // Byte displacement over a Vec's buffer is denominated in
    // the element type's size: byte_add(8) over a Vec<Point>
    // lands on element 1, not on element 8.
    let v: Vec<Point> = vec![
        Point { x: 1, y: 2 },
        Point { x: 3, y: 4 },
        Point { x: 5, y: 6 },
        Point { x: 7, y: 8 },
        Point { x: 9, y: 10 },
        Point { x: 11, y: 12 },
        Point { x: 13, y: 14 },
        Point { x: 15, y: 16 },
        Point { x: 17, y: 18 },
        Point { x: 19, y: 20 },
    ];
    let s: &[Point] = &v;
    let p = s.as_ptr();
    unsafe {
        assert_eq!(&*p.byte_add(8), &Point { x: 3, y: 4 });
        assert_eq!(&*p.byte_add(16), &Point { x: 5, y: 6 });
        assert_eq!(p.byte_add(16).offset_from(p), 2);
        assert_eq!(p.offset_from(p.byte_add(16)), -2);
        assert!(p.byte_add(8) == p.add(1));
    }
    // A write through a byte-displaced pointer lands in the
    // element the displacement denotes.
    let mut w: Vec<i32> = vec![10, 20, 30];
    let sw: &[i32] = &w;
    unsafe {
        let q = sw.as_ptr().byte_add(4) as *mut i32;
        *q = 99;
    }
    assert_eq!(w, vec![10, 99, 30]);
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_byte_add_over_vec_elements() {
    let dir = temp_dir("ptr-byte-vec");
    let src = fixture(&dir, "ptr_byte_vec.rs", PTR_BYTE_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "byte_add over Vec elements must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PTR_BYTE_VEC_FIELD_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct Point { x: i32, y: i32 }
struct Holder { v: Vec<Point> }
fn main() {
    // The same denomination holds for a Vec held in a struct
    // field: the position navigates to the field's sequence,
    // then ranges over its elements.
    let h = Holder {
        v: vec![
            Point { x: 1, y: 100 },
            Point { x: 2, y: 200 },
            Point { x: 3, y: 300 },
            Point { x: 4, y: 400 },
            Point { x: 5, y: 500 },
            Point { x: 6, y: 600 },
            Point { x: 7, y: 700 },
            Point { x: 8, y: 800 },
            Point { x: 9, y: 900 },
            Point { x: 10, y: 1000 },
        ],
    };
    let s: &[Point] = &h.v;
    let p = s.as_ptr();
    unsafe {
        assert_eq!(&*p.byte_add(8), &Point { x: 2, y: 200 });
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_byte_add_over_vec_in_struct_field() {
    let dir = temp_dir("ptr-byte-vec-field");
    let src = fixture(&dir, "ptr_byte_vec_field.rs", PTR_BYTE_VEC_FIELD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "byte_add over a Vec in a struct field must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PTR_BYTE_VEC_DISPLACED_READ_SRC: &str = r#"
#[derive(Debug)]
struct Point { x: i32, y: i32 }
fn main() {
    let v: Vec<Point> = vec![Point { x: 1, y: 2 }, Point { x: 3, y: 4 }];
    let s: &[Point] = &v;
    unsafe {
        let b = s.as_ptr().byte_add(4);
        println!("{:?}", &*b);
    }
    println!("unreachable");
}
"#;

#[test]
fn run_mode_ptr_byte_displaced_read_over_vec_is_explicit_unsupported() {
    // A byte displacement that lands between a Vec's elements
    // denotes no element: reading through it is an explicit
    // unsupported error, never the neighbouring element's value.
    let dir = temp_dir("ptr-byte-vec-displaced");
    let src = fixture(
        &dir,
        "ptr_byte_vec_displaced.rs",
        PTR_BYTE_VEC_DISPLACED_READ_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "mid-element byte-displaced read over a Vec must be an explicit unsupported; stderr={stderr}"
    );
    assert!(
        stderr.contains("displaced"),
        "the reason names the displacement; stderr={stderr}"
    );
}

const PTR_SLICE_SRC: &str = r#"
fn main() {
    let s: &[i32] = &[10, 20, 30];
    unsafe {
        let p = s.as_ptr();
        assert_eq!(*p.add(2), 30);
        assert_eq!(*p.offset(1), 20);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_offset_through_slice_pointer() {
    let dir = temp_dir("ptr-slice");
    let src = fixture(&dir, "ptr_slice.rs", PTR_SLICE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "offsets through a slice (fat) pointer must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

// ----- Lane HG: the ported intrinsic families, executed ------------

const SIMD_INTRINSICS_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::{simd_add, simd_eq, simd_extract, simd_reduce_add_unordered, simd_select, simd_splat};
#[repr(simd)]
#[derive(Copy, Clone)]
struct I32x4([i32; 4]);
fn main() {
    unsafe {
        let a = I32x4([1, 2, 3, 4]);
        let b = I32x4([10, 20, 30, 40]);
        let c = simd_add(a, b);
        let v: i32 = simd_extract(c, 2);
        assert_eq!(v, 33);
        let total: i32 = simd_reduce_add_unordered(c);
        assert_eq!(total, 110);
        let seven: I32x4 = simd_splat(7);
        let t2: i32 = simd_reduce_add_unordered(seven);
        assert_eq!(t2, 28);
        let mask: I32x4 = simd_eq(a, b);
        let picked = simd_select(mask, a, b);
        let p0: i32 = simd_extract(picked, 0);
        let p1: i32 = simd_extract(picked, 1);
        assert_eq!(p0, 10);
        assert_eq!(p1, 20);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_simd_intrinsics_execute() {
    let dir = temp_dir("simd-intrinsics");
    let src = fixture(&dir, "simd_intrinsics.rs", SIMD_INTRINSICS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the simd intrinsic family must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const INTRINSIC_LEFTOVERS_SRC: &str = r#"
#![feature(core_intrinsics)]
use std::sync::atomic::{AtomicI32, AtomicU8, Ordering};
fn main() {
    let x = AtomicI32::new(-5);
    assert_eq!(x.fetch_max(3, Ordering::SeqCst), -5);
    assert_eq!(x.load(Ordering::SeqCst), 3);
    let y = AtomicU8::new(200);
    assert_eq!(y.fetch_add(100, Ordering::SeqCst), 200);
    assert_eq!(y.load(Ordering::SeqCst), 44);
    unsafe {
        let f = core::intrinsics::fadd_fast(1.5f64, 2.25f64);
        assert_eq!(f, 3.75);
        let z: i32 = core::intrinsics::float_to_int_unchecked(3.75f64);
        assert_eq!(z, 3);
        let n: u32 = core::intrinsics::ctlz_nonzero(16u32);
        assert_eq!(n, 27);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_atomic_fetch_fast_float_and_nonzero_bit_intrinsics_execute() {
    let dir = temp_dir("intrinsic-leftovers");
    let src = fixture(&dir, "intrinsic_leftovers.rs", INTRINSIC_LEFTOVERS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the fetch family, fast float and nonzero bit intrinsics must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const MEMORY_INTRINSICS_SRC: &str = r#"
fn main() {
    let mut buf = [0u8; 4];
    unsafe {
        std::ptr::write_bytes(buf.as_mut_ptr(), 0xABu8, 4);
        assert_eq!(buf, [0xAB; 4]);
        let mut v: u32 = 0;
        std::ptr::write_volatile(&mut v, 0x0101_0101);
        assert_eq!(std::ptr::read_volatile(&v), 0x0101_0101);
        let arr = [10u32, 20, 30, 40, 50];
        let base = arr.as_ptr();
        let third = base.add(2);
        assert_eq!(third.offset_from(base), 2);
    }
    println!("ok");
}
"#;

const ELEM_PTR_BULK_SRC: &str = r#"
fn main() {
    let mut dst = [0u8; 6];
    let src = [9u8, 8, 7];
    unsafe {
        core::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().add(2), 3);
        std::ptr::write_bytes(dst.as_mut_ptr().add(5), 0xEEu8, 1);
        std::ptr::write_bytes(dst.as_mut_ptr(), 0xAAu8, 2);
    }
    println!("{:?}", dst);
}
"#;

#[test]
fn run_mode_bulk_ops_through_element_pointers() {
    // Bulk memory operations take a pointer to the sequence's
    // first element: through an element pointer (the shape an
    // array-to-element cast produces) the copy / fill must land on
    // that element and its siblings, not index into the element
    // itself.
    let dir = temp_dir("elem-ptr-bulk");
    let src = fixture(&dir, "elem_ptr_bulk.rs", ELEM_PTR_BULK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "bulk ops through element pointers must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "[170, 170, 9, 8, 7, 238]\n");
}

const RAW_BLOCK_FILL_SRC: &str = r#"
fn main() {
    let s = String::from("ok");
    assert_eq!(s.len(), 2);
}
"#;

#[test]
fn run_mode_raw_block_bulk_fill_under_optimized_mir() {
    // Under optimized MIR, `String::from` / `to_vec` allocate a
    // raw block through `__rust_alloc` and fill it with
    // `copy_nonoverlapping` before any element exists. The fill's
    // destination probe reads element 0 of the still-empty block:
    // an unwritten slot of a raw block must read as uninitialized
    // (the write side then grows the block), never as a guest
    // "index out of bounds" panic -- the guest indexed nothing.
    let dir = temp_dir("raw-block-fill");
    let src = fixture(&dir, "raw_block_fill.rs", RAW_BLOCK_FILL_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("-Copt-level=3")
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "bulk fill of a fresh raw block must execute; stderr={stderr}"
    );
}

#[test]
fn run_mode_write_bytes_volatile_and_offset_from_execute() {
    let dir = temp_dir("memory-intrinsics");
    let src = fixture(&dir, "memory_intrinsics.rs", MEMORY_INTRINSICS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "write_bytes, volatile accesses and ptr_offset_from must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

// ----- drop glue for modelled std shapes (Issue #114) --------------------------

const DROP_MAYBE_UNINIT_SRC: &str = r#"
use std::mem::MaybeUninit;

fn main() {
    let mut v: Vec<MaybeUninit<[u32; 2]>> = Vec::new();
    v.push(MaybeUninit::new([1, 2]));
    v.push(MaybeUninit::uninit());
    let b: Box<MaybeUninit<String>> = Box::new(MaybeUninit::new(String::from("held")));
    drop(v);
    drop(b);
    println!("ok");
}
"#;

#[test]
fn run_mode_drop_maybe_uninit_containers() {
    run_ok("drop-maybe-uninit", DROP_MAYBE_UNINIT_SRC, "ok\n");
}

const DROP_CELL_SRC: &str = r#"
use std::cell::Cell;

struct Flag(u32);
impl Drop for Flag {
    fn drop(&mut self) { println!("flag {} dropped", self.0); }
}

fn main() {
    let c = Cell::new(Some(Flag(7)));
    drop(c);
    println!("done");
}
"#;

#[test]
fn run_mode_drop_cell_runs_contents_glue() {
    // The modelled `Cell` value nests its contents two aggregates
    // deep; dropping the cell must reach the contents' destructor.
    run_ok("drop-cell", DROP_CELL_SRC, "flag 7 dropped\ndone\n");
}

const DROP_WAKER_SRC: &str = r#"
use std::sync::Arc;
use std::task::{Wake, Waker};

struct FlagWaker {
    held: String,
}
impl Wake for FlagWaker {
    fn wake(self: Arc<Self>) {}
}

fn main() {
    let w = Waker::from(Arc::new(FlagWaker { held: String::from("x") }));
    drop(w);
    println!("done");
}
"#;

#[test]
fn run_mode_drop_waker_releases_wake_object() {
    // Dropping a Waker releases the counted handle behind it;
    // the erased wake object's own glue is judged by shape (the
    // Dynamic plan), which carries the held string's release.
    run_ok("drop-waker", DROP_WAKER_SRC, "done\n");
}

const DROP_LOCK_GUARDS_SRC: &str = r#"
use std::sync::Mutex;

fn main() {
    let m = Mutex::new(1u32);
    {
        let _g = m.lock().unwrap();
    }
    println!("guards dropped");
}
"#;

#[test]
fn run_mode_drop_lock_guards() {
    // Acquiring and dropping each guard kind: the guards' glue is
    // their fields' (a lock reference and a poison flag), since
    // the lock state itself is not modelled.
    run_ok("drop-lock-guards", DROP_LOCK_GUARDS_SRC, "guards dropped\n");
}

const DROP_TYPEID_SRC: &str = r#"
use std::any::TypeId;

fn main() {
    let ids = [TypeId::of::<u32>(), TypeId::of::<String>()];
    println!("{}", ids[0] == TypeId::of::<u32>());
    drop(ids);
    println!("done");
}
"#;

#[test]
fn run_mode_drop_type_id_values() {
    run_ok("drop-typeid", DROP_TYPEID_SRC, "true\ndone\n");
}

const DROP_FLAT_MAP_SRC: &str = r#"
fn main() {
    let it = Some(())
        .into_iter()
        .flat_map(|_| Some(()).into_iter().flat_map(|_| vec![1, 2]));
    drop(it);
    println!("ok");
}
"#;

#[test]
fn run_mode_drop_flat_map_chain() {
    run_ok("drop-flat-map", DROP_FLAT_MAP_SRC, "ok\n");
}

const DROP_PHANTOM_PINNED_SRC: &str = r#"
use std::marker::PhantomPinned;

struct Pinned {
    value: u32,
    _pin: PhantomPinned,
}
impl Drop for Pinned {
    fn drop(&mut self) { println!("pinned {} dropped", self.value); }
}

fn main() {
    let p = Pinned { value: 3, _pin: PhantomPinned };
    drop(p);
    println!("done");
}
"#;

#[test]
fn run_mode_drop_phantom_pinned_struct() {
    run_ok("drop-phantom-pinned", DROP_PHANTOM_PINNED_SRC, "pinned 3 dropped\ndone\n");
}

const ARRAY_AS_PTR_SRC: &str = r#"
fn main() {
    let arr = [10i32, 20, 30];
    let p = arr.as_ptr();
    let q = &arr as *const i32;
    let r = &arr[0] as *const i32;
    unsafe {
        println!("{}", *p);
        println!("{}", *q);
        println!("{}", p == r);
    }
}
"#;

#[test]
fn run_mode_array_as_ptr_reads_first_element() {
    // An element pointer (from `as_ptr` or an `as` cast) denotes the
    // array's first element, not the whole array aggregate: reading
    // through it used to read the tuple value of the whole array.
    // It also denotes exactly the place `&arr[0]` denotes.
    let dir = temp_dir("array-as-ptr");
    let src = fixture(&dir, "array_as_ptr.rs", ARRAY_AS_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "10\n10\ntrue\n");
}

const ELEM_PTR_WRITE_SRC: &str = r#"
fn main() {
    let mut a = [10i32, 20, 30];
    let p = a.as_mut_ptr();
    unsafe { *p = 99; }
    let mut b = [10i32, 20, 30];
    let q = &mut b as *mut i32;
    unsafe { *q = 7; }
    println!("{} {}", a[0], b[0]);
}
"#;

#[test]
fn run_mode_element_pointer_writes_hit_first_element() {
    // Writing through an element pointer writes the first element;
    // it used to fail as an assignment of a scalar to the whole
    // array place.
    let dir = temp_dir("elem-ptr-write");
    let src = fixture(&dir, "elem_ptr_write.rs", ELEM_PTR_WRITE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "99 7\n");
}

const STR_AS_PTR_SRC: &str = r#"
fn main() {
    let s = "hello";
    let owned = String::from("hi");
    let bytes = b"abc";
    unsafe {
        println!("{}", *s.as_ptr());
        println!("{}", *owned.as_ptr());
        println!("{}", *bytes.as_ptr());
    }
}
"#;

#[test]
fn run_mode_str_as_ptr_reads_first_byte() {
    // `str::as_ptr` (and `String::as_ptr` through it) denotes the
    // first byte; it used to read the whole string value, which the
    // formatter then printed verbatim - a silent wrong result.
    let dir = temp_dir("str-as-ptr");
    let src = fixture(&dir, "str_as_ptr.rs", STR_AS_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "104\n104\n97\n");
}

const ELEM_PTR_SHAPES_SRC: &str = r#"
struct S { a: [i32; 3] }
static ARR: [i32; 3] = [1, 2, 3];
fn main() {
    let s = S { a: [10, 20, 30] };
    let arr = [40i32, 50, 60];
    let slice: &[i32] = &arr[..];
    unsafe {
        println!("{}", *s.a.as_ptr());
        println!("{}", *ARR.as_ptr());
        println!("{}", *slice.as_ptr());
    }
}
"#;

#[test]
fn run_mode_element_pointers_for_field_static_and_slice() {
    // The first-element retargeting composes with the projection
    // the base pointer already carries (a struct field), with
    // statics, and with slices (whose pointer rides a fat base).
    let dir = temp_dir("elem-ptr-shapes");
    let src = fixture(&dir, "elem_ptr_shapes.rs", ELEM_PTR_SHAPES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "10\n1\n40\n");
}

const ELEM_PTR_ROUNDTRIP_SRC: &str = r#"
fn main() {
    let arr = [10i32, 20, 30];
    let p = arr.as_ptr().cast::<[i32; 3]>();
    unsafe { println!("{}", (*p)[2]); }
}
"#;

#[test]
fn run_mode_element_pointer_cast_back_to_array() {
    // Casting an element pointer back to a whole-array pointer
    // denotes the array again (the index-0 projection the
    // first-element cast added is removed).
    let dir = temp_dir("elem-ptr-roundtrip");
    let src = fixture(&dir, "elem_ptr_roundtrip.rs", ELEM_PTR_ROUNDTRIP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "30\n");
}

// ----- one closure instance demanded by two collectors (Lane HM) -----------

const SUM_CLOSURE_SRC: &str = r#"
fn main() {
    let v: [i128; 3] = [10, 20, 12];
    let s: i128 = v.iter().sum();
    assert_eq!(s, 42);
    println!("ok");
}
"#;

#[test]
fn run_mode_closure_demanded_by_two_collectors_converts_once() {
    // `<i128 as Sum<&i128>>::sum` creates a closure and hands it
    // to `Iterator::fold`. Two collectors demand that one closure
    // instance: the reachability collection resolves the closures
    // in a call's generic arguments itself (the callee is
    // builtin-owned, so its body is excluded), and the closure
    // demand pass finds the same closure mentioned in the
    // collected `sum` body. Both used to convert it, the program
    // held two bodies under one DefId, and conversion failed with
    // "duplicate function name
    // `<i128 as Sum<&'a i128>>::sum::{closure#0}`" (the
    // tests/ui/numbers-arithmetic/i128.rs shape). The demand
    // passes now skip an instance another collector already
    // converted, and the reachability merges drop duplicates by
    // DefId, so the closure enters the program exactly once.
    let dir = temp_dir("sum-closure");
    let src = fixture(&dir, "sum_closure.rs", SUM_CLOSURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the doubly-demanded closure must convert once and run; stderr={stderr}"
    );
    assert!(
        !stderr.contains("duplicate function name"),
        "no duplicate registration; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const ASSERT_INHABITED_SRC: &str = r#"
#![feature(core_intrinsics)]
fn main() {
    unsafe {
        std::intrinsics::assert_inhabited::<u32>();
        std::intrinsics::assert_inhabited::<[u8; 3]>();
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_assert_inhabited_is_a_no_op() {
    let dir = temp_dir("assert-inhabited");
    let src = fixture(&dir, "assert_inhabited.rs", ASSERT_INHABITED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "assert_inhabited must run as a no-op; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}


const ALLOC_CYCLE_SRC: &str = r#"
use std::alloc::{alloc, alloc_zeroed, dealloc, realloc, Layout};
fn main() {
    unsafe {
        let layout = Layout::array::<u32>(4).unwrap();
        let p = alloc(layout) as *mut u32;
        assert!(!p.is_null());
        for i in 0..4 { p.add(i).write(10 + i as u32); }
        for i in 0..4 { assert_eq!(*p.add(i), 10 + i as u32); }
        let layout8 = Layout::array::<u32>(8).unwrap();
        let p = realloc(p as *mut u8, layout, layout8.size()) as *mut u32;
        assert!(!p.is_null());
        for i in 0..4 { assert_eq!(*p.add(i), 10 + i as u32); }
        dealloc(p as *mut u8, layout8);
        let zlayout = Layout::array::<u8>(3).unwrap();
        let z = alloc_zeroed(zlayout);
        assert!(!z.is_null());
        assert_eq!(*z, 0);
        assert_eq!(*z.add(2), 0);
        dealloc(z, zlayout);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_raw_allocator_cycle() {
    let dir = temp_dir("alloc-cycle");
    let src = fixture(&dir, "alloc_cycle.rs", ALLOC_CYCLE_SRC);
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const SIMD_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::*;
#[repr(simd)]
#[derive(Clone, Copy)]
struct Simd<T, const N: usize>([T; N]);
type I4 = Simd<i32, 4>;
fn main() {
    unsafe {
        let a: I4 = Simd([1, 2, 3, 4]);
        let b: I4 = Simd([10, 20, 30, 40]);
        let c: I4 = simd_mul(simd_add(a, b), Simd([2, 2, 2, 2]));
        let e0: i32 = simd_extract(c, 0);
        let e1: i32 = simd_extract(c, 1);
        let e3: i32 = simd_extract(c, 3);
        assert_eq!((e0, e1, e3), (22, 44, 88));
        let total: i32 = simd_reduce_add_unordered(c);
        assert_eq!(total, 220);
        let m: Simd<i32, 4> = simd_gt(c, Simd([50, 50, 50, 50]));
        let bits: u8 = simd_bitmask(m);
        assert_eq!(bits, 0b1100);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_simd_intrinsics() {
    let dir = temp_dir("simd");
    let src = fixture(&dir, "simd.rs", SIMD_SRC);
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const VOLATILE_SRC: &str = r#"
#![feature(core_intrinsics)]
use std::intrinsics::{volatile_load, volatile_store};
#[derive(Debug, Eq, PartialEq)]
struct A(u32);
fn main() {
    unsafe {
        let mut x = A(0);
        volatile_store(&mut x as *mut A, A(1));
        assert_eq!(x, A(1));
        let v = volatile_load(&x as *const A);
        assert_eq!(v, A(1));
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_volatile_composite_access() {
    let dir = temp_dir("volatile");
    let src = fixture(&dir, "volatile.rs", VOLATILE_SRC);
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const POW_PARSE_SRC: &str = r#"
fn main() {
    assert_eq!(2i32.pow(10), 1024);
    assert_eq!((-3i64).pow(3), -27);
    assert_eq!(7u32.pow(0), 1);
    let x: f64 = "3.1415".parse().unwrap();
    assert!(x > 3.1414 && x < 3.1416);
    let n: i32 = "-42".parse().unwrap();
    assert_eq!(n, -42);
    assert!("abc".parse::<i32>().is_err());
    assert!("99999999999999999999".parse::<i32>().is_err());
    let b: bool = "true".parse().unwrap();
    assert!(b);
    println!("ok");
}
"#;

#[test]
fn run_mode_pow_and_from_str_parse() {
    let dir = temp_dir("pow-parse");
    let src = fixture(&dir, "pow_parse.rs", POW_PARSE_SRC);
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const POW_OVERFLOW_SRC: &str = r#"
fn main() { let _ = 2i32.pow(1024); }
"#;

#[test]
fn run_mode_pow_overflow_panics() {
    let dir = temp_dir("pow-overflow");
    let src = fixture(&dir, "pow_overflow.rs", POW_OVERFLOW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "overflowing pow must exit 101; stderr={stderr}"
    );
    assert!(
        stderr.contains("attempt to exponentiate with overflow"),
        "overflow message goes to stderr: {stderr}"
    );
}

// ----- composite layouts and remaining Default shapes (Lane HQ) -----

const COMPOSITE_LAYOUT_SRC: &str = r#"
fn main() {
    let a: [u16; 3] = [1, 2, 3];
    println!("arr {} {}", std::mem::size_of_val(&a), std::mem::align_of_val(&a));
    let t: (u8, u32) = (1, 2);
    println!("tup {} {}", std::mem::size_of_val(&t), std::mem::align_of_val(&t));
    let n: [[u8; 3]; 2] = [[1, 2, 3], [4, 5, 6]];
    println!("nest {} {}", std::mem::size_of_val(&n), std::mem::align_of_val(&n));
    let m: (u16, [u8; 5]) = (7, [1, 2, 3, 4, 5]);
    println!("mix {} {}", std::mem::size_of_val(&m), std::mem::align_of_val(&m));
}
"#;

#[test]
fn run_mode_size_of_val_composite_layouts() {
    // The layouts are computed from the element / field layouts
    // (verified against a native 1.100.0 build of the same source).
    run_ok(
        "composite-layout",
        COMPOSITE_LAYOUT_SRC,
        "arr 6 2\ntup 8 4\nnest 6 1\nmix 8 2\n",
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

const HASH_COLLECTION_DEFAULT_SRC: &str = r#"
use std::collections::{HashMap, HashSet};

fn main() {
    let e: HashSet<i32> = Default::default();
    println!("empty contains={} len={}", e.contains(&9), e.len());
    let mut s: HashSet<i32> = Default::default();
    for i in 0..100 {
        s.insert(i * 3);
    }
    println!(
        "grown len={} has99={} has100={}",
        s.len(),
        s.contains(&99),
        s.contains(&100)
    );
    let sum: i32 = s.iter().sum();
    println!("iter sum={}", sum);
    let mut m: HashMap<String, i32> = Default::default();
    for i in 0..50 {
        m.insert(format!("k{}", i), i);
    }
    *m.get_mut(&"k7".to_string()).unwrap() += 100;
    m.remove(&"k0".to_string());
    println!(
        "map len={} k7={:?} k0={:?}",
        m.len(),
        m.get(&"k7".to_string()),
        m.get(&"k0".to_string())
    );
    println!("values sum={}", m.values().sum::<i32>());
}
"#;

/// `std::panic::set_backtrace_style` has no converted body
/// (it writes a std-internal atomic); the runtime records
/// the style. The rustdoc entry program installs its style
/// during startup.
const BACKTRACE_STYLE_SRC: &str = r#"#![feature(panic_backtrace_config)]
fn main() {
    std::panic::set_backtrace_style(std::panic::BacktraceStyle::Short);
    std::panic::set_backtrace_style(std::panic::BacktraceStyle::Full);
    println!("styled");
}
"#;

#[test]
fn run_mode_set_backtrace_style() {
    run_ok("backtrace-style", BACKTRACE_STYLE_SRC, "styled\n");
}

#[test]
fn run_mode_default_for_hash_collections() {
    // `Default::default` for the std hash collections has no
    // collected body (the trait path reaches the interpreter's
    // name fallback), so the model assembles the empty table
    // directly. The expected lines are the native 1.100.0
    // output of the same source: probes of the empty table,
    // growth out of the empty singleton, iteration, and
    // mutation of a default-built map.
    run_ok(
        "hash-collection-default",
        HASH_COLLECTION_DEFAULT_SRC,
        "empty contains=false len=0\ngrown len=100 has99=true has100=false\niter sum=14850\nmap len=49 k7=Some(107) k0=None\nvalues sum=1325\n",
    );
}

const NEXT_BACK_SLICE_SRC: &str = r#"
fn main() {
    let v = vec![10, 20, 30, 40];
    let mut it = v.iter();
    println!("back={:?}", it.next_back());
    println!("front={:?}", it.next());
    println!("back={:?}", it.next_back());
    println!("back={:?}", it.next_back());
    println!("back={:?}", it.next_back());
}
"#;

#[test]
fn run_mode_next_back_slice_iter() {
    run_ok(
        "next-back-slice",
        NEXT_BACK_SLICE_SRC,
        "back=Some(40)\nfront=Some(10)\nback=Some(30)\nback=Some(20)\nback=None\n",
    );
}

const NEXT_BACK_VEC_INTO_SRC: &str = r#"
fn main() {
    let mut it = vec![1, 2, 3].into_iter();
    println!("back={:?}", it.next_back());
    println!("front={:?}", it.next());
    println!("back={:?}", it.next_back());
    println!("back={:?}", it.next_back());
}
"#;

#[test]
fn run_mode_next_back_vec_into_iter() {
    run_ok(
        "next-back-vec-into",
        NEXT_BACK_VEC_INTO_SRC,
        "back=Some(3)\nfront=Some(1)\nback=Some(2)\nback=None\n",
    );
}

const NEXT_BACK_RANGE_SRC: &str = r#"
fn main() {
    let mut it = 1..5;
    println!("back={:?}", it.next_back());
    println!("front={:?}", it.next());
    println!("back={:?}", it.next_back());
    let mut incl = 1..=3;
    println!("incl back={:?}", incl.next_back());
    println!("incl front={:?}", incl.next());
    println!("incl back={:?}", incl.next_back());
    println!("incl back={:?}", incl.next_back());
}
"#;

#[test]
fn run_mode_next_back_ranges() {
    run_ok(
        "next-back-ranges",
        NEXT_BACK_RANGE_SRC,
        "back=Some(4)\nfront=Some(1)\nback=Some(3)\nincl back=Some(3)\nincl front=Some(1)\nincl back=Some(2)\nincl back=None\n",
    );
}

const NEXT_BACK_ADAPTERS_SRC: &str = r#"
fn main() {
    let v = vec![1, 2, 3];
    let mut copied = v.iter().copied();
    println!("copied back={:?}", copied.next_back());
    let mut mapped = (1..=3).map(|x| x * 10);
    println!("map back={:?}", mapped.next_back());
    println!("map front={:?}", mapped.next());
    let mut enumerated = v.iter().enumerate();
    println!("enum back={:?}", enumerated.next_back());
    println!("enum front={:?}", enumerated.next());
    let mut reversed = (1..=3).rev();
    println!("rev back={:?}", reversed.next_back());
    println!("rev front={:?}", reversed.next());
    let mut once = std::iter::once(42);
    println!("once back={:?}", once.next_back());
    println!("once back={:?}", once.next_back());
}
"#;

#[test]
fn run_mode_next_back_adapters() {
    run_ok(
        "next-back-adapters",
        NEXT_BACK_ADAPTERS_SRC,
        "copied back=Some(3)\nmap back=Some(30)\nmap front=Some(10)\nenum back=Some((2, 3))\nenum front=Some((0, 1))\nrev back=Some(1)\nrev front=Some(3)\nonce back=Some(42)\nonce back=None\n",
    );
}

const NEXT_BACK_OPTION_SRC: &str = r#"
fn main() {
    let mut it = Some(9).into_iter();
    println!("opt back={:?}", it.next_back());
    println!("opt back={:?}", it.next_back());
    let mut front = Some(7).into_iter();
    println!("opt front={:?}", front.next());
    println!("opt back={:?}", front.next_back());
}
"#;

#[test]
fn run_mode_next_back_option_into_iter() {
    run_ok(
        "next-back-option",
        NEXT_BACK_OPTION_SRC,
        "opt back=Some(9)\nopt back=None\nopt front=Some(7)\nopt back=None\n",
    );
}

const NEXT_BACK_USER_SRC: &str = r#"
struct Countdown(u32);
impl Iterator for Countdown {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.0 == 0 { None } else { self.0 -= 1; Some(self.0) }
    }
}
impl DoubleEndedIterator for Countdown {
    fn next_back(&mut self) -> Option<u32> {
        if self.0 == 0 { None } else { let v = self.0; self.0 -= 1; Some(v) }
    }
}
fn main() {
    let mut it = Countdown(3);
    println!("back={:?}", it.next_back());
    println!("front={:?}", it.next());
    println!("back={:?}", it.next_back());
    println!("back={:?}", it.next_back());
}
"#;

#[test]
fn run_mode_next_back_user_defined_iterator() {
    // A user-defined `DoubleEndedIterator` keeps its collected
    // body as the primary execution: the builtin only serves
    // the modelled iterator types.
    run_ok(
        "next-back-user",
        NEXT_BACK_USER_SRC,
        "back=Some(3)\nfront=Some(1)\nback=Some(1)\nback=None\n",
    );
}
const BOX_ASREF_IDENTITY_SRC: &str = r#"
#[repr(align(16))]
#[derive(Clone, Copy)]
struct Align16(i32);

fn main() {
    // The `AsRef` view of a box pointee is the pointee itself:
    // the result must be a reference to the box's storage, not
    // the pointee value.
    let b = Box::new(Align16(41));
    let r: &Align16 = b.as_ref();
    assert_eq!(r.0, 41);
    assert_eq!(std::mem::align_of_val(r), 16);
    assert_eq!(std::mem::size_of_val(r), 16);
    // An identity view of an aggregate receiver is a reference
    // to the receiver, not the receiver value.
    let v = vec![3u32, 4];
    let rv: &Vec<u32> = v.as_ref();
    assert_eq!(rv.len(), 2);
    assert_eq!(rv[0], 3);
    println!("box asref ok");
}
"#;

#[test]
fn run_mode_box_asref_identity_view_is_a_reference() {
    // Regression: the `AsRef` builtin (Lane HD) resolved a
    // `Box<T> -> &T` view to the bare pointee value, which the
    // destination's `Ref` type check rejects ("expected
    // Ref(Adt(..)), found ADT value") — this broke
    // tests/ui/structs-enums/align-struct.rs at both19. The
    // identity views (box pointee, aggregate self view) now
    // return reference-shaped results.
    let dir = temp_dir("box-asref");
    let src = fixture(&dir, "box_asref.rs", BOX_ASREF_IDENTITY_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "box asref ok\n");
}

// ---- Lane HU: body-less call models (issue #130) ----
// A trait method named by its plain trait path reaches the
// interpreter with no body under the name when it arrives as a
// function value or inside a body the converter kept in generic
// form; the value-semantics builtins serve those calls at run
// time. `HashMap::new` draws its `RandomState` keys from the
// modelled `hashmap_random_keys`.

const CLONED_SUM_SRC: &str = r#"
fn main() {
    let v = vec![1i64, 2, 3];
    let s: i64 = v.iter().cloned().sum();
    assert_eq!(s, 6);
    println!("ok");
}
"#;

#[test]
fn run_mode_cloned_sum_serves_trait_path_clone() {
    let dir = temp_dir("cloned-sum");
    let src = fixture(&dir, "cloned_sum.rs", CLONED_SUM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const MAP_CLONE_SRC: &str = r#"
fn main() {
    let v = vec![1i32, 2, 3];
    let w: Vec<i32> = v.iter().map(Clone::clone).collect();
    assert_eq!(w.len(), 3);
    println!("ok");
}
"#;

#[test]
fn run_mode_map_fn_item_clone() {
    let dir = temp_dir("map-clone");
    let src = fixture(&dir, "map_clone.rs", MAP_CLONE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

// The map is held behind a trait-object box, the shape of
// rustc's issue-6128 test: a plain local `HashMap` value's drop
// glue is a separate, still-unmodelled conversion gap.
const HASHMAP_NEW_SRC: &str = r#"
use std::collections::HashMap;
trait G { fn g(&self); }
impl G for HashMap<i32, i32> { fn g(&self) {} }
fn main() {
    let g: Box<HashMap<i32, i32>> = Box::new(HashMap::new());
    let _g2: Box<dyn G> = g as Box<dyn G>;
    println!("ok");
}
"#;

#[test]
fn run_mode_hashmap_new_uses_modelled_random_keys() {
    let dir = temp_dir("hashmap-new");
    let src = fixture(&dir, "hashmap_new.rs", HASHMAP_NEW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const RANDOM_STATE_DEFAULT_SRC: &str = r#"
use std::collections::HashMap;
use std::hash::RandomState;
trait G { fn g(&self); }
impl G for HashMap<i32, i32> { fn g(&self) {} }
fn main() {
    let h: Box<HashMap<i32, i32>> = Box::new(HashMap::with_hasher(RandomState::default()));
    let _h2: Box<dyn G> = h as Box<dyn G>;
    println!("ok");
}
"#;

#[test]
fn run_mode_random_state_default_builds_hasher() {
    let dir = temp_dir("random-state-default");
    let src = fixture(&dir, "random_state_default.rs", RANDOM_STATE_DEFAULT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const GENERIC_PATHS_SRC: &str = r#"
fn make<T>() -> Vec<T> { Vec::<T>::new() }
fn main() {
    let v: Vec<i32> = make();
    assert!(v.is_empty());
    let d = <i32 as std::default::Default>::default();
    assert_eq!(d, 0);
    println!("ok");
}
"#;

#[test]
fn run_mode_generic_context_vec_new_and_qualified_default() {
    let dir = temp_dir("generic-paths");
    let src = fixture(&dir, "generic_paths.rs", GENERIC_PATHS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PHANTOM_DEFAULT_SRC: &str = r#"
use std::marker::PhantomData;
#[derive(Default)]
struct Wrap<T> { _p: PhantomData<T>, x: i32 }
fn main() {
    let w: Wrap<String> = Wrap::default();
    assert_eq!(w.x, 0);
    println!("ok");
}
"#;

#[test]
fn run_mode_phantom_data_default() {
    let dir = temp_dir("phantom-default");
    let src = fixture(&dir, "phantom_default.rs", PHANTOM_DEFAULT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const FROM_INTO_WRAPPERS_SRC: &str = r#"
struct Outer { outer: std::cell::Cell<i32> }
fn main() {
    let o = Outer { outer: 1.into() };
    o.outer.set(0);
    println!("{}", o.outer.get());
    let c = std::cell::Cell::from(5);
    println!("{}", c.get());
    let rc: std::cell::RefCell<i32> = 7.into();
    println!("{}", *rc.borrow());
    let rf = std::cell::RefCell::from(9);
    println!("{}", *rf.borrow());
}
"#;

#[test]
fn run_mode_from_into_cell_and_refcell() {
    // The shape of tests/ui/contracts/
    // contracts-requires-is-not-inherited-when-nesting.rs: a
    // `Cell` field built with `1.into()`. `Into::into`'s instance
    // arguments are [Self, T] with T as the destination; reading
    // them in `From`'s order returned the bare integer here.
    let dir = temp_dir("from-into-wrappers");
    let src = fixture(&dir, "case.rs", FROM_INTO_WRAPPERS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cell/RefCell From and Into conversions must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "0\n5\n7\n9\n");
}

// ----- monomorphization validity checks (Lane DX) ---------------------------

const ATOMIC_BOOL_SRC: &str = r#"
#![feature(core_intrinsics)]
use std::intrinsics::{self, AtomicOrdering::*};
fn main() {
    let mut b = true;
    unsafe { let _ = intrinsics::atomic_load::<_, { SeqCst }, false>(&mut b); }
}
"#;

const ATOMIC_VALID_SRC: &str = r#"
#![feature(core_intrinsics)]
use std::intrinsics::{self, AtomicOrdering::*};
fn main() {
    let v = 41u32;
    unsafe {
        let _ = intrinsics::atomic_load::<_, { SeqCst }, false>(&v);
        let mut n = 1u64;
        let _ = intrinsics::atomic_xadd::<_, _, { SeqCst }>(&mut n, 2u64);
        let mut p = &v as *const u32;
        let _ = intrinsics::atomic_xadd::<_, _, { SeqCst }>(&mut p, 1usize);
    }
}
"#;

const SIMD_SATURATING_FLOAT_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_saturating_add;
#[repr(simd)]
#[derive(Copy, Clone)]
struct F32x4([f32; 4]);
fn main() { unsafe { let _ = simd_saturating_add(F32x4([0.0; 4]), F32x4([0.0; 4])); } }
"#;

const SIMD_BITMASK_BAD_RET_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_bitmask;
#[repr(simd)]
#[derive(Copy, Clone)]
struct M2([i32; 2]);
fn main() { unsafe { let _: u16 = simd_bitmask(M2([0, 0])); } }
"#;

const SIMD_BITMASK_OK_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_bitmask;
#[repr(simd)]
#[derive(Copy, Clone)]
struct M16([i32; 16]);
fn main() { unsafe { let _: u16 = simd_bitmask(M16([0; 16])); } }
"#;

const SIMD_SHUFFLE_OOB_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_shuffle;
#[repr(simd)]
#[derive(Copy, Clone)]
struct F32x2([f32; 2]);
#[repr(simd)]
#[derive(Copy, Clone)]
struct Idx2([u32; 2]);
const IDX: Idx2 = Idx2([4, 0]);
fn main() { unsafe { let _: F32x2 = simd_shuffle(F32x2([0.0, 0.0]), F32x2([1.0, 1.0]), IDX); } }
"#;

const SIMD_SHUFFLE_OK_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_shuffle;
#[repr(simd)]
#[derive(Copy, Clone)]
struct F32x2([f32; 2]);
#[repr(simd)]
#[derive(Copy, Clone)]
struct Idx2([u32; 2]);
const IDX: Idx2 = Idx2([3, 0]);
fn main() { unsafe { let _: F32x2 = simd_shuffle(F32x2([0.0, 0.0]), F32x2([1.0, 1.0]), IDX); } }
"#;

const SIMD_REDUCE_BAD_RET_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_reduce_and;
#[repr(simd)]
#[derive(Copy, Clone)]
struct U32x4([u32; 4]);
fn main() { unsafe { let _: f32 = simd_reduce_and(U32x4([0; 4])); } }
"#;

const SIMD_MASKED_LOAD_BAD_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::{SimdAlign, simd_masked_load};
#[repr(simd)]
#[derive(Copy, Clone)]
struct Simd<T, const N: usize>([T; N]);
fn main() {
    let arr = [4u8, 5, 6, 7];
    let default = Simd::<u8, 4>([9; 4]);
    unsafe {
        let _ = simd_masked_load::<_, _, _, { SimdAlign::Element }>(
            Simd::<i8, 8>([-1, 0, -1, -1, 0, 0, 0, 0]),
            arr.as_ptr(),
            default,
        );
    }
}
"#;

const SIMD_GATHER_BAD_PTR_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_gather;
#[repr(simd)]
#[derive(Copy, Clone)]
struct Simd<T, const N: usize>([T; N]);
fn main() {
    let default = Simd::<f32, 4>([0.0; 4]);
    let mask = Simd::<i32, 4>([-1, -1, -1, -1]);
    unsafe { let _ = simd_gather(default, mask, mask); }
}
"#;

#[test]
fn convert_mode_atomic_intrinsic_with_bool_value_is_a_compile_failure() {
    let dir = temp_dir("mono-atomic-bool");
    let src = fixture(&dir, "atomic_bool.rs", ATOMIC_BOOL_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "an atomic load of a bool must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("atomic_load")
            && stderr.contains("basic integer or pointer"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_atomic_intrinsics_with_valid_types_convert() {
    let dir = temp_dir("mono-atomic-valid");
    let src = fixture(&dir, "atomic_valid.rs", ATOMIC_VALID_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "valid atomic monomorphizations must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_saturating_add_on_floats_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-saturating");
    let src = fixture(&dir, "saturating.rs", SIMD_SATURATING_FLOAT_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "saturating arithmetic on a float vector must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_saturating_add") && stderr.contains("integer type"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_bitmask_with_wrong_return_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-bitmask-bad");
    let src = fixture(&dir, "bitmask_bad.rs", SIMD_BITMASK_BAD_RET_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a bitmask returned as the wrong integer width must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_bitmask") && stderr.contains("cannot return"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_bitmask_with_matching_return_converts() {
    let dir = temp_dir("mono-simd-bitmask-ok");
    let src = fixture(&dir, "bitmask_ok.rs", SIMD_BITMASK_OK_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a bitmask with the matching return width must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_shuffle_with_out_of_bounds_index_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-shuffle-oob");
    let src = fixture(&dir, "shuffle_oob.rs", SIMD_SHUFFLE_OOB_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a shuffle index past both inputs must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_shuffle") && stderr.contains("less than"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_shuffle_with_in_bounds_index_converts() {
    let dir = temp_dir("mono-simd-shuffle-ok");
    let src = fixture(&dir, "shuffle_ok.rs", SIMD_SHUFFLE_OK_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an in-bounds shuffle must convert; stderr={stderr}"
    );
}

const SIMD_INSERT_OOB_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_insert;
#[repr(simd)]
#[derive(Copy, Clone)]
struct U8x2([u8; 2]);
fn main() { unsafe { simd_insert(U8x2([0, 0]), 2, 0u8); } }
"#;

const SIMD_EXTRACT_OOB_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_extract;
#[repr(simd)]
#[derive(Copy, Clone)]
struct U8x2([u8; 2]);
fn main() { unsafe { let _x: u8 = simd_extract(U8x2([0, 0]), 5); } }
"#;

const SIMD_INSERT_EXTRACT_OK_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::{simd_extract, simd_insert};
#[repr(simd)]
#[derive(Copy, Clone)]
struct U8x2([u8; 2]);
fn main() {
    unsafe {
        let v = simd_insert(U8x2([0, 0]), 1, 7u8);
        let _x: u8 = simd_extract(v, 1);
    }
}
"#;

#[test]
fn convert_mode_simd_insert_with_out_of_bounds_index_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-insert-oob");
    let src = fixture(&dir, "insert_oob.rs", SIMD_INSERT_OOB_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "an insert index past the vector must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_insert") && stderr.contains("less than"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_extract_with_out_of_bounds_index_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-extract-oob");
    let src = fixture(&dir, "extract_oob.rs", SIMD_EXTRACT_OOB_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "an extract index past the vector must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_extract") && stderr.contains("less than"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_insert_extract_with_in_bounds_index_converts() {
    let dir = temp_dir("mono-simd-insert-extract-ok");
    let src = fixture(&dir, "insert_extract_ok.rs", SIMD_INSERT_EXTRACT_OK_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "in-bounds insert/extract must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_reduce_with_wrong_return_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-reduce-bad");
    let src = fixture(&dir, "reduce_bad.rs", SIMD_REDUCE_BAD_RET_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a reduction returned as a non-element type must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_reduce_and") && stderr.contains("expected return type"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_masked_load_with_wrong_length_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-masked-bad");
    let src = fixture(&dir, "masked_bad.rs", SIMD_MASKED_LOAD_BAD_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a masked load whose values length differs from the mask must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_masked_load") && stderr.contains("third argument"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_gather_with_non_pointer_vector_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-gather-bad");
    let src = fixture(&dir, "gather_bad.rs", SIMD_GATHER_BAD_PTR_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a gather whose second argument is not a pointer vector must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_gather") && stderr.contains("pointer"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}


// ----- statics holding slice references (Issue #140) ---------------------------

const STATIC_MUT_SLICE_SRC: &str = r#"
#![allow(static_mut_refs)]

static mut TEST: &'static mut [isize] = &mut [1];
static mut EMPTY: &'static mut [isize] = &mut [];
static mut INT: &'static mut isize = &mut 1;
static mut TEST_RAW: *mut [isize] = &mut [1isize] as *mut _;
static mut INT_RAW: *mut isize = &mut 1isize as *mut _;

pub fn main() {
    unsafe {
        TEST[0] += 1;
        assert_eq!(TEST[0], 2);
        assert_eq!(EMPTY.len(), 0);
        *INT += 1;
        assert_eq!(*INT, 2);
        (*TEST_RAW)[0] += 1;
        assert_eq!((*TEST_RAW)[0], 2);
        *INT_RAW += 1;
        assert_eq!(*INT_RAW, 2);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_static_mut_slice_references_write_through() {
    // A mutable slice reference stored in a static denotes the
    // promoted array's own storage (a fat reference over the
    // static reference), so a write through it is observed by
    // later reads of the static instead of landing in a detached
    // snapshot of the elements.
    let dir = temp_dir("static-mut-slice");
    let src = fixture(&dir, "static_mut_slice.rs", STATIC_MUT_SLICE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "writes through a static's slice reference must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const STATIC_SLICE_ALIAS_SRC: &str = r#"
use std::sync::atomic::{AtomicIsize, Ordering};

static A: [AtomicIsize; 1] = [AtomicIsize::new(7)];
static TEST: &[AtomicIsize] = &A;

pub fn main() {
    TEST[0].store(9, Ordering::SeqCst);
    assert_eq!(A[0].load(Ordering::SeqCst), 9);
    println!("ok");
}
"#;

#[test]
fn run_mode_static_slice_alias_observes_the_named_static() {
    // A slice reference to a whole named array static is the
    // same storage as the static itself: an interior write
    // through the alias is read back through the static.
    let dir = temp_dir("static-slice-alias");
    let src = fixture(&dir, "static_slice_alias.rs", STATIC_SLICE_ALIAS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a slice alias of an array static must share its storage; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
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

// ---- Lane II: process model (issue #144) ----
// The process model never starts a child process. Spawning a
// program that cannot be resolved answers the io::Error the OS
// would produce (shape and kind included, through the guest
// bitpacked representation and the modelled OS error table); a
// program that does resolve is a named unsupported error, never
// a silent success.

const SPAWN_NONEXISTENT_SRC: &str = r#"
use std::io::ErrorKind;
use std::process::Command;
fn main() {
    let e = Command::new("definitely-not-a-real-program-xyz").spawn().unwrap_err();
    println!("kind={:?}", e.kind());
    println!("raw={:?}", e.raw_os_error());
    assert_eq!(e.kind(), ErrorKind::NotFound);
    println!("spawn-nonexistent-ok");
}
"#;

#[test]
fn run_mode_process_spawn_nonexistent_error_kind() {
    let dir = temp_dir("proc-spawn-nonexistent");
    let src = fixture(&dir, "spawn_nonexistent.rs", SPAWN_NONEXISTENT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "kind=NotFound\nraw=Some(2)\nspawn-nonexistent-ok\n"
    );
}

const SPAWN_ERROR_SHAPES_SRC: &str = r#"
use std::io::ErrorKind;
use std::process::Command;
fn main() {
    let dir = Command::new("/tmp").spawn().unwrap_err().kind();
    assert_eq!(dir, ErrorKind::PermissionDenied);
    let nul = Command::new("whatever").arg("a\0b").spawn().unwrap_err().kind();
    assert_eq!(nul, ErrorKind::InvalidInput);
    let cwd = Command::new("whatever")
        .current_dir("/no/such/dir-xyz")
        .spawn()
        .unwrap_err()
        .kind();
    assert_eq!(cwd, ErrorKind::NotFound);
    println!("error-shapes-ok");
}
"#;

#[test]
fn run_mode_process_spawn_error_shapes() {
    let dir = temp_dir("proc-spawn-shapes");
    let src = fixture(&dir, "spawn_shapes.rs", SPAWN_ERROR_SHAPES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "error-shapes-ok\n");
}

const CURRENT_EXE_SRC: &str = r#"
fn main() {
    let p = std::env::current_exe().unwrap();
    let q = std::env::current_exe().unwrap();
    assert!(p == q);
    assert!(std::process::id() > 0);
    println!("current-exe-ok");
}
"#;

#[test]
fn run_mode_process_current_exe_and_id() {
    let dir = temp_dir("proc-current-exe");
    let src = fixture(&dir, "current_exe.rs", CURRENT_EXE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "current-exe-ok\n");
}

const SPAWN_RESOLVABLE_SRC: &str = r#"
fn main() {
    let _ = std::process::Command::new("ls").spawn();
    println!("must not reach here");
}
"#;

#[test]
fn run_mode_process_spawn_resolvable_is_named_error() {
    let dir = temp_dir("proc-spawn-resolvable");
    let src = fixture(&dir, "spawn_resolvable.rs", SPAWN_RESOLVABLE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("does not start child processes"),
        "stderr={stderr}"
    );
}

const FFI_HELPERS_SRC: &str = r#"
#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    fn rust_dbg_extern_identity_u32(x: u32) -> u32;
    fn rust_get_test_int() -> isize;
    fn rust_dbg_extern_return_TwoU16s() -> TwoU16s;
    fn rust_int8_to_int32(x: i8) -> i32;
    fn rust_interesting_average(n: u64, ...) -> f64;
    fn identity(x: u128) -> u128;
    fn square(x: i128) -> i128;
    fn sub(a: i128, b: i128) -> i128;
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TwoU16s { one: u16, two: u16 }

fn main() {
    unsafe {
        assert_eq!(rust_dbg_extern_identity_u32(0xDEADBEEF), 0xDEADBEEFu32);
        assert_eq!(rust_get_test_int(), 1);
        let t = rust_dbg_extern_return_TwoU16s();
        assert_eq!(t.one, 10);
        assert_eq!(t.two, 20);
        assert_eq!(rust_int8_to_int32(-1), -1);
        assert_eq!(rust_interesting_average(2, 10i64, 10.0f64, 20i64, 20.0f64) as i64, 30);
        assert!(rust_interesting_average(0).is_nan());
        let b: u128 = 0x33EE_0E2A_54E2_59DA_A0E7_8E41;
        assert_eq!(identity(b), b);
        assert_eq!(square(0x734C_C2F2_A521i128) as u128, b);
        assert_eq!(
            sub(
                0x2468_ACF1_3579_BDFF_DB97_530E_CA86_420i128,
                0x1234_5678_9ABC_DEFF_EDCB_A987_6543_210i128
            ),
            0x1234_5678_9ABC_DEFF_EDCB_A987_6543_210i128
        );
    }
    println!("ffi helpers ok");
}
"#;

#[test]
fn run_mode_ffi_helpers_execute() {
    // The `rust_test_helpers` C functions the rustc UI tests
    // link for ABI checks are body-less extern declarations;
    // the interpreter serves them from the foreign-helper
    // models (their semantics are the C source's).
    let dir = temp_dir("ffi-helpers");
    let src = fixture(&dir, "ffi_helpers.rs", FFI_HELPERS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "ffi helper calls must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "ffi helpers ok\n",
        "the guest's stdout is the driver's stdout"
    );
}

const FFI_STRUCT_SRC: &str = r#"
#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    fn rust_dbg_call(cb: extern "C" fn(u64) -> u64, data: u64) -> u64;
    fn get_x(s: S) -> u64;
    fn get_y(s: S) -> u64;
    fn get_z(s: S) -> u64;
    fn rust_dbg_abi_1(q: Quad) -> Quad;
    fn get_c_many_params(a: *const (), b: *const (), c: *const (), d: *const (), f: Quad) -> u64;
    fn issue_97463_leak_uninit_data(a: u32, b: u32, c: u32) -> u16;
    fn increment_all_parts(li: LARGE_INTEGER) -> LARGE_INTEGER;
    fn rust_dbg_new_some_u64u64(a: u64, b: u64) -> U8TaggedEnumOptionU64U64;
    fn rust_dbg_new_none_u64u64() -> U8TaggedEnumOptionU64U64;
    fn rust_dbg_unpack_option_u64u64(o: U8TaggedEnumOptionU64U64, a: *mut u64, b: *mut u64) -> i32;
}

extern "C" fn cb(data: u64) -> u64 { data + 1 }

#[derive(Copy, Clone)]
#[repr(C)]
pub struct S { x: u64, y: u64, z: u64 }

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Quad { a: u64, b: u64, c: u64, d: u64 }

#[derive(Clone, Copy)]
#[repr(C)]
struct LARGE_INTEGER_U { LowPart: u32, HighPart: u32 }

#[derive(Clone, Copy)]
#[repr(C)]
union LARGE_INTEGER {
    __unnamed__: LARGE_INTEGER_U,
    u: LARGE_INTEGER_U,
    QuadPart: u64,
}

#[repr(C, u8)]
pub enum U8TaggedEnumOptionU64U64 {
    None,
    Some(u64, u64),
}

fn indirect(func: unsafe extern "C" fn(s: S) -> u64, s: S) -> u64 { unsafe { func(s) } }

fn main() {
    unsafe {
        assert_eq!(rust_dbg_call(cb, 41), 42);
        let s = S { x: 1, y: 2, z: 3 };
        assert_eq!(indirect(get_x, s), 1);
        assert_eq!(indirect(get_y, s), 2);
        assert_eq!(indirect(get_z, s), 3);
        let q = Quad { a: 10, b: 20, c: 30, d: 40 };
        let qq = rust_dbg_abi_1(q);
        assert_eq!(qq.a, 31);
        assert_eq!(qq.b, 39);
        assert_eq!(qq.c, 11);
        assert_eq!(qq.d, 19);
        let null = std::ptr::null();
        assert_eq!(get_c_many_params(null, null, null, null, q), 30);
        assert_eq!(issue_97463_leak_uninit_data(0, 0x327b23c6, 0), 0x23c6u16);
        let li = LARGE_INTEGER { QuadPart: 0 };
        let li_c = increment_all_parts(li);
        assert_eq!(li_c.QuadPart, 0x0000_0002_0000_0003u64);
        let some = rust_dbg_new_some_u64u64(10, 20);
        if let U8TaggedEnumOptionU64U64::Some(a, b) = some {
            assert_eq!(a, 10);
            assert_eq!(b, 20);
        } else {
            panic!("unexpected none");
        }
        let mut a: u64 = 0;
        let mut b: u64 = 0;
        let r = rust_dbg_unpack_option_u64u64(some, &mut a as *mut _, &mut b as *mut _);
        assert_eq!(r, 1);
        assert_eq!(a, 10);
        assert_eq!(b, 20);
        let none = rust_dbg_new_none_u64u64();
        let r = rust_dbg_unpack_option_u64u64(none, &mut a as *mut _, &mut b as *mut _);
        assert_eq!(r, 0);
    }
    println!("ffi struct helpers ok");
}
"#;

#[test]
fn run_mode_ffi_struct_and_callback_helpers_execute() {
    // Callback invocation, field-returning helpers, the ABI
    // shuffle, the union increment and the tagged-option
    // constructor/unpack pair, all body-less extern
    // declarations served by the foreign-helper models.
    let dir = temp_dir("ffi-struct");
    let src = fixture(&dir, "ffi_struct.rs", FFI_STRUCT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "ffi struct helper calls must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "ffi struct helpers ok\n",
        "the guest's stdout is the driver's stdout"
    );
}
const BOX_CELL_SRC: &str = r#"
use std::cell::Cell;
fn main() {
    let b = Box::new(Cell::new(1u32));
    b.set(5);
    println!("{}", b.get());
    std::mem::forget(b);
}
"#;

#[test]
fn run_mode_box_cell_set_through_deref() {
    // Method resolution reaches the `Cell` through the `Deref`
    // builtin for `Box`. The builtin used to hand back a snapshot
    // of the pointee value, so `Cell::set` (a write through the
    // reference) failed with "expected mutable place, found
    // constant aggregate". A box is uniquely owned, so the builtin
    // now returns a reference to the place the pointee lives in.
    // (The box is forgotten because dropping a `Box<Cell<u32>>` is
    // a separate drop-glue gap.)
    let dir = temp_dir("box-cell");
    let src = fixture(&dir, "box_cell.rs", BOX_CELL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cell::set through a Box deref must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "5\n");
}

const RC_REFCELL_SRC: &str = r#"
use std::cell::RefCell;
use std::rc::Rc;
fn main() {
    let rc = Rc::new(RefCell::new(1u32));
    *rc.borrow_mut() = 7;
    let v = *rc.borrow();
    println!("{}", v);
    std::mem::forget(rc);
}
"#;

#[test]
fn run_mode_rc_refcell_borrow_mut_assignment() {
    // `RefCell::borrow_mut` is reached through the `Deref`
    // builtin for `Rc`; with a snapshot result the RefCell borrow
    // could not be tracked ("RefCell with no live place (constant
    // aggregate)"). The handle's pointee lives in a place in this
    // value model, so the builtin returns that place and the
    // borrow registers.
    let dir = temp_dir("rc-refcell");
    let src = fixture(&dir, "rc_refcell.rs", RC_REFCELL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "borrow_mut assignment through Rc<RefCell> must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "7\n");
}

const VEC_CELL_SRC: &str = r#"
use std::cell::Cell;
fn main() {
    let v = vec![Cell::new(0u32), Cell::new(0u32)];
    v[1].set(9);
    println!("{} {}", v[0].get(), v[1].get());
    std::mem::forget(v);
}
"#;

#[test]
fn run_mode_vec_cell_index_set() {
    // `v[1].set(..)` reaches the element through the `Index`
    // builtin; with a snapshot element the `Cell::set` write
    // failed with "expected mutable place, found constant
    // aggregate". When the indexed value lives in a place, the
    // builtin returns a reference to the element place.
    let dir = temp_dir("vec-cell");
    let src = fixture(&dir, "vec_cell.rs", VEC_CELL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cell::set through a Vec index must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "0 9\n");
}
const PANIC_BOX_HOLDER_SRC: &str = r#"
use std::cell::Cell;
use std::ops::Deref;
use std::panic;
struct Holder {
    c: Cell<u32>,
}
fn main() {
    let r = panic::catch_unwind(|| {
        panic::panic_any(Holder { c: Cell::new(1u32) });
    });
    let err = match r {
        Ok(()) => panic!("no panic"),
        Err(e) => e,
    };
    let b = match err.downcast::<Holder>() {
        Ok(b) => b,
        Err(_) => panic!("downcast failed"),
    };
    let hr: &Holder = Deref::deref(&b);
    hr.c.set(5);
    println!("{}", b.c.get());
    std::mem::forget(b);
}
"#;

#[test]
fn run_mode_panic_box_holder_cell_set_through_deref() {
    // A panic payload's by-value downcast hands back its payload
    // in an aggregate-shaped box (the payload value as the box's
    // single field), not an allocation-backed one. The box lives
    // in a place, so the `Deref` builtin's result must denote
    // that place's field 0: with a snapshot of the payload, the
    // `Cell::set` write through the deref cannot reach the box's
    // storage.
    let dir = temp_dir("panic-box-holder");
    let src = fixture(&dir, "panic_box_holder.rs", PANIC_BOX_HOLDER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cell::set through a panic-payload Box deref must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "5\n");
}

const PANIC_BOX_STRING_SRC: &str = r#"
use std::ops::DerefMut;
use std::panic;
fn main() {
    let r = panic::catch_unwind(|| {
        panic::panic_any(String::from("a"));
    });
    let err = match r {
        Ok(()) => panic!("no panic"),
        Err(e) => e,
    };
    let mut b = match err.downcast::<String>() {
        Ok(b) => b,
        Err(_) => panic!("downcast failed"),
    };
    let s: &mut String = DerefMut::deref_mut(&mut b);
    s.push_str("z");
    println!("{}", b.as_str());
    std::mem::forget(b);
}
"#;

#[test]
fn run_mode_panic_box_string_push_through_deref_mut() {
    // The mutable counterpart: an explicit `DerefMut` call on an
    // aggregate-shaped box in a place must yield a reference to
    // that place's field 0, so the write through the result lands
    // in the box instead of failing on the snapshot the builtin
    // used to hand back ("expected mutable place, found constant
    // aggregate").
    let dir = temp_dir("panic-box-string");
    let src = fixture(&dir, "panic_box_string.rs", PANIC_BOX_STRING_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "push_str through a panic-payload Box deref_mut must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "az\n");
}

// ----- monomorphization validity checks, deep rules (Lane HZ) ---------------

const SIMD_SHUFFLE_ELEM_MISMATCH_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_shuffle;
#[repr(simd)]
#[derive(Copy, Clone)]
struct U32x4([u32; 4]);
#[repr(simd)]
#[derive(Copy, Clone)]
struct Idx2([u32; 2]);
const IDX: Idx2 = Idx2([0, 1]);
fn main() {
    let v = U32x4([0; 4]);
    unsafe { let _: U32x4 = simd_shuffle(v, v, IDX); }
}
"#;

const SIMD_SHUFFLE_VALID_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_shuffle;
#[repr(simd)]
#[derive(Copy, Clone)]
struct U32x4([u32; 4]);
#[repr(simd)]
#[derive(Copy, Clone)]
struct Idx4([u32; 4]);
const IDX: Idx4 = Idx4([0, 1, 2, 3]);
fn main() {
    let v = U32x4([0; 4]);
    unsafe { let _: U32x4 = simd_shuffle(v, v, IDX); }
}
"#;

const SIMD_BITMASK_WIDTH_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_bitmask;
#[repr(simd)]
#[derive(Copy, Clone)]
struct I8x2([i8; 2]);
fn main() {
    let m = I8x2([-1, 0]);
    unsafe { let _: u16 = simd_bitmask(m); }
}
"#;

const MASKED_STORE_CONST_PTR_SRC: &str = r#"
#![feature(portable_simd, core_intrinsics)]
use std::intrinsics::simd::SimdAlign;
use std::{ptr::null, simd::Simd};
const _: () = {
    let c = Simd::from_array([0; 3]);
    unsafe {
        core::intrinsics::simd::simd_masked_store::<_, _, _, { SimdAlign::Element }>(
            c,
            null::<i32>(),
            c,
        )
    };
};
fn main() {}
"#;

#[test]
fn convert_mode_simd_shuffle_with_mismatched_index_length_is_a_compile_failure() {
    let dir = temp_dir("mono-shuffle-len");
    let src = fixture(&dir, "shuffle_len.rs", SIMD_SHUFFLE_ELEM_MISMATCH_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a shuffle whose index length differs from the return length must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("invalid monomorphization of `simd_shuffle` intrinsic"),
        "the failure must be the monomorphization check, not a front-end error; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_shuffle_with_matching_index_converts() {
    let dir = temp_dir("mono-shuffle-ok");
    let src = fixture(&dir, "shuffle_ok.rs", SIMD_SHUFFLE_VALID_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a valid shuffle must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_bitmask_with_wrong_width_is_a_compile_failure() {
    let dir = temp_dir("mono-bitmask-width");
    let src = fixture(&dir, "bitmask_width.rs", SIMD_BITMASK_WIDTH_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a bitmask result wider than the lane count requires must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_bitmask"),
        "the failure must name the intrinsic; stderr={stderr}"
    );
}

#[test]
fn convert_mode_masked_store_with_const_pointer_in_const_converts() {
    let dir = temp_dir("mono-masked-const");
    let src = fixture(&dir, "masked_const.rs", MASKED_STORE_CONST_PTR_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a const-evaluated masked store through a *const pointer must convert; stderr={stderr}"
    );
}


const SIMD_REDUCE_BAD_INITIAL_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_reduce_add_ordered;
#[repr(simd)]
#[derive(Copy, Clone)]
struct F32x4([f32; 4]);
fn reduce<U>(v: F32x4, init: U) -> U { unsafe { simd_reduce_add_ordered(v, init) } }
fn main() { let _ = reduce(F32x4([0.0; 4]), 0i32); }
"#;

#[test]
fn convert_mode_simd_reduce_ordered_with_mismatched_initial_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-reduce-initial");
    let src = fixture(&dir, "reduce_initial.rs", SIMD_REDUCE_BAD_INITIAL_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "an ordered reduction whose initial value is not the element type must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_reduce_add_ordered") && stderr.contains("initial value"),
        "the failure must name the intrinsic and the initial value; stderr={stderr}"
    );
}

// ----- stdout-mismatch bucket: TLS destructors, Debug wrapper
// ----- forms, SIMD cast, libtest failure sections (Lane IM) -----

const TLS_DTOR_SRC: &str = r#"
use std::thread;

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        println!("test2");
    }
}

thread_local!(static FOO: Foo = Foo);

fn main() {
    thread::spawn(|| {
        FOO.with(|_| {});
        println!("test1");
    })
    .join()
    .unwrap();
}
"#;

#[test]
fn run_mode_tls_destructor_runs_at_spawned_thread_exit() {
    // The shape of tests/ui/threads-sendsync/issue-29488.rs: a
    // thread-local's destructor runs when the thread that
    // registered it finishes, so the spawned thread's `Foo` is
    // dropped (printing test2) after its own output (test1).
    run_ok("tls-dtor", TLS_DTOR_SRC, "test1\ntest2\n");
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

const SIMD_CAST_SRC: &str = r#"
#![feature(repr_simd, core_intrinsics)]
#[repr(simd)]
#[derive(Debug, Copy, Clone)]
struct Simd<T, const N: usize>([T; N]);
fn main() {
    let a: Simd<i8, 3> = Simd([10, -3, -9]);
    let b: Simd<i16, 3> = unsafe { std::intrinsics::simd::simd_cast(a) };
    println!("{:?}", b);
}
"#;

#[test]
fn run_mode_simd_cast_sign_extends() {
    // `simd_cast` from a signed element type sign-extends from
    // the source width (the shape of tests/ui/simd/
    // issue-39720.rs): -3 and -9 stay negative as i16 lanes.
    run_ok("simd-cast", SIMD_CAST_SRC, "Simd([10, -3, -9])\n");
}

const LIBTEST_MISMATCH_SRC: &str = r#"
#[test]
#[should_panic(expected = "foo")]
fn bar_test() { panic!("bar"); }
"#;

#[test]
fn run_mode_libtest_should_panic_mismatch_section_has_report() {
    // A should-panic message mismatch fails with the panic
    // report ahead of the `did not contain` note in the
    // failure section, as libtest prints it.
    let dir = temp_dir("libtest-mismatch");
    let src = fixture(&dir, "mismatch.rs", LIBTEST_MISMATCH_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "--test"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    let report = stdout.find("panicked at").expect("panic report in section");
    let note = stdout
        .find("note: panic did not contain expected string")
        .expect("mismatch note in section");
    assert!(report < note, "report precedes the note: stdout={stdout}");
    assert!(
        stdout.contains("panic message: \"bar\""),
        "stdout={stdout}"
    );
}

const WINDOWS_DISPLACED_SRC: &str = r#"
pub trait Iter {
    type Item<'a> where Self: 'a;
    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>>;
    fn for_each<F>(mut self, mut f: F)
    where Self: Sized, F: for<'a> FnMut(Self::Item<'a>)
    {
        while let Some(item) = self.next() {
            f(item);
        }
    }
}
pub struct Windows<T> {
    items: Vec<T>,
    start: usize,
    len: usize,
}
impl<T> Windows<T> {
    pub fn new(items: Vec<T>, len: usize) -> Self {
        Self { items, start: 0, len }
    }
}
impl<T> Iter for Windows<T> {
    type Item<'a> = &'a mut [T] where T: 'a;
    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>> {
        let slice = self.items.get_mut(self.start..self.start + self.len)?;
        self.start += 1;
        Some(slice)
    }
}
fn main() {
    Windows::new(vec![1, 2, 3, 4, 5], 3)
        .for_each(|slice| {
            println!("{:?}", slice);
            assert_eq!(slice.len(), 3);
            let mut sum = 0;
            for x in slice.iter() {
                sum += x;
            }
            println!("sum={}", sum);
        });
    println!("ok");
}
"#;

#[test]
fn run_mode_windows_displaced_slice_reads_elements() {
    // A slice formed by displacing a container pointer by a whole
    // number of elements (the `windows` shape: a range index into
    // the middle of a Vec) denotes the elements from that position
    // on: its length, its iterated elements and its Debug rendering
    // all see the displaced window, not just the first element.
    let dir = temp_dir("windows-displaced");
    let src = fixture(&dir, "windows_displaced.rs", WINDOWS_DISPLACED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "displaced window slices must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 2, 3]\nsum=6\n[2, 3, 4]\nsum=9\n[3, 4, 5]\nsum=12\nok\n"
    );
}

const CHUNKS_WINDOWS_LEN_SRC: &str = r#"
use std::iter::ExactSizeIterator;
fn main() {
    let v = [1, 2, 3, 4, 5];
    println!("c_rem {}", v.chunks(2).len());
    println!("c_exact {}", v.chunks(5).len());
    println!("c_big {}", v.chunks(9).len());
    println!("c_one {}", v.chunks(1).len());
    println!("w_rem {}", v.windows(3).len());
    println!("w_exact {}", v.windows(5).len());
    println!("w_big {}", v.windows(6).len());
    println!("w_one {}", v.windows(1).len());
    let e: [i32; 0] = [];
    println!("c_empty {}", e.chunks(2).len());
    println!("w_empty {}", e.windows(1).len());
    println!("ce_empty {}", e.chunks_exact(2).len());
    println!("hint_c {:?}", v.chunks(2).size_hint());
    println!("hint_w {:?}", v.windows(3).size_hint());
    println!("hint_ce {:?}", v.chunks_exact(2).size_hint());
    let mut w = v.windows(2);
    let _ = w.next();
    println!("w_after1 {}", w.len());
    let _ = w.next();
    let _ = w.next();
    let _ = w.next();
    println!("w_after4 {}", w.len());
    let mut c = v.chunks(2);
    let rc = &mut c;
    println!("refmut_len {}", rc.len());
    let c2 = v.chunks(3);
    let r2 = &c2;
    println!("ref_len {}", r2.len());
    println!("ce_rem {:?}", v.chunks_exact(2).remainder());
    println!("rce_rem {:?}", v.rchunks_exact(2).remainder());
    let vec = vec![10, 20, 30, 40, 50, 60, 70];
    println!("vec_c {}", vec.chunks(3).len());
    println!("vec_w {}", vec.windows(4).len());
    let z = [(), (), (), ()];
    println!("zst_c {}", z.chunks(3).len());
    println!("zst_w {}", z.windows(2).len());
    let s = "hello".as_bytes();
    println!("bytes_c {}", s.chunks(2).len());
}
"#;

#[test]
fn run_mode_chunks_windows_len() {
    // `ExactSizeIterator::len` for the slice chunk/window
    // family: the trait default body asserts `size_hint`'s
    // upper bound equals its lower bound, and the modelled
    // `size_hint` answered the trait default `(0, None)` for
    // these types, so every `len()` call died in core's
    // `assert_eq!`. The modelled `size_hint` now computes
    // std's exact counts (ceiling division by the chunk size,
    // floor for the exact variants, `remaining - size + 1`
    // for windows), and the outputs below are native
    // rustc 1.100.0's.
    let dir = temp_dir("chunks-windows-len");
    let src = fixture(&dir, "chunks_windows_len.rs", CHUNKS_WINDOWS_LEN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "chunks/windows len must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c_rem 3\nc_exact 1\nc_big 1\nc_one 5\nw_rem 3\nw_exact 1\nw_big 0\nw_one 5\nc_empty 0\nw_empty 0\nce_empty 0\nhint_c (3, Some(3))\nhint_w (3, Some(3))\nhint_ce (2, Some(2))\nw_after1 3\nw_after4 0\nrefmut_len 3\nref_len 2\nce_rem [5]\nrce_rem [1]\nvec_c 3\nvec_w 4\nzst_c 2\nzst_w 3\nbytes_c 3\n"
    );
}

const CHUNKS_MUT_LEN_SRC: &str = r#"
use std::iter::ExactSizeIterator;
fn main() {
    let v = [1, 2, 3, 4, 5];
    println!("ce {}", v.chunks_exact(2).len());
    println!("rce {}", v.rchunks_exact(2).len());
    println!("rc {}", v.rchunks(2).len());
    let mut m = [1, 2, 3, 4, 5];
    println!("cm {}", m.chunks_mut(2).len());
    let mut m2 = [1, 2, 3, 4, 5];
    println!("cem {}", m2.chunks_exact_mut(2).len());
    let mut m3 = [1, 2, 3, 4, 5];
    println!("rcm {}", m3.rchunks_mut(2).len());
    let mut m4 = [1, 2, 3, 4, 5];
    println!("rcem {}", m4.rchunks_exact_mut(2).len());
}
"#;

#[test]
fn run_mode_chunks_mut_and_exact_len() {
    // The mutable and exact variants of the chunk family
    // store the remaining slice differently (a raw fat
    // pointer plus a marker field for the `Mut` shapes), but
    // `len` is the same count family: ceiling division for
    // the plain variants, floor division for the exact ones.
    let dir = temp_dir("chunks-mut-len");
    let src = fixture(&dir, "chunks_mut_len.rs", CHUNKS_MUT_LEN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "chunks mut/exact len must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "ce 2\nrce 2\nrc 3\ncm 3\ncem 2\nrcm 3\nrcem 2\n"
    );
}

const UFCS_FALLBACK_SRC: &str = r#"
use std::iter::FromIterator;

fn collect_into<I: IntoIterator<Item = i32>>(iter: I) -> Vec<i32> {
    <Vec<i32> as FromIterator<i32>>::from_iter(iter)
}

fn main() {
    let v = collect_into(Some(5).into_iter());
    assert_eq!(v.len(), 1);
    assert_eq!(v[0], 5);
    let add: fn(i32, i32) -> i32 = <i32 as std::ops::Add>::add;
    assert_eq!(add(5, 6), 11);
    let sub: fn(u32, u32) -> u32 = <u32 as std::ops::Sub>::sub;
    assert_eq!(sub(10, 4), 6);
    let neg: fn(i32) -> i32 = <i32 as std::ops::Neg>::neg;
    assert_eq!(neg(7), -7);
    let not: fn(bool) -> bool = <bool as std::ops::Not>::not;
    assert_eq!(not(true), false);
    println!("ok");
}
"#;

#[test]
fn run_mode_ufcs_impl_qualified_fallbacks() {
    // Calls whose names conversion kept in impl-qualified
    // generic form have no body and no plain trait-path lookup
    // entry: the run-time name fallback serves the ones whose
    // semantics are value-level — `FromIterator::from_iter` for
    // a Vec target (the name carries the target; the iterator
    // state steps by its shape) and the primitive arithmetic
    // traits (the name carries the Self type whose width the
    // operation computes at).
    let dir = temp_dir("ufcs-fallback");
    let src = fixture(&dir, "ufcs_fallback.rs", UFCS_FALLBACK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "impl-qualified UFCS calls must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const SCALAR_BYTE_COPY_SRC: &str = r#"
fn main() {
    // A bulk byte copy through a `*mut u8` cast of a scalar's
    // address writes the scalar's storage byte by byte (the
    // pattern `core::hash::Hasher::write` uses to assemble its
    // next word in a `u64` buffer), and a copy back out reads
    // the same bytes.
    let mut buf: u64 = 0;
    let msg: &[u8] = &[1, 2, 3];
    unsafe {
        let p = &mut buf as *mut u64 as *mut u8;
        std::ptr::copy_nonoverlapping(msg.as_ptr(), p, 3);
    }
    assert_eq!(buf, 0x030201);
    let mut out = [0u8; 2];
    unsafe {
        let p = &buf as *const u64 as *const u8;
        std::ptr::copy_nonoverlapping(p.add(1), out.as_mut_ptr(), 2);
    }
    assert_eq!(out, [2, 3]);
    println!("ok {buf}");
}
"#;

#[test]
fn run_mode_byte_copy_through_scalar_storage() {
    let dir = temp_dir("scalar-byte-copy");
    let src = fixture(&dir, "scalar_byte_copy.rs", SCALAR_BYTE_COPY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok 197121\n");
}

const C_MAIN_SRC: &str = r#"
#![no_std]
#![no_main]

extern crate std as other;

#[no_mangle]
extern "C" fn main(argc: core::ffi::c_int, argv: *const *const u8) -> core::ffi::c_int {
    // The modelled entry arguments: the program path alone,
    // NUL-terminated, followed by a null terminator slot.
    if argc != 1 || argv.is_null() {
        return 3;
    }
    unsafe {
        let path = *argv;
        if path.is_null() || *path == 0 || !(*argv.add(1)).is_null() {
            return 4;
        }
    }
    0
}
"#;

#[test]
fn run_mode_c_convention_main_gets_argc_argv() {
    let dir = temp_dir("c-main");
    let src = fixture(&dir, "c_main.rs", C_MAIN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
}

const HAND_BUILT_WAKER_SRC: &str = r#"
use std::sync::Arc;
use std::task::{RawWaker, RawWakerVTable, Waker};

struct Wake;

unsafe fn clone_raw(data: *const ()) -> RawWaker {
    let arc = Arc::from_raw(data as *const Wake);
    let cloned = arc.clone();
    let _ = Arc::into_raw(arc);
    let _ = Arc::into_raw(cloned);
    RawWaker::new(data, &VTABLE)
}

unsafe fn wake_raw(data: *const ()) {
    let arc = Arc::from_raw(data as *const Wake);
    drop(arc);
}

unsafe fn wake_by_ref_raw(_data: *const ()) {}

unsafe fn drop_raw(data: *const ()) {
    let arc = Arc::from_raw(data as *const Wake);
    drop(arc);
}

static VTABLE: RawWakerVTable =
    RawWakerVTable::new(clone_raw, wake_raw, wake_by_ref_raw, drop_raw);

fn main() {
    // A waker built by hand from a raw `Arc` (the `arc_wake`
    // pattern): dropping it releases the handle the data
    // pointer denotes, so the weak handle observes the strong
    // count reaching zero.
    let arc = Arc::new(Wake);
    let weak = Arc::downgrade(&arc);
    let raw = Arc::into_raw(arc) as *const ();
    let waker = unsafe { Waker::from_raw(RawWaker::new(raw, &VTABLE)) };
    assert!(weak.upgrade().is_some());
    drop(waker);
    println!("released={}", weak.upgrade().is_none());
}
"#;

#[test]
fn run_mode_hand_built_waker_drop_releases_arc() {
    let dir = temp_dir("hand-built-waker");
    let src = fixture(&dir, "hand_built_waker.rs", HAND_BUILT_WAKER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "released=true\n");
}

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

const UNSIZE_RC_WRAPPERS_SRC: &str = r#"
use std::cell::RefCell;
use std::rc::Rc;
struct Foo<T: ?Sized>(T);
fn main() {
    let rc: Rc<Foo<[u8]>> = Rc::new(Foo([7u8, 8, 9]));
    println!("foo {} {}", rc.0.len(), rc.0[2]);
    let a: Rc<RefCell<[isize; 3]>> = Rc::new(RefCell::new([10, 20, 30]));
    let b: Rc<RefCell<[isize]>> = a;
    println!("cell {} {}", b.borrow().len(), b.borrow()[1]);
    let s: std::sync::Arc<[i32]> = std::sync::Arc::new([4, 5]);
    let w = std::sync::Arc::downgrade(&s);
    println!("arc {} weak {}", s.len(), w.upgrade().map(|x| x.len()).unwrap_or(0));
}
"#;

#[test]
fn run_mode_unsize_rc_wrapped_and_refcell_slices() {
    run_ok(
        "unsize-rc-wrappers",
        UNSIZE_RC_WRAPPERS_SRC,
        "foo 3 9\ncell 3 20\narc 2 weak 2\n",
    );
}

const UNSIZE_CUSTOM_PTR_SRC: &str = r#"
#![feature(coerce_unsized, unsize)]
use std::marker::Unsize;
use std::ops::CoerceUnsized;
struct MyPtr<T: ?Sized> { p: *mut T, tag: u32 }
impl<T: ?Sized, U: ?Sized> CoerceUnsized<MyPtr<U>> for MyPtr<T> where T: Unsize<U> {}
fn main() {
    let mut arr = [10u8, 20, 30];
    let a: MyPtr<[u8; 3]> = MyPtr { p: &mut arr, tag: 7 };
    let b: MyPtr<[u8]> = a;
    println!("tag={}", b.tag);
    let s: &[u8] = unsafe { &*b.p };
    println!("len={} s1={}", s.len(), s[1]);
}
"#;

#[test]
fn run_mode_unsize_custom_pointer_multi_field() {
    run_ok("unsize-custom-ptr", UNSIZE_CUSTOM_PTR_SRC, "tag=7\nlen=3 s1=20\n");
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

// ---- Lane IW: Skip / Chain adapters over modelled iterators (issue #158) ----
// The `next` / `into_iter` / `size_hint` builtins step the
// adapter states directly, and the adapters drop glue is

// ---- Lane IW: Skip / Chain adapters over modelled iterators (issue #158) ----
// The `next` / `into_iter` / `size_hint` builtins step the
// adapter states directly, and the adapters' drop glue is
// accounted like the other modelled adapters': a `Skip` or
// `Chain` over the argument iterator runs end to end, and
// dropping one (a statement temporary, an explicit
// `mem::drop`) is a faithful no-op for the string elements.

const SKIP_CHAIN_SRC: &str = r#"
use std::env;
fn main() {
    let mut it = (0..6).skip(2);
    println!("{} {} {}", it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
    let mut short = (0..2).skip(5);
    println!("{:?} {:?}", short.next(), short.next());
    let hinted = (0..10).skip(3);
    println!("hint {:?}", hinted.size_hint());
    let rest: Vec<String> = env::args().skip(1).collect();
    println!("rest {:?}", rest);
    println!("skip-next {:?}", env::args().skip(2).next());
    let mut ch = (0..2).chain(5..7);
    println!("{} {} {} {} {:?}", ch.next().unwrap(), ch.next().unwrap(), ch.next().unwrap(), ch.next().unwrap(), ch.next());
    println!("chain-hint {:?}", (0..3).chain(0..4).size_hint());
    let all: Vec<i32> = (0..2).chain(9..11).collect();
    println!("all {:?}", all);
    println!("doubled {}", env::args().chain(env::args()).count());
    std::mem::drop(env::args().skip(1));
    std::mem::drop((0..3).chain(4..6));
    println!("done");
}
"#;

#[test]
fn run_mode_skip_and_chain_adapters_over_modelled_iterators() {
    let dir = temp_dir("skip-chain");
    let src = fixture(&dir, "skip_chain.rs", SKIP_CHAIN_SRC);
    let out = run_driver_extra(&src, &["--guest-arg", "one", "--guest-arg", "two"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "2 3 4\nNone None\nhint (7, Some(7))\nrest [\"one\", \"two\"]\nskip-next Some(\"two\")\n0 1 5 6 None\nchain-hint (7, Some(7))\nall [0, 1, 9, 10]\ndoubled 6\ndone\n"
    );
}

// ---- Lane IW: one-payload wrapper transmutes (issue #158) ----
// A one-field struct (or a `MaybeUninit`) transmuted to
// exactly its payload type reads the payload field out, and
// the reverse transmute wraps the value back up -- the
// `assume_init` shape an inlined `mem::zeroed` performs
// under optimization.

const TRANSMUTE_PAYLOAD_WRAPPER_SRC: &str = r#"
use std::mem::{ManuallyDrop, MaybeUninit};
struct Wrap(u32);
fn main() {
    let mu = MaybeUninit::new([1u32, 2, 3, 4]);
    let arr: [u32; 4] = unsafe { std::mem::transmute(mu) };
    println!("arr {:?}", arr);
    let mu2: MaybeUninit<[u32; 4]> = unsafe { std::mem::transmute(arr) };
    let arr2 = unsafe { mu2.assume_init() };
    println!("arr2 {:?}", arr2);
    let w = Wrap(41);
    let x: u32 = unsafe { std::mem::transmute(w) };
    let w2: Wrap = unsafe { std::mem::transmute(x + 1) };
    println!("wrap {} {}", x, w2.0);
    let md = ManuallyDrop::new(vec![5, 6, 7]);
    let v: Vec<i32> = unsafe { std::mem::transmute(md) };
    println!("vec {:?}", v);
    let md2: ManuallyDrop<Vec<i32>> = unsafe { std::mem::transmute(v) };
    println!("md {}", md2.len());
}
"#;

#[test]
fn run_mode_transmute_one_payload_wrapper_round_trips() {
    run_ok(
        "transmute-payload-wrapper",
        TRANSMUTE_PAYLOAD_WRAPPER_SRC,
        "arr [1, 2, 3, 4]\narr2 [1, 2, 3, 4]\nwrap 41 42\nvec [5, 6, 7]\nmd 3\n",
    );
}

// ---- Lane IW: thread spawn under optimization (issue #158) ----
// With `-O`, `mem::zeroed` inside the spawned closure is
// inlined into a `write_bytes` plus a `MaybeUninit` ->
// array `Transmute` cast in the closure body itself; the
// spawn / join path must carry it.

const SPAWN_ZEROED_OPT_SRC: &str = r#"
use std::thread;
fn main() {
    let h = thread::spawn(|| {
        let a: [u8; 65536] = unsafe { std::mem::zeroed() };
        a[0] as i32 + a[65535] as i32
    });
    std::process::exit(h.join().unwrap());
}
"#;

#[test]
fn run_mode_thread_spawn_zeroed_array_under_optimization() {
    // The joined sum of two zeroed bytes is the exit code:
    // reaching it at all proves the closure's `write_bytes`
    // plus `Transmute` executed and the handle joined.
    let dir = temp_dir("spawn-zeroed-opt");
    let src = fixture(&dir, "spawn_zeroed.rs", SPAWN_ZEROED_OPT_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-O"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "");
}

const DYN_ANY_GENERIC_PAYLOADS_SRC: &str = r#"
use std::any::Any;
use std::cell::{Cell, RefCell};
fn main() {
    let b: Box<dyn Any> = Box::new(Cell::new(1u32));
    println!("cell is: {}", b.is::<Cell<u32>>());
    match b.downcast::<Cell<u32>>() {
        Ok(c) => println!("cell: {}", c.get()),
        Err(_) => println!("cell FAILED"),
    }
    let b: Box<dyn Any> = Box::new(vec![1u32, 2, 3]);
    println!("vec is: {}", b.is::<Vec<u32>>());
    match b.downcast::<Vec<u32>>() {
        Ok(v) => println!("vec len: {}", v.len()),
        Err(_) => println!("vec FAILED"),
    }
    let b: Box<dyn Any> = Box::new(RefCell::new(4u32));
    println!("refcell is: {}", b.is::<RefCell<u32>>());
    match b.downcast::<RefCell<u32>>() {
        Ok(c) => println!("refcell: {}", c.into_inner()),
        Err(_) => println!("refcell FAILED"),
    }
}
"#;

#[test]
fn run_mode_dyn_any_downcast_generic_payloads() {
    // The concrete tag a `dyn Any` value carries is the fully
    // rendered type, generic arguments included — the same
    // rendering a downcast's generic argument carries. Payloads
    // whose types have generic arguments (Cell, Vec, RefCell)
    // therefore answer `is` and `downcast` like argument-free
    // payloads do.
    run_ok(
        "dyn-any-generic",
        DYN_ANY_GENERIC_PAYLOADS_SRC,
        "cell is: true\ncell: 1\nvec is: true\nvec len: 3\nrefcell is: true\nrefcell: 4\n",
    );
}

const DYN_ANY_DOWNCAST_WRITE_SRC: &str = r#"
use std::any::Any;
use std::cell::Cell;
use std::ops::{Deref, DerefMut};
fn main() {
    let b: Box<dyn Any> = Box::new(String::from("a"));
    let mut b = match b.downcast::<String>() {
        Ok(x) => x,
        Err(_) => panic!("downcast failed"),
    };
    let s: &mut String = DerefMut::deref_mut(&mut b);
    s.push_str("z");
    println!("{}", b.as_str());
    let b: Box<dyn Any> = Box::new(Cell::new(1u32));
    let b = match b.downcast::<Cell<u32>>() {
        Ok(x) => x,
        Err(_) => panic!("downcast failed"),
    };
    let r: &Cell<u32> = Deref::deref(&b);
    r.set(5);
    println!("{}", b.get());
}
"#;

#[test]
fn run_mode_dyn_any_downcast_constructed_box_writes_through_deref() {
    // A by-value downcast of a constructed `Box<dyn Any>` hands
    // back the box itself (the dyn value's base), not a box
    // wrapped around it: writes through the result's `Deref` /
    // `DerefMut` reach the payload, as they do for a box that was
    // never erased.
    run_ok("dyn-any-write", DYN_ANY_DOWNCAST_WRITE_SRC, "az\n5\n");
}

const DYN_ANY_DOWNCAST_REF_SRC: &str = r#"
use std::any::Any;
fn main() {
    let b: Box<dyn Any> = Box::new(vec![10u32, 20, 30]);
    match b.downcast_ref::<Vec<u32>>() {
        Some(v) => println!("ref len {} first {}", v.len(), v[0]),
        None => println!("ref FAILED"),
    }
}
"#;

#[test]
fn run_mode_dyn_any_downcast_ref_denotes_the_payload() {
    // `downcast_ref` denotes the payload the box points at, not
    // the box aggregate itself (which would report the box's own
    // one-field shape, e.g. a length of 1 for a 3-element Vec).
    run_ok("dyn-any-ref", DYN_ANY_DOWNCAST_REF_SRC, "ref len 3 first 10\n");
}

const PANIC_ANY_GENERIC_PAYLOAD_SRC: &str = r#"
use std::any::Any;
use std::cell::Cell;
use std::panic;
fn main() {
    let r = panic::catch_unwind(|| {
        panic::panic_any(Cell::new(7u32));
    });
    let err = match r {
        Ok(()) => panic!("no panic"),
        Err(e) => e,
    };
    println!("is: {}", err.is::<Cell<u32>>());
    match err.downcast_ref::<Cell<u32>>() {
        Some(c) => println!("ref: {}", c.get()),
        None => println!("ref FAILED"),
    }
    match err.downcast::<Cell<u32>>() {
        Ok(c) => println!("downcast: {}", c.get()),
        Err(_) => println!("downcast FAILED"),
    }
}
"#;

#[test]
fn run_mode_panic_any_generic_payload_downcast() {
    // A fabricated payload box (a caught panic's payload) has the
    // same box shape as a constructed one, so the full
    // is / downcast_ref / downcast surface works on a payload
    // whose type carries generic arguments.
    run_ok(
        "panic-any-generic",
        PANIC_ANY_GENERIC_PAYLOAD_SRC,
        "is: true\nref: 7\ndowncast: 7\n",
    );
}

const RESUME_UNWIND_GENERIC_PAYLOAD_SRC: &str = r#"
use std::any::Any;
use std::cell::Cell;
use std::panic;
fn main() {
    let r = panic::catch_unwind(|| {
        panic::panic_any(Cell::new(3u32));
    });
    let err = match r {
        Ok(()) => panic!("no panic"),
        Err(e) => e,
    };
    let r2 = panic::catch_unwind(panic::AssertUnwindSafe(move || {
        panic::resume_unwind(err);
    }));
    let err2 = match r2 {
        Ok(()) => panic!("no panic"),
        Err(e) => e,
    };
    match err2.downcast::<Cell<u32>>() {
        Ok(c) => println!("resumed: {}", c.get()),
        Err(_) => println!("resumed FAILED"),
    }
}
"#;

#[test]
fn run_mode_resume_unwind_generic_payload_downcast() {
    // A resumed panic re-raises the payload itself (the dyn box's
    // pointee), so the payload survives the round trip through
    // resume_unwind and a second catch with its type and value
    // intact.
    run_ok(
        "resume-generic",
        RESUME_UNWIND_GENERIC_PAYLOAD_SRC,
        "resumed: 3\n",
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

const PATH_TO_STR_SRC: &str = r#"
use std::path::Path;

fn main() {
    // `Path::to_str` goes through `str::from_utf8` on the
    // path's bytes.
    let p = Path::new("abc");
    assert_eq!(p.to_str().unwrap(), "abc");
    let ok = std::str::from_utf8(b"abc").unwrap();
    assert_eq!(ok, "abc");
    let bad = std::str::from_utf8(&[0x61, 0xFF, 0x62]).unwrap_err();
    assert_eq!(bad.valid_up_to(), 1);
    assert_eq!(bad.error_len(), Some(1));
    let incomplete = std::str::from_utf8(&[0x61, 0xE3, 0x81]).unwrap_err();
    assert_eq!(incomplete.valid_up_to(), 1);
    assert_eq!(incomplete.error_len(), None);
    println!("path-to-str-ok");
}
"#;

#[test]
fn run_mode_path_to_str_and_from_utf8() {
    let dir = temp_dir("path-to-str");
    let src = fixture(&dir, "path_to_str.rs", PATH_TO_STR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Path::to_str and str::from_utf8 run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "path-to-str-ok\n",
        "utf8 validation results and error fields match std"
    );
}

const PATH_DISPLAY_FMT_SRC: &str = r#"
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

fn main() {
    // The `fmt` bodies of the path / OS-string types are
    // foreign code, so their text comes from the byte
    // content the values hold: `Display` adapters forward
    // to the bytes (lossy), `Debug` is the `OsStr` form,
    // and neither is the aggregate expansion.
    let p = Path::new("/tmp/foo bar.txt");
    println!("{}", p.display());
    println!("{:?}", p);
    println!("{:?}", p.display());
    let pb = PathBuf::from("/tmp/foo bar.txt");
    println!("{}", pb.display());
    println!("{:?}", pb);
    let os = OsStr::new("hello");
    println!("{}", os.display());
    println!("{:?}", os);
    let osn = OsString::from("hello");
    println!("{}", osn.display());
    println!("{:?}", osn);
    println!("{:>24}", p.display());
    println!("{:>24?}", pb);
    println!("{}", p.parent().unwrap().display());
    println!("{}", p.to_string_lossy());
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let bad = OsStr::from_bytes(b"/tmp/a\xFFb\x80c");
        println!("{}", bad.display());
        println!("{:?}", bad);
    }
    let w = std::num::Wrapping(42u32);
    println!("{}", w);
    println!("{:x}", w);
    let r = std::cmp::Reverse(7);
    println!("{:?}", r);
}
"#;

#[test]
fn run_mode_path_display_and_os_str_formatting() {
    let dir = temp_dir("path-display-fmt");
    let src = fixture(&dir, "path_display_fmt.rs", PATH_DISPLAY_FMT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Path/OsStr Display and Debug formatting runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "/tmp/foo bar.txt\n\"/tmp/foo bar.txt\"\n\"/tmp/foo bar.txt\"\n/tmp/foo bar.txt\n\"/tmp/foo bar.txt\"\nhello\n\"hello\"\nhello\n\"hello\"\n        /tmp/foo bar.txt\n\"/tmp/foo bar.txt\"\n/tmp\n/tmp/foo bar.txt\n/tmp/a\u{FFFD}b\u{FFFD}c\n\"/tmp/a\\xFFb\\x80c\"\n42\n2a\nReverse(7)\n",
        "path and OS-string formatting matches std byte for byte"
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

const PATH_DISPLAY_PRECISION_SRC: &str = r#"
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

fn main() {
    // Precision and width on the `display()` adapters: the
    // adapters' `Display` text is the lossy content rendered
    // under the caller's options, so precision truncates and
    // width pads, exactly as `str` itself formats (measured
    // against native rustc 1.100.0).
    let os = OsStr::new("hello");
    println!("[{:.1}]", os.display());
    println!("[{:.0}]", os.display());
    println!("[{:>10.1}]", os.display());
    println!("[{:>10}]", os.display());
    let p = Path::new("hello");
    println!("[{:.1}]", p.display());
    println!("[{:>10.1}]", p.display());
    println!("[{:.3}]", PathBuf::from("hello world").display());
    println!("[{:.2}]", OsString::from("hello").display());
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let bad = OsStr::from_bytes(b"a\xFFbcdef");
        println!("[{:.2}]", bad.display());
        println!("[{:>12.2}]", bad.display());
    }
}
"#;

#[test]
fn run_mode_path_display_precision_and_width() {
    let dir = temp_dir("path-display-precision");
    let src = fixture(
        &dir,
        "path_display_precision.rs",
        PATH_DISPLAY_PRECISION_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "display-adapter precision and width runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[h]\n[]\n[         h]\n[     hello]\n[h]\n[         h]\n[hel]\n[he]\n[a\u{FFFD}]\n[          a\u{FFFD}]\n",
        "display adapters truncate by precision and pad by width like str"
    );
}

const PATH_NEW_SRC: &str = r#"
use std::path::Path;

fn main() {
    // `Path::new` reaches the path through raw pointer casts
    // over the string data: the projections through the
    // `Path`/`OsStr` wrappers read the string itself.
    let p = Path::new("abc");
    assert_eq!(p.as_os_str(), "abc");
    assert_eq!(p.as_os_str().len(), 3);
    assert_eq!(p.as_os_str().as_encoded_bytes(), b"abc");
    let o = std::ffi::OsStr::new("xy");
    assert_eq!(o.len(), 2);
    println!("path-new-ok");
}
"#;

#[test]
fn run_mode_path_new_and_os_str_projections() {
    let dir = temp_dir("path-new");
    let src = fixture(&dir, "path_new.rs", PATH_NEW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Path::new over a string literal runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "path-new-ok\n",
        "the path projections read the string data"
    );
}

const PATH_PARTS_SRC: &str = r#"
use std::path::Path;

fn main() {
    let p = Path::new("/foo/bar/baz.txt");
    assert_eq!(p.parent().unwrap(), Path::new("/foo/bar"));
    assert_eq!(p.file_name().unwrap(), "baz.txt");
    assert_eq!(p.extension().unwrap(), "txt");
    assert_eq!(p.file_stem().unwrap(), "baz");
    let root = Path::new("/");
    assert_eq!(root.parent(), None);
    assert_eq!(root.file_name(), None);
    let rel = Path::new("foo");
    assert_eq!(rel.parent().unwrap(), Path::new(""));
    assert_eq!(rel.extension(), None);
    let hidden = Path::new("/a/.gitignore");
    assert_eq!(hidden.file_stem().unwrap(), ".gitignore");
    assert_eq!(hidden.extension(), None);
    println!("path-parts-ok");
}
"#;

#[test]
fn run_mode_path_part_accessors() {
    let dir = temp_dir("path-parts");
    let src = fixture(&dir, "path_parts.rs", PATH_PARTS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the Path parsing accessors run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "path-parts-ok\n",
        "parent/file_name/file_stem/extension match std"
    );
}

const PATHBUF_DEREF_SRC: &str = r#"
fn main() {
    // `PathBuf` dereferences to `Path`: the methods below are
    // `Path` methods reached through the `Deref` impl.
    let p = std::env::current_exe().unwrap();
    assert!(p.as_os_str().len() > 0);
    assert!(p.parent().is_some());
    assert!(p.file_name().is_some());
    println!("pathbuf-deref-ok");
}
"#;

#[test]
fn run_mode_pathbuf_deref_to_path() {
    let dir = temp_dir("pathbuf-deref");
    let src = fixture(&dir, "pathbuf_deref.rs", PATHBUF_DEREF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "PathBuf deref to Path runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "pathbuf-deref-ok\n",
        "the dereferenced path reads the buffer's byte string"
    );
}

const PATH_COMPONENTS_SRC: &str = r#"
use std::path::{Component, Path};

fn main() {
    // The component kinds and the `as_os_str` view of each:
    // `Component::as_os_str`'s body transmutes a string
    // constant through a byte slice to the `OsStr` sys type,
    // and the consumers below run through library bodies.
    let kinds: Vec<u8> = Path::new("/a/../c")
        .components()
        .map(|c| match c {
            Component::Prefix(_) => 0,
            Component::RootDir => 1,
            Component::CurDir => 2,
            Component::ParentDir => 3,
            Component::Normal(_) => 4,
        })
        .collect();
    assert_eq!(kinds, vec![1, 4, 3, 4]);
    let lens: Vec<usize> = Path::new("/../x")
        .components()
        .map(|c| c.as_os_str().len())
        .collect();
    assert_eq!(lens, vec![1, 2, 1]);
    assert_eq!(Path::new("").components().count(), 0);
    assert_eq!(Path::new("./a").components().count(), 2);
    let mut sum = 0usize;
    for c in Path::new("a//b/").components() {
        sum += c.as_os_str().len();
    }
    assert_eq!(sum, 2);
    println!("path-components-ok");
}
"#;

#[test]
fn run_mode_path_components_iteration() {
    let dir = temp_dir("path-components");
    let src = fixture(&dir, "path_components.rs", PATH_COMPONENTS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Path::components iteration runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "path-components-ok\n",
        "component kinds and os_str views match std"
    );
}

const PATH_COMPONENTS_CONSUMERS_SRC: &str = r#"
use std::path::Path;

fn main() {
    // `count` is a library body whose closure drops each
    // component it consumes; the drop glue of the component
    // types is empty (they hold only string slices).
    assert_eq!(Path::new("a/b/c/d").components().count(), 4);
    // Front and back steps interleave over one state.
    let mut it = Path::new("a/b/c/d").components();
    assert_eq!(it.next().unwrap().as_os_str().len(), 1);
    assert_eq!(it.next_back().unwrap().as_os_str().len(), 1);
    assert_eq!(it.count(), 2);
    let mut it = Path::new("a").components();
    assert!(it.next().is_some());
    assert!(it.next_back().is_none());
    assert!(it.next().is_none());
    // `rev` composes the back step; an explicit `drop` of a
    // component accounts its (empty) glue as well.
    let names: Vec<String> = Path::new("a/bb")
        .components()
        .rev()
        .map(|c| c.as_os_str().to_str().unwrap().to_string())
        .collect();
    assert_eq!(names, vec!["bb".to_string(), "a".to_string()]);
    let c = Path::new("/x").components().next().unwrap();
    drop(c);
    println!("path-components-consumers-ok");
}
"#;

#[test]
fn run_mode_path_components_consumers() {
    let dir = temp_dir("path-components-consumers");
    let src = fixture(
        &dir,
        "path_components_consumers.rs",
        PATH_COMPONENTS_CONSUMERS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Components consumers (count/next_back/rev/drop) run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "path-components-consumers-ok\n",
        "front/back steps and consumer drops match std"
    );
}

const PATHBUF_BUILD_SRC: &str = r#"
use std::path::{Path, PathBuf};

fn main() {
    let mut pb = PathBuf::from("/tmp");
    pb.push("a");
    pb.push("b/c");
    assert_eq!(pb.to_str().unwrap(), "/tmp/a/b/c");
    assert!(pb.pop());
    assert_eq!(pb.to_str().unwrap(), "/tmp/a/b");
    // An absolute addition replaces the whole buffer.
    pb.push("/abs");
    assert_eq!(pb.to_str().unwrap(), "/abs");
    // The root and the empty buffer have nothing to pop.
    let mut root = PathBuf::from("/");
    assert!(!root.pop());
    assert_eq!(root.to_str().unwrap(), "/");
    let mut empty = PathBuf::new();
    assert!(!empty.pop());
    empty.push("x");
    assert_eq!(empty.to_str().unwrap(), "x");
    // One-byte buffers dereference to their path as well.
    let one = PathBuf::from(String::from("s"));
    assert_eq!(one.to_str().unwrap(), "s");
    assert_eq!(one.components().count(), 1);
    // `to_path_buf` copies a borrowed path into a buffer.
    let copied: PathBuf = Path::new("q/r").to_path_buf();
    assert_eq!(copied.to_str().unwrap(), "q/r");
    println!("pathbuf-build-ok");
}
"#;

#[test]
fn run_mode_pathbuf_construction_and_mutation() {
    let dir = temp_dir("pathbuf-build");
    let src = fixture(&dir, "pathbuf_build.rs", PATHBUF_BUILD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "PathBuf construction and mutation run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "pathbuf-build-ok\n",
        "from/push/pop/new/to_path_buf match std"
    );
}

// ---- Lane JL: one io::Error representation (issue #173) ----
// Every io::Error producer (the fs layer, the process model,
// From<ErrorKind>) builds the bitpacked `Repr` layout std's own
// converted `Error::kind` / `raw_os_error` bodies read. The fs
// layer previously built an aggregate `Error { repr }` shape
// those bodies cannot consume.

const FS_NUL_KIND_SRC: &str = r#"
use std::io::ErrorKind;
fn main() {
    let e = std::fs::metadata("\0").unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidInput);
    assert_eq!(e.raw_os_error(), None);
    let exe = std::env::current_exe().unwrap();
    let e = std::fs::rename(&exe, "\0").unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidInput);
    let e = std::fs::remove_file("\0").unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidInput);
    println!("fs-nul-kind-ok");
}
"#;

#[test]
fn run_mode_fs_nul_path_error_kind() {
    let dir = temp_dir("fs-nul-kind");
    let src = fixture(&dir, "fs_nul_kind.rs", FS_NUL_KIND_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "fs-nul-kind-ok\n");
}

const FS_OS_ERROR_SRC: &str = r#"
use std::io::ErrorKind;
fn main() {
    let e = std::fs::metadata("/no/such/path-xyz-123").unwrap_err();
    assert_eq!(e.kind(), ErrorKind::NotFound);
    assert_eq!(e.raw_os_error(), Some(2));
    println!("fs-os-error-ok");
}
"#;

#[test]
fn run_mode_fs_os_error_kind_and_raw() {
    let dir = temp_dir("fs-os-error");
    let src = fixture(&dir, "fs_os_error.rs", FS_OS_ERROR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "fs-os-error-ok\n");
}

const ERROR_FROM_KIND_SRC: &str = r#"
use std::io::ErrorKind;
fn main() {
    let e = std::io::Error::from(ErrorKind::PermissionDenied);
    assert_eq!(e.kind(), ErrorKind::PermissionDenied);
    assert_eq!(e.raw_os_error(), None);
    let e: std::io::Error = ErrorKind::Other.into();
    assert_eq!(e.kind(), ErrorKind::Other);
    println!("error-from-kind-ok");
}
"#;

#[test]
fn run_mode_io_error_from_error_kind() {
    let dir = temp_dir("error-from-kind");
    let src = fixture(&dir, "error_from_kind.rs", ERROR_FROM_KIND_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "error-from-kind-ok\n"
    );
}

// ----- uninitialized array constants under optimized MIR (Lane JS) ---------
// At opt-level 2 `[MaybeUninit::uninit(); N]` const-propagates into a
// single `<uninit>` constant of the array type, and an element written
// with `MaybeUninit::new(..)` arrives as a decoded union constant
// carrying both union slots. The converted values must carry the
// array's and the union's shapes, so a dynamic-index write followed by
// a read of the written element observes the payload -- while a
// never-written element's payload stays an error, exactly as in the
// unoptimized form. Verification is by exit code: under -O a plain
// `println!` reaches unrelated fmt machinery gaps on this tree.

const UNINIT_ARRAY_O2_SRC: &str = r#"
use std::mem::MaybeUninit;
fn idx() -> usize { 2 }
fn main() {
    let mut arr: [MaybeUninit<u32>; 4] = [MaybeUninit::uninit(); 4];
    let i = idx();
    arr[i] = MaybeUninit::new(42);
    unsafe {
        let v = arr[i].assume_init();
        std::process::exit(v as i32);
    }
}
"#;

#[test]
fn run_mode_uninit_array_o2_dynamic_index_write_then_read() {
    let dir = temp_dir("uninit-array-o2");
    let src = fixture(&dir, "uninit_array_o2.rs", UNINIT_ARRAY_O2_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(42),
        "the written element's payload must read back; stderr={stderr}"
    );
}

const UNINIT_ARRAY_O2_LOOP_SRC: &str = r#"
use std::mem::MaybeUninit;
fn main() {
    let mut arr: [MaybeUninit<u32>; 4] = [MaybeUninit::uninit(); 4];
    for i in 0..4 {
        arr[i].write(i as u32 * 10);
    }
    unsafe {
        let sum = arr[0].assume_init() + arr[1].assume_init()
            + arr[2].assume_init() + arr[3].assume_init();
        std::process::exit(sum as i32);
    }
}
"#;

#[test]
fn run_mode_uninit_array_o2_loop_write_then_read_all() {
    let dir = temp_dir("uninit-array-o2-loop");
    let src = fixture(&dir, "uninit_array_o2_loop.rs", UNINIT_ARRAY_O2_LOOP_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(60),
        "every written element must read back; stderr={stderr}"
    );
}

const UNINIT_SCALAR_CONST_O2_SRC: &str = r#"
use std::mem::MaybeUninit;
#[inline(never)]
fn read(x: MaybeUninit<u32>) -> u32 { unsafe { x.assume_init() } }
fn main() {
    let x: MaybeUninit<u32> = MaybeUninit::new(42);
    std::process::exit(read(x) as i32);
}
"#;

#[test]
fn run_mode_maybeuninit_scalar_const_o2_reads_payload() {
    let dir = temp_dir("uninit-scalar-o2");
    let src = fixture(&dir, "uninit_scalar_o2.rs", UNINIT_SCALAR_CONST_O2_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(42),
        "a decoded MaybeUninit constant must hold its payload at field 0; stderr={stderr}"
    );
}

const UNINIT_ARRAY_O2_UNWRITTEN_SRC: &str = r#"
use std::mem::MaybeUninit;
fn idx() -> usize { 2 }
fn main() {
    let mut arr: [MaybeUninit<u32>; 4] = [MaybeUninit::uninit(); 4];
    let i = idx();
    arr[i] = MaybeUninit::new(42);
    unsafe {
        let v = arr[1].assume_init();
        std::process::exit(v as i32);
    }
}
"#;

#[test]
fn run_mode_uninit_array_o2_unwritten_element_stays_an_error() {
    let dir = temp_dir("uninit-array-o2-unwritten");
    let src = fixture(&dir, "uninit_array_o2_unwritten.rs", UNINIT_ARRAY_O2_UNWRITTEN_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "reading a never-written element's payload must stay a named error; stderr={stderr}"
    );
}

const TRANSMUTE_OPTION_FIELDLESS_SRC: &str = r#"
#[repr(u8)]
pub enum Size { One = 1, Two = 2, Three = 3 }
fn main() {
    let n: u8 = unsafe { std::mem::transmute::<Option<Size>, u8>(None) };
    let s: u8 = unsafe { std::mem::transmute::<Option<Size>, u8>(Some(Size::Two)) };
    let back: Option<Size> = unsafe { std::mem::transmute::<u8, Option<Size>>(3) };
    println!("{} {} {}", n, s, matches!(back, Some(Size::Three)));
}
"#;

#[test]
fn run_mode_transmute_option_fieldless_enum_niche() {
    run_ok(
        "transmute-option-fieldless",
        TRANSMUTE_OPTION_FIELDLESS_SRC,
        "0 2 true\n",
    );
}

const TRANSMUTE_FN_PTR_RETURN_SRC: &str = r#"
use std::mem;
#[inline(never)]
extern "C" fn foo(x: &isize) -> Option<&isize> { Some(x) }
static FOO: isize = 0xDEADBEE;
pub fn main() {
    unsafe {
        let f: extern "C" fn(&isize) -> &isize =
            mem::transmute(foo as extern "C" fn(&isize) -> Option<&isize>);
        println!("{}", *f(&FOO));
    }
}
"#;

#[test]
fn run_mode_transmute_fn_ptr_return_niche_to_ref() {
    run_ok(
        "transmute-fn-ptr-return",
        TRANSMUTE_FN_PTR_RETURN_SRC,
        "233495534\n",
    );
}

const UNIQUE_DANGLING_SRC: &str = r#"
#![feature(ptr_internals)]
use std::ptr::Unique;
const PTR: *mut u32 = Unique::dangling().as_ptr();
fn main() {
    let p = Unique::<u32>::dangling().as_ptr();
    println!("{}", PTR == p);
}
"#;

#[test]
fn run_mode_unique_dangling_const_and_runtime() {
    run_ok("unique-dangling", UNIQUE_DANGLING_SRC, "true\n");
}

const NONNULL_DANGLING_SRC: &str = r#"
use std::ptr::NonNull;
use std::num::NonZero;
#[inline(never)]
fn dangling() -> NonNull<u32> { NonNull::dangling() }
fn main() {
    let a = dangling();
    let b = NonNull::<u32>::without_provenance(NonZero::new(4usize).unwrap());
    println!("{} {}", a.as_ptr() as usize, b.as_ptr() as usize);
}
"#;

#[test]
fn run_mode_nonnull_dangling_without_provenance() {
    run_ok("nonnull-dangling", NONNULL_DANGLING_SRC, "4 4\n");
}


const SLICE_TO_VEC_SRC: &str = r#"
fn bar(v: &mut [usize]) -> Vec<usize> {
    v.to_vec()
}

fn bip(v: &[usize]) -> Vec<usize> {
    v.to_vec()
}

fn main() {
    // A slice over an array literal.
    let s: &[usize] = &[1, 2, 3, 100];
    let r = s.to_vec();
    assert_eq!(r.len(), 4);
    assert_eq!(r[0], 1);
    assert_eq!(r[3], 100);
    assert_eq!(r, vec![1, 2, 3, 100]);
    // A slice reached by deref coercion of a Vec reference at a
    // call argument, in both the shared and mutable forms.
    let mut the_vec = vec![1, 2, 3, 100];
    assert_eq!(the_vec.clone(), bar(&mut the_vec));
    assert_eq!(the_vec.clone(), bip(&the_vec));
    // Elements are cloned: a slice of owned strings copies.
    let words: &[String] = &[String::from("a"), String::from("bc")];
    let owned = words.to_vec();
    assert_eq!(owned.len(), 2);
    assert_eq!(owned[1], "bc");
    println!("slice-to-vec-ok");
}
"#;

#[test]
fn run_mode_slice_to_vec() {
    let dir = temp_dir("slice-to-vec");
    let src = fixture(&dir, "slice_to_vec.rs", SLICE_TO_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "slice-to-vec-ok\n");
}

const DEREF_COPY_INTO_ALLOC_SRC: &str = r#"
use std::alloc::{alloc, dealloc, Layout};

fn main() {
    let layout = Layout::array::<usize>(4).unwrap();
    let p = unsafe { alloc(layout) as *mut usize };
    assert!(!p.is_null());
    // The destination pointer reaches the copy through a
    // dereference of the fresh block's base pointer, so it
    // denotes the block's first element position while the
    // block still holds no written elements.
    let r: &mut usize = unsafe { &mut *p };
    let q = r as *mut usize;
    unsafe {
        std::ptr::copy_nonoverlapping([10usize, 20, 30].as_ptr(), q, 3);
        assert_eq!(*q, 10);
        assert_eq!(*q.add(1), 20);
        assert_eq!(*q.add(2), 30);
        dealloc(p as *mut u8, layout);
    }
    println!("deref-copy-ok");
}
"#;

#[test]
fn run_mode_copy_nonoverlapping_through_deref_into_fresh_alloc() {
    let dir = temp_dir("deref-copy-alloc");
    let src = fixture(
        &dir,
        "deref_copy_alloc.rs",
        DEREF_COPY_INTO_ALLOC_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "deref-copy-ok\n");
}

// ---------------------------------------------------------------------
// Async execution base (Lane JF, Issue #167): the modelled waker
// machinery - RawWaker vtable function-pointer calls, `Arc::from_raw`
// reconstruction, waker drop through the vtable, and `Future::poll`
// dispatch, static and through a `dyn` receiver.

fn run_ok_edition(tag: &str, src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let src_path = fixture(&dir, "case.rs", src);
    let out = run_driver_extra(&src_path, &["--edition", "2021"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "case {tag} must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        expected_stdout,
        "case {tag} stdout mismatch; stderr={stderr}"
    );
}

const ARC_WAKE_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

static WAKES: AtomicUsize = AtomicUsize::new(0);

trait ArcWake {
    fn wake(self: Arc<Self>);
    fn wake_by_ref(arc_self: &Arc<Self>) { arc_self.clone().wake() }
    fn into_waker(wake: Arc<Self>) -> Waker where Self: Sized {
        let ptr = Arc::into_raw(wake) as *const ();
        unsafe { Waker::from_raw(RawWaker::new(ptr, vtable::<Self>())) }
    }
}

fn vtable<T: ArcWake>() -> &'static RawWakerVTable {
    &RawWakerVTable::new(clone_arc_raw::<T>, wake_arc_raw::<T>, wake_by_ref_arc_raw::<T>, drop_arc_raw::<T>)
}
unsafe fn clone_arc_raw<T: ArcWake>(data: *const ()) -> RawWaker {
    let arc: Arc<T> = Arc::from_raw(data as *const T);
    let c = arc.clone();
    let _ = Arc::into_raw(arc);
    let _ = Arc::into_raw(c);
    RawWaker::new(data, vtable::<T>())
}
unsafe fn wake_arc_raw<T: ArcWake>(data: *const ()) {
    let arc: Arc<T> = Arc::from_raw(data as *const T);
    ArcWake::wake(arc);
}
unsafe fn wake_by_ref_arc_raw<T: ArcWake>(data: *const ()) {
    let arc: Arc<T> = Arc::from_raw(data as *const T);
    ArcWake::wake_by_ref(&arc);
    let _ = Arc::into_raw(arc);
}
unsafe fn drop_arc_raw<T: ArcWake>(data: *const ()) {
    let _: Arc<T> = Arc::from_raw(data as *const T);
}

struct Counter;
impl ArcWake for Counter {
    fn wake(self: Arc<Self>) { WAKES.fetch_add(1, Ordering::SeqCst); }
}

struct YieldOnce(bool);
impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0 { Poll::Ready(()) } else { cx.waker().wake_by_ref(); self.0 = true; Poll::Pending }
    }
}

fn main() {
    let counter = Arc::new(Counter);
    let waker = ArcWake::into_waker(counter);
    let mut cx = Context::from_waker(&waker);
    let mut fut = YieldOnce(false);
    let mut pinned = unsafe { Pin::new_unchecked(&mut fut) };
    assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Pending);
    assert_eq!(WAKES.load(Ordering::SeqCst), 1);
    assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Ready(()));
    // Dropping the waker runs the vtable drop entry; cloning it
    // first runs the clone entry as well.
    let waker2 = waker.clone();
    drop(waker2);
    drop(waker);
    println!("arc-wake-ok");
}
"#;

#[test]
fn run_mode_arc_wake_vtable_waker() {
    // The whole RawWaker vtable surface as function-pointer
    // calls: the vtable entries live inside a constant
    // allocation, the waker is dropped through the vtable drop
    // entry, and the entries rebuild their `Arc` with
    // `Arc::from_raw`.
    run_ok_edition("arc-wake", ARC_WAKE_SRC, "arc-wake-ok\n");
}

const DYN_FUTURE_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

fn raw_waker() -> RawWaker {
    unsafe fn clone(_: *const ()) -> RawWaker { raw_waker() }
    unsafe fn no_op(_: *const ()) {}
    RawWaker::new(std::ptr::null(), &RawWakerVTable::new(clone, no_op, no_op, no_op))
}

async fn me() -> i32 { 41 }

fn main() {
    let waker = unsafe { Waker::from_raw(raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    let mut fut: Pin<Box<dyn Future<Output = i32>>> = Box::pin(me());
    match fut.as_mut().poll(&mut cx) {
        Poll::Ready(v) => assert_eq!(v, 41),
        Poll::Pending => panic!("pending"),
    }
    println!("dyn-future-ok");
}
"#;

#[test]
fn run_mode_dyn_future_poll() {
    // `Future::poll` through a `Pin<Box<dyn Future>>`: the
    // dispatch finds the coroutine body (the compiler's own
    // poll implementation for an async fn) through the vtable
    // demand, with the receiver rebuilt in the callee's
    // `Pin<&mut Self>` shape.
    run_ok_edition("dyn-future", DYN_FUTURE_SRC, "dyn-future-ok\n");
}

const ARC_FROM_RAW_SRC: &str = r#"
use std::sync::Arc;
fn main() {
    let arc = Arc::new(41u32);
    let raw = Arc::into_raw(arc);
    let p2 = unsafe { raw.byte_add(8) };
    let p3 = unsafe { p2.byte_sub(8) };
    let arc2 = unsafe { Arc::from_raw(p3) };
    assert_eq!(*arc2, 41);
    assert_eq!(Arc::strong_count(&arc2), 1);
    println!("from-raw-ok");
}
"#;

#[test]
fn run_mode_arc_from_raw_roundtrip() {
    run_ok_edition("arc-from-raw", ARC_FROM_RAW_SRC, "from-raw-ok\n");
}

const ENUM_WRAPPER_TRANSMUTE_SRC: &str = r#"
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(usize)]
enum AlignEnum { A1 = 1, A2 = 2, A4 = 4, A8 = 8 }
#[repr(transparent)]
#[derive(Clone, Copy)]
struct Wrap(AlignEnum);
fn main() {
    let w: Wrap = unsafe { std::mem::transmute(AlignEnum::A8) };
    let back: AlignEnum = unsafe { std::mem::transmute(w) };
    assert_eq!(back, AlignEnum::A8);
    let n: usize = unsafe { std::mem::transmute(w) };
    assert_eq!(n, 8);
    let w2: Wrap = unsafe { std::mem::transmute(4usize) };
    let back2: AlignEnum = unsafe { std::mem::transmute(w2) };
    assert_eq!(back2, AlignEnum::A4);
    println!("enum-wrapper-ok");
}
"#;

#[test]
fn run_mode_enum_wrapper_transmutes() {
    // The `mem::Alignment` shapes: a fieldless enum and the
    // one-field transparent struct wrapping it transmute in
    // both directions, and an integer transmutes to the
    // wrapper through the wrapped enum's discriminant table
    // (the transmute `Arc::from_raw`'s alignment path uses).
    run_ok_edition(
        "enum-wrapper-transmute",
        ENUM_WRAPPER_TRANSMUTE_SRC,
        "enum-wrapper-ok\n",
    );
}

// ---------------------------------------------------------------------
// Async block state storage (Lane JY, Issue #186): a coroutine's
// upvar storage and its per-state saved storage are distinct even
// where their field indices coincide. An async block keeps its
// captures in the upvar storage and reads them back after a
// suspension whose saved field 0 is the awaitee; an async fn's
// transform copies the capture into the state instead. Both shapes
// (and nesting, and dropping a suspended block future) must run.

const ASYNC_BLOCK_STATE_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

struct Nop;
impl Wake for Nop {
    fn wake(self: Arc<Self>) {}
}

struct YieldOnce(bool);
impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.0 = true;
            Poll::Pending
        }
    }
}

fn async_block(x: u8) -> impl Future<Output = u8> {
    async move {
        YieldOnce(false).await;
        x
    }
}

fn two_awaits(a: u8, b: u8) -> impl Future<Output = u8> {
    async move {
        YieldOnce(false).await;
        let c = a + 1;
        YieldOnce(false).await;
        c + b
    }
}

fn nested(x: u8) -> impl Future<Output = u8> {
    async move {
        let inner = async move {
            YieldOnce(false).await;
            x + 1
        };
        inner.await
    }
}

fn string_upvar() -> impl Future<Output = usize> {
    let s = String::from("hello");
    async move {
        YieldOnce(false).await;
        s.len()
    }
}

fn main() {
    let waker = Waker::from(Arc::new(Nop));
    let mut cx = Context::from_waker(&waker);

    let mut fut = Box::pin(async_block(9));
    assert_eq!(Poll::Pending, fut.as_mut().poll(&mut cx));
    assert_eq!(Poll::Ready(9), fut.as_mut().poll(&mut cx));

    let mut fut = Box::pin(two_awaits(9, 3));
    assert_eq!(Poll::Pending, fut.as_mut().poll(&mut cx));
    assert_eq!(Poll::Pending, fut.as_mut().poll(&mut cx));
    assert_eq!(Poll::Ready(13), fut.as_mut().poll(&mut cx));

    let mut fut = Box::pin(nested(10));
    assert_eq!(Poll::Pending, fut.as_mut().poll(&mut cx));
    assert_eq!(Poll::Ready(11), fut.as_mut().poll(&mut cx));

    // A suspended block future holding a non-Copy capture drops
    // its saved state and the capture.
    let mut fut = Box::pin(string_upvar());
    assert_eq!(Poll::Pending, fut.as_mut().poll(&mut cx));
    drop(fut);
    let mut fut = Box::pin(string_upvar());
    assert_eq!(Poll::Pending, fut.as_mut().poll(&mut cx));
    assert_eq!(Poll::Ready(5), fut.as_mut().poll(&mut cx));

    println!("async-block-state-ok");
}
"#;

#[test]
fn run_mode_async_block_state_storage() {
    run_ok_edition(
        "async-block-state",
        ASYNC_BLOCK_STATE_SRC,
        "async-block-state-ok\n",
    );
}

// ----- Constant pointer `as usize` (Lane KC) -----

const CONST_NULL_PTR_AS_USIZE_SRC: &str = r#"
const P: *mut u8 = core::ptr::null_mut();
fn main() {
    let a = P as usize;
    assert_eq!(a, 0);
}
"#;

/// A null pointer constant cast to an integer: the operand is a
/// constant, so it carries no place type; the constant's own form
/// is the evidence for the `PtrToInt` reading, and the null
/// pointer's address is 0.
#[test]
fn run_mode_const_null_ptr_as_usize() {
    let dir = temp_dir("const-null-ptr-usize");
    let src = fixture(&dir, "case.rs", CONST_NULL_PTR_AS_USIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a null pointer constant must cast to 0; stderr={stderr}"
    );
}

const CONST_ADDR_PTR_AS_USIZE_SRC: &str = r#"
const P: *const u8 = 0x1000 as *const u8;
fn main() {
    let a = P as usize;
    assert_eq!(a, 0x1000);
}
"#;

/// An address-only pointer constant cast to an integer reads back
/// the address it was made from, exactly.
#[test]
fn run_mode_const_addr_ptr_as_usize() {
    let dir = temp_dir("const-addr-ptr-usize");
    let src = fixture(&dir, "case.rs", CONST_ADDR_PTR_AS_USIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an address pointer constant must cast to its address; stderr={stderr}"
    );
}

/// The same reading under optimization, where the constant is
/// folded into the cast's operand position.
#[test]
fn run_mode_const_addr_ptr_as_usize_o2() {
    let dir = temp_dir("const-addr-ptr-usize-o2");
    let src = fixture(&dir, "case.rs", CONST_ADDR_PTR_AS_USIZE_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an address pointer constant must cast to its address under -O; stderr={stderr}"
    );
}

const CONST_STATIC_PTR_AS_USIZE_SRC: &str = r#"
static X: u32 = 42;
const P: *const u32 = &X;
fn main() {
    let a = P as usize;
    assert!(a != 0);
    let b = &X as *const u32 as usize;
    assert_eq!(a, b);
    assert_eq!(unsafe { *P }, 42);
}
"#;

/// A static-reference pointer constant cast to an integer yields
/// the same synthetic address as the place cast of the same
/// static, and the pointer still reads the static's value.
#[test]
fn run_mode_const_static_ptr_as_usize() {
    let dir = temp_dir("const-static-ptr-usize");
    let src = fixture(&dir, "case.rs", CONST_STATIC_PTR_AS_USIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a static pointer constant must cast to the static's address; stderr={stderr}"
    );
}

const CONST_FN_PTR_AS_USIZE_SRC: &str = r#"
fn f() {}
const P: fn() = f;
fn main() {
    let a = P as usize;
    assert!(a != 0);
    P();
}
"#;

/// A function-item pointer constant cast to an integer yields the
/// callable's synthetic address (nonzero), and the constant still
/// calls.
#[test]
fn run_mode_const_fn_ptr_as_usize() {
    let dir = temp_dir("const-fn-ptr-usize");
    let src = fixture(&dir, "case.rs", CONST_FN_PTR_AS_USIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a fn pointer constant must cast to its address; stderr={stderr}"
    );
}

const CONST_REF_AS_USIZE_SRC: &str = r#"
const R: &u32 = &42;
fn main() {
    let a = R as *const u32 as usize;
    assert!(a != 0);
    let b = &*R as *const u32 as usize;
    assert_eq!(a, b);
    assert_eq!(*R, 42);
}
"#;

/// A reference constant cast through a raw pointer to an integer
/// yields the same synthetic address as the place-derived pointer
/// to the same value.
#[test]
fn run_mode_const_ref_as_usize() {
    let dir = temp_dir("const-ref-usize");
    let src = fixture(&dir, "case.rs", CONST_REF_AS_USIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a reference constant must cast to its value's address; stderr={stderr}"
    );
}

const DROP_IN_PLACE_VEC_FORGET_SRC: &str = r#"
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("drop D({})", self.0); } }
fn main() {
    let mut v = vec![D(1), D(2)];
    unsafe { std::ptr::drop_in_place(&mut v) };
    std::mem::forget(v);
    println!("after");
}
"#;

/// `ptr::drop_in_place` on a `Vec` runs through the compiler's
/// drop-glue shim, whose field drops name the `RawVec` buffer:
/// the glue conversion must account for it (the buffer's own
/// glue is deallocation only), and the shim's drop must not
/// kill the caller's place -- `forget` still moves it after.
#[test]
fn run_mode_drop_in_place_vec_then_forget() {
    let dir = temp_dir("drop-in-place-vec-forget");
    let src = fixture(&dir, "case.rs", DROP_IN_PLACE_VEC_FORGET_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "drop D(1)\ndrop D(2)\nafter\n");
}

const DROP_IN_PLACE_NESTED_VEC_SRC: &str = r#"
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("drop D({})", self.0); } }
fn main() {
    let mut v = vec![vec![D(1)], vec![D(2), D(3)]];
    unsafe { std::ptr::drop_in_place(&mut v) };
    std::mem::forget(v);
    println!("after");
}
"#;

/// The nested-`Vec` form of the drop-glue shim path: the outer
/// buffer's field drop and the inner vectors' element glue.
#[test]
fn run_mode_drop_in_place_nested_vec() {
    let dir = temp_dir("drop-in-place-nested-vec");
    let src = fixture(&dir, "case.rs", DROP_IN_PLACE_NESTED_VEC_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "drop D(1)\ndrop D(2)\ndrop D(3)\nafter\n");
}

const DROP_IN_PLACE_RC_FORGET_SRC: &str = r#"
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("drop D({})", self.0); } }
fn main() {
    let mut r = std::rc::Rc::new(D(9));
    unsafe { std::ptr::drop_in_place(&mut r) };
    std::mem::forget(r);
    println!("after");
}
"#;

/// An `Rc` dropped in place through the shim: the pointee's
/// destructor runs once, and the caller's handle stays
/// movable (the drop through the pointer does not clear the
/// owner's slot).
#[test]
fn run_mode_drop_in_place_rc_then_forget() {
    let dir = temp_dir("drop-in-place-rc-forget");
    let src = fixture(&dir, "case.rs", DROP_IN_PLACE_RC_FORGET_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "drop D(9)\nafter\n");
}

const DROP_IN_PLACE_VEC_SCOPE_DROP_SRC: &str = r#"
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("drop D({})", self.0); } }
fn main() {
    let mut v = vec![D(1), D(2)];
    unsafe { std::ptr::drop_in_place(&mut v) };
    println!("after");
}
"#;

/// After `drop_in_place`, the scope-end `Drop` terminator for
/// the same local is unconditional in MIR (drop elaboration
/// cannot see through the raw pointer), so the glue runs on
/// the value as it stands -- the same behaviour arrays and
/// struct fields already had through this path.
#[test]
fn run_mode_drop_in_place_vec_scope_drop_after() {
    let dir = temp_dir("drop-in-place-vec-scope");
    let src = fixture(&dir, "case.rs", DROP_IN_PLACE_VEC_SCOPE_DROP_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "drop D(1)\ndrop D(2)\nafter\ndrop D(1)\ndrop D(2)\n");
}

const ADAPTERS_DROP_BY_VALUE_SRC: &str = r#"
fn sink<T>(_: T) {}
fn main() {
    let a = [1, 2, 3];
    sink(a.iter().cloned());
    sink(a.iter().cycle());
    sink(a.iter().zip([4, 5].iter()));
    sink(a.iter().peekable());
    sink(a.iter().filter(|_| true));
    sink(a.iter().rev());
    sink((1..10).step_by(2));
    sink("x".escape_debug());
    sink("x".escape_default());
    sink("x".escape_unicode());
    println!("done");
}
"#;

/// The field-only iterator adapters and the `str` escape
/// adapters have no `Drop` impl of their own: dropping one by
/// value (here, inside the generic `sink`) plans the glue
/// from the fields' evidence instead of refusing the drop.
#[test]
fn run_mode_iterator_adapters_drop_by_value() {
    let dir = temp_dir("adapters-drop-by-value");
    let src = fixture(&dir, "case.rs", ADAPTERS_DROP_BY_VALUE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "done\n");
}

// ----- uninitialized statics (Lane JZ) --------------------------------------
// A static whose initializer is (partly) uninitialized decodes to a
// bare `Uninit` seed: the storage holds no trace of the declared
// type's structure, so an element write lands in a shapeless value
// and every element read fails as an uninitialized-place read. The
// converted seed carries the declared type's skeleton (the same
// expansion the thread-local table applies to its seeds), so writes
// and reads through the static's storage observe the array's shape.
// Verification is by exit code, the sum of the payloads read back.

const STATIC_UNINIT_ARRAY_SRC: &str = r#"
use std::mem::MaybeUninit;
static mut ARR: [MaybeUninit<u32>; 4] = [MaybeUninit::uninit(); 4];
fn main() {
    unsafe {
        ARR[1] = MaybeUninit::new(42);
        let v = ARR[1].assume_init();
        std::process::exit(v as i32);
    }
}
"#;

#[test]
fn run_mode_static_uninit_array_write_then_read() {
    let dir = temp_dir("static-uninit-array");
    let src = fixture(&dir, "static_uninit_array.rs", STATIC_UNINIT_ARRAY_SRC);
    let out = run_driver_extra(&src, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(42),
        "the written element's payload must read back; stderr={stderr}"
    );
}

const STATIC_UNINIT_ARRAY_LOOP_SRC: &str = r#"
use std::mem::MaybeUninit;
static mut ARR: [MaybeUninit<u32>; 4] = [MaybeUninit::uninit(); 4];
fn main() {
    unsafe {
        let mut i = 0usize;
        while i < 4 {
            ARR[i] = MaybeUninit::new((i * 10) as u32);
            i += 1;
        }
        let mut s = 0u32;
        let mut j = 0usize;
        while j < 4 {
            s += ARR[j].assume_init();
            j += 1;
        }
        std::process::exit(s as i32);
    }
}
"#;

#[test]
fn run_mode_static_uninit_array_dynamic_index_loop() {
    let dir = temp_dir("static-uninit-array-loop");
    let src = fixture(
        &dir,
        "static_uninit_array_loop.rs",
        STATIC_UNINIT_ARRAY_LOOP_SRC,
    );
    let out = run_driver_extra(&src, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(60),
        "every written element must read back; stderr={stderr}"
    );
}

const STATIC_UNINIT_SLICE_SRC: &str = r#"
use std::mem::MaybeUninit;
static mut S: &mut [MaybeUninit<u32>] = &mut [MaybeUninit::uninit(); 3];
fn main() {
    unsafe {
        S[0] = MaybeUninit::new(5);
        S[2] = MaybeUninit::new(7);
        let s = S[0].assume_init() + S[2].assume_init();
        std::process::exit(s as i32);
    }
}
"#;

#[test]
fn run_mode_static_uninit_slice_write_then_read() {
    let dir = temp_dir("static-uninit-slice");
    let src = fixture(&dir, "static_uninit_slice.rs", STATIC_UNINIT_SLICE_SRC);
    let out = run_driver_extra(&src, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(12),
        "elements written through the slice static must read back; stderr={stderr}"
    );
}

const STATIC_UNINIT_SCALAR_AND_FIELD_SRC: &str = r#"
use std::mem::MaybeUninit;
static mut MU: MaybeUninit<u32> = MaybeUninit::uninit();
struct Holder {
    tag: u32,
    buf: [MaybeUninit<u16>; 2],
}
static mut H: Holder = Holder {
    tag: 9,
    buf: [MaybeUninit::uninit(); 2],
};
fn main() {
    unsafe {
        MU.as_mut_ptr().write(77);
        H.buf[1] = MaybeUninit::new(55u16);
        let s = MU.assume_init() + H.tag + H.buf[1].assume_init() as u32;
        std::process::exit(s as i32);
    }
}
"#;

#[test]
fn run_mode_static_uninit_scalar_and_struct_field() {
    let dir = temp_dir("static-uninit-scalar-field");
    let src = fixture(
        &dir,
        "static_uninit_scalar_field.rs",
        STATIC_UNINIT_SCALAR_AND_FIELD_SRC,
    );
    let out = run_driver_extra(&src, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(141),
        "the scalar static and the struct field array must both read back; stderr={stderr}"
    );
}

const RAW_PARTS_REBUILD_SRC: &str = r#"
fn main() {
    let s: &[u8] = &[7, 8, 9];
    let sp = s.as_ptr();
    let sr = unsafe { std::slice::from_raw_parts(sp, 3) };
    println!("rebuild: {} {} {}", sr[0], sr[1], sr[2]);
    let sr2 = unsafe { std::slice::from_raw_parts(sp.add(1), 2) };
    println!("rebuild-off: {} {}", sr2[0], sr2[1]);
    let a: [u8; 4] = [10, 20, 30, 40];
    let ap = a.as_ptr();
    let ar = unsafe { std::slice::from_raw_parts(ap, 4) };
    println!("arr rebuild: {} {} {} {}", ar[0], ar[1], ar[2], ar[3]);
    let mut b: [u8; 4] = [0, 0, 0, 0];
    {
        let p = b.as_mut_ptr();
        let w = unsafe { std::slice::from_raw_parts_mut(p.add(1), 2) };
        w[0] = 7;
        w[1] = 9;
    }
    println!("wback {} {} {} {}", b[0], b[1], b[2], b[3]);
    // A row of a matrix reached through an element pointer is
    // the element itself, not a window over the outer array.
    let m: [[u8; 3]; 2] = [[1, 2, 3], [4, 5, 6]];
    let row: &[u8] = &m[1];
    println!("row {} {} {} len {}", row[0], row[1], row[2], row.len());
}
"#;

#[test]
fn run_mode_raw_parts_rebuilt_slice_reads() {
    run_ok(
        "raw-parts-rebuild",
        RAW_PARTS_REBUILD_SRC,
        "rebuild: 7 8 9\nrebuild-off: 8 9\narr rebuild: 10 20 30 40\nwback 0 7 9 0\nrow 4 5 6 len 3\n",
    );
}

const PATH_COMPONENT_EQ_SRC: &str = r#"
use std::path::Path;
fn e(a: &str, b: &str) -> bool {
    Path::new(a) == Path::new(b)
}
fn main() {
    println!("eq1 {}", e("a//b", "a/b"));
    println!("eq2 {}", e("a/", "a"));
    println!("eq3 {}", e("a/./b", "a/b"));
    println!("eq4 {}", e("./a", "a"));
    println!("eq5 {}", e("/a", "a"));
    println!("eq6 {}", e("", "."));
    println!("eq7 {}", e("../a", "../a"));
    println!("eq8 {}", e("//a", "/a"));
    println!("eq9 {}", e("a/.", "a"));
    println!("eq10 {}", e("a/../b", "a/b"));
    println!("eq11 {}", e("/", "//"));
    println!("eq12 {}", e("a", "a/"));
    println!("cross {}", Path::new("a/b") == "a//b");
    println!("cross2 {}", "a//b" == Path::new("a/b"));
    println!("ne {}", Path::new("a//b") != Path::new("a/b"));
}
"#;

#[test]
fn run_mode_path_component_equality() {
    run_ok(
        "path-component-eq",
        PATH_COMPONENT_EQ_SRC,
        "eq1 true\neq2 true\neq3 true\neq4 false\neq5 false\neq6 false\neq7 true\neq8 true\neq9 true\neq10 false\neq11 true\neq12 true\ncross true\ncross2 true\nne false\n",
    );
}

const PATH_HASH_EQ_SRC: &str = r#"
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
fn hp(p: &str) -> u64 {
    let mut h = DefaultHasher::new();
    Path::new(p).hash(&mut h);
    h.finish()
}
fn main() {
    println!("h1 {}", hp("a//b") == hp("a/b"));
    println!("h2 {}", hp("a/") == hp("a"));
    println!("h3 {}", hp("a/./b") == hp("a/b"));
    println!("h4 {}", hp("a/b") == hp("a/b"));
    println!("h5 {}", hp("/a/b") == hp("/a/b"));
    println!("h6 {}", hp("x") != hp("y"));
    // The key path a hash map exercises: the same key spelled
    // two ways must hash and compare equal.
    let stored = Path::new("some/key");
    let looked_up = Path::new("some//key");
    println!("key {}", stored == looked_up && hp("some/key") == hp("some//key"));
}
"#;

#[test]
fn run_mode_path_hash_eq_consistency() {
    run_ok(
        "path-hash-eq",
        PATH_HASH_EQ_SRC,
        "h1 true\nh2 true\nh3 true\nh4 true\nh5 true\nh6 true\nkey true\n",
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

const FMT_GENERIC_AS_ARRAY_SRC: &str = r#"
use std::fmt;
struct S<T, const N: usize>(pub [T; N]);
impl<T, const N: usize> S<T, N> {
    pub const fn as_array(&self) -> &[T; N] {
        let p: *const Self = self;
        unsafe { &*p.cast::<[T; N]>() }
    }
}
impl<T: fmt::Debug, const N: usize> fmt::Debug for S<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <[T; N] as fmt::Debug>::fmt(self.as_array(), f)
    }
}
fn main() {
    println!("{:?}", S::<i16, 3>([10, -3, -9]));
}
"#;

#[test]
fn run_mode_generic_single_field_pointer_cast_enters_the_field() {
    let dir = temp_dir("fmt-generic-as-array");
    let src = fixture(&dir, "generic_as_array.rs", FMT_GENERIC_AS_ARRAY_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "must run; stdout={stdout}");
    assert_eq!(
        stdout, "[10, -3, -9]\n",
        "a raw-pointer cast from a single-field struct to its \
         field type denotes the field, also from generic code"
    );
}

const FMT_SIMD_CAST_SIGN_SRC: &str = r#"
#![feature(repr_simd, core_intrinsics)]
#[repr(simd)]
#[derive(Copy, Clone)]
struct Simd<T, const N: usize>([T; N]);
impl<T, const N: usize> Simd<T, N> {
    fn as_array(&self) -> &[T; N] {
        let p: *const Self = self;
        unsafe { &*p.cast::<[T; N]>() }
    }
}
fn main() {
    let a = Simd::<i8, 3>([10, -3, -9]);
    let b: Simd<i16, 3> = unsafe { std::intrinsics::simd::simd_cast(a) };
    println!("{:?}", b.as_array());
}
"#;

#[test]
fn run_mode_simd_cast_sign_extends_signed_lanes() {
    let dir = temp_dir("fmt-simd-cast-sign");
    let src = fixture(&dir, "simd_cast_sign.rs", FMT_SIMD_CAST_SIGN_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "must run; stdout={stdout}");
    assert_eq!(
        stdout, "[10, -3, -9]\n",
        "simd_cast from a signed element type sign-extends its lanes"
    );
}

// ----- HashMap / hashbrown raw-table execution (Lane HX) -----

const HASHMAP_BASIC_SRC: &str = r#"
use std::collections::HashMap;
fn main() {
    let mut m: HashMap<i32, i32> = HashMap::new();
    m.insert(1, 2);
    println!("{}", m[&1]);
}
"#;

#[test]
fn run_mode_hashmap_insert_and_index() {
    let dir = temp_dir("hashmap-basic");
    let src = fixture(&dir, "hashmap_basic.rs", HASHMAP_BASIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "2\n");
}

const HASHMAP_RESIZE_SRC: &str = r#"
use std::collections::HashMap;
fn main() {
    let mut m: HashMap<i32, i32> = HashMap::new();
    for i in 0..10 {
        m.insert(i, i * 10);
    }
    println!("len {}", m.len());
    for i in 0..10 {
        assert_eq!(m[&i], i * 10);
    }
    println!("resize ok");
}
"#;

#[test]
fn run_mode_hashmap_resize_rehashes_entries() {
    let dir = temp_dir("hashmap-resize");
    let src = fixture(&dir, "hashmap_resize.rs", HASHMAP_RESIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len 10\nresize ok\n");
}

const HASHMAP_STRING_KEYS_SRC: &str = r#"
use std::collections::HashMap;
fn main() {
    let mut m: HashMap<String, String> = HashMap::new();
    for (k, v) in [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")] {
        m.insert(k.to_string(), v.to_string());
    }
    println!("len {}", m.len());
    println!("got {}", &m["c"]);
}
"#;

#[test]
fn run_mode_hashmap_string_keys_across_resize() {
    let dir = temp_dir("hashmap-string-keys");
    let src = fixture(&dir, "hashmap_string_keys.rs", HASHMAP_STRING_KEYS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len 4\ngot 3\n");
}

const HASHMAP_COW_KEYS_SRC: &str = r#"
use std::borrow::Cow;
use std::collections::HashMap;
fn main() {
    let mut m: HashMap<Cow<str>, i32> = HashMap::new();
    m.insert(Cow::Borrowed("foo"), 42);
    assert_eq!(m.get(&Cow::Owned("foo".to_string())), Some(&42));
    assert_eq!(m.insert(Cow::Owned("foo".to_string()), 43), Some(42));
    println!("cow ok {}", m.len());
}
"#;

#[test]
fn run_mode_hashmap_cow_keys_borrowed_owned_agree() {
    let dir = temp_dir("hashmap-cow-keys");
    let src = fixture(&dir, "hashmap_cow_keys.rs", HASHMAP_COW_KEYS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "cow ok 1\n");
}

const HASHMAP_ITER_SRC: &str = r#"
use std::collections::HashMap;
fn main() {
    let mut m: HashMap<i32, i32> = HashMap::new();
    m.insert(1, 10);
    m.insert(2, 20);
    m.insert(3, 30);
    let mut sum = 0;
    for (k, v) in &m {
        sum += k + v;
    }
    assert_eq!(sum, 66);
    let mut it = m.iter();
    let mut seen = 0;
    for _ in it.by_ref() {
        seen += 1;
        break;
    }
    for _ in it {
        seen += 1;
    }
    assert_eq!(seen, 3);
    println!("iter ok");
}
"#;

#[test]
fn run_mode_hashmap_iter_and_by_ref_restart() {
    let dir = temp_dir("hashmap-iter");
    let src = fixture(&dir, "hashmap_iter.rs", HASHMAP_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "iter ok\n");
}

const HASHSET_ITER_SRC: &str = r#"
use std::collections::HashSet;
fn main() {
    let mut s: HashSet<i32> = HashSet::new();
    s.insert(1);
    s.insert(2);
    let mut n = 0;
    for x in &s {
        n += x;
    }
    assert_eq!(n, 3);
    println!("set ok");
}
"#;

#[test]
fn run_mode_hashset_iter() {
    let dir = temp_dir("hashset-iter");
    let src = fixture(&dir, "hashset_iter.rs", HASHSET_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "set ok\n");
}

const HASHMAP_WITH_CAPACITY_SRC: &str = r#"
use std::collections::HashMap;
fn main() {
    let mut m: HashMap<u8, u8> = HashMap::with_capacity(100);
    m.insert(7, 9);
    println!("{}", m[&7]);
}
"#;

#[test]
fn run_mode_hashmap_with_capacity() {
    let dir = temp_dir("hashmap-capacity");
    let src = fixture(&dir, "hashmap_capacity.rs", HASHMAP_WITH_CAPACITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "9\n");
}

const NONNULL_CAST_DENOMINATION_SRC: &str = r#"
use std::ptr::NonNull;
struct Inner { ctrl: NonNull<u8>, mask: usize }
struct Bucket<T> { ptr: NonNull<T> }
impl Inner {
    fn data_end<T>(&self) -> NonNull<T> { self.ctrl.cast() }
    unsafe fn bucket<T>(&self, index: usize) -> Bucket<T> {
        unsafe { Bucket::from_base_index(self.data_end(), index) }
    }
}
impl<T> Bucket<T> {
    unsafe fn from_base_index(base: NonNull<T>, index: usize) -> Self {
        let ptr = unsafe { base.as_ptr().sub(index) };
        Bucket { ptr: unsafe { NonNull::new_unchecked(ptr) } }
    }
    unsafe fn write(&self, value: T) {
        unsafe { self.ptr.as_ptr().sub(1).write(value) }
    }
}
fn main() {
    unsafe {
        let layout = std::alloc::Layout::from_size_align(64, 8).unwrap();
        let base = std::alloc::alloc(layout);
        let inner = Inner { ctrl: NonNull::new_unchecked(base.add(32)), mask: 3 };
        let b: Bucket<(i32, i32)> = inner.bucket(0);
        b.write((7i32, 8i32));
        let back = base.add(24) as *const (i32, i32);
        let v = back.read();
        assert_eq!(v, (7, 8));
        std::alloc::dealloc(base, layout);
    }
    println!("denomination ok");
}
"#;

#[test]
fn run_mode_nonnull_cast_denominates_pointer_units() {
    let dir = temp_dir("nonnull-cast-units");
    let src = fixture(&dir, "nonnull_cast_units.rs", NONNULL_CAST_DENOMINATION_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "denomination ok\n");
}

const SWAP_STRUCT_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct TableLike { mask: usize, ctrl: u64, growth: usize, items: usize }
fn main() {
    let mut a = TableLike { mask: 3, ctrl: 11, growth: 5, items: 2 };
    let mut b = TableLike { mask: 7, ctrl: 22, growth: 1, items: 6 };
    std::mem::swap(&mut a, &mut b);
    assert_eq!(a, TableLike { mask: 7, ctrl: 22, growth: 1, items: 6 });
    assert_eq!(b, TableLike { mask: 3, ctrl: 11, growth: 5, items: 2 });
    println!("swap ok");
}
"#;

#[test]
fn run_mode_swap_nonoverlapping_whole_struct() {
    let dir = temp_dir("swap-struct");
    let src = fixture(&dir, "swap_struct.rs", SWAP_STRUCT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "swap ok\n");
}

// ----- Lane KE: reborrowed &mut Vec slice positions ---------------

const MUT_VEC_REBORROW_REVERSE_SRC: &str = r#"
fn reverse_arg(v: &mut [usize]) {
    v.reverse();
}
fn bar(v: &mut [usize]) {
    v.reverse();
    reverse_arg(v);
    v.reverse();
}
fn main() {
    let mut the_vec = vec![0x0102usize, 0x0304, 0x0506, 0x0708];
    bar(&mut the_vec);
    assert_eq!(the_vec, [0x0708, 0x0506, 0x0304, 0x0102]);
    println!("ok");
}
"#;

#[test]
fn run_mode_mut_vec_reborrow_reverse() {
    // A `&mut Vec` reborrowed as `&mut [usize]` and reversed
    // through the receiver and through a helper argument: the
    // slice machinery (`split_at_mut`, `revswap`, `mem::swap` over
    // byte-chunk pointers) must address whole elements of the
    // backing Vec, in both directions. Multi-byte element values
    // pin the semantics: a byte-wise swap would corrupt them.
    run_ok("mut-vec-reborrow", MUT_VEC_REBORROW_REVERSE_SRC, "ok\n");
}

const SUBSLICE_SPLIT_AT_MUT_SRC: &str = r#"
fn main() {
    let mut v = vec![10usize, 20, 30, 40, 50, 60];
    let sub = &mut v[1..5];
    let (a, b) = sub.split_at_mut(2);
    a[1] = 99;
    b[0] = 77;
    a.reverse();
    assert_eq!(v, [10, 99, 20, 77, 50, 60]);
    println!("ok");
}
"#;

#[test]
fn run_mode_subslice_split_at_mut_positions() {
    // `split_at_mut` on a range sub-slice: the front half's data
    // pointer is the sub-slice's first element (not that
    // element's own first element), indexing either half composes with the
    // sub-slice offset in the backing Vec, and the back half's data
    // pointer absorbs its displacement through the sub-slice step
    // instead of degrading to a displaced pointer.
    run_ok("subslice-split", SUBSLICE_SPLIT_AT_MUT_SRC, "ok\n");
}
const TLS_DTOR_BOTH_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};

static DROPPED: AtomicUsize = AtomicUsize::new(0);

struct Foo;
impl Drop for Foo {
    fn drop(&mut self) {
        DROPPED.fetch_add(1, Ordering::SeqCst);
    }
}

thread_local!(static FOO: Foo = Foo);
thread_local!(static BAR: Foo = Foo);

fn main() {
    // A spawned thread's TLS destructors run at its exit, in
    // reverse registration order; try_with inside a destructor
    // observes the destruction.
    std::thread::spawn(|| {
        FOO.with(|_| ());
        BAR.with(|_| ());
    })
    .join()
    .unwrap();
    assert_eq!(DROPPED.load(Ordering::SeqCst), 2);
    // The entry thread's destructors run at program exit.
    FOO.with(|_| ());
    println!("ok");
}
"#;

#[test]
fn run_mode_tls_destructors_run_at_thread_exit() {
    // TLS destructors are registered through std's
    // `__cxa_thread_atexit` wrapper; the model records the
    // registrations per thread and runs them (as ordinary guest
    // code, so the destroy shim's state transitions happen) when
    // the owning thread exits.
    let dir = temp_dir("tls-dtor");
    let src = fixture(&dir, "tls_dtor.rs", TLS_DTOR_BOTH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "TLS destructors must run at thread exit; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}


const PANIC_IN_DTOR_SRC: &str = r#"
use std::sync::atomic::{AtomicBool, Ordering};

static FIELD_DROPPED: AtomicBool = AtomicBool::new(false);

struct Field {
    _foo: isize,
}

impl Drop for Field {
    fn drop(&mut self) {
        FIELD_DROPPED.store(true, Ordering::SeqCst);
    }
}

struct Outer {
    _field: Field,
}

impl Drop for Outer {
    fn drop(&mut self) {
        panic!("outer destructor panics");
    }
}

fn main() {
    let handle = std::thread::spawn(|| {
        let _outer = Outer { _field: Field { _foo: 3 } };
    });
    assert!(handle.join().is_err());
    assert!(FIELD_DROPPED.load(Ordering::SeqCst));
    println!("ok");
}
"#;

#[test]
fn run_mode_panicking_destructor_still_drops_fields() {
    // When a destructor panics, the remaining drop glue of the
    // value (its fields) is the cleanup of the drop itself: the
    // field destructors still run while the panic unwinds, and
    // the panic then continues to the thread boundary.
    let dir = temp_dir("panic-in-dtor");
    let src = fixture(&dir, "panic_in_dtor.rs", PANIC_IN_DTOR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a panicking destructor must still drop the fields; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PUNNED_ACCESS_SRC: &str = r#"
#![feature(core_intrinsics)]

use std::intrinsics::{unaligned_volatile_load, unaligned_volatile_store};

fn main() {
    unsafe {
        let mut arr: [u8; 3] = [1, 2, 3];
        // A u16 load through a pointer into byte storage composes
        // the two stored bytes.
        let ptr = arr[1..].as_mut_ptr() as *mut u16;
        unaligned_volatile_store(ptr, 0);
        assert_eq!(arr, [1, 0, 0]);
        unaligned_volatile_store(ptr, 0x0504);
        assert_eq!(arr, [1, 4, 5]);
        assert_eq!(unaligned_volatile_load(ptr as *const u16), 0x0504);
        println!("ok");
    }
}
"#;

#[test]
fn run_mode_unaligned_volatile_access_composes_bytes() {
    // An unaligned (type-punned) volatile access through a
    // pointer into byte storage reads and writes the composed
    // bytes of the wider integer, not just the first byte.
    let dir = temp_dir("punned-access");
    let src = fixture(&dir, "punned_access.rs", PUNNED_ACCESS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "unaligned volatile accesses must compose bytes; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const CONST_CAST_SRC: &str = r#"
fn main() {
    // A constant operand of a cast is read under its own
    // (source) type: u128::MAX is not -1i128.
    assert!(u128::MAX as f32 == f32::INFINITY);
    assert!(u64::MAX as f32 == 18446744073709551616.0);
    assert!((-1i8) as u32 == u32::MAX);
    assert!((-1i128) as f64 == -1.0);
    println!("ok");
}
"#;

#[test]
fn run_mode_constant_cast_operand_keeps_source_type() {
    // The operand of a cast on a constant must be converted
    // under the constant's own type, not the cast destination's:
    // typing it by the destination read `u128::MAX` as a signed
    // value and the int-to-float cast produced -1.0.
    let dir = temp_dir("const-cast");
    let src = fixture(&dir, "const_cast.rs", CONST_CAST_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "constant cast operands keep their source type; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const INDEX_LOCATION_SRC: &str = r#"
#[track_caller]
fn caller_line() -> u32 {
    std::panic::Location::caller().line()
}

struct A {
    prev_line: u32,
}

impl std::ops::Index<usize> for A {
    type Output = ();
    fn index(&self, _idx: usize) -> &() {
        assert_eq!(caller_line(), self.prev_line + 1);
        &()
    }
}

fn main() {
    let prev_line = caller_line();
    (A { prev_line })[0];
    println!("ok");
}
"#;

#[test]
fn run_mode_track_caller_location_forwards_through_index() {
    // `Index::index` is `#[track_caller]` through the trait:
    // a location taken inside an index impl names the indexing
    // expression in the caller, not the call inside the impl.
    // The call names the trait method, so the stamp must reach
    // the impl body the dispatch lands on.
    let dir = temp_dir("index-location");
    let src = fixture(&dir, "index_location.rs", INDEX_LOCATION_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "caller location must forward through Index::index; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const COUNTING_ALLOCATOR_SRC: &str = r#"#![feature(slice_ptr_get)]

use std::alloc::{self, Allocator, Global, Layout, System};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

static HITS: AtomicUsize = AtomicUsize::new(0);

struct A;

unsafe impl alloc::GlobalAlloc for A {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        HITS.fetch_add(1, Ordering::SeqCst);
        alloc::GlobalAlloc::alloc(&alloc::System, layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        HITS.fetch_add(1, Ordering::SeqCst);
        alloc::GlobalAlloc::dealloc(&alloc::System, ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: A = A;

fn hits() -> usize {
    HITS.load(Ordering::SeqCst)
}

fn main() {
    println!("hello!");
    // std's lazy stdout buffer goes through the global
    // allocator once.
    let n = hits();
    assert!(n >= 1, "stdout buffer allocation not observed: {n}");

    unsafe {
        let layout = Layout::new::<u32>();
        // The trait entry points count; the System allocator
        // behind them does not.
        let memory = Global.allocate(layout).unwrap();
        assert_eq!(hits(), n + 1);
        Global.deallocate(NonNull::new(memory.as_non_null_ptr().as_ptr() as *mut u8).unwrap(), layout);
        assert_eq!(hits(), n + 2);

        let s = String::with_capacity(10);
        assert_eq!(hits(), n + 3);
        drop(s);
        assert_eq!(hits(), n + 4);

        let memory = System.allocate(layout).unwrap();
        assert_eq!(hits(), n + 4);
        System.deallocate(NonNull::new(memory.as_non_null_ptr().as_ptr() as *mut u8).unwrap(), layout);
        assert_eq!(hits(), n + 4);

        let b = Box::new(42u32);
        assert_eq!(hits(), n + 5);
        assert_eq!(*b, 42);
        drop(b);
        assert_eq!(hits(), n + 6);
    }
    println!("counting ok");
}
"#;

#[test]
fn run_mode_guest_global_allocator_observes_allocations() {
    // A guest `#[global_allocator]` must observe the
    // allocations the abstract std model performs: the lazy
    // stdout buffer, `String::with_capacity` and its drop,
    // `Box::new` and its drop, and the `Global` trait entry
    // points — while direct `System` allocations stay
    // invisible, exactly as in the ui allocator tests.
    let dir = temp_dir("counting-allocator");
    let src = fixture(&dir, "counting_allocator.rs", COUNTING_ALLOCATOR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "guest allocator must observe modelled allocations; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hello!\ncounting ok\n"
    );
}

const SWAPPED_DROP_IMPL_SRC: &str = r#"
use std::mem;
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNT: AtomicUsize = AtomicUsize::new(0);

struct S<U, V> {
    _u: U,
    size_of_u: usize,
    _v: V,
    size_of_v: usize,
}

impl<U, V> S<U, V> {
    fn new(u: U, v: V) -> Self {
        S {
            _u: u,
            size_of_u: mem::size_of::<U>(),
            _v: v,
            size_of_v: mem::size_of::<V>(),
        }
    }
}

// The impl declares its parameters in the opposite order
// from the self type: the destructor instance for
// `S<u8, u16>` must still substitute U = u8, V = u16.
impl<V, U> Drop for S<U, V> {
    fn drop(&mut self) {
        assert_eq!(mem::size_of::<U>(), self.size_of_u);
        assert_eq!(mem::size_of::<V>(), self.size_of_v);
        COUNT.fetch_add(1, Ordering::SeqCst);
    }
}

fn main() {
    assert_eq!(COUNT.load(Ordering::SeqCst), 0);
    {
        S::new(0u8, 1u16);
    }
    assert_eq!(COUNT.load(Ordering::SeqCst), 1);
    println!("swapped drop ok");
}
"#;

#[test]
fn run_mode_drop_impl_with_swapped_generic_params() {
    // A `Drop` impl whose generic parameters are declared in
    // a different order than the self type's must be
    // monomorphized by unifying the impl's self type with the
    // concrete type, not by copying the self type's arguments
    // positionally (which would run the destructor of the
    // swapped instantiation).
    let dir = temp_dir("swapped-drop-impl");
    let src = fixture(&dir, "swapped_drop_impl.rs", SWAPPED_DROP_IMPL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "swapped-parameter Drop impl must substitute correctly; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "swapped drop ok\n");
}

const BOX_DYN_SEND_DROP_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};

static DROPPED: AtomicUsize = AtomicUsize::new(0);

struct Marker;

impl Drop for Marker {
    fn drop(&mut self) {
        DROPPED.fetch_add(1, Ordering::SeqCst);
    }
}

fn main() {
    {
        let _: Box<dyn Send> = Box::new(Marker);
    }
    assert_eq!(DROPPED.load(Ordering::SeqCst), 1);
    println!("dyn drop ok");
}
"#;

#[test]
fn run_mode_box_dyn_drop_runs_concrete_dtor() {
    // Dropping a `Box<dyn Trait>` runs the concrete type's
    // destructor through the vtable: the unsizing cast's
    // source pointee is a drop candidate like any dropped
    // type, so its destructor instance is demanded, and the
    // glue the cast carries is the concrete type's.
    let dir = temp_dir("box-dyn-send-drop");
    let src = fixture(&dir, "box_dyn_send_drop.rs", BOX_DYN_SEND_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<dyn Trait> drop must run the concrete destructor; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn drop ok\n");
}

const DEFAULT_FIELD_VEC_SRC: &str = r#"
#![feature(default_field_values)]

pub struct Foo {
    pub baz: i32 = 42 + 3,
    pub bak: Vec<i32> = Vec::new(),
}

fn main() {
    let x = Foo { .. };
    assert_eq!(x.baz, 45);
    assert!(x.bak.is_empty());
    println!("default field values ok");
}
"#;

#[test]
fn run_mode_default_field_value_empty_vec_is_empty() {
    // A field default is materialised as a constant in the
    // type's raw layout; for a modelled container (`Vec`) the
    // constant must be translated into the model's
    // contents representation, or the raw tree's field count
    // reads back as the vector's length.
    let dir = temp_dir("default-field-vec");
    let src = fixture(&dir, "default_field_vec.rs", DEFAULT_FIELD_VEC_SRC);
    let out = run_driver_extra(&src, &["--edition", "2024"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a Vec::new() field default must be an empty vector; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "default field values ok\n");
}

const CSTR_NONASCII_SRC: &str = r#"
fn main() {
    assert_eq!(
        c"\xEF\x80\u{1F980}\u{1F980}".to_bytes_with_nul(),
        &[0xEF, 0x80, 0xF0, 0x9F, 0xA6, 0x80, 0xF0, 0x9F, 0xA6, 0x80, 0x00],
    );
    println!("cstr non-ascii ok");
}
"#;

#[test]
fn run_mode_cstr_literal_non_ascii_bytes() {
    // A `CStr`'s field is a `[c_char]` slice and `c_char` is
    // signed on this platform: decoding the constant as
    // signed-char elements would leak the sign into every byte
    // view of the string (indexing would read -17 for 0xEF and
    // byte equality against `u8` data would fail). The constant
    // decodes as the byte string it is.
    let dir = temp_dir("cstr-nonascii");
    let src = fixture(&dir, "cstr_nonascii.rs", CSTR_NONASCII_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "non-ASCII CStr literal bytes must compare as u8 data; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "cstr non-ascii ok\n");
}

const COROUTINE_DROP_ENV_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
#![allow(dropping_copy_types)]

use std::ops::Coroutine;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

static A: AtomicUsize = AtomicUsize::new(0);

struct B;

impl Drop for B {
    fn drop(&mut self) {
        A.fetch_add(1, Ordering::SeqCst);
    }
}

fn main() {
    let b = B;
    let mut foo = #[coroutine] || {
        yield;
        drop(b);
    };
    let n = A.load(Ordering::SeqCst);
    drop(Pin::new(&mut foo).resume(()));
    assert_eq!(A.load(Ordering::SeqCst), n);
    drop(foo);
    assert_eq!(A.load(Ordering::SeqCst), n + 1);
    println!("coroutine drop env ok");
}
"#;

#[test]
fn run_mode_suspended_coroutine_drop_runs_capture_dtor() {
    // Dropping a suspended coroutine drops the captures it
    // still holds: the coroutine's storage is one field list
    // shared by every state, so the suspended state's glue is
    // the unresumed state's fields (plus its own saved locals),
    // not the empty plan an unevidenced variant would get.
    let dir = temp_dir("coroutine-drop-env");
    let src = fixture(&dir, "coroutine_drop_env.rs", COROUTINE_DROP_ENV_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a suspended coroutine's drop must run its captures' destructors; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "coroutine drop env ok\n");
}

const PIN_DROP_SRC: &str = r#"
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[pin_v2]
struct Foo {
    dropped: Arc<AtomicBool>,
}

impl Drop for Foo {
    fn pin_drop(self: Pin<&mut Self>) {
        self.dropped.store(true, Ordering::Relaxed);
    }
}

fn main() {
    let dropped = Arc::new(AtomicBool::new(false));
    {
        let _f = Foo { dropped: dropped.clone() };
    }
    assert!(dropped.load(Ordering::Relaxed));
    println!("pin drop ok");
}
"#;

#[test]
fn run_mode_pin_drop_runs_as_the_destructor() {
    // Under pin ergonomics a type's destructor entry point is
    // `pin_drop(self: Pin<&mut Self>)` and there is no `drop`
    // method: the glue must call it, with the argument in the
    // one-field `Pin` aggregate shape the value model uses for
    // `Pin` everywhere else.
    let dir = temp_dir("pin-drop");
    let src = fixture(&dir, "pin_drop.rs", PIN_DROP_SRC);
    let out = run_driver_extra(&src, &["--edition", "2024"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "pin_drop must run as the type's destructor; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "pin drop ok\n");
}

const GEN_BLOCK_PANIC_SRC: &str = r#"
#![feature(gen_blocks)]

fn main() {
    let mut iter = gen {
        yield 42;
        panic!("foo");
    };
    assert_eq!(iter.next(), Some(42));
    let mut tmp = std::panic::AssertUnwindSafe(&mut iter);
    match std::panic::catch_unwind(move || tmp.next()) {
        Ok(_) => unreachable!(),
        Err(err) => assert_eq!(*err.downcast::<&'static str>().unwrap(), "foo"),
    }
    match std::panic::catch_unwind(move || iter.next()) {
        Ok(_) => unreachable!(),
        Err(err) => assert_eq!(
            *err.downcast::<&'static str>().unwrap(),
            "`gen fn` should just keep returning `None` after panicking",
        ),
    }
    println!("gen block panic ok");
}
"#;

#[test]
fn run_mode_gen_block_resume_after_panic_payload() {
    // Resuming a panicked `gen` coroutine panics with the
    // `panic_const_gen_fn_none_panic` payload, not with a
    // generic assertion-failure message: the resume guard's
    // assert carries a `ResumedAfterPanic` message whose text
    // is that exact string.
    let dir = temp_dir("gen-block-panic");
    let src = fixture(&dir, "gen_block_panic.rs", GEN_BLOCK_PANIC_SRC);
    let out = run_driver_extra(&src, &["--edition", "2024"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "resume after panic must carry the gen panic payload; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "gen block panic ok\n");
}

const TUPLE_CLONE_UNWIND_SRC: &str = r#"
use std::rc::Rc;

struct S(Rc<()>);

impl Clone for S {
    fn clone(&self) -> Self {
        if Rc::strong_count(&self.0) == 7 {
            panic!("oops");
        }
        S(self.0.clone())
    }
}

fn main() {
    let counter = Rc::new(());
    let ccounter = counter.clone();
    let result = std::panic::catch_unwind(move || {
        let _ =
            (S(ccounter.clone()), S(ccounter.clone()), S(ccounter.clone()), S(ccounter)).clone();
    });
    assert!(result.is_err());
    assert_eq!(1, Rc::strong_count(&counter));
    println!("tuple clone unwind ok");
}
"#;

#[test]
fn run_mode_tuple_clone_runs_element_impls_and_cleans_up() {
    // A tuple's `Clone` impl is a compiler shim: with its body
    // demanded, cloning a tuple runs each element's own `Clone`
    // impl in order, and a panic mid-clone drops the elements
    // already cloned (the shim's unwind cleanup), which the
    // reference count observes.
    let dir = temp_dir("tuple-clone-unwind");
    let src = fixture(&dir, "tuple_clone_unwind.rs", TUPLE_CLONE_UNWIND_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "tuple clone must run element impls and clean up on panic; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "tuple clone unwind ok\n");
}

const BOXED_FUTURE_PARAM_DROP_SRC: &str = r#"
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

static DROPS: AtomicUsize = AtomicUsize::new(0);

struct D;

impl Drop for D {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

struct YieldOnce(bool);

impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: std::pin::Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.0 = true;
            Poll::Pending
        }
    }
}

async fn foo(x: D, _y: D) {
    let _ = &x;
    YieldOnce(false).await;
}

fn dummy_waker() -> Waker {
    fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VT)
    }
    fn noop(_: *const ()) {}
    static VT: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VT)) }
}

fn main() {
    let mut fut = Box::pin(foo(D, D));
    let w = dummy_waker();
    let mut cx = Context::from_waker(&w);
    let r = fut.as_mut().poll(&mut cx);
    assert!(matches!(r, Poll::Pending));
    assert_eq!(DROPS.load(Ordering::SeqCst), 0);
    drop(fut);
    assert_eq!(DROPS.load(Ordering::SeqCst), 2);
    println!("boxed future param drop ok");
}
"#;

#[test]
fn run_mode_boxed_suspended_future_drops_parameters_once() {
    // A boxed future's prologue moves its parameters out of the
    // environment (the allocation's upvar storage); the slots
    // clear as they are moved from, so when the suspended future
    // is dropped, its glue drops each parameter exactly once -
    // through the saved storage it now lives in, not again
    // through the environment.
    let dir = temp_dir("boxed-future-param-drop");
    let src = fixture(&dir, "boxed_future_param_drop.rs", BOXED_FUTURE_PARAM_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a boxed suspended future must drop its parameters exactly once; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "boxed future param drop ok\n"
    );
}

const TOOL_GUEST_SRC: &str = r#"
use std::process::Command;

fn main() {
    // The harness passes the tool program's path as the first
    // guest argument; every compile request below is answered
    // by the driver's own conversion machinery (the tool
    // interception), never by a real child process.
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();

    std::fs::write("t.rs", "pub fn f() -> u32 { 7 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--crate-type=rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("t.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "good compile failed");
    assert!(std::path::Path::new("libt.rlib").exists(), "rlib artifact missing");

    // dep-info is real text naming the input.
    let out = Command::new(&rustc)
        .arg("--crate-type=rlib")
        .arg("--emit=dep-info")
        .arg("--out-dir")
        .arg(".")
        .arg("t.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "dep-info compile failed");
    let dep = std::fs::read("t.d").unwrap();
    assert!(!dep.is_empty(), "dep-info is empty");

    // A failing source: exit code 1 and diagnostics on stderr.
    std::fs::write("bad.rs", "fn main() { let _x: i32 = \"s\"; }\n").unwrap();
    let out = Command::new(&rustc).arg("bad.rs").output().unwrap();
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
    assert!(!out.stderr.is_empty(), "diagnostics missing");

    // `status()` reports the code without an Output value.
    let st = Command::new(&rustc)
        .arg("--crate-type=rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("t.rs")
        .status()
        .unwrap();
    assert!(st.success());

    // A consumer bound through `--extern` resolves the
    // artifact back to its source.
    std::fs::write("u.rs", "extern crate t; fn main() { assert_eq!(t::f(), 7); }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--extern")
        .arg("t=libt.rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("u.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "extern consumer failed");

    // `--print` rides the informational path.
    let out = Command::new(&rustc)
        .arg("--print")
        .arg("target-list")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(!out.stdout.is_empty(), "print output missing");

    println!("tool guest ok");
}
"#;

#[test]
fn run_mode_tool_interception_answers_compiler_calls() {
    let dir = temp_dir("tool-intercept");
    let src = fixture(&dir, "tool_guest.rs", TOOL_GUEST_SRC);
    // The tool program is this very driver binary: the string
    // only has to name a program that would start (so the
    // process model lets the call through to the interception),
    // and the handler answers with the driver's own conversion.
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(
        &src,
        &["--tool-program", driver, "--guest-arg", driver],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "tool guest ok\n", "stderr={stderr}");
}

const ARTIFACT_GUEST_SRC: &str = r#"
use std::process::Command;

fn main() {
    // The harness passes the tool program's path as the first
    // guest argument; compiler calls are answered by the tool
    // interception, and running a binary built during the run
    // is answered from the interception's artifact registry:
    // the recorded source is interpreted with the bindings its
    // build invocation carried.
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();

    // A library, then a binary that uses it through `--extern`.
    std::fs::write("t.rs", "pub fn f() -> u32 { 7 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--crate-type=rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("t.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "library compile failed");
    std::fs::write(
        "u.rs",
        "extern crate t; fn main() { println!(\"u says {}\", t::f()); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--extern")
        .arg("t=libt.rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("u.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "binary compile failed");
    let out = Command::new(dir.join("u")).output().unwrap();
    assert!(out.status.success(), "running u failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "u says 7\n");

    // Arguments, stderr, and `status()` ride the same path.
    std::fs::write(
        "b.rs",
        "fn main() { let a: Vec<String> = std::env::args().collect(); println!(\"b args: {}\", a[1..].join(\",\")); eprintln!(\"b err\"); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("b.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "b compile failed");
    let out = Command::new(dir.join("b")).arg("x").arg("y").output().unwrap();
    assert!(out.status.success(), "running b failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "b args: x,y\n");
    assert_eq!(String::from_utf8(out.stderr).unwrap(), "b err\n");
    let st = Command::new(dir.join("b")).status().unwrap();
    assert!(st.success(), "status on b failed");

    // A nonzero exit code is the program's own.
    std::fs::write("c.rs", "fn main() { std::process::exit(3); }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("c.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "c compile failed");
    let out = Command::new(dir.join("c")).output().unwrap();
    assert_eq!(out.status.code(), Some(3));

    println!("artifact guest ok");
}
"#;

#[test]
fn run_mode_tool_interception_runs_built_binaries() {
    let dir = temp_dir("tool-artifact");
    let src = fixture(&dir, "artifact_guest.rs", ARTIFACT_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The guest's one `status()` run (the `b` binary)
    // inherits the guest's streams, so the binary's own
    // line stands in the guest's stdout ahead of the
    // final marker.
    assert_eq!(stdout, "b args: \nartifact guest ok\n", "stderr={stderr}");
}

const ARTIFACT_FIDELITY_GUEST_SRC: &str = r#"
use std::process::Command;

fn build(rustc: &str, name: &str, src: &str, extra: &[&str]) {
    std::fs::write(format!("{name}.rs"), src).unwrap();
    let mut cmd = Command::new(rustc);
    cmd.arg("--edition=2021").arg("--out-dir").arg(".");
    for a in extra {
        cmd.arg(a);
    }
    cmd.arg(format!("{name}.rs"));
    let out = cmd.output().unwrap();
    assert!(
        out.status.success(),
        "build of {name} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();

    // The producing invocation's compilation options ride
    // along: a binary built with `--cfg foo` runs its cfg
    // branch when the recipe executes it.
    build(
        &rustc,
        "cfgprobe",
        "fn main() { if cfg!(foo) { println!(\"cfg foo on\"); } else { println!(\"cfg foo off\"); } }\n",
        &["--cfg", "foo"],
    );
    let out = Command::new(dir.join("cfgprobe")).output().unwrap();
    assert!(out.status.success(), "running cfgprobe failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "cfg foo on\n");

    // A built program's exit code 2 is the program's own,
    // not a driver failure: both the `process::exit` form
    // and the `ExitCode` form reach the recipe as code 2.
    build(&rustc, "exit2", "fn main() { std::process::exit(2); }\n", &[]);
    let out = Command::new(dir.join("exit2")).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    build(
        &rustc,
        "exit2b",
        "fn main() -> std::process::ExitCode { std::process::ExitCode::from(2) }\n",
        &[],
    );
    let out = Command::new(dir.join("exit2b")).output().unwrap();
    assert_eq!(out.status.code(), Some(2));

    // The built program's stderr is relayed byte for byte:
    // no newline is invented, CRLF survives, and a guest
    // line that begins with the driver's own prefix is
    // still the guest's.
    build(&rustc, "errprobe", "fn main() { eprint!(\"error\"); }\n", &[]);
    let out = Command::new(dir.join("errprobe")).output().unwrap();
    assert_eq!(out.stderr, b"error".to_vec());
    build(
        &rustc,
        "errprobe2",
        "fn main() { eprint!(\"a\\r\\nb\\n\"); eprintln!(\"sparagmos-driver impostor\"); }\n",
        &[],
    );
    let out = Command::new(dir.join("errprobe2")).output().unwrap();
    assert_eq!(
        out.stderr,
        b"a\r\nb\nsparagmos-driver impostor\n".to_vec()
    );

    // A panicking built program: its own stderr comes
    // first, then the panic report, and the code is the
    // panic code.
    build(
        &rustc,
        "panicprobe",
        "fn main() { eprintln!(\"before panic\"); panic!(\"artifact boom\"); }\n",
        &[],
    );
    let out = Command::new(dir.join("panicprobe")).output().unwrap();
    assert_eq!(out.status.code(), Some(101));
    assert_eq!(
        String::from_utf8(out.stderr).unwrap(),
        "before panic\nthread 'main' panicked: artifact boom\n"
    );

    // The built program's argv[0] is the name it was
    // invoked under, not the source path the run is
    // reconstructed from.
    build(
        &rustc,
        "argvprobe",
        "fn main() { println!(\"argv0={}\", std::env::args().next().unwrap()); }\n",
        &[],
    );
    let expect0 = dir.join("argvprobe");
    let out = Command::new(&expect0).output().unwrap();
    assert!(out.status.success(), "running argvprobe failed");
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        format!("argv0={}\n", expect0.display())
    );

    // The compilation environment and the run environment
    // are kept apart: `option_env!` sees the value the
    // build was invoked with, the program's own `env::var`
    // sees the value it was invoked with, and a variable
    // set only for the build does not leak into the run.
    std::fs::write(
        "envprobe.rs",
        "fn main() { println!(\"baked={:?} runtime={:?} leak={:?}\", option_env!(\"BUILD_VALUE\"), std::env::var(\"BUILD_VALUE\").unwrap_or_default(), std::env::var(\"BUILD_ONLY\").ok()); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("envprobe.rs")
        .env("BUILD_VALUE", "old")
        .env("BUILD_ONLY", "build-secret")
        .output()
        .unwrap();
    assert!(out.status.success(), "envprobe build failed");
    let out = Command::new(dir.join("envprobe"))
        .env("BUILD_VALUE", "new")
        .output()
        .unwrap();
    assert!(out.status.success(), "running envprobe failed");
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "baked=Some(\"old\") runtime=\"new\" leak=None\n"
    );

    // A built binary denotes the source its build
    // converted: rewriting the source afterwards does not
    // change what an earlier artifact runs, and deleting
    // the source does not orphan it.
    std::fs::write("ver.rs", "fn main() { println!(\"version 1\"); }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("-o")
        .arg("ver1")
        .arg("ver.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "ver1 build failed");
    std::fs::write("ver.rs", "fn main() { println!(\"version 2\"); }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("-o")
        .arg("ver2")
        .arg("ver.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "ver2 build failed");
    let out = Command::new(dir.join("ver1")).output().unwrap();
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "version 1\n");
    let out = Command::new(dir.join("ver2")).output().unwrap();
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "version 2\n");
    build(&rustc, "gone", "fn main() { println!(\"still here\"); }\n", &[]);
    std::fs::remove_file("gone.rs").unwrap();
    let out = Command::new(dir.join("gone")).output().unwrap();
    assert!(out.status.success(), "running gone failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "still here\n");

    // The same holds for a library the binary was built
    // against: the library's recorded source, not the
    // file as it later stands, is what the binary's run
    // converts.
    std::fs::write("libv.rs", "pub fn f() -> u32 { 1 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--crate-type=rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("libv.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "libv build failed");
    std::fs::write(
        "useslib.rs",
        "extern crate libv; fn main() { println!(\"lib says {}\", libv::f()); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--extern")
        .arg("libv=liblibv.rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("useslib.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "useslib build failed");
    std::fs::write("libv.rs", "pub fn f() -> u32 { 2 }\n").unwrap();
    let out = Command::new(dir.join("useslib")).output().unwrap();
    assert!(out.status.success(), "running useslib failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "lib says 1\n");

    // A binary keeps the library version it was built
    // against: rebuilding the library at the same output
    // path afterwards does not change what the earlier
    // binary runs.
    build(
        &rustc,
        "libw",
        "pub fn f() -> u32 { 1 }\n",
        &["--crate-type=rlib"],
    );
    build(
        &rustc,
        "usesw",
        "extern crate libw; fn main() { println!(\"lib says {}\", libw::f()); }\n",
        &["--extern", "libw=liblibw.rlib"],
    );
    build(
        &rustc,
        "libw",
        "pub fn f() -> u32 { 2 }\n",
        &["--crate-type=rlib"],
    );
    let out = Command::new(dir.join("usesw")).output().unwrap();
    assert!(out.status.success(), "running usesw failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "lib says 1\n");

    // Two library outputs built from one source at
    // different times bind by the artifact each binary
    // named, not by whichever artifact the source path
    // resolves to later: a binary linked against the
    // second build runs the second build's code.
    build(
        &rustc,
        "libx",
        "pub fn f() -> u32 { 1 }\n",
        &["--crate-type=rlib", "-o", "libx_a.rlib"],
    );
    build(
        &rustc,
        "libx",
        "pub fn f() -> u32 { 2 }\n",
        &["--crate-type=rlib", "-o", "libx_b.rlib"],
    );
    build(
        &rustc,
        "usesx",
        "extern crate libx; fn main() { println!(\"lib says {}\", libx::f()); }\n",
        &["--extern", "libx=libx_b.rlib"],
    );
    let out = Command::new(dir.join("usesx")).output().unwrap();
    assert!(out.status.success(), "running usesx failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "lib says 2\n");

    // A binary bound through the `extern crate` scan (the
    // recipe passes no `--extern`) keeps the library
    // version its build bound: rebuilding the library at
    // the same output path afterwards does not change
    // what the earlier binary runs.
    build(
        &rustc,
        "scanlib",
        "pub fn f() -> u32 { 1 }\n",
        &["--crate-type=rlib"],
    );
    build(
        &rustc,
        "usesscan",
        "extern crate scanlib; fn main() { println!(\"lib says {}\", scanlib::f()); }\n",
        &[],
    );
    build(
        &rustc,
        "scanlib",
        "pub fn f() -> u32 { 2 }\n",
        &["--crate-type=rlib"],
    );
    let out = Command::new(dir.join("usesscan")).output().unwrap();
    assert!(out.status.success(), "running usesscan failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "lib says 1\n");

    // The same through a source rewrite with no rebuild:
    // the scan-bound pin is the tree the binary's build
    // converted, not the file as it later stands.
    build(
        &rustc,
        "rewlib",
        "pub fn f() -> u32 { 1 }\n",
        &["--crate-type=rlib"],
    );
    build(
        &rustc,
        "usesrew",
        "extern crate rewlib; fn main() { println!(\"lib says {}\", rewlib::f()); }\n",
        &[],
    );
    std::fs::write("rewlib.rs", "pub fn f() -> u32 { 2 }\n").unwrap();
    let out = Command::new(dir.join("usesrew")).output().unwrap();
    assert!(out.status.success(), "running usesrew failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "lib says 1\n");

    // A built binary denotes its whole source tree, not
    // just the crate root: rewriting the root AND its
    // module file after the build changes nothing the
    // earlier binary runs.
    let mtree = dir.join("mtree");
    std::fs::create_dir_all(&mtree).unwrap();
    std::fs::write(
        mtree.join("modprobe.rs"),
        "mod helper; fn main() { println!(\"value {}\", helper::v()); }\n",
    )
    .unwrap();
    std::fs::write(mtree.join("helper.rs"), "pub fn v() -> u32 { 1 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(&mtree)
        .arg(mtree.join("modprobe.rs"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "modprobe build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::write(
        mtree.join("modprobe.rs"),
        "mod helper; fn main() { println!(\"value {} newroot\", helper::v()); }\n",
    )
    .unwrap();
    std::fs::write(mtree.join("helper.rs"), "pub fn v() -> u32 { 2 }\n").unwrap();
    let out = Command::new(mtree.join("modprobe")).output().unwrap();
    assert!(out.status.success(), "running modprobe failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "value 1\n");

    // Modifying only the module file leaves the root's
    // bytes untouched, and still must not change what
    // the earlier binary runs: the snapshot compares
    // the whole tree, not the root alone.
    let htree = dir.join("htree");
    std::fs::create_dir_all(&htree).unwrap();
    std::fs::write(
        htree.join("modprobe.rs"),
        "mod helper; fn main() { println!(\"value {}\", helper::v()); }\n",
    )
    .unwrap();
    std::fs::write(htree.join("helper.rs"), "pub fn v() -> u32 { 1 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(&htree)
        .arg(htree.join("modprobe.rs"))
        .output()
        .unwrap();
    assert!(out.status.success(), "htree modprobe build failed");
    std::fs::write(htree.join("helper.rs"), "pub fn v() -> u32 { 2 }\n").unwrap();
    let out = Command::new(htree.join("modprobe")).output().unwrap();
    assert!(out.status.success(), "running htree modprobe failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "value 1\n");

    // An `include!` input is part of the tree too: the
    // root that only includes its `main` builds a binary,
    // and modifying only the included file afterwards
    // does not change what it runs.
    let itree = dir.join("itree");
    std::fs::create_dir_all(&itree).unwrap();
    std::fs::write(itree.join("incprobe.rs"), "include!(\"part.rs\");\n").unwrap();
    std::fs::write(
        itree.join("part.rs"),
        "fn main() { println!(\"part 1\"); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(&itree)
        .arg(itree.join("incprobe.rs"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "incprobe build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::write(
        itree.join("part.rs"),
        "fn main() { println!(\"part 2\"); }\n",
    )
    .unwrap();
    let out = Command::new(itree.join("incprobe")).output().unwrap();
    assert!(out.status.success(), "running incprobe failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "part 1\n");

    // Deleting the whole tree after the build does not
    // orphan the binary: the snapshot holds every file.
    let dtree = dir.join("dtree");
    std::fs::create_dir_all(&dtree).unwrap();
    std::fs::write(
        dtree.join("modprobe.rs"),
        "mod helper; fn main() { println!(\"value {}\", helper::v()); }\n",
    )
    .unwrap();
    std::fs::write(dtree.join("helper.rs"), "pub fn v() -> u32 { 1 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(&dtree)
        .arg(dtree.join("modprobe.rs"))
        .output()
        .unwrap();
    assert!(out.status.success(), "dtree modprobe build failed");
    std::fs::remove_file(dtree.join("modprobe.rs")).unwrap();
    std::fs::remove_file(dtree.join("helper.rs")).unwrap();
    let out = Command::new(dtree.join("modprobe")).output().unwrap();
    assert!(out.status.success(), "running dtree modprobe failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "value 1\n");

    // A library's tree rides its pin the same way:
    // modifying only the library's module file after the
    // binary was built, or deleting the library's tree,
    // does not change what the binary runs.
    let tlib = dir.join("tlib");
    std::fs::create_dir_all(&tlib).unwrap();
    std::fs::write(
        tlib.join("treelib.rs"),
        "mod helper; pub fn f() -> u32 { helper::v() }\n",
    )
    .unwrap();
    std::fs::write(tlib.join("helper.rs"), "pub fn v() -> u32 { 1 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--crate-type=rlib")
        .arg("--out-dir")
        .arg(&tlib)
        .arg(tlib.join("treelib.rs"))
        .output()
        .unwrap();
    assert!(out.status.success(), "treelib build failed");
    std::fs::write(
        tlib.join("uses.rs"),
        "extern crate treelib; fn main() { println!(\"lib says {}\", treelib::f()); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--extern")
        .arg(format!("treelib={}", tlib.join("libtreelib.rlib").display()))
        .arg("--out-dir")
        .arg(&tlib)
        .arg(tlib.join("uses.rs"))
        .output()
        .unwrap();
    assert!(out.status.success(), "tlib uses build failed");
    std::fs::write(tlib.join("helper.rs"), "pub fn v() -> u32 { 2 }\n").unwrap();
    let out = Command::new(tlib.join("uses")).output().unwrap();
    assert!(out.status.success(), "running tlib uses failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "lib says 1\n");
    std::fs::remove_file(tlib.join("treelib.rs")).unwrap();
    std::fs::remove_file(tlib.join("helper.rs")).unwrap();
    let out = Command::new(tlib.join("uses")).output().unwrap();
    assert!(out.status.success(), "running tlib uses after delete failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "lib says 1\n");

    // A bare program name invoked under a
    // command-specific `PATH` denotes the artifact that
    // search finds, not a namesake in the working
    // directory; the spelled name stays its argv[0].
    // Without the `PATH` entry the same bare name does
    // not resolve at all -- the working directory is not
    // searched.
    let pbin = dir.join("pbin");
    std::fs::create_dir_all(&pbin).unwrap();
    std::fs::write(
        pbin.join("pathprobe.rs"),
        "fn main() { println!(\"bin prog argv0={}\", std::env::args().next().unwrap()); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(&pbin)
        .arg(pbin.join("pathprobe.rs"))
        .output()
        .unwrap();
    assert!(out.status.success(), "pbin pathprobe build failed");
    build(
        &rustc,
        "pathprobe",
        "fn main() { println!(\"cwd prog\"); }\n",
        &[],
    );
    let out = Command::new("pathprobe")
        .env("PATH", &pbin)
        .output()
        .unwrap();
    assert!(out.status.success(), "running pathprobe via PATH failed");
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "bin prog argv0=pathprobe\n"
    );
    assert!(Command::new("pathprobe").output().is_err());

    println!("artifact fidelity guest ok");
}
"#;

#[test]
fn run_mode_tool_interception_built_binary_fidelity() {
    let dir = temp_dir("tool-artifact-fidelity");
    let src = fixture(&dir, "artifact_fidelity_guest.rs", ARTIFACT_FIDELITY_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "artifact fidelity guest ok\n", "stderr={stderr}");
}


const STATUS_RELAY_GUEST_SRC: &str = r#"use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    std::fs::write(
        "p.rs",
        "fn main() { println!(\"child out\"); eprintln!(\"child err\"); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("p.rs")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    println!("before");
    let st = Command::new(dir.join("p")).status().unwrap();
    assert!(st.success(), "status failed");
    println!("after");
}"#;

#[test]
fn run_mode_tool_interception_status_relays_inherited_streams() {
    let dir = temp_dir("tool-status-relay");
    let src = fixture(&dir, "status_relay_guest.rs", STATUS_RELAY_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // `status()` inherits the child's streams: the built
    // program's lines stand in the guest's own streams,
    // in invocation order around the guest's prints.
    assert_eq!(stdout, "before\nchild out\nafter\n", "stderr={stderr}");
    assert_eq!(stderr, "child err\n", "stdout={stdout}");
}

const FILE_SNAPSHOT_GUEST_SRC: &str = r#"use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    let fbin = dir.join("fbin");
    std::fs::create_dir_all(&fbin).unwrap();
    std::fs::write(
        fbin.join("fprobe.rs"),
        "fn main() { println!(\"{}\", file!()); }\n",
    )
    .unwrap();
    std::fs::write(fbin.join("other.txt"), "one\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(&fbin)
        .arg(fbin.join("fprobe.rs"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new(fbin.join("fprobe")).output().unwrap();
    assert!(out.status.success(), "first run failed");
    println!("first: {}", String::from_utf8(out.stdout).unwrap().trim());
    // Change a recorded file of the tree (not the root): the
    // next run converts from the snapshot.
    std::fs::write(fbin.join("other.txt"), "two\n").unwrap();
    let out = Command::new(fbin.join("fprobe")).output().unwrap();
    assert!(out.status.success(), "second run failed");
    println!("second: {}", String::from_utf8(out.stdout).unwrap().trim());
}"#;

#[test]
fn run_mode_tool_interception_snapshot_keeps_source_spellings() {
    let dir = temp_dir("tool-file-snapshot");
    let src = fixture(&dir, "file_snapshot_guest.rs", FILE_SNAPSHOT_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The second run converts from the source snapshot
    // (the guest rewrote a file of the tree); its
    // `file!()` spelling stays the recorded source's.
    let probe = dir.join("fbin").join("fprobe.rs");
    assert_eq!(
        stdout,
        format!("first: {0}\nsecond: {0}\n", probe.display()),
        "stderr={stderr}"
    );
}

const CWD_SEPARATION_GUEST_SRC: &str = r#"use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    std::fs::write(
        "prog.rs",
        "extern crate dep; fn main() { println!(\"prog cwd {}\", std::env::current_dir().unwrap().display()); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--extern")
        .arg("dep=libs/libdep.rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("prog.rs")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let other = dir.join("elsewhere");
    std::fs::create_dir_all(&other).unwrap();
    let out = Command::new(dir.join("prog"))
        .current_dir(&other)
        .output()
        .unwrap();
    assert!(out.status.success(), "run failed");
    println!("ran: {}", String::from_utf8(out.stdout).unwrap().trim());
}"#;

#[test]
fn run_mode_tool_interception_artifact_run_separates_directories() {
    let dir = temp_dir("tool-artifact-cwd");
    // A prebuilt dependency the run did not produce: a
    // real rlib from the pinned toolchain's own rustc,
    // which the guest names by a path relative to its
    // build directory.
    let libs = dir.join("libs");
    std::fs::create_dir_all(&libs).unwrap();
    let dep = fixture(&dir, "dep.rs", "pub fn f() -> u32 { 7 }\n");
    let rustc = Path::new(env!("SPARAGMOS_BUILD_SYSROOT"))
        .join("bin")
        .join("rustc");
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--crate-type=rlib")
        .arg("--crate-name")
        .arg("dep")
        .arg("-o")
        .arg(libs.join("libdep.rlib"))
        .arg(&dep)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "dep build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let src = fixture(&dir, "cwd_guest.rs", CWD_SEPARATION_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The run re-converts in the build's directory (the
    // relative `--extern` resolves there) and executes
    // in the directory the invocation requested.
    assert_eq!(
        stdout,
        format!("ran: prog cwd {}\n", dir.join("elsewhere").display()),
        "stderr={stderr}"
    );
}

const NEVER_BUILT_GUEST_SRC: &str = r#"
fn main() {
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    // The file exists and is executable, so the process
    // model's spawn gate passes; the run just never produced
    // it, so the interception cannot answer for it.
    let _out = std::process::Command::new(dir.join("never-built"))
        .output()
        .unwrap();
    println!("never-built guest unexpectedly ran the program");
}
"#;

#[test]
fn run_mode_tool_interception_unproduced_program_stays_unsupported() {
    let dir = temp_dir("tool-unproduced");
    let src = fixture(&dir, "never_built_guest.rs", NEVER_BUILT_GUEST_SRC);
    let fake = dir.join("never-built");
    std::fs::write(&fake, b"").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("is not an artifact this run produced"),
        "stderr={stderr}"
    );
}

const RELATIVE_CHILD_DIR_GUEST_SRC: &str = r#"use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    std::fs::create_dir_all("sub").unwrap();
    std::fs::write("sub/prog.rs", "fn main() { println!(\"child five\"); }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg("sub")
        .arg("sub/prog.rs")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new("./prog").current_dir("sub").output().unwrap();
    assert!(out.status.success(), "run failed");
    println!("got: {}", String::from_utf8(out.stdout).unwrap().trim());
}"#;

#[test]
fn run_mode_tool_interception_relative_child_dir_resolves_once() {
    let dir = temp_dir("tool-relative-child-dir");
    let src = fixture(
        &dir,
        "relative_child_dir_guest.rs",
        RELATIVE_CHILD_DIR_GUEST_SRC,
    );
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The spawn resolution applies the child's directory
    // to the spelled `./prog` once: the artifact lookup
    // stands the resolved `sub/./prog` on the directory the
    // resolution ran in, not on the execution directory
    // (which would look under `sub/sub/`).
    assert_eq!(stdout, "got: child five\n", "stderr={stderr}");
}

const STDIO_NULL_GUEST_SRC: &str = r#"use std::process::{Command, Stdio};

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    std::fs::write(
        "p.rs",
        "fn main() { println!(\"child out\"); eprintln!(\"child err\"); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("p.rs")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    println!("before");
    let st = Command::new(dir.join("p"))
        .stdout(Stdio::null())
        .status()
        .unwrap();
    assert!(st.success());
    let st = Command::new(dir.join("p"))
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(st.success());
    println!("after");
}"#;

#[test]
fn run_mode_tool_interception_status_honors_stdio_null() {
    let dir = temp_dir("tool-stdio-null");
    let src = fixture(&dir, "stdio_null_guest.rs", STDIO_NULL_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // Each `status()` call relays only the stream it
    // inherits: the stdout-nulled call contributes its
    // stderr line, the stderr-nulled call its stdout line,
    // and the nulled streams' bytes appear nowhere.
    assert_eq!(stdout, "before\nchild out\nafter\n", "stderr={stderr}");
    assert_eq!(stderr, "child err\n", "stdout={stdout}");
}

const STDIO_OUTPUT_GUEST_SRC: &str = r#"use std::process::{Command, Stdio};

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    std::fs::write(
        "p.rs",
        "fn main() { println!(\"child out\"); eprintln!(\"child err\"); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("p.rs")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new(dir.join("p"))
        .stdout(Stdio::inherit())
        .output()
        .unwrap();
    println!(
        "inherit collected stdout: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
    println!(
        "inherit collected stderr: {:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new(dir.join("p"))
        .stderr(Stdio::null())
        .output()
        .unwrap();
    println!(
        "null collected stdout: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
    println!(
        "null collected stderr: {:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new(dir.join("p"))
        .stdout(Stdio::piped())
        .output()
        .unwrap();
    println!(
        "piped collected stdout: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}"#;

#[test]
fn run_mode_tool_interception_output_honors_stdio_overrides() {
    let dir = temp_dir("tool-stdio-output");
    let src = fixture(&dir, "stdio_output_guest.rs", STDIO_OUTPUT_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // Under `output()`, an inherited stream joins the
    // guest's own stream instead of the `Output`, a nulled
    // stream is neither collected nor shown, and an
    // explicit pipe collects as the default does.
    assert_eq!(
        stdout,
        "child out\n\
         inherit collected stdout: \"\"\n\
         inherit collected stderr: \"child err\\n\"\n\
         null collected stdout: \"child out\\n\"\n\
         null collected stderr: \"\"\n\
         piped collected stdout: \"child out\\n\"\n",
        "stderr={stderr}"
    );
    assert_eq!(stderr, "", "stdout={stdout}");
}

const NON_UTF8_ARG_GUEST_SRC: &str = r#"use std::os::unix::ffi::OsStringExt;
use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    std::fs::write(
        "p.rs",
        "fn main() { println!(\"child ran\"); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("p.rs")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let arg = std::ffi::OsString::from_vec(vec![b'a', 0xff, b'b']);
    let out = Command::new(dir.join("p")).arg(arg).output().unwrap();
    assert!(out.status.success(), "run failed");
    println!("arg guest unexpectedly ran the program");
}"#;

#[test]
fn run_mode_tool_interception_non_utf8_argument_is_unsupported() {
    let dir = temp_dir("tool-non-utf8-arg");
    let src = fixture(&dir, "non_utf8_arg_guest.rs", NON_UTF8_ARG_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    // The argument's bytes cannot be carried to the run
    // unchanged, so the invocation is unsupported with the
    // reason named -- never run with a lossy replacement.
    assert_eq!(
        out.status.code(),
        Some(2),
        "stdout={stdout} stderr={stderr}"
    );
    assert!(
        stderr.contains("a guest argument is not valid UTF-8"),
        "stderr={stderr}"
    );
}

const ADDED_MODULE_CANDIDATE_GUEST_SRC: &str = r#"use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    let tree = dir.join("mtree");
    std::fs::create_dir_all(&tree).unwrap();
    std::fs::write(
        tree.join("modprobe.rs"),
        "mod helper; fn main() { println!(\"value {}\", helper::v()); }\n",
    )
    .unwrap();
    std::fs::write(tree.join("helper.rs"), "pub fn v() -> u32 { 1 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(&tree)
        .arg(tree.join("modprobe.rs"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    // Every recorded file stands unchanged, but a second
    // module candidate appears beside the one the build
    // resolved: the live tree no longer describes the
    // recorded conversion (a fresh conversion of it fails
    // with E0761), while the program the build produced
    // is unchanged.
    std::fs::create_dir_all(tree.join("helper")).unwrap();
    std::fs::write(
        tree.join("helper").join("mod.rs"),
        "pub fn v() -> u32 { 2 }\n",
    )
    .unwrap();
    let out = Command::new(tree.join("modprobe")).output().unwrap();
    assert!(out.status.success(), "run failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "value 1\n");
    println!("added module candidate guest ok");
}
"#;

#[test]
fn run_mode_tool_interception_added_module_candidate_uses_snapshot() {
    let dir = temp_dir("tool-added-module-candidate");
    let src = fixture(
        &dir,
        "added_module_candidate_guest.rs",
        ADDED_MODULE_CANDIDATE_GUEST_SRC,
    );
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The added `helper/mod.rs` ends the recorded tree's
    // intact reading, so the run converts the recorded
    // snapshot (which holds only `helper.rs`) and the
    // built program runs as built.
    assert_eq!(
        stdout, "added module candidate guest ok\n",
        "stderr={stderr}"
    );
}

const ENV_GUEST_SRC: &str = r#"
fn main() {
    assert!(std::env::var_os("SPARAGMOS_NO_SUCH_VAR").is_none());
    match std::env::var("SPARAGMOS_NO_SUCH_VAR") {
        Err(std::env::VarError::NotPresent) => {}
        other => panic!("expected NotPresent, got {other:?}"),
    }
    std::env::set_var("SPARAGMOS_GUEST_VAR", "hello");
    assert_eq!(std::env::var("SPARAGMOS_GUEST_VAR").unwrap(), "hello");
    assert_eq!(
        std::env::var_os("SPARAGMOS_GUEST_VAR").unwrap().to_str().unwrap(),
        "hello"
    );
    std::env::remove_var("SPARAGMOS_GUEST_VAR");
    assert!(std::env::var_os("SPARAGMOS_GUEST_VAR").is_none());
    let cwd = std::env::current_dir().unwrap();
    assert!(cwd.is_absolute());
    assert!(std::env::temp_dir().is_absolute());
    println!("env guest ok");
}
"#;

#[test]
fn run_mode_env_builtins_roundtrip() {
    let dir = temp_dir("env-guest");
    let src = fixture(&dir, "env_guest.rs", ENV_GUEST_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "env guest ok\n", "stderr={stderr}");
}

const FS_GUEST_SRC: &str = r#"
fn main() {
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().join("work");
    std::fs::create_dir_all(dir.join("a/b")).unwrap();
    std::fs::write(dir.join("a/b/f.txt"), b"abc").unwrap();
    assert_eq!(
        std::fs::read(dir.join("a/b/f.txt")).unwrap(),
        vec![b'a', b'b', b'c']
    );
    let md = std::fs::metadata(dir.join("a/b/f.txt")).unwrap();
    assert!(md.is_file());
    assert_eq!(md.len(), 3);
    let mut names = Vec::new();
    for e in std::fs::read_dir(dir.join("a")).unwrap() {
        let e = e.unwrap();
        names.push(e.file_name().to_str().unwrap().to_string());
    }
    assert_eq!(names.len(), 1);
    assert_eq!(names[0], "b");
    match std::fs::read(dir.join("missing")) {
        Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
        Ok(_) => panic!("missing file read succeeded"),
    }
    match std::fs::create_dir(dir.join("a")) {
        Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::AlreadyExists),
        Ok(_) => panic!("second create_dir succeeded"),
    }
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(!dir.exists());
    println!("fs guest ok");
}
"#;

#[test]
fn run_mode_fs_builtins_roundtrip() {
    let dir = temp_dir("fs-guest");
    let src = fixture(&dir, "fs_guest.rs", FS_GUEST_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "fs guest ok\n", "stderr={stderr}");
}

const NESTED_REVERSE_SRC: &str = r#"
fn main() {
    let mut v: Vec<[u32; 4]> = vec![[1, 2, 3, 4], [5, 6, 7, 8]];
    v[0].reverse();
    v[1].reverse();
    println!("{:?}", v);
    let mut a = [[1u32, 2, 3], [4, 5, 6]];
    a[1].reverse();
    println!("{:?}", a);
}
"#;

#[test]
fn run_mode_nested_aggregate_element_reverse() {
    // Reversing one row of a nested aggregate (`v[0].reverse()`)
    // swaps through pointers displaced from an element position
    // inside the row: the displacement must absorb into the
    // row's own index space (the innermost container the
    // pointer's element size denominates), or the swap reads and
    // writes the wrong elements — it used to splice bytes into
    // the first element's storage and leave the row's tail
    // untouched.
    let dir = temp_dir("nestedreverse");
    let src = fixture(&dir, "nestedreverse.rs", NESTED_REVERSE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested aggregate element reverse must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[[4, 3, 2, 1], [8, 7, 6, 5]]\n[[1, 2, 3], [6, 5, 4]]\n",
        "each row reverses in place, exactly as natively"
    );
}

const NESTED_VEC_SRC: &str = r#"
fn main() {
    let mut v: Vec<Vec<u32>> = vec![vec![1, 2, 3, 4], vec![5, 6]];
    assert_eq!(v[0][1], 2);
    assert_eq!(v[1][0], 5);
    v[0].reverse();
    v[1].push(7);
    assert_eq!(v[0], vec![4, 3, 2, 1]);
    assert_eq!(v[1][2], 7);
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_nested_vec_index_and_reverse() {
    // A `Vec` value is the aggregate of its elements, so an
    // index step into it is element access and a following index
    // step ranges inside it: composing the two steps into one
    // index over the outer vector (the flat reading a data
    // pointer's position takes) reads the sibling vector whole
    // instead of the element. Nested reads, writes through
    // `v[0].reverse()`'s swaps, and growth of an inner vector
    // must all observe the inner elements.
    let dir = temp_dir("nestedvec");
    let src = fixture(&dir, "nestedvec.rs", NESTED_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested vec index/reverse must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[[4, 3, 2, 1], [5, 6, 7]]\n",
        "the inner vectors reverse and grow independently, exactly as natively"
    );
}

const PATH_ORDERING_SRC: &str = r#"
use std::cmp::Ordering;
use std::path::Path;

fn main() {
    // std's `Path` ordering compares the parsed component
    // sequences, not the raw bytes: the first components
    // here are "a" and "a.b", so "a/c" < "a.b/c" — byte
    // order would give the opposite ('.' = 0x2e < '/' =
    // 0x2f).
    assert_eq!(Path::new("a/c").cmp(Path::new("a.b/c")), Ordering::Less);
    assert!(Path::new("a/c") < Path::new("a.b/c"));
    assert!(Path::new("a.b/c") > Path::new("a/c"));
    assert!(Path::new("a/c") <= Path::new("a.b/c"));
    // Component kinds order by kind before any name bytes:
    // root directory, then current directory, then parent
    // directory, then normal names.
    assert_eq!(Path::new("/a").cmp(Path::new("./a")), Ordering::Less);
    assert_eq!(Path::new("./a").cmp(Path::new("../a")), Ordering::Less);
    assert_eq!(Path::new("../a").cmp(Path::new("a")), Ordering::Less);
    assert_eq!(Path::new("/a").cmp(Path::new("a")), Ordering::Less);
    // A strict component prefix is smaller, and paths whose
    // components parse the same compare equal.
    assert_eq!(Path::new("a").cmp(Path::new("a/b")), Ordering::Less);
    assert_eq!(Path::new("a/..").cmp(Path::new("a")), Ordering::Greater);
    assert_eq!(Path::new("foo").cmp(Path::new("foo")), Ordering::Equal);
    assert_eq!(Path::new("a//b").cmp(Path::new("a/b")), Ordering::Equal);
    assert_eq!(
        Path::new("a/b").partial_cmp(Path::new("a/b")),
        Some(Ordering::Equal)
    );
    println!("path-ordering-ok");
}
"#;

#[test]
fn run_mode_path_component_ordering() {
    run_ok("path-ordering", PATH_ORDERING_SRC, "path-ordering-ok\n");
}

const PATH_SORT_SRC: &str = r#"
use std::ffi::OsString;
use std::path::PathBuf;

fn main() {
    // Sorting owned buffers goes through the generic sort
    // machinery, whose comparisons carry no type tag; the
    // order must still be the component order.
    let mut v = vec![
        PathBuf::from("a.b/c"),
        PathBuf::from("a/c"),
        PathBuf::from("/x"),
    ];
    v.sort();
    assert_eq!(v[0], PathBuf::from("/x"));
    assert_eq!(v[1], PathBuf::from("a/c"));
    assert_eq!(v[2], PathBuf::from("a.b/c"));
    let mut w = vec![PathBuf::from("a.b/c"), PathBuf::from("a/c")];
    w.sort_unstable();
    assert_eq!(w[0], PathBuf::from("a/c"));
    assert_eq!(w[1], PathBuf::from("a.b/c"));
    // Binary search uses the same ordering, and a
    // component-equal key ("a//c") finds its entry.
    assert_eq!(v.binary_search(&PathBuf::from("a/c")), Ok(1));
    assert!(v.binary_search(&PathBuf::from("a//c")).is_ok());
    // The neighbouring string-like types keep their byte
    // order in the same generic machinery.
    let mut oss = vec![OsString::from("a/c"), OsString::from("a.b/c")];
    oss.sort();
    assert_eq!(oss[0], OsString::from("a.b/c"));
    let mut vecs = vec![b"a/c".to_vec(), b"a.b/c".to_vec()];
    vecs.sort();
    assert_eq!(vecs[0], b"a.b/c".to_vec());
    println!("path-sort-ok");
}
"#;

#[test]
fn run_mode_pathbuf_sort_component_order() {
    run_ok("path-sort", PATH_SORT_SRC, "path-sort-ok\n");
}

const PATH_ANCESTORS_SRC: &str = r#"
use std::path::Path;

fn main() {
    // `ancestors` yields the path itself, then each parent
    // up to (and including) the root / the empty path.
    let anc: Vec<&Path> = Path::new("/a/b/c").ancestors().collect();
    assert_eq!(
        anc,
        vec![
            Path::new("/a/b/c"),
            Path::new("/a/b"),
            Path::new("/a"),
            Path::new("/")
        ]
    );
    let anc2: Vec<String> = Path::new("a/b")
        .ancestors()
        .map(|p| p.to_str().unwrap().to_string())
        .collect();
    assert_eq!(
        anc2,
        vec!["a/b".to_string(), "a".to_string(), "".to_string()]
    );
    assert_eq!(Path::new("").ancestors().count(), 1);
    assert_eq!(Path::new("/").ancestors().count(), 1);
    // The parents are lexical: a trailing separator takes
    // no part, and `..` components are kept as components.
    let anc3: Vec<String> = Path::new("a/b/")
        .ancestors()
        .map(|p| p.to_str().unwrap().to_string())
        .collect();
    assert_eq!(
        anc3,
        vec!["a/b/".to_string(), "a".to_string(), "".to_string()]
    );
    let anc4: Vec<String> = Path::new("a/../b")
        .ancestors()
        .map(|p| p.to_str().unwrap().to_string())
        .collect();
    assert_eq!(
        anc4,
        vec![
            "a/../b".to_string(),
            "a/..".to_string(),
            "a".to_string(),
            "".to_string()
        ]
    );
    // An exhausted iterator stays exhausted.
    let mut it = Path::new("x/y").ancestors();
    assert_eq!(it.next().unwrap(), Path::new("x/y"));
    assert_eq!(it.next().unwrap(), Path::new("x"));
    assert_eq!(it.next().unwrap(), Path::new(""));
    assert!(it.next().is_none());
    assert!(it.next().is_none());
    println!("path-ancestors-ok");
}
"#;

#[test]
fn run_mode_path_ancestors() {
    run_ok("path-ancestors", PATH_ANCESTORS_SRC, "path-ancestors-ok\n");
}

const PATH_SET_EXTENSION_SRC: &str = r#"
use std::path::PathBuf;

fn main() {
    let mut pb = PathBuf::from("foo.rs");
    assert!(pb.set_extension("txt"));
    assert_eq!(pb.to_str().unwrap(), "foo.txt");
    // An empty extension removes the old one (and the dot).
    assert!(pb.set_extension(""));
    assert_eq!(pb.to_str().unwrap(), "foo");
    // A path without an extension gains one.
    let mut bare = PathBuf::from("foo");
    assert!(bare.set_extension("md"));
    assert_eq!(bare.to_str().unwrap(), "foo.md");
    // No file name: the buffer is left alone, result false.
    let mut dot = PathBuf::from("..");
    assert!(!dot.set_extension("txt"));
    assert_eq!(dot.to_str().unwrap(), "..");
    let mut root = PathBuf::from("/");
    assert!(!root.set_extension("txt"));
    assert_eq!(root.to_str().unwrap(), "/");
    // `with_extension` replaces only the final extension.
    let w = PathBuf::from("a/b.tar.gz").with_extension("zip");
    assert_eq!(w.to_str().unwrap(), "a/b.tar.zip");
    let w2 = PathBuf::from("a/b").with_extension("md");
    assert_eq!(w2.to_str().unwrap(), "a/b.md");
    // The extension argument may be an OsStr too.
    let w3 = PathBuf::from("a/b.rs").with_extension(std::ffi::OsStr::new("toml"));
    assert_eq!(w3.to_str().unwrap(), "a/b.toml");
    println!("path-set-extension-ok");
}
"#;

#[test]
fn run_mode_pathbuf_set_extension() {
    run_ok(
        "path-set-extension",
        PATH_SET_EXTENSION_SRC,
        "path-set-extension-ok\n",
    );
}

// ---------------------------------------------------------------------
// C-variadic (VaList) value model: a variadic callee's tail
// arguments are packed into one ordered list at the call boundary;
// `next_arg` reads them back in order, `clone` forks the cursor,
// and a `VaList` parameter forwards the same list.
// ---------------------------------------------------------------------

const VALIST_BASIC_SRC: &str = r#"
use std::ffi::*;
unsafe extern "C" fn sum3(mut ap: ...) -> i64 {
    let mut total: i64 = 0;
    for _ in 0..3 {
        total += ap.next_arg::<c_int>() as i64;
    }
    total
}
fn main() {
    let s = unsafe { sum3(10 as c_int, 20 as c_int, 30 as c_int) };
    println!("sum={s}");
}
"#;

const VALIST_MIXED_SRC: &str = r#"
use std::ffi::*;
unsafe extern "C" fn weighted(mut ap: ...) -> f64 {
    let w = ap.next_arg::<c_double>();
    let x = ap.next_arg::<c_int>();
    let y = ap.next_arg::<c_double>();
    w * x as f64 + y
}
fn main() {
    let w = unsafe { weighted(2.5 as c_double, 4 as c_int, 0.5 as c_double) };
    println!("weighted={w}");
}
"#;

const VALIST_CLONE_SRC: &str = r#"
use std::ffi::*;
unsafe extern "C" fn clone_check(mut ap: ...) -> i64 {
    let mut ap2 = ap.clone();
    let a = ap2.next_arg::<c_long>();
    let b = ap.next_arg::<c_long>();
    a * 100 + b
}
fn main() {
    let c = unsafe { clone_check(7 as c_long, 9 as c_long) };
    println!("clone={c}");
}
"#;

const VALIST_FORWARD_SRC: &str = r#"
use std::ffi::*;
unsafe extern "C" fn helper(mut ap: VaList) -> i64 {
    ap.next_arg::<c_long>() + ap.next_arg::<c_long>()
}
unsafe extern "C" fn outer(ap: ...) -> i64 {
    unsafe { helper(ap) }
}
fn main() {
    let v = unsafe { outer(40 as c_long, 2 as c_long) };
    println!("fwd={v}");
}
"#;

const VALIST_FIXED_AND_TAIL_SRC: &str = r#"
use std::ffi::*;
unsafe extern "C" fn scaled(scale: c_int, mut ap: ...) -> i64 {
    let a = ap.next_arg::<c_long>();
    let b = ap.next_arg::<c_long>();
    scale as i64 * (a + b)
}
fn main() {
    let v = unsafe { scaled(3 as c_int, 10 as c_long, 5 as c_long) };
    println!("scaled={v}");
}
"#;

#[test]
fn run_mode_valist_basic() {
    run_ok_edition("valist-basic", VALIST_BASIC_SRC, "sum=60\n");
}

#[test]
fn run_mode_valist_mixed_types() {
    run_ok_edition("valist-mixed", VALIST_MIXED_SRC, "weighted=10.5\n");
}

#[test]
fn run_mode_valist_clone_independent_cursor() {
    run_ok_edition("valist-clone", VALIST_CLONE_SRC, "clone=707\n");
}

#[test]
fn run_mode_valist_forwarding() {
    run_ok_edition("valist-forward", VALIST_FORWARD_SRC, "fwd=42\n");
}

#[test]
fn run_mode_valist_fixed_and_tail() {
    run_ok_edition("valist-fixed-tail", VALIST_FIXED_AND_TAIL_SRC, "scaled=45\n");
}
const ARRAY_CLONE_UNWIND_MULTIFIELD_SRC: &str = r#"
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
static N: AtomicUsize = AtomicUsize::new(0);
struct S {
    id: u32,
    rc: Rc<()>,
}
impl Clone for S {
    fn clone(&self) -> Self {
        let n = N.fetch_add(1, Ordering::SeqCst);
        if n == 2 {
            panic!("oops on third clone");
        }
        S { id: self.id, rc: self.rc.clone() }
    }
}
fn main() {
    let counter = Rc::new(());
    let arr = [
        S { id: 0, rc: counter.clone() },
        S { id: 1, rc: counter.clone() },
        S { id: 2, rc: counter.clone() },
        S { id: 3, rc: counter.clone() },
    ];
    println!("built: {}", Rc::strong_count(&counter));
    let child = std::panic::catch_unwind(move || {
        let _ = arr.clone();
    });
    println!("is_err: {}", child.is_err());
    println!("count after: {}", Rc::strong_count(&counter));
}
"#;
#[test]
fn run_mode_array_clone_unwind_drops_cloned_prefix() {
    // Cloning an array of multi-field elements through the
    // spec-clone guard: when an element's `Clone` panics, the
    // guard drops exactly the already-cloned prefix. The guard's
    // erased buffer pointer denotes element 0 entered through
    // its `MaybeUninit` wrapper; the drop's slice length is the
    // cloned count, which for a two-field element coincides with
    // the element's own field count -- the two must not be
    // confused when the drop's base place is composed.
    let dir = temp_dir("array-clone-unwind-multifield");
    let src = fixture(&dir, "array_clone_unwind.rs", ARRAY_CLONE_UNWIND_MULTIFIELD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "built: 5\nis_err: true\ncount after: 1\n"
    );
}
const MAYBE_UNINIT_ARRAY_BUFFER_SRC: &str = r#"
use std::mem::MaybeUninit;
use std::rc::Rc;
struct S {
    id: u32,
    rc: Rc<()>,
}
impl Clone for S {
    fn clone(&self) -> Self {
        S { id: self.id, rc: self.rc.clone() }
    }
}
fn main() {
    let counter = Rc::new(());
    let mut buf: MaybeUninit<[S; 4]> = MaybeUninit::uninit();
    let ptr = buf.as_mut_ptr() as *mut S;
    unsafe {
        for i in 0..4 {
            ptr.add(i).write(S { id: i as u32, rc: counter.clone() });
        }
        let arr = buf.assume_init();
        println!("after assume_init: {}", Rc::strong_count(&counter));
        println!("ids: {} {} {} {}", arr[0].id, arr[1].id, arr[2].id, arr[3].id);
        let arr2 = arr.clone();
        println!("after clone: {}", Rc::strong_count(&counter));
        drop(arr2);
        drop(arr);
    }
    println!("final: {}", Rc::strong_count(&counter));
}
"#;
#[test]
fn run_mode_maybe_uninit_array_buffer_element_access() {
    // A whole-buffer `MaybeUninit<[S; 4]>`: the `uninit()`
    // construction must materialize the payload slot as the
    // array's uninitialized skeleton (the model's field-0
    // payload convention), so element writes through
    // `as_mut_ptr` land in array slots and `assume_init` reads
    // the filled array back.
    let dir = temp_dir("maybe-uninit-array-buffer");
    let src = fixture(&dir, "maybe_uninit_buffer.rs", MAYBE_UNINIT_ARRAY_BUFFER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "after assume_init: 5\nids: 0 1 2 3\nafter clone: 9\nfinal: 1\n"
    );
}
const NESTED_MAYBE_UNINIT_BUFFER_SRC: &str = r#"
use std::mem::MaybeUninit;
fn main() {
    let mut buf: MaybeUninit<MaybeUninit<[u32; 2]>> = MaybeUninit::uninit();
    let ptr = buf.as_mut_ptr() as *mut u32;
    unsafe {
        ptr.write(11);
        ptr.add(1).write(22);
        let inner = buf.assume_init();
        let arr = inner.assume_init();
        println!("nested: {} {}", arr[0], arr[1]);
    }
}
"#;
#[test]
fn run_mode_nested_maybe_uninit_buffer_cast() {
    // `buf.as_mut_ptr() as *mut u32` over a nested
    // `MaybeUninit<MaybeUninit<[u32; 2]>>` crosses both wrapper
    // layers and the array layer in a single cast; the resulting
    // pointer denotes the first array element, so element writes
    // through it fill the array's slots in order.
    let dir = temp_dir("nested-maybe-uninit-buffer");
    let src = fixture(&dir, "nested_maybe_uninit.rs", NESTED_MAYBE_UNINIT_BUFFER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "nested: 11 22\n");
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

const IO_ERROR_INTO_INNER_SRC: &str = r#"
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

static DROPS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct Counted { code: i32 }
impl fmt::Display for Counted {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "counted {}", self.code) }
}
impl std::error::Error for Counted {}
impl Drop for Counted {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
        println!("dropping Counted {}", self.code);
    }
}

fn main() {
    let e = std::io::Error::new(std::io::ErrorKind::InvalidInput, Counted { code: 5 });
    match e.into_inner() {
        Some(boxed) => {
            println!("got inner");
            println!("drops before box drop: {}", DROPS.load(Ordering::SeqCst));
            drop(boxed);
            println!("drops after box drop: {}", DROPS.load(Ordering::SeqCst));
        }
        None => println!("no inner"),
    }
    println!("done");
}
"#;

#[test]
fn run_mode_io_error_into_inner_drops_payload_once() {
    // `io::Error::into_inner` rebuilds the payload's
    // `Box<dyn Error + Send + Sync>` with `Box::from_raw` over
    // the pointer the packed representation carried. Reading
    // the pointee back through that cast descends the pointer
    // wrappers (`Box` over `Unique` over the raw pointer, each
    // carrying trailing zero-sized fields), and dropping the
    // rebuilt box reaches the `DynRef` at the bottom of the
    // same wrapper chain - the payload destructor runs exactly
    // once, when the box drops, not inside `into_inner`.
    let dir = temp_dir("io-error-into-inner");
    let src = fixture(&dir, "io_error_into_inner.rs", IO_ERROR_INTO_INNER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "got inner\n\
         drops before box drop: 0\n\
         dropping Counted 5\n\
         drops after box drop: 1\n\
         done\n"
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

const BOX_DYN_TAIL_STRUCT_SRC: &str = r#"
struct Wrapper<'a, T: ?Sized>(&'a mut i32, T);
impl<'a, T: ?Sized> Drop for Wrapper<'a, T> {
    fn drop(&mut self) { *self.0 = 432; }
}
fn main() {
    let mut x = 0;
    {
        let wrapper = Box::new(Wrapper(&mut x, [1u32, 2, 3]));
        let _: Box<Wrapper<dyn Send>> = wrapper;
    }
    println!("x is {}", x);
}
"#;

#[test]
fn run_mode_box_dyn_tail_struct_drop_runs_dtor() {
    // A `Box` over a struct whose unsized tail is a trait object:
    // the box value is the trait-object form the unsizing
    // coercion produced, and dropping it runs the struct's
    // destructor against the pointee in the allocation — the
    // write through the borrowed field reaches the caller's
    // local, exactly as the sized box's drop does.
    let dir = temp_dir("box-dyn-tail-struct-drop");
    let src = fixture(&dir, "box_dyn_tail_struct_drop.rs", BOX_DYN_TAIL_STRUCT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<dyn-tail struct> drop program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "x is 432\n",
        "the dyn-tail struct's destructor runs at the box drop"
    );
}

const RC_DYN_TAIL_STRUCT_SRC: &str = r#"
use std::rc::Rc;
struct Wrapper<T: ?Sized>(i32, T);
impl<T: ?Sized> Drop for Wrapper<T> {
    fn drop(&mut self) { println!("dtor {}", self.0); }
}
fn main() {
    let r: Rc<Wrapper<dyn Send>> = Rc::new(Wrapper(5, [1u32, 2]));
    let r2 = r.clone();
    drop(r);
    println!("one left");
    drop(r2);
    println!("gone");
}
"#;

#[test]
fn run_mode_rc_dyn_tail_struct_drop_runs_dtor() {
    // The reference-counted form of the dyn-tail shape: the
    // handle value is the trait-object form, and the pointee's
    // destructor runs when the last strong handle drops.
    let dir = temp_dir("rc-dyn-tail-struct-drop");
    let src = fixture(&dir, "rc_dyn_tail_struct_drop.rs", RC_DYN_TAIL_STRUCT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Rc<dyn-tail struct> drop program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "one left\ndtor 5\ngone\n",
        "the dyn-tail struct's destructor runs at the last handle drop"
    );
}

const BOX_POINTEE_MENTIONS_DYN_SRC: &str = r#"
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
struct Holder(Vec<Box<dyn Fn()>>);
impl Drop for Holder {
    fn drop(&mut self) { println!("holder dtor"); }
}
fn main() {
    let m = M(9);
    let h = Box::new(Holder(vec![Box::new(move || { let _ = &m; })]));
    drop(h);
    println!("done");
}
"#;

#[test]
fn run_mode_box_pointee_mentions_dyn_plain_form_drop() {
    // A pointee that merely mentions a trait object (a `Vec` of
    // boxed closures in a field) is a plain box value, not the
    // trait-object form: dropping it runs the pointee's own
    // destructor and its fields' glue as usual.
    let dir = temp_dir("box-pointee-mentions-dyn-drop");
    let src = fixture(&dir, "box_pointee_mentions_dyn.rs", BOX_POINTEE_MENTIONS_DYN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "plain Box over a dyn-mentioning pointee must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "holder dtor\ndtor 9\ndone\n",
        "the plain box's pointee destructor and field glue run at the drop"
    );
}

const RC_DYN_TRAIT_COUNT_SRC: &str = r#"
use std::rc::Rc;
trait Tr { fn hi(&self) -> i32; }
struct S(i32);
impl Tr for S { fn hi(&self) -> i32 { self.0 } }
impl Drop for S { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let r: Rc<dyn Tr> = Rc::new(S(5));
    let r2 = r.clone();
    println!("strong {}", Rc::strong_count(&r));
    drop(r);
    println!("one left");
    drop(r2);
    println!("gone");
}
"#;

#[test]
fn run_mode_rc_dyn_trait_clone_and_drop_counts() {
    // A cloned `Rc<dyn Trait>` handle is a second live handle on
    // the same allocation: the strong count moves with the clone
    // (the handle value is in trait-object form, and the count
    // machinery sees through it), and the pointee's destructor
    // runs only when the last handle drops.
    let dir = temp_dir("rc-dyn-trait-count");
    let src = fixture(&dir, "rc_dyn_trait_count.rs", RC_DYN_TRAIT_COUNT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Rc<dyn Trait> count program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "strong 2\none left\ndtor 5\ngone\n",
        "the count moves with the clone and the destructor runs at the last drop"
    );
}

const SPARSE_VEC_SRC: &str = r#"
fn main() {
    // A length past the value model's materialisation limit is
    // held sparsely: the length is exact, the stored prefix is
    // kept, and the tail denotes uninitialised slots.
    let mut vec: Vec<String> = Vec::with_capacity(5_000_000);
    unsafe { vec.set_len(5_000_000) }
    assert_eq!(vec.len(), 5_000_000);
    drop(vec);
    println!("sparse string drop ok");

    let mut v2: Vec<String> = Vec::new();
    v2.push("a".to_string());
    v2.push("bb".to_string());
    unsafe { v2.set_len(5_000_000) }
    assert_eq!(v2.len(), 5_000_000);
    assert_eq!(v2[0], "a");
    assert_eq!(v2[1], "bb");
    drop(v2);
    println!("sparse prefix drop ok");

    let mut v3: Vec<u8> = Vec::with_capacity(5_000_000);
    unsafe { v3.set_len(5_000_000) }
    let s = &v3[..];
    assert_eq!(s.len(), 5_000_000);
    assert_eq!(v3.len(), 5_000_000);
    drop(v3);
    println!("sparse u8 ok");

    // Shrinking back under the limit materialises the vector.
    let mut v4: Vec<u8> = Vec::with_capacity(5_000_000);
    unsafe { v4.set_len(5_000_000) }
    unsafe { v4.set_len(3) }
    assert_eq!(v4.len(), 3);
    v4.push(9);
    assert_eq!(v4[3], 9);
    println!("shrink ok");

    // Growing a materialised vector past the limit keeps its
    // stored elements.
    let mut v5: Vec<u8> = vec![1, 2, 3];
    unsafe { v5.set_len(5_000_000) }
    assert_eq!(v5.len(), 5_000_000);
    assert_eq!(v5[0], 1);
    assert_eq!(v5[2], 3);
    let s5 = &v5[..];
    assert_eq!(s5.len(), 5_000_000);
    drop(v5);
    println!("grow ok");
}
"#;

#[test]
fn run_mode_sparse_vec_set_len() {
    run_ok(
        "sparse-vec",
        SPARSE_VEC_SRC,
        "sparse string drop ok\nsparse prefix drop ok\nsparse u8 ok\nshrink ok\ngrow ok\n",
    );
}

const SPARSE_VEC_UNINIT_READ_SRC: &str = r#"
fn main() {
    let mut vec: Vec<u8> = Vec::with_capacity(5_000_000);
    unsafe { vec.set_len(5_000_000) }
    // A tail slot was never written: reading it is the same
    // uninitialised read a materialised `set_len` vector's
    // padding raises.
    println!("{}", vec[0]);
}
"#;

#[test]
fn run_mode_sparse_vec_uninit_read() {
    let dir = temp_dir("sparse-vec-uninit");
    let src = fixture(&dir, "case.rs", SPARSE_VEC_UNINIT_READ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("uninitialized"),
        "tail read must be an uninitialised read; stderr={stderr}"
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

const MAYBE_UNINIT_COPY_MOVE_SRC: &str = r#"
fn main() {
    let m: std::mem::MaybeUninit<u8> = std::mem::MaybeUninit::uninit();
    unsafe {
        let mut b: std::mem::MaybeUninit<u8> = std::mem::MaybeUninit::uninit();
        std::ptr::copy_nonoverlapping(&m, &mut b, 1);
        let c = b;
        std::mem::forget(c);
    }
    println!("done");
}
"#;

#[test]
fn run_mode_maybe_uninit_copy_then_move_uninit() {
    // A `MaybeUninit` value whose payload is uninitialized is
    // the type's ordinary content: copying it element-wise and
    // moving the destination place carries the union value
    // (only a read of the payload itself is an uninitialized
    // read).
    run_ok("maybe-uninit-copy-move", MAYBE_UNINIT_COPY_MOVE_SRC, "done\n");
}

const MAYBE_UNINIT_COPY_DEFINED_SRC: &str = r#"
fn main() {
    let m: std::mem::MaybeUninit<u8> = std::mem::MaybeUninit::new(7);
    unsafe {
        let mut b: std::mem::MaybeUninit<u8> = std::mem::MaybeUninit::uninit();
        std::ptr::copy_nonoverlapping(&m, &mut b, 1);
        let c = b;
        println!("val={}", c.assume_init());
    }
}
"#;

#[test]
fn run_mode_maybe_uninit_copy_defined_assume_init() {
    // The element a `MaybeUninit`-typed copy moves is the union
    // value, not the bare payload: the destination place keeps
    // the union form its own `assume_init` reads.
    run_ok(
        "maybe-uninit-copy-defined",
        MAYBE_UNINIT_COPY_DEFINED_SRC,
        "val=7\n",
    );
}

const MAYBE_UNINIT_ALLOC_READ_SRC: &str = r#"
fn main() {
    unsafe {
        let layout = std::alloc::Layout::new::<std::mem::MaybeUninit<u8>>();
        let p = std::alloc::alloc(layout) as *mut std::mem::MaybeUninit<u8>;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let b: std::mem::MaybeUninit<u8> = std::ptr::read(p);
        std::mem::forget(b);
        std::alloc::dealloc(p as *mut u8, layout);
    }
    println!("done");
}
"#;

#[test]
fn run_mode_maybe_uninit_ptr_read_unwritten_alloc() {
    // A `MaybeUninit` place inside a raw allocation no write has
    // ever touched holds the union's uninitialized form, the
    // type's ordinary content: `ptr::read` of it succeeds (only
    // reading the payload itself is an uninitialized read).
    run_ok(
        "maybe-uninit-alloc-read",
        MAYBE_UNINIT_ALLOC_READ_SRC,
        "done\n",
    );
}

const MAYBE_UNINIT_ALLOC_READ_WRITTEN_SRC: &str = r#"
fn main() {
    unsafe {
        let layout = std::alloc::Layout::new::<std::mem::MaybeUninit<u8>>();
        let p = std::alloc::alloc(layout) as *mut std::mem::MaybeUninit<u8>;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        std::ptr::write(p, std::mem::MaybeUninit::new(41));
        let b: std::mem::MaybeUninit<u8> = std::ptr::read(p);
        println!("val={}", b.assume_init());
        std::alloc::dealloc(p as *mut u8, layout);
    }
}
"#;

#[test]
fn run_mode_maybe_uninit_ptr_read_written_alloc() {
    // The same read over storage a write did fill assembles the
    // payload the write stored.
    run_ok(
        "maybe-uninit-alloc-read-written",
        MAYBE_UNINIT_ALLOC_READ_WRITTEN_SRC,
        "val=41\n",
    );
}

const PLAIN_ALLOC_READ_UNWRITTEN_SRC: &str = r#"
fn main() {
    unsafe {
        let layout = std::alloc::Layout::new::<u8>();
        let p = std::alloc::alloc(layout) as *mut u8;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let b: u8 = std::ptr::read(p);
        println!("val={}", b);
        std::alloc::dealloc(p, layout);
    }
}
"#;

#[test]
fn run_mode_ptr_read_unwritten_alloc_plain_payload_errors() {
    // The unwritten-allocation allowance is the `MaybeUninit`
    // type's, not the read's: a plain payload type read from
    // never-written storage stays an uninitialized read.
    let dir = temp_dir("plain-alloc-read-unwritten");
    let src = fixture(&dir, "case.rs", PLAIN_ALLOC_READ_UNWRITTEN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("uninitialized"),
        "plain payload read must be an uninitialised read; stderr={stderr}"
    );
}

const MAYBE_UNINIT_ALLOC_READ_ASSUME_SRC: &str = r#"
fn main() {
    unsafe {
        let layout = std::alloc::Layout::new::<std::mem::MaybeUninit<u8>>();
        let p = std::alloc::alloc(layout) as *mut std::mem::MaybeUninit<u8>;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let b: std::mem::MaybeUninit<u8> = std::ptr::read(p);
        println!("val={}", b.assume_init());
        std::alloc::dealloc(p as *mut u8, layout);
    }
}
"#;

#[test]
fn run_mode_maybe_uninit_ptr_read_then_assume_init_errors() {
    // Reading the union out of unwritten storage succeeds, but
    // the value is still uninitialized: using its payload through
    // `assume_init` fails instead of fabricating a byte.
    let dir = temp_dir("maybe-uninit-alloc-read-assume");
    let src = fixture(&dir, "case.rs", MAYBE_UNINIT_ALLOC_READ_ASSUME_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
}
const ASSUME_INIT_SLICE_SRC: &str = r#"
use std::mem::MaybeUninit;
fn main() {
    let a: [MaybeUninit<i32>; 3] =
        [MaybeUninit::new(10), MaybeUninit::new(20), MaybeUninit::new(30)];
    let s = unsafe { &*(&a as *const [MaybeUninit<i32>] as *const [i32]) };
    // Every element of a slice whose element type changed across
    // a pointer cast is readable: by index, by iteration, and by
    // raw pointer offset.
    println!("len={} sum={}", s.len(), s.iter().sum::<i32>());
    println!("idx={} {} {}", s[0], s[1], s[2]);
    println!("last={:?} count={}", s.last(), s.iter().count());
    let q = s.as_ptr();
    unsafe {
        println!("raw={} {} {}", *q, *q.add(1), *q.add(2));
    }
    let v = s.to_vec();
    println!("vec={:?}", v);
}
"#;

#[test]
fn run_mode_assume_init_slice_reads_all_elements() {
    run_ok(
        "assume-init-slice",
        ASSUME_INIT_SLICE_SRC,
        "len=3 sum=60\nidx=10 20 30\nlast=Some(30) count=3\nraw=10 20 30\nvec=[10, 20, 30]\n",
    );
}

const ASSUME_INIT_SLICE_MUT_SRC: &str = r#"
use std::mem::MaybeUninit;
fn main() {
    let mut a: [MaybeUninit<(i32, i32)>; 4] = [
        MaybeUninit::new((1, 2)),
        MaybeUninit::new((3, 4)),
        MaybeUninit::new((5, 6)),
        MaybeUninit::new((7, 8)),
    ];
    let s = unsafe { &mut *(&mut a as *mut [MaybeUninit<(i32, i32)>] as *mut [(i32, i32)]) };
    // A write through the cast slice lands in the addressed
    // element, not in element zero's payload.
    s[1].0 = 99;
    let sub = &mut s[1..4];
    sub[2].1 = 66;
    let t = unsafe { &*(&a as *const [MaybeUninit<(i32, i32)>] as *const [(i32, i32)]) };
    println!("e0={:?} e1={:?} e2={:?} e3={:?}", t[0], t[1], t[2], t[3]);
}
"#;

#[test]
fn run_mode_assume_init_slice_mut_writes_addressed_elements() {
    run_ok(
        "assume-init-slice-mut",
        ASSUME_INIT_SLICE_MUT_SRC,
        "e0=(1, 2) e1=(99, 4) e2=(5, 6) e3=(7, 66)\n",
    );
}

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

const PATHBUF_DEDUP_SRC: &str = r#"
use std::path::PathBuf;

fn main() {
    let mut v: Vec<PathBuf> = vec![
        PathBuf::from("a/b"),
        PathBuf::from("a/b"),
        PathBuf::from("c"),
        PathBuf::from("a-b"),
        PathBuf::from("a-b"),
        PathBuf::from("a-b"),
        PathBuf::from("d/e"),
        PathBuf::from("c"),
    ];
    v.dedup();
    println!("after len: {}", v.len());
    for p in &v {
        println!("item: {}", p.to_str().unwrap());
    }
    // Path equality is component-wise: "a//b", "a/./b" and
    // "a/b/" are the same path as "a/b".
    let mut w: Vec<PathBuf> = vec![
        PathBuf::from("a/b"),
        PathBuf::from("a//b"),
        PathBuf::from("a/./b"),
        PathBuf::from("a/b/"),
    ];
    w.dedup();
    println!("normalised dedup len: {}", w.len());
    println!("pathbuf-dedup-ok");
}
"#;

#[test]
fn run_mode_pathbuf_dedup() {
    run_ok(
        "pathbuf-dedup",
        PATHBUF_DEDUP_SRC,
        "after len: 5\nitem: a/b\nitem: c\nitem: a-b\nitem: d/e\nitem: c\nnormalised dedup len: 1\npathbuf-dedup-ok\n",
    );
}

const PATH_MIXED_CMP_SRC: &str = r#"
use std::path::{Path, PathBuf};

fn generic_lt<T: PartialOrd>(a: T, b: T) -> bool { a < b }

fn main() {
    let buf_ab = PathBuf::from("a/b");
    let buf_a_b = PathBuf::from("a-b");
    let p_ab: &Path = Path::new("a/b");
    assert!(buf_ab.as_path() < buf_a_b.as_path());
    assert!(p_ab < &buf_a_b);
    assert!(&buf_ab.as_path() < &buf_a_b);
    assert!(buf_ab.as_path() == p_ab);
    assert!(generic_lt(buf_ab.as_path(), buf_a_b.as_path()));
    assert!(generic_lt(&buf_ab, &buf_a_b));
    assert_eq!(std::cmp::min(buf_ab.as_path(), buf_a_b.as_path()), p_ab);
    // A sort through the generic Ord adapter keeps component
    // order ("a/b" < "a-b" component-wise, not byte-wise).
    let mut v: Vec<&Path> = vec![buf_a_b.as_path(), buf_ab.as_path()];
    v.sort();
    assert!(v[0] == p_ab);
    let mut w: Vec<PathBuf> = vec![
        PathBuf::from("a/b"),
        PathBuf::from("c"),
        PathBuf::from("a/b"),
        PathBuf::from("a-b"),
        PathBuf::from("c"),
    ];
    w.sort();
    w.dedup();
    assert_eq!(w.len(), 3);
    assert_eq!(w[0].to_str().unwrap(), "a/b");
    assert_eq!(w[1].to_str().unwrap(), "a-b");
    assert_eq!(w[2].to_str().unwrap(), "c");
    println!("path-mixed-cmp-ok");
}
"#;

#[test]
fn run_mode_path_mixed_cmp() {
    run_ok("path-mixed-cmp", PATH_MIXED_CMP_SRC, "path-mixed-cmp-ok\n");
}

const PATHBUF_PTR_SWAP_SRC: &str = r#"
use std::path::PathBuf;

fn main() {
    // A raw-pointer element copy (the shift a dedup performs)
    // and a raw-pointer swap must move the whole element
    // value, wrapper chain included.
    let mut v: Vec<PathBuf> = vec![PathBuf::from("a/b"), PathBuf::from("c"), PathBuf::from("d/e")];
    unsafe {
        let base = v.as_mut_ptr();
        std::ptr::copy_nonoverlapping(base.add(2), base.add(1), 1);
    }
    println!("copied: {}", v[1].to_str().unwrap());
    let mut u: Vec<PathBuf> = vec![PathBuf::from("m/n"), PathBuf::from("o")];
    unsafe {
        std::ptr::swap(u.as_mut_ptr(), u.as_mut_ptr().add(1));
    }
    println!("swapped 0: {}", u[0].to_str().unwrap());
    println!("swapped 1: {}", u[1].to_str().unwrap());
    let mut s: Vec<String> = vec!["hello".to_string(), "world".to_string()];
    unsafe {
        std::ptr::swap(s.as_mut_ptr(), s.as_mut_ptr().add(1));
    }
    println!("strings: {} {}", s[0], s[1]);
    println!("pathbuf-ptr-swap-ok");
}
"#;

#[test]
fn run_mode_pathbuf_ptr_swap() {
    run_ok(
        "pathbuf-ptr-swap",
        PATHBUF_PTR_SWAP_SRC,
        "copied: d/e\nswapped 0: o\nswapped 1: m/n\nstrings: world hello\npathbuf-ptr-swap-ok\n",
    );
}

const VEC_POP_AFTER_SHRINK_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::with_capacity(16);
    v.push(1);
    v.push(2);
    v.shrink_to_fit();
    println!("shrunk len={}", v.len());
    match v.pop() {
        Some(x) => println!("popped {}", x),
        None => println!("none"),
    }
    println!("after len={}", v.len());
    match v.pop() {
        Some(x) => println!("popped {}", x),
        None => println!("none"),
    }
    println!("final len={} empty={}", v.len(), v.is_empty());
}
"#;

#[test]
fn run_mode_vec_pop_after_shrink() {
    // `pop`'s collected body reads the length slot, writes the
    // decremented length back, reads the capacity slot for its
    // `assert_unchecked(len < capacity)`, and reads the element
    // at the new length through the buffer pointer: the length
    // write must not destroy the element it is about to read,
    // and the capacity answer must stay the storage extent
    // across the write (see the vec length/storage rule in
    // `vec_with_len`).
    run_ok(
        "vec-pop-after-shrink",
        VEC_POP_AFTER_SHRINK_SRC,
        "shrunk len=2\npopped 2\nafter len=1\npopped 1\nfinal len=0 empty=true\n",
    );
}

const VEC_APPEND_TAIL_SRC: &str = r#"
fn main() {
    let mut a = vec![1, 2];
    let mut b = vec![3, 4, 5];
    a.append(&mut b);
    println!("a={:?} len={} cap={}", a, a.len(), a.capacity());
    println!("b={:?} len={}", b, b.len());
    println!("a[0]={} a[3]={} a[4]={}", a[0], a[3], a[4]);
}
"#;

#[test]
fn run_mode_vec_append_tail() {
    // `append`'s collected body copies the source elements
    // through the destination's buffer pointer at the old
    // length and only then writes the length slot: the raw
    // writes extend the storage without moving the declared
    // length, and the final length write lands on the extended
    // storage.
    run_ok(
        "vec-append-tail",
        VEC_APPEND_TAIL_SRC,
        "a=[1, 2, 3, 4, 5] len=5 cap=5\nb=[] len=0\na[0]=1 a[3]=4 a[4]=5\n",
    );
}

const VEC_SPLIT_OFF_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3, 4, 5, 6];
    v.truncate(3);
    println!("trunc v={:?} len={}", v, v.len());
    let mut w = vec![1, 2, 3, 4, 5];
    let tail = w.split_off(2);
    println!("w={:?} tail={:?}", w, tail);
}
"#;

#[test]
fn run_mode_vec_split_off() {
    // `split_off` builds its result in the real aggregate form
    // inside its body (an inlined `with_capacity_in`, `set_len`,
    // and a raw copy into the buffer allocation): the value
    // translates to the element form as it returns to the
    // caller (see `vec_from_real_form`).
    run_ok(
        "vec-split-off",
        VEC_SPLIT_OFF_SRC,
        "trunc v=[1, 2, 3] len=3\nw=[1, 2] tail=[3, 4, 5]\n",
    );
}

const VEC_DRAIN_COLLECT_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3, 4, 5];
    let taken: Vec<i32> = v.drain(1..3).collect();
    println!("taken={:?}", taken);
    let mut u = vec![10, 20, 30];
    let all: Vec<i32> = u.drain(..).collect();
    println!("all={:?}", all);
}
"#;

#[test]
fn run_mode_vec_drain_collect() {
    // A drain's iteration steps its slice iterator through the
    // source vector's storage (see `drain_next_state`); the
    // collected elements are the drained range.
    run_ok(
        "vec-drain-collect",
        VEC_DRAIN_COLLECT_SRC,
        "taken=[2, 3]\nall=[10, 20, 30]\n",
    );
}

const VEC_INSERT_GROW_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3];
    v.insert(3, 4);
    println!("{:?} len={} cap={}", v, v.len(), v.capacity());
    v.insert(1, 99);
    println!("{:?}", v);
    v.insert(0, 0);
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_vec_insert_positions_with_growth() {
    // `Vec::insert` runs its collected body: the capacity
    // check and the regrowth walk the buffer wrapper chain
    // (`RawVec` / `RawVecInner`) through references handed
    // across calls, and the chain's bookkeeping slots are the
    // vector's own storage extent and declared length in this
    // model. End, middle and start inserts over a growing
    // vector land exactly as natively, capacity included.
    let dir = temp_dir("vecinsertgrow");
    let src = fixture(&dir, "vecinsertgrow.rs", VEC_INSERT_GROW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::insert with growth must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 2, 3, 4] len=4 cap=6\n[1, 99, 2, 3, 4]\n[0, 1, 99, 2, 3, 4]\n"
    );
}

const VEC_INSERT_RESERVED_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::with_capacity(16);
    v.push(10);
    v.push(20);
    v.push(30);
    v.insert(1, 15);
    println!("{:?} cap={}", v, v.capacity());
    assert_eq!(v, vec![10, 15, 20, 30]);
    let mut s = vec![String::from("a"), String::from("c")];
    s.insert(1, String::from("b"));
    assert_eq!(s, vec![String::from("a"), String::from("b"), String::from("c")]);
    println!("ok");
}
"#;

#[test]
fn run_mode_vec_insert_within_capacity_and_equality() {
    // An insert inside the reserved capacity does not grow:
    // `with_capacity` materialises the requested storage
    // extent, so the capacity stays 16 and the elements shift
    // in place. Whole-vector equality after a mutation
    // compares the logical element sequence: the sparse form
    // a growth or a shrink leaves behind (declared length
    // below the stored elements) equals the plain aggregate
    // form of the same elements.
    let dir = temp_dir("vecinsertreserved");
    let src = fixture(&dir, "vecinsertreserved.rs", VEC_INSERT_RESERVED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::insert within capacity must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[10, 15, 20, 30] cap=16\nok\n"
    );
}

const VEC_INSERT_NESTED_SRC: &str = r#"
fn main() {
    let mut v = vec![vec![1, 2], vec![4]];
    v.insert(1, vec![3]);
    println!("{:?}", v);
    let mut t = (vec![1, 3], 7);
    t.0.insert(1, 2);
    println!("{:?}", t);
}
"#;

#[test]
fn run_mode_vec_insert_nested_and_tuple_field() {
    // The inserted element may itself be a vector, and the
    // mutated vector may live inside another aggregate: the
    // chain re-entry judges the place the buffer pointer
    // denotes by the value it currently holds, not by the
    // declared type of the local that happens to contain it.
    let dir = temp_dir("vecinsertnested");
    let src = fixture(&dir, "vecinsertnested.rs", VEC_INSERT_NESTED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::insert on nested/tuple-field vectors must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[[1, 2], [3], [4]]\n([1, 2, 3], 7)\n"
    );
}

const VEC_INSERT_SHRINK_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::with_capacity(16);
    v.push(1);
    v.push(2);
    v.insert(1, 9);
    println!("{:?} cap={}", v, v.capacity());
    v.shrink_to_fit();
    println!("{:?} cap={}", v, v.capacity());
    v.insert(3, 5);
    println!("{:?} cap={}", v, v.capacity());
}
"#;

#[test]
fn run_mode_vec_insert_after_shrink_to_fit() {
    // After a shrink the storage extent is the length itself,
    // so the next insert grows again from there: the capacity
    // slot writes of both paths (the shrink's truncation and
    // the regrowth's amortised capacity) rebuild the vector's
    // storage extent exactly.
    let dir = temp_dir("vecinsertshrink");
    let src = fixture(&dir, "vecinsertshrink.rs", VEC_INSERT_SHRINK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::insert after shrink_to_fit must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 9, 2] cap=16\n[1, 9, 2] cap=3\n[1, 9, 2, 5] cap=6\n"
    );
}

const BOX_VEC_GROW_SRC: &str = r#"
struct D(&'static str);
impl Drop for D {
    fn drop(&mut self) {
        println!("drop {}", self.0);
    }
}
fn main() {
    let mut b: Box<Vec<D>> = Box::new(Vec::with_capacity(2));
    b.push(D("a"));
    b.push(D("b"));
    b.insert(1, D("x"));
    b.reserve(16);
    for d in b.iter() {
        println!("elem {}", d.0);
    }
    drop(b);
    println!("done");
}
"#;

#[test]
fn run_mode_box_vec_insert_and_reserve_with_growth() {
    // Growing a `Vec` that is the whole content of a `Box`
    // allocation through the collected `insert` / `reserve`
    // bodies: the vector's buffer pointer is derived from the
    // vector place inside the box's allocation, and the
    // reallocation must not free or move the box's storage --
    // the elements and their drop order land exactly as
    // natively and the box stays live throughout.
    let dir = temp_dir("boxvecgrow");
    let src = fixture(&dir, "boxvecgrow.rs", BOX_VEC_GROW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<Vec> growth must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "elem a\nelem x\nelem b\ndrop a\ndrop x\ndrop b\ndone\n"
    );
}

const BOX_BOX_VEC_GROW_SRC: &str = r#"
fn main() {
    let mut b: Box<Box<Vec<i32>>> = Box::new(Box::new(Vec::with_capacity(2)));
    b.push(1);
    b.push(2);
    b.push(3);
    b.insert(1, 9);
    println!("{:?}", b);
}
"#;

#[test]
fn run_mode_box_box_vec_insert_with_growth() {
    // The same growth through two owning wrappers: the inner
    // vector's buffer pointer denotes a position inside the
    // inner box's stored value, never a wrapper allocation's
    // base.
    let dir = temp_dir("boxboxvecgrow");
    let src = fixture(&dir, "boxboxvecgrow.rs", BOX_BOX_VEC_GROW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<Box<Vec>> growth must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "[1, 9, 2, 3]\n");
}

const NESTED_VEC_POP_SRC: &str = r#"
fn main() {
    let mut v = vec![vec![1, 2], vec![4]];
    let x = v[0].pop();
    println!("{:?} {:?}", v, x);
}
"#;

#[test]
fn run_mode_nested_vec_pop_after_shrink() {
    // A collected method (`pop`) on a vector element reached
    // through an index projection: `pop` shrinks the element
    // in place, so the element takes the sparse form, and the
    // body's own reads of the element's elements (through the
    // buffer pointer the method forms) must range inside the
    // element, not compose into positions of the outer vector.
    run_ok(
        "nested-vec-pop",
        NESTED_VEC_POP_SRC,
        "[[1], [4]] Some(2)\n",
    );
}

const NESTED_VEC_INSERT_SRC: &str = r#"
fn main() {
    let mut v = vec![vec![1, 2], vec![4]];
    v.insert(1, vec![3]);
    println!("{:?}", v);
    v[0].insert(1, 9);
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_nested_vec_insert_after_outer_insert() {
    // After the outer vector's own collected `insert` leaves
    // it in the sparse form, a collected `insert` on one of its
    // elements forms its mutable receiver through `IndexMut`
    // over the outer place: the sparse form is the same element
    // aggregate as the plain form and is addressable the same
    // way.
    run_ok(
        "nested-vec-insert",
        NESTED_VEC_INSERT_SRC,
        "[[1, 2], [3], [4]]\n[[1, 9, 2], [3], [4]]\n",
    );
}

const NESTED_VEC_SET_LEN_READS_SRC: &str = r#"
fn main() {
    let mut v = vec![vec![1, 2], vec![4]];
    unsafe { v[0].set_len(1); }
    println!("len={}", v[0].len());
    println!("e={}", v[0][0]);
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_nested_vec_reads_after_set_len() {
    // `set_len` rewrites the element's declared length in
    // place, leaving the sparse form behind; element reads
    // through the index projection then denote the element's
    // own elements.
    run_ok(
        "nested-vec-set-len",
        NESTED_VEC_SET_LEN_READS_SRC,
        "len=1\ne=1\n[[1], [4]]\n",
    );
}

const NESTED_VEC_PUSH_TRUNC_SRC: &str = r#"
fn main() {
    let mut w = vec![vec![vec![7, 8]], vec![vec![3]]];
    w.truncate(1);
    w[0][0].push(10);
    println!("{:?}", w);
}
"#;

#[test]
fn run_mode_nested_vec_push_after_outer_truncate() {
    // With the outer vector in the sparse form (after
    // `truncate`), a method on a doubly-indexed element forms
    // its receiver through the outer element position: the
    // steps nest into the element rather than composing into
    // the outer vector's positions.
    run_ok(
        "nested-vec-push-trunc",
        NESTED_VEC_PUSH_TRUNC_SRC,
        "[[[7, 8, 10]]]\n",
    );
}

// ----- zero-sized pointer addresses (Lane NA) -------------------------------

const ZST_VEC_PTR_EQ_SRC: &str = r#"
fn main() {
    let v: Vec<()> = vec![(), (), ()];
    let p = v.as_ptr();
    unsafe {
        println!("eq {}", p.add(1) == p);
        println!("eq2 {}", p.add(2) == p.add(0));
        println!("ne {}", p.add(1) != p);
        println!("lt {}", p.add(1) < p);
        println!("le {}", p.add(1) <= p);
        println!("gt {}", p.add(1) > p);
        println!("int {}", p.add(1) as usize == p as usize);
        println!("ptreq {}", std::ptr::eq(p.add(1), p));
    }
    println!("done");
}
"#;

const ZST_ARRAY_PTR_EQ_SRC: &str = r#"
struct Z;
fn main() {
    let a = [Z, Z, Z, Z];
    let p = a.as_ptr();
    unsafe {
        println!("arr {}", p.add(3) == p);
    }
    let s: &[Z] = &a;
    println!("slice {}", s.as_ptr() == p);
    let mut m: Vec<std::mem::MaybeUninit<()>> = Vec::new();
    m.push(std::mem::MaybeUninit::new(()));
    m.push(std::mem::MaybeUninit::uninit());
    let q = m.as_ptr();
    unsafe {
        println!("mu {}", q.add(1) == q);
    }
    println!("done");
}
"#;

const SIZED_PTR_POSITIONS_SRC: &str = r#"
fn main() {
    let v: Vec<i32> = vec![10, 20, 30];
    let p = v.as_ptr();
    unsafe {
        println!("eq0 {}", p.add(0) == p);
        println!("eq1 {}", p.add(1) == p);
        println!("ne12 {}", p.add(1) != p.add(2));
        println!("ordered {}", (p < p.add(1)) ^ (p.add(1) < p));
        println!("refl {}", p.add(1) <= p.add(1) && p.add(1) >= p.add(1));
    }
    println!("done");
}
"#;

#[test]
fn run_mode_zst_vec_pointer_positions_share_the_base_address() {
    // A zero-sized element occupies no bytes, so every element
    // position of a `Vec<()>` is the same address natively:
    // `p.add(1) == p`, no strict ordering between positions,
    // and the integer casts agree (Lane MV denominated the
    // offsets in elements, which had made the positions
    // distinct synthetic addresses).
    let dir = temp_dir("zst-vec-ptr-eq");
    let src = fixture(&dir, "zst_vec_ptr_eq.rs", ZST_VEC_PTR_EQ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "ZST pointer equality program must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "eq true\neq2 true\nne false\nlt false\nle true\ngt false\nint true\nptreq true\ndone\n"
    );
}

#[test]
fn run_mode_zst_array_slice_and_maybeuninit_pointer_equality() {
    // The same address rule through an array root, a slice
    // fat pointer and a `Vec<MaybeUninit<()>>` buffer.
    let dir = temp_dir("zst-array-ptr-eq");
    let src = fixture(&dir, "zst_array_ptr_eq.rs", ZST_ARRAY_PTR_EQ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "ZST array pointer equality program must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "arr true\nslice true\nmu true\ndone\n"
    );
}

#[test]
fn run_mode_sized_pointer_positions_stay_distinct() {
    // Sized elements keep their position distinction: only
    // zero-sized element positions collapse onto the base.
    // Distinct positions stay distinct and strictly ordered
    // one way or the other (the synthetic address order itself
    // is the hash order, as before this lane).
    let dir = temp_dir("sized-ptr-positions");
    let src = fixture(&dir, "sized_ptr_positions.rs", SIZED_PTR_POSITIONS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "sized pointer position program must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "eq0 true\neq1 false\nne12 true\nordered true\nrefl true\ndone\n"
    );
}

const VEC_ZST_SHIFT_SRC: &str = r#"
struct D;
impl Drop for D {
    fn drop(&mut self) {
        println!("drop D");
    }
}
fn main() {
    let mut u: Vec<()> = vec![(), ()];
    u.insert(0, ());
    println!("ulen {}", u.len());
    u.remove(0);
    println!("ulen {}", u.len());
    let mut v = vec![D, D];
    v.insert(1, D);
    println!("len {}", v.len());
    v.remove(0);
    println!("len {}", v.len());
    drop(v);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_macro_zst_elements_shift() {
    // The `vec![...]` list form over zero-sized elements: rustc
    // elides the element writes into the `Box::new_uninit`
    // buffer (a zero-sized write is a no-op), so the slots the
    // interpreter's `box_assume_init_into_vec_unsafe` model
    // reads still hold the uninitialized marker. Conversion
    // bakes the element type's one value into the call and the
    // model materializes the unwritten slots from it, so the
    // shift operations (`insert` / `remove`, which copy the
    // buffer's elements) read real elements and the destructors
    // run in order, byte for byte as natively.
    let dir = temp_dir("veczstshift");
    let src = fixture(&dir, "veczstshift.rs", VEC_ZST_SHIFT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec! of ZST elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "ulen 3\nulen 2\nlen 3\ndrop D\nlen 2\ndrop D\ndrop D\ndone\n",
        "vec! ZST elements shift and drop exactly as natively"
    );
}

const VEC_SET_LEN_MAYBE_UNINIT_SRC: &str = r#"
fn main() {
    // A slot the length declares but no write reached holds the
    // element type's uninitialized form: for a `MaybeUninit`
    // element that is the uninitialized `MaybeUninit` value
    // itself, which reads back and can be carried (here,
    // forgotten) without touching its payload.
    let mut v: Vec<std::mem::MaybeUninit<u8>> = Vec::with_capacity(2);
    unsafe {
        v.set_len(2);
        let b = std::ptr::read(v.as_ptr());
        std::mem::forget(b);
        let c = v[1];
        std::mem::forget(c);
    }
    println!("carried");

    // The slots are real storage: writing through the buffer
    // pointer and reading back observes the writes.
    let mut w: Vec<std::mem::MaybeUninit<u8>> = Vec::with_capacity(2);
    unsafe {
        w.set_len(2);
        w.as_mut_ptr().write(std::mem::MaybeUninit::new(41));
        w.as_mut_ptr().add(1).write(std::mem::MaybeUninit::new(42));
        println!("{} {}", w[0].assume_init(), w[1].assume_init());
    }

    // A written prefix survives a growth: the first element
    // keeps its value, the grown slot stays uninitialized.
    let mut p: Vec<std::mem::MaybeUninit<u8>> = Vec::with_capacity(2);
    p.push(std::mem::MaybeUninit::new(7));
    unsafe {
        p.set_len(2);
        println!("first {}", std::ptr::read(p.as_ptr()).assume_init());
        std::mem::forget(std::ptr::read(p.as_ptr().add(1)));
    }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_set_len_maybe_uninit_slots() {
    run_ok(
        "vec-set-len-maybe-uninit",
        VEC_SET_LEN_MAYBE_UNINIT_SRC,
        "carried\n41 42\nfirst 7\ndone\n",
    );
}

const VEC_SET_LEN_PLAIN_UNINIT_READ_SRC: &str = r#"
fn main() {
    let mut vec: Vec<u8> = Vec::with_capacity(2);
    unsafe { vec.set_len(2) }
    // A plain element type's grown slot has no value: reading
    // it is an uninitialised read, exactly as a sparse tail
    // slot's read is.
    println!("{}", vec[0]);
}
"#;

#[test]
fn run_mode_vec_set_len_plain_uninit_read() {
    let dir = temp_dir("vec-set-len-uninit");
    let src = fixture(&dir, "case.rs", VEC_SET_LEN_PLAIN_UNINIT_READ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("uninitialized"),
        "grown slot read must be an uninitialised read; stderr={stderr}"
    );
}
const VEC_INSERT_WRAPPER_ELEM_SRC: &str = r#"
struct D(&'static str);
impl Drop for D {
    fn drop(&mut self) { println!("drop {}", self.0); }
}
struct N(i32);
fn main() {
    let mut v: Vec<D> = Vec::with_capacity(1);
    v.push(D("a"));
    v.push(D("b"));
    v.push(D("c"));
    v.insert(1, D("x"));
    for d in v.iter() { println!("elem {}", d.0); }
    drop(v);
    let mut n: Vec<N> = Vec::with_capacity(8);
    n.push(N(1)); n.push(N(2)); n.push(N(3));
    n.insert(1, N(9));
    for e in n.iter() { println!("num {}", e.0); }
    let mut a = [D("a"), D("b"), D("c"), D("?")];
    unsafe {
        let p = a.as_mut_ptr();
        std::ptr::copy(p.add(1), p.add(2), 2);
    }
    for d in a.iter() { println!("arr {}", d.0); }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_insert_one_field_wrapper_elements() {
    // The shift inside `Vec::insert` is an element copy
    // (`ptr::copy` over the buffer), and its elements here are
    // one-field wrapper structs. The copy must move the
    // elements whole: a wrapper over a scalar is not a byte
    // tree to patch with the following elements' bytes, and a
    // wrapper over a fat pointer is not a wrapper place to
    // rewrap the incoming element inside. Shifted elements
    // read back exactly, fields and drops included, as does a
    // direct `ptr::copy` over an array of the same elements.
    let dir = temp_dir("vecinsertwrapper");
    let src = fixture(&dir, "vecinsertwrapper.rs", VEC_INSERT_WRAPPER_ELEM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::insert of one-field wrapper elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "elem a\nelem x\nelem b\nelem c\ndrop a\ndrop x\ndrop b\ndrop c\nnum 1\nnum 9\nnum 2\nnum 3\narr a\narr b\narr b\narr c\ndone\ndrop a\ndrop b\ndrop b\ndrop c\n"
    );
}


const NOUNWIND_FROM_STR_SRC: &str = r#"
#![allow(invalid_null_arguments)]
fn main() {
    let p: *const u8 = std::ptr::null();
    let s = unsafe { std::slice::from_raw_parts(p, 0) };
    println!("{}", s.len());
}
"#;

#[test]
fn run_mode_nounwind_from_str_panic_message() {
    // A UB-check failure inside a library body reports through
    // `panic_nounwind_fmt` with an `Arguments::from_str` value
    // the body built itself: the run must surface the check's
    // own message as the panic payload, not die in the fmt
    // coercion with a type mismatch.
    let dir = temp_dir("nounwind-from-str");
    let src = fixture(&dir, "case.rs", NOUNWIND_FROM_STR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "nounwind panic must exit 101; stderr={stderr}"
    );
    assert!(
        stderr.contains(
            "unsafe precondition(s) violated: slice::from_raw_parts requires the pointer to be aligned and non-null"
        ),
        "the check's message is the payload: {stderr}"
    );
    assert!(
        stderr.contains("This indicates a bug in the program. This Undefined Behavior check is optional"),
        "the full message renders: {stderr}"
    );
    assert!(
        !stderr.contains("expected byte data"),
        "no fmt coercion failure masks the message: {stderr}"
    );
}

const SLICE_JOIN_SRC: &str = r#"
fn main() {
    // `slice::join` builds its `String` through
    // `join_generic_copy`: a `Vec<u8>` in its real aggregate
    // form, filled through raw-pointer copies, returned as a
    // `String`. Its core UB checks (`ptr::offset_from_unsigned`
    // preconditions) compare positions of one allocation, and
    // the result denotes its string through the buffer.
    let parts: &[&str] = &["a", "bb", "ccc"];
    let s = parts.join("-");
    println!("{}", s);
    println!("{:?}", s);
    println!("len={} lt={}", s.len(), s < "b".to_string());
    assert_eq!(s, "a-bb-ccc");
    let e: &[&str] = &[];
    println!("[{}]", e.join("-"));
    let one: &[&str] = &["x"];
    println!("{}", one.join("---"));
    let v = vec!["p".to_string(), "q".to_string()];
    println!("{}", v.join("::"));
    println!("{}", parts.concat());
    println!("{}", s.to_uppercase());
}
"#;

#[test]
fn run_mode_slice_join() {
    let dir = temp_dir("slice-join");
    let src = fixture(&dir, "slice_join.rs", SLICE_JOIN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "slice::join must run without a UB-check failure; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "a-bb-ccc\n\"a-bb-ccc\"\nlen=8 lt=true\n[]\nx\np::q\nabbccc\nA-BB-CCC\n",
        "the joined string reads back through its real aggregate form"
    );
    assert!(
        !stderr.contains("offset_from_unsigned"),
        "no UB-check message fires: {stderr}"
    );
}

const WAKER_FROM_ARC_SRC: &str = r#"
use std::future::Future;
use std::pin::pin;
use std::sync::{mpsc, Arc};
use std::task::{Context, Poll, Wake, Waker};

fn simple_waker() -> (Waker, mpsc::Receiver<()>) {
    struct SimpleWaker {
        tx: mpsc::Sender<()>,
    }
    impl Wake for SimpleWaker {
        fn wake(self: Arc<Self>) {
            self.tx.send(()).unwrap();
        }
    }
    let (tx, rx) = mpsc::channel();
    (Waker::from(Arc::new(SimpleWaker { tx })), rx)
}

fn main() {
    let (waker, _rx) = simple_waker();
    let mut cx = Context::from_waker(&waker);
    let fut = pin!(async { 42 });
    let mut fut = fut;
    match fut.as_mut().poll(&mut cx) {
        Poll::Ready(v) => println!("ready {v}"),
        Poll::Pending => println!("pending"),
    }
    waker.wake_by_ref();
    println!("woken");
}
"#;

#[test]
fn run_mode_waker_from_arc_wake_by_ref_default() {
    // `Waker::from(Arc<T: Wake>)` with only `wake` defined:
    // `wake_by_ref` runs the trait's default body (clone the
    // handle, wake it), which the wake demand converts.
    run_ok_edition("waker-from-arc", WAKER_FROM_ARC_SRC, "ready 42\nwoken\n");
}

const WAKER_OVERRIDE_SRC: &str = r#"
use std::sync::{Arc, Weak};
use std::task::{Wake, Waker};

struct W2 {
    tag: u32,
}
impl Wake for W2 {
    fn wake(self: Arc<Self>) {
        println!("wake {}", self.tag);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        println!("wake_by_ref override {}", self.tag);
    }
}

fn main() {
    let arc = Arc::new(W2 { tag: 9 });
    let weak: Weak<W2> = Arc::downgrade(&arc);
    let waker = Waker::from(arc);
    println!("count {}", weak.strong_count());
    waker.wake_by_ref();
    println!("count {}", weak.strong_count());
    let waker2 = waker.clone();
    waker2.wake_by_ref();
    waker.wake();
    println!("count {}", weak.strong_count());
    waker2.wake_by_ref();
    drop(waker2);
    println!("count {}", weak.strong_count());
}
"#;

#[test]
fn run_mode_waker_wake_override_and_counts() {
    // An overridden `wake_by_ref` runs the override, and the
    // strong count the wake consumes / preserves is the native
    // one at every step.
    run_ok_edition(
        "waker-override",
        WAKER_OVERRIDE_SRC,
        "count 1\nwake_by_ref override 9\ncount 1\nwake_by_ref override 9\nwake 9\ncount 1\nwake_by_ref override 9\ncount 0\n",
    );
}

const WAKER_RAW_VTABLE_SRC: &str = r#"
use std::sync::Arc;
use std::task::{RawWaker, RawWakerVTable, Wake, Waker};

struct W3 {
    tag: u32,
}
impl Wake for W3 {
    fn wake(self: Arc<Self>) {
        println!("w3 wake {}", self.tag);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        println!("w3 wake_by_ref {}", self.tag);
    }
}

unsafe fn clone_fn(data: *const ()) -> RawWaker {
    let arc = unsafe { Arc::<W3>::from_raw(data as *const W3) };
    let cloned = arc.clone();
    std::mem::forget(arc);
    RawWaker::new(Arc::into_raw(cloned) as *const (), &VTABLE)
}
unsafe fn wake_fn(data: *const ()) {
    let arc = unsafe { Arc::<W3>::from_raw(data as *const W3) };
    arc.wake();
}
unsafe fn wake_by_ref_fn(data: *const ()) {
    let arc = unsafe { Arc::<W3>::from_raw(data as *const W3) };
    arc.wake_by_ref();
    std::mem::forget(arc);
}
unsafe fn drop_fn(data: *const ()) {
    let arc = unsafe { Arc::<W3>::from_raw(data as *const W3) };
    drop(arc);
}
static VTABLE: RawWakerVTable = RawWakerVTable::new(clone_fn, wake_fn, wake_by_ref_fn, drop_fn);

fn main() {
    let arc = Arc::new(W3 { tag: 4 });
    let raw = RawWaker::new(Arc::into_raw(arc) as *const (), &VTABLE);
    let waker = unsafe { Waker::from_raw(raw) };
    waker.wake_by_ref();
    let waker2 = waker.clone();
    waker2.wake_by_ref();
    waker.wake();
    println!("mid");
    waker2.wake();
    println!("done");
}
"#;

#[test]
fn run_mode_waker_from_raw_vtable_wake() {
    // A hand-built `RawWaker`: wake and wake_by_ref run the
    // vtable's entries with the data pointer.
    run_ok_edition(
        "waker-raw-vtable",
        WAKER_RAW_VTABLE_SRC,
        "w3 wake_by_ref 4\nw3 wake_by_ref 4\nw3 wake 4\nmid\nw3 wake 4\ndone\n",
    );
}

const SLICE_JOIN_CLOSURE_CONST_SRC: &str = r#"
fn main() {
    let a: &[&[i32]] = &[&[1, 2], &[3]];
    let j = a.join(&0);
    println!("done {}", j.len());
}
"#;

#[test]
fn run_mode_slice_join_closure_constant_converts() {
    // The length fold inside slice Join::join reaches conversion as a
    // closure *type constant* whose environment captures the sum
    // closure (const-prop folded, all captures zero-sized).
    // Conversion must materialize that environment instead of
    // rejecting the constant; before the fix this program died in
    // conversion with "closure constant ... with a captured
    // environment". The joined length itself is Vec bookkeeping
    // territory and is not what this test pins.
    let dir = temp_dir("slice-join-closure-const");
    let src = fixture(&dir, "case.rs", SLICE_JOIN_CLOSURE_CONST_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "join runs; stderr={stderr}");
    assert!(
        String::from_utf8_lossy(&out.stdout).starts_with("done "),
        "join completed: {stderr}"
    );
}

const ZIP_SRC: &str = r#"
fn main() {
    let a = [1, 2, 3];
    let b = [10, 20, 30, 40];
    for (x, y) in a.iter().zip(b.iter()) {
        println!("{} {}", x, y);
    }
    let v: Vec<(&i32, &i32)> = a.iter().zip(b.iter()).collect();
    println!("collected {}", v.len());
    println!("count {}", a.iter().zip(b.iter()).count());
    for ((x, y), w) in a.iter().zip(b.iter()).zip([100, 200, 300].iter()) {
        println!("{} {} {}", x, y, w);
    }
    let s: i32 = a.iter().zip(b.iter()).map(|(x, y)| x + y).sum();
    println!("sum {}", s);
    for (x, y) in std::iter::zip([1, 2], [3, 4]) {
        println!("arr {} {}", x, y);
    }
}
"#;

#[test]
fn run_mode_zip_iteration() {
    let dir = temp_dir("zip-iteration");
    let src = fixture(&dir, "zip_iteration.rs", ZIP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Zip over slice iterators and by-value arrays must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "1 10\n2 20\n3 30\ncollected 3\ncount 3\n1 10 100\n2 20 200\n3 30 300\nsum 66\narr 1 3\narr 2 4\n",
        "pairs are yielded front-aligned up to the shorter side"
    );
}

const ZIP_DOUBLE_ENDED_SRC: &str = r#"
fn main() {
    let a = [1, 2, 3, 4];
    let b = [10, 20, 30];
    let mut z = a.iter().zip(b.iter());
    println!("hint {:?}", z.size_hint());
    println!("back {:?}", z.next_back());
    println!("next {:?}", z.next());
    println!("back {:?}", z.next_back());
    println!("back {:?}", z.next_back());
    println!("len {}", a.iter().zip(b.iter()).len());
    let mut z2 = std::iter::zip([1, 2, 3], [7, 8, 9, 10]);
    println!("mix {:?}", z2.next_back());
    println!("mix {:?}", z2.next());
    println!("mix {:?}", z2.next_back());
    println!("mix {:?}", z2.next());
}
"#;

#[test]
fn run_mode_zip_double_ended() {
    let dir = temp_dir("zip-double-ended");
    let src = fixture(&dir, "zip_double_ended.rs", ZIP_DOUBLE_ENDED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Zip next_back with unequal lengths must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hint (3, Some(3))\nback Some((3, 30))\nnext Some((1, 10))\nback Some((2, 20))\nback None\nlen 3\nmix Some((3, 9))\nmix Some((1, 7))\nmix Some((2, 8))\nmix None\n",
        "the back step drops the longer side's unpaired excess first"
    );
}


// ----- Zero-sized Vec elements' destructors (Lane MH) -------------------------

const ZST_VEC_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
fn main() {
    let v = vec![DropMe, DropMe];
    drop(v);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_of_zero_sized_elements_drops_them() {
    // A zero-sized element's store is elided in MIR, so the Vec's
    // value holds uninitialised slots for live elements; their
    // destructors still run, once each, as natively.
    let dir = temp_dir("zstvec");
    let src = fixture(&dir, "zstvec.rs", ZST_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec! of zero-sized elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "drop\ndrop\ndone\n",
        "both elided elements' destructors run at the Vec's drop"
    );
}

const ZST_ENUM_VEC_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
enum Never {}
enum CS { Yielded(Never), Complete(DropMe) }
fn main() {
    let v = vec![CS::Complete(DropMe)];
    drop(v);
    let mut v2: Vec<CS> = Vec::new();
    v2.push(CS::Complete(DropMe));
    drop(v2);
    let x = CS::Complete(DropMe);
    drop(x);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_of_zero_sized_enum_elements_drops_them() {
    // The element type here is zero-sized with a single
    // inhabitant, so its values are built as constants: no
    // aggregate statement records the variant's field, yet the
    // field's destructor runs — for vec! elements, for pushed
    // elements and for a plain local alike.
    let dir = temp_dir("zstenumvec");
    let src = fixture(&dir, "zstenumvec.rs", ZST_ENUM_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "zero-sized enum element program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "drop\ndrop\ndrop\ndone\n",
        "the constant-built elements' and local's destructors all run"
    );
}

const CLONED_SRC: &str = r#"
fn main() {
    let v = vec![1, 2, 3];
    for x in v.iter().cloned() {
        print!("{} ", x);
    }
    println!();
    let c: Vec<i32> = v.iter().cloned().collect();
    println!("collect={c:?}");
    println!("count={}", v.iter().cloned().count());
    let mut it = v.iter().cloned();
    println!("hint={:?}", it.size_hint());
    println!("next={:?}", it.next());
    println!("next_back={:?}", it.next_back());
    let words = vec![String::from("a"), String::from("bb")];
    for s in words.iter().cloned() {
        print!("{}:{} ", s, s.len());
    }
    println!();
    let w: Vec<String> = words.iter().cloned().collect();
    println!("{w:?}");
}
"#;

#[test]
fn run_mode_cloned_iteration() {
    let dir = temp_dir("cloned-iteration");
    let src = fixture(&dir, "cloned_iteration.rs", CLONED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cloned over slice iterators must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "1 2 3 \ncollect=[1, 2, 3]\ncount=3\nhint=(3, Some(3))\nnext=Some(1)\nnext_back=Some(3)\na:1 bb:2 \n[\"a\", \"bb\"]\n",
        "stderr={stderr}"
    );
}

const CLONED_HANDLES_SRC: &str = r#"
use std::rc::Rc;
fn main() {
    let v: Vec<Rc<i32>> = vec![Rc::new(1), Rc::new(2)];
    println!("before: {} {}", Rc::strong_count(&v[0]), Rc::strong_count(&v[1]));
    let w: Vec<Rc<i32>> = v.iter().cloned().collect();
    println!("after: {} {}", Rc::strong_count(&v[0]), Rc::strong_count(&v[1]));
    println!("vals: {} {}", w[0], w[1]);
    drop(w);
    println!("dropped: {}", Rc::strong_count(&v[0]));
    let boxes: Vec<Box<i32>> = vec![Box::new(10), Box::new(20)];
    let mut clones: Vec<Box<i32>> = boxes.iter().cloned().collect();
    *clones[0] = 99;
    println!("orig {} clone {}", boxes[0], clones[0]);
}
"#;

#[test]
fn run_mode_cloned_handle_elements() {
    let dir = temp_dir("cloned-handles");
    let src = fixture(&dir, "cloned_handles.rs", CLONED_HANDLES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cloned over Rc/Box elements must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "before: 1 1\nafter: 2 2\nvals: 1 2\ndropped: 1\norig 10 clone 99\n",
        "stderr={stderr}"
    );
}
const VEC_ZST_ENUM_ITER_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) {} }
enum CS<Y, R> { Yielded(Y), Complete(R) }
enum Never {}
fn main() {
    let v: Vec<CS<Never, DropMe>> = vec![CS::Complete(DropMe), CS::Complete(DropMe)];
    for x in &v {
        match x { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
    }
    let s: &[CS<Never, DropMe>] = &v;
    for x in s {
        match x { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
    }
    for x in &v[..] {
        match x { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
    }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_of_zero_sized_enum_elements_read_by_iteration() {
    // The elements of a `vec!` of a zero-sized enum reach an
    // iteration snapshot as uninitialized slots (their stores
    // are elided like the index-read case); the snapshot rescue
    // denotes the element type's single value, so borrowed
    // iteration, slice iteration and range-slice iteration all
    // read the `Complete` variant, exactly as natively.
    let dir = temp_dir("veczstenumiter");
    let src = fixture(&dir, "veczstenumiter.rs", VEC_ZST_ENUM_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a vec! of zero-sized enum elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\nc\nc\nc\nc\nc\ndone\n",
        "zero-sized enum elements read by iteration denote their single value"
    );
}

const VEC_ZST_ENUM_NESTED_ITER_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) {} }
enum CS<Y, R> { Yielded(Y), Complete(R) }
enum Never {}
fn main() {
    let v: Vec<Vec<CS<Never, DropMe>>> = vec![vec![CS::Complete(DropMe)]];
    for inner in &v {
        for x in inner {
            match x { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
        }
    }
    println!("done");
}
"#;

#[test]
fn run_mode_nested_vec_of_zero_sized_enum_elements_read_by_iteration() {
    // The inner `Vec` reaches the outer iteration's snapshot as
    // a value, detached from any place; the rescue follows the
    // recorded element path one index step deeper, so the
    // inner iteration reads the `Complete` variant too.
    let dir = temp_dir("veczstenumnestediter");
    let src = fixture(&dir, "veczstenumnestediter.rs", VEC_ZST_ENUM_NESTED_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a nested vec! of zero-sized enum elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\ndone\n",
        "nested zero-sized enum elements read by iteration denote their single value"
    );
}

const VEC_ZST_ENUM_STRUCT_FIELD_ITER_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
enum CS<Y, R> { Yielded(Y), Complete(R) }
enum Never {}
struct S { v: Vec<CS<Never, DropMe>> }
fn main() {
    let outer: Vec<S> = vec![
        S { v: vec![CS::Complete(DropMe)] },
        S { v: vec![CS::Complete(DropMe), CS::Complete(DropMe)] },
    ];
    for s in &outer {
        for x in &s.v { match x { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") } }
    }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_of_struct_with_zero_sized_enum_vec_field_read_by_iteration() {
    // The inner `Vec` sits in a field of the outer snapshot's
    // elements: the recorded path crosses a field step between
    // the two index steps, and the snapshot rescue follows it,
    // so the inner iteration reads the `Complete` variant
    // exactly as natively.
    let dir = temp_dir("veczstenumstructfielditer");
    let src = fixture(&dir, "veczstenumstructfielditer.rs", VEC_ZST_ENUM_STRUCT_FIELD_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a struct field's vec! of zero-sized enum elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\nc\nc\ndone\ndrop\ndrop\ndrop\n",
        "zero-sized enum elements behind a struct field denote their single value"
    );
}

const VEC_ZST_ENUM_TUPLE_FIELD_ITER_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
enum CS<Y, R> { Yielded(Y), Complete(R) }
enum Never {}
fn main() {
    let outer: Vec<(Vec<CS<Never, DropMe>>, u32)> = vec![
        (vec![CS::Complete(DropMe)], 7),
        (vec![CS::Complete(DropMe), CS::Complete(DropMe)], 8),
    ];
    for t in &outer {
        println!("n={}", t.1);
        for x in &t.0 { match x { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") } }
    }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_of_tuple_with_zero_sized_enum_vec_field_read_by_iteration() {
    // The same field crossing with a tuple element: the `Vec`
    // sits under the tuple's first field step.
    let dir = temp_dir("veczstenumtuplefielditer");
    let src = fixture(&dir, "veczstenumtuplefielditer.rs", VEC_ZST_ENUM_TUPLE_FIELD_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a tuple field's vec! of zero-sized enum elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "n=7\nc\nn=8\nc\nc\ndone\ndrop\ndrop\ndrop\n",
        "zero-sized enum elements behind a tuple field denote their single value"
    );
}

const SPARSE_TAIL_STRUCT_FIELD_ITER_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
enum CS<Y, R> { Yielded(Y), Complete(R) }
enum Never {}
struct S { v: Vec<CS<Never, DropMe>> }
fn main() {
    let mut inner: Vec<CS<Never, DropMe>> = Vec::new();
    unsafe { inner.set_len(2); }
    let outer: Vec<S> = vec![S { v: inner }];
    for s in &outer {
        for x in &s.v { match x { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") } }
    }
    println!("done");
}
"#;

#[test]
fn run_mode_sparse_tail_in_struct_field_read_by_nested_iteration() {
    // The inner `Vec`'s declared length runs past its stored
    // elements (a `set_len` growth): the virtual tail denotes
    // the element type's single value, and the snapshot rescue
    // materialises it behind the struct field's step, so the
    // nested iteration reads the `Complete` variant exactly as
    // natively.
    let dir = temp_dir("sparsetailstructfielditer");
    let src = fixture(&dir, "sparsetailstructfielditer.rs", SPARSE_TAIL_STRUCT_FIELD_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a sparse vec! behind a struct field must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\nc\ndone\ndrop\ndrop\n",
        "a sparse tail behind a struct field denotes the element type's single value"
    );
}


const ZIP_FOLD_SRC: &str = r#"
fn main() {
    let a = [1, 2, 3, 4];
    let b = [10, 20, 30, 40];
    let f = a.into_iter().zip(b.into_iter()).fold(0, |acc, (x, y)| acc + x * y);
    println!("fold {}", f);
    let s: i32 = a.into_iter().zip(b.into_iter()).map(|(x, y)| x + y).sum();
    println!("sum {}", s);
    println!("count {}", a.into_iter().zip(b.into_iter()).count());
    let mut fe = 0;
    a.into_iter().zip(b.into_iter()).for_each(|(x, y)| fe += x * y);
    println!("for_each {}", fe);
    println!("last {:?}", a.into_iter().zip(b.into_iter()).last());
    println!("vec {}", vec![1, 2, 3].into_iter().zip(vec![10, 20, 30].into_iter()).fold(0, |acc, (x, y)| acc + x * y));
    println!("uneven {}", (0..3).zip(0..10).fold(0, |acc, (x, y)| acc + x + y));
    println!("uneven count {}", (0..3).zip(0..10).count());
}
"#;

#[test]
fn run_mode_zip_fold_specialized_consumers() {
    // `fold` over a `Zip` whose sides are `TrustedRandomAccess`
    // iterators (array / `vec` into-iterators) resolves to
    // core's specialized `ZipImpl::fold`, which reads the sides
    // through `__iterator_get_unchecked` and the state's stored
    // index/length counters — machinery the modelled state does
    // not maintain, so the collected body failed (an
    // `IndexRange` type mismatch) or read uninitialized places.
    // The driver detects the specialization in the resolved
    // instance's delegation graph and lets the `Iterator::fold`
    // builtin serve the call; `sum` / `count` / `for_each` /
    // `last` reach the same point through their collected
    // default bodies, which delegate to `fold`.
    let dir = temp_dir("zip-fold");
    let src = fixture(&dir, "zip_fold.rs", ZIP_FOLD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Zip fold/sum/count/for_each over owned iterators must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "fold 300\nsum 110\ncount 4\nfor_each 300\nlast Some((4, 40))\nvec 140\nuneven 6\nuneven count 3\n",
        "the specialized consumers produce the general fold's results"
    );
}

const ZIP_FOLD_MIXED_SRC: &str = r#"
fn main() {
    let mut z = [1, 2, 3, 4].into_iter().zip([10, 20, 30, 40].into_iter());
    println!("first {:?}", z.next());
    println!("rest {}", z.fold(0, |acc, (x, y)| acc + x + y));
    let mut z2 = [1, 2, 3, 4].into_iter().zip([10, 20, 30, 40].into_iter());
    let _ = z2.next();
    println!("count {}", z2.count());
    println!("max {:?}", (0..4).zip(10..14).max());
    println!("reduce {:?}", (0..4).zip(10..14).reduce(|p, q| if q.0 > p.0 { q } else { p }));
}
"#;

#[test]
fn run_mode_zip_fold_after_next() {
    // Mixed consumption: a modelled `next` step followed by a
    // specialized consumer. The builtin `next` leaves the Zip
    // state's specialization counters untouched, so a collected
    // specialized `fold` afterwards reads a stale length (and,
    // over `Range` sides, fabricated the one-past-the-end pair
    // in `max` / `reduce`). Serving `fold` through the builtin
    // keeps one state discipline for the whole consumption.
    let dir = temp_dir("zip-fold-mixed");
    let src = fixture(&dir, "zip_fold_mixed.rs", ZIP_FOLD_MIXED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Zip next-then-fold must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "first Some((1, 10))\nrest 99\ncount 3\nmax Some((3, 13))\nreduce Some((3, 13))\n",
        "consumers after a modelled next see exactly the remaining items"
    );
}

const ZST_STRUCT_ITER_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
struct S { d: DropMe }
fn main() {
    let v = vec![S { d: DropMe }, S { d: DropMe }];
    for s in v {
        let _ = s;
        println!("c");
    }
    println!("done");
}
"#;

#[test]
fn run_mode_by_value_iteration_drops_zst_struct_elements() {
    // The loop variable of a by-value iteration over zero-sized
    // struct elements is never written — the payload move out of
    // the iterator is a no-op the MIR pipeline deletes — yet each
    // element's field destructors run at the end of its iteration,
    // once each, as natively.
    let dir = temp_dir("zststructiter");
    let src = fixture(&dir, "zststructiter.rs", ZST_STRUCT_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "by-value iteration over zero-sized structs must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\ndrop\nc\ndrop\ndone\n",
        "each iterated element's field destructor runs"
    );
}

const ZST_ENUM_MATCH_ITER_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
enum Never {}
enum CS { Yielded(Never), Complete(DropMe) }
fn main() {
    let v = vec![CS::Complete(DropMe), CS::Complete(DropMe)];
    let mut i = 0;
    for x in v {
        match x {
            CS::Yielded(_) => println!("y"),
            CS::Complete(_) => println!("{i}"),
        }
        i += 1;
    }
    println!("done");
}
"#;

#[test]
fn run_mode_match_consumed_zst_enum_elements_drop() {
    // An enum element consumed by a match in a by-value iteration
    // still drops its payload when the scrutinee goes out of scope —
    // for the second element exactly as for the first, even though
    // the scrutinee local's write was elided as a no-op.
    let dir = temp_dir("zstenummatchiter");
    let src = fixture(&dir, "zstenummatchiter.rs", ZST_ENUM_MATCH_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "match-consumed zero-sized enum iteration must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "0\ndrop\n1\ndrop\ndone\n",
        "both matched elements' payloads drop"
    );
}

const ZST_NESTED_ITER_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
enum CS<Y, R> { Yielded(Y), Complete(R) }
enum Never {}
struct S { v: Vec<CS<Never, DropMe>> }
fn main() {
    let outer: Vec<S> = vec![
        S { v: vec![CS::Complete(DropMe)] },
        S { v: vec![CS::Complete(DropMe), CS::Complete(DropMe)] },
    ];
    for s in outer {
        for x in s.v {
            match x {
                CS::Yielded(_) => println!("y"),
                CS::Complete(_) => println!("c"),
            }
        }
    }
    println!("done");
}
"#;

#[test]
fn run_mode_nested_by_value_iteration_drops_zst_payloads() {
    // Nested by-value iteration whose elements are structs holding
    // Vecs of zero-sized enums: every element's payload destructor
    // runs in iteration order, as natively.
    let dir = temp_dir("zstnestediter");
    let src = fixture(&dir, "zstnestediter.rs", ZST_NESTED_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested by-value iteration must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\ndrop\nc\ndrop\nc\ndrop\ndone\n",
        "every nested element's payload destructor runs in order"
    );
}

const SINGLE_ARRAY_FIELD_PUN_SRC: &str = r#"
struct S([u8; 4]);
struct W([u32; 2]);
fn main() {
    let a: [u8; 4] = [11, 22, 33, 44];
    let r: &S = unsafe { std::mem::transmute(&a) };
    println!("ref {} {} {} {}", r.0[0], r.0[1], r.0[2], r.0[3]);
    let f: [u8; 4] = r.0;
    println!("field {} {} {} {}", f[0], f[1], f[2], f[3]);
    let w: [u32; 2] = [0x11223344, 0x55667788];
    let p = &w as *const [u32; 2] as *const W;
    let w2 = unsafe { &*p };
    println!("ptr {:x} {:x}", w2.0[0], w2.0[1]);
    let mut b: [u8; 4] = [0, 0, 0, 0];
    let m: &mut S = unsafe { std::mem::transmute(&mut b) };
    m.0[2] = 77;
    println!("mut {} {} {} {}", b[0], b[1], b[2], b[3]);
    m.0 = [9, 8, 7, 6];
    println!("whole {} {} {} {}", b[0], b[1], b[2], b[3]);
    let bx: Box<[u8; 4]> = Box::new([5, 6, 7, 8]);
    let bs: Box<S> = unsafe { std::mem::transmute(bx) };
    println!("box {} {} {} {}", bs.0[0], bs.0[1], bs.0[2], bs.0[3]);
    let t = ([11u8, 22, 33, 44],);
    let tr: &[u8; 4] = unsafe { std::mem::transmute(&t) };
    println!("tup {} {} {} {}", tr[0], tr[1], tr[2], tr[3]);
    let br: &([u8; 4],) = unsafe { std::mem::transmute(&a) };
    println!("back {} {} {} {}", br.0[0], br.0[1], br.0[2], br.0[3]);
}
"#;

#[test]
fn run_mode_single_array_field_pun_reference_views() {
    // A struct (or one-element tuple) whose only field is an
    // array is layout-identical to the array, and a reference,
    // raw pointer or box transmuted or cast between the two
    // views denotes the same storage: the storage keeps the
    // form it was written with (the bare array), so reading
    // the wrapper's field through the other view must read
    // the array itself -- its elements at their indices, the
    // whole field as the whole array -- not the first element
    // and that element's bytes. Writes through the wrapper
    // view land in the array's elements.
    run_ok(
        "arrayfieldpun",
        SINGLE_ARRAY_FIELD_PUN_SRC,
        "ref 11 22 33 44\nfield 11 22 33 44\nptr 11223344 55667788\nmut 0 0 77 0\nwhole 9 8 7 6\nbox 5 6 7 8\ntup 11 22 33 44\nback 11 22 33 44\n",
    );
}

const ONE_ELEMENT_TUPLE_TRANSMUTE_SRC: &str = r#"
fn main() {
    let t = ([3u8, 1, 4, 1],);
    let a: [u8; 4] = unsafe { std::mem::transmute(t) };
    println!("out {} {} {} {}", a[0], a[1], a[2], a[3]);
    let b: ([u8; 4],) = unsafe { std::mem::transmute([9u8, 9, 9, 9]) };
    println!("in {} {} {} {}", b.0[0], b.0[1], b.0[2], b.0[3]);
    let s = (42u32,);
    let x: u32 = unsafe { std::mem::transmute(s) };
    println!("scalar {}", x);
    let s2: (u32,) = unsafe { std::mem::transmute(7u32) };
    println!("wrapped {}", s2.0);
}
"#;

#[test]
fn run_mode_one_element_tuple_transmute_crossings() {
    // A one-element tuple is a one-field wrapper, with the
    // same transmute crossings the one-field struct has: to
    // exactly the field's type it reads the field out, and
    // from exactly the field's type it wraps the value into
    // the tuple.
    run_ok(
        "onetupletransmute",
        ONE_ELEMENT_TUPLE_TRANSMUTE_SRC,
        "out 3 1 4 1\nin 9 9 9 9\nscalar 42\nwrapped 7\n",
    );
}

const SPARSE_NESTED_ARRAY_SRC: &str = r#"
use std::alloc::{alloc, Layout};

fn main() {
    unsafe {
        let layout = Layout::new::<[[u32; 2]; 2]>();
        let p = alloc(layout) as *mut [[u32; 2]; 2];
        (*p)[0] = [11, 22];
        println!("row {:?}", (*p)[0]);
        println!("elem {}", (*p)[0][1]);
        let q = p as *const [u32; 2];
        println!("via_row_ptr {}", (*q)[1]);
        (*p)[0][1] = 99;
        println!("written {:?}", (*p)[0]);
        println!("via_row_ptr_after {}", (*q)[1]);
    }
}
"#;

#[test]
fn run_mode_sparse_nested_array_prefix_is_not_a_tuple_wrapper() {
    // A nested array whose first row alone is materialised
    // stores as a one-item tuple holding the row -- the same
    // shape a one-element tuple wrapper around the whole array
    // has. The dereference of the array pointer must keep the
    // genuine reading (the tuple is the array's own sparse
    // prefix): reading the row reads the row, indexing through
    // it reaches the row's elements, and writes through the
    // array view land in the row the row view reads back.
    run_ok(
        "sparsenestedarray",
        SPARSE_NESTED_ARRAY_SRC,
        "row [11, 22]\nelem 22\nvia_row_ptr 22\nwritten [11, 99]\nvia_row_ptr_after 99\n",
    );
}

const ZST_RAW_SLICE_ITER_SRC: &str = r#"
use std::slice;

fn foo<T>(v: &[T]) -> Option<&[T]> {
    let mut it = v.iter();
    for _ in 0..5 {
        let _ = it.next();
    }
    Some(it.as_slice())
}

fn foo_mut<T>(v: &mut [T]) -> Option<&mut [T]> {
    let mut it = v.iter_mut();
    for _ in 0..5 {
        let _ = it.next();
    }
    Some(it.into_slice())
}

fn main() {
    let slice: &[()] = unsafe { slice::from_raw_parts(-5isize as *const (), 10) };
    println!("len {}", slice.len());
    println!("count {}", slice.iter().count());
    let mut it = slice.iter();
    println!("nth some {}", it.nth(5).is_some());
    println!("rest {}", it.count());
    println!("foo some {}", foo(slice).is_some());
    let slice: &mut [()] = unsafe { slice::from_raw_parts_mut(-5isize as *mut (), 10) };
    println!("mcount {}", slice.iter_mut().count());
    {
        let mut it = slice.iter_mut();
        println!("mnth some {}", it.nth(5).is_some());
        println!("mrest {}", it.count());
    }
    println!("foo_mut some {}", foo_mut(slice).is_some());
}
"#;

#[test]
fn run_mode_zst_raw_slice_iterators() {
    // A slice of zero-sized elements over a raw-parts pointer
    // at the edge of the address space: the iterator's end
    // pointer carries the remaining count in its own bits, a
    // representation the collected core bodies read through
    // pointer-to-integer puns the value model does not
    // reproduce. Iteration is served by the modelled `next`
    // state machine (whose address-pair step synthesises the
    // elements), and the consumers (`count`, `nth`) are served
    // by the same machine so the accounting cannot disagree.
    // This program used to spin forever in `slice_elements`
    // (the fat pointer over the raw address resolved to
    // itself).
    run_ok(
        "zst-raw-slice-iterators",
        ZST_RAW_SLICE_ITER_SRC,
        "len 10\ncount 10\nnth some true\nrest 4\nfoo some true\nmcount 10\nmnth some true\nmrest 4\nfoo_mut some true\n",
    );
}

const ZST_ARRAY_ITER_NTH_SRC: &str = r#"
fn main() {
    let a = [(), (), (), (), (), (), (), (), (), ()];
    println!("count {}", a.iter().count());
    let mut it = a.iter();
    println!("nth some {}", it.nth(5).is_some());
    println!("rest {}", it.count());
    let mut m = [(), (), (), ()];
    println!("mcount {}", m.iter_mut().count());
}
"#;

#[test]
fn run_mode_zst_array_iter_nth_count() {
    // `nth` on a zero-sized snapshot iterator: core's body
    // reads the end pointer's bits as the remaining count
    // through a `*mut usize` pun, which the value model cannot
    // serve; the modelled `nth` drives the `next` transition
    // instead.
    run_ok(
        "zst-array-iter-nth-count",
        ZST_ARRAY_ITER_NTH_SRC,
        "count 10\nnth some true\nrest 4\nmcount 4\n",
    );
}

const NTH_SKIPPED_DROP_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};

static DROPS: AtomicUsize = AtomicUsize::new(0);

struct Frag(u32);
impl Drop for Frag {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

fn main() {
    let mut it = vec![Frag(1), Frag(2), Frag(3)].into_iter();
    let second = it.nth(1);
    println!("got {}", second.unwrap().0);
    println!("drops after nth {}", DROPS.load(Ordering::SeqCst));
    drop(it);
    println!("drops after drop(it) {}", DROPS.load(Ordering::SeqCst));
}
"#;

#[test]
fn run_mode_iterator_nth_drops_skipped_items() {
    // The items `nth` skips are consumed by the skip: the
    // modelled `nth` runs each skipped item's drop glue
    // (baked by conversion from the item type) before
    // stepping on, exactly as core's default body drops the
    // temporary each `advance_by` iteration yields. Before
    // the glue ran, the counter read 1 after `nth` (only the
    // returned item's temporary) and 2 at the end: the
    // skipped element's destructor never ran, and its slot
    // in the iterator was already marked consumed, so the
    // iterator's own drop could not recover it.
    run_ok(
        "iterator-nth-drops-skipped-items",
        NTH_SKIPPED_DROP_SRC,
        "got 2\ndrops after nth 2\ndrops after drop(it) 3\n",
    );
}

const NTH_RANGE_ADVANCE_SRC: &str = r#"
fn main() {
    let mut r = 0usize..2_000_000usize;
    println!("nth {:?}", r.nth(1_500_000));
    let mut ri = 0usize..=2_000_000usize;
    println!("incl nth {:?}", ri.nth(1_500_000));
}
"#;

#[test]
fn run_mode_iterator_nth_range_advances_directly() {
    // An arithmetic range's `nth` advances in one step, as
    // the specialised body in core does: driving the
    // modelled `next` transition once per skipped item
    // spends a step per item and a 1.5M skip exceeds the
    // default step limit before it can answer.
    run_ok(
        "iterator-nth-range-advances-directly",
        NTH_RANGE_ADVANCE_SRC,
        "nth Some(1500000)\nincl nth Some(1500000)\n",
    );
}

const NTH_CHAR_RANGE_GAP_SRC: &str = r#"
fn show(tag: &str, v: Option<char>) {
    match v {
        Some(c) => println!("{} Some({})", tag, c as u32),
        None => println!("{} None", tag),
    }
}

fn main() {
    // The Codex repro: skipping one item of '\u{d7ff}'..'\u{e001}'
    // must land on '\u{e000}', not on the surrogate 0xD800.
    show("codex", ('\u{d7ff}'..'\u{e001}').nth(1));
    show("nth0", ('\u{d7ff}'..'\u{e001}').nth(0));
    // Exactly the range length (2) and beyond it: no element.
    show("nth-len", ('\u{d7ff}'..'\u{e001}').nth(2));
    show("nth-over", ('\u{d7ff}'..'\u{e001}').nth(3));
    // A range whose raw span crosses the whole gap but holds
    // one element: '\u{d7ff}'..'\u{e000}'.
    show("gap1-nth0", ('\u{d7ff}'..'\u{e000}').nth(0));
    show("gap1-nth1", ('\u{d7ff}'..'\u{e000}').nth(1));
    // Crossing the gap with a wider skip.
    show("wide", ('\u{d7f0}'..'\u{e010}').nth(17));
    // Starting above the gap.
    show("above", ('\u{e000}'..'\u{e003}').nth(1));
    // Inclusive ranges across the gap.
    show("incl-nth1", ('\u{d7ff}'..='\u{e000}').nth(1));
    show("incl-nth-len", ('\u{d7ff}'..='\u{e000}').nth(2));
    show("incl-single", ('\u{d7ff}'..='\u{d7ff}').nth(0));
    show("incl-single-over", ('\u{d7ff}'..='\u{d7ff}').nth(1));
    // An inclusive range exhausted by a skip stays exhausted
    // from both ends.
    let mut ri = '\u{d7ff}'..='\u{e000}';
    show("incl-exh-nth", ri.nth(5));
    show("incl-exh-next", ri.next());
    show("incl-exh-back", ri.next_back());
    // A skip that lands just below the gap: the item `nth`
    // returns is '\u{d7ff}', and the following step crosses
    // the gap.
    let mut r = '\u{d7fe}'..'\u{e001}';
    show("land-below", r.nth(1));
    show("after-nth", r.next());
    show("after-nth2", r.next());
    // A skip past the end leaves an exclusive range exhausted.
    let mut ro = 'a'..'c';
    show("over-nth", ro.nth(2));
    show("over-next", ro.next());
    // Plain char ranges are unchanged.
    show("normal", ('a'..'f').nth(3));
    show("normal-incl", ('a'..='f').nth(5));
    show("normal-incl-over", ('a'..='f').nth(6));
    // Forward and backward per-item steps cross the gap too.
    let fwd: Vec<u32> = ('\u{d7ff}'..'\u{e001}').map(|c| c as u32).collect();
    println!("fwd {:?}", fwd);
    let back: Vec<u32> = ('\u{d7ff}'..'\u{e001}').rev().map(|c| c as u32).collect();
    println!("back {:?}", back);
    let incl: Vec<u32> = ('\u{d7fe}'..='\u{e000}').map(|c| c as u32).collect();
    println!("incl {:?}", incl);
    // A large char skip still advances in one step.
    show("big", ('\0'..='\u{10ffff}').nth(1_000_000));
}
"#;

#[test]
fn run_mode_iterator_nth_char_range_surrogate_gap() {
    // `nth` on a `Range<char>` / `RangeInclusive<char>`:
    // the one-step advance and the range's length must both
    // skip the surrogate gap [0xD800, 0xE000), as
    // `range_count`'s length already did — the raw
    // code-point arithmetic returned the invalid char
    // 0xD800 for ('\u{d7ff}'..'\u{e001}').nth(1) and
    // served elements past the end of gap-crossing ranges
    // whose raw span overstates their length. The per-item
    // steps share the same arithmetic, so the sequences
    // after a skip (and plain collect / rev) cross the gap
    // the same way.
    run_ok(
        "iterator-nth-char-range-surrogate-gap",
        NTH_CHAR_RANGE_GAP_SRC,
        "codex Some(57344)\nnth0 Some(55295)\nnth-len None\nnth-over None\ngap1-nth0 Some(55295)\ngap1-nth1 None\nwide Some(57345)\nabove Some(57345)\nincl-nth1 Some(57344)\nincl-nth-len None\nincl-single Some(55295)\nincl-single-over None\nincl-exh-nth None\nincl-exh-next None\nincl-exh-back None\nland-below Some(55295)\nafter-nth Some(57344)\nafter-nth2 None\nover-nth None\nover-next None\nnormal Some(100)\nnormal-incl Some(102)\nnormal-incl-over None\nfwd [55295, 57344]\nback [57344, 55295]\nincl [55294, 55295, 57344]\nbig Some(1002048)\n",
    );
}

const ITER_MUT_RAW_PARTS_SRC: &str = r#"
use std::slice;

fn main() {
    let mut v = vec![10, 20, 30, 40];
    let ptr = v.as_mut_ptr();
    let s: &mut [i32] = unsafe { slice::from_raw_parts_mut(ptr, 4) };
    let mut it = s.iter_mut();
    println!("first {:?}", it.next());
    let out = it.into_slice();
    println!("rest {:?}", out);
}
"#;

#[test]
fn run_mode_slice_iter_mut_raw_parts_next() {
    // `next` on the real field layout (ptr, end, marker) the
    // collected `<[T]>::iter_mut` body builds over a raw-parts
    // slice: the state machine's `IterMut` arm used to serve
    // only the [container, index, length] state `IntoIterator`
    // builds, and reported this shape as a type mismatch.
    run_ok(
        "slice-iter-mut-raw-parts-next",
        ITER_MUT_RAW_PARTS_SRC,
        "first Some(10)\nrest [20, 30, 40]\n",
    );
}

const LIBC_ATOL_SRC: &str = r#"
use std::ffi::{c_char, c_int, c_long, c_longlong, CString};

mod mlibc {
    use std::ffi::{c_char, c_long, c_longlong};
    extern "C" {
        pub fn atol(x: *const c_char) -> c_long;
        pub fn atoll(x: *const c_char) -> c_longlong;
    }
}

extern "C" {
    fn atoi(x: *const c_char) -> c_int;
}

fn main() {
    let a = CString::new("1024").unwrap();
    let b = CString::new("  -42xyz").unwrap();
    let c = CString::new("11111111111111111").unwrap();
    let d = CString::new("+77").unwrap();
    let e = CString::new("zzz").unwrap();
    unsafe {
        println!("atol={}", mlibc::atol(a.as_ptr()));
        println!("atol_neg={}", mlibc::atol(b.as_ptr()));
        println!("atoll={}", mlibc::atoll(c.as_ptr()));
        println!("atoi={}", atoi(d.as_ptr()));
        println!("atoi_none={}", atoi(e.as_ptr()));
    }
}
"#;

#[test]
fn run_mode_libc_atol_family() {
    // Body-less extern declarations of the libc
    // string-to-integer family are served by delegating to
    // the host C library function of the same name (the
    // shape of tests/ui/abi/c-stack-returning-int64.rs,
    // whose declarations live in a module).
    let dir = temp_dir("libc-atol");
    let src = fixture(&dir, "libc_atol.rs", LIBC_ATOL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the atol family must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "atol=1024\natol_neg=-42\natoll=11111111111111111\natoi=77\natoi_none=0\n",
        "the host libc functions' own results"
    );
}

const LIBC_STRTOL_SRC: &str = r#"
use std::ffi::{c_char, c_int, c_long, c_ulong, CString};

extern "C" {
    fn strtol(x: *const c_char, end: *mut *mut c_char, base: c_int) -> c_long;
    fn strtoul(x: *const c_char, end: *mut *mut c_char, base: c_int) -> c_ulong;
}

fn main() {
    let a = CString::new("0x1F").unwrap();
    let b = CString::new("-1").unwrap();
    let c = CString::new("99999999999999999999999999").unwrap();
    let d = CString::new("12ab34").unwrap();
    unsafe {
        println!("hex={}", strtol(a.as_ptr(), std::ptr::null_mut(), 0));
        println!("neg_ul={}", strtoul(b.as_ptr(), std::ptr::null_mut(), 10));
        println!("sat={}", strtol(c.as_ptr(), std::ptr::null_mut(), 10));
        println!("mid={}", strtol(d.as_ptr().add(4), std::ptr::null_mut(), 10));
    }
}
"#;

#[test]
fn run_mode_libc_strtol_null_endptr() {
    // The strto* family delegates with a NULL endptr:
    // base-0 prefix detection, the unsigned wrap of a
    // negative input, saturation on overflow, and a
    // mid-string argument pointer all take the host
    // function's own results.
    let dir = temp_dir("libc-strtol");
    let src = fixture(&dir, "libc_strtol.rs", LIBC_STRTOL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "strtol/strtoul with a NULL endptr must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hex=31\nneg_ul=18446744073709551615\nsat=9223372036854775807\nmid=34\n",
        "the host libc functions' own results"
    );
}

const LIBC_SIGNED_CHAR_SRC: &str = r#"
use std::ffi::{c_char, c_int, c_long};

extern "C" {
    fn atol(x: *const c_char) -> c_long;
    fn atoi(x: *const c_char) -> c_int;
}

fn main() {
    let a: [i8; 3] = [49, -1, 0];
    let b: [i8; 4] = [53, 50, -128, 0];
    let c: [u8; 3] = [49, 50, 0];
    let d: [i8; 3] = [49, 50, 0];
    unsafe {
        println!("signed_high={}", atol(a.as_ptr()));
        println!("signed_high2={}", atoi(b.as_ptr()));
        println!("plain_u8={}", atol(c.as_ptr().cast()));
        println!("plain_i8={}", atol(d.as_ptr()));
    }
}
"#;

#[test]
fn run_mode_libc_signed_char_bytes() {
    // A signed `c_char` array holding a byte above 127: the
    // scan keeps the byte's bit pattern (the host reads
    // "1\xff" and stops there), instead of rejecting the
    // negative element as a type mismatch.
    let dir = temp_dir("libc-signed-char");
    let src = fixture(&dir, "libc_signed_char.rs", LIBC_SIGNED_CHAR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "signed char byte strings must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "signed_high=1\nsigned_high2=52\nplain_u8=12\nplain_i8=12\n",
        "the host libc functions' own results"
    );
}

const COMPARE_BYTES_SIGNED_SRC: &str = r#"
#![feature(core_intrinsics)]
use std::intrinsics::compare_bytes;

fn main() {
    let a: [i8; 4] = [1, -128, 3, 0];
    let b: [i8; 4] = [1, 127, 3, 0];
    let c: [i8; 4] = [1, -128, 3, 0];
    let d: [i8; 4] = [1, -128, 2, 0];
    unsafe {
        let gt = compare_bytes(a.as_ptr().cast(), b.as_ptr().cast(), 4);
        let eq = compare_bytes(a.as_ptr().cast(), c.as_ptr().cast(), 4);
        let lt = compare_bytes(b.as_ptr().cast(), a.as_ptr().cast(), 4);
        let tail = compare_bytes(a.as_ptr().cast(), d.as_ptr().cast(), 4);
        println!("gt={} eq={} lt={} tail={}", gt > 0, eq == 0, lt < 0, tail > 0);
    }
}
"#;

#[test]
fn run_mode_compare_bytes_signed_elements() {
    // `compare_bytes` over signed-element storage: an `i8`
    // element holding a byte above 127 keeps its bit pattern
    // (0x80 compares above 0x7f), instead of being rejected
    // as a type mismatch -- the same acceptance `read_c_str`
    // already gives the libc string scans.
    let dir = temp_dir("compare-bytes-signed");
    let src = fixture(&dir, "compare_bytes_signed.rs", COMPARE_BYTES_SIGNED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "compare_bytes over signed bytes must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "gt=true eq=true lt=true tail=true\n",
        "unsigned byte order over the signed elements' bit patterns"
    );
}

const LIBC_WIDE_ELEM_SRC: &str = r#"
use std::ffi::{c_char, c_long};

extern "C" {
    fn atol(x: *const c_char) -> c_long;
}

fn main() {
    let a: [u16; 3] = [49, 50, 0];
    let b: [u16; 3] = [0x3231, 0x0033, 0];
    let c: [u32; 2] = [0x00323331, 0];
    let d: [i16; 3] = [49, -1, 0];
    let e: [i16; 3] = [49, -256, 0];
    unsafe {
        println!("u16_cast={}", atol(a.as_ptr().cast()));
        println!("u16_packed={}", atol(b.as_ptr().cast()));
        println!("u32_packed={}", atol(c.as_ptr().cast()));
        println!("i16_cast={}", atol(d.as_ptr().cast()));
        println!("byte_disp={}", atol(b.as_ptr().cast::<c_char>().add(1)));
        println!("elem_disp={}", atol(b.as_ptr().add(1).cast()));
        println!("i16_neg={}", atol(e.as_ptr().cast()));
    }
}
"#;

#[test]
fn run_mode_libc_wide_element_storage() {
    // A `*const c_char` cast into wider-element storage
    // denotes the storage's little-endian bytes, not its
    // elements: the scan reads the declared element type's
    // bytes, so a high byte of an element terminates (or
    // joins) the string exactly as on the host.
    let dir = temp_dir("libc-wide-elem");
    let src = fixture(&dir, "libc_wide_elem.rs", LIBC_WIDE_ELEM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "wide-element storage casts must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "u16_cast=1\nu16_packed=123\nu32_packed=132\ni16_cast=1\nbyte_disp=23\nelem_disp=3\ni16_neg=1\n",
        "the host libc functions' own results"
    );
}

const LIBC_STRTOL_ENDPTR_SRC: &str = r#"
use std::ffi::{c_char, c_int, c_long, CString};

extern "C" {
    fn strtol(x: *const c_char, end: *mut *mut c_char, base: c_int) -> c_long;
}

fn main() {
    let s = CString::new("123").unwrap();
    let mut end: *mut c_char = std::ptr::null_mut();
    unsafe {
        let v = strtol(s.as_ptr(), &mut end, 10);
        println!("v={v}");
    }
}
"#;

#[test]
fn run_mode_libc_strtol_nonnull_endptr_is_explicit_error() {
    // A non-NULL endptr is an explicit unsupported error,
    // never a silently skipped write-back.
    let dir = temp_dir("libc-strtol-endptr");
    let src = fixture(&dir, "libc_strtol_endptr.rs", LIBC_STRTOL_ENDPTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a non-NULL endptr must be refused; stderr={stderr}"
    );
    assert!(
        stderr.contains("non-NULL endptr is not modelled"),
        "the refusal names the endptr model limit; stderr={stderr}"
    );
}
// ----- modelled handle values under declared-type drop glue ---------
// Drop glue built from a declared type can name a handle at a
// position where a builtin model keeps a scalar stand-in: the
// thread model's `Thread` value is an id, although the declared
// type's field is an `Arc`. Dropping a struct that holds such a
// value runs the handle plan against the id; a scalar holds no
// count, so the drop completes (and the struct's own destructor
// still runs). A real handle aggregate in the same position is
// still released.

const THREAD_HOLDER_SRC: &str = r#"
struct ThreadHolder(std::thread::Thread);

struct Observed(std::thread::Thread);
impl Drop for Observed {
    fn drop(&mut self) {
        println!("observed");
    }
}

fn main() {
    let h = ThreadHolder(std::thread::current());
    drop(h);
    let o = Observed(std::thread::current());
    drop(o);
    let same = std::thread::current().id() == std::thread::current().id();
    println!("same {same}");
    println!("done");
}
"#;

#[test]
fn run_mode_thread_value_in_struct_field_drops() {
    let dir = temp_dir("thread-holder-drop");
    let src = fixture(&dir, "thread_holder.rs", THREAD_HOLDER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dropping a struct holding a modelled Thread value must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "observed\nsame true\ndone\n",
        "the struct destructor runs and the modelled handle drop is inert"
    );
}

const ARC_HOLDER_SRC: &str = r#"
use std::sync::Arc;
struct Holder(Arc<u32>);
fn main() {
    let a = Arc::new(7u32);
    let h = Holder(a.clone());
    println!("during {}", Arc::strong_count(&a));
    drop(h);
    println!("after {}", Arc::strong_count(&a));
}
"#;

#[test]
fn run_mode_arc_handle_in_struct_field_still_releases() {
    let dir = temp_dir("arc-holder-drop");
    let src = fixture(&dir, "arc_holder.rs", ARC_HOLDER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dropping a struct holding a real Arc must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "during 2\nafter 1\n",
        "a real handle aggregate in a struct field is still released"
    );
}
const VEC_CAPACITY_GROWTH_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::new();
    for i in 0..5 {
        v.push(i);
    }
    println!("push5 cap {} spare {}", v.capacity(), v.spare_capacity_mut().len());
    let mut b: Vec<u8> = Vec::new();
    b.push(1);
    println!("u8 push1 cap {}", b.capacity());
    let mut w: Vec<i32> = Vec::with_capacity(1);
    w.push(1);
    w.push(2);
    println!("withcap1 push2 cap {}", w.capacity());
    let mut r = vec![1, 2, 3];
    r.reserve(1);
    println!("reserve cap {}", r.capacity());
    let mut e = vec![1, 2, 3];
    e.reserve_exact(1);
    println!("reserve_exact cap {}", e.capacity());
    let mut t = vec![1, 2, 3];
    println!("try_reserve ok {}", t.try_reserve(20).is_ok());
    println!("try_reserve cap {}", t.capacity());
}
"#;

#[test]
fn run_mode_vec_capacity_growth_matches_rawvec() {
    // Capacity is the storage extent in this model, and growth
    // follows RawVec's rules exactly: an append that outgrows
    // the storage doubles it (floored at the minimum non-zero
    // capacity for the element size -- 8 for bytes, 4 for
    // 4-byte elements), `reserve` grows amortized,
    // `reserve_exact` to the exact requirement, and a
    // successful `try_reserve` grows like `reserve`. The
    // answers are observable through `capacity()` and
    // `spare_capacity_mut().len()`, as natively.
    let dir = temp_dir("veccapacitygrowth");
    let src = fixture(&dir, "veccapacitygrowth.rs", VEC_CAPACITY_GROWTH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec capacity growth must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "push5 cap 8 spare 3\nu8 push1 cap 8\nwithcap1 push2 cap 4\nreserve cap 6\nreserve_exact cap 4\ntry_reserve ok true\ntry_reserve cap 23\n"
    );
}

const VEC_SPARE_WRITE_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::with_capacity(8);
    v.push(10);
    v.push(20);
    let spare = v.spare_capacity_mut();
    println!("spare {}", spare.len());
    spare[0].write(30);
    spare[1].write(40);
    unsafe { v.set_len(4) };
    println!("v = {:?}", v);
    println!("idx {}", v[2]);
    println!("pop {:?}", v.pop());
}
"#;

#[test]
fn run_mode_vec_spare_capacity_write_then_set_len_readback() {
    // Elements written through the spare-capacity slice are
    // stored as `MaybeUninit` payload slots; once `set_len`
    // declares them, every read of them -- indexing, the
    // vector's own Debug list, `pop` -- sees the payload, as
    // natively. (Formatting a spare-written element goes
    // through the fmt engine's storage-wrapper unwrap.)
    let dir = temp_dir("vecsparewrite");
    let src = fixture(&dir, "vecsparewrite.rs", VEC_SPARE_WRITE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "spare-capacity write + set_len readback must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "spare 6\nv = [10, 20, 30, 40]\nidx 30\npop Some(40)\n"
    );
}

const VEC_ZST_CAPACITY_SRC: &str = r#"
fn main() {
    let mut v: Vec<()> = Vec::with_capacity(10);
    println!("cap {}", v.capacity());
    println!("spare {}", v.spare_capacity_mut().len());
    v.push(());
    println!("cap {}", v.capacity());
}
"#;

#[test]
fn run_mode_vec_zero_sized_element_capacity() {
    // A zero-sized element's allocation is the dangling
    // pointer: std's capacity for it is `usize::MAX` by
    // definition, not the storage extent the model holds.
    let dir = temp_dir("veczstcapacity");
    let src = fixture(&dir, "veczstcapacity.rs", VEC_ZST_CAPACITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "ZST Vec capacity must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "cap 18446744073709551615\nspare 18446744073709551615\ncap 18446744073709551615\n"
    );
}

const VEC_SPARSE_ORDERING_SRC: &str = r#"
use std::cmp::Ordering;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Wrap(Vec<i32>);

fn main() {
    let mut v: Vec<i32> = Vec::new();
    v.push(1);
    println!("self le {} cmp {:?}", v <= v, v.cmp(&v));
    println!("partial {:?}", v.partial_cmp(&v));

    let mut a: Vec<i32> = Vec::new();
    a.push(1);
    a.push(3);
    let mut b: Vec<i32> = Vec::new();
    b.push(1);
    b.push(2);
    b.push(9);
    println!("a<b {} a>b {} a==b {}", a < b, a > b, a == b);
    println!("a cmp {:?}", a.cmp(&b));

    let lit = vec![1, 3];
    println!("a==lit {} a<=lit {} a>=lit {}", a == lit, a <= lit, a >= lit);
    let lit2 = vec![1, 4];
    println!("a<lit2 {} a cmp lit2 {:?}", a < lit2, a.cmp(&lit2));

    let mut short: Vec<i32> = Vec::new();
    short.push(1);
    println!("short<a {} short cmp a {:?}", short < a, short.cmp(&a));

    let w1 = Wrap(a.clone());
    let w2 = Wrap(vec![1, 3]);
    println!("wrap cmp {:?}", w1.cmp(&w2));
    let _ = Ordering::Equal;
}
"#;

#[test]
fn run_mode_vec_sparse_ordering_after_growth() {
    // A push that leaves spare capacity puts the vector in
    // its sparse form (declared length below the stored
    // elements); ordering -- `PartialOrd`, `cmp`,
    // `partial_cmp`, and derived `Ord` over a field holding
    // such a vector -- compares the declared elements only,
    // exactly as equality does, against sparse and plain
    // forms alike. The spare storage past the length is not
    // part of the value and never enters the comparison.
    let dir = temp_dir("vecsparseord");
    let src = fixture(&dir, "vecsparseord.rs", VEC_SPARSE_ORDERING_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "ordering of spare-capacity vectors must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "self le true cmp Equal\npartial Some(Equal)\na<b false a>b true a==b false\na cmp Greater\na==lit true a<=lit true a>=lit true\na<lit2 true a cmp lit2 Less\nshort<a true short cmp a Less\nwrap cmp Equal\n"
    );
}

const VEC_SPARSE_CONSUMERS_SRC: &str = r#"
use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

fn main() {
    let mut v = Vec::new();
    v.push(10);
    v.push(20);
    for x in v {
        println!("elem {x}");
    }

    let mut v = Vec::new();
    v.push(10);
    v.push(20);
    let r: Rc<[i32]> = v.into();
    println!("rc {:?} len {}", r, r.len());
    let mut w = Vec::new();
    w.push(7);
    let a: Arc<[i32]> = w.into();
    println!("arc {:?} len {}", a, a.len());

    let mut v = Vec::new();
    v.push(10);
    v.push(20);
    let mut d: VecDeque<i32> = v.into();
    println!("deque len {}", d.len());
    println!("front {:?} back {:?}", d.front(), d.back());
    println!("pop {:?}", d.pop_front());

    let mut bytes: Vec<u8> = Vec::new();
    bytes.push(65u8);
    write!(bytes, "BC").unwrap();
    println!("written {:?}", String::from_utf8(bytes).unwrap());

    let mut paths = Vec::new();
    paths.push(PathBuf::from("/a"));
    paths.push(PathBuf::from("/b"));
    let joined = std::env::join_paths(paths).unwrap();
    println!("joined {}", joined.to_string_lossy());
}
"#;

#[test]
fn run_mode_vec_sparse_consumers_after_growth() {
    // Consumers of a vector the growth path left in its
    // sparse form take its declared elements, exactly as
    // equality and ordering do: owning iteration, the
    // counted-slice conversions (`Rc<[T]>` / `Arc<[T]>`),
    // the `VecDeque` conversion, a formatted write into a
    // byte vector, and `join_paths` over a by-value vector.
    // The spare storage past the length is not part of the
    // value and never reaches any of them.
    let dir = temp_dir("vecsparseuse");
    let src = fixture(&dir, "vecsparseuse.rs", VEC_SPARSE_CONSUMERS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "consumers of spare-capacity vectors must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "elem 10\nelem 20\nrc [10, 20] len 2\narc [7] len 1\ndeque len 2\nfront Some(10) back Some(20)\npop Some(10)\nwritten \"ABC\"\njoined /a:/b\n"
    );
}

const PATH_FMT_SRC: &str = r#"
use std::ffi::OsString;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Wrap {
    p: PathBuf,
    n: i32,
}

fn main() {
    // The Debug of the path / OS-string wrapper types is the
    // quoted content, not the wrapper structure; width and
    // precision are not consulted by std's impls.
    println!("{:?}", PathBuf::from("a/b"));
    println!("{:?}", Path::new("a/b"));
    println!("{:?}", OsString::from("os/s"));
    println!("{:?}", vec![PathBuf::from("x"), PathBuf::from("y/z")]);
    println!("{:.2?}", PathBuf::from("abcdef"));
    println!("{:>12?}", PathBuf::from("ab"));
    println!("{:?}", Wrap { p: PathBuf::from("in/struct"), n: 3 });
    println!("{:?}", PathBuf::from("a\\b\"c"));
    println!("{:?}", PathBuf::from(""));
    println!("path-fmt-ok");
}
"#;

#[test]
fn run_mode_path_and_os_string_debug_formats_content() {
    let dir = temp_dir("path-fmt");
    let src = fixture(&dir, "path_fmt.rs", PATH_FMT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "path Debug formatting runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "\"a/b\"\n\"a/b\"\n\"os/s\"\n[\"x\", \"y/z\"]\n\"abcdef\"\n\"ab\"\nWrap { p: \"in/struct\", n: 3 }\n\"a\\\\b\\\"c\"\n\"\"\npath-fmt-ok\n",
        "PathBuf/Path/OsString Debug renders the quoted content like std"
    );
}

const PATH_DISPLAY_SRC: &str = r#"
use std::path::{Path, PathBuf};

fn main() {
    println!("{}", Path::new("a/b").display());
    println!("{}", PathBuf::from("dir/file.txt").display());
    let s = Path::new("x/y").display().to_string();
    println!("{s}");
    let pb = PathBuf::from("p/q");
    println!("{}", pb.display());
    println!("path-display-ok");
}
"#;

#[test]
fn run_mode_path_display_and_to_string() {
    let dir = temp_dir("path-display");
    let src = fixture(&dir, "path_display.rs", PATH_DISPLAY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "path Display formatting runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "a/b\ndir/file.txt\nx/y\np/q\npath-display-ok\n",
        "Path::display renders the path content and to_string converts it"
    );
}

const NON_UTF8_PATH_DISPLAY_SRC: &str = r#"
use std::ffi::OsString;
use std::os::unix::ffi::OsStringExt;

fn main() {
    // A non-UTF-8 `OsString` (built from raw bytes): its
    // `&OsStr` form is a fat pointer to the byte elements in
    // this model, and the `OsStr` / `Slice` field projections
    // inside `OsStr::len` and `Path::display` walk the
    // one-field wrapper chain over that fat pointer.
    let s = OsString::from_vec(vec![0x61, 0xff, 0x62]);
    println!("len={}", s.len());
    let pb = std::path::PathBuf::from(s);
    println!("{}", pb.display());
    println!("len={}", pb.as_os_str().len());
    println!("non-utf8-path-display-ok");
}
"#;

#[test]
fn run_mode_non_utf8_path_display() {
    let dir = temp_dir("non-utf8-path-display");
    let src = fixture(&dir, "non_utf8_path_display.rs", NON_UTF8_PATH_DISPLAY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "non-UTF-8 path display runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len=3\na\u{fffd}b\nlen=3\nnon-utf8-path-display-ok\n",
        "display() renders the lossy content and len() reads the byte length"
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

const PATHBUF_AS_PATH_SRC: &str = r#"
use std::path::{Path, PathBuf};
fn main() {
    let pb = PathBuf::from("abc");
    let p = pb.as_path();
    println!("len {}", p.as_os_str().len());
    println!("eq {}", p == Path::new("abc"));
    println!("done");
}
"#;

#[test]
fn run_mode_pathbuf_as_path() {
    run_ok(
        "pathbuf-as-path",
        PATHBUF_AS_PATH_SRC,
        "len 3\neq true\ndone\n",
    );
}

const OSSTRING_AS_OS_STR_SRC: &str = r#"
use std::ffi::OsString;
fn main() {
    let s = OsString::from("a/b");
    println!("len {}", s.as_os_str().len());
    let b = &s[..];
    println!("sub {}", b.to_str().unwrap());
    println!("enc {}", s.as_os_str().as_encoded_bytes().len());
    println!("done");
}
"#;

#[test]
fn run_mode_osstring_as_os_str_and_index() {
    run_ok(
        "osstring-as-os-str",
        OSSTRING_AS_OS_STR_SRC,
        "len 3\nsub a/b\nenc 3\ndone\n",
    );
}

const VEC_AS_PTR_WRAPPER_SRC: &str = r#"
struct Holder {
    v: Vec<u32>,
}
fn main() {
    let h = Holder { v: vec![10, 20, 30] };
    let p = h.v.as_ptr();
    unsafe {
        println!("{} {} {}", *p, *p.add(1), *p.add(2));
    }
    let v = vec![1u8, 2, 3];
    let q = v.as_ptr();
    unsafe {
        println!("first {}", *q);
    }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_as_ptr_through_wrapper() {
    run_ok(
        "vec-as-ptr-wrapper",
        VEC_AS_PTR_WRAPPER_SRC,
        "10 20 30\nfirst 1\ndone\n",
    );
}

const VEC_DEDUP_LEN_WRITE_SRC: &str = r#"
fn main() {
    let mut v = vec![1u8, 1, 2, 2, 3];
    v.dedup();
    println!("{:?}", v);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_dedup_len_slot_write() {
    run_ok("vec-dedup", VEC_DEDUP_LEN_WRITE_SRC, "[1, 2, 3]\ndone\n");
}

const VEC_TRUNCATE_LEN_WRITE_SRC: &str = r#"
fn main() {
    let mut v = vec![1u8, 1, 2, 2, 3];
    v.truncate(3);
    println!("{:?}", v);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_truncate_len_slot_write() {
    run_ok(
        "vec-truncate",
        VEC_TRUNCATE_LEN_WRITE_SRC,
        "[1, 1, 2]\ndone\n",
    );
}

const CSTR_WITH_NUL_SRC: &str = r#"
use std::ffi::CString;
fn main() {
    let c = CString::new("hello").unwrap();
    let cs = c.as_c_str();
    println!("len {}", cs.to_bytes().len());
    println!("with_nul {:?}", cs.to_bytes_with_nul());
    println!("done");
}
"#;

#[test]
fn run_mode_cstr_bytes_with_nul() {
    run_ok(
        "cstr-with-nul",
        CSTR_WITH_NUL_SRC,
        "len 5\nwith_nul [104, 101, 108, 108, 111, 0]\ndone\n",
    );
}

const IO_ERROR_CUSTOM_SRC: &str = r#"
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

fn main() {
    let e = std::io::Error::new(std::io::ErrorKind::InvalidInput, MyErr);
    println!("kind: {:?}", e.kind());
    println!("display: {e}");
    println!("debug: {e:?}");
    let e2 = std::io::Error::new(std::io::ErrorKind::Other, MyErr);
    println!("other debug: {e2:?}");
    let e3 = std::io::Error::new(std::io::ErrorKind::InvalidData, "literal payload");
    println!("str: {e3} | {e3:?}");
    let e4 = std::io::Error::new(std::io::ErrorKind::NotFound, String::from("owned payload"));
    println!("string: {e4} | {e4:?}");
    let e5 = std::io::Error::new(std::io::ErrorKind::InvalidInput, MyErr);
    match e5.get_ref() {
        Some(r) => println!("get_ref: {r:?}"),
        None => println!("get_ref none"),
    }
    println!("done");
}
"#;

#[test]
fn run_mode_io_error_custom_payload() {
    // `io::Error::new` with a custom payload: the packed
    // representation's word names the payload's allocation by
    // registered synthetic address, and every access goes
    // through the reverse registry - `kind()` decodes in std's
    // own body, `Display` forwards to the payload, `Debug` is
    // the `Custom { kind, error }` struct (with the kind named
    // as an `ErrorKind` value: `Other`'s variant index is not
    // its discriminant), and `get_ref` reaches the payload
    // through `decode_repr`'s closure.
    let dir = temp_dir("io-error-custom");
    let src = fixture(&dir, "io_error_custom.rs", IO_ERROR_CUSTOM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "kind: InvalidInput\n\
         display: my error display\n\
         debug: Custom { kind: InvalidInput, error: MyErr }\n\
         other debug: Custom { kind: Other, error: MyErr }\n\
         str: literal payload | Custom { kind: InvalidData, error: \"literal payload\" }\n\
         string: owned payload | Custom { kind: NotFound, error: \"owned payload\" }\n\
         get_ref: MyErr\n\
         done\n"
    );
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

const IO_ERROR_CUSTOM_DROP_SRC: &str = r#"
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

static DROPS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct Counted;
impl fmt::Display for Counted {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "counted error")
    }
}
impl std::error::Error for Counted {}
impl Drop for Counted {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

struct Holder {
    err: std::io::Error,
}

fn main() {
    {
        let e = std::io::Error::new(std::io::ErrorKind::Other, Counted);
        println!("in scope: {e}");
        println!("mid drops: {}", DROPS.load(Ordering::SeqCst));
    }
    println!("after scope drops: {}", DROPS.load(Ordering::SeqCst));
    {
        let h = Holder {
            err: std::io::Error::new(std::io::ErrorKind::Other, Counted),
        };
        println!("holder: {}", h.err);
    }
    println!("after holder drops: {}", DROPS.load(Ordering::SeqCst));
    let inner = std::io::Error::new(std::io::ErrorKind::Other, Counted);
    let outer = std::io::Error::new(std::io::ErrorKind::InvalidData, inner);
    drop(outer);
    println!("after nested drops: {}", DROPS.load(Ordering::SeqCst));
    let e5 = std::io::Error::new(std::io::ErrorKind::Other, Counted);
    std::mem::forget(e5);
    println!("after forget drops: {}", DROPS.load(Ordering::SeqCst));
}
"#;

#[test]
fn run_mode_io_error_custom_payload_drop() {
    // Dropping an `io::Error` in the `Custom` representation
    // runs the payload's destructor exactly once - at scope
    // end, through a struct field, and through a nested error
    // payload - and a forgotten error runs it never: the packed
    // representation's drop plan judges the word's tag at run
    // time and releases the registered allocation's trait
    // object.
    let dir = temp_dir("io-error-custom-drop");
    let src = fixture(&dir, "io_error_custom_drop.rs", IO_ERROR_CUSTOM_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "in scope: counted error\n\
         mid drops: 0\n\
         after scope drops: 1\n\
         holder: counted error\n\
         after holder drops: 2\n\
         after nested drops: 3\n\
         after forget drops: 3\n"
    );
}


const FILE_MODEL_SRC: &str = r#"
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};

fn main() {
    let p = "DATA_PATH";
    {
        let mut f = File::create(p).unwrap();
        f.write_all(b"hello").unwrap();
        writeln!(f, " {}", 42).unwrap();
        f.sync_all().unwrap();
    }
    let mut f = OpenOptions::new().read(true).write(true).open(p).unwrap();
    let mut s = String::new();
    f.read_to_string(&mut s).unwrap();
    println!("content={s:?}");
    f.seek(SeekFrom::Start(6)).unwrap();
    let mut buf = [0u8; 2];
    f.read_exact(&mut buf).unwrap();
    println!("at6={:?}", &buf);
    let mut g = f.try_clone().unwrap();
    println!("clone pos={}", g.stream_position().unwrap());
    println!("len={}", f.metadata().unwrap().len());
    // Open/drop in a loop: the handle's drop plan must close
    // the host file every iteration, or the loop exhausts the
    // process's descriptor budget long before it finishes.
    let mut total = 0usize;
    for _ in 0..1500 {
        let mut h = File::open(p).unwrap();
        let mut b = [0u8; 1];
        h.read_exact(&mut b).unwrap();
        total += b[0] as usize;
    }
    println!("loop total={total}");
}
"#;

#[test]
fn run_mode_file_handle_value_model() {
    let dir = temp_dir("file-model");
    let data = dir.join("data.txt");
    let src = FILE_MODEL_SRC.replace("DATA_PATH", &data.to_string_lossy());
    let src_path = fixture(&dir, "file_model.rs", &src);
    let out = run_driver(&src_path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "file model case must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "content=\"hello 42\\n\"\nat6=[52, 50]\nclone pos=8\nlen=9\nloop total=156000\n",
        "file model stdout mismatch; stderr={stderr}"
    );
}

// ----- exposed-provenance pointer round trips ----------------------
//
// A pointer exposed as an integer (a cast or a scalar transmute)
// records its target under the synthetic address it produced;
// materialising a pointer from that integer restores the target,
// so a rebuilt `Box` dereferences to its pointee instead of an
// address-only pointer that denotes no readable place.

const EXPOSED_TRANSMUTE_BOX_SRC: &str = r#"
fn main() {
    let b = Box::new(7u32);
    let n: usize = unsafe { std::mem::transmute(b) };
    let b2: Box<u32> = unsafe { std::mem::transmute(n) };
    println!("{}", *b2);
}
"#;

#[test]
fn run_mode_exposed_provenance_transmute_box_round_trip() {
    run_ok("exposed-transmute-box", EXPOSED_TRANSMUTE_BOX_SRC, "7\n");
}

const EXPOSED_CAST_FROM_RAW_SRC: &str = r#"
fn main() {
    let b = Box::new(7u32);
    let raw = Box::into_raw(b);
    let n = raw as usize;
    let raw2 = n as *mut u32;
    println!("via raw: {}", unsafe { *raw2 });
    let b2 = unsafe { Box::from_raw(raw2) };
    println!("via box: {}", *b2);
}
"#;

#[test]
fn run_mode_exposed_provenance_cast_from_raw_round_trip() {
    run_ok(
        "exposed-cast-from-raw",
        EXPOSED_CAST_FROM_RAW_SRC,
        "via raw: 7\nvia box: 7\n",
    );
}

const EXPOSED_DROP_ONCE_SRC: &str = r#"
struct Counted(u32);
static mut DROPS: u32 = 0;
impl Drop for Counted {
    fn drop(&mut self) { unsafe { DROPS += 1; } }
}
fn main() {
    let b = Box::new(Counted(11));
    let n: usize = unsafe { std::mem::transmute(b) };
    let mut b2: Box<Counted> = unsafe { std::mem::transmute(n) };
    println!("v={}", b2.0);
    b2.0 = 12;
    println!("v={}", (*b2).0);
    drop(b2);
    println!("drops={}", unsafe { DROPS });
}
"#;
#[test]
fn run_mode_box_enum_tag_byte_view_matches_native() {
    // A byte pointer at a `Box` allocation's storage reads and
    // writes the enum's tag span, as it does for a place: the
    // allocation's recorded declared type lets the byte view
    // and the `MaybeUninit` entry gate judge the position as
    // the enum's own storage (Lane OC, #292). Before the
    // record existed, the byte read answered the payload's
    // first byte and the byte write spliced into the payload.
    let dir = temp_dir("box-enum-tag-byte-view");
    let src = fixture(&dir, "box_enum_tag_byte_view.rs", BOX_ENUM_TAG_BYTE_VIEW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "box enum tag byte view must run with native values; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const BOX_ENUM_TAG_BYTE_VIEW_SRC: &str = r#"
enum E { A(u32), B(u16) }
fn main() {
    let b = Box::new(E::A(0x11223344));
    let tag = unsafe { *(&*b as *const E as *const u8) };
    assert_eq!(tag, 0);
    let mut b2 = Box::new(E::A(0x11223344));
    unsafe { *(&mut *b2 as *mut E as *mut u8) = 1; }
    assert!(matches!(*b2, E::B(_)));
    let raw = Box::into_raw(Box::new(E::A(0x11223344)));
    let tag = unsafe { *(raw as *const u8) };
    assert_eq!(tag, 0);
    unsafe { *(raw as *mut u8) = 1; }
    let b3 = unsafe { Box::from_raw(raw) };
    assert!(matches!(*b3, E::B(_)));
    println!("ok");
}
"#;

#[test]
fn run_mode_exposed_provenance_round_trip_drops_once() {
    run_ok(
        "exposed-drop-once",
        EXPOSED_DROP_ONCE_SRC,
        "v=11\nv=12\ndrops=1\n",
    );
}
#[test]
fn run_mode_box_byte_add_over_adt_elements_matches_native() {
    // A byte-denominated pointer step over a `Box<[E; N]>`
    // lands on the element it denotes natively: the
    // allocation's recorded array type denominates the
    // displacement by the element size (Lane OC, #292).
    // Before the record existed, the displaced read was an
    // explicit "displaced from a value boundary" error.
    let dir = temp_dir("box-byte-add-adt");
    let src = fixture(&dir, "box_byte_add_adt.rs", BOX_BYTE_ADD_ADT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "box byte_add over ADT elements must run with native values; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const BOX_BYTE_ADD_ADT_SRC: &str = r#"
#[derive(Clone, Copy, PartialEq, Debug)]
enum E { A(u32), B(u16) }
fn main() {
    let b = Box::new([E::A(11), E::B(22), E::A(33)]);
    let mut p = b.as_ptr();
    unsafe {
        assert_eq!(*p, E::A(11));
        p = p.byte_add(std::mem::size_of::<E>());
        assert_eq!(*p, E::B(22));
        p = p.byte_add(std::mem::size_of::<E>());
        assert_eq!(*p, E::A(33));
        p = p.byte_sub(2 * std::mem::size_of::<E>());
        assert_eq!(*p, E::A(11));
    }
    // The same denomination over a boxed slice (a recorded
    // slice type) and for byte-denominated writes.
    let v = vec![E::A(11), E::B(22), E::A(33)];
    let bs = v.into_boxed_slice();
    let mut q = bs.as_ptr();
    unsafe {
        q = q.byte_add(std::mem::size_of::<E>());
        assert_eq!(*q, E::B(22));
    }
    let mut w = Box::new([0u32; 4]);
    let wp = w.as_mut_ptr();
    unsafe {
        *wp.byte_add(4) = 77;
        *wp.byte_add(12) = 99;
    }
    assert_eq!(*w, [0, 77, 0, 99]);
    println!("ok");
}
"#;

#[test]
fn run_mode_box_array_byte_ptr_cast_back_matches_native() {
    // A byte pointer advanced into a `Box<[u32; 3]>` is
    // absorbed as an index denominated by the allocation's
    // recorded element type (Lane OC, #292); casting it back
    // to `*const u32` must not redenominate that index by the
    // byte pointer's units -- the cast does not move the
    // address. Before the cast-side fix, the index was
    // rewritten as a one- or two-byte offset and the reads
    // answered the elements' shifted bytes.
    let dir = temp_dir("box-array-byte-ptr-cast-back");
    let src = fixture(&dir, "box_array_byte_ptr_cast_back.rs", BOX_ARRAY_BYTE_PTR_CAST_BACK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "byte pointer cast back over a boxed array must run with native values; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "20\n30\n20\n30\n");
}

const BOX_ARRAY_BYTE_PTR_CAST_BACK_SRC: &str = r#"
fn main() {
    let b: Box<[u32; 3]> = Box::new([10, 20, 30]);
    let p = &*b as *const [u32; 3] as *const u8;
    unsafe {
        let r = p.add(4) as *const u32;
        println!("{}", *r);
        let r2 = p.add(8) as *const u32;
        println!("{}", *r2);
    }
    // The same denomination over a boxed slice (a recorded
    // slice type).
    let bs = vec![10u32, 20, 30].into_boxed_slice();
    let q = bs.as_ptr() as *const u8;
    unsafe {
        let r = q.add(4) as *const u32;
        println!("{}", *r);
        let r2 = q.add(8) as *const u32;
        println!("{}", *r2);
    }
}
"#;

#[test]
fn run_mode_box_transparent_wrapper_cast_read_matches_native() {
    // Reading a boxed transparent wrapper's payload through a
    // raw pointer cast to the payload type: the allocation's
    // recorded ADT type must not refuse the payload entry --
    // the stored form is a wrapper layer around the payload,
    // at the payload's own width and at byte width alike.
    // (The recorded type still judges byte positions of a
    // genuine enum's storage; see
    // `run_mode_box_enum_tag_byte_view_matches_native`.)
    let dir = temp_dir("box-transparent-wrapper-cast-read");
    let src = fixture(
        &dir,
        "box_transparent_wrapper_cast_read.rs",
        BOX_TRANSPARENT_WRAPPER_CAST_READ_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "transparent wrapper payload reads through casts must run with native values; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const BOX_TRANSPARENT_WRAPPER_CAST_READ_SRC: &str = r#"
#[repr(transparent)]
struct W<T>(T);

fn main() {
    let b: Box<W<u32>> = Box::new(W(42));
    let p = &*b as *const W<u32> as *const u32;
    let v = unsafe { *p };
    assert_eq!(v, 42);
    // The payload at byte width: a single-field struct's
    // storage is its field's, with no tag span in front.
    let b2: Box<W<u8>> = Box::new(W(0x5a));
    let bp = &*b2 as *const W<u8> as *const u8;
    let byte = unsafe { *bp };
    assert_eq!(byte, 0x5a);
    println!("ok");
}
"#;

const DEEP_RECURSION_SRC: &str = r#"
fn f(x: isize) -> isize {
    if x == 1 { return 1; } else { let y: isize = 1 + f(x - 1); return y; }
}

fn main() { println!("depth = {}", f(5000)); }
"#;

#[test]
fn run_mode_deep_recursion_completes() {
    run_ok("deep-recursion", DEEP_RECURSION_SRC, "depth = 5000\n");
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

const ERROR_SOURCE_CHAIN_SRC: &str = r#"
use std::error::Error;
use std::fmt;
#[derive(Debug)]
struct Inner;
impl fmt::Display for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "inner") }
}
impl Error for Inner {}
#[derive(Debug)]
struct Outer { src: Inner }
impl fmt::Display for Outer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "outer") }
}
impl Error for Outer {
    fn source(&self) -> Option<&(dyn Error + 'static)> { Some(&self.src) }
}
#[derive(Debug)]
struct Plain;
impl fmt::Display for Plain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "plain") }
}
impl Error for Plain {}
fn main() {
    let outer = Outer { src: Inner };
    let e: &dyn Error = &outer;
    let s = e.source().expect("outer has source");
    println!("source display: {}", s);
    println!("source-of-source is none: {}", s.source().is_none());
    let plain = Plain;
    let e2: &dyn Error = &plain;
    println!("plain source is none: {}", e2.source().is_none());
    let boxed: Box<dyn Error> = Box::new(Outer { src: Inner });
    println!("boxed source: {}", boxed.source().map(|s| s.to_string()).unwrap_or_else(|| "none".into()));
    let pie = "x".parse::<u32>().unwrap_err();
    let e3: &dyn Error = &pie;
    println!("foreign source is none: {}", e3.source().is_none());
}
"#;

#[test]
fn run_mode_error_source_chain_and_default() {
    // The vtable demand puts every `Error` method of a coerced
    // concrete type in the program — an overriding `source`, and
    // the trait's default body (which answers `None`) for types
    // that do not override it, local and foreign alike.
    run_ok(
        "error-source-chain",
        ERROR_SOURCE_CHAIN_SRC,
        "source display: inner\nsource-of-source is none: true\nplain source is none: true\nboxed source: inner\nforeign source is none: true\n",
    );
}

const ERROR_DOWNCAST_SRC: &str = r#"
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
static DROPS: AtomicUsize = AtomicUsize::new(0);
#[derive(Debug)]
struct Counted { code: u32 }
impl fmt::Display for Counted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "counted {}", self.code) }
}
impl Error for Counted {}
impl Drop for Counted { fn drop(&mut self) { DROPS.fetch_add(1, Ordering::SeqCst); } }
#[derive(Debug)]
struct Other;
impl fmt::Display for Other {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "other") }
}
impl Error for Other {}
fn main() {
    let boxed: Box<dyn Error> = Box::new(Counted { code: 42 });
    println!("is Counted: {}", boxed.is::<Counted>());
    println!("is Other: {}", boxed.is::<Other>());
    println!("ref code: {}", boxed.downcast_ref::<Counted>().map(|r| r.code).unwrap_or(0));
    let b = boxed.downcast::<Counted>().unwrap();
    println!("box debug: {:?}", b);
    println!("code: {}", b.code);
    let e: Box<dyn Error> = b;
    println!("back via ref: {}", e.downcast_ref::<Counted>().unwrap().code);
    drop(e);
    println!("drops after ok path: {}", DROPS.load(Ordering::SeqCst));
    let boxed2: Box<dyn Error> = Box::new(Counted { code: 2 });
    let back = boxed2.downcast::<Other>().unwrap_err();
    println!("err display: {}", back);
    drop(back);
    println!("drops after err path: {}", DROPS.load(Ordering::SeqCst));
    let send_box: Box<dyn Error + Send> = Box::new(Counted { code: 3 });
    println!("send code: {}", send_box.downcast::<Counted>().unwrap().code);
    println!("drops final: {}", DROPS.load(Ordering::SeqCst));
}
"#;

#[test]
fn run_mode_error_downcast_by_value() {
    // By-value `Box<dyn Error>::downcast` runs std's collected
    // body: the `type_id` vtable dispatch decides, the fat-to-thin
    // pointer cast extracts the data pointer, and the rebuilt
    // box owns the payload exactly once on both the Ok and the
    // Err path (one drop each, counted).
    run_ok(
        "error-downcast",
        ERROR_DOWNCAST_SRC,
        "is Counted: true\nis Other: false\nref code: 42\nbox debug: Counted { code: 42 }\ncode: 42\nback via ref: 42\ndrops after ok path: 1\nerr display: counted 2\ndrops after err path: 2\nsend code: 3\ndrops final: 3\n",
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

const PANIC_PAYLOAD_INTO_RAW_SRC: &str = r#"
use std::panic;
#[derive(Clone, Copy)]
struct Pair { a: i64, b: i64 }
fn main() {
    let result = panic::catch_unwind(|| {
        panic::panic_any(Pair { a: 11, b: 22 });
    });
    let err = result.unwrap_err();
    let raw = Box::into_raw(err) as *mut Pair;
    unsafe {
        println!("fields: {} {}", (*raw).a, (*raw).b);
        let p = *raw;
        println!("whole: {} {}", p.a, p.b);
    }
}
"#;

#[test]
fn run_mode_panic_payload_into_raw_cast_reads_whole_struct() {
    // A caught panic payload travels as a trait object whose
    // base is the one-field box aggregate holding the payload
    // inline. Casting `Box::into_raw` of it to the concrete
    // pointer type must produce a data pointer denoting the
    // payload as a whole — field reads and a whole-struct read
    // through it read the payload, not its first field.
    run_ok(
        "panic-payload-into-raw",
        PANIC_PAYLOAD_INTO_RAW_SRC,
        "fields: 11 22\nwhole: 11 22\n",
    );
}

const JOIN_PAYLOAD_INTO_RAW_SRC: &str = r#"
use std::thread;
struct Pair { a: i64, b: i64 }
fn main() {
    let handle = thread::spawn(|| -> () {
        std::panic::panic_any(Pair { a: 33, b: 44 });
    });
    let err = handle.join().unwrap_err();
    let raw = Box::into_raw(err) as *mut Pair;
    unsafe {
        println!("fields: {} {}", (*raw).a, (*raw).b);
    }
}
"#;

#[test]
fn run_mode_join_payload_into_raw_cast_reads_whole_struct() {
    // The thread-join error payload has the same inline-box
    // trait-object shape as the catch_unwind cleanup product:
    // the data pointer extracted for a cast to the concrete
    // pointer type denotes the whole payload struct.
    run_ok(
        "join-payload-into-raw",
        JOIN_PAYLOAD_INTO_RAW_SRC,
        "fields: 33 44\n",
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

const IPV4_FROM_CONVERSIONS_SRC: &str = r#"
use std::net::Ipv4Addr;
fn main() {
    println!("from array: {}", Ipv4Addr::from([1u8, 2, 3, 4]));
    println!("from u32: {}", Ipv4Addr::from(0x7f000001u32));
    println!("into addr: {}", <Ipv4Addr as From<[u8; 4]>>::from([10u8, 0, 0, 255]));
    let a = Ipv4Addr::new(127, 0, 0, 1);
    println!("to u32: {}", u32::from(a));
    println!("bits: {}", a.to_bits());
}
"#;

#[test]
fn run_mode_ipv4_addr_from_conversions() {
    // The `From` conversions between `Ipv4Addr`, `[u8; 4]`
    // and `u32` (big-endian) build / read the same octet
    // shape the formatter renders.
    run_ok(
        "ipv4-from",
        IPV4_FROM_CONVERSIONS_SRC,
        "from array: 1.2.3.4\nfrom u32: 127.0.0.1\ninto addr: 10.0.0.255\nto u32: 2130706433\nbits: 2130706433\n",
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

const CSTRING_DYN_MEASURE_SRC: &str = r#"
use std::ffi::CString;
trait Measure { fn measure(&self) -> usize; }
impl Measure for CString { fn measure(&self) -> usize { self.as_bytes().len() } }
struct S { name: CString, n: u32 }
fn main() {
    let c = CString::new("hello").unwrap();
    let d: &dyn Measure = &c;
    println!("ref_dyn={}", d.measure());
    let b: Box<dyn Measure> = Box::new(CString::new("world!").unwrap());
    println!("box_dyn={}", b.measure());
    let rc: std::rc::Rc<dyn Measure> = std::rc::Rc::new(CString::new("rc").unwrap());
    println!("rc_dyn={}", rc.measure());
    let s = S { name: CString::new("field").unwrap(), n: 1 };
    let d2: &dyn Measure = &s.name;
    println!("field_dyn={}", d2.measure());
}
"#;

#[test]
fn run_mode_cstring_dyn_method_reads_whole_value() {
    // Method dispatch through trait objects over a `CString`
    // (by reference, boxed, reference-counted, and as a struct
    // field) reads the whole value through the receiver.
    run_ok(
        "cstring-dyn-measure",
        CSTRING_DYN_MEASURE_SRC,
        "ref_dyn=5\nbox_dyn=6\nrc_dyn=2\nfield_dyn=5\n",
    );
}

const TRANSMUTE_TUPLE_FIELD_SRC: &str = r#"
#[repr(C, packed)]
struct S {
    a: (u8, u16, u8),
    b: u8,
}

fn main() {
    let bytes: [u8; 5] = [0x34, 0x12, 0x56, 0x78, 0x9a];
    let s: S = unsafe { std::mem::transmute(bytes) };
    let a = s.a;
    let b = s.b;
    println!("{} {} {} {}", a.0, a.1, a.2, b);
}
"#;

#[test]
fn run_mode_transmute_tuple_field_at_recorded_offsets() {
    // A tuple field of a packed struct transmuted from a
    // byte array: the tuple's storage order is the
    // compiler's choice (here the `u16` sits first, at
    // offset 0), so the field's fields are read at the
    // tuple's recorded layout offsets, not in declaration
    // order.
    run_ok(
        "transmute-tuple-field",
        TRANSMUTE_TUPLE_FIELD_SRC,
        "86 4660 120 154\n",
    );
}

const TRANSMUTE_TUPLE_LAYOUT_SRC: &str = r#"
fn main() {
    let b4: [u8; 4] = [0x34, 0x12, 0x56, 0x78];
    let t: (u8, u16, u8) = unsafe { std::mem::transmute(b4) };
    println!("{} {} {}", t.0, t.1, t.2);
    let back: [u8; 4] = unsafe { std::mem::transmute(t) };
    println!("{:?}", back);
    let v: u64 = 0x1122_3344_5566_7788;
    let p: (u8, u32) = unsafe { std::mem::transmute(v) };
    println!("{} {}", p.0, p.1);
    let v2: u64 = unsafe { std::mem::transmute(p) };
    println!("{:x}", v2);
    let nb: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let n: ((u8, u32), u8) = unsafe { std::mem::transmute(nb) };
    println!("{} {} {}", (n.0).0, (n.0).1, n.1);
}
"#;

#[test]
fn run_mode_transmute_tuple_at_recorded_offsets() {
    // Bare tuples transmuted to and from byte arrays and
    // integers at their recorded layout offsets: a
    // reordered tuple (`(u8, u16, u8)` stores its `u16`
    // first), a padded tuple (`(u8, u32)` spans eight
    // bytes, its pattern the full `u64` width in both
    // directions), and a nested tuple read field by field.
    run_ok(
        "transmute-tuple-layout",
        TRANSMUTE_TUPLE_LAYOUT_SRC,
        "86 4660 120\n[52, 18, 86, 120]\n136 287454020\n1122334400000088\n1 134678021 9\n",
    );
}

const TRANSMUTE_ENUM_TUPLE_PAYLOAD_SRC: &str = r#"
#[repr(u8)]
enum E {
    A((u8, u16, u8)),
    B(u32),
}

fn main() {
    let e = E::A((0x56, 0x1234, 0x78));
    let bytes: [u8; 8] = unsafe { std::mem::transmute(e) };
    let back: E = unsafe { std::mem::transmute(bytes) };
    match back {
        E::A(t) => println!("A {} {} {}", t.0, t.1, t.2),
        E::B(v) => println!("B {}", v),
    }
    let e2 = E::B(0xdeadbeef);
    let bytes2: [u8; 8] = unsafe { std::mem::transmute(e2) };
    let back2: E = unsafe { std::mem::transmute(bytes2) };
    match back2 {
        E::A(t) => println!("A {} {} {}", t.0, t.1, t.2),
        E::B(v) => println!("B {:x}", v),
    }
}
"#;

#[test]
fn run_mode_transmute_enum_tuple_payload() {
    // A direct-tag enum whose one variant carries a
    // tuple: the tuple payload is rebuilt at its recorded
    // layout offsets when the enum is reconstructed from
    // its bytes, for both variants.
    run_ok(
        "transmute-enum-tuple-payload",
        TRANSMUTE_ENUM_TUPLE_PAYLOAD_SRC,
        "A 86 4660 120\nB deadbeef\n",
    );
}

const TRANSMUTE_NICHE_WRAP_SRC: &str = r#"
#[repr(u8)]
#[derive(Clone, Copy)]
enum E { A = 253, B = 254 }
#[repr(C)]
#[derive(Clone, Copy)]
struct W { a: E, b: u8 }
#[derive(Clone, Copy)]
enum N { Payload(W), Empty1, Empty2 }
fn code(n: N) -> u32 {
    match n {
        N::Payload(w) => 1000 + (w.a as u32) + ((w.b as u32) << 8),
        N::Empty1 => 1,
        N::Empty2 => 2,
    }
}
fn main() {
    let a: N = unsafe { std::mem::transmute(0u16) };
    let b: N = unsafe { std::mem::transmute(255u16) };
    let c: N = unsafe { std::mem::transmute(253u16) };
    let d: N = unsafe { std::mem::transmute(0x5AFEu16) };
    println!("{} {} {} {}", code(a), code(b), code(c), code(d));
    let e2: u16 = unsafe { std::mem::transmute(N::Empty2) };
    let e1: u16 = unsafe { std::mem::transmute(N::Empty1) };
    let p: u16 = unsafe { std::mem::transmute(N::Payload(W { a: E::B, b: 0x5A })) };
    println!("{} {} {}", e2, e1, p);
}
"#;

#[test]
fn run_mode_transmute_niche_range_wraps_at_scalar_width() {
    // A niche range that starts at the niche scalar's
    // maximum (255 in a `u8` tag) wraps: the second niche
    // variant encodes as 0, in both directions. Decoding
    // measures the distance from the range's start at the
    // scalar's width, and encoding masks to that width
    // before shifting instead of spilling into the
    // adjacent payload byte.
    run_ok(
        "transmute-niche-wrap",
        TRANSMUTE_NICHE_WRAP_SRC,
        "2 1 1253 24294\n0 255 23294\n",
    );
}

const TRANSMUTE_NICHE_WRAP_SHIFTED_SRC: &str = r#"
#[repr(u8)]
#[derive(Clone, Copy)]
enum E { A = 253, B = 254 }
#[repr(C)]
#[derive(Clone, Copy)]
struct W { b: u8, a: E }
#[derive(Clone, Copy)]
enum N { Payload(W), Empty1, Empty2 }
fn code(n: N) -> u32 {
    match n {
        N::Payload(w) => 1000 + (w.a as u32) + (w.b as u32),
        N::Empty1 => 1,
        N::Empty2 => 2,
    }
}
fn main() {
    let a: N = unsafe { std::mem::transmute(0x0000u16) };
    let b: N = unsafe { std::mem::transmute(0xFF00u16) };
    let c: N = unsafe { std::mem::transmute(0xFE5Au16) };
    println!("{} {} {}", code(a), code(b), code(c));
    let e2: u16 = unsafe { std::mem::transmute(N::Empty2) };
    let p: u16 = unsafe { std::mem::transmute(N::Payload(W { b: 0x5A, a: E::B })) };
    println!("{} {}", e2, p);
}
"#;

#[test]
fn run_mode_transmute_niche_range_wraps_at_shifted_scalar() {
    // The same wrapping niche range with the niche scalar
    // in the payload's second byte: the wrapped value is
    // masked to the scalar's width before it is shifted
    // into its window, in both directions.
    run_ok(
        "transmute-niche-wrap-shifted",
        TRANSMUTE_NICHE_WRAP_SHIFTED_SRC,
        "2 1 1344\n0 65114\n",
    );
}



const EXTERN_STATIC_CONVERT_SRC: &str = r#"
pub mod inner {
    extern "C" {
        pub static GLOBAL1: u8;
        pub static mut GLOBAL2: u32;
        pub static GLOBAL3: [u8; 1];
    }
    pub fn call() {
        let _a = std::ptr::addr_of!(GLOBAL1);
        let _b = std::ptr::addr_of_mut!(GLOBAL2);
        let _c = std::ptr::addr_of!(GLOBAL3);
    }
}
fn main() { inner::call(); }
"#;

#[test]
fn convert_mode_extern_static_reference_converts() {
    // A foreign static (an `extern` block item) has no
    // initializer; asking the compiler to evaluate one ICEs
    // inside the typeck query and used to crash the whole
    // conversion (exit 101). The conversion must instead
    // complete, with the reference denoting the static's
    // program-level storage.
    let dir = temp_dir("extern-static-convert");
    let src = fixture(&dir, "case.rs", EXTERN_STATIC_CONVERT_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "conversion must complete; stderr={stderr}"
    );
    assert!(
        !stderr.contains("internal compiler error"),
        "no compiler ICE may be involved; stderr={stderr}"
    );
    let main_line = stdout
        .lines()
        .find(|l| l.contains("::main:"))
        .expect("summary lists main");
    assert!(
        main_line.contains("[converted]"),
        "main is converted: {main_line}"
    );
    let target = stdout
        .lines()
        .find(|l| l.starts_with("target-crate converted:"))
        .expect("summary reports the target crate counts");
    assert!(
        target.contains("skipped: 0"),
        "the target crate converts completely: {target}"
    );
}

const EXTERN_STATIC_INIT_SRC: &str = r#"
extern "C" { static EXT_STAT: u32; }
static R: &u32 = unsafe { &EXT_STAT };
fn main() {
    println!("nonnull={}", !std::ptr::addr_of!(*R).is_null());
    println!("same={}", std::ptr::addr_of!(*R) == std::ptr::addr_of!(EXT_STAT));
}
"#;

#[test]
fn convert_mode_extern_static_in_static_initializer_converts() {
    // The same reference reached through another static's
    // initializer allocation (the shape of a vtable or a
    // pointer table naming an extern static).
    let dir = temp_dir("extern-static-init");
    let src = fixture(&dir, "case.rs", EXTERN_STATIC_INIT_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "conversion must complete; stderr={stderr}"
    );
    let target = stdout
        .lines()
        .find(|l| l.starts_with("target-crate converted:"))
        .expect("summary reports the target crate counts");
    assert!(
        target.contains("skipped: 0"),
        "the target crate converts completely: {target}"
    );
}

#[test]
fn run_mode_extern_static_initializer_reference_resolves() {
    run_ok(
        "extern-static-init-run",
        EXTERN_STATIC_INIT_SRC,
        "nonnull=true\nsame=true\n",
    );
}

const EXTERN_STATIC_RUN_SRC: &str = r#"
extern "C" {
    static EXT_STAT: u32;
    static mut EXT_MUT: u32;
}
fn main() {
    let a = std::ptr::addr_of!(EXT_STAT);
    let b = std::ptr::addr_of!(EXT_STAT);
    println!("same={} nonnull={}", a == b, !a.is_null());
    unsafe {
        std::ptr::addr_of_mut!(EXT_MUT).write(42);
        println!("roundtrip={}", std::ptr::addr_of!(EXT_MUT).read());
    }
}
"#;

#[test]
fn run_mode_extern_static_address_identity_and_roundtrip() {
    // The storage identity is observable (same address, not
    // null) and the guest's own writes read back, even though
    // the static's initial contents are not the interpreter's
    // to invent.
    run_ok(
        "extern-static-run",
        EXTERN_STATIC_RUN_SRC,
        "same=true nonnull=true\nroundtrip=42\n",
    );
}

const EXTERN_STATIC_READ_SRC: &str = r#"
extern "C" { static EXT_STAT: u32; }
fn main() {
    let v = unsafe { std::ptr::addr_of!(EXT_STAT).read() };
    println!("value={}", v);
}
"#;

#[test]
fn run_mode_extern_static_unwritten_read_is_explicit_error() {
    // Reading storage the guest never wrote must meet the
    // explicit uninitialized-read error, not a fabricated
    // value and not a crash.
    let dir = temp_dir("extern-static-read");
    let src = fixture(&dir, "case.rs", EXTERN_STATIC_READ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "the read is an explicit unsupported outcome; stderr={stderr}"
    );
    assert!(
        stderr.contains("read of uninitialized place static"),
        "the error names the uninitialized static read; stderr={stderr}"
    );
    assert!(
        stderr.contains("EXT_STAT"),
        "the error names the static; stderr={stderr}"
    );
}

const EXTERN_STATIC_FIRST_ELEM_SRC: &str = r#"
extern "C" {
    static mut EXT_ARR: [u32; 2];
}
const FIRST: *mut u32 = unsafe { &raw mut EXT_ARR[0] };
fn main() {
    unsafe {
        std::ptr::addr_of_mut!(EXT_ARR).write([10, 20]);
        FIRST.write(33);
        println!("e0={} e1={}", EXT_ARR[0], EXT_ARR[1]);
    }
}
"#;

#[test]
fn run_mode_extern_static_first_element_write_preserves_rest() {
    // A constant pointer to the FIRST element of an aggregate
    // foreign static has byte offset zero, but it denotes that
    // element, not the whole static: the projection must say so,
    // or a write through the pointer replaces the complete
    // storage and the other element is lost.
    run_ok(
        "extern-static-first-elem",
        EXTERN_STATIC_FIRST_ELEM_SRC,
        "e0=33 e1=20\n",
    );
}

const EXTERN_STATIC_SLICE_FIELD_SRC: &str = r#"
#[repr(C)]
pub struct Pair {
    data: [u32; 2],
    sentinel: u32,
}
extern "C" {
    static mut EXT_PAIR: Pair;
}
const S: *mut [u32] = unsafe { &raw mut EXT_PAIR.data };
fn main() {
    unsafe {
        std::ptr::addr_of_mut!(EXT_PAIR).write(Pair { data: [10, 20], sentinel: 7 });
        (*S)[1] = 33;
        println!(
            "d0={} d1={} sentinel={}",
            EXT_PAIR.data[0], EXT_PAIR.data[1], EXT_PAIR.sentinel
        );
    }
}
"#;

#[test]
fn run_mode_extern_static_slice_field_write_hits_field() {
    // A constant slice pointer to an array FIELD of a foreign
    // static: the field sits at byte offset zero, but the slice
    // denotes the field's array, not the whole static. The
    // projection must name the field, or a write through the
    // slice indexes the static's whole value and lands on the
    // wrong element (here it silently corrupted `sentinel`).
    run_ok(
        "extern-static-slice-field",
        EXTERN_STATIC_SLICE_FIELD_SRC,
        "d0=10 d1=33 sentinel=7\n",
    );
}

const EXTERN_STATIC_SLICE_OFFSET_FIELD_SRC: &str = r#"
#[repr(C)]
pub struct Pair2 {
    sentinel: u32,
    data: [u32; 2],
}
extern "C" {
    static mut EXT_PAIR2: Pair2;
}
const S2: *mut [u32] = unsafe { &raw mut EXT_PAIR2.data };
fn main() {
    unsafe {
        std::ptr::addr_of_mut!(EXT_PAIR2).write(Pair2 { sentinel: 7, data: [10, 20] });
        (*S2)[0] = 33;
        println!(
            "d0={} d1={} sentinel={}",
            EXT_PAIR2.data[0], EXT_PAIR2.data[1], EXT_PAIR2.sentinel
        );
    }
}
"#;

#[test]
fn run_mode_extern_static_slice_offset_field_write_hits_field() {
    // The same shape with the array field at a nonzero byte
    // offset: the slice reference decodes with the field's
    // projection instead of staying undecodable.
    run_ok(
        "extern-static-slice-offset-field",
        EXTERN_STATIC_SLICE_OFFSET_FIELD_SRC,
        "d0=33 d1=20 sentinel=7\n",
    );
}

const EXTERN_STATIC_PREFIX_SLICE_SRC: &str = r#"
extern "C" {
    static mut EXT_ARR: [u32; 2];
}
const PREFIX: *mut [u32] =
    unsafe { std::ptr::slice_from_raw_parts_mut(std::ptr::addr_of_mut!(EXT_ARR).cast::<u32>(), 1) };
fn main() {
    unsafe {
        (*PREFIX)[0] = 10;
        std::ptr::addr_of_mut!(EXT_ARR).cast::<u32>().add(1).write(20);
        println!("e0={} e1={}", EXT_ARR[0], EXT_ARR[1]);
    }
}
"#;

#[test]
fn run_mode_extern_static_prefix_slice_keeps_full_storage() {
    // A constant slice covering only a PREFIX of a foreign
    // array static, evaluated before any whole-static access:
    // the slice's length bounds the view it denotes, never the
    // static's storage, so seeding the storage from this
    // reference must still give it the declared array's full
    // extent -- a later write to the second element is valid
    // and must not fail a bounds check against a one-element
    // stored value.
    run_ok(
        "extern-static-prefix-slice",
        EXTERN_STATIC_PREFIX_SLICE_SRC,
        "e0=10 e1=20\n",
    );
}

const EXTERN_STATIC_INTERIOR_FIRST_SRC: &str = r#"
extern "C" {
    static mut EXT_ARR: [u32; 2];
}
const FIRST: *mut u32 = unsafe { &raw mut EXT_ARR[0] };
fn main() {
    unsafe {
        FIRST.write(10);
        std::ptr::addr_of_mut!(EXT_ARR).cast::<u32>().add(1).write(20);
        println!("e0={} e1={}", EXT_ARR[0], EXT_ARR[1]);
    }
}
"#;

#[test]
fn run_mode_extern_static_interior_write_first_keeps_full_storage() {
    // A constant pointer to one element of a foreign array
    // static, written through before any whole-static access:
    // the reference's seed is the static's full initializer, so
    // the storage it seeds must have the declared array's full
    // extent -- a later write to the second element is valid
    // and must observe the first write.
    run_ok(
        "extern-static-interior-first",
        EXTERN_STATIC_INTERIOR_FIRST_SRC,
        "e0=10 e1=20\n",
    );
}

const EXTERN_STATIC_FIELD_SLICE_FIRST_SRC: &str = r#"
#[repr(C)]
pub struct Pair {
    data: [u32; 2],
    sentinel: u32,
}
extern "C" {
    static mut EXT_PAIR: Pair;
}
const S: *mut [u32] = unsafe { &raw mut EXT_PAIR.data };
fn main() {
    unsafe {
        (*S)[0] = 10;
        std::ptr::addr_of_mut!(EXT_PAIR.data)
            .cast::<u32>()
            .add(1)
            .write(20);
        std::ptr::addr_of_mut!(EXT_PAIR.sentinel).write(7);
        println!(
            "d0={} d1={} sentinel={}",
            EXT_PAIR.data[0], EXT_PAIR.data[1], EXT_PAIR.sentinel
        );
    }
}
"#;

#[test]
fn run_mode_extern_static_field_slice_first_keeps_full_storage() {
    // A constant slice of an array FIELD of a foreign static,
    // evaluated before any whole-static access: the projection
    // is nonempty, but the seed is still the static's full
    // initializer, so the storage must be seeded with the
    // declared struct's skeleton. A bare seed left the storage
    // shapeless: the first write collapsed the field slot to
    // the written scalar, and the later write to the field's
    // second element failed with a type mismatch against it.
    run_ok(
        "extern-static-field-slice-first",
        EXTERN_STATIC_FIELD_SLICE_FIRST_SRC,
        "d0=10 d1=20 sentinel=7\n",
    );
}

const EXTERN_STATIC_FIELD_PREFIX_FIRST_SRC: &str = r#"
#[repr(C)]
pub struct Pair {
    data: [u32; 2],
    sentinel: u32,
}
extern "C" {
    static mut EXT_PAIR: Pair;
}
const PFX: *mut [u32] = unsafe {
    std::ptr::slice_from_raw_parts_mut(
        std::ptr::addr_of_mut!(EXT_PAIR.data).cast::<u32>(),
        1,
    )
};
fn main() {
    unsafe {
        (*PFX)[0] = 10;
        std::ptr::addr_of_mut!(EXT_PAIR.data)
            .cast::<u32>()
            .add(1)
            .write(20);
        std::ptr::addr_of_mut!(EXT_PAIR.sentinel).write(7);
        println!(
            "d0={} d1={} sentinel={}",
            EXT_PAIR.data[0], EXT_PAIR.data[1], EXT_PAIR.sentinel
        );
    }
}
"#;

#[test]
fn run_mode_extern_static_field_prefix_slice_first_keeps_full_storage() {
    // A constant slice covering only a PREFIX of an array
    // field of a foreign static, evaluated first: the slice's
    // length bounds only the view it denotes, and the field
    // projection does not shrink the seed either -- the
    // storage is the declared struct's full extent, so the
    // later writes to the field's second element and to the
    // sibling field are valid.
    run_ok(
        "extern-static-field-prefix-first",
        EXTERN_STATIC_FIELD_PREFIX_FIRST_SRC,
        "d0=10 d1=20 sentinel=7\n",
    );
}

// ----- Required constants of collected instances (mono collector) -----

const REQUIRED_CONST_DEAD_FN_SRC: &str = r#"
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
#[inline(never)]
fn not_called<T>() {
    if false {
        let _ = Fail::<T>::C;
    }
}
#[inline(never)]
fn called<T>() {
    if false {
        not_called::<T>();
    }
}
fn main() {
    called::<i32>();
}
"#;

const REQUIRED_CONST_FNPTR_IN_CONST_SRC: &str = r#"
struct Late<T>(T);
impl<T> Late<T> {
    const FAIL: () = panic!();
    const FNPTR: fn() = || Self::FAIL;
}
#[inline(never)]
fn not_called<T>() {
    if false {
        let _ = Late::<T>::FNPTR;
    }
}
#[inline(never)]
fn called<T>() {
    if false {
        not_called::<T>();
    }
}
fn main() {
    called::<i32>();
}
"#;

const REQUIRED_CONST_UNMENTIONED_SRC: &str = r#"
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
#[inline(never)]
fn not_called<T>() {
    if false {
        let _ = Fail::<T>::C;
    }
}
fn main() {}
"#;

#[test]
fn convert_mode_required_const_in_dead_mention_is_a_compile_failure() {
    // rustc's mono collector evaluates the required constants of
    // every instance it collects, following mentions recorded
    // before optimization: `called::<i32>` mentions
    // `not_called::<i32>` from dead code, and that body's required
    // constant `Fail::<i32>::C` panics in evaluation (E0080), so
    // the crate fails to compile even though the bad constant is
    // never demanded by the converted (post-optimization) bodies.
    let dir = temp_dir("required-const-dead-fn");
    let src = fixture(&dir, "required_const_dead_fn.rs", REQUIRED_CONST_DEAD_FN_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a failing required constant behind a dead-code mention must fail \
         compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("E0080"),
        "the failure must surface the evaluation error; stderr={stderr}"
    );
}

#[test]
fn convert_mode_required_const_behind_fnptr_const_is_a_compile_failure() {
    // The failing constant hides one more hop away: evaluating
    // `Late::<i32>::FNPTR` succeeds and yields a closure pointer,
    // and the collector collects that closure instance, whose own
    // required constant `Late::<i32>::FAIL` fails to evaluate.
    let dir = temp_dir("required-const-fnptr");
    let src = fixture(
        &dir,
        "required_const_fnptr.rs",
        REQUIRED_CONST_FNPTR_IN_CONST_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a failing required constant behind a function-pointer constant \
         must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("E0080"),
        "the failure must surface the evaluation error; stderr={stderr}"
    );
}

#[test]
fn convert_mode_required_const_never_mentioned_converts() {
    // The same failing constant in a generic function that no
    // collected instance mentions: rustc never collects it, never
    // evaluates the constant, and the crate compiles. The check
    // must not fire either — it only walks from the instances
    // collection actually resolved.
    let dir = temp_dir("required-const-unmentioned");
    let src = fixture(
        &dir,
        "required_const_unmentioned.rs",
        REQUIRED_CONST_UNMENTIONED_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing required constant in a never-mentioned generic \
         function must convert; stderr={stderr}"
    );
}

const REQUIRED_CONST_BEHIND_UNREACHABLE_SRC: &str = r#"
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
fn leaf<T>() {
    if false {
        let _ = Fail::<T>::C;
    }
}
fn bad<T>() {
    if false {
        leaf::<T>();
    }
}
fn unused() {
    bad::<i32>();
}
fn main() {}
"#;

#[test]
fn convert_mode_required_const_behind_unreachable_caller_converts() {
    // `bad::<i32>` is demanded by conversion only because the
    // unreachable private function `unused` calls it; the mono
    // collector never roots it, so it never reaches `leaf::<i32>`
    // and never evaluates the panicking constant. rustc compiles
    // the crate, and the check must not fire either: its roots are
    // the collector's roots, not every instance conversion
    // demanded. (The constant sits one mention hop behind the
    // demanded instance because conversion itself materializes the
    // constants of the bodies it converts; what is exercised here
    // is that the check does not start from `bad` at all.)
    let dir = temp_dir("required-const-unreachable");
    let src = fixture(
        &dir,
        "required_const_unreachable.rs",
        REQUIRED_CONST_BEHIND_UNREACHABLE_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing required constant reachable only from an \
         unreachable function must convert; stderr={stderr}"
    );
}

const REQUIRED_CONST_BEHIND_FN_ITEM_VALUE_SRC: &str = r#"
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
fn bad<T>() {
    if false {
        let _ = Fail::<T>::C;
    }
}
#[inline(never)]
fn hold<U>(x: U) {
    std::hint::black_box(x);
}
#[inline(never)]
fn outer<T>() {
    hold(bad::<T>);
}
fn main() {
    if false {
        outer::<i32>();
    }
}
"#;

#[test]
fn convert_mode_required_const_behind_fn_item_value_converts() {
    // Passing the function item `bad::<T>` as a value to a generic
    // function that never calls it does not reify a function
    // pointer and does not codegen `bad`'s body: a bare `FnDef`
    // constant operand is a zero-sized function-item value, not a
    // collected item. The walk reaches `outer::<i32>` through its
    // mention from `main`, but must not demand `bad::<i32>` from
    // the operand, so the panicking constant is never evaluated
    // and the crate converts, as it does under rustc.
    let dir = temp_dir("required-const-fn-item-value");
    let src = fixture(
        &dir,
        "required_const_fn_item_value.rs",
        REQUIRED_CONST_BEHIND_FN_ITEM_VALUE_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing required constant behind a bare function-item \
         value must convert; stderr={stderr}"
    );
}
const REQUIRED_CONST_BEHIND_UNCALLED_CLOSURE_SRC: &str = r#"
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
#[inline(never)]
fn hold<U>(x: U) {
    std::hint::black_box(x);
}
#[inline(never)]
fn outer<T>() {
    hold(|| {
        if false {
            let _ = Fail::<T>::C;
        }
    });
}
fn main() {
    if false {
        outer::<i32>();
    }
}
"#;

#[test]
fn convert_mode_required_const_behind_uncalled_closure_converts() {
    // Passing a captureless closure as a value to a generic
    // function that never calls it does not codegen the closure
    // body: a bare closure constant operand is a zero-sized
    // value, not a collected item. The walk reaches
    // `outer::<i32>` through its mention from `main`, but must
    // not demand the closure instance from the operand, so the
    // panicking constant in the closure body is never evaluated
    // and the crate converts, as it does under rustc.
    let dir = temp_dir("required-const-uncalled-closure");
    let src = fixture(
        &dir,
        "required_const_uncalled_closure.rs",
        REQUIRED_CONST_BEHIND_UNCALLED_CLOSURE_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing required constant behind an uncalled closure \
         value must convert; stderr={stderr}"
    );
}

const REQUIRED_CONST_IN_CALLED_CLOSURE_SRC: &str = r#"
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
fn main() {
    let f = || {
        let _ = Fail::<i32>::C;
    };
    f();
}
"#;

#[test]
fn convert_mode_required_const_in_called_closure_is_a_compile_failure() {
    // A closure that is actually called IS collected: the call
    // names the `FnOnce` trait method, whose resolution reaches
    // the closure instance, and the collector evaluates the
    // closure body's required constants. `Fail::<i32>::C`
    // panics in evaluation (E0080), so the crate fails to
    // compile, exactly as under rustc.
    let dir = temp_dir("required-const-called-closure");
    let src = fixture(
        &dir,
        "required_const_called_closure.rs",
        REQUIRED_CONST_IN_CALLED_CLOSURE_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a failing required constant in a called closure must \
         fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("E0080"),
        "the failure must surface the evaluation error; stderr={stderr}"
    );
}

const REQUIRED_CONST_IN_COERCED_CLOSURE_SRC: &str = r#"
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
fn main() {
    let f: fn() = || {
        let _ = Fail::<i32>::C;
    };
    std::hint::black_box(f);
}
"#;

#[test]
fn convert_mode_required_const_in_coerced_closure_is_a_compile_failure() {
    // A closure coerced to a function pointer IS collected:
    // the coercion is a mentioned item that resolves the
    // closure instance, and the collector evaluates its
    // required constants. `Fail::<i32>::C` panics in evaluation
    // (E0080), so the crate fails to compile, exactly as under
    // rustc.
    let dir = temp_dir("required-const-coerced-closure");
    let src = fixture(
        &dir,
        "required_const_coerced_closure.rs",
        REQUIRED_CONST_IN_COERCED_CLOSURE_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a failing required constant in a closure coerced to a \
         function pointer must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("E0080"),
        "the failure must surface the evaluation error; stderr={stderr}"
    );
}

const REQUIRED_CONST_UNSIZE_DOUBLE_REF_SRC: &str = r#"
trait Trait {
    fn method(&self) -> usize;
}
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
struct W<T>(T);
impl<T> Trait for W<T> {
    fn method(&self) -> usize {
        if false {
            let _ = Fail::<T>::C;
        }
        1
    }
}
impl<T> Trait for &W<T> {
    fn method(&self) -> usize {
        42
    }
}
fn main() {
    let w = W(7u8);
    let d: &dyn Trait = &&w;
    println!("{}", d.method());
}
"#;

#[test]
fn convert_mode_required_const_behind_sibling_unsize_impl_converts() {
    // The unsizing cast `&&W<u8>` -> `&dyn Trait` forms the
    // vtable for `&W<u8>`, whose method carries no required
    // constants. The sibling implementation for `W<u8>` is never
    // collected, so its method's failing required constant is
    // never evaluated and the crate compiles, exactly as under
    // rustc. (A tail search that peels the cast's source and
    // target independently strips both source references, forms
    // the vtable for `W<u8>` instead, and wrongly rejects this.)
    let dir = temp_dir("required-const-unsize-double-ref");
    let src = fixture(
        &dir,
        "required_const_unsize_double_ref.rs",
        REQUIRED_CONST_UNSIZE_DOUBLE_REF_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing required constant in the sibling impl of an \
         unsizing cast must convert; stderr={stderr}"
    );
}

const REQUIRED_CONST_UNSIZE_DEAD_GENERIC_SRC: &str = r#"
trait Trait {
    fn method(&self) -> usize;
}
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
struct W<T>(T);
impl<T> Trait for W<T> {
    fn method(&self) -> usize {
        if false {
            let _ = Fail::<T>::C;
        }
        1
    }
}
impl<T> Trait for &W<T> {
    fn method(&self) -> usize {
        42
    }
}
#[inline(never)]
fn cast_dead<T>(w: &W<T>) {
    let d: &dyn Trait = &w;
    println!("{}", d.method());
}
#[inline(never)]
fn caller<T>(w: &W<T>) {
    if false {
        cast_dead::<T>(w);
    }
}
fn main() {
    let w = W(7u8);
    caller::<u8>(&w);
    println!("done");
}
"#;

#[test]
fn convert_mode_required_const_behind_sibling_unsize_impl_in_dead_generic_converts() {
    // The same cast one mention further away: the generic
    // function holding it is mentioned only from dead code of a
    // called function, which the walk still follows. The cast
    // forms the vtable for `&W<u8>`, so the failing required
    // constant in `W<u8>`'s sibling implementation is never
    // evaluated and the crate compiles.
    let dir = temp_dir("required-const-unsize-dead-generic");
    let src = fixture(
        &dir,
        "required_const_unsize_dead_generic.rs",
        REQUIRED_CONST_UNSIZE_DEAD_GENERIC_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing required constant in the sibling impl of an \
         unsizing cast inside a dead-mentioned generic function \
         must convert; stderr={stderr}"
    );
}

const REQUIRED_CONST_UNSIZE_VTABLE_SRC: &str = r#"
trait Trait {
    fn method(&self) -> usize;
}
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
struct W<T>(T);
impl<T> Trait for W<T> {
    fn method(&self) -> usize {
        if false {
            let _ = Fail::<T>::C;
        }
        1
    }
}
impl<T> Trait for &W<T> {
    fn method(&self) -> usize {
        42
    }
}
fn main() {
    let w = W(7u8);
    let d: &dyn Trait = &w;
    println!("{}", d.method());
}
"#;

#[test]
fn convert_mode_required_const_behind_unsize_vtable_is_a_compile_failure() {
    // The direct shape, as a control: `&W<u8>` unsizing to
    // `&dyn Trait` forms the vtable for `W<u8>` itself, so the
    // collector collects its method and evaluates its failing
    // required constant (E0080); the crate fails to compile,
    // exactly as under rustc.
    let dir = temp_dir("required-const-unsize-vtable");
    let src = fixture(
        &dir,
        "required_const_unsize_vtable.rs",
        REQUIRED_CONST_UNSIZE_VTABLE_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a failing required constant behind an unsizing cast's \
         own vtable must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("E0080"),
        "the failure must surface the evaluation error; stderr={stderr}"
    );
}

// ----- rustc_private type defaults (rustdoc execution path) -----------------

// The rustdoc entry program's session configuration asks for
// `ErrorOutputType::default()` before anything else observable
// happens; the value is the `#[default]` variant with its field
// defaults (`HumanReadable` with a non-short, non-unicode kind
// and `ColorConfig::Auto`), served by the interpreter's modelled
// `Default::default`. The expected line is the native 1.100.0
// output of the same program.
const ERROR_OUTPUT_TYPE_DEFAULT_SRC: &str = r#"
#![feature(rustc_private)]
extern crate rustc_errors;
extern crate rustc_session;

use rustc_errors::emitter::ColorConfig;
use rustc_session::config::ErrorOutputType;

fn main() {
    let v = ErrorOutputType::default();
    match v {
        ErrorOutputType::HumanReadable { kind, color_config } => {
            println!(
                "human short={} unicode={} auto={}",
                kind.short,
                kind.unicode,
                matches!(color_config, ColorConfig::Auto)
            );
        }
        ErrorOutputType::Json { .. } => println!("json"),
    }
}
"#;

#[test]
fn run_mode_error_output_type_default() {
    let dir = temp_dir("error-output-type-default");
    let src = fixture(&dir, "case.rs", ERROR_OUTPUT_TYPE_DEFAULT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "human short=false unicode=false auto=true\n",
        "ErrorOutputType::default() must be the HumanReadable default; stderr={stderr}"
    );
}

// The rustdoc entry program's diagnostics context setup asks
// for `DiagCtxtFlags::default()` (the fourth stop on the
// rustdoc execution path); the value is the derived `Default`
// of `rustc_errors::DiagCtxtFlags` -- every field at its own
// default -- served by the interpreter's modelled
// `Default::default`. The guest reads every field of the
// default, then mutates three fields and reads them back; the
// expected lines are the native 1.100.0 output of the same
// program.
const DIAG_CTXT_FLAGS_DEFAULT_SRC: &str = r#"
#![feature(rustc_private)]
extern crate rustc_errors;

use std::num::NonZero;
use rustc_errors::DiagCtxtFlags;

fn show(f: DiagCtxtFlags) {
    let err = match f.treat_err_as_bug {
        None => "none",
        Some(n) => {
            if n.get() == 3 {
                "some3"
            } else {
                "some?"
            }
        }
    };
    println!(
        "warn={} err={} eager={} macro={} dedup={} track={}",
        f.can_emit_warnings,
        err,
        f.eagerly_emit_delayed_bugs,
        f.macro_backtrace,
        f.deduplicate_diagnostics,
        f.track_diagnostics
    );
}

fn main() {
    let mut f = DiagCtxtFlags::default();
    show(f);
    f.can_emit_warnings = true;
    f.treat_err_as_bug = Some(NonZero::new(3usize).unwrap());
    f.track_diagnostics = true;
    show(f);
}
"#;

#[test]
fn run_mode_diag_ctxt_flags_default() {
    let dir = temp_dir("diag-ctxt-flags-default");
    let src = fixture(&dir, "case.rs", DIAG_CTXT_FLAGS_DEFAULT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "warn=false err=none eager=false macro=false dedup=false track=false\n\
         warn=true err=some3 eager=false macro=false dedup=false track=true\n",
        "DiagCtxtFlags::default() must be the all-defaults flags value; stderr={stderr}"
    );
}

// ----- libc isatty (rustdoc execution path) ---------------------------------

// Session setup in compiler programs asks whether the standard
// streams are terminals (colour configuration); the answer comes
// from the body-less libc `isatty`, served by the interpreter's
// foreign-helper table: descriptors 0/1/2 are the host process's
// own streams, file-model descriptors are answered from the host
// file table, and unissued descriptors are not terminals. The
// expected line is the native output of the same program with
// both standard streams captured through pipes.
const LIBC_ISATTY_SRC: &str = r#"
use std::ffi::c_int;
use std::os::unix::io::AsRawFd;

extern "C" {
    fn isatty(fd: c_int) -> c_int;
}

fn main() {
    let path = std::env::temp_dir().join("sparagmos-isatty-probe.txt");
    let f = std::fs::File::create(&path).unwrap();
    let (o, e, d, x) = unsafe { (isatty(1), isatty(2), isatty(f.as_raw_fd()), isatty(9999)) };
    drop(f);
    let _ = std::fs::remove_file(&path);
    println!("stdout={o} stderr={e} file={d} bogus={x}");
}
"#;

#[test]
fn run_mode_libc_isatty() {
    let dir = temp_dir("libc-isatty");
    let src = fixture(&dir, "libc_isatty.rs", LIBC_ISATTY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "stdout=0 stderr=0 file=0 bogus=0\n",
        "piped streams, a regular file and an unissued descriptor are not terminals; stderr={stderr}"
    );
}

const EAGER_CONST_UNREACHABLE_FN_SRC: &str = r#"
struct Fail<T>(std::marker::PhantomData<T>);
impl<T> Fail<T> {
    const C: u32 = panic!("fail const evaluated");
}
fn bad() {
    if false {
        let _ = Fail::<i32>::C;
    }
}
fn main() {
    println!("ok");
}
"#;

#[test]
fn convert_mode_eager_const_in_unreachable_fn_body_converts() {
    // Issue #417: `bad` is a non-generic local function, so the
    // item loop materializes its body even though nothing calls
    // it, and materializing a body makes the Stable MIR builder
    // evaluate every constant operand in it. The failing
    // constant's evaluation report used to stay in the session's
    // error count and fail the whole conversion, even though
    // rustc never collects `bad`, never evaluates the constant,
    // and compiles the crate. A body materialization's
    // evaluation diagnostics are not a verdict; the
    // required-constant check over the collector's instances is.
    let dir = temp_dir("eager-const-unreachable-fn");
    let src = fixture(
        &dir,
        "eager_const_unreachable_fn.rs",
        EAGER_CONST_UNREACHABLE_FN_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing constant in an unreachable function's body must \
         not fail conversion; stderr={stderr}"
    );
}

#[test]
fn run_mode_eager_const_in_unreachable_fn_body_executes() {
    // The same program through run mode: conversion must not be
    // poisoned either, and the program runs and prints.
    let dir = temp_dir("eager-const-unreachable-fn-run");
    let src = fixture(
        &dir,
        "eager_const_unreachable_fn_run.rs",
        EAGER_CONST_UNREACHABLE_FN_SRC,
    );
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "run mode must execute the program; stderr={stderr}"
    );
    assert!(
        stdout.contains("ok"),
        "the program's output must be produced; stdout={stdout}"
    );
}

const EAGER_CONST_DEAD_GENERIC_CALL_SRC: &str = r#"
struct Fail<T>(std::marker::PhantomData<T>);
impl<T> Fail<T> {
    const C: u32 = panic!("fail const evaluated");
}
fn bad<T>() {
    if false {
        let _ = Fail::<T>::C;
    }
}
fn unused() {
    bad::<i32>();
}
fn main() {
    println!("ok");
}
"#;

#[test]
fn convert_mode_eager_const_behind_dead_generic_call_converts() {
    // Issue #417's isolated shape: conversion demands
    // `bad::<i32>` because the unreachable `unused` calls it,
    // and materializing that body evaluates `Fail::<i32>::C`.
    // The mono collector roots nothing at `unused`, so rustc
    // compiles the crate; conversion must too.
    let dir = temp_dir("eager-const-dead-generic-call");
    let src = fixture(
        &dir,
        "eager_const_dead_generic_call.rs",
        EAGER_CONST_DEAD_GENERIC_CALL_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing constant in a body demanded only from dead code \
         must not fail conversion; stderr={stderr}"
    );
}

const EAGER_CONST_DEMANDED_SRC: &str = r#"
struct Fail<T>(std::marker::PhantomData<T>);
impl<T> Fail<T> {
    const C: u32 = panic!("fail const evaluated");
}
fn bad<T>() {
    if false {
        let _ = Fail::<T>::C;
    }
}
fn main() {
    bad::<i32>();
}
"#;

#[test]
fn convert_mode_eager_const_in_demanded_body_still_fails() {
    // Control: the same constant in a body the collector does
    // collect — `main` calls `bad::<i32>` directly — is a
    // required constant of a collected instance, so its failure
    // is a compilation error. The verdict comes from the
    // required-constant check; discounting the materialization's
    // own report must not turn this into a conversion.
    let dir = temp_dir("eager-const-demanded");
    let src = fixture(
        &dir,
        "eager_const_demanded.rs",
        EAGER_CONST_DEMANDED_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a failing required constant of a collected instance must \
         fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("E0080"),
        "the failure must surface the evaluation error; stderr={stderr}"
    );
}

const MATERIALIZATION_LINT_ENTRY_SRC: &str = r#"
fn main() {
    let array = [std::env::args().len()];
    array[1];
}
"#;

#[test]
fn convert_mode_materialization_lint_in_entry_body_still_fails() {
    // Materializing `main`'s body runs the MIR lint passes over
    // it, and the known-panics lint reports the out-of-bounds
    // index at deny level. That diagnostic is a compilation
    // error under rustc (it lints the bodies it materializes),
    // so the discount for speculative materializations must not
    // swallow it: the verdict records it as a lint finding.
    let dir = temp_dir("materialization-lint-entry");
    let src = fixture(
        &dir,
        "materialization_lint_entry.rs",
        MATERIALIZATION_LINT_ENTRY_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a lint error in the entry body's materialization must \
         fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed"),
        "the failure must be the driver's compilation failure; \
         stderr={stderr}"
    );
}

const MATERIALIZATION_LINT_DEAD_FN_SRC: &str = r#"
fn bad() {
    let array = [std::env::args().len()];
    array[1];
}
fn main() {
    println!("ok");
}
"#;

#[test]
fn convert_mode_materialization_lint_in_dead_body_still_fails() {
    // The same lint in a function nothing calls: unlike a
    // failing constant (which rustc evaluates only for the
    // bodies it collects), the lint is reported for every local
    // body rustc builds, so this is a compilation error too —
    // the lint registry is not scoped to collected instances.
    let dir = temp_dir("materialization-lint-dead-fn");
    let src = fixture(
        &dir,
        "materialization_lint_dead_fn.rs",
        MATERIALIZATION_LINT_DEAD_FN_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a lint error in a dead function's materialization must \
         fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed"),
        "the failure must be the driver's compilation failure; \
         stderr={stderr}"
    );
}

const MATERIALIZATION_VALIDITY_ENTRY_SRC: &str = r#"
use std::cell::Cell;
use std::mem::ManuallyDrop;

#[repr(C)]
struct S {
    x: u32,
    y: E,
}

#[repr(u32)]
enum E {
    A,
    B(U)
}

union U {
    cell: ManuallyDrop<Cell<u32>>,
}

const C: S = {
    let mut s = S { x: 0, y: E::A };
    let p = &mut s.x as *mut u32;
    unsafe { *p.add(1) = 1 };
    s
};

fn main() {
    let _: &'static _ = &C;
}
"#;

#[test]
fn convert_mode_materialization_validity_in_entry_body_still_fails() {
    // Materializing `main`'s body evaluates its constant
    // operands, and validating `&C`'s value fails (interior
    // mutability in read-only memory, E0080). The constant is
    // an operand of a body the collector collects, so the
    // evaluation-class record becomes a finding even though
    // evaluating the constant itself succeeds — the required-
    // constant check alone does not validate.
    let dir = temp_dir("materialization-validity-entry");
    let src = fixture(
        &dir,
        "materialization_validity_entry.rs",
        MATERIALIZATION_VALIDITY_ENTRY_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a validity error in the entry body's materialization \
         must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("E0080"),
        "the failure must surface the evaluation error; stderr={stderr}"
    );
}
// ----- rustc_private type defaults (rustdoc execution path) -----------------

// The rustdoc entry program's emitter setup asks for
// `anstream::adapter::strip::StripBytes::default()` (the third
// stop on the rustdoc execution path). The distributed sysroot
// carries anstream as metadata only, so no `Default` body is
// ever collected and the interpreter's modelled
// `Default::default` serves the value: the ANSI-strip state
// machine in its initial state (`state: State::Ground`,
// wrapping a fresh `utf8parse::Parser { point: 0, state:
// Ground }`). The guest prints the value through the derived
// `Debug`, which the interpreter renders from the recorded
// shape; the expected line is the native 1.100.0 output of the
// same program. (Against the MIR-carrying artifact-tree build
// of anstream 0.6.21, the collected strip machinery also runs
// over this modelled value and renders it identically.)
const STRIP_BYTES_DEFAULT_SRC: &str = r#"
#![feature(rustc_private)]

extern crate anstream;

use anstream::adapter::StripBytes;

fn main() {
    let s = StripBytes::default();
    println!("{:?}", s);
}
"#;

#[test]
fn run_mode_strip_bytes_default() {
    let dir = temp_dir("strip-bytes-default");
    let src = fixture(&dir, "case.rs", STRIP_BYTES_DEFAULT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "StripBytes { state: Ground, utf8parser: Utf8Parser { utf8_parser: Parser { point: 0, state: Ground } } }\n",
        "StripBytes::default() must be the initial strip state; stderr={stderr}"
    );
}
fn debug_driver(dir: &Path, src: &Path, script: &str) -> Output {
    let script_path = dir.join("session.script");
    std::fs::write(&script_path, script).unwrap();
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("debug")
        .arg(src)
        .arg("--edition")
        .arg("2021")
        .arg("--debug-script")
        .arg(&script_path)
        .arg("--debug-dialect")
        .arg("gdb")
        .output()
        .expect("failed to run sparagmos-driver")
}
const DEBUG_UNION_SRC: &str = r#"union U { bits: u32, value: f32 }
fn main() {
    let u = U { bits: 0x3f800000 };
    zzz();
}
fn zzz() {}
"#;
#[test]
fn debug_mode_union_field_read_reinterprets_storage() {
    // A union stores only the written field; printing the
    // other field reinterprets that storage (the reading the
    // whole-union print applies) instead of reporting an
    // uninitialized read.
    let dir = temp_dir("debug-union");
    let src = fixture(&dir, "union_field.rs", DEBUG_UNION_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint u\nprint u.value\nprint u.bits\nwhatis u.value\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "{bits: 1065353216, value: 1}",
        "$2 = 1",
        "$3 = 1065353216",
        "type = f32",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
    assert!(
        !stdout.contains("uninitialized"),
        "union field read must not report an uninitialized read: {stdout}"
    );
}
const DEBUG_DYN_BINDINGS_SRC: &str = r#"trait Tr {
    type A;
    type B;
    fn get(&self) -> u8;
}
struct S;
impl Tr for S {
    type A = u8;
    type B = u16;
    fn get(&self) -> u8 { 7 }
}
fn main() {
    let s = S;
    let d: &dyn Tr<A = u8, B = u16> = &s;
    zzz();
    let _z = d.get();
}
fn zzz() {}
"#;
#[test]
fn debug_mode_dyn_two_bindings_stay_on_one_trait() {
    // Two associated-type bindings of one trait fold into
    // that trait's name (`Tr<A=u8, B=u16>`); the second
    // binding must not stand up as a separate bound
    // (`Tr<A=u8> + Tr<B=u16>`).
    let dir = temp_dir("debug-dyn-bindings");
    let src = fixture(&dir, "dyn_bindings.rs", DEBUG_DYN_BINDINGS_SRC);
    let out = debug_driver(&dir, &src, "break 15\nrun\nwhatis d\n");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    assert!(
        stdout.contains("type = &dyn dyn_bindings::Tr<A=u8, B=u16>"),
        "bindings must combine on one trait: {stdout}"
    );
    assert!(
        !stdout.contains("Tr<A=u8> +"),
        "second binding must not become a separate bound: {stdout}"
    );
}

const DEBUG_FN_BP_SRC: &str = r#"fn helper<T: Copy>(x: T) -> T { x }
fn main() {
    let a = helper(1i32);
    let b = helper(2u8);
    zzz();
    let _ = (a, b);
}
fn zzz() {}
"#;

#[test]
fn debug_mode_fn_breakpoint_stops_in_every_instance() {
    // The driver names a generic function's later instances
    // `helper [helper::<u8>]`; a function breakpoint names
    // the definition, so it must stop in every instance, not
    // just the first.
    let dir = temp_dir("debug-fn-bp");
    let src = fixture(&dir, "fnbp.rs", DEBUG_FN_BP_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break-fn helper\nrun\nprint x\ncontinue\nprint x\ncontinue\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    assert!(
        stdout.contains("$1 = 1"),
        "first instance stop missing: {stdout}"
    );
    assert!(
        stdout.contains("$2 = 2"),
        "second instance stop missing: {stdout}"
    );
}

const DEBUG_LOCALS_FRAMES_SRC: &str = r#"fn deep() {
    let d = 1;
    marker();
    let _ = d;
}
fn marker() {}
fn main() {
    let x = 42;
    deep();
    let _ = x;
}
"#;

#[test]
fn debug_mode_info_locals_uses_selected_frame_position() {
    // After `up`, the caller's locals are scoped by the
    // caller's own position (its suspended call line), not by
    // the callee's stop line: `x` is declared before the call
    // but on a later line than the callee stopped on, and it
    // must still list.
    let dir = temp_dir("debug-locals-frames");
    let src = fixture(&dir, "locals_frames.rs", DEBUG_LOCALS_FRAMES_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 3\nrun\nup\ninfo-locals\ndown\ninfo-locals\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    assert!(
        stdout.contains("x = 42"),
        "caller local missing after `up`: {stdout}"
    );
    assert!(
        stdout.contains("d = 1"),
        "callee local missing after `down`: {stdout}"
    );
}

const DEBUG_NULL_PTR_SRC: &str = r#"fn main() {
    let p: *const i32 = std::ptr::null();
    let q: *mut f64 = std::ptr::null_mut();
    zzz();
    let _ = (p, q);
}
fn zzz() {}
"#;

#[test]
fn debug_mode_whatis_through_null_pointer_resolves_the_type() {
    // A type query reads no pointee: `whatis *p` over a null
    // pointer still names the pointee type, resolved from the
    // recorded type alone.
    let dir = temp_dir("debug-null-ptr");
    let src = fixture(&dir, "null_ptr.rs", DEBUG_NULL_PTR_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nwhatis *p\nptype *p\nwhatis *q\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    assert!(
        stdout.contains("type = i32"),
        "whatis *p must name i32: {stdout}"
    );
    assert!(
        stdout.contains("type = f64"),
        "whatis *q must name f64: {stdout}"
    );
    assert!(
        !stdout.contains("<error"),
        "type query must not surface the dereference error: {stdout}"
    );
}

const DEBUG_CAST_LEN_SRC: &str = r#"fn main() {
    let a: [i32; 3] = [1, 2, 3];
    let r: &[i32; 3] = &a;
    zzz();
    let _ = r;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_cast_to_shorter_array_limits_the_view() {
    // A cast to a shorter array type views only that many of
    // the backing elements (indexing bounds against the cast
    // length), and a cast to a longer array than the backing
    // holds is an error rather than a short read.
    let dir = temp_dir("debug-cast-len");
    let src = fixture(&dir, "cast_len.rs", DEBUG_CAST_LEN_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint *(r as *const [i32; 2])\nprint *(r as *const [i32; 2])[2]\nprint *(r as *const [i32; 5])\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    assert!(
        stdout.contains("$1 = [1, 2]"),
        "shorter cast must limit the elements: {stdout}"
    );
    assert!(
        !stdout.contains("[1, 2, 3]"),
        "the cast view must not show the whole backing array: {stdout}"
    );
    assert!(
        stdout.contains("<error"),
        "out-of-view index and over-long cast must error: {stdout}"
    );
}

const DEBUG_UNION_TUPLE_SRC: &str = r#"union U { a: u64, b: (u8, u32, u8) }
union V { a: u64, b: (u8, u8, u32) }
union W { a: u32, b: (u8, bool, u8) }
fn main() {
    let u = U { a: 0x030100000002 };
    let v = V { a: 0x0000000300000201 };
    let w = W { a: 0x00090701 };
    zzz();
    let _ = (u, v, w);
}
fn zzz() {}
"#;

#[test]
fn debug_mode_union_tuple_field_uses_compiler_layout() {
    // A union's tuple field is reinterpreted at the
    // compiler's tuple offsets, not in declaration order:
    // `(u8, u32, u8)` places the `u32` first, `(u8, u8, u32)`
    // keeps its tail last, and `(u8, bool, u8)` moves the
    // niche-carrying `bool` ahead of the plain `u8`.
    let dir = temp_dir("debug-union-tuple");
    let src = fixture(&dir, "union_tuple.rs", DEBUG_UNION_TUPLE_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 8\nrun\nprint u\nprint u.b\nprint v.b\nprint w.b\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["b: (1, 2, 3)", "$2 = (1, 2, 3)", "$3 = (1, 2, 3)", "$4 = (7, true, 9)"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}

fn debug_driver_lldb(dir: &Path, src: &Path, script: &str) -> Output {
    let script_path = dir.join("session.script");
    std::fs::write(&script_path, script).unwrap();
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("debug")
        .arg(src)
        .arg("--edition")
        .arg("2021")
        .arg("--debug-script")
        .arg(&script_path)
        .arg("--debug-dialect")
        .arg("lldb")
        .output()
        .expect("failed to run sparagmos-driver")
}

const DEBUG_CAST_REINTERPRET_SRC: &str = r#"fn main() {
    let a: [u32; 1] = [1065353216];
    let r: &[u32; 1] = &a;
    zzz();
    let _ = r;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_cast_reinterprets_array_elements() {
    // A pointer cast that changes the array element type
    // reinterprets the backing storage through the target
    // element layout, comparing byte counts rather than
    // element counts; a target the storage cannot fill, and
    // a change the scalar layouts cannot express, are
    // explicit errors rather than wrong values.
    let dir = temp_dir("debug-cast-reinterpret");
    let src = fixture(&dir, "cast_reinterpret.rs", DEBUG_CAST_REINTERPRET_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint *(r as *const [f32; 1])\nprint *(r as *const [u16; 2])\nprint *(r as *const [f32])\nprint *(r as *const [f32; 2])\nprint *(r as *const [f16; 2])\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "$1 = [1]",
        "$2 = [0, 16256]",
        "$3 = [1]",
        "cast to an array of 2 elements over 4 bytes of backing storage",
        "cast changing element type from u32 to f16 is not supported",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}

const DEBUG_CAST_SCALAR_SRC: &str = r#"fn main() {
    let a: u32 = 1065353216;
    let r: &u32 = &a;
    zzz();
    let _ = r;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_cast_reinterprets_scalar_pointee() {
    // A pointer cast that changes the scalar pointee type
    // re-reads the target storage through the target layout
    // at the dereference: the `u32` bits of 1.0f32 print as
    // `1`, not as the integer they spell in the source type.
    // A target wider than the storage, and a change the
    // scalar layouts cannot express, are explicit errors
    // rather than wrong values.
    let dir = temp_dir("debug-cast-scalar");
    let src = fixture(&dir, "cast_scalar.rs", DEBUG_CAST_SCALAR_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint *(r as *const f32)\nprint *(r as *const u32)\nprint *(r as *const f64)\nprint *(r as *const f16)\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "$1 = 1",
        "$2 = 1065353216",
        "cast to a 8-byte pointee over 4 bytes of backing storage",
        "cast changing pointee type from u32 to f16 is not supported",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}

const DEBUG_CAST_PTR_TYPE_SRC: &str = r#"fn main() {
    let a: [i32; 2] = [1, 2];
    let r: &[i32; 2] = &a;
    zzz();
    let _ = r;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_whatis_keeps_array_pointer_cast_type() {
    // A type query over a cast to an array pointer names the
    // cast's own type, pointer layer included; the array
    // view the cast materializes is the pointee's form, and
    // only a dereference consumes the pointer layer.
    let dir = temp_dir("debug-cast-ptr-type");
    let src = fixture(&dir, "cast_ptr_type.rs", DEBUG_CAST_PTR_TYPE_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nwhatis (r as *const [i32; 2])\nwhatis (r as *mut [i32; 2])\nwhatis *(r as *const [i32; 2])\nprint *(r as *const [i32; 2])\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "type = *const [i32; 2]",
        "type = *mut [i32; 2]",
        "type = [i32; 2]",
        "$1 = [1, 2]",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}

const DEBUG_PTR_QUALIFIERS_SRC: &str = r#"struct S { p: *mut i32, q: *const i32 }
struct R<'a> { r: &'a mut i32 }
fn main() {
    let mut x = 5i32;
    let t: (*const i32,) = (&x as *const i32,);
    let s = S { p: &mut x as *mut i32, q: &x as *const i32 };
    let rr = R { r: &mut x };
    zzz();
    let _ = (t, s, rr);
}
fn zzz() {}
"#;

#[test]
fn debug_mode_whatis_preserves_pointer_qualifiers() {
    // A projected pointer or reference value reports its
    // recorded type with its qualifier: `*const`, `*mut`
    // and `&mut` do not collapse into a shared `&`.
    let dir = temp_dir("debug-ptr-qualifiers");
    let src = fixture(&dir, "ptr_qualifiers.rs", DEBUG_PTR_QUALIFIERS_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 8\nrun\nwhatis t.0\nwhatis s.p\nwhatis s.q\nwhatis rr.r\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["type = *const i32", "type = *mut i32", "type = &mut i32"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
    assert!(
        !stdout.contains("type = &i32"),
        "pointer qualifiers must not collapse into a shared reference: {stdout}"
    );
}

const DEBUG_LLDB_UNSIGNED_SRC: &str = r#"fn main() {
    let b: i8 = -1;
    zzz();
    let _ = b;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_lldb_unsigned_format_masks_to_type_width() {
    // An lldb unsigned format registration renders the
    // unsigned reading of the value's stored bytes: an
    // `i8` holding -1 prints as 255, not as the 128-bit
    // sign-extension of the interpreter's scalar.
    let dir = temp_dir("debug-lldb-unsigned");
    let src = fixture(&dir, "lldb_unsigned.rs", DEBUG_LLDB_UNSIGNED_SRC);
    let out = debug_driver_lldb(
        &dir,
        &src,
        "type-format unsigned i8\nbreak 3\nrun\nprint b\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    assert!(
        stdout.contains("(i8) b = 255"),
        "unsigned format must mask to the i8 width: {stdout}"
    );
    assert!(
        !stdout.contains("340282366920938463463374607431768211455"),
        "unsigned format must not print the 128-bit sign-extension: {stdout}"
    );
}

const DEBUG_MUT_SLICE_SRC: &str = r#"fn b() {}
fn main() {
    let mut_slice: &mut [i32] = &mut [2, 3, 5, 7];
    let mut buf = String::from("mutable string slice");
    let mut_str_slice: &mut str = buf.as_mut_str();
    b();
    let _ = (mut_slice, mut_str_slice);
}
"#;

#[test]
fn debug_mode_mut_slice_and_str_render_with_pretty_printers() {
    // A mutable reference's rendering dispatches on the same
    // slice/string summaries as a shared reference's: the
    // qualifier changes the recorded descriptor, not the
    // summary form.
    let dir = temp_dir("debug-mut-slice");
    let src = fixture(&dir, "mut_slice.rs", DEBUG_MUT_SLICE_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "pretty-printers\nbreak 6\nrun\nprint mut_slice\nprint mut_str_slice\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "$1 = &mut [i32](size=4) = {2, 3, 5, 7}",
        "$2 = \"mutable string slice\"",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}

const DEBUG_MUT_CAPTURE_SRC: &str = r#"struct MyStruct { my_field1: u32, my_field2: u32 }
fn _zzz() {}
fn main() {
    let mut my_var = MyStruct { my_field1: 11, my_field2: 22 };
    let my_ref = &mut my_var;
    let test = || {
        let a = my_ref;
    };
    _zzz();
    let _ = test;
}
"#;

#[test]
fn debug_mode_lldb_env_field_reads_through_mut_capture() {
    // A closure environment field holding a `&mut` renders
    // its pointee in the lldb environment form, the same
    // reading a shared-reference capture gets.
    let dir = temp_dir("debug-mut-capture");
    let src = fixture(&dir, "mut_capture.rs", DEBUG_MUT_CAPTURE_SRC);
    let out = debug_driver_lldb(
        &dir,
        &src,
        "break 9\nrun\nv test\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    assert!(
        stdout.contains("test = {my_ref:{my_field1:11, my_field2:22}}"),
        "mutable capture must render its pointee: {stdout}"
    );
}

const DEBUG_CAST_PTR_VALUE_SRC: &str = r#"fn main() {
    let a: [i32; 2] = [1, 2];
    let r: &[i32; 2] = &a;
    zzz();
    let _ = r;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_cast_to_array_pointer_keeps_the_pointer_value() {
    // A cast to an array pointer is still a pointer: printing
    // it renders the pointer, not the pointee. The array view
    // materializes only at the step that consumes the pointer
    // (a dereference, or a projection through it), and an
    // element-changing cast reinterprets at that step.
    let dir = temp_dir("debug-cast-ptr-value");
    let src = fixture(&dir, "cast_ptr_value.rs", DEBUG_CAST_PTR_VALUE_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint r\nprint (r as *const [i32; 2])\nprint *(r as *const [i32; 2])\nprint (r as *const [i32; 2])[1]\nprint *(r as *const [u16; 4])\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    let printed = |n: u32| -> String {
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(&format!("${n} = ")))
            .unwrap_or_else(|| panic!("no ${n} line in debug transcript: {stdout}"))
            .to_string()
    };
    assert_eq!(
        printed(2),
        printed(1),
        "cast print must render the pointer: {stdout}"
    );
    assert_eq!(printed(3), "[1, 2]");
    assert_eq!(printed(4), "2");
    assert_eq!(printed(5), "[1, 0, 2, 0]");
}

const DEBUG_UNION_TUPLE_COMMA_SRC: &str = r#"union U { bits: u32, value: (u32,) }
fn main() {
    let u = U { bits: 0x2a };
    zzz();
    let _ = &u;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_union_one_tuple_field_reads_through_trailing_comma() {
    // The canonical one-element tuple spelling carries a
    // trailing comma (`(u32,)`): the layout walk must not
    // read it as a second, empty component, or the field's
    // reinterpretation fails to resolve the layout and the
    // read reports an uninitialized place.
    let dir = temp_dir("debug-union-tuple-comma");
    let src = fixture(&dir, "union_tuple_comma.rs", DEBUG_UNION_TUPLE_COMMA_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint u.value\nprint u.value.0\nprint u\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["$1 = (42,)", "$2 = 42", "value: (42,)"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}

const DEBUG_NESTED_CAST_SRC: &str = r#"fn main() {
    let x: u32 = 0x3f800000;
    let r: &u32 = &x;
    zzz();
    let _ = r;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_nested_casts_split_at_the_top_level() {
    // The cast separator of a parenthesised expression is
    // the top-level ` as `: a chain grouped as
    // `((r as *const u32) as *const f32)`, and an extra
    // grouping layer around a single cast, both reach the
    // evaluator instead of splitting at the inner separator
    // and failing to parse.
    let dir = temp_dir("debug-nested-cast");
    let src = fixture(&dir, "nested_cast.rs", DEBUG_NESTED_CAST_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint *((r as *const u32) as *const f32)\nprint *((r as *const f32))\nwhatis *((r as *const u32) as *const f32)\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["$1 = 1", "$2 = 1", "type = f32"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
    assert!(
        !stdout.contains("cannot parse expression"),
        "nested casts must parse: {stdout}"
    );
}

const DEBUG_LLDB_PROJECTED_I8_SRC: &str = r#"fn main() {
    let t: (i8,) = (65,);
    zzz();
    let _ = t;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_lldb_int_format_applies_to_projected_values() {
    // A format registration names a type, not a variable:
    // printing a projected value of that type (`t.0`, whose
    // recorded tag the expression steps cleared) applies the
    // registration through the value's concrete descriptor.
    let dir = temp_dir("debug-lldb-projected");
    let src = fixture(&dir, "lldb_projected.rs", DEBUG_LLDB_PROJECTED_I8_SRC);
    let out = debug_driver_lldb(&dir, &src, "type-format hex i8\nbreak 3\nrun\nprint t.0\n");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    assert!(
        stdout.contains("= 0x41"),
        "hex registration must apply to the projected i8: {stdout}"
    );
}

const DEBUG_DEREF_PROJECTED_SRC: &str = r#"struct S { p: *mut i32 }
fn main() {
    let mut v: i32 = 9;
    let t: (*const i32,) = (&v as *const i32,);
    let s = S { p: &mut v as *mut i32 };
    let arr: [&i32; 2] = [&v, &v];
    zzz();
    let _ = (&t, &s, &arr);
}
fn zzz() {}
"#;

#[test]
fn debug_mode_deref_applies_after_projection_steps() {
    // A leading dereference reads the projected value's
    // pointee (`*t.0` is `*(t.0)`, as in Rust and C), not
    // the base value's: dereferencing the tuple or the
    // struct itself is a type mismatch, and the type query
    // for the dereferenced projection reports the pointee.
    let dir = temp_dir("debug-deref-projected");
    let src = fixture(&dir, "deref_projected.rs", DEBUG_DEREF_PROJECTED_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 7\nrun\nprint *t.0\nwhatis *t.0\nprint *s.p\nwhatis *s.p\nprint *arr[1]\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["$1 = 9", "$2 = 9", "$3 = 9"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
    assert_eq!(
        stdout.matches("type = i32").count(),
        2,
        "both dereferenced projections must report the pointee type: {stdout}"
    );
    assert!(
        !stdout.contains("type mismatch in dereference"),
        "the base value must not be dereferenced: {stdout}"
    );
}

const DEBUG_CHAINED_ARRAY_CAST_SRC: &str = r#"fn main() {
    let a: [u32; 1] = [1065353216];
    let r: &[u32; 1] = &a;
    zzz();
    let _ = r;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_chained_array_casts_keep_the_backing_element_type() {
    // A second array-pointer cast before the dereference
    // re-spells the pending view but does not reinterpret
    // the storage: materialization reads the bytes through
    // the final cast's element type, starting from the
    // backing's actual element type -- not the intermediate
    // cast's, which renders the stored integer unchanged,
    // and not a byte-count comparison against the
    // intermediate array's extent.
    let dir = temp_dir("debug-chained-array-cast");
    let src = fixture(&dir, "chained_array_cast.rs", DEBUG_CHAINED_ARRAY_CAST_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint *(r as *const [f32; 1])\nprint *((r as *const [f32; 1]) as *const [f32; 1])\nprint *((r as *const [u8; 4]) as *const [f32; 1])\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["$1 = [1]", "$2 = [1]", "$3 = [1]"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
    assert!(
        !stdout.contains("<error"),
        "chained array casts must not error: {stdout}"
    );
}

const DEBUG_DEREF_PAREN_PROJECTED_SRC: &str = r#"fn main() {
    let v: i32 = 42;
    let t: (&i32,) = (&v,);
    zzz();
    let _ = t;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_deref_of_parenthesised_projection_projects_first() {
    // `*(t).0` groups the projection with the parenthesised
    // base: the field step applies to the tuple and the
    // leading dereference reads the field's pointee -- the
    // same order the unparenthesised `*t.0` takes. Appending
    // the dereference before the trailing projection
    // dereferences the tuple itself.
    let dir = temp_dir("debug-deref-paren");
    let src = fixture(&dir, "deref_paren.rs", DEBUG_DEREF_PAREN_PROJECTED_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint *t.0\nprint *(t).0\nwhatis *(t).0\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["$1 = 42", "$2 = 42", "type = i32"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
    assert!(
        !stdout.contains("type mismatch in dereference"),
        "the parenthesised base must not be dereferenced: {stdout}"
    );
}

const DEBUG_DATA_PTR_CAST_SRC: &str = r#"fn main() {
    let a: [u32; 1] = [1065353216];
    let s: &[u32] = &a;
    zzz();
    let _ = s;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_data_ptr_cast_recovers_source_element_layout() {
    // A fat pointer's `data_ptr` carries a pointer-to-scalar
    // descriptor, not an array one: the cast's source element
    // type is the scalar it names, recovered when the backing
    // value is the element aggregate. Without that recovery
    // the reinterpretation is skipped (the stored integer
    // prints unchanged) and a wider-element-count cast
    // misreports the backing extent in element units.
    let dir = temp_dir("debug-data-ptr-cast");
    let src = fixture(&dir, "data_ptr_cast.rs", DEBUG_DATA_PTR_CAST_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint *(s.data_ptr as *const [f32; 1])\nprint *(s.data_ptr as *const [u16; 2])\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["$1 = [1]", "$2 = [0, 16256]"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
    assert!(
        !stdout.contains("<error"),
        "data_ptr casts must not error: {stdout}"
    );
}

const DEBUG_MIXED_CAST_CHAIN_SRC: &str = r#"fn main() {
    let v: u32 = 1065353216;
    let r: *const u32 = &v;
    zzz();
    let _ = r;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_cast_through_array_pointer_keeps_storage_type() {
    // A scalar pointer cast through an array-pointer type
    // and back composes its dereference reinterpretation
    // from the descriptor the pending array view recorded --
    // the storage's actual type -- not from the array
    // spelling the second cast replaces, which discards the
    // recorded source and renders the stored integer
    // unchanged.
    let dir = temp_dir("debug-mixed-cast-chain");
    let src = fixture(&dir, "mixed_cast_chain.rs", DEBUG_MIXED_CAST_CHAIN_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint *(r as *const f32)\nprint *((r as *const [u32; 1]) as *const f32)\nprint *((r as *const [f32; 1]) as *const f32)\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["$1 = 1", "$2 = 1", "$3 = 1"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
    assert!(
        !stdout.contains("<error"),
        "mixed cast chains must not error: {stdout}"
    );
}

const DEBUG_MIXED_CAST_CHAIN_REF_SRC: &str = r#"fn main() {
    let a: [u32; 1] = [1065353216];
    let r: &[u32; 1] = &a;
    zzz();
    let _ = r;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_cast_through_array_pointer_from_array_ref() {
    // The same family switch from an array reference: the
    // recorded source is the array descriptor itself, whose
    // element is the storage's scalar, and the dereference
    // re-reads the element aggregate's leading bytes through
    // the target layout.
    let dir = temp_dir("debug-mixed-cast-chain-ref");
    let src = fixture(
        &dir,
        "mixed_cast_chain_ref.rs",
        DEBUG_MIXED_CAST_CHAIN_REF_SRC,
    );
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint *((r as *const [f32; 1]) as *const f32)\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    assert!(
        stdout.contains("$1 = 1"),
        "missing `$1 = 1` in debug transcript: {stdout}"
    );
    assert!(
        !stdout.contains("<error"),
        "mixed cast chains must not error: {stdout}"
    );
}

const DEBUG_SCALAR_THEN_ARRAY_CAST_SRC: &str = r#"fn main() {
    let a: [u32; 1] = [1065353216];
    let r: &[u32; 1] = &a;
    zzz();
    let _ = r;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_scalar_cast_then_array_view_uses_backing_type() {
    // A scalar cast followed by an array-pointer cast keeps
    // the backing storage's type for the view's
    // materialization: the first cast records the pending
    // reinterpretation from the array pointee's element, and
    // the array view records that source rather than the
    // intermediate cast's descriptor, whose pointee the
    // storage does not hold.
    let dir = temp_dir("debug-scalar-then-array-cast");
    let src = fixture(
        &dir,
        "scalar_then_array_cast.rs",
        DEBUG_SCALAR_THEN_ARRAY_CAST_SRC,
    );
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint *((r as *const f32) as *const [f32; 1])\nprint *((r as *const f32) as *const [u16; 2])\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["$1 = [1]", "$2 = [0, 16256]"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
    assert!(
        !stdout.contains("<error"),
        "scalar-to-array cast chains must not error: {stdout}"
    );
}

const DEBUG_CAST_INDEX_SRC: &str = r#"fn main() {
    let a: [u32; 2] = [1065353216, 1073741824];
    let s: &[u32] = &a;
    zzz();
    let _ = s;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_cast_pointer_index_reinterprets_elements() {
    // Indexing a cast pointer dereferences it, so the
    // pending pointee reinterpretation applies to the
    // indexed element as it does to a plain dereference --
    // at the target type's width, so a narrower target
    // strides through the storage at its own element size,
    // and an index past the storage is an explicit error.
    let dir = temp_dir("debug-cast-index");
    let src = fixture(&dir, "cast_index.rs", DEBUG_CAST_INDEX_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint *(s.data_ptr as *const f32)\nprint (s.data_ptr as *const f32)[0]\nprint (s.data_ptr as *const f32)[1]\nprint (s.data_ptr as *const u16)[3]\nprint (s.data_ptr as *const f64)[1]\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["$1 = 1", "$2 = 1", "$3 = 2", "$4 = 16384"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
    assert!(
        stdout.contains("index 1 of a 8-byte element over 8 bytes of backing storage"),
        "an out-of-storage index must error explicitly: {stdout}"
    );
}

const DEBUG_CAST_NOOP_SRC: &str = r#"fn main() {
    let a: [u32; 2] = [42, 7];
    let s: &[u32] = &a;
    zzz();
    let _ = s;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_noop_cast_deref_consumes_one_element() {
    // A cast to the pointer's existing element type changes
    // no bytes, so no reinterpretation is owed -- but the
    // dereference still reads through the cast pointer, whose
    // pointee storage is the whole element aggregate: the
    // read consumes the first scalar element, the value
    // indexing the same pointer reports, rather than
    // rendering the entire aggregate under the element type.
    let dir = temp_dir("debug-cast-noop");
    let src = fixture(&dir, "cast_noop.rs", DEBUG_CAST_NOOP_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint *(s.data_ptr as *const u32)\nprint (s.data_ptr as *const u32)[0]\nprint (s.data_ptr as *const u32)[1]\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["$1 = 42", "$2 = 42", "$3 = 7"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
    assert!(
        !stdout.contains("(42, 7)"),
        "a no-op cast dereference must not render the whole aggregate: {stdout}"
    );
}

const DEBUG_ARRAY_VIEW_SCALAR_BACKING_SRC: &str = r#"fn main() {
    let v: u32 = 1065353216;
    let r: &u32 = &v;
    zzz();
    let _ = r;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_array_view_over_scalar_backing_materializes() {
    // An array view over a scalar pointee materializes the
    // scalar's bytes as the target's elements: the backing
    // is a one-element storage, not an aggregate the plain
    // view can pass through, and even an element type equal
    // to the scalar's own still forms the array value.
    let dir = temp_dir("debug-array-view-scalar-backing");
    let src = fixture(
        &dir,
        "array_view_scalar_backing.rs",
        DEBUG_ARRAY_VIEW_SCALAR_BACKING_SRC,
    );
    let out = debug_driver(
        &dir,
        &src,
        "break 4\nrun\nprint *(r as *const [f32; 1])\nprint *(r as *const [u16; 2])\nprint *(r as *const [u32; 1])\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["$1 = [1]", "$2 = [0, 16256]", "$3 = [1065353216]"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
    assert!(
        !stdout.contains("<error"),
        "array views over scalar backing must not error: {stdout}"
    );
}

const DEBUG_UNION_PTR_FIELD_SRC: &str = r#"union U { bits: u32, value: f32, ptr: *const u32 }
fn main() {
    let u = U { bits: 0x3f800000 };
    zzz();
}
fn zzz() {}
"#;

#[test]
fn debug_mode_union_field_read_with_unsupported_sibling() {
    // Reading one field of a union serializes the stored
    // field and deserializes the requested field
    // independently: a sibling whose layout the byte
    // conversion cannot express (here a raw pointer) does
    // not disable the supported fields' reads, the way the
    // whole-union reading -- which must produce every field
    // at once -- is disabled by it.
    let dir = temp_dir("debug-union-ptr-field");
    let src = fixture(&dir, "union_ptr_field.rs", DEBUG_UNION_PTR_FIELD_SRC);
    let out = debug_driver(&dir, &src, "break 4\nrun\nprint u.value\nprint u.bits\n");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["$1 = 1", "$2 = 1065353216"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
    assert!(
        !stdout.contains("uninitialized"),
        "union field read must not report an uninitialized read: {stdout}"
    );
}

const DEBUG_LLDB_NEG_HEX_SRC: &str = r#"fn main() {
    let b: i8 = -1;
    zzz();
    let _ = b;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_lldb_hex_format_applies_to_negative_values() {
    // A hexadecimal registration reads the stored bits at the
    // registered width for negative values too: -1 in an
    // 8-bit hex format is 0xff (the reading lldb gives),
    // not a fall back to the decimal rendering.
    let dir = temp_dir("debug-lldb-neg-hex");
    let src = fixture(&dir, "lldb_neg_hex.rs", DEBUG_LLDB_NEG_HEX_SRC);
    let out = debug_driver_lldb(&dir, &src, "type-format hex i8\nbreak 3\nrun\nprint b\n");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    assert!(
        stdout.contains("= 0xff"),
        "hex registration must mask a negative i8 to its width: {stdout}"
    );
    assert!(
        !stdout.contains("= -1"),
        "the decimal rendering must not survive the hex registration: {stdout}"
    );
}

#[test]
fn debug_mode_lldb_format_registration_applies_in_command_order() {
    // A registration takes effect when its command runs, not
    // before the session starts: re-registering a type's
    // format mid-session changes the later prints and leaves
    // the earlier ones under the previous registration (the
    // order lldb applies `type format add` in).
    let dir = temp_dir("debug-lldb-format-order");
    let src = fixture(&dir, "lldb_format_order.rs", DEBUG_LLDB_NEG_HEX_SRC);
    let out = debug_driver_lldb(
        &dir,
        &src,
        "type-format hex i8\nbreak 3\nrun\nprint b\ntype-format dec i8\nprint b\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    let hex_at = stdout.find("= 0xff").expect("first print renders hex");
    let dec_at = stdout.find("= -1").expect("second print renders decimal");
    assert!(
        hex_at < dec_at,
        "the hex print must precede the decimal one: {stdout}"
    );
}

const DEBUG_LLDB_COMMAND_FORMAT_SRC: &str = r#"fn main() {
    let b: i8 = -1;
    let c: char = 'a';
    zzz();
    let _ = (b, c);
}
fn zzz() {}
"#;

#[test]
fn debug_mode_lldb_command_formats_are_honored() {
    // A per-command format prescribes the print's own
    // reading: `print/u` reads a negative value unsigned at
    // its width (255, not -1), and `print/d` of a char reads
    // its code point (97, not the character summary) -- the
    // semantics the runner's translation now carries into
    // the session instead of dropping the format.
    let dir = temp_dir("debug-lldb-command-format");
    let src = fixture(
        &dir,
        "lldb_command_format.rs",
        DEBUG_LLDB_COMMAND_FORMAT_SRC,
    );
    let out = debug_driver_lldb(
        &dir,
        &src,
        "break 4\nrun\nprint/u b\nprint/d c\nprint/d b\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["(i8) b = 255", "(char) c = 97", "(i8) b = -1"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
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

const BOX_DEFAULT_SLICE_OF_ARRAYS_SRC: &str = r#"
fn main() {
    let b: Box<[[i32; 2]]> = Default::default();
    println!("slice {} {:?}", b.len(), b);
    let c: Box<[i32; 2]> = Default::default();
    println!("array {:?}", c);
    let d: Box<[[u8; 3]]> = Box::default();
    println!("via-box {} {:?}", d.len(), d);
}
"#;

#[test]
fn run_mode_box_default_slice_of_arrays() {
    // `Box<[[i32; 2]]>::default()` is an empty boxed slice, but
    // the pointee check in `default_value` judged slice-ness
    // with a raw `!contains(';')`, fell through to the array
    // default of the outer tag, and gave up as unsupported.
    // The `Box<[i32; 2]>` line pins the array direction: an
    // array pointee still defaults elementwise.
    run_ok(
        "box-default-slice-of-arrays",
        BOX_DEFAULT_SLICE_OF_ARRAYS_SRC,
        "slice 0 []\narray [0, 0]\nvia-box 0 []\n",
    );
}

const AS_REF_SLICE_OF_ARRAYS_SRC: &str = r#"
fn main() {
    let a: [[i32; 2]; 2] = [[1, 2], [3, 4]];
    let s: &[[i32; 2]] = a.as_ref();
    println!("asref {} {:?}", s.len(), s);
    let t: &[[i32; 2]] = std::borrow::Borrow::borrow(&a);
    println!("borrow {} {:?}", t.len(), t);
}
"#;

#[test]
fn run_mode_as_ref_slice_of_arrays() {
    // The array-to-slice view in `view_value_as` used the same
    // raw `contains(';')` pair to recognize its case; it now
    // goes through `is_slice_tag` on both sides like the other
    // consumers. (The pre-unification code reached the same
    // output here through the reference fallback, so this test
    // pins the behavior while the predicate is unified.)
    run_ok(
        "as-ref-slice-of-arrays",
        AS_REF_SLICE_OF_ARRAYS_SRC,
        "asref 2 [[1, 2], [3, 4]]\nborrow 2 [[1, 2], [3, 4]]\n",
    );
}
