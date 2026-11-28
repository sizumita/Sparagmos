use super::{Command, aux_spec, fixture, run_driver, run_driver_extra, run_driver_with_aux, run_ok, run_ok_edition, summarize_driver, temp_dir};


#[test]
fn run_mode_cstr_table_over_budget_stays_undecoded() {
    // A table of references to one shared 1 MiB `CStr`: each
    // element's decode reaches the `CStr` payload copy in
    // `decode_adt`, so the copies are charged against the
    // shared expansion budget and the table stays undecoded
    // instead of materialising one 1 MiB copy per element.
    let lit = "a".repeat(1048576);
    let src = format!(
        "use std::ffi::CStr; static BIG: &CStr = c\"{lit}\"; static TABLE: [&CStr; 5000] = [BIG; 5000]; fn main() {{ println!(\"{{}} {{}}\", TABLE.len(), TABLE[0].to_bytes().len()); }}"
    );
    let dir = temp_dir("cstr-table-over-budget");
    let src = fixture(&dir, "cstr_table_over_budget.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an over-budget shared CStr table must stay undecoded; stderr={stderr}"
    );
}

#[test]
fn run_mode_cstr_table_under_budget_reads_back() {
    // The same shared-`CStr` table shape at a size the budget
    // covers still materialises and reads back exactly.
    let lit = "b".repeat(4096);
    let src = format!(
        "use std::ffi::CStr; static BIG: &CStr = c\"{lit}\"; static TABLE: [&CStr; 100] = [BIG; 100]; fn main() {{ println!(\"{{}} {{}}\", TABLE.len(), TABLE[99].to_bytes().len()); }}"
    );
    let dir = temp_dir("cstr-table-under-budget");
    let src = fixture(&dir, "cstr_table_under_budget.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an under-budget shared CStr table must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "100 4096\n",
        "decoded table values must match the source program"
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

const LOCAL_EXPORT_PTR_SRC: &str = r#"
mod defs {
    #[no_mangle]
    pub fn ptr_export(x: i32) -> i32 { x + 40 }
}

unsafe extern "Rust" {
    safe fn ptr_export(x: i32) -> i32;
}

fn main() {
    println!("direct={}", ptr_export(2));
    println!("indirect={}", (ptr_export as fn(i32) -> i32)(3));
}
"#;

#[test]
fn run_mode_extern_decl_fn_pointer_binds_local_export() {
    // A function-item value names the extern declaration
    // itself, not the definition a direct call is rewritten
    // to: the value's call resolves through the link alias
    // to the in-program definition exporting the
    // declaration's symbol, as a native link binds it.
    let dir = temp_dir("local-export-ptr");
    let src = fixture(&dir, "localexportptr.rs", LOCAL_EXPORT_PTR_SRC);
    let out = run_driver_extra(&src, &["--edition", "2021"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a call through a function value naming the declaration must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "direct=42\nindirect=43\n",
        "both the direct and the value call execute the exported definition"
    );
}

const TRACK_CALLER_FFI_SRC: &str = r#"
use std::panic::Location;

unsafe extern "Rust" {
    #[track_caller]
    safe fn ffi_tracked() -> &'static Location<'static>;
    safe fn ffi_untracked() -> &'static Location<'static>;
}

fn nested_tracked() -> &'static Location<'static> {
    ffi_tracked()
}

mod provides {
    use std::panic::Location;
    #[track_caller]
    #[no_mangle]
    fn ffi_tracked() -> &'static Location<'static> {
        Location::caller()
    }
    #[no_mangle]
    fn ffi_untracked() -> &'static Location<'static> {
        Location::caller()
    }
}

fn main() {
    let tracked = ffi_tracked();
    println!("tracked={}:{}", tracked.line(), tracked.column());
    let untracked = ffi_untracked();
    println!("untracked={}:{}", untracked.line(), untracked.column());
    let contained = nested_tracked();
    println!("contained={}:{}", contained.line(), contained.column());
    let indirect = (ffi_tracked as fn() -> &'static Location<'static>)();
    println!("indirect={}:{}", indirect.line(), indirect.column());
}
"#;

#[test]
fn run_mode_track_caller_extern_decl_locations() {
    // The rfc-2091 track-caller FFI shape: `#[track_caller]`
    // extern declarations whose `#[no_mangle]` definitions
    // live in the same crate. Direct calls present their
    // call sites, the untracked definition presents its own
    // `Location::caller()` site, and a call through a
    // function-item value naming the declaration presents
    // the declaration's own definition location (line 6),
    // as the compiler's shim for the declaration does.
    let dir = temp_dir("track-caller-ffi");
    let src = fixture(&dir, "trackcallerffi.rs", TRACK_CALLER_FFI_SRC);
    let out = run_driver_extra(&src, &["--edition", "2021"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "track-caller FFI calls must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "tracked=28:19\nuntracked=23:9\ncontained=11:5\nindirect=6:5\n",
        "each call shape presents the location the compiler's resolution assigns it"
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

const IMPORTED_MAIN_AUX_SRC: &str = r#"
pub fn boilerplate() {
    println!("Hello world!");
}
"#;

const IMPORTED_MAIN_EXTERN_SRC: &str = r#"
extern crate main_functions;
pub use main_functions::boilerplate as main;
"#;

#[test]
fn run_mode_imported_main_from_extern_crate() {
    let dir = temp_dir("imported-main-extern");
    let aux = fixture(&dir, "main_functions.rs", IMPORTED_MAIN_AUX_SRC);
    let src = fixture(
        &dir,
        "imported_main_from_extern_crate.rs",
        IMPORTED_MAIN_EXTERN_SRC,
    );
    let spec = format!("main_functions={}", aux.display());
    let out = run_driver_extra(&src, &["--aux", &spec]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr} stdout={stdout}");
    assert_eq!(stdout, "Hello world!\n", "stderr={stderr}");
}

// ---------------------------------------------------------------------------
// Sysroot extern pins (issue #532): an auxiliary crate whose
// source declares `extern crate libc;` must compile in a
// rustup sysroot that offers two libc candidates for the host
// target (the rust-std rlib+rmeta pair and the rustc-dev
// component's rmeta-only build) among the dozens of targets
// a toolchain can carry std for. The pin scan considers the
// host target's lib dir only; scanning every target's dir
// finds a complete candidate under most of them and pins
// nothing (E0464).
// ---------------------------------------------------------------------------

const USES_LIBC_AUX_SRC: &str = r#"
#![feature(rustc_private)]
extern crate libc;

pub fn answer() -> i32 {
    42
}
"#;

const USES_LIBC_MAIN_SRC: &str = r#"
extern crate uses_libc;

fn main() {
    assert_eq!(uses_libc::answer(), 42);
    println!("answer ok");
}
"#;

#[test]
fn run_mode_aux_extern_crate_libc_compiles() {
    let dir = temp_dir("aux-libc-pin");
    let aux = fixture(&dir, "uses_libc.rs", USES_LIBC_AUX_SRC);
    let src = fixture(&dir, "uses_libc_main.rs", USES_LIBC_MAIN_SRC);
    let spec = format!("uses_libc={}", aux.display());
    let out = run_driver_extra(&src, &["--aux", &spec]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr} stdout={stdout}");
    assert_eq!(stdout, "answer ok\n", "stderr={stderr}");
}

const EXTERN_WEAK_UNDEFINED_SRC: &str = r#"
#![feature(linkage)]
extern "C" {
    #[linkage = "extern_weak"]
    static MISSING: *const u32;
}
fn main() {
    let p = unsafe { MISSING };
    println!("null={}", p.is_null());
}
"#;

#[test]
fn run_mode_extern_weak_undefined_static_reads_null() {
    // An `extern_weak` import with no definition anywhere in
    // the session denotes the undefined weak symbol, whose
    // address is the null address: taking the declaration's
    // value reads the null pointer, not uninitialized storage.
    run_ok(
        "extern-weak-undefined",
        EXTERN_WEAK_UNDEFINED_SRC,
        "null=true\n",
    );
}

const LINKAGE_AUX_SRC: &str = r#"
#[no_mangle]
pub static foo: isize = 3;

pub fn bar() {}

pub fn get_foo() -> isize {
    unsafe { foo }
}
"#;

const LINKAGE_MAIN_SRC: &str = r#"
#![feature(linkage)]
extern crate auxlib as other;

extern "C" {
    #[linkage = "extern_weak"]
    static foo: *const isize;
    #[linkage = "extern_weak"]
    static never_defined: *mut isize;
}

fn main() {
    other::bar();
    unsafe {
        println!("defined={} value={}", !foo.is_null(), *foo);
        *(foo as *mut isize) = 9;
        println!("after={}", other::get_foo());
        println!("missing={}", never_defined.is_null());
    }
}
"#;

#[test]
fn run_mode_extern_static_defined_in_aux_crate_resolves() {
    // A foreign declaration whose symbol an auxiliary crate
    // defines denotes the definition's one storage: the
    // declaration's value is the definition's address, reads
    // through it see the initializer, and a write through it
    // is visible to the defining crate. A second declaration
    // with no definition in the session reads as null.
    let dir = temp_dir("extern-static-defined");
    let aux = fixture(&dir, "auxlib.rs", LINKAGE_AUX_SRC);
    let src = fixture(&dir, "case.rs", LINKAGE_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("auxlib", &aux)]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the defined import must resolve; stderr={stderr}"
    );
    assert_eq!(
        stdout, "defined=true value=3\nafter=9\nmissing=true\n",
        "stderr={stderr}"
    );
}

const THREAD_LOCAL_EXTERN_AUX_SRC: &str = r#"
#![feature(thread_local)]

use std::cell::Cell;

#[no_mangle]
#[thread_local]
pub static FOO: Cell<u32> = Cell::new(3);
"#;

const THREAD_LOCAL_EXTERN_MAIN_SRC: &str = r#"
#![feature(thread_local)]

extern crate auxlib;

use std::cell::Cell;

extern "C" {
    #[thread_local]
    static FOO: Cell<u32>;
}

fn main() {
    unsafe {
        println!("foo={}", FOO.get());
        FOO.set(9);
        println!("foo2={}", FOO.get());
    }
}
"#;

#[test]
fn run_mode_thread_local_extern_static_defined_in_aux_resolves() {
    // A thread-local foreign declaration whose symbol an
    // auxiliary crate defines denotes the definition's
    // thread-local storage (Issue #395): the declaration's
    // `ThreadLocalRef` is recorded from the definition's
    // initializer, so reads see the initializer and writes
    // through the declaration stick, exactly as for a plain
    // foreign static defined in the session.
    let dir = temp_dir("thread-local-extern-static");
    let aux = fixture(&dir, "auxlib.rs", THREAD_LOCAL_EXTERN_AUX_SRC);
    let src = fixture(&dir, "case.rs", THREAD_LOCAL_EXTERN_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("auxlib", &aux)]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the defined thread-local import must resolve; stderr={stderr}"
    );
    assert_eq!(stdout, "foo=3\nfoo2=9\n", "stderr={stderr}");
}


// ----- `OsStr::to_ascii_lowercase` -----
//
// The method has no MIR body; the interpreter answers it
// with the byte-wise ASCII case folding std defines, over
// the receiver's platform bytes.

const OS_STR_TO_ASCII_LOWERCASE_SRC: &str = r#"
use std::ffi::OsStr;
fn main() {
    let s = OsStr::new("AbC dEf XYZ");
    println!("{:?}", s.to_ascii_lowercase());
    let owned = std::ffi::OsString::from("MiXeD");
    println!("{:?}", owned.as_os_str().to_ascii_lowercase());
    let lower = OsStr::new("already lower 123");
    println!("{:?}", lower.to_ascii_lowercase());
}
"#;

#[test]
fn run_mode_os_str_to_ascii_lowercase() {
    run_ok(
        "os-str-to-ascii-lowercase",
        OS_STR_TO_ASCII_LOWERCASE_SRC,
        "\"abc def xyz\"\n\"mixed\"\n\"already lower 123\"\n",
    );
}

const BOXED_CSTR_FROM_CSTRING_SRC: &str = r#"
use std::ffi::{CStr, CString};
fn main() {
    let b: Box<CStr> = Box::from(CString::new("abc").unwrap());
    println!("bytes_len={}", b.to_bytes().len());
    println!("first={}", b.to_bytes()[0]);
    println!("with_nul={:?}", b.to_bytes_with_nul());
    println!("debug={:?}", b);
    let d: &CStr = &b;
    println!("deref_debug={:?}", d);
    let c: Box<CStr> = Box::from(CString::new("abd").unwrap());
    println!("eq={} ord={:?}", b == Box::from(CString::new("abc").unwrap()), b.cmp(&c));
    let e: Box<CStr> = CString::new("").unwrap().into();
    println!("empty={} {}", e.to_bytes().len(), e.to_bytes_with_nul().len());
    let n = unsafe { CString::from_vec_unchecked(vec![b'x', 0x80, b'y']) };
    let nb: Box<CStr> = Box::from(n);
    println!("nonascii={:?}", nb.to_bytes_with_nul());
    println!("lossy_len={}", nb.to_string_lossy().chars().count());
}
"#;

#[test]
fn run_mode_boxed_cstr_from_cstring_keeps_its_length() {
    // `Box<CStr>: From<CString>` is `into_boxed_c_str`: the
    // box holds the C string's fat slice itself, not the
    // `CString` aggregate. Boxing the aggregate verbatim (the
    // generic `Box` conversion's move) left the
    // length-trusting consumers reading the aggregate's
    // field count (1) as the string's length, so `to_bytes`
    // came out empty and `to_bytes_with_nul` a single byte,
    // while the byte-scanning consumers (Debug, equality,
    // ordering) read through the aggregate and hid it.
    run_ok(
        "boxed-cstr-from-cstring",
        BOXED_CSTR_FROM_CSTRING_SRC,
        "bytes_len=3\nfirst=97\nwith_nul=[97, 98, 99, 0]\ndebug=\"abc\"\nderef_debug=\"abc\"\neq=true ord=Less\nempty=0 1\nnonascii=[120, 128, 121, 0]\nlossy_len=3\n",
    );
}

const BOXED_CSTR_FROM_COW_SRC: &str = r#"
use std::borrow::Cow;
use std::ffi::{CStr, CString};
fn main() {
    let owned: Box<CStr> = Box::from(Cow::Owned(CString::new("abc").unwrap()));
    println!("owned={} {}", owned.to_bytes().len(), owned.to_bytes_with_nul().len());
    let cs = CString::new("abc").unwrap();
    let cow: Cow<'_, CStr> = Cow::Borrowed(&cs);
    let borrowed: Box<CStr> = Box::from(cow);
    println!("borrowed={} {}", borrowed.to_bytes().len(), borrowed.to_bytes_with_nul().len());
    let from_ref: Box<CStr> = Box::from(&*cs);
    println!("from_ref={} {}", from_ref.to_bytes().len(), from_ref.to_bytes_with_nul().len());
}
"#;

#[test]
fn run_mode_boxed_cstr_from_cow_keeps_its_length() {
    // The Cow sources of `Box<CStr>` end at the same fat
    // slice pointee: the owned variant through its `CString`
    // (which had the same aggregate-boxing defect as the
    // direct `CString` source), the borrowed variant through
    // the `&CStr` it holds, as the plain `&CStr` source does.
    run_ok(
        "boxed-cstr-from-cow",
        BOXED_CSTR_FROM_COW_SRC,
        "owned=3 4\nborrowed=3 4\nfrom_ref=3 4\n",
    );
}

const INTO_BOXED_CSTR_SRC: &str = r#"
use std::ffi::{CStr, CString};
fn main() {
    let b = CString::new("hello").unwrap().into_boxed_c_str();
    println!("bytes={:?}", b.to_bytes());
    println!("with_nul={:?}", b.to_bytes_with_nul());
    println!("debug={:?}", b);
    println!("first={} len={}", b.to_bytes()[0], b.to_bytes().len());
    let d: &CStr = &b;
    println!("deref={:?}", d.to_bytes());
    let e = CString::new("").unwrap().into_boxed_c_str();
    println!("empty={} {}", e.to_bytes().len(), e.to_bytes_with_nul().len());
    let n = unsafe { CString::from_vec_unchecked(vec![b'x', 0x80, b'y']) };
    let nb = n.into_boxed_c_str();
    println!("nonascii={:?}", nb.to_bytes_with_nul());
    println!("lossy_len={}", nb.to_string_lossy().chars().count());
    let c = CString::new("abd").unwrap().into_boxed_c_str();
    println!("eq={} ord={:?}", b == CString::new("hello").unwrap().into_boxed_c_str(), b.cmp(&c));
}
"#;
const COW_CSTR_BORROWED_DROP_SRC: &str = r#"
use std::borrow::Cow;
use std::ffi::CStr;
fn main() {
    {
        let borrowed: Cow<'_, CStr> = Cow::Borrowed(c"abc");
        println!("borrowed len={}", borrowed.to_bytes().len());
    }
    println!("after borrowed drop");
}
"#;

#[test]
fn run_mode_cow_cstr_borrowed_literal_drop_converts() {
    // Dropping a `Cow<CStr>` local converts the enum's field
    // glue, whose `Owned` field is declared as the projection
    // `<CStr as ToOwned>::Owned`: the glue builder resolves
    // the field type to `CString` from body evidence and asks
    // for its destructor body, but the demand walk cannot
    // substitute the projection, so `CString` only became a
    // drop candidate when some other body dropped one
    // directly. With no plain `CString` local anywhere in
    // this guest, the drop used to stop the conversion with
    // `unsupported terminator: Drop`.
    run_ok(
        "cow-cstr-borrowed-drop",
        COW_CSTR_BORROWED_DROP_SRC,
        "borrowed len=3\nafter borrowed drop\n",
    );
}

const COW_CSTR_OWNED_DROP_SRC: &str = r#"
use std::borrow::Cow;
use std::ffi::{CStr, CString};
fn main() {
    {
        let owned: Cow<'_, CStr> = Cow::Owned(CString::new("hello").unwrap());
        println!("owned len={}", owned.to_bytes().len());
    }
    println!("after owned drop");
    let explicit: Cow<'_, CStr> = Cow::Owned(CString::new("hi").unwrap());
    drop(explicit);
    println!("after explicit drop");
}
"#;

#[test]
fn run_mode_cstring_into_boxed_c_str_keeps_its_bytes() {
    // `CString::into_boxed_c_str` has no MIR body: the box
    // holds the C string's bytes themselves, terminator
    // included. Storing the fat reference instead left the
    // place consumers (indexing `to_bytes()`, iterating
    // it) projecting into a reference value, while the
    // length-trusting and byte-scanning consumers read
    // through it and hid the defect.
    run_ok(
        "cstring-into-boxed-cstr",
        INTO_BOXED_CSTR_SRC,
        "bytes=[104, 101, 108, 108, 111]\nwith_nul=[104, 101, 108, 108, 111, 0]\ndebug=\"hello\"\nfirst=104 len=5\nderef=[104, 101, 108, 108, 111]\nempty=0 1\nnonascii=[120, 128, 121, 0]\nlossy_len=3\neq=true ord=Greater\n",
    );
}

const CSTRING_FROM_BOXED_CSTR_SRC: &str = r#"
use std::ffi::CString;
fn main() {
    let b = CString::new("world").unwrap().into_boxed_c_str();
    let back: CString = CString::from(b);
    println!("back={:?}", back.to_bytes());
    println!("debug={:?}", back);
    let b2: Box<std::ffi::CStr> = Box::from(CString::new("abc").unwrap());
    let back2: CString = b2.into();
    println!("back2={:?} len={}", back2.to_bytes(), back2.to_bytes_with_nul().len());
    let empty: CString = CString::from(CString::new("").unwrap().into_boxed_c_str());
    println!("empty={} {}", empty.to_bytes().len(), empty.to_bytes_with_nul().len());
}
"#;

#[test]
fn run_mode_cstring_from_boxed_cstr_round_trips() {
    // `CString: From<Box<CStr>>` is `into_c_string`: the
    // owned C string over the box's bytes. The generic
    // conversion had no arm for it and stopped as not
    // runtime-supported.
    run_ok(
        "cstring-from-boxed-cstr",
        CSTRING_FROM_BOXED_CSTR_SRC,
        "back=[119, 111, 114, 108, 100]\ndebug=\"world\"\nback2=[97, 98, 99] len=4\nempty=0 1\n",
    );
}

#[test]
fn run_mode_cow_cstr_owned_drop_converts() {
    // The owned variant of the same drop: the `CString` is
    // built in expression position and moved into the `Cow`,
    // so no body ever drops a `CString` place directly and
    // the destructor demand again depends on seeing through
    // the `ToOwned::Owned` projection — the demand pass now
    // scans the compiler's own `drop_in_place` shim body for
    // `Cow<CStr>`, whose `Drop` terminator names the `CString`
    // field place directly. The explicit `std::mem::drop`
    // call exercises the same glue without a `Drop`
    // terminator in the guest body.
    run_ok(
        "cow-cstr-owned-drop",
        COW_CSTR_OWNED_DROP_SRC,
        "owned len=5\nafter owned drop\nafter explicit drop\n",
    );
}

const BOXED_CSTR_FROM_REF_ACCESS_SRC: &str = r#"
use std::ffi::{CStr, CString};
fn main() {
    let b: Box<CStr> = Box::from(&*CString::new("abc").unwrap());
    println!("bytes={:?}", b.to_bytes());
    println!("first={}", b.to_bytes()[0]);
    println!("sub={:?}", &b.to_bytes()[1..]);
    println!("iter_sum={}", b.to_bytes().iter().map(|x| *x as u32).sum::<u32>());
    println!("with_nul={:?}", b.to_bytes_with_nul());
    println!("debug={:?}", b);
    let d: &CStr = &b;
    println!("deref={:?}", d.to_bytes());
    let e: Box<CStr> = Box::from(&*CString::new("").unwrap());
    println!("empty={} {}", e.to_bytes().len(), e.to_bytes_with_nul().len());
}
"#;

#[test]
fn run_mode_boxed_cstr_from_ref_element_access() {
    // `From<&CStr> for Box<CStr>`: the box holds a copy of
    // the borrowed string's bytes, terminator included --
    // not the fat reference itself. Boxing the reference
    // (the generic handler's catch-all move) left the
    // allocation holding a reference value, and every
    // element read of the boxed string (indexing
    // `to_bytes()`, subslicing it, iterating it) stopped
    // in the index projection on that allocation.
    run_ok(
        "boxed-cstr-from-ref-access",
        BOXED_CSTR_FROM_REF_ACCESS_SRC,
        "bytes=[97, 98, 99]\nfirst=97\nsub=[98, 99]\niter_sum=294\nwith_nul=[97, 98, 99, 0]\ndebug=\"abc\"\nderef=[97, 98, 99]\nempty=0 1\n",
    );
}

const RC_CSTR_FROM_BOXED_SRC: &str = r#"
use std::ffi::{CStr, CString};
use std::rc::Rc;
fn main() {
    let b: Box<CStr> = Box::from(CString::new("abc").unwrap());
    let r: Rc<CStr> = Rc::from(b);
    println!("bytes={:?}", r.to_bytes());
    println!("first={}", r.to_bytes()[0]);
    println!("debug={:?}", r);
    let b2: Box<CStr> = Box::from(&*CString::new("xy").unwrap());
    let r2: Rc<CStr> = b2.into();
    println!("from_ref_src={:?} len={}", r2.to_bytes(), r2.to_bytes().len());
    let empty: Rc<CStr> = Rc::from(CString::new("").unwrap().into_boxed_c_str());
    println!("empty={} {}", empty.to_bytes().len(), empty.to_bytes_with_nul().len());
    println!("count={}", Rc::strong_count(&r));
}
"#;

#[test]
fn run_mode_rc_cstr_from_boxed_cstr_reads_its_bytes() {
    // `Rc<CStr>: From<Box<CStr>>`: the counted handle
    // takes over the box's byte value, so the handle's
    // element reads work for every box source (owned,
    // borrowed, `into_boxed_c_str`).
    run_ok(
        "rc-cstr-from-boxed-cstr",
        RC_CSTR_FROM_BOXED_SRC,
        "bytes=[97, 98, 99]\nfirst=97\ndebug=\"abc\"\nfrom_ref_src=[120, 121] len=2\nempty=0 1\ncount=1\n",
    );
}

const ARC_CSTR_FROM_BOXED_SRC: &str = r#"
use std::ffi::{CStr, CString};
use std::sync::Arc;
fn main() {
    let b: Box<CStr> = Box::from(CString::new("abc").unwrap());
    let a: Arc<CStr> = Arc::from(b);
    println!("bytes={:?}", a.to_bytes());
    println!("first={}", a.to_bytes()[0]);
    println!("debug={:?}", a);
    let b2: Box<CStr> = Box::from(&*CString::new("xy").unwrap());
    let a2: Arc<CStr> = b2.into();
    println!("from_ref_src={:?} len={}", a2.to_bytes(), a2.to_bytes().len());
    let empty: Arc<CStr> = Arc::from(CString::new("").unwrap().into_boxed_c_str());
    println!("empty={} {}", empty.to_bytes().len(), empty.to_bytes_with_nul().len());
    println!("count={}", Arc::strong_count(&a));
}
"#;

#[test]
fn run_mode_arc_cstr_from_boxed_cstr_reads_its_bytes() {
    // `Arc<CStr>: From<Box<CStr>>`: the `Arc` twin of the
    // counted-handle takeover above.
    run_ok(
        "arc-cstr-from-boxed-cstr",
        ARC_CSTR_FROM_BOXED_SRC,
        "bytes=[97, 98, 99]\nfirst=97\ndebug=\"abc\"\nfrom_ref_src=[120, 121] len=2\nempty=0 1\ncount=1\n",
    );
}
