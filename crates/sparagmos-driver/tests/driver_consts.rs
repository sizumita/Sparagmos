//! Allocated constants of ADT type: constants whose type is a
//! tuple, struct, enum or nested ADT are materialised from their
//! allocations into structured values, guided by the type layout —
//! the stored enum tag is read with the layout tag scalar's width
//! (not the discriminant type's), a niche-encoded enum's niche is
//! read at the absolute offset the layout records for it, and
//! pattern-typed fields decode as their inner type. Collected
//! std bodies full of such constants (`RawVecInner`, `HashMap`)
//! convert, and programs using ADT constants execute.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Fixture directories this test process created under the
/// system temp directory, removed when the process exits.
/// Tests spawn the driver per case and used to leave every
/// fixture directory behind (#537); the registry keeps the
/// per-test `temp_dir` shape and sweeps at exit instead.
static SCRATCH_DIRS: std::sync::Mutex<Vec<std::path::PathBuf>> =
    std::sync::Mutex::new(Vec::new());

fn register_scratch_dir(dir: &std::path::Path) {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| unsafe {
        let _ = libc::atexit(remove_scratch_dirs);
    });
    if let Ok(mut dirs) = SCRATCH_DIRS.lock() {
        if !dirs.iter().any(|d| d == dir) {
            dirs.push(dir.to_path_buf());
        }
    }
}

extern "C" fn remove_scratch_dirs() {
    if let Ok(dirs) = SCRATCH_DIRS.try_lock() {
        for dir in dirs.iter() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

fn driver() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_sparagmos-driver"))
}

fn write_fixture(name: &str, src: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-consts-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    register_scratch_dir(&dir);
    let path = dir.join(format!("{name}.rs"));
    std::fs::write(&path, src).expect("write fixture");
    path
}

fn run_driver(args: &[&str], fixture: &Path) -> (String, String, std::process::ExitStatus) {
    let out = Command::new(driver())
        .args(args)
        .arg(fixture)
        .env_remove("SPARAGMOS_DUMP")
        .output()
        .expect("spawn driver");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status,
    )
}

fn dump_of(name: &str, src: &str) -> String {
    let fixture = write_fixture(name, src);
    let dump_path = fixture.with_extension("dump.txt");
    let dump_arg = dump_path.to_string_lossy().into_owned();
    let (stdout, _, status) = run_driver(&["--dump", &dump_arg], &fixture);
    assert!(status.success(), "driver failed: {stdout}");
    let dump = std::fs::read_to_string(&dump_path).expect("read dump file");
    // The dump's pretty Debug rendering spreads values over many
    // lines; collapse whitespace so value patterns can be matched.
    dump.split_whitespace().collect::<Vec<_>>().join("")
}

const ENUM_RS: &str = "\
enum Shape { Circle(u32), Rect { w: u32, h: u32 }, Unit }
const S: Shape = Shape::Rect { w: 2, h: 5 };
const C: Shape = Shape::Circle(3);
fn area(s: &Shape) -> u32 {
    match s {
        Shape::Circle(r) => r * r,
        Shape::Rect { w, h } => w * h,
        Shape::Unit => 0,
    }
}
fn main() {
    println!(\"{} {}\", area(&S), area(&C));
}
";

#[test]
fn enum_constants_execute() {
    let fixture = write_fixture("enum_run", ENUM_RS);
    let (stdout, _, status) = run_driver(&["run"], &fixture);
    assert!(status.success(), "run failed: {stdout}");
    assert_eq!(stdout, "10 9\n");
}

const STRUCT_RS: &str = "\
struct Point { x: i32, y: i32 }
const P: Point = Point { x: 3, y: 4 };
const T: (i32, bool) = (7, true);
fn main() {
    println!(\"{} {} {} {}\", P.x + P.y, T.0, T.1 as u8, P.x);
}
";

#[test]
fn struct_and_tuple_constants_execute() {
    let fixture = write_fixture("struct_run", STRUCT_RS);
    let (stdout, _, status) = run_driver(&["run"], &fixture);
    assert!(status.success(), "run failed: {stdout}");
    assert_eq!(stdout, "7 7 1 3\n");
}

const RESULT_RS: &str = "\
const R: Result<i32, u8> = Err(42);
const O: Option<i32> = Some(11);
fn main() {
    let a = match R { Ok(v) => v, Err(e) => e as i32 };
    let b = O.unwrap_or(0);
    println!(\"{} {}\", a, b);
}
";

#[test]
fn result_and_option_constants_execute() {
    let fixture = write_fixture("result_run", RESULT_RS);
    let (stdout, _, status) = run_driver(&["run"], &fixture);
    assert!(status.success(), "run failed: {stdout}");
    assert_eq!(stdout, "42 11\n");
}

const SMALL_ENUM_RS: &str = "\
#[repr(u8)]
enum Small { A = 3, B = 9 }
const B: Small = Small::B;
fn main() {
    println!(\"{}\", match B { Small::A => 1, Small::B => 2 });
}
";

#[test]
fn narrow_tag_enum_constant_decodes_to_its_variant() {
    // A fieldless `#[repr(u8)]` enum stores its tag in one byte
    // although the discriminant type is word-sized; the stored
    // value 9 must resolve to variant index 1 (`B`), not be read
    // as an 8-byte tag.
    let dump = dump_of("small_enum", SMALL_ENUM_RS);
    assert!(
        dump.contains("Adt{variant:1,fields:[]"),
        "Small::B constant not decoded as variant 1:\n{dump}"
    );
}

const NICHE_RS: &str = "\
const R: Result<i32, u8> = Err(42);
const O: Option<i32> = Some(11);
const N: Option<i32> = None;
fn main() {
    println!(\"{}\", match R { Ok(v) => v, Err(e) => e as i32 } + O.unwrap_or(0) + N.unwrap_or(1));
}
";

#[test]
fn result_option_constants_decode_with_payloads() {
    let dump = dump_of("niche_consts", NICHE_RS);
    assert!(
        dump.contains("Adt{variant:1,fields:[Scalar(Int{data:42,size:1"),
        "Err(42) constant not decoded:\n{dump}"
    );
    assert!(
        dump.contains("Adt{variant:1,fields:[Scalar(Int{data:11,size:4"),
        "Some(11) constant not decoded:\n{dump}"
    );
    assert!(
        dump.contains("Adt{variant:0,fields:[]"),
        "None constant not decoded:\n{dump}"
    );
}

const NESTED_RS: &str = "\
struct Inner { v: u8 }
struct Outer { inner: Inner, flag: bool }
const X: Outer = Outer { inner: Inner { v: 9 }, flag: true };
fn main() {
    println!(\"{}\", X.inner.v);
}
";

#[test]
fn nested_adt_constant_decodes() {
    let dump = dump_of("nested", NESTED_RS);
    assert!(
        dump.contains("Adt{variant:0,fields:[Adt{variant:0,fields:[Scalar(Int{data:9,size:1"),
        "nested ADT constant not decoded:\n{dump}"
    );
}

const LAYOUT_RS: &str = "\
use std::alloc::Layout;
const L: Layout = Layout::new::<u32>();
fn main() {
    println!(\"{}\", L.size());
}
";

#[test]
fn layout_constant_converts() {
    // `Layout` wraps an `Alignment`, whose representation is an
    // enum over power-of-two tags; the constant appears throughout
    // the collected allocator bodies.
    let fixture = write_fixture("layout", LAYOUT_RS);
    let (stdout, _, status) = run_driver(&[], &fixture);
    assert!(status.success(), "driver failed: {stdout}");
    assert!(
        stdout.contains("fn layout::main: blocks="),
        "main with a Layout constant not converted:\n{stdout}"
    );
    assert!(
        !stdout.contains("ConstantKind::Allocated"),
        "allocated constant still unsupported:\n{stdout}"
    );
}

const WITH_CAPACITY_RS: &str = "\
use std::collections::VecDeque;
fn main() {
    let d: VecDeque<i32> = VecDeque::with_capacity(4);
    println!(\"{}\", d.len());
}
";

#[test]
fn collected_rawvec_bodies_convert() {
    // The collected `RawVecInner` bodies carry the constants this
    // lane is about: a fieldless `AllocInit` (1-byte tag), the
    // pattern-typed capacity wrapper, and niche-encoded
    // `Result<Layout, LayoutError>` / `TryReserveError` values.
    // The route in is `VecDeque::with_capacity`, which builds its
    // buffer through `RawVecInner`; `Vec::with_capacity` itself is
    // an interpreter builtin (the abstract Vec model), so its body
    // is not collected.
    let fixture = write_fixture("with_capacity", WITH_CAPACITY_RS);
    let (stdout, _, status) = run_driver(&[], &fixture);
    assert!(status.success(), "driver failed: {stdout}");
    assert!(
        stdout.contains("fn alloc::raw_vec::RawVecInner::<A>::try_allocate_in: blocks="),
        "try_allocate_in not converted:\n{stdout}"
    );
    assert!(
        !stdout.contains("ConstantKind::Allocated"),
        "allocated constant still unsupported:\n{stdout}"
    );
}

const HASHMAP_RS: &str = "\
use std::collections::HashMap;
fn main() {
    let m: HashMap<String, i32> = HashMap::new();
    println!(\"{}\", m.len());
}
";

#[test]
fn collected_hashmap_new_converts() {
    // `HashMap::new` reaches a `hashbrown::raw::RawTableInner`
    // constant (a struct wrapping niche-typed fields).
    let fixture = write_fixture("hashmap", HASHMAP_RS);
    let (stdout, _, status) = run_driver(&[], &fixture);
    assert!(status.success(), "driver failed: {stdout}");
    assert!(
        stdout.contains("fn std::collections::HashMap::<K, V>::new: blocks="),
        "HashMap::new not converted:\n{stdout}"
    );
    assert!(
        !stdout.contains("ConstantKind::Allocated"),
        "allocated constant still unsupported:\n{stdout}"
    );
}

// ----- Lane CE: by-reference constant shapes ---------------------

fn run_ok(name: &str, src: &str) {
    let fixture = write_fixture(name, src);
    let (stdout, stderr, status) = run_driver(&["run"], &fixture);
    assert!(
        status.success(),
        "run failed ({status:?}): {stdout} {stderr}"
    );
    assert!(
        stdout.contains("ok"),
        "expected ok output, got: {stdout} {stderr}"
    );
}

const NEG_DISCR_RS: &str = "\
#[derive(PartialEq, Debug)]
enum E { A = -1, B = 0, C = 1 }
const X: &E = &E::A;
const Y: &E = &E::C;
fn main() {
    assert_eq!(*X, E::A);
    assert_eq!(*Y, E::C);
    assert_ne!(*X, *Y);
    println!(\"ok\");
}
";

#[test]
fn negative_discriminant_reference_constant_runs() {
    // The stored tag of a negative discriminant is the bit pattern
    // in the discriminant type's own width; matching it against the
    // zero-extended `Discriminant::val` used to fail every such
    // constant into the by-reference fallback.
    run_ok("neg_discr", NEG_DISCR_RS);
}

const FN_PTR_TABLE_RS: &str = "\
struct VTable { make: fn() -> u32, size: usize }
fn mk() -> u32 { 7 }
const VT: &VTable = &VTable { make: mk, size: 3 };
fn main() {
    assert_eq!((VT.make)(), 7);
    assert_eq!(VT.size, 3);
    println!(\"ok\");
}
";

#[test]
fn fn_pointer_table_constant_runs() {
    // A vtable-shaped static: an ADT constant behind a reference
    // whose fields are function pointers (the RawWakerVTable shape).
    run_ok("fn_table", FN_PTR_TABLE_RS);
}

const FN_UNDER_CAST_RS: &str = "\
fn target() {}
const P: *const u8 = target as *const u8;
const Q: *const u8 = target as *const u8;
fn main() {
    assert_eq!(P, Q);
    assert!(!P.is_null());
    println!(\"ok\");
}
";

#[test]
fn function_pointer_under_cast_constant_runs() {
    // A pointer to a function item stored under a data-pointer
    // cast: the pointee is a function allocation, and the value is
    // the callable itself.
    run_ok("fn_cast", FN_UNDER_CAST_RS);
}

const DANGLING_RS: &str = "\
use std::ptr::NonNull;
const D: NonNull<u8> = NonNull::dangling();
const D2: NonNull<u8> = NonNull::dangling();
fn main() {
    assert_eq!(D, D2);
    println!(\"ok\");
}
";

#[test]
fn dangling_pointer_constant_runs() {
    // A dangling pointer constant has no provenance; the stored
    // integer is the address itself.
    run_ok("dangling", DANGLING_RS);
}

const TYPEID_RS: &str = "\
use std::any::TypeId;
fn main() {
    assert_eq!(TypeId::of::<u32>(), TypeId::of::<u32>());
    assert_ne!(TypeId::of::<u32>(), TypeId::of::<u64>());
    println!(\"ok\");
}
";

#[test]
fn type_id_constant_runs() {
    run_ok("typeid", TYPEID_RS);
}

const STR_NEWTYPE_RS: &str = "\
#[repr(transparent)]
struct Wrapper(str);
const W: &Wrapper = unsafe { &*(\"abc\" as *const str as *const Wrapper) };
fn main() {
    assert_eq!(W.0.len(), 3);
    println!(\"ok\");
}
";

#[test]
fn trailing_str_field_constant_runs() {
    // An ADT whose trailing field is unsized `str`: the field's
    // bytes run to the end of the allocation.
    run_ok("str_newtype", STR_NEWTYPE_RS);
}

// ----- Lane DR: multi-payload unions, vtable / fat pointer constants, f128 -----

const UNION_MULTI_RS: &str = "\
union U { a: u32, b: f32 }
const V: U = U { a: 0x3fc0_0000 };
const W: U = U { b: 2.5 };
fn main() {
    unsafe {
        assert_eq!(V.a, 0x3fc0_0000);
        assert_eq!(V.b, 1.5);
        assert_eq!(W.b, 2.5);
        assert_eq!(W.a, 0x4020_0000);
    }
    println!(\"ok\");
}
";

#[test]
fn multi_payload_union_constant_runs() {
    // A union with two payload fields: the allocation does not say
    // which field is live, but the value model carries every field
    // and a field read reinterprets the same bytes, so both reads
    // are faithful.
    run_ok("union_multi", UNION_MULTI_RS);
}

const UNION_PTR_RS: &str = "\
union P { p: *const u8, n: usize }
static X: u8 = 9;
const PU: P = P { p: &X };
fn main() {
    unsafe {
        assert_eq!(*PU.p, 9);
    }
    println!(\"ok\");
}
";

#[test]
fn union_with_pointer_field_constant_runs() {
    // The provenance entry at the union's offset identifies the
    // pointer field as the stored one; the overlapping integer
    // field decodes as uninitialised rather than as a fabricated
    // reinterpretation of the pointer's bytes.
    run_ok("union_ptr", UNION_PTR_RS);
}

const DANGLING_SLICE_RS: &str = "\
const P: *const [u16] = unsafe { std::slice::from_raw_parts(0x10 as *const u16, 3) };
fn main() {
    assert_eq!(P.len(), 3);
    assert_eq!((P as *const u16) as usize, 0x10);
    println!(\"ok\");
}
";

#[test]
fn dangling_slice_constant_runs() {
    // A provenance-less fat pointer constant: there is no pointee,
    // but the address identity and the length are exactly what the
    // program observes. (A dangling slice *reference* constant is
    // rejected by rustc's own validity check, so the raw-pointer
    // form is the shape that reaches the decoder.)
    run_ok("dangling_slice", DANGLING_SLICE_RS);
}

const VTABLE_CONST_RS: &str = "\
trait T { fn m(&self) -> u8; }
impl T for u32 { fn m(&self) -> u8 { 1 } }
impl T for i32 { fn m(&self) -> u8 { 2 } }
const VT1: [*const (); 2] = unsafe {
    std::mem::transmute::<&dyn T, [*const (); 2]>(&42_u32 as &dyn T)
};
const VT1B: [*const (); 2] = unsafe {
    std::mem::transmute::<&dyn T, [*const (); 2]>(&7_u32 as &dyn T)
};
const VT2: [*const (); 2] = unsafe {
    std::mem::transmute::<&dyn T, [*const (); 2]>(&42_i32 as &dyn T)
};
fn main() {
    assert!(!VT1[1].is_null());
    assert_eq!(VT1[1], VT1B[1]);
    assert_ne!(VT1[1], VT2[1]);
    println!(\"ok\");
}
";

#[test]
fn vtable_pointer_constant_runs() {
    // A vtable pointer decodes to a stable identity for the
    // (concrete type, trait) pair: the same pair compares equal,
    // different pairs compare unequal.
    run_ok("vtable_const", VTABLE_CONST_RS);
}

const F128_CONST_RS: &str = "\
#![feature(f128)]
const A: f128 = 1.5;
const B: f128 = 2.25;
fn main() {
    assert!(A + A == 3.0f128);
    assert!(A * B == 3.375f128);
    assert!((A * B) as f64 == 3.375f64);
    assert!(A < B);
    assert!(B.sqrt() == 1.5f128);
    assert!(A.to_bits() == 0x3fff_8000_0000_0000_0000_0000_0000_0000u128);
    println!(\"ok\");
}
";

#[test]
fn f128_constant_runs() {
    // f128 constants decode to their raw binary128 bit pattern and
    // execute in the exact software arithmetic; nothing is rounded
    // through f64.
    run_ok("f128_const", F128_CONST_RS);
}

// ----- Target pointer width in allocated constants -------------

/// Convert a fixture for a foreign `target` in summary mode.
/// `None` when the target is not installed in the toolchain
/// this test runs against (its conversion cannot be judged
/// there), so the caller skips instead of failing.
fn convert_with_target(name: &str, src: &str, target: &str) -> Option<String> {
    let fixture = write_fixture(name, src);
    let (stdout, stderr, status) = run_driver(
        &["--rustc-arg", "--target", "--rustc-arg", target],
        &fixture,
    );
    if !status.success()
        || stdout.contains("compilation failed")
        || stderr.contains("can't find crate")
    {
        return None;
    }
    Some(stdout)
}

const STATIC_PTR_RS: &str = "\
static mut BAR: u32 = 12;
static mut FOO: fn(&usize, &mut u32) -> Result<(), ()> = bar_fn;
fn bar_fn(ptr: &usize, _: &mut u32) -> Result<(), ()> { let _ = *ptr; Ok(()) }
pub fn get_bar() -> u32 { unsafe { BAR } }
pub fn call_foo() { let mut buf = 7u32; unsafe { FOO(&1, &mut buf); } }
fn main() { println!(\"{}\", get_bar()); call_foo(); }
";

#[test]
fn static_pointer_constants_convert_on_32bit_target() {
    // Reading a `static mut` in MIR goes through a constant of
    // type `*mut T` naming the static. On a 32-bit target the
    // allocation stores the pointer in 4 bytes; decoding it at
    // the host's 8-byte pointer width failed the constant, and
    // every function touching the static was skipped
    // (`ConstantKind::Allocated (type ... RawPtr(Uint(U32)))`
    // and `... RawPtr(FnPtr ...)`).
    let Some(stdout) = convert_with_target("target_ptr32", STATIC_PTR_RS, "i686-unknown-linux-gnu")
    else {
        return;
    };
    assert!(
        stdout.contains("fn target_ptr32::get_bar: blocks="),
        "get_bar not converted on a 32-bit target:\n{stdout}"
    );
    assert!(
        stdout.contains("fn target_ptr32::call_foo: blocks="),
        "call_foo not converted on a 32-bit target:\n{stdout}"
    );
    assert!(
        !stdout.contains("ConstantKind::Allocated"),
        "allocated constant still unsupported on a 32-bit target:\n{stdout}"
    );
}

const FOREIGN_PTR_RS: &str = "\
unsafe extern \"C\" { safe static VAL: *mut (); }
pub fn get_class() -> *mut () { VAL }
fn main() { println!(\"{:?}\", get_class()); }
";

#[test]
fn foreign_static_pointer_constant_converts_on_32bit_target() {
    // Reading a foreign static goes through a constant of type
    // `*mut *mut ()` naming it: the nested pointer word is
    // target-sized too, so the same host-width decode failed
    // it on 32-bit targets.
    let Some(stdout) =
        convert_with_target("foreign_ptr32", FOREIGN_PTR_RS, "i686-unknown-linux-gnu")
    else {
        return;
    };
    assert!(
        stdout.contains("fn foreign_ptr32::get_class: blocks="),
        "get_class not converted on a 32-bit target:\n{stdout}"
    );
    assert!(
        !stdout.contains("ConstantKind::Allocated"),
        "allocated constant still unsupported on a 32-bit target:\n{stdout}"
    );
}

const MDROP_PTR_RS: &str = "\
use std::mem::ManuallyDrop;
static mut X: u32 = 3;
const M: ManuallyDrop<*mut u32> = ManuallyDrop::new(std::ptr::addr_of_mut!(X));
fn main() { unsafe { println!(\"{}\", **M); } }
";

#[test]
fn manually_drop_pointer_constant_converts_on_32bit_target() {
    // A `ManuallyDrop` wrapping a raw-pointer constant: the
    // pointer field inside the ADT is a target-sized word, so
    // the aggregate decode needs the target width as well.
    let Some(stdout) = convert_with_target("mdrop_ptr32", MDROP_PTR_RS, "i686-unknown-linux-gnu")
    else {
        return;
    };
    assert!(
        stdout.contains("fn mdrop_ptr32::main: blocks="),
        "main not converted on a 32-bit target:\n{stdout}"
    );
    assert!(
        !stdout.contains("ConstantKind::Allocated"),
        "allocated constant still unsupported on a 32-bit target:\n{stdout}"
    );
}
