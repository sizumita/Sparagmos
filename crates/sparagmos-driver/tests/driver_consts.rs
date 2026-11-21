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

fn driver() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_sparagmos-driver"))
}

fn write_fixture(name: &str, src: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-consts-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
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
