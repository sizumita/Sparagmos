// Default construction tests.


const COMPOSITE_DEFAULT_SRC: &str = r#"
#[derive(Debug, Default)]
struct Point {
    x: u32,
    y: bool,
}

#[derive(Debug, Default)]
struct Outer {
    inner: Point,
    n: u32,
}

fn main() {
    let arr: [Point; 3] = Default::default();
    println!("arr={:?}", arr);
    let pair: (Point, u32) = Default::default();
    println!("pair=({:?}, {})", pair.0, pair.1);
    let boxed: Box<Outer> = Default::default();
    println!("boxed={:?}", boxed);
}
"#;

#[test]
fn run_mode_default_for_composites_over_guest_adts() {
    // The `Default` impls of arrays, tuples and `Box` live in
    // core / alloc, so their calls reach the interpreter's
    // name fallback with the composite tag, whose elements
    // are guest structs: the model fabricates each element's
    // derived `Default` from the instance shape the driver
    // recorded for it. The expected lines are the native
    // 1.100.0 output of the same source.
    run_ok(
        "composite-default",
        COMPOSITE_DEFAULT_SRC,
        "arr=[Point { x: 0, y: false }, Point { x: 0, y: false }, Point { x: 0, y: false }]\npair=(Point { x: 0, y: false }, 0)\nboxed=Outer { inner: Point { x: 0, y: false }, n: 0 }\n",
    );
}

const NESTED_WRAPPER_DEFAULT_SRC: &str = r#"
use std::rc::Rc;
use std::sync::Arc;

#[derive(Default)]
struct Ty {
    v: u32,
}

impl Ty {
    fn get(&self) -> u32 {
        self.v + 7
    }
}

fn main() {
    let ty = <Arc<Rc<Box<Ty>>>>::default();
    println!("val={}", ty.get());
}
"#;

#[test]
fn run_mode_default_for_nested_wrappers_over_guest_adt() {
    // `Arc` / `Rc` / `Box` defaults wrap the inner type's
    // default; the innermost type here is a guest struct,
    // whose derived `Default` the model fabricates from the
    // recorded instance shape (the survey case is
    // tests/ui/self/arbitrary_self_types_nested.rs).
    run_ok(
        "nested-wrapper-default",
        NESTED_WRAPPER_DEFAULT_SRC,
        "val=7\n",
    );
}

const POINTER_DEFAULT_SRC: &str = r#"
use std::cell::Cell;

fn main() {
    let p: *const u32 = Default::default();
    println!("thin null={}", p.is_null());
    let m: *mut u32 = Default::default();
    println!("mut null={}", m.is_null());
    let f: *const [i32; 2] = Default::default();
    println!("arrptr null={}", f.is_null());
    let cell: Cell<*const u32> = Default::default();
    println!("cell null={}", cell.get().is_null());
    let x = 41u32;
    cell.set(&x);
    println!("cell val={}", unsafe { *cell.get() });
    let s: &[i32] = Default::default();
    println!("slice len={} empty={}", s.len(), s.is_empty());
    println!(
        "slice ptr eq={}",
        s.as_ptr() == Vec::<i32>::new().as_ptr() && s.as_ptr() == (&[]).as_ptr()
    );
}
"#;

#[test]
fn run_mode_default_for_pointers_and_slice_refs() {
    // A raw pointer's `Default` is the null pointer, and
    // `&[T]` defaults to the empty slice, in the shape a
    // converted `&[]` constant takes, so its data pointer
    // prices as the element type's dangling address -- the
    // same address an empty `Vec`'s buffer pointer has
    // (tests/ui/consts/zst_no_llvm_alloc.rs asserts exactly
    // that chain natively).
    run_ok(
        "pointer-default",
        POINTER_DEFAULT_SRC,
        "thin null=true\nmut null=true\narrptr null=true\ncell null=true\ncell val=41\nslice len=0 empty=true\nslice ptr eq=true\n",
    );
}