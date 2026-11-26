// whatis and type-name query tests.

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

const DEBUG_FOREIGN_PTR_SRC: &str = r#"#![feature(extern_types)]
extern "C" {
    type Opaque;
}
fn main() {
    let p = 0 as *const Opaque;
    let q = p as *mut Opaque;
    zzz();
    let _ = (p, q);
}
fn zzz() {}
"#;

#[test]
fn debug_mode_foreign_type_pointers_convert_and_answer_whatis() {
    // A foreign (extern) type is opaque and unsized: no value
    // of it can be formed, but pointers to it are ordinary
    // thin pointers, and a body that only forms such pointers
    // (here, null casts) must convert and run in a debugger
    // session. Conversion used to refuse the type outright
    // ("unsupported type in local ... Foreign (...)"), which
    // held back the debuginfo suite's type-names record at
    // the session stage.
    let dir = temp_dir("debug-foreign-ptr");
    let src = fixture(&dir, "foreign_ptr.rs", DEBUG_FOREIGN_PTR_SRC);
    let out = debug_driver_bootstrapped(
        &dir,
        &src,
        "break 8\nrun\nwhatis p\nwhatis q\nprint p\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "type = *const foreign_ptr::Opaque",
        "type = *mut foreign_ptr::Opaque",
        "$1 = 0x0",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}

const DEBUG_TYPE_NAMES_SRC: &str = r#"
use std::marker::PhantomData;
struct Struct1;
struct GenericStruct<T1, T2>(PhantomData<(T1, T2)>);
mod mod1 {
    pub struct Struct2;
}
trait Trait2<T1, T2> {
    fn dummy(&self, _: T1, _: T2) {}
}
impl<T1, T2> Trait2<T1, T2> for isize {}
fn rust_fn(_: Option<isize>) {}
extern "C" fn extern_c_fn(_: isize) {}
fn main() {
    let generic_struct2: GenericStruct<
        Struct1,
        extern "system" fn(isize) -> usize,
    > = GenericStruct(PhantomData);
    let generic_box_trait =
        Box::new(0isize) as Box<dyn Trait2<i32, mod1::Struct2>>;
    let rust_fn = (rust_fn, 0usize);
    let extern_c_fn = (extern_c_fn, 0usize);
    let closure1 = (|x: isize| {}, 0usize);
    let closure2 = (|x: i8, y: f32| (x as f32) + y, 0usize);
    zzz();
}
#[inline(never)]
fn zzz() {
    ()
}
"#;

#[test]
fn debug_mode_type_names_display_fidelity() {
    // The debuginfo type-names record's display fidelity: a
    // function pointer renders with its full signature, a
    // `dyn Trait<args>` keeps the trait's generic arguments,
    // a function item's type is its fn-pointer spelling from
    // the display-name table, and a closure's type is its
    // environment name with no generic arguments.
    let dir = temp_dir("debug-type-names");
    let src = fixture(&dir, "type_names.rs", DEBUG_TYPE_NAMES_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 25\nrun\nwhatis generic_struct2\nwhatis generic_box_trait\nwhatis rust_fn\nwhatis extern_c_fn\nwhatis closure1\nwhatis closure2\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "type = type_names::GenericStruct<type_names::Struct1, extern \"system\" fn(isize) -> usize>",
        "type = alloc::boxed::Box<dyn type_names::Trait2<i32, type_names::mod1::Struct2>, alloc::alloc::Global>",
        "type = (fn(core::option::Option<isize>), usize)",
        "type = (extern \"C\" fn(isize), usize)",
        "type = (type_names::main::{closure_env#0}, usize)",
        "type = (type_names::main::{closure_env#1}, usize)",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}
