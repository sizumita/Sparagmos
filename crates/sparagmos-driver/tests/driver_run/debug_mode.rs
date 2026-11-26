use super::{Command, Output, Path, fixture, temp_dir};

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

const DEBUG_FAT_PTR_FORMS_SRC: &str = r#"struct Foo<T: ?Sized> {
    value: T,
}

fn main() {
    let foo: Foo<Foo<[u8; 4]>> = Foo { value: Foo { value: *b"abc\0" } };
    let a: &Foo<[u8]> = &foo.value;
    let b: &Foo<Foo<[u8]>> = &foo;
    let c: &Foo<dyn std::fmt::Debug> = &Foo { value: 7i32 };
    let _box: Box<Foo<dyn std::fmt::Debug>> = Box::new(Foo { value: 8i32 });
    zzz();
}

fn zzz() {}
"#;

#[test]
fn debug_mode_unsized_fat_pointer_forms() {
    // gdb prints a fat pointer as the struct its DWARF
    // describes -- the pointer type's name, then the metadata
    // fields (`data_ptr`/`length` for a slice tail,
    // `pointer`/`vtable` for a trait-object tail) -- not as a
    // bare address. The addresses themselves are the model's
    // synthetic ones, so the assertions pin the form around
    // them.
    let dir = temp_dir("debug-fat-ptr-forms");
    let src = fixture(&dir, "fat_ptr_forms.rs", DEBUG_FAT_PTR_FORMS_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 11\nrun\nprint a\nprint b\nprint c\nprint _box\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "&fat_ptr_forms::Foo<[u8]> {data_ptr: 0x",
        "&fat_ptr_forms::Foo<fat_ptr_forms::Foo<[u8]>> {data_ptr: 0x",
        ", length: 4}",
        "&fat_ptr_forms::Foo<dyn core::fmt::Debug> {pointer: 0x",
        ", vtable: 0x",
        "alloc::boxed::Box<fat_ptr_forms::Foo<dyn core::fmt::Debug>, \
         alloc::alloc::Global> {pointer: 0x",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}

const DEBUG_STR_TEXT_SRC: &str = r#"struct Holder<'a> {
    inner: &'a str,
}

fn main() {
    let plain: &str = "Hello";
    let holder = Holder { inner: "Hello" };
    let pair = ("Hello", "World");
    let slice: &[u8] = &[1, 2, 3];
    zzz();
}

fn zzz() {}
"#;

#[test]
fn debug_mode_str_values_render_as_text() {
    // gdb prints a `str` value -- a `&str` local, a `&str`
    // field, a tuple of them -- as its quoted text, in both
    // of the value forms the model gives it (the text itself,
    // or a fat reference whose base is a snapshot of the
    // text). A slice reference prints in the fat-pointer
    // struct form.
    let dir = temp_dir("debug-str-text");
    let src = fixture(&dir, "str_text.rs", DEBUG_STR_TEXT_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 10\nrun\nprint plain\nprint holder\nprint pair\nprint slice\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "$1 = \"Hello\"",
        "str_text::Holder {inner: \"Hello\"}",
        "$3 = (\"Hello\", \"World\")",
        "&[u8] {data_ptr: 0x",
        ", length: 3}",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}

fn debug_driver_g(dir: &Path, src: &Path, script: &str, dialect: &str) -> Output {
    let script_path = dir.join("session.script");
    std::fs::write(&script_path, script).unwrap();
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("debug")
        .arg(src)
        .arg("--edition")
        .arg("2021")
        .arg("--rustc-arg")
        .arg("-g")
        .arg("--rustc-arg")
        .arg("-Zmir-opt-level=0")
        .arg("--debug-script")
        .arg(&script_path)
        .arg("--debug-dialect")
        .arg(dialect)
        .output()
        .expect("failed to run sparagmos-driver")
}

const DEBUG_BOX_DST_SRC: &str = r#"struct Foo<T: ?Sized> {
    a: u32,
    value: T,
}
struct Holder {
    p: *const Foo<[u8]>,
}
fn main() {
    let box_str: Box<str> = "World".into();
    let box_slice: Box<[u8]> = Box::new([10u8, 20, 30]);
    let box_dst: Box<Foo<[u8]>> = Box::new(Foo { a: 7, value: [1u8, 2] });
    let rc_str: std::rc::Rc<str> = "World".into();
    let str_in_rc: std::rc::Rc<&str> = std::rc::Rc::new("Hello");
    let raw_slice: *const [u8] = &[9u8, 8, 7];
    let raw_dst: *const Foo<[u8]> = &Foo { a: 3, value: [5u8, 6, 7, 8] };
    let holder = Holder { p: raw_dst };
    zzz();
    let _ = (box_str, box_slice, box_dst, rc_str, str_in_rc, raw_slice, raw_dst, holder);
}
fn zzz() {}
"#;

#[test]
fn debug_mode_box_unsized_pointees_print_the_pointee() {
    // gdb prints a fat pointer's pointee, not its metadata:
    // a `Box<str>` as the box's name over the text's bytes,
    // a `Box<[T]>` over the elements, a `Box` of a
    // dynamically sized struct and a raw pointer to one as
    // the struct itself (sized fields by value, the unsized
    // tail as an address), and a raw slice pointer as its
    // name over the elements. An `Rc<str>`'s inner pointer
    // prints the `RcInner<str>` struct the same way, while
    // a sized pointee (`RcInner<&str>`) stays an address;
    // both print `NonNull` by its definition path.
    let dir = temp_dir("debug-box-dst");
    let src = fixture(&dir, "box_str_dst.rs", DEBUG_BOX_DST_SRC);
    let out = debug_driver_g(
        &dir,
        &src,
        "break 17\nrun\nprint box_str\nprint box_slice\nprint box_dst\nprint rc_str\nprint str_in_rc\nprint raw_slice\nprint holder\n",
        "gdb",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "alloc::boxed::Box<str, alloc::alloc::Global> [87, 111, 114, 108, 100]",
        "alloc::boxed::Box<[u8], alloc::alloc::Global> [10, 20, 30]",
        "box_str_dst::Foo<[u8]> {a: 7, value: 0x",
        "core::ptr::non_null::NonNull<alloc::rc::RcInner<str>> {pointer: alloc::rc::RcInner<str> {strong: core::cell::Cell<usize> {value: core::cell::UnsafeCell<usize> {value: 1}}",
        "core::ptr::non_null::NonNull<alloc::rc::RcInner<&str>> {pointer: 0x",
        "*const [u8] [9, 8, 7]",
        "p: box_str_dst::Foo<[u8]> {a: 3, value: 0x",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
    assert!(
        !stdout.contains("RcInner<&str> {strong"),
        "a sized pointee must stay an address, not the struct: {stdout}"
    );
}

#[test]
fn debug_mode_box_str_lldb_provider_line() {
    // The lldb provider line for a `Box` over an unsized
    // pointee shows the pointee's own summary under the
    // box's type name: the text and its character children
    // for `str`, the size and element children for a slice.
    let dir = temp_dir("debug-box-lldb");
    let src = fixture(&dir, "box_str_dst.rs", DEBUG_BOX_DST_SRC);
    let out = debug_driver_g(
        &dir,
        &src,
        "break 17\nrun\nprint box_str\nprint box_slice\n",
        "lldb",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "(alloc::boxed::Box<str, alloc::alloc::Global>) box_str = \"World\" { [0] = 'W' [1] = 'o' [2] = 'r' [3] = 'l' [4] = 'd' }",
        "box_slice = size=3 { [0] = 10 [1] = 20 [2] = 30 }",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}
