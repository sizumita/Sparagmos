// Chained cast and projection-step tests.

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

const DEBUG_NESTED_BOX_SRC: &str = r#"enum Opt<T> { Empty, Val { val: T } }
struct Node { next: Opt<Box<Box<Box<Node>>>>, value: usize }
struct Pair(usize, usize);
fn main() {
    let deep = Box::new(Box::new(Box::new(Node {
        next: Opt::Val { val: Box::new(Box::new(Box::new(Node { next: Opt::Empty, value: 31 }))) },
        value: 30 })));
    let pair = Box::new(Box::new(Pair(41, 42)));
    zzz();
    let _ = &pair;
}
fn zzz() {}
"#;

#[test]
fn debug_mode_field_steps_deref_nested_boxes() {
    // A field step dereferences implicitly through every
    // smart-pointer layer, the reading gdb applies: the
    // debuginfo recursive-struct record's anonymous-types
    // local wraps its struct in five `Box` layers, and both
    // the named field and the enum-variant field behind them
    // must resolve (a single-layer dereference stops at the
    // next `Box`, whose shape has no such field). Positional
    // fields take the same path.
    let dir = temp_dir("debug-nested-box");
    let src = fixture(&dir, "nested_box.rs", DEBUG_NESTED_BOX_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "break 9\nrun\nprint deep.value\nprint deep.next.val.value\nprint pair.1\nprint pair.0\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["$1 = 30", "$2 = 31", "$3 = 42", "$4 = 41"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}
