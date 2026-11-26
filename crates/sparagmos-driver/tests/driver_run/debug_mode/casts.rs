// Pointer cast reinterpretation tests.

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
