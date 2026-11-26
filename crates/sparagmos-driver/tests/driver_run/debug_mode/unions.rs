// Union field read tests.

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
