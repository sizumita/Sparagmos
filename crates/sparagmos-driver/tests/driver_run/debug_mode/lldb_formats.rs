// LLDB format handling tests.

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
