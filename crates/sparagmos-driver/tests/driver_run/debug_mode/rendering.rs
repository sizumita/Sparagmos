// Value rendering and pretty-printer tests.

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

const DEBUG_PRETTY_STD_SRC: &str = r#"use std::ffi::OsString;
struct Holder {
    text: String,
}
fn main() {
    let os_string = OsString::from("IAMA OS string \u{1F603}");
    let some_string = Some("IAMA optional string!".to_owned());
    let holder = Holder { text: "held".to_owned() };
    zzz();
    let _ = (&os_string, &some_string, &holder);
}
fn zzz() {}
"#;

#[test]
fn debug_mode_pretty_std_os_string_and_option_string() {
    // The pretty-std shapes: an `OsString` renders as its
    // text (its recorded type names the re-export path
    // `std::ffi::OsString`, not the definition path the
    // provider classification was written against), an
    // `Option<String>` payload renders through the `String`
    // provider inside the raw enum form (gdb applies the
    // printers at every level), and a `print elements`
    // limit set mid-session truncates the payload string
    // of a later print.
    let dir = temp_dir("debug-pretty-std");
    let src = fixture(&dir, "pretty_std.rs", DEBUG_PRETTY_STD_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "pretty-printers\nbreak 9\nrun\nprint os_string\nprint some_string\nprint holder\nprint-elements 5\nprint some_string\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "$1 = \"IAMA OS string ",
        "$2 = core::option::Option<alloc::string::String>::Some(\"IAMA optional string!\")",
        "{text: \"held\"}",
        "$4 = core::option::Option<alloc::string::String>::Some(\"IAMA \"...",
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

const DEBUG_HUGE_VEC_SRC: &str = r#"fn main() {
    let mut vec: Vec<u8> = Vec::with_capacity(1_000_000_000);
    unsafe { vec.set_len(1_000_000_000) }
    let slice = &vec[..];
    zzz();
    let _ = slice.len();
}
fn zzz() {}
"#;

#[test]
fn debug_mode_huge_sparse_vec_prints_without_materialising() {
    // The `pretty-huge-vec` shape: a vector whose declared
    // length runs into the billions past its stored elements
    // (the sparse form). Slicing it must not materialise one
    // host slot per element (that allocation exhausts the
    // host), the pretty printers report the declared size with
    // the bounded element block, and the function listing
    // carries the slicing instance: its signature's return
    // type is the declaration's associated-type projection
    // (`Index::Output`), which the display-name computation
    // must normalise to `&[u8]` -- fed to the compiler's
    // debuginfo name machinery unnormalised, the projection
    // is rejected and the instance silently drops out of the
    // table.
    let dir = temp_dir("debug-huge-vec");
    let src = fixture(&dir, "huge_vec.rs", DEBUG_HUGE_VEC_SRC);
    let out = debug_driver(
        &dir,
        &src,
        "pretty-printers\nbreak 5\nrun\nprint vec\nprint slice\ninfo-functions index\n",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "$1 = Vec(size=1000000000) = {",
        "$2 = &[u8](size=1000000000) = {",
        "::index<u8, core::ops::range::RangeFull, alloc::alloc::Global>(\
            &alloc::vec::Vec<u8, alloc::alloc::Global>, core::ops::range::RangeFull) -> &[u8]",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}

const DEBUG_STRING_FORMS_SRC: &str = r#"use std::rc::Rc;

struct Foo {
    inner: &'static str,
}

fn main() {
    let plain_string = String::from("Hello");
    let plain_str = "Hello";
    let str_in_struct = Foo { inner: "Hello" };
    let str_in_tuple = ("Hello", "World");
    let str_in_rc = Rc::new("Hello");
    let box_str: Box<str> = Box::from("World");
    let rc_str: Rc<str> = Rc::from("Hello");
    zzz(); // #break
}
fn zzz() {}
"#;

#[test]
fn debug_mode_string_forms_render_in_gdb() {
    // The gdb forms of the string values: a `String` whose
    // value the model holds as text renders in the DWARF
    // structural aggregate form (with the `non_null`
    // definition path and the niche capacity), a boxed `str`
    // renders as its byte list, and an `Rc<str>` expands its
    // `RcInner` inline with the unsized tail as the data
    // address.
    let dir = temp_dir("debug-string-forms-gdb");
    let src = fixture(&dir, "string_forms.rs", DEBUG_STRING_FORMS_SRC);
    let out = debug_driver_session(
        &dir,
        &src,
        "break 15\nrun\nprint plain_string\nprint str_in_struct\nprint str_in_rc\nprint box_str\nprint rc_str\n",
        "gdb",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "alloc::string::String {vec: alloc::vec::Vec<u8, \
         alloc::alloc::Global> {buf: alloc::raw_vec::RawVec<u8, \
         alloc::alloc::Global> {inner: \
         alloc::raw_vec::RawVecInner<alloc::alloc::Global> {ptr: \
         core::ptr::non_null::NonNull<u8> {pointer: 0x",
        "cap: core::num::niche_types::UsizeNoHighBit (5)",
        "string_forms::Foo {inner: \"Hello\"}",
        "alloc::boxed::Box<str, alloc::alloc::Global> \
         [87, 111, 114, 108, 100]",
        "core::ptr::non_null::NonNull<alloc::rc::RcInner<str>> \
         {pointer: alloc::rc::RcInner<str> {strong:",
        "value: 0x",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}

#[test]
fn debug_mode_string_forms_render_in_lldb() {
    // The lldb forms of the same values: a `String` and an
    // `Rc`'s `str` pointee render through their provider
    // lines with the character children appended.
    let dir = temp_dir("debug-string-forms-lldb");
    let src = fixture(&dir, "string_forms.rs", DEBUG_STRING_FORMS_SRC);
    let out = debug_driver_session(
        &dir,
        &src,
        "break 15\nrun\nv plain_string\nv str_in_rc\nv box_str\n",
        "lldb",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "(alloc::string::String) plain_string = \"Hello\" \
         { [0] = 'H' [1] = 'e' [2] = 'l' [3] = 'l' [4] = 'o' }",
        "str_in_rc = strong=1, weak=0 \
         { value = \"Hello\" { [0] = 'H'",
        "box_str = \"World\" { [0] = 'W'",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}

const DEBUG_BOX_DST_SRC: &str = r#"use std::rc::Rc;

struct Foo<T: ?Sized> {
    vals: T,
}

struct Holder {
    p: *const Foo<[u8]>,
}

fn main() {
    let box_str: Box<str> = Box::from("World");
    let box_slice: Box<[u8]> = Box::new([9, 8, 7]);
    let box_dst: Box<Foo<[u8]>> = Box::new(Foo { vals: [5, 6] });
    let raw_slice: *const [u8] = &[9, 8, 7];
    let holder = Holder { p: &Foo { vals: [5, 6] } };
    let rc_str: Rc<str> = Rc::from("Hello");
    zzz(); // #break
}
fn zzz() {}
"#;

#[test]
fn debug_mode_box_unsized_pointees_print_the_pointee() {
    // A box over an unsized pointee prints the pointee: a
    // `str` as its byte list, a slice as its elements, a
    // DST struct as the struct itself with the slice tail's
    // elements. A raw slice pointer prints the same element
    // list, and a raw pointer to a DST struct held in a
    // field expands the pointee struct inline.
    let dir = temp_dir("debug-box-dst");
    let src = fixture(&dir, "box_dst.rs", DEBUG_BOX_DST_SRC);
    let out = debug_driver_session(
        &dir,
        &src,
        "break 18\nrun\nprint box_str\nprint box_slice\nprint box_dst\nprint raw_slice\nprint holder\nprint rc_str\n",
        "gdb",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in [
        "alloc::boxed::Box<str, alloc::alloc::Global> \
         [87, 111, 114, 108, 100]",
        "alloc::boxed::Box<[u8], alloc::alloc::Global> [9, 8, 7]",
        "alloc::boxed::Box<box_dst::Foo<[u8]>, \
         alloc::alloc::Global> box_dst::Foo<[u8]> {vals: [5, 6]}",
        "*const [u8] [9, 8, 7]",
        "box_dst::Holder {p: box_dst::Foo<[u8]> {vals: [5, 6]}}",
        "value: 0x",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}

#[test]
fn debug_mode_box_str_lldb_provider_line() {
    // A boxed `str` takes the pointee's own lldb provider
    // form: the string summary with its character children.
    let dir = temp_dir("debug-box-str-lldb");
    let src = fixture(&dir, "string_forms.rs", DEBUG_STRING_FORMS_SRC);
    let out = debug_driver_session(
        &dir,
        &src,
        "break 15\nrun\nv box_str\n",
        "lldb",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    assert!(
        stdout.contains(
            "(alloc::boxed::Box<str, alloc::alloc::Global>) \
             box_str = \"World\" \
             { [0] = 'W' [1] = 'o' [2] = 'r' [3] = 'l' [4] = 'd' }"
        ),
        "missing the box provider line in debug transcript: {stdout}"
    );
}
