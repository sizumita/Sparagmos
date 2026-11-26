// Breakpoint, frame and history session tests.

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

const DEBUG_MULTILINE_CHAIN_SRC: &str = r#"struct Foo;
impl Foo {
    fn bar(self) -> Foo {
        println!("bar");
        self
    }
    fn baz(self) -> Foo {
        println!("baz");
        self
    }
}
fn main() {
    let f = Foo;
    f.bar()
        .baz();
}
"#;

#[test]
fn debug_mode_multiline_chain_up_shows_the_called_segment() {
    // A method chain split across lines: a native debugger
    // attributes each call in the chain to the line of its own
    // segment, so `up` from inside `baz` shows the `.baz()`
    // line in the caller, not the line the chain starts on --
    // and `up` from inside `bar` shows the `f.bar()` line.
    let dir = temp_dir("debug-multiline-chain");
    let src = fixture(&dir, "multiline_chain.rs", DEBUG_MULTILINE_CHAIN_SRC);
    let out = debug_driver(&dir, &src, "break 8\nrun\nup\n");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    assert!(
        stdout.contains("15\t        .baz();"),
        "`up` from baz did not show the `.baz()` line: {stdout}"
    );
    assert!(
        !stdout.contains("14\t    f.bar()"),
        "`up` from baz showed the chain's first line: {stdout}"
    );
    let out = debug_driver(&dir, &src, "break 4\nrun\nup\n");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    assert!(
        stdout.contains("14\t    f.bar()"),
        "`up` from bar did not show the `f.bar()` line: {stdout}"
    );
}

const DEBUG_HISTORY_EQ_SRC: &str = r#"struct Foo {
    a: usize,
}

impl Foo {
    #[inline(never)]
    fn get_a(&self) -> Option<usize> {
        Some(self.a)
    }

    #[inline(never)]
    fn var_return_opt_try(&self) -> Option<usize> {
        let r = self.get_a()?;
        Some(r)
    }
}

fn main() {
    let f = Foo { a: 1 };
    let _ = f.var_return_opt_try();
}
"#;

#[test]
fn debug_mode_history_equality_at_fn_breakpoint_gdb() {
    // The debuginfo suite's zst-interferes-with-prologue
    // session: stop in a method by function breakpoint, print
    // `self`, step once, print it again, and print the history
    // comparison `$1 == $2` — the session must answer `true`,
    // the receiver being the same object at both stops.
    let dir = temp_dir("debug-history-eq-gdb");
    let src = fixture(&dir, "zst_prologue.rs", DEBUG_HISTORY_EQ_SRC);
    let out = debug_driver_session(
        &dir,
        &src,
        "pretty-printers\nbreak-fn zst_prologue::Foo::var_return_opt_try\nrun\nprint self\nnext\nprint self\nprint $1 == $2\n",
        "gdb",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    for want in ["$1 = ", "$2 = ", "$3 = true"] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in debug transcript: {stdout}"
        );
    }
}

#[test]
fn debug_mode_history_equality_at_fn_breakpoint_lldb() {
    // The same session in lldb form: history numbers run
    // from $0, and the comparison prints as a bare result.
    let dir = temp_dir("debug-history-eq-lldb");
    let src = fixture(&dir, "zst_prologue.rs", DEBUG_HISTORY_EQ_SRC);
    let out = debug_driver_session(
        &dir,
        &src,
        "break-fn zst_prologue::Foo::var_return_opt_try\nrun\nv self\nnext\nv self\nv $0 == $1\n",
        "lldb",
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "debug session failed: {stdout}");
    assert!(
        stdout.contains("= true"),
        "missing the comparison result in debug transcript: {stdout}"
    );
}
