use super::{fixture, run_driver, temp_dir};


const SILENT_SRC: &str = r#"
fn main() {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < 100 { s += i; i += 1; }
    let _ = s;
}
"#;

#[test]
fn run_mode_silent_program_exits_zero() {
    // A plain computation-only main converts and executes to the end.
    // This used to be blocked in the interpreter: real MIR never
    // writes the return place `_0` of a unit-returning function, and
    // `Return` read it as uninitialized (exit 2, reason naming `_0`).
    // The interpreter now reads an unwritten unit `_0` as `()`, so
    // the program terminates normally.
    let dir = temp_dir("silent");
    let src = fixture(&dir, "silent.rs", SILENT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "silent program must terminate normally; stderr={stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "run mode keeps stdout clean: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

const BUILTIN_FN_VALUE_DROP_SRC: &str = r#"
struct Guard(&'static str);
impl Drop for Guard {
    fn drop(&mut self) { println!("dropped {}", self.0); }
}
fn apply<F: FnOnce(T), T>(f: F, x: T) { f(x) }
fn main() {
    apply(drop, Guard("a"));
    let w = vec![1, 2, 3];
    let m = w.iter().map(drop).count();
    println!("mapped {m}");
}
"#;

#[test]
fn run_mode_builtin_fn_item_as_value_drop() {
    // `std::mem::drop` used *as a value*: conversion has no call
    // site at which to bake the builtin (the call goes through an
    // `FnOnce` shim inside a generic consumer, or through the
    // `Map` adapter), so the runtime is handed the plain
    // definition name. The definition's own body is collected for
    // the value demand, and running it drops the argument with
    // its glue -- the guard's `Drop` must print, in order.
    let dir = temp_dir("builtin-fn-value-drop");
    let src = fixture(&dir, "builtin_fn_value_drop.rs", BUILTIN_FN_VALUE_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "drop as a function value must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "dropped a\nmapped 3\n"
    );
}

const BUILTIN_FN_VALUE_DEREF_SRC: &str = r#"
use std::ops::Deref;
fn main() {
    let r = Ok::<String, ()>(String::from("foo-bar"));
    let v = r.as_ref().map(Deref::deref).ok();
    assert_eq!(v, Some("foo-bar"));
    println!("deref ok");
}
"#;

#[test]
fn run_mode_builtin_fn_item_as_value_deref() {
    // `Deref::deref` used *as a value* (the
    // llvm-miscompile-metadata-invalidation-36023 shape): the
    // builtin that serves a direct call cannot serve a call
    // through the value, because it needs the Self type
    // conversion would have baked in. The impl instance's body
    // is collected for the value demand instead.
    let dir = temp_dir("builtin-fn-value-deref");
    let src = fixture(&dir, "builtin_fn_value_deref.rs", BUILTIN_FN_VALUE_DEREF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Deref::deref as a function value must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "deref ok\n");
}

const DYN_FNPTR_SIGNATURE_SRC: &str = r#"
trait T { fn get(&self) -> u32; }
struct Foo<F>(F);
impl T for Foo<fn(u8) -> u8> { fn get(&self) -> u32 { 1 } }
impl T for Foo<fn(u8) -> u16> { fn get(&self) -> u32 { 2 } }
fn main() {
    let a: Box<dyn T> = Box::new(Foo((|x: u8| x) as fn(u8) -> u8));
    let b: Box<dyn T> = Box::new(Foo((|x: u8| x as u16) as fn(u8) -> u16));
    assert_eq!(a.get(), 1);
    assert_eq!(b.get(), 2);
    println!("dynsig ok");
}
"#;

#[test]
fn run_mode_dyn_dispatch_distinguishes_fn_ptr_signatures() {
    // The concrete type a `dyn` value records must spell a
    // function-pointer generic argument with its signature, the
    // way the driver spells it in instance names: with the
    // argument collapsed to the kind name `FnPtr`, the two impls
    // above are indistinguishable and the dispatch scan finds
    // an ambiguity instead of the one impl the vtable names.
    let dir = temp_dir("dyn-fnptr-signature");
    let src = fixture(&dir, "dyn_fnptr_signature.rs", DYN_FNPTR_SIGNATURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dyn dispatch over fn-pointer signatures must resolve; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dynsig ok\n");
}

#[test]
fn run_mode_deeply_nested_zst_range_chain_return() {
    // The `punch_card` shape of tests/ui/expr/weird-exprs.rs: a
    // return type nesting dozens of one-field zero-sized structs
    // (`RangeToInclusive` / `RangeTo` over `RangeFull`). rustc
    // MIR writes none of the places of a zero-sized type, so the
    // callee's `_0` is unwritten and the return reads as the
    // type's single value -- which conversion only synthesises
    // when its structural proof reaches the innermost type. The
    // proof's recursion cap (32) used to give up a dozen levels
    // short of this chain's end, and the return failed as a read
    // of uninitialized place `_0` (exit 2).
    let mut expr = "..".to_string();
    for _ in 0..48 {
        expr = format!("..={expr}");
    }
    let src = format!(
        "fn punch_card() -> impl std::fmt::Debug {{\n    {expr}\n}}\nfn main() {{ punch_card(); println!(\"done\"); }}\n"
    );
    let dir = temp_dir("deep-zst-range-chain");
    let src_path = fixture(&dir, "deep_zst_range_chain.rs", &src);
    let out = run_driver(&src_path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a deeply nested zero-sized return must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "done\n");
}

#[test]
fn run_mode_deeply_nested_zst_wrapper_chain_return() {
    // The same proof-depth point through user-defined wrappers:
    // a one-field struct nested forty deep over `()` is a
    // zero-sized type with a single value, so MIR leaves the
    // return place unwritten and the value is the conversion
    // synthesis -- which must reach the bottom of the chain.
    let mut ty = "()".to_string();
    let mut val = "()".to_string();
    for _ in 0..40 {
        ty = format!("W<{ty}>");
        val = format!("W({val})");
    }
    let src = format!(
        "struct W<T>(T);\nfn deep() -> {ty} {{ {val} }}\nfn main() {{ let _x = deep(); println!(\"done\"); }}\n"
    );
    let dir = temp_dir("deep-zst-wrapper-chain");
    let src_path = fixture(&dir, "deep_zst_wrapper_chain.rs", &src);
    let out = run_driver(&src_path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a deeply nested zero-sized wrapper return must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "done\n");
}

const IMPORTED_MAIN_SRC: &str = r#"
pub mod foo {
    pub fn bar() {
        println!("Hello world!");
    }
}
use foo::bar as main;
"#;

#[test]
fn run_mode_imported_main_from_inner_mod() {
    // tests/ui/entry-point/imported_main_from_inner_mod.rs
    // (Issue #532): the entry point is a re-export, not a
    // definition in the root module; run mode enters through
    // the compiler-designated entry id.
    let dir = temp_dir("imported-main");
    let src = fixture(&dir, "imported_main.rs", IMPORTED_MAIN_SRC);
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "Hello world!\n");
}
