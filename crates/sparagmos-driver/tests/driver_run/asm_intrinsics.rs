use super::{Command, fixture, run_driver, run_driver_with_aux, run_ok, summarize_driver, temp_dir};


const INTRINSIC_NAMED_WRAP_BINDER_SRC: &str = r#"
#![feature(unsafe_binders)]
#![allow(incomplete_features)]
use std::unsafe_binder::wrap_binder;
fn atomic_fence() {
    unsafe {
        let x = 1;
        let _binder: unsafe<'a> i32 = wrap_binder!(x);
    }
}
fn main() {
    atomic_fence();
    println!("main done");
}
"#;

#[test]
fn run_mode_wrap_binder_body_named_like_an_intrinsic_executes() {
    // An ordinary function that merely shares a compiler
    // intrinsic's name (`atomic_fence`) and whose body wraps a
    // value with `wrap_binder!`: the wrap's Stable MIR
    // conversion is unimplemented, so the body materializes
    // through the compiler's internal MIR with the wrap
    // erased to the identity (Issue #510), conversion
    // completes, and the call executes the body's own
    // computation -- the run completes (exit 0, "main done"
    // printed) on the strength of the converted body, not by
    // diverting the call to the intrinsic's builtin. (This
    // fixture used to pin the opposite outcome -- the body
    // stayed unconverted and the call exited 2 -- back when
    // it carried the unconverted-body contract; wrap bodies
    // convert since the internal MIR materialization, the
    // same transition tail-call bodies made before them.)
    let dir = temp_dir("intrinsicname");
    let src = fixture(&dir, "intrinsicname.rs", INTRINSIC_NAMED_WRAP_BINDER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the converted wrap body executes and the run completes; stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("main done"),
        "the run completes past the call: {stdout}"
    );
    assert!(
        !stderr.contains("no converted body"),
        "the call is not reported as an unconverted body: {stderr}"
    );
}

// ----- monomorphization validity checks (Lane CA) ---------------------------

const SIMD_SCALAR_SRC: &str = r#"
#![feature(core_intrinsics)]
use std::intrinsics::simd::simd_add;
fn main() { unsafe { let _ = simd_add(0, 1); } }
"#;

const SIMD_VECTOR_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_add;
#[repr(simd)]
#[derive(Copy, Clone)]
struct I32x4([i32; 4]);
fn main() { unsafe { let _ = simd_add(I32x4([1, 2, 3, 4]), I32x4([5, 6, 7, 8])); } }
"#;

const SIMD_TOO_MANY_LANES_SRC: &str = r#"
#![feature(portable_simd)]
use std::simd::Simd;
fn main() { let _x = Simd::<u16, 54321>::splat(0); }
"#;

#[test]
fn convert_mode_simd_intrinsic_with_scalar_args_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-scalar");
    let src = fixture(&dir, "simd_scalar.rs", SIMD_SCALAR_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "an invalid SIMD monomorphization must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("simd_add")
            && stderr.contains("expected SIMD input type"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_intrinsic_with_vector_args_converts() {
    let dir = temp_dir("mono-simd-vector");
    let src = fixture(&dir, "simd_vector.rs", SIMD_VECTOR_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a valid SIMD monomorphization must convert; stderr={stderr}"
    );
}

// ----- locally declared compiler intrinsics (Issue #384) --------------------

// The `no_core` / minicore test crates declare the intrinsics
// they exercise themselves with `#[rustc_intrinsic]` instead of
// linking core's, so the callee is named by the declaration
// site's own path and the qualified builtin registry never
// matches it. A modelled intrinsic is served by its builtin
// (its identity is its item name); an unmodelled one stays an
// unmet demand.
const LOCAL_INTRINSICS_SRC: &str = r#"
#![feature(intrinsics, repr_simd)]
#[rustc_intrinsic]
unsafe fn simd_add<T>(x: T, y: T) -> T;
#[rustc_intrinsic]
unsafe fn simd_reduce_add_ordered<T, U>(x: T, init: U) -> U;
#[rustc_intrinsic]
unsafe fn nontemporal_store<T>(ptr: *mut T, val: T);
#[repr(simd)]
#[derive(Copy, Clone)]
struct I32x4([i32; 4]);
fn main() {
    let c = unsafe { simd_add(I32x4([1, 2, 3, 4]), I32x4([5, 6, 7, 8])) };
    let sum: i32 = unsafe { simd_reduce_add_ordered(c, 0i32) };
    println!("{sum}");
    let mut v: u32 = 0;
    unsafe { nontemporal_store(&mut v as *mut u32, 42u32) };
    println!("{v}");
}
"#;

const LOCAL_UNMODELLED_INTRINSIC_SRC: &str = r#"
#![feature(intrinsics)]
#[rustc_intrinsic]
fn prefetch_read_data<T, const LOCALITY: i32>(data: *const T);
fn main() {
    let v: u32 = 0;
    prefetch_read_data::<u32, 0>(&v as *const u32);
}
"#;

fn target_crate_line(stdout: &str) -> &str {
    stdout
        .lines()
        .find(|line| line.starts_with("target-crate converted:"))
        .expect("the summary prints a target-crate line")
}

#[test]
fn run_mode_locally_declared_intrinsics_run() {
    let dir = temp_dir("local-intrinsics");
    let src = fixture(&dir, "local_intrinsics.rs", LOCAL_INTRINSICS_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "36\n42\n");
}

#[test]
fn convert_mode_locally_declared_intrinsics_convert() {
    let dir = temp_dir("local-intrinsics-convert");
    let src = fixture(&dir, "local_intrinsics.rs", LOCAL_INTRINSICS_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        target_crate_line(&stdout).ends_with("skipped: 0"),
        "a crate whose intrinsics are all modelled must convert fully; stdout={stdout}"
    );
    assert!(
        !stdout.contains("compiler intrinsic (no fallback body)"),
        "no local intrinsic may be left unmet; stdout={stdout}"
    );
}

const LOCAL_UNREACHABLE_SRC: &str = r#"
#![feature(intrinsics)]
#[rustc_intrinsic]
const unsafe fn unreachable() -> !;
fn do_or_die(cond: bool) {
    if !cond {
        unsafe { unreachable() }
    }
}
fn main() {
    do_or_die(true);
    println!("alive");
    do_or_die(false);
    println!("must not print");
}
"#;

#[test]
fn convert_mode_locally_declared_unreachable_converts() {
    // The shape of tests/ui/target-feature/no-llvm-leaks.rs: a
    // minicore-style crate declares `unreachable` itself and a
    // body calls it. The intrinsic is modelled (executing it is
    // undefined behaviour, so the run aborts), so the demand is
    // met and the crate converts fully.
    let dir = temp_dir("local-unreachable-convert");
    let src = fixture(&dir, "local_unreachable.rs", LOCAL_UNREACHABLE_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        target_crate_line(&stdout).ends_with("skipped: 0"),
        "a crate calling the modelled unreachable intrinsic must convert fully; stdout={stdout}"
    );
}

#[test]
fn run_mode_locally_declared_unreachable_aborts() {
    // rustc's MIR building turns a direct call to the
    // intrinsic into an `Unreachable` terminator, whose
    // execution is the guest panic "entered unreachable code"
    // (exit 101); what the model changes is that the crate
    // converts at all — without it the caller stays
    // unconverted and the run exits 2.
    let dir = temp_dir("local-unreachable");
    let src = fixture(&dir, "local_unreachable.rs", LOCAL_UNREACHABLE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "reaching the unreachable intrinsic ends the guest run; stdout={stdout} stderr={stderr}"
    );
    assert!(
        stderr.contains("entered unreachable code"),
        "the report names the unreachable point: {stderr}"
    );
    assert!(stdout.contains("alive"), "stdout={stdout}");
    assert!(
        !stdout.contains("must not print"),
        "the run must not continue past the intrinsic: {stdout}"
    );
}

#[test]
fn convert_mode_locally_declared_unmodelled_intrinsic_stays_skipped() {
    let dir = temp_dir("local-intrinsic-unmodelled");
    let src = fixture(
        &dir,
        "local_unmodelled.rs",
        LOCAL_UNMODELLED_INTRINSIC_SRC,
    );
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        target_crate_line(&stdout).ends_with("skipped: 1"),
        "an intrinsic with no interpreter model must not be credited; stdout={stdout}"
    );
    assert!(
        stdout.contains("compiler intrinsic (no fallback body)"),
        "the skip must keep the intrinsic reason; stdout={stdout}"
    );
}

#[test]
fn convert_mode_simd_type_with_too_many_lanes_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-lanes");
    let src = fixture(&dir, "simd_lanes.rs", SIMD_TOO_MANY_LANES_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a SIMD type over the lane limit must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("invalid number of lanes"),
        "the failure must name the reason; stderr={stderr}"
    );
}

// ----- InlineAsm: the evaluable register class (Lane GI) -----
// Expected outputs are the native results of the same sources.

const ASM_MOV_ADD_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let x: u32 = 40;
    let mut y: u32 = 0;
    unsafe {
        asm!("mov {0:e}, {1:e}", "add {0:e}, 2", out(reg) y, in(reg) x);
    }
    let mut z: u64 = 5;
    unsafe {
        asm!("add {}, 3", inout(reg) z);
    }
    println!("y={y} z={z}");
}
"#;

#[test]
fn run_mode_inline_asm_mov_add_inout() {
    run_ok("asm-mov-add", ASM_MOV_ADD_SRC, "y=42 z=8\n");
}

const ASM_SHIFT_LOGIC_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let mut a: i32 = -16;
    let mut b: u32 = 3;
    let mut c: u32 = 0b1100;
    unsafe {
        asm!("sar {0:e}, 2", inout(reg) a);
        asm!("shl {}, 4", inout(reg) b);
        asm!("and {}, 0b1010", inout(reg) c);
        asm!("or {}, 0b0101", inout(reg) c);
        asm!("xor {}, {}", inout(reg) c, in(reg) 0b1111u32);
    }
    println!("{a} {b} {c}");
}
"#;

#[test]
fn run_mode_inline_asm_shifts_and_logic() {
    run_ok("asm-shift-logic", ASM_SHIFT_LOGIC_SRC, "-4 48 2\n");
}

const ASM_SUBREG_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let mut x: u16 = 0x1234;
    unsafe {
        asm!("mov {0:l}, 0xAB", inout(reg) x);
        asm!("mov {0:x}, 0xCD", inout(reg) x);
    }
    println!("{x}");
}
"#;

#[test]
fn run_mode_inline_asm_subregister_modifiers() {
    run_ok("asm-subreg", ASM_SUBREG_SRC, "205\n");
}

const ASM_ZERO_EXTEND_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let mut x: u64 = u64::MAX;
    unsafe {
        asm!("mov {0:e}, 1", inout(reg) x);
    }
    println!("{x}");
}
"#;

#[test]
fn run_mode_inline_asm_32bit_write_zero_extends() {
    run_ok("asm-zero-extend", ASM_ZERO_EXTEND_SRC, "1\n");
}

const ASM_CONST_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let y: u32;
    let mut z: u32 = 10;
    unsafe {
        asm!("mov {}, {}", out(reg) y, const 7);
        asm!("add {}, {}", inout(reg) z, const 5);
    }
    println!("{y} {z}");
}
"#;

#[test]
fn run_mode_inline_asm_const_operands() {
    run_ok("asm-const", ASM_CONST_SRC, "7 15\n");
}

const ASM_UNARY_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let mut a: i32 = 5;
    let mut b: u8 = 0x0f;
    let mut c: u32 = 41;
    unsafe {
        asm!("neg {}", inout(reg) a);
        asm!("not {}", inout(reg_byte) b);
        asm!("inc {}", inout(reg) c);
        asm!("dec {}", inout(reg) c);
        asm!("dec {}", inout(reg) c);
    }
    println!("{a} {b} {c}");
}
"#;

#[test]
fn run_mode_inline_asm_unary_ops() {
    run_ok("asm-unary", ASM_UNARY_SRC, "-5 240 40\n");
}

const ASM_NAKED_SRC: &str = r#"
use std::arch::naked_asm;

#[unsafe(naked)]
pub extern "C" fn naked_ret() {
    naked_asm!("ret");
}

fn main() {
    naked_ret();
    println!("back");
}
"#;

#[test]
fn run_mode_naked_fn_ret() {
    run_ok("asm-naked-ret", ASM_NAKED_SRC, "back\n");
}

const ASM_NAKED_ARGS_SRC: &str = r#"
use std::arch::naked_asm;

#[unsafe(naked)]
pub extern "sysv64" fn identity(x: u64) -> u64 {
    naked_asm!("mov rax, rdi", "ret");
}

#[unsafe(naked)]
pub extern "sysv64" fn add2(a: u64, b: u64) -> u64 {
    naked_asm!("mov rax, rdi", "add rax, rsi", "ret");
}

#[unsafe(naked)]
pub extern "win64" fn identity_win(x: u64) -> u64 {
    naked_asm!("mov rax, rcx", "ret");
}

#[unsafe(naked)]
pub extern "C" fn float_bits(x: f64) -> u64 {
    naked_asm!("movq rax, xmm0", "ret");
}

#[unsafe(naked)]
pub extern "C" fn sink(x: u64) {
    naked_asm!("ret");
}

fn main() {
    sink(9);
    println!(
        "{} {} {} {}",
        identity(42),
        add2(40, 2),
        identity_win(42),
        float_bits(1.5)
    );
}
"#;

#[test]
fn run_mode_naked_fn_seeds_incoming_abi_registers() {
    // A naked function's parameters arrive through the enclosing
    // function's ABI, not through asm operands: the register
    // file is seeded from the frame's argument locals (SysV64
    // integer registers, the Win64 position registers, and the
    // xmm sequence for a float), so the programs read their
    // actual arguments. A bare-`ret` body with a parameter it
    // never reads needs no seeding and still runs.
    run_ok(
        "asm-naked-args",
        ASM_NAKED_ARGS_SRC,
        "42 42 42 4609434218613702656\n",
    );
}

const ASM_NAKED_AGGREGATE_SRC: &str = r#"
use std::arch::naked_asm;

pub struct Pair(pub u64, pub u64);

#[unsafe(naked)]
pub extern "C" fn first(p: Pair) -> u64 {
    naked_asm!("mov rax, rdi", "ret");
}

fn main() {
    println!("{}", first(Pair(42, 1)));
}
"#;

#[test]
fn run_mode_naked_fn_unseedable_parameters_are_a_named_error() {
    // An aggregate parameter is not register-passed in the
    // model: the naked body is a named error rather than a
    // block that reads zeroed registers as if they were the
    // argument.
    let dir = temp_dir("asm-naked-aggregate");
    let src = fixture(&dir, "case.rs", ASM_NAKED_AGGREGATE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an unseedable naked parameter must exit two; stderr={stderr}"
    );
    assert!(
        stderr.contains("incoming ABI registers cannot be seeded"),
        "the reason names the unseedable parameters: {stderr}"
    );
}

const ASM_SYM_LOCAL_SRC: &str = r#"
use std::arch::asm;

pub extern "C" fn add_one(x: u64) -> u64 {
    x + 1
}

mod inner {
    pub extern "C" fn add_two(x: u64) -> u64 {
        x + 2
    }
}

fn main() {
    let out1: u64;
    let out2: u64;
    unsafe {
        asm!(
            "mov rdi, {inp}",
            "call {target}",
            inp = in(reg) 41u64,
            target = sym add_one,
            lateout("rax") out1,
        );
        asm!(
            "mov rdi, {inp}",
            "call {target}",
            inp = in(reg) 41u64,
            target = sym inner::add_two,
            lateout("rax") out2,
        );
    }
    println!("{out1} {out2}");
}
"#;

#[test]
fn run_mode_asm_sym_call_resolves_local_functions() {
    // A `sym` operand denoting a function of the local crate —
    // at the crate root or in a module — resolves by its
    // complete path and the call executes that function.
    run_ok("asm-sym-local", ASM_SYM_LOCAL_SRC, "42 43\n");
}

const ASM_SYM_AUX_LIB_SRC: &str = r#"
pub extern "C" fn foo(x: u64) -> u64 {
    x + 1000
}
"#;

const ASM_SYM_AUX_MAIN_SRC: &str = r#"
extern crate auxlib;

use std::arch::asm;

// A local function whose last name segment collides with the
// external `auxlib::foo` the asm block actually names. It is
// never called from Rust code.
pub extern "C" fn foo(x: u64) -> u64 {
    x + 1
}

fn main() {
    let out: u64;
    unsafe {
        asm!(
            "mov rdi, {inp}",
            "call {target}",
            inp = in(reg) 41u64,
            target = sym auxlib::foo,
            lateout("rax") out,
        );
    }
    println!("{out}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_an_external_function_is_rejected() {
    // The `sym` operand names `auxlib::foo`, a function of
    // another crate: it must not resolve to the unrelated local
    // `foo` whose last name segment happens to match. The call
    // stays outside the evaluable class (exit two) instead of
    // executing the local function (which would print 42; the
    // named function would print 1041).
    let dir = temp_dir("asm-sym-aux");
    let aux = fixture(&dir, "auxlib.rs", ASM_SYM_AUX_LIB_SRC);
    let src = fixture(&dir, "auxmain.rs", ASM_SYM_AUX_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("auxlib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an external asm call target must exit two; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "",
        "the unrelated local function must not execute"
    );
}

#[test]
fn convert_mode_asm_with_generic_const_and_sym_operands_converts() {
    // Regression: recording function ABIs for the naked-body
    // argument seeding queried `Instance::ty` on still-generic
    // instances (here `core::mem::size_of::<T>`, named by a
    // `const` asm operand), which panics with "needs further
    // instantiation"; the panic escaped the ABI query's
    // containment and read as a conversion crash (exit 101) in
    // convert mode.
    let dir = temp_dir("asm-generic-const");
    let src = fixture(
        &dir,
        "generic_const.rs",
        r#"use std::arch::asm;

fn foofoo<const N: usize>() {}

unsafe fn foo<const N: usize>() {
    asm!("/* {0} */", const N);
    asm!("/* {0} */", sym foofoo::<N>);
}

fn barbar<T>() {}

unsafe fn bar<T>() {
    asm!("/* {0} */", const std::mem::size_of::<T>());
    asm!("/* {0} */", sym barbar::<T>);
}

fn main() {
    unsafe {
        foo::<0>();
        bar::<usize>();
    }
}
"#,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "generic asm const/sym operands must convert; stderr={stderr}"
    );
}

const ASM_PUSH_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let x: u32 = 1;
    unsafe {
        asm!("push {}", in(reg) x);
    }
    println!("done");
}
"#;

#[test]
fn run_mode_inline_asm_outside_class_is_named_error() {
    // `push` is outside the evaluable class: conversion carries
    // the block faithfully, and the interpreter refuses it at
    // execution time — exit 2, and the reason names InlineAsm.
    let dir = temp_dir("asm-push");
    let src = fixture(&dir, "case.rs", ASM_PUSH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("unsupported terminator: InlineAsm"),
        "the reason names InlineAsm: {stderr}"
    );
}

const ASM_LATEOUT_ALIAS_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let y: u64;
    unsafe {
        asm!("mov rcx, rax", in("rax") 42u64, lateout("rax") y, out("rcx") _);
    }
    println!("{y}");
}
"#;

#[test]
fn run_mode_inline_asm_lateout_aliased_with_input() {
    // The template never writes rax, but the lateout bound to
    // it still observes the incoming content the aliased input
    // seeded: natively y == 42.
    run_ok("asm-lateout-alias", ASM_LATEOUT_ALIAS_SRC, "42\n");
}

const ASM_CALL_BOOL_ABI_WIDTH_SRC: &str = r#"
use std::arch::asm;

pub extern "sysv64" fn choose(x: bool) -> u64 {
    if x {
        42
    } else {
        43
    }
}

fn main() {
    let out1: u64;
    let out2: u64;
    unsafe {
        asm!(
            "call {target}",
            target = sym choose,
            in("rdi") 0x1_0000_0000u64,
            lateout("rax") out1,
        );
        asm!(
            "call {target}",
            target = sym choose,
            in("rdi") 0x1_0000_0001u64,
            lateout("rax") out2,
        );
    }
    println!("{out1} {out2}");
}
"#;

#[test]
fn run_mode_asm_call_bool_argument_reads_only_the_abi_byte() {
    // A boolean argument occupies one byte of its register: the
    // callee reads only the low byte, and the upper bits are
    // unspecified incoming state. The first call passes a false
    // whose upper bits are nonzero (natively 43); the second a
    // genuine true in the low byte (natively 42).
    run_ok(
        "asm-call-bool-width",
        ASM_CALL_BOOL_ABI_WIDTH_SRC,
        "43 42\n",
    );
}

const ASM_LEA_WRAP_SRC: &str = r#"
use std::arch::asm;
fn main() {
    let y: u64;
    let z: u64;
    unsafe {
        asm!("lea {0}, [ecx + 1]", out(reg) y, in("rcx") 0xffff_ffffu64);
        asm!("lea {0}, [rcx + 1]", out(reg) z, in("rcx") 0xffff_ffffu64);
    }
    println!("{y} {z}");
}
"#;

#[test]
fn run_mode_inline_asm_lea_wraps_at_address_size() {
    // The 32-bit base form wraps the effective address at 32
    // bits before it is zero-extended into the destination (0);
    // the 64-bit base form keeps the full sum (2^32).
    run_ok("asm-lea-wrap", ASM_LEA_WRAP_SRC, "0 4294967296\n");
}

const ASM_NAKED_SIGNED_RET_SRC: &str = r#"
use std::arch::naked_asm;

#[unsafe(naked)]
pub extern "sysv64" fn neg_one64() -> i64 {
    naked_asm!("mov rax, -1", "ret");
}

#[unsafe(naked)]
pub extern "sysv64" fn neg_one32() -> i32 {
    naked_asm!("mov eax, -1", "ret");
}

#[unsafe(naked)]
pub extern "sysv64" fn neg_one16() -> i16 {
    naked_asm!("mov ax, -1", "ret");
}

#[unsafe(naked)]
pub extern "sysv64" fn neg_one8() -> i8 {
    naked_asm!("mov al, -1", "ret");
}

fn main() {
    println!(
        "{} {} {} {}",
        neg_one64(),
        neg_one32(),
        neg_one16(),
        neg_one8()
    );
}
"#;

#[test]
fn run_mode_naked_fn_signed_return_values() {
    // A naked body's return value is the register content read
    // at the return type's shape: a signed type reads the bits
    // as two's complement at its width, so each body above
    // returns -1 — natively `-1 -1 -1 -1`.
    run_ok(
        "asm-naked-signed-ret",
        ASM_NAKED_SIGNED_RET_SRC,
        "-1 -1 -1 -1\n",
    );
}

const ASM_NAKED_RSP_SRC: &str = r#"
use std::arch::naked_asm;

#[unsafe(naked)]
pub extern "sysv64" fn alignment() -> u64 {
    naked_asm!("mov rax, rsp", "and rax, 15", "ret");
}

fn main() {
    println!("{}", alignment());
}
"#;

#[test]
fn run_mode_naked_fn_incoming_rsp_read_is_a_named_error() {
    // The incoming `rsp` is fixed by the caller's frame layout
    // (natively this body returns 8, for the return address on
    // the stack); the register model does not represent it, so
    // reading it is a named error — exit 2 — rather than a
    // computation with a fabricated zero stack pointer.
    let dir = temp_dir("asm-naked-rsp");
    let src = fixture(&dir, "case.rs", ASM_NAKED_RSP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a read of the incoming stack pointer must exit two; stderr={stderr}"
    );
    assert!(
        stderr.contains("stack pointer"),
        "the reason names the incoming stack pointer: {stderr}"
    );
}

const ASM_SYM_NAKED_SRC: &str = r#"
use std::arch::{asm, naked_asm};

#[unsafe(naked)]
pub extern "sysv64" fn read_r10() -> u64 {
    naked_asm!("mov rax, r10", "ret");
}

#[unsafe(naked)]
pub extern "sysv64" fn add_r10(x: u64) -> u64 {
    naked_asm!("mov rax, rdi", "add rax, r10", "ret");
}

fn main() {
    let out1: u64;
    let out2: u64;
    unsafe {
        asm!(
            "call {target}",
            target = sym read_r10,
            in("r10") 42u64,
            lateout("rax") out1,
        );
        asm!(
            "mov rdi, {inp}",
            "call {target}",
            inp = in(reg) 41u64,
            target = sym add_r10,
            in("r10") 100u64,
            lateout("rax") out2,
        );
    }
    println!("{out1} {out2}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_naked_fn_forwards_caller_registers() {
    // A `sym` call into a naked function preserves the calling
    // block's architectural register state: the callee's body
    // reads `r10`, a register no argument seeding supplies,
    // whose value the caller established with an explicit
    // `in("r10")` operand. The second callee also takes an
    // argument: its argument register is seeded from the
    // argument as usual, and the forwarded `r10` composes with
    // it. Natively this prints `42 141`.
    run_ok("asm-sym-naked", ASM_SYM_NAKED_SRC, "42 141\n");
}

const ASM_SYM_NAKED_RSP_SRC: &str = r#"
use std::arch::{asm, naked_asm};

#[unsafe(naked)]
pub extern "sysv64" fn read_rsp() -> u64 {
    naked_asm!("mov rax, rsp", "and rax, 15", "ret");
}

fn main() {
    let out: u64;
    unsafe {
        asm!(
            "call {target}",
            target = sym read_rsp,
            lateout("rax") out,
        );
    }
    println!("{out}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_naked_fn_rsp_read_stays_a_named_error() {
    // Forwarding the calling block's register state does not
    // fabricate an incoming stack pointer for the callee: the
    // caller never established `rsp`, so the naked callee's
    // read of it stays the named error (exit 2), exactly as
    // for a naked function called from Rust code.
    let dir = temp_dir("asm-sym-naked-rsp");
    let src = fixture(&dir, "case.rs", ASM_SYM_NAKED_RSP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a forwarded read of the incoming stack pointer must exit two; stderr={stderr}"
    );
    assert!(
        stderr.contains("stack pointer"),
        "the reason names the incoming stack pointer: {stderr}"
    );
}

const ASM_SYM_NAKED_OUTGOING_SRC: &str = r#"
use std::arch::{asm, naked_asm};

#[unsafe(naked)]
pub extern "sysv64" fn set_r10() -> u64 {
    naked_asm!("mov r10, 42", "mov rax, 1", "ret");
}

fn main() {
    let via_bound: u64;
    let via_step: u64;
    let ret: u64;
    unsafe {
        asm!(
            "call {helper}",
            helper = sym set_r10,
            lateout("rax") ret,
            lateout("r10") via_bound,
        );
        asm!(
            "call {helper}",
            "mov {out}, r10",
            helper = sym set_r10,
            out = lateout(reg) via_step,
        );
    }
    println!("{ret} {via_bound} {via_step}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_naked_fn_propagates_outgoing_registers() {
    // A `sym` call into a naked function hands the callee's
    // outgoing register state back to the calling block: the
    // helper writes `r10`, which is not its return register,
    // and the caller observes 42 both through an output bound
    // to `r10` and through a step that reads `r10` after the
    // call. The declared return value still comes back in
    // `rax`. Natively this prints `1 42 42`.
    run_ok(
        "asm-sym-naked-outgoing",
        ASM_SYM_NAKED_OUTGOING_SRC,
        "1 42 42\n",
    );
}

const ASM_SYM_NAKED_PRESERVE_SRC: &str = r#"
use std::arch::{asm, naked_asm};

#[unsafe(naked)]
pub extern "sysv64" fn five() -> u64 {
    naked_asm!("mov rax, 5", "ret");
}

fn main() {
    let kept: u64;
    let ret: u64;
    unsafe {
        asm!(
            "call {helper}",
            helper = sym five,
            in("r10") 7u64,
            lateout("rax") ret,
            lateout("r10") kept,
        );
    }
    println!("{ret} {kept}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_naked_fn_preserves_untouched_registers() {
    // The outgoing hand-back covers only the registers the
    // callee's body actually changed: a register the body
    // never writes keeps the calling block's own state across
    // the call, exactly as a native `call` preserves whatever
    // the callee does not overwrite. Natively this prints
    // `5 7`.
    run_ok(
        "asm-sym-naked-preserve",
        ASM_SYM_NAKED_PRESERVE_SRC,
        "5 7\n",
    );
}

const ASM_SYM_NAKED_WIDE_ARG_SRC: &str = r#"
use std::arch::{asm, naked_asm};

#[unsafe(naked)]
pub extern "sysv64" fn helper(x: u8) -> u64 {
    naked_asm!("mov rax, rdi", "ret");
}

fn main() {
    let out: u64;
    unsafe {
        asm!(
            "call {target}",
            target = sym helper,
            in("rdi") 0x1_0000_0001u64,
            lateout("rax") out,
        );
    }
    println!("{out}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_naked_fn_keeps_forwarded_argument_bits() {
    // The caller establishes rdi at full width; the callee's
    // parameter is a u8, but its body reads the register itself,
    // whose upper bits are the caller's and observable: argument
    // seeding supplies only the registers the forwarded state
    // did not establish, so it cannot narrow this one. Natively
    // this prints `4294967297`.
    run_ok(
        "asm-sym-naked-wide-arg",
        ASM_SYM_NAKED_WIDE_ARG_SRC,
        "4294967297\n",
    );
}

const ASM_SYM_NAKED_WIDE_RET_SRC: &str = r#"
use std::arch::{asm, naked_asm};

#[unsafe(naked)]
pub extern "sysv64" fn helper() -> u8 {
    naked_asm!("mov rax, 0x100000001", "ret");
}

fn main() {
    let y: u64;
    unsafe {
        asm!(
            "call {target}",
            target = sym helper,
            lateout("rax") y,
        );
    }
    println!("{y}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_naked_fn_keeps_raw_return_register() {
    // The callee's declared return is a u8, but the calling
    // block's output reads rax itself at u64 width: the
    // callee's outgoing register state is the register's
    // content, and the narrowed Rust return value must not be
    // scattered over it. Natively this prints `4294967297`.
    run_ok(
        "asm-sym-naked-wide-ret",
        ASM_SYM_NAKED_WIDE_RET_SRC,
        "4294967297\n",
    );
}

const ASM_SYM_NAKED_BARE_RET_SRC: &str = r#"
use std::arch::{asm, naked_asm};

#[unsafe(naked)]
pub extern "sysv64" fn id_u64() -> u64 {
    naked_asm!("ret");
}

#[unsafe(naked)]
pub extern "sysv64" fn id_u8() -> u8 {
    naked_asm!("ret");
}

fn main() {
    let a: u64;
    let b: u64;
    unsafe {
        asm!(
            "call {target}",
            target = sym id_u64,
            inout("rax") 42u64 => a,
        );
        asm!(
            "call {target}",
            target = sym id_u8,
            inout("rax") 0x1_0000_0001u64 => b,
        );
    }
    println!("{a} {b}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_bare_ret_naked_fn_reads_the_return_register() {
    // A bare-`ret` naked body returns whatever the caller left
    // in the return register: the scalar return shape is read
    // from the register file even though the body has no steps
    // before the `ret`, and the register itself is preserved
    // across the call at full width — the u8 callee's upper
    // bits are the caller's. Natively this prints
    // `42 4294967297`.
    run_ok(
        "asm-sym-naked-bare-ret",
        ASM_SYM_NAKED_BARE_RET_SRC,
        "42 4294967297\n",
    );
}

const ASM_SYM_NAKED_CALLS_ORDINARY_SRC: &str = r#"
use std::arch::{asm, naked_asm};

pub extern "sysv64" fn normal() -> u64 {
    42
}

#[unsafe(naked)]
pub extern "sysv64" fn helper() -> u64 {
    naked_asm!("call {f}", "ret", f = sym normal);
}

fn main() {
    let direct = helper();
    let via: u64;
    unsafe {
        asm!(
            "call {t}",
            t = sym helper,
            lateout("rax") via,
        );
    }
    println!("{direct} {via}");
}
"#;

#[test]
fn run_mode_asm_sym_call_to_naked_fn_propagates_ordinary_callee_return() {
    // The naked helper's body obtains its return value by
    // calling an ordinary function: the call establishes the
    // return register, so the helper's outgoing state carries
    // it and a caller entering through an asm `call` observes
    // the same value as a direct call. Natively this prints
    // `42 42`.
    run_ok(
        "asm-sym-naked-calls-ordinary",
        ASM_SYM_NAKED_CALLS_ORDINARY_SRC,
        "42 42\n",
    );
}

#[test]
fn convert_mode_inline_asm_outside_class_converts() {
    // The same out-of-class block converts in full: conversion is
    // faithful for every asm shape, and only execution is gated
    // on the evaluable classes.
    let dir = temp_dir("asm-push-convert");
    let src = fixture(&dir, "case.rs", ASM_PUSH_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg(&src)
        .output()
        .expect("failed to run sparagmos-driver");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "convert succeeds: {stdout} {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("target-crate converted: 1, skipped: 0"),
        "the asm body converts: {stdout}"
    );
}

// ----- Lane HG: the ported intrinsic families, executed ------------

const SIMD_INTRINSICS_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::{simd_add, simd_eq, simd_extract, simd_reduce_add_unordered, simd_select, simd_splat};
#[repr(simd)]
#[derive(Copy, Clone)]
struct I32x4([i32; 4]);
fn main() {
    unsafe {
        let a = I32x4([1, 2, 3, 4]);
        let b = I32x4([10, 20, 30, 40]);
        let c = simd_add(a, b);
        let v: i32 = simd_extract(c, 2);
        assert_eq!(v, 33);
        let total: i32 = simd_reduce_add_unordered(c);
        assert_eq!(total, 110);
        let seven: I32x4 = simd_splat(7);
        let t2: i32 = simd_reduce_add_unordered(seven);
        assert_eq!(t2, 28);
        let mask: I32x4 = simd_eq(a, b);
        let picked = simd_select(mask, a, b);
        let p0: i32 = simd_extract(picked, 0);
        let p1: i32 = simd_extract(picked, 1);
        assert_eq!(p0, 10);
        assert_eq!(p1, 20);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_simd_intrinsics_execute() {
    let dir = temp_dir("simd-intrinsics");
    let src = fixture(&dir, "simd_intrinsics.rs", SIMD_INTRINSICS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the simd intrinsic family must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const INTRINSIC_LEFTOVERS_SRC: &str = r#"
#![feature(core_intrinsics)]
use std::sync::atomic::{AtomicI32, AtomicU8, Ordering};
fn main() {
    let x = AtomicI32::new(-5);
    assert_eq!(x.fetch_max(3, Ordering::SeqCst), -5);
    assert_eq!(x.load(Ordering::SeqCst), 3);
    let y = AtomicU8::new(200);
    assert_eq!(y.fetch_add(100, Ordering::SeqCst), 200);
    assert_eq!(y.load(Ordering::SeqCst), 44);
    unsafe {
        let f = core::intrinsics::fadd_fast(1.5f64, 2.25f64);
        assert_eq!(f, 3.75);
        let z: i32 = core::intrinsics::float_to_int_unchecked(3.75f64);
        assert_eq!(z, 3);
        let n: u32 = core::intrinsics::ctlz_nonzero(16u32);
        assert_eq!(n, 27);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_atomic_fetch_fast_float_and_nonzero_bit_intrinsics_execute() {
    let dir = temp_dir("intrinsic-leftovers");
    let src = fixture(&dir, "intrinsic_leftovers.rs", INTRINSIC_LEFTOVERS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the fetch family, fast float and nonzero bit intrinsics must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const SIMD_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::*;
#[repr(simd)]
#[derive(Clone, Copy)]
struct Simd<T, const N: usize>([T; N]);
type I4 = Simd<i32, 4>;
fn main() {
    unsafe {
        let a: I4 = Simd([1, 2, 3, 4]);
        let b: I4 = Simd([10, 20, 30, 40]);
        let c: I4 = simd_mul(simd_add(a, b), Simd([2, 2, 2, 2]));
        let e0: i32 = simd_extract(c, 0);
        let e1: i32 = simd_extract(c, 1);
        let e3: i32 = simd_extract(c, 3);
        assert_eq!((e0, e1, e3), (22, 44, 88));
        let total: i32 = simd_reduce_add_unordered(c);
        assert_eq!(total, 220);
        let m: Simd<i32, 4> = simd_gt(c, Simd([50, 50, 50, 50]));
        let bits: u8 = simd_bitmask(m);
        assert_eq!(bits, 0b1100);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_simd_intrinsics() {
    let dir = temp_dir("simd");
    let src = fixture(&dir, "simd.rs", SIMD_SRC);
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

// ----- monomorphization validity checks (Lane DX) ---------------------------

const ATOMIC_BOOL_SRC: &str = r#"
#![feature(core_intrinsics)]
use std::intrinsics::{self, AtomicOrdering::*};
fn main() {
    let mut b = true;
    unsafe { let _ = intrinsics::atomic_load::<_, { SeqCst }, false>(&mut b); }
}
"#;

const ATOMIC_VALID_SRC: &str = r#"
#![feature(core_intrinsics)]
use std::intrinsics::{self, AtomicOrdering::*};
fn main() {
    let v = 41u32;
    unsafe {
        let _ = intrinsics::atomic_load::<_, { SeqCst }, false>(&v);
        let mut n = 1u64;
        let _ = intrinsics::atomic_xadd::<_, _, { SeqCst }>(&mut n, 2u64);
        let mut p = &v as *const u32;
        let _ = intrinsics::atomic_xadd::<_, _, { SeqCst }>(&mut p, 1usize);
    }
}
"#;

#[test]
fn convert_mode_atomic_intrinsic_with_bool_value_is_a_compile_failure() {
    let dir = temp_dir("mono-atomic-bool");
    let src = fixture(&dir, "atomic_bool.rs", ATOMIC_BOOL_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "an atomic load of a bool must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("atomic_load")
            && stderr.contains("basic integer or pointer"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_atomic_intrinsics_with_valid_types_convert() {
    let dir = temp_dir("mono-atomic-valid");
    let src = fixture(&dir, "atomic_valid.rs", ATOMIC_VALID_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "valid atomic monomorphizations must convert; stderr={stderr}"
    );
}

const EXACT_DIV_SRC: &str = r#"
#![feature(core_intrinsics)]
fn main() {
    let a = unsafe { std::intrinsics::exact_div(84i32, 7) };
    let b = unsafe { std::intrinsics::exact_div(-84i64, -7) };
    let c = unsafe { std::intrinsics::exact_div(100u32, 10) };
    println!("{a} {b} {c}");
}
"#;

const SIMD_FUNNEL_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::{simd_funnel_shl, simd_funnel_shr};
#[repr(simd)]
#[derive(Clone, Copy, PartialEq)]
struct I32x4([i32; 4]);
fn main() {
    let x1 = I32x4([1, 2, 3, 4]);
    let x2 = I32x4([2, 3, 4, 5]);
    // The funnel shifts are ternary: the concatenated pair
    // (x1 high, x2 low) shifted by the third operand's lanes.
    let l = unsafe { simd_funnel_shl(x1, x2, x1) };
    assert!(l == I32x4([
        (1 << 1) | (2 >> 31),
        (2 << 2) | (3 >> 30),
        (3 << 3) | (4 >> 29),
        (4 << 4) | (5 >> 28),
    ]));
    let r = unsafe { simd_funnel_shr(I32x4([0; 4]), I32x4([16, 32, 64, 128]), x1) };
    assert!(r == I32x4([8, 8, 8, 8]));
    let r2 = unsafe { simd_funnel_shr(x1, I32x4([0; 4]), x1) };
    assert!(r2 == I32x4([1 << 31, 2 << 30, 3 << 29, 4 << 28]));
    // A zero shift takes the near operand whole.
    assert!(unsafe { simd_funnel_shl(x1, x2, I32x4([0; 4])) } == x1);
    assert!(unsafe { simd_funnel_shr(x1, x2, I32x4([0; 4])) } == x2);
    println!("ok");
}
"#;

#[test]
fn run_mode_exact_div_intrinsic() {
    run_ok("exact-div", EXACT_DIV_SRC, "12 12 10\n");
}

const RAW_EQ_SRC: &str = r#"
#![feature(core_intrinsics)]
fn main() {
    let a = [1u8, 2, 3];
    let b = [1u8, 2, 3];
    let c = [1u8, 2, 4];
    let x = unsafe { std::intrinsics::raw_eq(&a, &b) };
    let y = unsafe { std::intrinsics::raw_eq(&a, &c) };
    let z = unsafe { std::intrinsics::raw_eq(&7u32, &7u32) };
    println!("{x} {y} {z}");
}
"#;

#[test]
fn run_mode_raw_eq_intrinsic() {
    run_ok("raw-eq", RAW_EQ_SRC, "true false true\n");
}

#[test]
fn run_mode_simd_funnel_shifts_are_ternary() {
    let dir = temp_dir("simd-funnel");
    let src = fixture(&dir, "funnel.rs", SIMD_FUNNEL_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert!(stdout.contains("ok"), "stdout={stdout}");
}
