// Remaining closure-adjacent run-mode cases: shared closure demand, assert_inhabited, a raw allocator cycle, and pow/parse.

// ----- one closure instance demanded by two collectors (Lane HM) -----------

const SUM_CLOSURE_SRC: &str = r#"
fn main() {
    let v: [i128; 3] = [10, 20, 12];
    let s: i128 = v.iter().sum();
    assert_eq!(s, 42);
    println!("ok");
}
"#;

#[test]
fn run_mode_closure_demanded_by_two_collectors_converts_once() {
    // `<i128 as Sum<&i128>>::sum` creates a closure and hands it
    // to `Iterator::fold`. Two collectors demand that one closure
    // instance: the reachability collection resolves the closures
    // in a call's generic arguments itself (the callee is
    // builtin-owned, so its body is excluded), and the closure
    // demand pass finds the same closure mentioned in the
    // collected `sum` body. Both used to convert it, the program
    // held two bodies under one DefId, and conversion failed with
    // "duplicate function name
    // `<i128 as Sum<&'a i128>>::sum::{closure#0}`" (the
    // tests/ui/numbers-arithmetic/i128.rs shape). The demand
    // passes now skip an instance another collector already
    // converted, and the reachability merges drop duplicates by
    // DefId, so the closure enters the program exactly once.
    let dir = temp_dir("sum-closure");
    let src = fixture(&dir, "sum_closure.rs", SUM_CLOSURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the doubly-demanded closure must convert once and run; stderr={stderr}"
    );
    assert!(
        !stderr.contains("duplicate function name"),
        "no duplicate registration; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const ASSERT_INHABITED_SRC: &str = r#"
#![feature(core_intrinsics)]
fn main() {
    unsafe {
        std::intrinsics::assert_inhabited::<u32>();
        std::intrinsics::assert_inhabited::<[u8; 3]>();
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_assert_inhabited_is_a_no_op() {
    let dir = temp_dir("assert-inhabited");
    let src = fixture(&dir, "assert_inhabited.rs", ASSERT_INHABITED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "assert_inhabited must run as a no-op; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const ALLOC_CYCLE_SRC: &str = r#"
use std::alloc::{alloc, alloc_zeroed, dealloc, realloc, Layout};
fn main() {
    unsafe {
        let layout = Layout::array::<u32>(4).unwrap();
        let p = alloc(layout) as *mut u32;
        assert!(!p.is_null());
        for i in 0..4 { p.add(i).write(10 + i as u32); }
        for i in 0..4 { assert_eq!(*p.add(i), 10 + i as u32); }
        let layout8 = Layout::array::<u32>(8).unwrap();
        let p = realloc(p as *mut u8, layout, layout8.size()) as *mut u32;
        assert!(!p.is_null());
        for i in 0..4 { assert_eq!(*p.add(i), 10 + i as u32); }
        dealloc(p as *mut u8, layout8);
        let zlayout = Layout::array::<u8>(3).unwrap();
        let z = alloc_zeroed(zlayout);
        assert!(!z.is_null());
        assert_eq!(*z, 0);
        assert_eq!(*z.add(2), 0);
        dealloc(z, zlayout);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_raw_allocator_cycle() {
    let dir = temp_dir("alloc-cycle");
    let src = fixture(&dir, "alloc_cycle.rs", ALLOC_CYCLE_SRC);
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const POW_PARSE_SRC: &str = r#"
fn main() {
    assert_eq!(2i32.pow(10), 1024);
    assert_eq!((-3i64).pow(3), -27);
    assert_eq!(7u32.pow(0), 1);
    let x: f64 = "3.1415".parse().unwrap();
    assert!(x > 3.1414 && x < 3.1416);
    let n: i32 = "-42".parse().unwrap();
    assert_eq!(n, -42);
    assert!("abc".parse::<i32>().is_err());
    assert!("99999999999999999999".parse::<i32>().is_err());
    let b: bool = "true".parse().unwrap();
    assert!(b);
    println!("ok");
}
"#;

#[test]
fn run_mode_pow_and_from_str_parse() {
    let dir = temp_dir("pow-parse");
    let src = fixture(&dir, "pow_parse.rs", POW_PARSE_SRC);
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}
