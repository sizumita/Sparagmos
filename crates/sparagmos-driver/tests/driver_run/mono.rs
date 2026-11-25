use super::{EXTERNAL_DEMAND_SRC, fixture, run_driver, run_ok, summarize_driver, temp_dir};


#[test]
fn run_mode_generic_instance_demanded_through_foreign_body_executes() {
    // Neither generic definition is named by a local body:
    // `Total::sum` (generic over the iterator type) is demanded
    // only from the collected body of `Iterator::sum`, and
    // `RepeatN::next` (a method of a generic impl) only from
    // `Total::sum`'s instance body. Both instances are resolved
    // and converted, so the program runs end to end.
    let dir = temp_dir("external-demand");
    let src = fixture(&dir, "external_demand.rs", EXTERNAL_DEMAND_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "externally demanded generic instances must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "21\n");
}

// ----- driver demand round 2 (Lane CP) -----

const EXT_CTOR_HOF_SRC: &str = r#"
fn apply<F: FnOnce(u32) -> Option<u32>>(f: F, v: u32) -> u32 {
    match f(v) { Some(x) => x, None => 0 }
}
fn main() {
    println!("{}", apply(Some, 40) + 2);
}
"#;

const EXT_CTOR_VALUE_SRC: &str = r#"
fn main() {
    let f = Some;
    let x = f(3);
    match x { Some(v) => println!("{}", v + 1), None => println!("none") }
}
"#;

const EXT_FN_VALUE_SRC: &str = r#"
fn main() {
    let f = std::num::Wrapping;
    let w = f(20u32) + std::num::Wrapping(22u32);
    println!("{}", w.0);
    let g = String::new;
    let mut s = g();
    s.push_str("hi");
    println!("{}", s.len());
}
"#;

#[test]
fn run_mode_foreign_ctor_as_value_executes() {
    let dir = temp_dir("ext-ctor-value");
    let src = fixture(&dir, "ext_ctor_value.rs", EXT_CTOR_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a foreign constructor used as a value must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "4\n");
}

#[test]
fn run_mode_foreign_ctor_passed_to_hof_executes() {
    let dir = temp_dir("ext-ctor-hof");
    let src = fixture(&dir, "ext_ctor_hof.rs", EXT_CTOR_HOF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a foreign constructor passed to a higher-order function must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_foreign_fn_and_tuple_struct_ctor_as_values_execute() {
    let dir = temp_dir("ext-fn-value");
    let src = fixture(&dir, "ext_fn_value.rs", EXT_FN_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a foreign plain function and tuple-struct constructor used as values must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n2\n");
}

const LARGE_BUT_LEGAL_SRC: &str = r#"
struct Big([u8; 1_000_000]);
fn main() { let _b = Big([7u8; 1_000_000]); }
"#;

const POINTER_TO_HUGE_SRC: &str = r#"
fn main() {
    let _p: *const [u8; usize::MAX] = core::ptr::null();
}
"#;

const TOO_BIG_LOCAL_SRC: &str = r#"
fn main() { let _a = [0u8; usize::MAX]; }
"#;

const TOO_BIG_SIZE_OF_SRC: &str = r#"
struct S { x: [u8; usize::MAX] }
fn f() -> usize { std::mem::size_of::<S>() }
fn main() { let _ = f(); }
"#;

const UNBOUNDED_INSTANTIATION_SRC: &str = r#"
fn g<T>() { g::<Option<T>>(); }
fn main() { g::<i32>(); }
"#;

/// The shape of tests/ui/recursion/infinite-instantiation-via-
/// size-increasing-tuple.rs: each instantiation demands the
/// next with the type doubled, so the nesting depth grows only
/// linearly while the type expansion grows exponentially.
const SIZE_INCREASING_TUPLE_SRC: &str = r#"
trait Foo {
    fn recurse(&self);
}

impl<T> Foo for T {
    fn recurse(&self) {
        (self, self).recurse();
    }
}

fn main() {
    ().recurse();
}
"#;

const UNSIZED_NONFINAL_SRC: &str = r#"
#![feature(trivial_bounds)]
#![expect(trivial_bounds)]
struct TwoStrs(str, str)
where
    str: Sized;
fn main() {}
"#;

#[test]
fn convert_mode_large_but_legal_type_converts() {
    let dir = temp_dir("mono-large-legal");
    let src = fixture(&dir, "large_legal.rs", LARGE_BUT_LEGAL_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a large type within the target's bound must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_pointer_to_huge_type_converts() {
    // A pointer to a too-big type is itself an ordinary pointer;
    // merely pointing at the type does not materialize it, so
    // the program converts.
    let dir = temp_dir("mono-pointer-to-huge");
    let src = fixture(&dir, "pointer_to_huge.rs", POINTER_TO_HUGE_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a pointer to a huge type must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_too_big_local_type_is_a_compile_failure() {
    let dir = temp_dir("mono-too-big-local");
    let src = fixture(&dir, "too_big_local.rs", TOO_BIG_LOCAL_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a local of a type too big for the target must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("too big for the target architecture"),
        "the failure must carry rustc's too-big wording; stderr={stderr}"
    );
}

#[test]
fn convert_mode_too_big_size_of_argument_is_a_compile_failure() {
    let dir = temp_dir("mono-too-big-size-of");
    let src = fixture(&dir, "too_big_size_of.rs", TOO_BIG_SIZE_OF_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "measuring a too-big type with size_of must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("too big for the target architecture"),
        "the failure must carry rustc's too-big wording; stderr={stderr}"
    );
}

#[test]
fn convert_mode_unbounded_self_instantiation_is_a_recursion_limit_failure() {
    let dir = temp_dir("mono-recursion-limit");
    let src = fixture(&dir, "unbounded.rs", UNBOUNDED_INSTANTIATION_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "unbounded self-instantiation must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("reached the recursion limit while instantiating"),
        "the failure must name the recursion limit; stderr={stderr}"
    );
}

#[test]
fn convert_mode_unsized_field_in_non_final_position_converts() {
    // A struct with an unsized field in a non-final position is
    // accepted under trivial bounds but can never be
    // materialized; asking the compiler for its layout creates
    // a delayed internal compiler error instead of returning
    // one. The too-big scan must screen the type out and leave
    // the compilation alone.
    let dir = temp_dir("mono-unsized-nonfinal");
    let src = fixture(&dir, "unsized_nonfinal.rs", UNSIZED_NONFINAL_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a never-materializable unsized-field struct must not fail compilation; stderr={stderr}"
    );
}

#[test]
fn run_mode_unbounded_self_instantiation_is_unsupported() {
    let dir = temp_dir("mono-recursion-limit-run");
    let src = fixture(&dir, "unbounded_run.rs", UNBOUNDED_INSTANTIATION_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "run mode reports a monomorphization failure as unsupported; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("reached the recursion limit while instantiating"),
        "the failure must name the recursion limit; stderr={stderr}"
    );
}

#[test]
fn convert_mode_size_increasing_tuple_is_a_size_limit_failure() {
    // The depth and instance-count caps cannot bind here: the
    // demand path nests the definition once per instantiation
    // while the type doubles, so conversion exhausts memory
    // long before either cap is reached. The expanded-size cap
    // must stop it, as the same recursion-limit failure rustc
    // reports.
    let dir = temp_dir("mono-size-increasing-tuple");
    let src = fixture(&dir, "size_increasing.rs", SIZE_INCREASING_TUPLE_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "size-increasing instantiation must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("reached the recursion limit while instantiating")
            && stderr.contains("(instantiation size limit)"),
        "the failure must name the instantiation size limit; stderr={stderr}"
    );
}

const SIMD_SATURATING_FLOAT_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_saturating_add;
#[repr(simd)]
#[derive(Copy, Clone)]
struct F32x4([f32; 4]);
fn main() { unsafe { let _ = simd_saturating_add(F32x4([0.0; 4]), F32x4([0.0; 4])); } }
"#;

const SIMD_BITMASK_BAD_RET_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_bitmask;
#[repr(simd)]
#[derive(Copy, Clone)]
struct M2([i32; 2]);
fn main() { unsafe { let _: u16 = simd_bitmask(M2([0, 0])); } }
"#;

const SIMD_BITMASK_OK_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_bitmask;
#[repr(simd)]
#[derive(Copy, Clone)]
struct M16([i32; 16]);
fn main() { unsafe { let _: u16 = simd_bitmask(M16([0; 16])); } }
"#;

const SIMD_SHUFFLE_OOB_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_shuffle;
#[repr(simd)]
#[derive(Copy, Clone)]
struct F32x2([f32; 2]);
#[repr(simd)]
#[derive(Copy, Clone)]
struct Idx2([u32; 2]);
const IDX: Idx2 = Idx2([4, 0]);
fn main() { unsafe { let _: F32x2 = simd_shuffle(F32x2([0.0, 0.0]), F32x2([1.0, 1.0]), IDX); } }
"#;

const SIMD_SHUFFLE_OK_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_shuffle;
#[repr(simd)]
#[derive(Copy, Clone)]
struct F32x2([f32; 2]);
#[repr(simd)]
#[derive(Copy, Clone)]
struct Idx2([u32; 2]);
const IDX: Idx2 = Idx2([3, 0]);
fn main() { unsafe { let _: F32x2 = simd_shuffle(F32x2([0.0, 0.0]), F32x2([1.0, 1.0]), IDX); } }
"#;

const SIMD_REDUCE_BAD_RET_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_reduce_and;
#[repr(simd)]
#[derive(Copy, Clone)]
struct U32x4([u32; 4]);
fn main() { unsafe { let _: f32 = simd_reduce_and(U32x4([0; 4])); } }
"#;

const SIMD_MASKED_LOAD_BAD_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::{SimdAlign, simd_masked_load};
#[repr(simd)]
#[derive(Copy, Clone)]
struct Simd<T, const N: usize>([T; N]);
fn main() {
    let arr = [4u8, 5, 6, 7];
    let default = Simd::<u8, 4>([9; 4]);
    unsafe {
        let _ = simd_masked_load::<_, _, _, { SimdAlign::Element }>(
            Simd::<i8, 8>([-1, 0, -1, -1, 0, 0, 0, 0]),
            arr.as_ptr(),
            default,
        );
    }
}
"#;

const SIMD_GATHER_BAD_PTR_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_gather;
#[repr(simd)]
#[derive(Copy, Clone)]
struct Simd<T, const N: usize>([T; N]);
fn main() {
    let default = Simd::<f32, 4>([0.0; 4]);
    let mask = Simd::<i32, 4>([-1, -1, -1, -1]);
    unsafe { let _ = simd_gather(default, mask, mask); }
}
"#;

#[test]
fn convert_mode_simd_saturating_add_on_floats_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-saturating");
    let src = fixture(&dir, "saturating.rs", SIMD_SATURATING_FLOAT_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "saturating arithmetic on a float vector must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_saturating_add") && stderr.contains("integer type"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_bitmask_with_wrong_return_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-bitmask-bad");
    let src = fixture(&dir, "bitmask_bad.rs", SIMD_BITMASK_BAD_RET_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a bitmask returned as the wrong integer width must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_bitmask") && stderr.contains("cannot return"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_bitmask_with_matching_return_converts() {
    let dir = temp_dir("mono-simd-bitmask-ok");
    let src = fixture(&dir, "bitmask_ok.rs", SIMD_BITMASK_OK_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a bitmask with the matching return width must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_shuffle_with_out_of_bounds_index_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-shuffle-oob");
    let src = fixture(&dir, "shuffle_oob.rs", SIMD_SHUFFLE_OOB_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a shuffle index past both inputs must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_shuffle") && stderr.contains("less than"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_shuffle_with_in_bounds_index_converts() {
    let dir = temp_dir("mono-simd-shuffle-ok");
    let src = fixture(&dir, "shuffle_ok.rs", SIMD_SHUFFLE_OK_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an in-bounds shuffle must convert; stderr={stderr}"
    );
}

const SIMD_INSERT_OOB_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_insert;
#[repr(simd)]
#[derive(Copy, Clone)]
struct U8x2([u8; 2]);
fn main() { unsafe { simd_insert(U8x2([0, 0]), 2, 0u8); } }
"#;

const SIMD_EXTRACT_OOB_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_extract;
#[repr(simd)]
#[derive(Copy, Clone)]
struct U8x2([u8; 2]);
fn main() { unsafe { let _x: u8 = simd_extract(U8x2([0, 0]), 5); } }
"#;

const SIMD_INSERT_EXTRACT_OK_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::{simd_extract, simd_insert};
#[repr(simd)]
#[derive(Copy, Clone)]
struct U8x2([u8; 2]);
fn main() {
    unsafe {
        let v = simd_insert(U8x2([0, 0]), 1, 7u8);
        let _x: u8 = simd_extract(v, 1);
    }
}
"#;

#[test]
fn convert_mode_simd_insert_with_out_of_bounds_index_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-insert-oob");
    let src = fixture(&dir, "insert_oob.rs", SIMD_INSERT_OOB_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "an insert index past the vector must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_insert") && stderr.contains("less than"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_extract_with_out_of_bounds_index_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-extract-oob");
    let src = fixture(&dir, "extract_oob.rs", SIMD_EXTRACT_OOB_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "an extract index past the vector must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_extract") && stderr.contains("less than"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_insert_extract_with_in_bounds_index_converts() {
    let dir = temp_dir("mono-simd-insert-extract-ok");
    let src = fixture(&dir, "insert_extract_ok.rs", SIMD_INSERT_EXTRACT_OK_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "in-bounds insert/extract must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_reduce_with_wrong_return_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-reduce-bad");
    let src = fixture(&dir, "reduce_bad.rs", SIMD_REDUCE_BAD_RET_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a reduction returned as a non-element type must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_reduce_and") && stderr.contains("expected return type"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_masked_load_with_wrong_length_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-masked-bad");
    let src = fixture(&dir, "masked_bad.rs", SIMD_MASKED_LOAD_BAD_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a masked load whose values length differs from the mask must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_masked_load") && stderr.contains("third argument"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_gather_with_non_pointer_vector_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-gather-bad");
    let src = fixture(&dir, "gather_bad.rs", SIMD_GATHER_BAD_PTR_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a gather whose second argument is not a pointer vector must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_gather") && stderr.contains("pointer"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

// ----- monomorphization validity checks, deep rules (Lane HZ) ---------------

const SIMD_SHUFFLE_ELEM_MISMATCH_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_shuffle;
#[repr(simd)]
#[derive(Copy, Clone)]
struct U32x4([u32; 4]);
#[repr(simd)]
#[derive(Copy, Clone)]
struct Idx2([u32; 2]);
const IDX: Idx2 = Idx2([0, 1]);
fn main() {
    let v = U32x4([0; 4]);
    unsafe { let _: U32x4 = simd_shuffle(v, v, IDX); }
}
"#;

const SIMD_SHUFFLE_VALID_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_shuffle;
#[repr(simd)]
#[derive(Copy, Clone)]
struct U32x4([u32; 4]);
#[repr(simd)]
#[derive(Copy, Clone)]
struct Idx4([u32; 4]);
const IDX: Idx4 = Idx4([0, 1, 2, 3]);
fn main() {
    let v = U32x4([0; 4]);
    unsafe { let _: U32x4 = simd_shuffle(v, v, IDX); }
}
"#;

const SIMD_BITMASK_WIDTH_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_bitmask;
#[repr(simd)]
#[derive(Copy, Clone)]
struct I8x2([i8; 2]);
fn main() {
    let m = I8x2([-1, 0]);
    unsafe { let _: u16 = simd_bitmask(m); }
}
"#;

const MASKED_STORE_CONST_PTR_SRC: &str = r#"
#![feature(portable_simd, core_intrinsics)]
use std::intrinsics::simd::SimdAlign;
use std::{ptr::null, simd::Simd};
const _: () = {
    let c = Simd::from_array([0; 3]);
    unsafe {
        core::intrinsics::simd::simd_masked_store::<_, _, _, { SimdAlign::Element }>(
            c,
            null::<i32>(),
            c,
        )
    };
};
fn main() {}
"#;

#[test]
fn convert_mode_simd_shuffle_with_mismatched_index_length_is_a_compile_failure() {
    let dir = temp_dir("mono-shuffle-len");
    let src = fixture(&dir, "shuffle_len.rs", SIMD_SHUFFLE_ELEM_MISMATCH_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a shuffle whose index length differs from the return length must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("invalid monomorphization of `simd_shuffle` intrinsic"),
        "the failure must be the monomorphization check, not a front-end error; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_shuffle_with_matching_index_converts() {
    let dir = temp_dir("mono-shuffle-ok");
    let src = fixture(&dir, "shuffle_ok.rs", SIMD_SHUFFLE_VALID_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a valid shuffle must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_bitmask_with_wrong_width_is_a_compile_failure() {
    let dir = temp_dir("mono-bitmask-width");
    let src = fixture(&dir, "bitmask_width.rs", SIMD_BITMASK_WIDTH_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a bitmask result wider than the lane count requires must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_bitmask"),
        "the failure must name the intrinsic; stderr={stderr}"
    );
}

#[test]
fn convert_mode_masked_store_with_const_pointer_in_const_converts() {
    let dir = temp_dir("mono-masked-const");
    let src = fixture(&dir, "masked_const.rs", MASKED_STORE_CONST_PTR_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a const-evaluated masked store through a *const pointer must convert; stderr={stderr}"
    );
}


const SIMD_REDUCE_BAD_INITIAL_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_reduce_add_ordered;
#[repr(simd)]
#[derive(Copy, Clone)]
struct F32x4([f32; 4]);
fn reduce<U>(v: F32x4, init: U) -> U { unsafe { simd_reduce_add_ordered(v, init) } }
fn main() { let _ = reduce(F32x4([0.0; 4]), 0i32); }
"#;

#[test]
fn convert_mode_simd_reduce_ordered_with_mismatched_initial_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-reduce-initial");
    let src = fixture(&dir, "reduce_initial.rs", SIMD_REDUCE_BAD_INITIAL_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "an ordered reduction whose initial value is not the element type must fail; stderr={stderr}"
    );
    assert!(
        stderr.contains("simd_reduce_add_ordered") && stderr.contains("initial value"),
        "the failure must name the intrinsic and the initial value; stderr={stderr}"
    );
}

// ----- Required constants of collected instances (mono collector) -----

const REQUIRED_CONST_DEAD_FN_SRC: &str = r#"
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
#[inline(never)]
fn not_called<T>() {
    if false {
        let _ = Fail::<T>::C;
    }
}
#[inline(never)]
fn called<T>() {
    if false {
        not_called::<T>();
    }
}
fn main() {
    called::<i32>();
}
"#;

const REQUIRED_CONST_FNPTR_IN_CONST_SRC: &str = r#"
struct Late<T>(T);
impl<T> Late<T> {
    const FAIL: () = panic!();
    const FNPTR: fn() = || Self::FAIL;
}
#[inline(never)]
fn not_called<T>() {
    if false {
        let _ = Late::<T>::FNPTR;
    }
}
#[inline(never)]
fn called<T>() {
    if false {
        not_called::<T>();
    }
}
fn main() {
    called::<i32>();
}
"#;

const REQUIRED_CONST_UNMENTIONED_SRC: &str = r#"
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
#[inline(never)]
fn not_called<T>() {
    if false {
        let _ = Fail::<T>::C;
    }
}
fn main() {}
"#;

#[test]
fn convert_mode_required_const_in_dead_mention_is_a_compile_failure() {
    // rustc's mono collector evaluates the required constants of
    // every instance it collects, following mentions recorded
    // before optimization: `called::<i32>` mentions
    // `not_called::<i32>` from dead code, and that body's required
    // constant `Fail::<i32>::C` panics in evaluation (E0080), so
    // the crate fails to compile even though the bad constant is
    // never demanded by the converted (post-optimization) bodies.
    let dir = temp_dir("required-const-dead-fn");
    let src = fixture(&dir, "required_const_dead_fn.rs", REQUIRED_CONST_DEAD_FN_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a failing required constant behind a dead-code mention must fail \
         compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("E0080"),
        "the failure must surface the evaluation error; stderr={stderr}"
    );
}

#[test]
fn convert_mode_required_const_behind_fnptr_const_is_a_compile_failure() {
    // The failing constant hides one more hop away: evaluating
    // `Late::<i32>::FNPTR` succeeds and yields a closure pointer,
    // and the collector collects that closure instance, whose own
    // required constant `Late::<i32>::FAIL` fails to evaluate.
    let dir = temp_dir("required-const-fnptr");
    let src = fixture(
        &dir,
        "required_const_fnptr.rs",
        REQUIRED_CONST_FNPTR_IN_CONST_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a failing required constant behind a function-pointer constant \
         must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("E0080"),
        "the failure must surface the evaluation error; stderr={stderr}"
    );
}

#[test]
fn convert_mode_required_const_never_mentioned_converts() {
    // The same failing constant in a generic function that no
    // collected instance mentions: rustc never collects it, never
    // evaluates the constant, and the crate compiles. The check
    // must not fire either — it only walks from the instances
    // collection actually resolved.
    let dir = temp_dir("required-const-unmentioned");
    let src = fixture(
        &dir,
        "required_const_unmentioned.rs",
        REQUIRED_CONST_UNMENTIONED_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing required constant in a never-mentioned generic \
         function must convert; stderr={stderr}"
    );
}

const REQUIRED_CONST_BEHIND_UNREACHABLE_SRC: &str = r#"
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
fn leaf<T>() {
    if false {
        let _ = Fail::<T>::C;
    }
}
fn bad<T>() {
    if false {
        leaf::<T>();
    }
}
fn unused() {
    bad::<i32>();
}
fn main() {}
"#;

#[test]
fn convert_mode_required_const_behind_unreachable_caller_converts() {
    // `bad::<i32>` is demanded by conversion only because the
    // unreachable private function `unused` calls it; the mono
    // collector never roots it, so it never reaches `leaf::<i32>`
    // and never evaluates the panicking constant. rustc compiles
    // the crate, and the check must not fire either: its roots are
    // the collector's roots, not every instance conversion
    // demanded. (The constant sits one mention hop behind the
    // demanded instance because conversion itself materializes the
    // constants of the bodies it converts; what is exercised here
    // is that the check does not start from `bad` at all.)
    let dir = temp_dir("required-const-unreachable");
    let src = fixture(
        &dir,
        "required_const_unreachable.rs",
        REQUIRED_CONST_BEHIND_UNREACHABLE_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing required constant reachable only from an \
         unreachable function must convert; stderr={stderr}"
    );
}

const REQUIRED_CONST_BEHIND_FN_ITEM_VALUE_SRC: &str = r#"
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
fn bad<T>() {
    if false {
        let _ = Fail::<T>::C;
    }
}
#[inline(never)]
fn hold<U>(x: U) {
    std::hint::black_box(x);
}
#[inline(never)]
fn outer<T>() {
    hold(bad::<T>);
}
fn main() {
    if false {
        outer::<i32>();
    }
}
"#;

#[test]
fn convert_mode_required_const_behind_fn_item_value_converts() {
    // Passing the function item `bad::<T>` as a value to a generic
    // function that never calls it does not reify a function
    // pointer and does not codegen `bad`'s body: a bare `FnDef`
    // constant operand is a zero-sized function-item value, not a
    // collected item. The walk reaches `outer::<i32>` through its
    // mention from `main`, but must not demand `bad::<i32>` from
    // the operand, so the panicking constant is never evaluated
    // and the crate converts, as it does under rustc.
    let dir = temp_dir("required-const-fn-item-value");
    let src = fixture(
        &dir,
        "required_const_fn_item_value.rs",
        REQUIRED_CONST_BEHIND_FN_ITEM_VALUE_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing required constant behind a bare function-item \
         value must convert; stderr={stderr}"
    );
}
const REQUIRED_CONST_BEHIND_UNCALLED_CLOSURE_SRC: &str = r#"
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
#[inline(never)]
fn hold<U>(x: U) {
    std::hint::black_box(x);
}
#[inline(never)]
fn outer<T>() {
    hold(|| {
        if false {
            let _ = Fail::<T>::C;
        }
    });
}
fn main() {
    if false {
        outer::<i32>();
    }
}
"#;

#[test]
fn convert_mode_required_const_behind_uncalled_closure_converts() {
    // Passing a captureless closure as a value to a generic
    // function that never calls it does not codegen the closure
    // body: a bare closure constant operand is a zero-sized
    // value, not a collected item. The walk reaches
    // `outer::<i32>` through its mention from `main`, but must
    // not demand the closure instance from the operand, so the
    // panicking constant in the closure body is never evaluated
    // and the crate converts, as it does under rustc.
    let dir = temp_dir("required-const-uncalled-closure");
    let src = fixture(
        &dir,
        "required_const_uncalled_closure.rs",
        REQUIRED_CONST_BEHIND_UNCALLED_CLOSURE_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing required constant behind an uncalled closure \
         value must convert; stderr={stderr}"
    );
}

const REQUIRED_CONST_IN_CALLED_CLOSURE_SRC: &str = r#"
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
fn main() {
    let f = || {
        let _ = Fail::<i32>::C;
    };
    f();
}
"#;

#[test]
fn convert_mode_required_const_in_called_closure_is_a_compile_failure() {
    // A closure that is actually called IS collected: the call
    // names the `FnOnce` trait method, whose resolution reaches
    // the closure instance, and the collector evaluates the
    // closure body's required constants. `Fail::<i32>::C`
    // panics in evaluation (E0080), so the crate fails to
    // compile, exactly as under rustc.
    let dir = temp_dir("required-const-called-closure");
    let src = fixture(
        &dir,
        "required_const_called_closure.rs",
        REQUIRED_CONST_IN_CALLED_CLOSURE_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a failing required constant in a called closure must \
         fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("E0080"),
        "the failure must surface the evaluation error; stderr={stderr}"
    );
}

const REQUIRED_CONST_IN_COERCED_CLOSURE_SRC: &str = r#"
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
fn main() {
    let f: fn() = || {
        let _ = Fail::<i32>::C;
    };
    std::hint::black_box(f);
}
"#;

#[test]
fn convert_mode_required_const_in_coerced_closure_is_a_compile_failure() {
    // A closure coerced to a function pointer IS collected:
    // the coercion is a mentioned item that resolves the
    // closure instance, and the collector evaluates its
    // required constants. `Fail::<i32>::C` panics in evaluation
    // (E0080), so the crate fails to compile, exactly as under
    // rustc.
    let dir = temp_dir("required-const-coerced-closure");
    let src = fixture(
        &dir,
        "required_const_coerced_closure.rs",
        REQUIRED_CONST_IN_COERCED_CLOSURE_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a failing required constant in a closure coerced to a \
         function pointer must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("E0080"),
        "the failure must surface the evaluation error; stderr={stderr}"
    );
}

const REQUIRED_CONST_UNSIZE_DOUBLE_REF_SRC: &str = r#"
trait Trait {
    fn method(&self) -> usize;
}
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
struct W<T>(T);
impl<T> Trait for W<T> {
    fn method(&self) -> usize {
        if false {
            let _ = Fail::<T>::C;
        }
        1
    }
}
impl<T> Trait for &W<T> {
    fn method(&self) -> usize {
        42
    }
}
fn main() {
    let w = W(7u8);
    let d: &dyn Trait = &&w;
    println!("{}", d.method());
}
"#;

#[test]
fn convert_mode_required_const_behind_sibling_unsize_impl_converts() {
    // The unsizing cast `&&W<u8>` -> `&dyn Trait` forms the
    // vtable for `&W<u8>`, whose method carries no required
    // constants. The sibling implementation for `W<u8>` is never
    // collected, so its method's failing required constant is
    // never evaluated and the crate compiles, exactly as under
    // rustc. (A tail search that peels the cast's source and
    // target independently strips both source references, forms
    // the vtable for `W<u8>` instead, and wrongly rejects this.)
    let dir = temp_dir("required-const-unsize-double-ref");
    let src = fixture(
        &dir,
        "required_const_unsize_double_ref.rs",
        REQUIRED_CONST_UNSIZE_DOUBLE_REF_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing required constant in the sibling impl of an \
         unsizing cast must convert; stderr={stderr}"
    );
}

const REQUIRED_CONST_UNSIZE_DEAD_GENERIC_SRC: &str = r#"
trait Trait {
    fn method(&self) -> usize;
}
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
struct W<T>(T);
impl<T> Trait for W<T> {
    fn method(&self) -> usize {
        if false {
            let _ = Fail::<T>::C;
        }
        1
    }
}
impl<T> Trait for &W<T> {
    fn method(&self) -> usize {
        42
    }
}
#[inline(never)]
fn cast_dead<T>(w: &W<T>) {
    let d: &dyn Trait = &w;
    println!("{}", d.method());
}
#[inline(never)]
fn caller<T>(w: &W<T>) {
    if false {
        cast_dead::<T>(w);
    }
}
fn main() {
    let w = W(7u8);
    caller::<u8>(&w);
    println!("done");
}
"#;

#[test]
fn convert_mode_required_const_behind_sibling_unsize_impl_in_dead_generic_converts() {
    // The same cast one mention further away: the generic
    // function holding it is mentioned only from dead code of a
    // called function, which the walk still follows. The cast
    // forms the vtable for `&W<u8>`, so the failing required
    // constant in `W<u8>`'s sibling implementation is never
    // evaluated and the crate compiles.
    let dir = temp_dir("required-const-unsize-dead-generic");
    let src = fixture(
        &dir,
        "required_const_unsize_dead_generic.rs",
        REQUIRED_CONST_UNSIZE_DEAD_GENERIC_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing required constant in the sibling impl of an \
         unsizing cast inside a dead-mentioned generic function \
         must convert; stderr={stderr}"
    );
}

const REQUIRED_CONST_UNSIZE_VTABLE_SRC: &str = r#"
trait Trait {
    fn method(&self) -> usize;
}
struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}
struct W<T>(T);
impl<T> Trait for W<T> {
    fn method(&self) -> usize {
        if false {
            let _ = Fail::<T>::C;
        }
        1
    }
}
impl<T> Trait for &W<T> {
    fn method(&self) -> usize {
        42
    }
}
fn main() {
    let w = W(7u8);
    let d: &dyn Trait = &w;
    println!("{}", d.method());
}
"#;

#[test]
fn convert_mode_required_const_behind_unsize_vtable_is_a_compile_failure() {
    // The direct shape, as a control: `&W<u8>` unsizing to
    // `&dyn Trait` forms the vtable for `W<u8>` itself, so the
    // collector collects its method and evaluates its failing
    // required constant (E0080); the crate fails to compile,
    // exactly as under rustc.
    let dir = temp_dir("required-const-unsize-vtable");
    let src = fixture(
        &dir,
        "required_const_unsize_vtable.rs",
        REQUIRED_CONST_UNSIZE_VTABLE_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a failing required constant behind an unsizing cast's \
         own vtable must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("E0080"),
        "the failure must surface the evaluation error; stderr={stderr}"
    );
}

const ELEM_BLOCK_NEWTYPE_SRC: &str = r#"
use std::alloc::{alloc, dealloc, Layout};
#[derive(Clone, Copy)]
struct Inner { a: i32, b: i32 }
struct Outer { inner: Inner }
fn main() {
    unsafe {
        let layout = Layout::new::<[Outer; 2]>();
        let p = alloc(layout) as *mut Outer;
        p.write(Outer { inner: Inner { a: 10, b: 20 } });
        p.add(1).write(Outer { inner: Inner { a: 30, b: 40 } });
        let inner = (*p).inner;
        println!("{} {}", inner.a, inner.b);
        let inner2 = (*p.add(1)).inner;
        println!("{} {}", inner2.a, inner2.b);
        (*p).inner = Inner { a: 1, b: 2 };
        let inner3 = (*p).inner;
        println!("{} {}", inner3.a, inner3.b);
        dealloc(p as *mut u8, layout);
    }
}
"#;

#[test]
fn run_mode_elem_block_genuine_newtype_field_read_write() {
    // A genuine one-field struct (`Outer`) around an aggregate
    // (`Inner`) in a raw allocation block is a real field, not a
    // `MaybeUninit` wrapper: reading the field yields the whole
    // aggregate (not its first field), and writing it replaces
    // the field (not the element). The pointee field-0 rewrite
    // and the resolver's bare-wrapper judgment must both stand
    // down for it, judged by the place's declared type and by
    // the element's own stored form.
    run_ok(
        "elemblock-newtype",
        ELEM_BLOCK_NEWTYPE_SRC,
        "10 20\n30 40\n1 2\n",
    );
}

const SLICE_WRAP_CAST_SRC: &str = r#"
fn main() {
    let b: Box<[i32]> = Box::new([1, 2, 3]);
    let p: *const [i32] = &*b;
    let q = p as *const [std::mem::MaybeUninit<i32>];
    let _ = q;
    println!("{}", unsafe { (*p)[2] });
    println!("{}", b[1]);
}
"#;

#[test]
fn run_mode_slice_wrap_cast_preserves_original_view() {
    // Casting a slice pointer to the `MaybeUninit` element view
    // (without ever dereferencing the result) must not rewrite
    // the backing elements: the original view keeps reading the
    // original integers.
    run_ok("slice-wrap-cast", SLICE_WRAP_CAST_SRC, "3\n2\n");
}

const SLICE_UNWRAP_TWICE_SRC: &str = r#"
struct N(i32);
fn main() {
    let v: Vec<std::mem::MaybeUninit<N>> = vec![std::mem::MaybeUninit::new(N(7)), std::mem::MaybeUninit::new(N(8))];
    let s1: &[N] = unsafe { &*(&v[..] as *const [std::mem::MaybeUninit<N>] as *const [N]) };
    println!("{}", s1[0].0);
    let s2: &[N] = unsafe { &*(&v[..] as *const [std::mem::MaybeUninit<N>] as *const [N]) };
    println!("{}", s2[1].0);
}
"#;

#[test]
fn run_mode_slice_unwrap_casts_keep_genuine_newtype() {
    // Repeated unwrap casts over a slice of wrapped genuine
    // one-field elements (`N`) must not strip the newtype layer:
    // the elements read back as `N` values through both casts.
    run_ok("slice-unwrap-twice", SLICE_UNWRAP_TWICE_SRC, "7\n8\n");
}

const ELEM_BLOCK_PARTIAL_INIT_SRC: &str = r#"
use std::alloc::{alloc, dealloc, Layout};
struct S { first: (i32, i32), second: i32 }
fn main() {
    unsafe {
        let layout = Layout::new::<S>();
        let p = alloc(layout) as *mut S;
        (*p).first = (10, 20);
        (*p).second = 30;
        println!("{} {}", (*p).first.0, (*p).first.1);
        println!("{}", (*p).second);
        dealloc(p as *mut u8, layout);
    }
}
"#;

#[test]
fn run_mode_elem_block_partial_struct_init_keeps_siblings() {
    // A struct in a raw allocation block initialized field by
    // field: after the first (aggregate) field is stored, the
    // partially materialized value has the shape of a one-field
    // wrapper around an aggregate, but the second field's write
    // must grow the struct's own field vector, not descend into
    // the first field and corrupt it. The navigation's wrapper
    // descent is gated on the declared type: the stored field
    // is the declared type's own first field, not a complete
    // value of it.
    run_ok(
        "elemblock-partial-init",
        ELEM_BLOCK_PARTIAL_INIT_SRC,
        "10 20\n30\n",
    );
}
