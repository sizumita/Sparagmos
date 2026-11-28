//! Deref patterns whose scrutinee chains run through a
//! one-element vector and through a `Cow` of a `Cow`
//! (the rustc suite programs `pattern/deref-patterns/branch.rs`
//! and `pattern/deref-patterns/implicit-cow-deref.rs`, reduced
//! to the shapes that failed; directives stripped, the feature
//! attribute is accepted because the driver is built and run
//! with RUSTC_BOOTSTRAP=1, like compiletest).

use super::{fixture, run_driver, temp_dir};

// Matching a `Vec<Vec<u32>>` against a nested deref pattern
// switches on the inner vector's single element: the place
// resolves to an index run over the outer vector's aggregate,
// and the run must stay nested -- the inner vector holding
// exactly one scalar element has the shape of a `MaybeUninit`
// wrapper over a scalar, which the projection composition
// mistook for a flat-sequence position and read the inner
// vector itself as the element.
const NESTED_ONE_ELEMENT_SRC: &str = r#"#![feature(deref_patterns)]

fn main() {
    assert!(matches!(vec![vec![1]], deref!([deref!([1])])));
    assert!(matches!(vec![vec![1, 2, 3]], deref!([deref!([1, _, 3])])));
    assert!(!matches!(vec![vec![2]], deref!([deref!([1])])));

    let v = vec![vec![7]];
    let s: &[i32] = &v[0];
    match s[0] {
        7 => {}
        _ => panic!("wrong element"),
    }
    let x = s[0];
    assert_eq!(x, 7);
}"#;

#[test]
fn run_mode_deref_pattern_nested_vec_one_element() {
    let dir = temp_dir("deref-pattern-nested-vec");
    let src = fixture(&dir, "nested_vec.rs", NESTED_ONE_ELEMENT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}

// Matching a `Cow<Cow<[u8]>>` against a slice pattern chains
// two `Deref::deref` calls for `Cow`: the outer call's target
// is the inner `Cow` itself, so its result must be a reference
// to the owned field -- the bare field value leaves the second
// call holding the field's contents (the borrowed slice) where
// it expects the `Cow` aggregate.
const COW_OF_COW_SRC: &str = r#"#![feature(deref_patterns)]

use std::borrow::Cow;

fn main() {
    let cow: Cow<'static, [u8]> = Cow::Borrowed(&[1, 2, 3]);
    let cow_of_cow: Cow<'_, Cow<'static, [u8]>> = Cow::Owned(cow);

    match cow_of_cow {
        [..] => {}
    }

    match cow_of_cow {
        Cow::Borrowed(_) => unreachable!(),
        Cow::Owned(_) => {}
    }
}"#;

#[test]
fn run_mode_deref_pattern_cow_of_cow() {
    let dir = temp_dir("deref-pattern-cow-of-cow");
    let src = fixture(&dir, "cow_of_cow.rs", COW_OF_COW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}
