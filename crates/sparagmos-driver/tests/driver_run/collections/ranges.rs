// Range / RangeFrom / StepBy tests.


/// The field-only iterator adapters and the `str` escape
/// adapters have no `Drop` impl of their own: dropping one by
/// value (here, inside the generic `sink`) plans the glue
/// from the fields' evidence instead of refusing the drop.
const RANGE_FROM_SRC: &str = r#"
fn main() {
    let mut it = 3_u8..;
    println!("{:?} {:?}", it.next(), it.next());
    println!("hint {:?}", it.size_hint());
    let v: Vec<u8> = (10_u8..).take(3).collect();
    println!("{:?}", v);
    let mut s = 0_i32;
    for x in 1_i32.. {
        if x > 4 { break; }
        s += x;
    }
    println!("sum {s}");
    let mut it = 100_u32..;
    println!("nth {:?}", it.nth(5));
    println!("after {:?}", it.next());
    let c: Vec<char> = ('x'..).take(3).collect();
    println!("{:?}", c);
}
"#;

/// The unbounded ranges are modelled iterators: the legacy
/// `RangeFrom` step (direct `next`, the `IntoIterator`
/// identity the `for` loop uses, the `nth` / `size_hint`
/// builtins and the consumers over them) and its wrapping
/// successor arithmetic.
#[test]
fn run_mode_range_from_next_and_consumers() {
    run_ok(
        "range-from",
        RANGE_FROM_SRC,
        "Some(3) Some(4)\nhint (18446744073709551615, None)\n[10, 11, 12]\nsum 10\nnth Some(105)\nafter Some(106)\n['x', 'y', 'z']\n",
    );
}

const RANGE_FROM_OVERFLOW_ON_SRC: &str = r#"
fn main() {
    let r = std::panic::catch_unwind(|| {
        let mut it = u8::MAX..;
        assert_eq!(it.next().unwrap(), 255);
        let _ = it.next().unwrap();
    });
    println!("legacy panics {}", r.is_err());
    let r = std::panic::catch_unwind(|| {
        let mut it = i8::MAX..;
        assert_eq!(it.next().unwrap(), 127);
        let _ = it.next().unwrap();
    });
    println!("signed panics {}", r.is_err());
}
"#;

/// With overflow checks on (the driver-run default session),
/// stepping a legacy `RangeFrom` past the element maximum is
/// core's addition-overflow panic, caught by the guest.
#[test]
fn run_mode_range_from_overflow_checks_on() {
    run_ok(
        "range-from-overflow-on",
        RANGE_FROM_OVERFLOW_ON_SRC,
        "legacy panics true\nsigned panics true\n",
    );
}

const RANGE_FROM_OVERFLOW_OFF_SRC: &str = r#"
fn main() {
    let mut it = u8::MAX..;
    println!("{:?} {:?}", it.next(), it.next());
    let mut it = i8::MAX..;
    println!("{:?} {:?}", it.next(), it.next());
}
"#;

/// With overflow checks off, the same step wraps to the
/// element type's minimum and iteration continues.
#[test]
fn run_mode_range_from_overflow_checks_off() {
    let dir = temp_dir("range-from-overflow-off");
    let src = fixture(&dir, "case.rs", RANGE_FROM_OVERFLOW_OFF_SRC);
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "-C", "--rustc-arg", "overflow-checks=no"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "Some(255) Some(0)\nSome(127) Some(-128)\n"
    );
}

const NEW_RANGE_FROM_SRC: &str = r#"
#![feature(new_range_remainder)]
fn main() {
    // The legacy <-> new `From` conversions, both ways.
    let new = std::range::RangeFrom::from(7_u8..);
    let back: std::ops::RangeFrom<u8> = std::ops::RangeFrom::from(new);
    println!("start {}", back.start);
    // The new iterator steps in lockstep with the legacy one,
    // `nth` included (the shape tests/ui/iterators/fromrangeiter.rs pins).
    let mut a = std::range::RangeFrom::from(0_u8..).into_iter();
    let mut b = 0_u8..;
    assert_eq!(a.next(), b.next());
    assert_eq!(a.nth(5), b.nth(5));
    assert_eq!(a.nth(0), b.next());
    println!("lockstep ok");
    // `remainder` reads the state the modelled step maintains.
    let range = std::range::RangeFrom::from(3_u32..);
    let mut it = range.into_iter();
    assert_eq!(it.clone().remainder(), range);
    it.next();
    assert_eq!(it.remainder(), std::range::RangeFrom::from(4_u32..));
    println!("remainder ok");
    // Conversions of the bounded and one-sided pairs as well.
    let r: std::range::Range<u32> = std::range::Range::from(1_u32..5);
    let l: std::ops::Range<u32> = std::ops::Range::from(r);
    println!("range {}..{}", l.start, l.end);
    let t: std::range::RangeToInclusive<u32> =
        std::range::RangeToInclusive::from(..=9_u32);
    let lt: std::ops::RangeToInclusive<u32> = std::ops::RangeToInclusive::from(t);
    println!("to-inclusive {}", lt.end);
}
"#;

/// The new `core::range` types: the `From` conversions from
/// the legacy `ops::` types are field copies, and
/// `RangeFromIter`'s modelled step maintains the
/// [start, exhausted] state its converted `remainder` body
/// reads back.
#[test]
fn run_mode_new_range_from_conversion_and_iter() {
    run_ok(
        "new-range-from",
        NEW_RANGE_FROM_SRC,
        "start 7\nlockstep ok\nremainder ok\nrange 1..5\nto-inclusive 9\n",
    );
}

const RANGE_FROM_ITER_OVERFLOW_SRC: &str = r#"
#![feature(new_range_remainder)]
fn main() {
    // Checks on: the maximum is yielded once, and reading the
    // remainder of the exhausted iterator panics.
    let mut it = std::range::RangeFrom::from(u8::MAX..).into_iter();
    assert_eq!(it.next().unwrap(), 255);
    let r = std::panic::catch_unwind(|| {
        let _ = it.remainder();
    });
    println!("remainder panics {}", r.is_err());
    // A further step of the exhausted iterator panics too.
    let mut it = std::range::RangeFrom::from(u8::MAX..).into_iter();
    assert_eq!(it.next().unwrap(), 255);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = it.next();
    }));
    println!("next panics {}", r.is_err());
}
"#;

#[test]
fn run_mode_range_from_iter_overflow_checks_on() {
    run_ok(
        "range-from-iter-overflow-on",
        RANGE_FROM_ITER_OVERFLOW_SRC,
        "remainder panics true\nnext panics true\n",
    );
}

const RANGE_FROM_ITER_WRAP_SRC: &str = r#"
#![feature(new_range_remainder)]
fn main() {
    let mut it = std::range::RangeFrom::from(u8::MAX..).into_iter();
    assert_eq!(it.next().unwrap(), 255);
    println!("wrapped start {}", it.clone().remainder().start);
    println!("next {:?}", it.next());
}
"#;

/// Checks off: the exhausted flag is never set, the bound
/// wraps, and `remainder` reports the wrapped start.
#[test]
fn run_mode_range_from_iter_overflow_checks_off() {
    let dir = temp_dir("range-from-iter-overflow-off");
    let src = fixture(&dir, "case.rs", RANGE_FROM_ITER_WRAP_SRC);
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "-C", "--rustc-arg", "overflow-checks=no"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "wrapped start 0\nnext Some(0)\n"
    );
}

const STEP_BY_RANGES_SRC: &str = r#"
fn main() {
    let v: Vec<usize> = (0_usize..=10).step_by(2).collect();
    println!("{:?}", v);
    let v: Vec<u32> = (0_u32..10).step_by(3).collect();
    println!("{:?}", v);
    let v: Vec<u8> = (250_u8..=255).step_by(2).collect();
    println!("{:?}", v);
    let v: Vec<i32> = (-5_i32..5).step_by(4).collect();
    println!("{:?}", v);
    let v: Vec<u32> = (0_u32..3).step_by(100).collect();
    println!("{:?}", v);
    let mut it = (10_u32..20).step_by(5);
    println!("{:?} {:?} {:?}", it.next(), it.next(), it.next());
    let v: Vec<i32> = vec![0, 1, 2, 3, 4, 5].into_iter().step_by(2).collect();
    println!("{:?}", v);
}
"#;

/// `StepBy`'s modelled step: the generic `nth`-skip form
/// (over `RangeInclusive`, a signed `Range`, and an owning
/// iterator) and core's specialised counter form over an
/// unsigned integer `Range`, including a step wider than
/// the range and the exact-end boundary.
#[test]
fn run_mode_step_by_over_ranges() {
    run_ok(
        "step-by-ranges",
        STEP_BY_RANGES_SRC,
        "[0, 2, 4, 6, 8, 10]\n[0, 3, 6, 9]\n[250, 252, 254]\n[-5, -1, 3]\n[0]\nSome(10) Some(15) None\n[0, 2, 4]\n",
    );
}const RANGE_STEP_AUX_YES_SRC: &str = r#"
//@ compile-flags: -C overflow-checks=yes
#![crate_type = "lib"]
pub fn next_from(iter: &mut std::ops::RangeFrom<u8>) -> u8 {
    iter.next().unwrap()
}
pub fn next_from_iter(iter: &mut std::range::RangeFromIter<u8>) -> u8 {
    iter.next().unwrap()
}
"#;

const RANGE_STEP_AUX_NO_SRC: &str = r#"
//@ compile-flags: -C overflow-checks=no
#![crate_type = "lib"]
pub fn next_from(iter: &mut std::ops::RangeFrom<u8>) -> u8 {
    iter.next().unwrap()
}
pub fn next_from_iter(iter: &mut std::range::RangeFromIter<u8>) -> u8 {
    iter.next().unwrap()
}
"#;

const RANGE_FROM_STEP_PER_CRATE_MAIN_SRC: &str = r#"
extern crate ocyes;
extern crate ocno;
fn main() {
    // The legacy `RangeFrom` step performed inside an
    // auxiliary crate follows that crate's own
    // overflow-checks setting, not the main compilation's:
    // the no-checks crate wraps at the maximum, the checks
    // crate panics stepping at the maximum.
    let mut it = u8::MAX..;
    assert_eq!(ocno::next_from(&mut it), 255);
    assert_eq!(ocno::next_from(&mut it), 0);
    let mut it = u8::MAX..;
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = ocyes::next_from(&mut it);
    }));
    assert!(r.is_err());
    println!("ok");
}
"#;

/// The `Step::forward` behind a legacy `RangeFrom` step is
/// an `#[rustc_inherit_overflow_checks]` operation: when an
/// auxiliary crate performs the step, the auxiliary crate's
/// setting decides -- under both session configurations of
/// the main compilation.
#[test]
fn run_mode_range_from_step_follows_the_stepping_crate() {
    let dir = temp_dir("range-from-step-per-crate");
    let yes = fixture(&dir, "ocyes.rs", RANGE_STEP_AUX_YES_SRC);
    let no = fixture(&dir, "ocno.rs", RANGE_STEP_AUX_NO_SRC);
    let src = fixture(&dir, "main.rs", RANGE_FROM_STEP_PER_CRATE_MAIN_SRC);
    let yes_arg = format!("ocyes={}", yes.display());
    let no_arg = format!("ocno={}", no.display());
    let out = run_driver_extra(&src, &["--aux", &yes_arg, "--aux", &no_arg]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
    let out = run_driver_extra(
        &src,
        &[
            "--aux",
            &yes_arg,
            "--aux",
            &no_arg,
            "--rustc-arg",
            "-C",
            "--rustc-arg",
            "overflow-checks=no",
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const RANGE_FROM_ITER_STEP_PER_CRATE_MAIN_SRC: &str = r#"
#![feature(new_range_remainder)]
extern crate ocyes;
extern crate ocno;
fn main() {
    // The new `RangeFromIter` step follows the crate that
    // performs it (the shape of
    // tests/ui/iterators/rangefrom-overflow-2crates.rs):
    // stepped by the no-checks crate the bound wraps;
    // stepped by the checks crate the maximum is yielded
    // once and the next step panics -- while the exhausted
    // iterator still wraps in the no-checks crate's hands.
    let mut it = std::range::RangeFrom::from(u8::MAX..).into_iter();
    assert_eq!(ocno::next_from_iter(&mut it), 255);
    assert_eq!(ocno::next_from_iter(&mut it), 0);
    let mut it = std::range::RangeFrom::from(u8::MAX..).into_iter();
    assert_eq!(ocyes::next_from_iter(&mut it), 255);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = ocyes::next_from_iter(&mut it);
    }));
    assert!(r.is_err());
    assert_eq!(ocno::next_from_iter(&mut it), 0);
    println!("ok");
}
"#;

/// The same per-crate resolution for `RangeFromIter`: its
/// step replays core's checked/wrapping forms, and which
/// form applies is the stepping crate's setting.
#[test]
fn run_mode_range_from_iter_step_follows_the_stepping_crate() {
    let dir = temp_dir("range-from-iter-step-per-crate");
    let yes = fixture(&dir, "ocyes.rs", RANGE_STEP_AUX_YES_SRC);
    let no = fixture(&dir, "ocno.rs", RANGE_STEP_AUX_NO_SRC);
    let src = fixture(
        &dir,
        "main.rs",
        RANGE_FROM_ITER_STEP_PER_CRATE_MAIN_SRC,
    );
    let yes_arg = format!("ocyes={}", yes.display());
    let no_arg = format!("ocno={}", no.display());
    let out = run_driver_extra(&src, &["--aux", &yes_arg, "--aux", &no_arg]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
    let out = run_driver_extra(
        &src,
        &[
            "--aux",
            &yes_arg,
            "--aux",
            &no_arg,
            "--rustc-arg",
            "-C",
            "--rustc-arg",
            "overflow-checks=no",
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}
