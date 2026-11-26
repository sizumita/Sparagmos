// Iterator adapters and consumers.


const ADAPTER_INTO_ITER_COLLECT_SRC: &str = r#"
fn main() {
    let v = vec![5, 12, 7, 20];
    let out: Vec<i32> = v.into_iter().map(|x| x + 1).collect();
    println!("map {out:?}");
    let e: Vec<i32> = Vec::new();
    let empty: Vec<i32> = e.into_iter().map(|x| x + 1).collect();
    println!("empty {}", empty.len());
    let n = vec![1, 2, 3];
    let nested: Vec<i32> = n.into_iter().map(|x| x * 2).map(|x| x + 10).collect();
    println!("nested {nested:?}");
    let t = vec![10, 20, 30, 40, 50];
    let taken: Vec<i32> = t.into_iter().take(2).collect();
    println!("take {taken:?}");
    let r = vec![1, 2, 3, 4];
    let rev: Vec<i32> = r.into_iter().rev().collect();
    println!("rev {rev:?}");
    let w = vec!["a".to_string(), "bb".to_string()];
    let strs: Vec<String> = w
        .into_iter()
        .map(|mut s| {
            s.push_str("!");
            s
        })
        .collect();
    println!("strs {strs:?}");
}
"#;

#[test]
fn run_mode_adapter_over_vec_into_iter_collect() {
    // Adapters over `vec::IntoIter` collect through the
    // specialized `SpecFromIter::from_iter`, whose body calls
    // `std::vec::in_place_collect::from_iter_in_place` through a
    // function-pointer constant — a shape the name-based
    // collection never sees, so the raw-allocation body was
    // collected by the const-fn demand pass and ran into the
    // body-less leaf specializations (`SourceIter::as_inner`
    // and friends). The call is served by the
    // `VecFromIterNested` builtin, the entry point's own model
    // (drain the iterator into a fresh `Vec`).
    let dir = temp_dir("adapter-into-iter-collect");
    let src = fixture(&dir, "adaptercollect.rs", ADAPTER_INTO_ITER_COLLECT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "map [6, 13, 8, 21]\nempty 0\nnested [12, 14, 16]\ntake [10, 20]\nrev [4, 3, 2, 1]\nstrs [\"a!\", \"bb!\"]\n",
        "stdout={stdout}"
    );
}

const FILTER_ZIP_INTO_ITER_COLLECT_SRC: &str = r#"
fn main() {
    let v = vec![1, 2, 3, 4, 5, 6];
    let evens: Vec<i32> = v.into_iter().filter(|x| x % 2 == 0).collect();
    println!("filter {evens:?}");
    let a = vec![1, 2, 3];
    let b = vec![10, 20, 30, 40];
    let pairs: Vec<(i32, i32)> = a.into_iter().zip(b.into_iter()).collect();
    println!("zip {pairs:?}");
    let e: Vec<i32> = Vec::new();
    let no_pairs: Vec<(i32, i32)> = e.into_iter().zip(vec![1]).collect();
    println!("zip-empty {}", no_pairs.len());
    let w = vec![String::from("a"), String::from("bb")];
    let cloned: Vec<String> = w.iter().cloned().collect();
    println!("cloned {cloned:?}");
}
"#;

#[test]
fn run_mode_filter_zip_cloned_over_vec_into_iter_collect() {
    // The same specialization path as the Map case, for the
    // adapters the iterator framework models: `Filter` over
    // `vec::IntoIter` delegates to the in-place entry, and
    // `Zip` (no single source allocation to reuse) delegates
    // to the nested entry's collected impl body, which drives
    // `Vec::with_capacity` and the raw-buffer machinery. Both
    // calls are served by the `VecFromIterNested` builtin.
    let dir = temp_dir("filter-zip-into-iter-collect");
    let src = fixture(
        &dir,
        "filterzipcollect.rs",
        FILTER_ZIP_INTO_ITER_COLLECT_SRC,
    );
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "filter [2, 4, 6]\nzip [(1, 10), (2, 20), (3, 30)]\nzip-empty 0\ncloned [\"a\", \"bb\"]\n",
        "stdout={stdout}"
    );
}

const GUEST_ITERATOR_COLLECT_SRC: &str = r#"
struct Counter { n: u32, max: u32 }
impl Iterator for Counter {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.n < self.max { self.n += 1; Some(self.n) } else { None }
    }
}
struct Wrap<M> { inner: M }
impl<M: Iterator> Iterator for Wrap<M> {
    type Item = M::Item;
    fn next(&mut self) -> Option<M::Item> { self.inner.next() }
}
fn main() {
    let v: Vec<u32> = Counter { n: 0, max: 3 }.collect();
    println!("direct {v:?}");
    let m: Vec<u32> = Counter { n: 0, max: 3 }.map(|x| x * 10).collect();
    println!("map {m:?}");
    let w: Vec<u32> = Wrap { inner: Counter { n: 0, max: 2 } }.collect();
    println!("wrap {w:?}");
    let mut c = Counter { n: 0, max: 2 };
    let mut seen = Vec::new();
    for x in &mut c { seen.push(x); }
    println!("byref {seen:?} after {}", c.n);
}
"#;

#[test]
fn run_mode_guest_iterator_collect_and_adapters() {
    // A user-defined iterator is not a type the modelled
    // `next` state machine serves, so the builtin consumers'
    // drain reaches the machine with a tag no arm claims; the
    // step is the type's own collected `next` body, run on the
    // state parked in an allocation so its mutations stick
    // (the `byref` line's counter is the witness). The generic
    // `Wrap` impl has no monomorphic instance as an item and
    // no call edge names it -- the consumer's call is
    // builtin-served -- so its instance is brought in by the
    // iterator-next demand pass, like the generic `Drop` impls
    // the drop demand brings in.
    let dir = temp_dir("guest-iterator-collect");
    let src = fixture(&dir, "guestiter.rs", GUEST_ITERATOR_COLLECT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "direct [1, 2, 3]\nmap [10, 20, 30]\nwrap [1, 2]\nbyref [1, 2] after 2\n",
        "stdout={stdout}"
    );
}

const ITER_COUNT_SRC: &str = r#"
fn main() {
    let a = [10, 20, 30];
    println!("range={}", (0..5).count());
    println!("slice={}", a.iter().count());
    println!("chars={}", "abcd".chars().count());
}
"#;

#[test]
fn run_mode_iterator_count_over_modelled_iterators() {
    let dir = temp_dir("iter_count");
    let src = fixture(&dir, "iter_count.rs", ITER_COUNT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "count runs; stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "range=5\nslice=3\nchars=4\n"
    );
}

const ITER_FOR_EACH_SRC: &str = r#"
fn main() {
    let mut total = 0;
    (0..5).for_each(|x| total += x);
    let mut seen = String::new();
    "abc".chars().for_each(|c| seen.push(c));
    println!("total={} seen={}", total, seen);
}
"#;

#[test]
fn run_mode_iterator_for_each_mutates_captures() {
    let dir = temp_dir("iter_for_each");
    let src = fixture(&dir, "iter_for_each.rs", ITER_FOR_EACH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "for_each runs; stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "total=10 seen=abc\n");
}

const ITER_COLLECT_UNSUPPORTED_SRC: &str = r#"
fn main() {
    let list: std::collections::LinkedList<i32> = (0..3).collect();
    println!("n={}", list.len());
}
"#;

#[test]
fn run_mode_iterator_collect_names_unmodelled_targets() {
    // Targets without a value-model shape fail with the target
    // named, not with an approximation or a conversion error
    // from deep inside the collected FromIterator chain.
    // (`HashSet` / `HashMap` used to be this test's example;
    // since #350 their collected bodies execute instead, and
    // `BTreeSet` / `BTreeMap` followed in #545, so the
    // still-unmodelled `LinkedList` stands in.)
    let dir = temp_dir("iter_collect_unsupported");
    let src = fixture(&dir, "iter_collect_unsupported.rs", ITER_COLLECT_UNSUPPORTED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "unmodelled collect target must exit 2; stderr={stderr}"
    );
    assert!(
        stderr.contains("collect into std::collections::LinkedList")
            && stderr.contains("is not runtime-supported"),
        "the target is named in the reason; stderr={stderr}"
    );
}

const OPTION_BY_VALUE_SRC: &str = r#"
fn main() {
    let mut n = 0;
    for x in Some(41) {
        n += x + 1;
    }
    for x in None::<i32> {
        n += 100;
    }
    let o = Some(String::from("hi"));
    for s in o {
        n += s.len() as i32;
    }
    println!("n={n}");
    let mut m = Some(7);
    for x in &m {
        n += x;
    }
    for x in &mut m {
        *x += 1;
    }
    println!("m={}", m.unwrap_or(0));
    let p = Some(9);
    let mut t = 0;
    for x in p.iter() {
        t += x;
    }
    println!("t={t} n={n}");
}
"#;

#[test]
fn run_mode_option_iteration() {
    let dir = temp_dir("option-iteration");
    let src = fixture(&dir, "option_iteration.rs", OPTION_BY_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Option iteration (by value, by reference, .iter()) must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "n=44\nm=8\nt=9 n=51\n",
        "Some yields its payload once, None yields nothing"
    );
}

const MAP_SRC: &str = r#"
fn double(x: i32) -> i32 { x * 2 }
fn main() {
    let mut s = 0;
    for x in (1..4).map(|x| x * 10) {
        s += x;
    }
    println!("s={s}");
    let mut calls = 0;
    let v = vec![1, 2, 3];
    let mut t = 0;
    for x in v.into_iter().map(|x| { calls += 1; x + calls }) {
        t += x;
    }
    println!("t={t} calls={calls}");
    let mut u = 0;
    for x in (5..8).map(double) {
        u += x;
    }
    println!("u={u}");
}
"#;

#[test]
fn run_mode_map_iteration() {
    // The FnMut case pins that capture writes persist across
    // steps (the closure is called through its own storage).
    let dir = temp_dir("map-iteration");
    let src = fixture(&dir, "map_iteration.rs", MAP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Map over Range and vec::IntoIter must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "s=60\nt=12 calls=3\nu=36\n",
        "closure, capturing closure and fn pointer all apply per element"
    );
}

const ADAPTERS_SRC: &str = r#"
use std::iter::{empty, once, repeat};
fn main() {
    let a = [10, 20, 30];
    let mut e = 0;
    for (i, x) in a.iter().enumerate() {
        e += (i as i32) * x;
    }
    println!("e={e}");
    let mut c = 0;
    for x in a.iter().copied() {
        c += x;
    }
    println!("c={c}");
    let mut t = 0;
    for x in (0..100).take(3) {
        t += x;
    }
    println!("t={t}");
    let mut o = 0;
    for x in once(5) {
        o += x;
    }
    for x in empty::<i32>() {
        o += 100;
    }
    println!("o={o}");
    let mut r = 0;
    for x in repeat(2).take(3) {
        r += x;
    }
    println!("r={r}");
    let v = vec![1, 2, 3, 1, 4];
    let mut s = 0;
    for x in v.into_iter().skip_while(|x| *x < 3) {
        s += x;
    }
    println!("s={s}");
    let mut calls = 0;
    let w = vec![5, 6, 7];
    let mut u = 0;
    for x in w.into_iter().skip_while(|x| { calls += 1; *x < 6 }) {
        u += x;
    }
    println!("u={u} calls={calls}");
}
"#;

#[test]
fn run_mode_adapter_iteration() {
    // The SkipWhile capture count pins that the predicate is not
    // consulted again after its first failure.
    let dir = temp_dir("adapter-iteration");
    let src = fixture(&dir, "adapter_iteration.rs", ADAPTERS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Enumerate/Copied/Take/Once/Empty/Repeat/SkipWhile must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "e=80\nc=60\nt=3\no=5\nr=6\ns=8\nu=13 calls=2\n",
        "each adapter yields exactly the elements its semantics prescribe"
    );
}

const FILTER_SRC: &str = r#"
fn main() {
    let v = vec![1, 2, 3, 4, 5, 6];
    let mut sum = 0;
    for x in v.iter().filter(|x| **x % 2 == 0) {
        sum += x;
    }
    println!("sum={sum}");
    let mut calls = 0;
    let mut it = v.iter().filter(|x| { calls += 1; **x % 2 == 0 });
    println!("{:?}", it.next());
    println!("{:?}", it.next());
    println!("{:?}", it.next());
    println!("calls={calls}");
    let kept: Vec<i32> = (0..10).filter(|x| x % 3 == 0).collect();
    println!("{kept:?}");
    let n = (0..20).filter(|x| x % 2 == 1).count();
    println!("n={n}");
    let it2 = v.iter().filter(|x| **x > 1);
    println!("hint={:?}", it2.size_hint());
}
"#;

#[test]
fn run_mode_filter_iteration() {
    // The call count pins that the predicate runs once per
    // pulled element and that its capture writes persist
    // across steps (the predicate is called through its own
    // storage); the size hint is core's `(0, inner upper)`.
    let dir = temp_dir("filter-iteration");
    let src = fixture(&dir, "filter_iteration.rs", FILTER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Filter over slice::Iter and Range must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "sum=12\nSome(2)\nSome(4)\nSome(6)\ncalls=6\n[0, 3, 6, 9]\nn=10\nhint=(0, Some(6))\n",
        "the predicate accepts exactly the even elements, once per pull"
    );
}

const FILTER_BACK_SRC: &str = r#"
fn main() {
    let v = vec![1, 2, 3, 4, 5, 6];
    let mut it = v.iter().filter(|x| **x % 2 == 0);
    println!("{:?}", it.next());
    println!("{:?}", it.next_back());
    println!("{:?}", it.next());
    println!("{:?}", it.next_back());
    println!("{:?}", it.next());
    let mut rsum = 0;
    for x in (0..10).filter(|x| x % 2 == 0).rev() {
        rsum += x;
    }
    println!("revsum={rsum}");
    let out: Vec<i32> = v.iter().filter(|x| **x % 2 == 0).copied().collect();
    println!("{out:?}");
    let out2: Vec<i32> = v.iter().copied().filter(|x| x % 2 == 1).collect();
    println!("{out2:?}");
    let mut seen = 0;
    let stateful: Vec<i32> = (0..10).filter(move |x| {
        if *x % 2 == 0 {
            seen += 1;
            seen % 2 == 0
        } else {
            false
        }
    }).collect();
    println!("{stateful:?}");
}
"#;

#[test]
fn run_mode_filter_next_back_and_composition() {
    // The interleaved steps pin that front and back pulls
    // meet in the middle without re-yielding an element; the
    // stateful predicate pins that a by-value capture's
    // writes persist from one pull to the next.
    let dir = temp_dir("filter-next-back");
    let src = fixture(&dir, "filter_next_back.rs", FILTER_BACK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Filter next_back, Rev and Copied composition must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "Some(2)\nSome(6)\nSome(4)\nNone\nNone\nrevsum=20\n[2, 4, 6]\n[1, 3, 5]\n[2, 6]\n",
        "front and back steps share one predicate-gated sequence"
    );
}

const FILTER_MAP_SRC: &str = r#"
fn keep_even_times_ten(x: i32) -> Option<i32> {
    if x % 2 == 0 { Some(x * 10) } else { None }
}

fn main() {
    let v = vec![1, 2, 3, 4, 5, 6];
    let kept: Vec<i32> = v.iter().filter_map(|x| if *x % 2 == 0 { Some(*x * 10) } else { None }).collect();
    println!("{kept:?}");
    let kept2: Vec<i32> = (0..7).filter_map(keep_even_times_ten).collect();
    println!("{kept2:?}");
    let mut calls = 0;
    let mut it = v.iter().filter_map(|x| { calls += 1; if *x % 3 == 0 { Some(*x) } else { None } });
    println!("{:?}", it.next());
    println!("{:?}", it.next());
    println!("{:?}", it.next());
    println!("calls={calls}");
    let n = (0..10).filter_map(|x| if x > 100 { Some(x) } else { None }).count();
    println!("n={n}");
    let it2 = v.iter().filter_map(|x| Some(*x));
    println!("hint={:?}", it2.size_hint());
    let text = "a,,bb,ccc";
    let base = 100usize;
    let lens: Vec<usize> = text
        .split(',')
        .filter(|s| !s.is_empty())
        .filter_map(|s| if s.len() > 1 { Some(base + s.len()) } else { None })
        .collect();
    println!("{lens:?}");
}
"#;

#[test]
fn run_mode_filter_map_iteration() {
    // The call count pins that the closure runs once per pulled
    // element and that its capture writes persist across steps
    // (the closure is called through its own storage); the
    // split/filter/filter_map chain is the `parse_lossy` shape
    // of the rustdoc execution chain (Issue #438), and the size
    // hint is core's `(0, inner upper)`.
    let dir = temp_dir("filter-map-iteration");
    let src = fixture(&dir, "filter_map_iteration.rs", FILTER_MAP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "FilterMap over slice::Iter, Range and the Split chain must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[20, 40, 60]\n[0, 20, 40, 60]\nSome(3)\nSome(6)\nNone\ncalls=6\nn=0\nhint=(0, Some(6))\n[102, 103]\n",
        "the closure's Some results are yielded in order, None results dropped"
    );
}

const FILTER_MAP_BACK_SRC: &str = r#"
fn main() {
    let v = vec![1, 2, 3, 4, 5, 6];
    let mut it = v.iter().filter_map(|x| if *x % 2 == 0 { Some(*x) } else { None });
    println!("{:?}", it.next());
    println!("{:?}", it.next_back());
    println!("{:?}", it.next());
    println!("{:?}", it.next_back());
    println!("{:?}", it.next());
    let mut rsum = 0;
    for x in (0..10).filter_map(|x| if x % 2 == 0 { Some(x) } else { None }).rev() {
        rsum += x;
    }
    println!("revsum={rsum}");
    let out: Vec<i32> = v.iter().filter_map(|x| if *x % 2 == 1 { Some(*x * 2) } else { None }).collect();
    println!("{out:?}");
}
"#;

#[test]
fn run_mode_filter_map_next_back_and_composition() {
    // The interleaved steps pin that front and back pulls meet
    // in the middle without re-yielding an element, exactly as
    // the Filter test pins for its predicate.
    let dir = temp_dir("filter-map-next-back");
    let src = fixture(&dir, "filter_map_next_back.rs", FILTER_MAP_BACK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "FilterMap next_back and Rev composition must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "Some(2)\nSome(6)\nSome(4)\nNone\nNone\nrevsum=20\n[2, 6, 10]\n",
        "front and back steps share one closure-gated sequence"
    );
}

const MAP_CLONE_SRC: &str = r#"
fn main() {
    let v = vec![1i32, 2, 3];
    let w: Vec<i32> = v.iter().map(Clone::clone).collect();
    assert_eq!(w.len(), 3);
    println!("ok");
}
"#;

#[test]
fn run_mode_map_fn_item_clone() {
    let dir = temp_dir("map-clone");
    let src = fixture(&dir, "map_clone.rs", MAP_CLONE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

// ---- Lane IW: Skip / Chain adapters over modelled iterators (issue #158) ----
// The `next` / `into_iter` / `size_hint` builtins step the
// adapter states directly, and the adapters drop glue is

// ---- Lane IW: Skip / Chain adapters over modelled iterators (issue #158) ----
// The `next` / `into_iter` / `size_hint` builtins step the
// adapter states directly, and the adapters' drop glue is
// accounted like the other modelled adapters': a `Skip` or
// `Chain` over the argument iterator runs end to end, and
// dropping one (a statement temporary, an explicit
// `mem::drop`) is a faithful no-op for the string elements.

const SKIP_CHAIN_SRC: &str = r#"
use std::env;
fn main() {
    let mut it = (0..6).skip(2);
    println!("{} {} {}", it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
    let mut short = (0..2).skip(5);
    println!("{:?} {:?}", short.next(), short.next());
    let hinted = (0..10).skip(3);
    println!("hint {:?}", hinted.size_hint());
    let rest: Vec<String> = env::args().skip(1).collect();
    println!("rest {:?}", rest);
    println!("skip-next {:?}", env::args().skip(2).next());
    let mut ch = (0..2).chain(5..7);
    println!("{} {} {} {} {:?}", ch.next().unwrap(), ch.next().unwrap(), ch.next().unwrap(), ch.next().unwrap(), ch.next());
    println!("chain-hint {:?}", (0..3).chain(0..4).size_hint());
    let all: Vec<i32> = (0..2).chain(9..11).collect();
    println!("all {:?}", all);
    println!("doubled {}", env::args().chain(env::args()).count());
    std::mem::drop(env::args().skip(1));
    std::mem::drop((0..3).chain(4..6));
    println!("done");
}
"#;

#[test]
fn run_mode_skip_and_chain_adapters_over_modelled_iterators() {
    let dir = temp_dir("skip-chain");
    let src = fixture(&dir, "skip_chain.rs", SKIP_CHAIN_SRC);
    let out = run_driver_extra(&src, &["--guest-arg", "one", "--guest-arg", "two"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "2 3 4\nNone None\nhint (7, Some(7))\nrest [\"one\", \"two\"]\nskip-next Some(\"two\")\n0 1 5 6 None\nchain-hint (7, Some(7))\nall [0, 1, 9, 10]\ndoubled 6\ndone\n"
    );
}

const ADAPTERS_DROP_BY_VALUE_SRC: &str = r#"
fn sink<T>(_: T) {}
fn main() {
    let a = [1, 2, 3];
    sink(a.iter().cloned());
    sink(a.iter().cycle());
    sink(a.iter().zip([4, 5].iter()));
    sink(a.iter().peekable());
    sink(a.iter().filter(|_| true));
    sink(a.iter().rev());
    sink((1..10).step_by(2));
    sink("x".escape_debug());
    sink("x".escape_default());
    sink("x".escape_unicode());
    println!("done");
}
"#;

#[test]
fn run_mode_iterator_adapters_drop_by_value() {
    let dir = temp_dir("adapters-drop-by-value");
    let src = fixture(&dir, "case.rs", ADAPTERS_DROP_BY_VALUE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "done\n");
}

const COLLECTION_ITERATORS_SRC: &str = r#"
use std::collections::{BinaryHeap, LinkedList, VecDeque};
fn main() {
    let mut l = LinkedList::new();
    l.push_back(5); l.push_back(7); l.push_back(9);
    let mut it = l.iter();
    println!("{:?} {:?} {:?}", it.next(), it.next_back(), it.next());
    println!("count {}", l.iter().count());
    let mut s = 0;
    for x in l.into_iter() { s += x; }
    println!("sum {}", s);
    let mut h = BinaryHeap::with_capacity(4);
    h.push(3); h.push(9); h.push(5);
    let mut hi = h.iter();
    println!("{:?} {:?}", hi.next(), hi.next());
    println!("hcount {}", h.iter().count());
    let mut d = VecDeque::new();
    d.push_back(4); d.push_back(6); d.push_front(2);
    let mut di = d.iter();
    println!("{:?} {:?} {:?}", di.next(), di.next_back(), di.next());
    let mut dsum = 0;
    for x in d.into_iter() { dsum += x; }
    println!("dsum {}", dsum);
}
"#;

#[test]
fn run_mode_collection_iterators_next() {
    // The concrete collection iterators (`LinkedList` /
    // `VecDeque` / `BinaryHeap` in their `Iter` and
    // `IntoIter` forms) step through their collected bodies,
    // like the hash iterators: neither reach nor conversion
    // demanded or preferred those bodies, so `next` and the
    // containers' by-value `into_iter` died as not
    // runtime-supported (the iterators' own identity
    // `into_iter` is served by the builtin). The outputs are
    // native rustc 1.100.0's.
    let dir = temp_dir("collection-iterators");
    let src = fixture(&dir, "collection_iterators.rs", COLLECTION_ITERATORS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "collection iterators must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "Some(5) Some(9) Some(7)\ncount 3\nsum 21\nSome(9) Some(3)\nhcount 3\nSome(2) Some(6) Some(4)\ndsum 12\n"
    );
}

const MUT_REF_ITERATOR_SRC: &str = r#"
fn main() {
    let v = vec![10u32, 20, 30, 40];
    let mut it = v.iter().cloned();
    let mut sum = 0u32;
    for x in &mut it { sum += x; }
    println!("sum={} after={:?}", sum, it.next());
    let a = [1u64, 2, 3];
    let b = [100u64, 200, 300];
    let mut ci = a.iter().cloned();
    let z: Vec<(u64, u64)> = b.iter().cloned().zip(&mut ci).collect();
    println!("{:?} ci={:?}", z, ci.next());
    let mut m = v.iter().map(|x| x + 1);
    println!("{:?}", (&mut m).next());
    println!("{:?}", m.next());
}
"#;

#[test]
fn run_mode_mut_ref_iterator_forwarding() {
    // The `&mut I: Iterator` / `IntoIterator` blanket shapes:
    // `into_iter` on a `&mut` iterator is the identity (the
    // reference itself is the iterator), and its `next`
    // forwards to the pointee -- including the `&mut` side a
    // `Zip` state holds, which the modelled consumer steps
    // as a state field. All three shapes died as not
    // runtime-supported. The outputs are native
    // rustc 1.100.0's.
    let dir = temp_dir("mut-ref-iterator");
    let src = fixture(&dir, "mut_ref_iterator.rs", MUT_REF_ITERATOR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "&mut iterator forwarding must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "sum=100 after=None\n[(100, 1), (200, 2), (300, 3)] ci=None\nSome(11)\nSome(21)\n"
    );
}

const FLAT_MAP_SRC: &str = r#"
fn twice(x: i32) -> Vec<i32> {
    vec![x, x]
}
fn main() {
    let v: Vec<i32> = (0..3).flat_map(|i| vec![i, i * 10]).collect();
    println!("{v:?}");
    let factor = 3;
    let c: Vec<i32> = (1..4).flat_map(|i| vec![i * factor]).collect();
    println!("{c:?}");
    let f: Vec<i32> = (0..3).flat_map(twice as fn(i32) -> Vec<i32>).collect();
    println!("{f:?}");
    let o: Vec<i32> = (0..4)
        .flat_map(|i| if i % 2 == 0 { Some(i) } else { None })
        .collect();
    println!("{o:?}");
    let a: Vec<i32> = (0..3).flat_map(|i| [i, -i]).collect();
    println!("{a:?}");
    let n: Vec<i32> = (0..3)
        .flat_map(|i| (0..i).flat_map(move |j| vec![j]))
        .collect();
    println!("{n:?}");
    let e: Vec<i32> = (0..3).flat_map(|_| Vec::<i32>::new()).collect();
    println!("{e:?}");
    let cnt = (0..3).flat_map(|i| vec![i, 1]).count();
    println!("cnt={cnt}");
}
"#;

#[test]
fn run_mode_flat_map_iteration() {
    // The composition's step is core's FlattenCompat machine:
    // the in-flight inner iterator is drained, then the outer
    // iterator is pulled and its item converted with the same
    // into_iter transformation the guest's own call performs.
    // The shapes pin the conversions (Vec, capturing closure,
    // fn pointer, Option, array, a nested FlatMap) and the
    // empty-inner accounting.
    let dir = temp_dir("flat-map-iteration");
    let src = fixture(&dir, "flat_map_iteration.rs", FLAT_MAP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "FlatMap over Range with Vec/Option/array/nested inners must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[0, 0, 1, 10, 2, 20]\n[3, 6, 9]\n[0, 0, 1, 1, 2, 2]\n[0, 2]\n[0, 0, 1, -1, 2, -2]\n[0, 0, 1]\n[]\ncnt=6\n",
        "each inner iterator is drained in order before the outer advances"
    );
}

const FLATTEN_SRC: &str = r#"
fn main() {
    let w: Vec<i32> = vec![vec![1, 2], vec![3], vec![]]
        .into_iter()
        .flatten()
        .collect();
    println!("{w:?}");
    let vv = vec![vec![1, 2], vec![3]];
    let flat: Vec<&i32> = vv.iter().flatten().collect();
    println!("{flat:?}");
    let fr: Vec<i32> = vec![0..2, 5..7].into_iter().flatten().collect();
    println!("{fr:?}");
    let mut it = (0..3).flat_map(|i| vec![i, i * 10]);
    let mut out = Vec::new();
    while let Some(x) = it.next() {
        out.push(x);
        if let Some(y) = it.next_back() {
            out.push(y);
        }
    }
    println!("{out:?}");
    let r: Vec<i32> = (0..3).flat_map(|i| vec![i, i * 10]).rev().collect();
    println!("{r:?}");
}
"#;

#[test]
fn run_mode_flatten_and_flat_map_next_back() {
    // Flatten over owned, borrowed and range inners, and the
    // back step of the composition: the interleaved pulls pin
    // that front and back meet in the middle (the back inner
    // iterator is drained from the back, and an outer pull for
    // the back side takes the outer iterator's own back step).
    let dir = temp_dir("flatten-next-back");
    let src = fixture(&dir, "flatten_next_back.rs", FLATTEN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Flatten over Vec/&Vec/Range inners and FlatMap next_back must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 2, 3]\n[1, 2, 3]\n[0, 1, 5, 6]\n[0, 20, 0, 2, 1, 10]\n[20, 2, 10, 1, 0, 0]\n",
        "front and back steps drain the composition symmetrically"
    );
}

const CUSTOM_ITER_CONSUMERS_SRC: &str = r#"
struct Count { n: u32, max: u32 }
impl Iterator for Count {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.n < self.max { self.n += 1; Some(self.n) } else { None }
    }
}
fn main() {
    println!("uall={}", (Count { n: 0, max: 3 }).all(|x| x > 0));
    println!("uany={}", (Count { n: 0, max: 3 }).any(|x| x == 2));
    let mut it = Count { n: 0, max: 5 };
    println!("unth={:?}", it.nth(2));
    println!("ucount={}", (Count { n: 0, max: 4 }).count());
}
"#;

#[test]
fn run_mode_custom_iterator_consumers() {
    // The same consumer rule for a user-defined iterator:
    // its `next` is an ordinary collected body, so the
    // `try_fold` / `nth` instance bodies that compose it are
    // collected and preferred over the builtin consumers,
    // exactly as for the hash iterators.
    let dir = temp_dir("custom_iter_consumers");
    let src = fixture(&dir, "custom_iter_consumers.rs", CUSTOM_ITER_CONSUMERS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "try_fold / nth consumers over a user-defined iterator must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "uall=true\nuany=true\nunth=Some(3)\nucount=4\n"
    );
}

const COLLECTION_SIZE_HINT_SRC: &str = r#"
use std::collections::{LinkedList, VecDeque};
fn main() {
    let mut l: LinkedList<i32> = LinkedList::new();
    l.push_back(1); l.push_back(2); l.push_back(3);
    println!("hint {:?}", l.iter().size_hint());
    let mut it = l.iter();
    it.next();
    println!("hint {:?}", it.size_hint());
    println!("len {}", l.iter().len());
    let mut d: VecDeque<i32> = VecDeque::new();
    d.push_back(4); d.push_back(5);
    println!("hint {:?}", d.iter().size_hint());
}
"#;

#[test]
fn run_mode_collection_iterator_size_hint_is_exact() {
    // A body-served collection iterator's `size_hint` is
    // its own override (an exact remaining count), served
    // by the collected body: the builtin's trait-default
    // `(0, None)` answer for a real-layout state is
    // observable here directly, not only through the
    // `TrustedLen` collect entry that panics on it.
    let dir = temp_dir("collectionsizehint");
    let src = fixture(&dir, "collectionsizehint.rs", COLLECTION_SIZE_HINT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "size_hint of LinkedList / VecDeque iterators must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hint (3, Some(3))\nhint (2, Some(2))\nlen 3\nhint (2, Some(2))\n"
    );
}

const FLAT_MAP_SIZE_HINT_SRC: &str = r#"
fn main() {
    println!("map={:?}", (0..3).map(|x| x * 2).size_hint());
    let mut fm1 = vec![1, 2, 3].into_iter().flat_map(|x| Some(x));
    println!("opt-fresh={:?}", fm1.size_hint());
    println!("opt-first={:?}", fm1.next());
    println!("opt-after1={:?}", fm1.size_hint());
    let fm2 = vec![1, 2].into_iter().flat_map(|x| [x, x]);
    println!("arr={:?}", fm2.size_hint());
    let mut fm3 = vec![1, 2].into_iter().flat_map(|x| vec![x, x]);
    println!("vec-fresh={:?}", fm3.size_hint());
    println!("vec-first={:?}", fm3.next());
    println!("vec-mid={:?}", fm3.size_hint());
    let mut fl = vec![vec![1, 2], vec![3]].into_iter().flatten();
    println!("fl-first={:?}", fl.next());
    println!("fl-mid={:?}", fl.size_hint());
}
"#;

#[test]
fn run_mode_flat_map_size_hint() {
    // `size_hint` of `Map` and the flatten
    // compositions: `Map` forwards its inner
    // iterator's hint unchanged, and
    // `FlattenCompat::size_hint` is replayed on the
    // modelled state -- including the one-shot
    // specialisation (an `Option` inner's hint
    // ignores the front iterator) and the
    // fixed-size scaling of an array inner.
    // Expected output was produced by the native
    // 1.100.0 toolchain.
    let dir = temp_dir("flatmapsizehint");
    let src = fixture(&dir, "flatmapsizehint.rs", FLAT_MAP_SIZE_HINT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "size_hint of Map / FlatMap / Flatten must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "map=(3, Some(3))\nopt-fresh=(0, Some(3))\nopt-first=Some(1)\nopt-after1=(0, Some(2))\narr=(4, Some(4))\nvec-fresh=(0, None)\nvec-first=Some(1)\nvec-mid=(1, None)\nfl-first=Some(1)\nfl-mid=(1, None)\n"
    );
}