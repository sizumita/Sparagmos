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


/// A guest-defined `Iterator` impl whose only demand
/// arrives through the builtin-diverted consumer names:
/// `group_by` (an `IteratorExt`-style constructor) builds
/// a `GroupBy` over a `Peekable`, a modelled `Zip` steps
/// it, and `collect` drains the group iterators. The
/// `GroupBy::next` body -- a generic impl whose `G`
/// parameter is not determined by `Self` -- must be
/// materialised by the iterator demand pass, its inner
/// `Peekable::next` call must be served by the collected
/// core body, and the key closure it stores must be
/// callable through the `&mut` reference `Option::map`
/// hands to `call_once`. Mirrors the shape of
/// `tests/ui/impl-trait/example-calendar.rs`'s
/// `test_group_by`.
const GROUP_BY_SRC: &str = r#"
struct GroupBy<It: Iterator, F> {
    it: std::iter::Peekable<It>,
    f: F,
}

impl<It, F> Clone for GroupBy<It, F>
where
    It: Iterator + Clone,
    It::Item: Clone,
    F: Clone,
{
    fn clone(&self) -> Self {
        GroupBy { it: self.it.clone(), f: self.f.clone() }
    }
}

impl<G, It, F> Iterator for GroupBy<It, F>
where
    It: Iterator + Clone,
    It::Item: Clone,
    F: Clone + FnMut(&It::Item) -> G,
    G: Eq + Clone,
{
    type Item = (G, InGroup<std::iter::Peekable<It>, F, G>);

    fn next(&mut self) -> Option<Self::Item> {
        self.it.peek().map(&mut self.f).map(|key| {
            let start = self.it.clone();
            while let Some(k) = self.it.peek().map(&mut self.f) {
                if key != k {
                    break;
                }
                self.it.next();
            }
            (key.clone(), InGroup { it: start, f: self.f.clone(), g: key })
        })
    }
}

#[derive(Copy, Clone)]
struct InGroup<It, F, G> {
    it: It,
    f: F,
    g: G,
}

impl<It: Iterator, F: FnMut(&It::Item) -> G, G: Eq> Iterator for InGroup<It, F, G> {
    type Item = It::Item;

    fn next(&mut self) -> Option<It::Item> {
        self.it.next().and_then(|x| {
            if (self.f)(&x) == self.g { Some(x) } else { None }
        })
    }
}

trait IteratorExt: Iterator + Sized {
    fn group_by<G, F>(self, f: F) -> GroupBy<Self, F>
    where
        F: Clone + FnMut(&Self::Item) -> G,
        G: Eq,
    {
        GroupBy { it: self.peekable(), f }
    }
}

impl<It> IteratorExt for It where It: Iterator {}

fn main() {
    let input = [[1, 1], [1, 1], [1, 2], [2, 2], [2, 3], [2, 3], [3, 3]];
    let by_x = input.iter().cloned().group_by(|a| a[0]);
    let expected_1: &[&[[i32; 2]]] = &[
        &[[1, 1], [1, 1], [1, 2]],
        &[[2, 2], [2, 3], [2, 3]],
        &[[3, 3]],
    ];
    for ((_, a), b) in by_x.zip(expected_1.iter().cloned()) {
        assert_eq!(&a.collect::<Vec<_>>()[..], b);
    }
    println!("group_by ok");
}
"#;

#[test]
fn run_mode_group_by_guest_iterator_impl() {
    run_ok("group-by-demand", GROUP_BY_SRC, "group_by ok\n");
}

/// The `example-calendar` compositions around the same
/// guest adapters: two `GroupBy` instances over one source
/// type, differing only in their key closure, stepped side
/// by side through a modelled `Zip` (the impl scan must
/// tell the instances apart by their rendered type tags,
/// not by the shared base name); an `InGroup` over a
/// `RangeFrom` stepped through two nested modelled `Map`s
/// (the adapter step extracts the inner type from a tag
/// whose closure argument carries a function-pointer
/// signature); and `GroupBy`s consumed only through
/// adapter constructors -- a `map` source and a `chain`
/// second source -- inside generic helper bodies, whose
/// instance bodies the second `next`-demand run scans.
/// Mirrors the shapes of
/// `tests/ui/impl-trait/example-calendar.rs`'s
/// `test_group_by`, `test_isoweekdate` and `format_month`.
const GROUP_BY_COMPOSITIONS_SRC: &str = r#"
struct GroupBy<It: Iterator, F> {
    it: std::iter::Peekable<It>,
    f: F,
}

impl<It, F> Clone for GroupBy<It, F>
where
    It: Iterator + Clone,
    It::Item: Clone,
    F: Clone,
{
    fn clone(&self) -> Self {
        GroupBy { it: self.it.clone(), f: self.f.clone() }
    }
}

impl<G, It, F> Iterator for GroupBy<It, F>
where
    It: Iterator + Clone,
    It::Item: Clone,
    F: Clone + FnMut(&It::Item) -> G,
    G: Eq + Clone,
{
    type Item = (G, InGroup<std::iter::Peekable<It>, F, G>);

    fn next(&mut self) -> Option<Self::Item> {
        self.it.peek().map(&mut self.f).map(|key| {
            let start = self.it.clone();
            while let Some(k) = self.it.peek().map(&mut self.f) {
                if key != k {
                    break;
                }
                self.it.next();
            }
            (key.clone(), InGroup { it: start, f: self.f.clone(), g: key })
        })
    }
}

#[derive(Copy, Clone)]
struct InGroup<It, F, G> {
    it: It,
    f: F,
    g: G,
}

impl<It: Iterator, F: FnMut(&It::Item) -> G, G: Eq> Iterator for InGroup<It, F, G> {
    type Item = It::Item;

    fn next(&mut self) -> Option<It::Item> {
        self.it.next().and_then(|x| {
            if (self.f)(&x) == self.g { Some(x) } else { None }
        })
    }
}

trait IteratorExt: Iterator + Sized {
    fn group_by<G, F>(self, f: F) -> GroupBy<Self, F>
    where
        F: Clone + FnMut(&Self::Item) -> G,
        G: Eq,
    {
        GroupBy { it: self.peekable(), f }
    }
}

impl<It> IteratorExt for It where It: Iterator {}

fn dates() -> impl Iterator<Item = i32> + Clone {
    InGroup { it: 0.., f: |x: &i32| *x / 10, g: 0 }
}

fn weeks_uniq() -> Vec<(i32, i32)> {
    let mut weeks = dates().map(|d| (d / 7, d)).map(|(w, _)| w);
    let mut out = vec![];
    let mut acc = (weeks.next().unwrap(), 1);
    for w in weeks {
        if acc.0 == w {
            acc.1 += 1;
        } else {
            out.push(acc);
            acc = (w, 1);
        }
    }
    out.push(acc);
    out
}

fn summarize<I>(it: I) -> Vec<(i32, usize)>
where
    I: Iterator<Item = i32> + Clone,
{
    it.group_by(|x| x / 10).map(|(k, g)| (k, g.count())).collect()
}

struct Countdown(u32);

impl Iterator for Countdown {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        if self.0 == 0 {
            None
        } else {
            self.0 -= 1;
            Some(self.0 as i32)
        }
    }
}

fn chained_groups<I>(it: I) -> usize
where
    I: Iterator<Item = i32> + Clone,
{
    it.group_by(|x| x / 10)
        .map(|(k, _)| k)
        .chain(Countdown(2))
        .count()
}

"#;

const GROUP_BY_ZIP_MAIN: &str = r#"
fn main() {
    // Two instances of one adapter type, told apart by
    // their key closures, stepped through one `Zip`.
    let input = [[1, 1], [1, 1], [1, 2], [2, 2], [2, 3], [2, 3], [3, 3]];
    let by_x = input.iter().cloned().group_by(|a| a[0]);
    let by_y = input.iter().cloned().group_by(|a| a[1]);
    let expected: &[(&[[i32; 2]], &[[i32; 2]])] = &[
        (&[[1, 1], [1, 1], [1, 2]], &[[1, 1], [1, 1]]),
        (&[[2, 2], [2, 3], [2, 3]], &[[1, 2], [2, 2]]),
        (&[[3, 3]], &[[2, 3], [2, 3], [3, 3]]),
    ];
    for (((kx, gx), (ky, gy)), (ex, ey)) in by_x.zip(by_y).zip(expected.iter()) {
        assert_eq!(kx, ky);
        assert_eq!(&gx.collect::<Vec<_>>()[..], *ex);
        assert_eq!(&gy.collect::<Vec<_>>()[..], *ey);
    }
    println!("group_by zip ok");
}
"#;

#[test]
fn run_mode_group_by_instances_through_zip() {
    run_ok(
        "group-by-zip",
        &format!("{GROUP_BY_COMPOSITIONS_SRC}{GROUP_BY_ZIP_MAIN}"),
        "group_by zip ok\n",
    );
}

const GROUP_BY_WEEKS_MAIN: &str = r#"
fn main() {
    // An `InGroup` over a `RangeFrom` through two `Map`s.
    assert_eq!(weeks_uniq(), vec![(0, 7), (1, 3)]);
    println!("group_by weeks ok");
}
"#;

#[test]
fn run_mode_group_by_through_maps() {
    run_ok(
        "group-by-weeks",
        &format!("{GROUP_BY_COMPOSITIONS_SRC}{GROUP_BY_WEEKS_MAIN}"),
        "group_by weeks ok\n",
    );
}

const GROUP_BY_HELPERS_MAIN: &str = r#"
fn main() {
    // Adapter-only consumption inside generic helpers.
    assert_eq!(summarize(dates()), vec![(0, 10)]);
    assert_eq!(chained_groups(dates()), 3);
    println!("group_by helpers ok");
}
"#;

#[test]
fn run_mode_group_by_adapter_only_consumption() {
    run_ok(
        "group-by-helpers",
        &format!("{GROUP_BY_COMPOSITIONS_SRC}{GROUP_BY_HELPERS_MAIN}"),
        "group_by helpers ok\n",
    );
}

const GROUP_BY_COMBINED_MAIN: &str = r#"
fn main() {
    // Both consumption shapes in one program: the
    // direct-`next` composition and the generic helpers'
    // collect / count.
    assert_eq!(weeks_uniq(), vec![(0, 7), (1, 3)]);
    assert_eq!(summarize(dates()), vec![(0, 10)]);
    assert_eq!(chained_groups(dates()), 3);
    println!("group_by combined ok");
}
"#;

#[test]
fn run_mode_group_by_combined_consumption_shapes() {
    // The helpers' collect demands the `Map` over the
    // `GroupBy` composition's collected `next` body; the
    // direct-`next` composition's own `Map` instantiations
    // are served by the modelled state machine and collect
    // no body. The conversion scan for a trait-method call
    // must not bind the direct composition's `next` call to
    // the one collected `Map` body just because the base
    // names agree: it did, and the foreign body walked the
    // machine-shaped value with its own type's layout,
    // stepping the innermost `RangeFrom` one field too deep
    // ("type mismatch in iterator state of
    // std::ops::RangeFrom<i32>: expected ADT value, found
    // signed integer").
    run_ok(
        "group-by-combined",
        &format!("{GROUP_BY_COMPOSITIONS_SRC}{GROUP_BY_COMBINED_MAIN}"),
        "group_by combined ok\n",
    );
}

/// `Peekable::next` has no modelled state-machine step:
/// its state is the genuine struct the collected `peek`
/// body manipulates, so a direct `next` call on a
/// `Peekable` -- here from a plain local function -- is
/// served by the collected core body.
const PEEKABLE_NEXT_SRC: &str = r#"
fn step<I: Iterator>(it: &mut std::iter::Peekable<I>) -> Option<I::Item> {
    it.next()
}

fn main() {
    let input = [[1, 1], [1, 2], [2, 2]];
    let mut p = input.iter().cloned().peekable();
    assert_eq!(p.peek(), Some(&[1, 1]));
    assert_eq!(step(&mut p), Some([1, 1]));
    assert_eq!(step(&mut p), Some([1, 2]));
    assert_eq!(step(&mut p), Some([2, 2]));
    assert_eq!(step(&mut p), None);
    println!("peekable next ok");
}
"#;

#[test]
fn run_mode_peekable_next_direct_call() {
    run_ok("peekable-next", PEEKABLE_NEXT_SRC, "peekable next ok\n");
}


/// `FnOnce::call_once` handed a closure through a
/// reference: inside `Option::map`'s body the callable
/// operand is the `&mut` field borrow, and the `CallOnce`
/// dispatch must follow the reference to the closure
/// (the shape `Peekable::peek`'s internal
/// `Option::map(&mut self.f)`-style calls take).
const MAP_CALL_ONCE_SRC: &str = r#"
struct S<F: FnMut(i32) -> i32> {
    it: std::iter::Peekable<std::vec::IntoIter<i32>>,
    f: F,
}

impl<F: FnMut(i32) -> i32> S<F> {
    fn peek_key(&mut self) -> Option<i32> {
        self.it.peek().copied().map(&mut self.f)
    }
}

fn main() {
    let mut s = S {
        it: vec![10, 20].into_iter().peekable(),
        f: |x| x + 1,
    };
    assert_eq!(s.peek_key(), Some(11));
    assert_eq!(s.it.next(), Some(10));
    assert_eq!(s.peek_key(), Some(21));
    println!("map call_once ok");
}
"#;

#[test]
fn run_mode_map_call_once_through_mut_ref() {
    run_ok("map-call-once-ref", MAP_CALL_ONCE_SRC, "map call_once ok\n");
}


const PEEKABLE_CONSUMER_SRC: &str = r#"
struct Count {
    next: u32,
    end: u32,
}

impl Iterator for Count {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.next < self.end {
            let v = self.next;
            self.next += 1;
            Some(v)
        } else {
            None
        }
    }
}

fn main() {
    let input: Vec<i32> = vec![1, 2, 3, 4];
    let out: Vec<i32> = input.iter().cloned().peekable().collect();
    println!("collect {out:?}");
    let mut p = input.iter().cloned().peekable();
    assert_eq!(p.peek(), Some(&1));
    let rest: Vec<i32> = p.collect();
    println!("peeked {rest:?}");
    let n = input.iter().cloned().peekable().count();
    println!("count {n}");
    let s: i32 = input.iter().cloned().peekable().sum();
    println!("sum {s}");
    let guest: Vec<u32> = Count { next: 0, end: 3 }.peekable().collect();
    println!("guest {guest:?}");
}
"#;

#[test]
fn run_mode_peekable_owned_consumers() {
    // An owned `Peekable` handed to a consumer builtin: the
    // `into_iter` call is the blanket identity (a `Peekable`
    // is its own iterator), and the consumer's drain steps the
    // adapter through its collected core `next` body -- the
    // modelled state machine has no `Peekable` arm, since the
    // state carries the peeked slot no builtin-produced state
    // has, so the demand pass converts the body for the
    // machine's unclaimed-tag fallback (the same route a
    // guest-defined iterator takes). Before, the run stopped
    // at `IntoIterator::into_iter for std::iter::Peekable<...>`
    // as not runtime-supported.
    let dir = temp_dir("peekable-consumers");
    let src = fixture(&dir, "peekableconsumers.rs", PEEKABLE_CONSUMER_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "collect [1, 2, 3, 4]\npeeked [1, 2, 3, 4]\ncount 4\nsum 10\nguest [0, 1, 2]\n",
        "stdout={stdout}"
    );
}

const COLLECT_PAIRS_ANY_SRC: &str = r#"
fn main() {
    let all: Vec<(String, String)> = (0..17)
        .map(|i| (format!("k{i}"), format!("v{i}")))
        .collect();
    println!("any-last {}", all.iter().any(|(k, _v)| k == "k16"));
    println!("any-none {}", all.iter().any(|(k, _v)| k == "nope"));
    println!("count {}", all.iter().count());
}
"#;

#[test]
fn run_mode_collect_pairs_iter_any_full_scan() {
    // A full-scan `any` over a collect-built vector of pairs:
    // the scan drives the borrowed iteration to the end
    // position and through the closure's string comparisons
    // on every element, and has to find the last element and
    // report a miss correctly. (The collect-built vector is
    // stored as a field sequence, so its iteration keeps the
    // element snapshot; the shape pins that the scan over it
    // stays exact.)
    run_ok(
        "collect-pairs-any",
        COLLECT_PAIRS_ANY_SRC,
        "any-last true\nany-none false\ncount 17\n",
    );
}

const FN_DEF_STEPPED_ADAPTERS_SRC: &str = r#"
struct Counter { n: u32, max: u32 }

impl Iterator for Counter {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.n < self.max { self.n += 1; Some(self.n) } else { None }
    }
}

trait JoinExt: Iterator + Sized {
    fn join(mut self) -> String
    where Self::Item: std::fmt::Display {
        let mut s = String::new();
        if let Some(e) = self.next() {
            s.push_str(&e.to_string());
            for e in self {
                s.push(',');
                s.push_str(&e.to_string());
            }
        }
        s
    }
}

impl<It> JoinExt for It where It: Iterator {}

fn main() {
    let chained = Counter { n: 0, max: 2 }.chain(Counter { n: 10, max: 12 }.map(|x| x));
    println!("chain {}", chained.join());
    let chained2 = vec![7, 8].into_iter().chain(Counter { n: 20, max: 22 });
    println!("chain2 {}", chained2.join());
}
"#;

#[test]
fn run_mode_fn_def_stepped_adapters_over_guest_iterator() {
    // `Chain` steps its first iterator by handing
    // `Iterator::next` itself to the `and_then_or_clear`
    // helper as a function item, so a collected `next` body
    // for it cannot run end to end (no runtime dispatch
    // serves a trait method held as a value): the run stops
    // at the bare trait name when the first side is the
    // guest composition (the example-calendar record), and
    // silently drops the first side's items when the first
    // side is a modelled iterator (the `chain2` line). The
    // demand judgement
    // must leave `Chain` to the modelled state machine even
    // over a guest iterator -- the shape of the
    // example-calendar record, where a generic consumer
    // (the extension-trait `join` here) drains a `Chain`
    // whose second half is a `Map` over a guest-defined
    // iterator (and, mirrored, a guest iterator chained
    // behind a modelled one).
    run_ok(
        "fn-def-stepped-adapters",
        FN_DEF_STEPPED_ADAPTERS_SRC,
        "chain 1,2,11,12\nchain2 7,8,21,22\n",
    );
}

const ENUMERATE_OVER_GUEST_SRC: &str = r#"
struct Counter { n: u32, max: u32 }

impl Iterator for Counter {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.n < self.max { self.n += 1; Some(self.n) } else { None }
    }
}

fn drain<I: Iterator>(mut it: I) -> Vec<I::Item> {
    let mut v = Vec::new();
    while let Some(x) = it.next() { v.push(x); }
    v
}

fn main() {
    let e: Vec<(usize, u32)> = drain(Counter { n: 0, max: 3 }.enumerate());
    println!("enum {e:?}");
}
"#;

#[test]
fn run_mode_enumerate_over_guest_iterator_generic_consumer() {
    // The shape PR #924's demand arm exists for: an
    // `Enumerate` over a guest-defined iterator, drained by a
    // generic consumer. The adapter's own `next` body calls
    // the inner `next` directly, so collecting it is what
    // demands the guest instance; the fn-def-stepped
    // exception must not narrow this back out.
    run_ok(
        "enumerate-over-guest",
        ENUMERATE_OVER_GUEST_SRC,
        "enum [(0, 1), (1, 2), (2, 3)]\n",
    );
}
