use super::{fixture, run_driver, run_driver_extra, run_ok, temp_dir};


const BTREE_BORROWED_ITER_SRC: &str = r#"
use std::collections::{BTreeMap, BTreeSet};
fn main() {
    let s: BTreeSet<i32> = BTreeSet::new();
    let m: BTreeMap<i32, i32> = BTreeMap::new();
    let mut n = 0;
    for _x in &s {
        n += 1;
    }
    for (_k, _v) in &m {
        n += 1;
    }
    for _x in s.iter() {
        n += 1;
    }
    for _kv in m.iter() {
        n += 1;
    }
    for _k in m.keys() {
        n += 1;
    }
    for _v in m.values() {
        n += 1;
    }
    println!("count {}", n);
    println!("hint {:?}", s.iter().size_hint());
    let mut it = m.iter();
    println!("back none {}", it.next_back().is_none());
}
"#;

#[test]
fn run_mode_btree_borrowed_iteration() {
    // Borrowed iteration over the B-tree collections:
    // `IntoIterator` for `&BTreeSet` / `&BTreeMap` and the
    // `iter` / `keys` / `values` methods yield iterators over an
    // element snapshot walked from the tree's node storage. An
    // empty tree's snapshot is empty, so every form yields
    // nothing; a non-empty tree cannot be built yet (insertion
    // itself is blocked by the raw-pointer write path), so the
    // empty forms are the reachable surface.
    let dir = temp_dir("btreeborrowediter");
    let src = fixture(&dir, "btreeborrowediter.rs", BTREE_BORROWED_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating &BTreeSet / &BTreeMap must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "count 0\nhint (0, Some(0))\nback none true\n"
    );
}

const BTREE_OWNED_ITER_SRC: &str = r#"
use std::collections::{BTreeMap, BTreeSet};
fn main() {
    let s: BTreeSet<i32> = BTreeSet::new();
    let mut n = 0;
    for _x in s {
        n += 1;
    }
    let m: BTreeMap<i32, i32> = BTreeMap::new();
    for _kv in m {
        n += 1;
    }
    println!("owned {}", n);
}
"#;

#[test]
fn run_mode_btree_owned_iteration() {
    // By-value iteration consumes the collection into the
    // owning-iterator snapshot state; dropping the partially or
    // fully consumed iterator afterwards drops exactly the
    // unyielded snapshot elements (the drop plan entry for the
    // btree `IntoIter` types).
    let dir = temp_dir("btreeownediter");
    let src = fixture(&dir, "btreeownediter.rs", BTREE_OWNED_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a BTreeSet / BTreeMap by value must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "owned 0\n");
}

const TRANSMUTE_DEEP_ZST_CHAIN_SRC: &str = r#"
#[repr(C)]
struct Inner2((), *mut u32);
#[repr(C)]
struct Outer2(Inner2, ());
fn main() {
    let mut x: u32 = 42;
    let p = &mut x as *mut u32;
    let outer: Outer2 = unsafe { std::mem::transmute(p) };
    let q = outer.0.1;
    unsafe { *q += 1; }
    println!("{}", x);
}
"#;

#[test]
fn run_mode_transmute_deep_zst_chain_is_explicit_unsupported() {
    // The chain enters a deeper layer at a non-zero field;
    // rebuilding it would need an aggregate nested inside
    // an operand, which the IR does not express. The
    // transmute stays an explicit unsupported instead of
    // building a value whose fields were never constructed.
    let dir = temp_dir("transmute-deep-zst-chain");
    let src = fixture(&dir, "case.rs", TRANSMUTE_DEEP_ZST_CHAIN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "deep non-zero chain transmute must be a named unsupported; stderr={stderr}"
    );
    assert!(
        stderr.contains("Cast Transmute"),
        "stderr must name the transmute; stderr={stderr}"
    );
}

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

const BTREE_SET_COLLECT_SRC: &str = r#"
use std::collections::BTreeSet;
fn main() {
    let s: BTreeSet<i32> = (0..3).collect();
    let sum: i32 = s.iter().sum();
    println!("len={} sum={}", s.len(), sum);
}
"#;

#[test]
fn run_mode_btree_set_collect() {
    // Collecting into a `BTreeSet` executes the collection's
    // own `FromIterator` body (the bulk bottom-up build over
    // the collected vector), as for the hash collections in
    // #350: the elements land sorted and deduplicated in the
    // tree's node storage, iteration reads them back in
    // order, and the set drops cleanly at the end of `main`.
    let dir = temp_dir("btreesetcollect");
    let src = fixture(&dir, "btreesetcollect.rs", BTREE_SET_COLLECT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "collecting into BTreeSet must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len=3 sum=3\n");
}

const BTREE_SET_FROM_ITER_SRC: &str = r#"
use std::collections::BTreeSet;
use std::iter::FromIterator;
fn main() {
    let s = BTreeSet::from_iter([5, 4, 4]);
    let sum: i32 = s.iter().sum();
    println!("len={} sum={}", s.len(), sum);
}
"#;

#[test]
fn run_mode_btree_set_from_iter() {
    // `BTreeSet::from_iter` over an array: the source
    // iterator's elements are collected into a vector, sorted,
    // and built into the tree bottom-up; the duplicate
    // element is dropped by the build.
    let dir = temp_dir("btreesetfromiter");
    let src = fixture(&dir, "btreesetfromiter.rs", BTREE_SET_FROM_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "BTreeSet::from_iter must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len=2 sum=9\n");
}

const BTREE_SET_COLLECT_FROM_VEC_SRC: &str = r#"
use std::collections::BTreeSet;
fn main() {
    let v = vec![5, 4, 4];
    let s: BTreeSet<i32> = v.into_iter().collect();
    let sum: i32 = s.iter().sum();
    println!("len={} sum={}", s.len(), sum);
}
"#;

#[test]
fn run_mode_btree_set_collect_from_vec_iter() {
    // Collecting a `vec::IntoIter`: the B-tree build wraps
    // the collected vector's iterator in its deduplicating
    // sorted iterator and steps it in its real layout.
    let dir = temp_dir("btreesetcollectvec");
    let src = fixture(&dir, "btreesetcollectvec.rs", BTREE_SET_COLLECT_FROM_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "collecting a vec::IntoIter into BTreeSet must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len=2 sum=9\n");
}

const BTREE_MAP_COLLECT_SRC: &str = r#"
use std::collections::BTreeMap;
use std::iter::FromIterator;
fn main() {
    let m: BTreeMap<i32, i32> = [(1, 10), (2, 20)].into_iter().collect();
    println!("len={} v1={:?}", m.len(), m.get(&1));
    let m2 = BTreeMap::from_iter([(3, 30), (1, 10)]);
    println!("len={} v3={:?}", m2.len(), m2.get(&3));
}
"#;

#[test]
fn run_mode_btree_map_collect_and_from_iter() {
    // The map forms of the same paths: `collect` and
    // `from_iter` build the tree from the key/value pairs and
    // lookups read the values back out of the node storage.
    let dir = temp_dir("btreemapcollect");
    let src = fixture(&dir, "btreemapcollect.rs", BTREE_MAP_COLLECT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "collecting into BTreeMap must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len=2 v1=Some(10)\nlen=2 v3=Some(30)\n"
    );
}

const BTREE_SET_MULTILEVEL_SRC: &str = r#"
use std::collections::BTreeSet;
fn main() {
    let mut s = BTreeSet::new();
    for i in 0..30 {
        s.insert(i);
    }
    println!("count={} sum={}", s.iter().count(), s.iter().sum::<i32>());
}
"#;

#[test]
fn run_mode_btree_set_multilevel_iterate() {
    // A set grown past one leaf by insertion alone: the
    // root is an internal node (height 1) whose leaf data
    // the node pointers name through the cast into the leaf
    // type, and iteration walks the internal node and its
    // leaf children in order.
    let dir = temp_dir("btreesetmultilevel");
    let src = fixture(&dir, "btreesetmultilevel.rs", BTREE_SET_MULTILEVEL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a multi-level BTreeSet must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "count=30 sum=435\n");
}

const BTREE_SET_MULTILEVEL_FROM_ITER_SRC: &str = r#"
use std::collections::BTreeSet;
fn main() {
    let s: BTreeSet<i32> = (0..80).collect();
    println!("count={} sum={}", s.iter().count(), s.iter().sum::<i32>());
}
"#;

#[test]
fn run_mode_btree_set_multilevel_from_iter() {
    // The same multi-level shape built by the bulk
    // bottom-up build instead of insertion: iteration
    // reads back every element in order.
    let dir = temp_dir("btreesetmultilevelfi");
    let src = fixture(
        &dir,
        "btreesetmultilevelfi.rs",
        BTREE_SET_MULTILEVEL_FROM_ITER_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a from_iter-built multi-level BTreeSet must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "count=80 sum=3160\n");
}

const BTREE_MAP_MULTILEVEL_SRC: &str = r#"
use std::collections::BTreeMap;
fn main() {
    let mut m = BTreeMap::new();
    for i in 0..60 {
        m.insert(i, i * i);
    }
    println!("count={} sum={}", m.iter().count(), m.iter().map(|(k, v)| k + v).sum::<i32>());
    let mut i = 0;
    while i < 60 {
        m.remove(&i);
        i += 2;
    }
    println!("count={}", m.iter().count());
}
"#;

#[test]
fn run_mode_btree_map_multilevel_iterate_remove() {
    // A multi-level map: iteration interleaves the
    // internal nodes' own pairs with their subtrees, and
    // removals that rebalance the leaves leave the walk
    // exact.
    let dir = temp_dir("btreemapmultilevel");
    let src = fixture(&dir, "btreemapmultilevel.rs", BTREE_MAP_MULTILEVEL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a multi-level BTreeMap must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "count=60 sum=71980\ncount=30\n"
    );
}

const BTREE_SET_MULTILEVEL_REMOVE_SRC: &str = r#"
use std::collections::BTreeSet;
fn main() {
    let mut s = BTreeSet::new();
    for i in 0..88 {
        s.insert(i);
    }
    let mut j = 0;
    while j < 88 {
        s.remove(&j);
        j += 3;
    }
    println!("count={} sum={}", s.iter().count(), s.iter().sum::<i32>());
}
"#;

#[test]
fn run_mode_btree_set_multilevel_remove() {
    // Removals across every leaf of a multi-level set,
    // with the merges and steals rebalancing performs:
    // the remaining elements still walk in order.
    let dir = temp_dir("btreesetmultilevelrm");
    let src = fixture(
        &dir,
        "btreesetmultilevelrm.rs",
        BTREE_SET_MULTILEVEL_REMOVE_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a rebalanced multi-level BTreeSet must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "count=58 sum=2523\n");
}

const BTREE_SET_MERGE_DRAIN_INSERT_SRC: &str = r#"
use std::collections::BTreeSet;
fn main() {
    let mut s = BTreeSet::new();
    for i in 0..200 {
        s.insert(i);
    }
    for i in 0..190 {
        s.remove(&i);
    }
    println!("count={} sum={}", s.iter().count(), s.iter().sum::<i32>());
}
"#;

#[test]
fn run_mode_btree_set_merge_drain_insert_built() {
    // Draining an insert-built multi-level set from the
    // front until its leaves merge: the merge pulls the
    // parent pair down through slice_remove on the set's
    // zero-sized value slots, and the remaining elements
    // still walk in order.
    let dir = temp_dir("btreesetmergedrainins");
    let src = fixture(
        &dir,
        "btreesetmergedrainins.rs",
        BTREE_SET_MERGE_DRAIN_INSERT_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "draining a multi-level BTreeSet until leaves merge must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "count=10 sum=1945\n");
}

const BTREE_SET_MERGE_DRAIN_BULK_SRC: &str = r#"
use std::collections::BTreeSet;
fn main() {
    let mut s: BTreeSet<i32> = (0..150).collect();
    for i in 0..140 {
        s.remove(&i);
    }
    println!("count={} sum={}", s.iter().count(), s.iter().sum::<i32>());
}
"#;

#[test]
fn run_mode_btree_set_merge_drain_bulk_built() {
    // The same front drain on a bulk-built set: the first
    // leaf merge reads the parent node's zero-sized value
    // slot, and the remaining elements still walk in order.
    let dir = temp_dir("btreesetmergedrainbulk");
    let src = fixture(
        &dir,
        "btreesetmergedrainbulk.rs",
        BTREE_SET_MERGE_DRAIN_BULK_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "draining a bulk-built multi-level BTreeSet until leaves merge must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "count=10 sum=1445\n");
}

const BTREE_SET_HEIGHT2_INSERT_SRC: &str = r#"
use std::collections::BTreeSet;
fn main() {
    let mut s = BTreeSet::new();
    for i in 0..200 {
        s.insert(i);
    }
    println!("count={} sum={}", s.iter().count(), s.iter().sum::<i32>());
}
"#;

#[test]
fn run_mode_btree_set_height2_insert() {
    // A set grown by insertion until the root itself
    // splits (height 2): the split moves the right half of
    // the full internal root's keys and edges into a new
    // internal node, and iteration afterwards reads every
    // element back in order.
    let dir = temp_dir("btreesetheight2");
    let src = fixture(&dir, "btreesetheight2.rs", BTREE_SET_HEIGHT2_INSERT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a height-2 BTreeSet must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "count=200 sum=19900\n");
}

const BTREE_SET_HEIGHT2_FROM_ITER_SRC: &str = r#"
use std::collections::BTreeSet;
fn main() {
    let s: BTreeSet<i32> = (0..150).collect();
    println!("count={} sum={}", s.iter().count(), s.iter().sum::<i32>());
}
"#;

#[test]
fn run_mode_btree_set_height2_from_iter() {
    // The same height-2 shape produced by the bulk build:
    // pushing past a full root creates the second level,
    // and iteration reads every element back in order.
    let dir = temp_dir("btreesetheight2fi");
    let src = fixture(
        &dir,
        "btreesetheight2fi.rs",
        BTREE_SET_HEIGHT2_FROM_ITER_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a from_iter-built height-2 BTreeSet must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "count=150 sum=11175\n");
}

const BTREE_MAP_HEIGHT2_FROM_ITER_SRC: &str = r#"
use std::collections::BTreeMap;
fn main() {
    let m: BTreeMap<i32, i32> = (0..150).map(|i| (i, i * 3)).collect();
    println!("count={} sum={}", m.iter().count(), m.iter().map(|(k, v)| k + v).sum::<i32>());
}
"#;

#[test]
fn run_mode_btree_map_height2_from_iter() {
    // The bulk-built height-2 shape for a map: the second
    // level's internal nodes carry their separator pairs,
    // and iteration interleaves them with the leaves.
    let dir = temp_dir("btreemapheight2fi");
    let src = fixture(
        &dir,
        "btreemapheight2fi.rs",
        BTREE_MAP_HEIGHT2_FROM_ITER_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a from_iter-built height-2 BTreeMap must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "count=150 sum=44700\n");
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

const ITER_MUT_SRC: &str = r#"
fn bump(xs: &mut [i32]) {
    for x in xs {
        *x += 1;
    }
}
fn main() {
    let mut a = [1, 2, 3];
    for x in &mut a {
        *x += 10;
    }
    println!("a={} {} {}", a[0], a[1], a[2]);
    let mut v = vec![1, 2, 3];
    for x in &mut v {
        *x *= 2;
    }
    println!("v={} {} {}", v[0], v[1], v[2]);
    let mut s = [5, 6, 7];
    bump(&mut s);
    println!("s={} {} {}", s[0], s[1], s[2]);
}
"#;

#[test]
fn run_mode_iter_mut() {
    let dir = temp_dir("iter-mut");
    let src = fixture(&dir, "iter_mut.rs", ITER_MUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "IterMut over array, Vec and slice must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "a=11 12 13\nv=2 4 6\ns=6 7 8\n",
        "writes through the yielded references land in the container"
    );
}

const ITER_MUT_METHOD_SRC: &str = r#"
fn main() {
    // The `iter_mut` method builds the iterator's real state;
    // its consumers step element pointers through the
    // container's own storage.
    let mut a = [1, 2, 3, 4];
    let n = a.iter_mut().count();
    println!("count {}", n);
    for x in a.iter_mut() {
        *x *= 10;
    }
    println!("{:?}", a);
    let mut v = vec![5, 6, 7];
    let mut it = v.iter_mut();
    println!("len {} hint {:?}", it.len(), it.size_hint());
    *it.next().unwrap() += 100;
    *it.next_back().unwrap() += 100;
    println!("len {} {:?}", it.len(), v);
    let mut boxed: Box<[i32]> = Box::new([1, 2]);
    for x in boxed.iter_mut() {
        *x += 1;
    }
    println!("{:?}", boxed);
    let mut e: Vec<i32> = Vec::new();
    println!("empty {}", (&mut e).into_iter().count());
    let mut g = [1, 2, 3];
    println!("refcount {}", (&mut g).into_iter().count());
}
"#;

#[test]
fn run_mode_iter_mut_method_forms() {
    let dir = temp_dir("iter-mut-method");
    let src = fixture(&dir, "iter_mut_method.rs", ITER_MUT_METHOD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "slice::IterMut method forms must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "count 4\n[10, 20, 30, 40]\nlen 3 hint (3, Some(3))\nlen 1 [105, 6, 107]\n[2, 3]\nempty 0\nrefcount 3\n",
        "IterMut steps, counts and lengths read the real iterator state"
    );
}

const ITER_MUT_ZST_SRC: &str = r#"
fn main() {
    // A zero-sized element type stores the remaining length in
    // the iterator's end field as a provenance-free address.
    let mut z = [(), (), ()];
    let mut n = 0;
    for _ in z.iter_mut() {
        n += 1;
    }
    println!("zst {}", n);
    let mut w = [(), (), (), ()];
    let mut it = (&mut w).into_iter();
    it.next_back();
    println!("zlen {}", it.len());
    let mut m = 0;
    while it.next().is_some() {
        m += 1;
    }
    println!("zrest {}", m);
}
"#;

#[test]
fn run_mode_iter_mut_zst() {
    let dir = temp_dir("iter-mut-zst");
    let src = fixture(&dir, "iter_mut_zst.rs", ITER_MUT_ZST_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "slice::IterMut over a zero-sized element type must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "zst 3\nzlen 3\nzrest 3\n",
        "the length stored in the end field drives every step"
    );
}

const ITER_MUT_ZST_MIXED_SRC: &str = r#"
fn main() {
    // Regression for the front/back count disagreement: the
    // front step of a storage-backed ZST `IterMut` advances the
    // front pointer's element index while the end field carries
    // the terminal index, so a back step must read the end
    // field as that terminal index, not as a remaining count.
    // Front step then two back steps over a two-element array
    // yield two items total.
    let mut a = [(); 2];
    let mut it = (&mut a).into_iter();
    println!("repro {} {} {} {}",
        it.next().is_some(), it.next_back().is_some(),
        it.next_back().is_some(), it.next().is_some());
    // Back first, then front, on three elements.
    let mut b = [(); 3];
    let mut it = (&mut b).into_iter();
    println!("bf {} {} {} {} {}",
        it.next_back().is_some(), it.next().is_some(),
        it.next_back().is_some(), it.next().is_some(),
        it.next_back().is_some());
    // Alternating drain of five, with the size hint after
    // each step.
    let mut c = [(); 5];
    let mut it = (&mut c).into_iter();
    let mut total = 0;
    println!("hint0 {:?}", it.size_hint());
    if it.next().is_some() { total += 1; }
    println!("hint1 {:?}", it.size_hint());
    if it.next_back().is_some() { total += 1; }
    println!("hint2 {:?}", it.size_hint());
    if it.next_back().is_some() { total += 1; }
    if it.next().is_some() { total += 1; }
    if it.next_back().is_some() { total += 1; }
    println!("drain {} end {} {}", total,
        it.next().is_some(), it.next_back().is_some());
    // The collected `iter_mut` body shape, over a Vec and an
    // array, mixes the same way.
    let mut v = vec![(); 3];
    let mut it = v.iter_mut();
    let mut n = 0;
    if it.next().is_some() { n += 1; }
    while it.next_back().is_some() { n += 1; }
    println!("vec {}", n);
    let mut d = [(); 4];
    let mut it = d.iter_mut();
    let mut n = 0;
    if it.next_back().is_some() { n += 1; }
    while it.next().is_some() { n += 1; }
    println!("arr {}", n);
    // A named zero-sized struct element type.
    #[derive(Clone, Copy)]
    struct Zst;
    let mut e = [Zst; 3];
    let mut it = (&mut e).into_iter();
    let mut n = 0;
    if it.next().is_some() { n += 1; }
    while it.next_back().is_some() { n += 1; }
    println!("named {}", n);
    // The immutable iterator over the same shapes.
    let f = [(); 3];
    let mut it = f.iter();
    let mut n = 0;
    if it.next().is_some() { n += 1; }
    while it.next_back().is_some() { n += 1; }
    println!("iter {}", n);
    // Rev composes the back step.
    let mut g = [(); 3];
    let n = (&mut g).into_iter().rev().count();
    println!("rev {}", n);
}
"#;

#[test]
fn run_mode_iter_mut_zst_front_back_mixed() {
    let dir = temp_dir("iter-mut-zst-mixed");
    let src = fixture(&dir, "iter_mut_zst_mixed.rs", ITER_MUT_ZST_MIXED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "mixed front/back steps of a ZST slice::IterMut must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "repro true true false false\nbf true true true false false\nhint0 (5, Some(5))\nhint1 (4, Some(4))\nhint2 (3, Some(3))\ndrain 5 end false false\nvec 3\narr 4\nnamed 3\niter 3\nrev 3\n",
        "front and back steps agree on the remaining count in every shape"
    );
}

const HASH_COLLECTION_DEFAULT_SRC: &str = r#"
use std::collections::{HashMap, HashSet};

fn main() {
    let e: HashSet<i32> = Default::default();
    println!("empty contains={} len={}", e.contains(&9), e.len());
    let mut s: HashSet<i32> = Default::default();
    for i in 0..100 {
        s.insert(i * 3);
    }
    println!(
        "grown len={} has99={} has100={}",
        s.len(),
        s.contains(&99),
        s.contains(&100)
    );
    let sum: i32 = s.iter().sum();
    println!("iter sum={}", sum);
    let mut m: HashMap<String, i32> = Default::default();
    for i in 0..50 {
        m.insert(format!("k{}", i), i);
    }
    *m.get_mut(&"k7".to_string()).unwrap() += 100;
    m.remove(&"k0".to_string());
    println!(
        "map len={} k7={:?} k0={:?}",
        m.len(),
        m.get(&"k7".to_string()),
        m.get(&"k0".to_string())
    );
    println!("values sum={}", m.values().sum::<i32>());
}
"#;

#[test]
fn run_mode_default_for_hash_collections() {
    // `Default::default` for the std hash collections has no
    // collected body (the trait path reaches the interpreter's
    // name fallback), so the model assembles the empty table
    // directly. The expected lines are the native 1.100.0
    // output of the same source: probes of the empty table,
    // growth out of the empty singleton, iteration, and
    // mutation of a default-built map.
    run_ok(
        "hash-collection-default",
        HASH_COLLECTION_DEFAULT_SRC,
        "empty contains=false len=0\ngrown len=100 has99=true has100=false\niter sum=14850\nmap len=49 k7=Some(107) k0=None\nvalues sum=1325\n",
    );
}

const NEXT_BACK_SLICE_SRC: &str = r#"
fn main() {
    let v = vec![10, 20, 30, 40];
    let mut it = v.iter();
    println!("back={:?}", it.next_back());
    println!("front={:?}", it.next());
    println!("back={:?}", it.next_back());
    println!("back={:?}", it.next_back());
    println!("back={:?}", it.next_back());
}
"#;

#[test]
fn run_mode_next_back_slice_iter() {
    run_ok(
        "next-back-slice",
        NEXT_BACK_SLICE_SRC,
        "back=Some(40)\nfront=Some(10)\nback=Some(30)\nback=Some(20)\nback=None\n",
    );
}

const NEXT_BACK_VEC_INTO_SRC: &str = r#"
fn main() {
    let mut it = vec![1, 2, 3].into_iter();
    println!("back={:?}", it.next_back());
    println!("front={:?}", it.next());
    println!("back={:?}", it.next_back());
    println!("back={:?}", it.next_back());
}
"#;

#[test]
fn run_mode_next_back_vec_into_iter() {
    run_ok(
        "next-back-vec-into",
        NEXT_BACK_VEC_INTO_SRC,
        "back=Some(3)\nfront=Some(1)\nback=Some(2)\nback=None\n",
    );
}

const NEXT_BACK_RANGE_SRC: &str = r#"
fn main() {
    let mut it = 1..5;
    println!("back={:?}", it.next_back());
    println!("front={:?}", it.next());
    println!("back={:?}", it.next_back());
    let mut incl = 1..=3;
    println!("incl back={:?}", incl.next_back());
    println!("incl front={:?}", incl.next());
    println!("incl back={:?}", incl.next_back());
    println!("incl back={:?}", incl.next_back());
}
"#;

#[test]
fn run_mode_next_back_ranges() {
    run_ok(
        "next-back-ranges",
        NEXT_BACK_RANGE_SRC,
        "back=Some(4)\nfront=Some(1)\nback=Some(3)\nincl back=Some(3)\nincl front=Some(1)\nincl back=Some(2)\nincl back=None\n",
    );
}

const NEXT_BACK_ADAPTERS_SRC: &str = r#"
fn main() {
    let v = vec![1, 2, 3];
    let mut copied = v.iter().copied();
    println!("copied back={:?}", copied.next_back());
    let mut mapped = (1..=3).map(|x| x * 10);
    println!("map back={:?}", mapped.next_back());
    println!("map front={:?}", mapped.next());
    let mut enumerated = v.iter().enumerate();
    println!("enum back={:?}", enumerated.next_back());
    println!("enum front={:?}", enumerated.next());
    let mut reversed = (1..=3).rev();
    println!("rev back={:?}", reversed.next_back());
    println!("rev front={:?}", reversed.next());
    let mut once = std::iter::once(42);
    println!("once back={:?}", once.next_back());
    println!("once back={:?}", once.next_back());
}
"#;

#[test]
fn run_mode_next_back_adapters() {
    run_ok(
        "next-back-adapters",
        NEXT_BACK_ADAPTERS_SRC,
        "copied back=Some(3)\nmap back=Some(30)\nmap front=Some(10)\nenum back=Some((2, 3))\nenum front=Some((0, 1))\nrev back=Some(1)\nrev front=Some(3)\nonce back=Some(42)\nonce back=None\n",
    );
}

const NEXT_BACK_OPTION_SRC: &str = r#"
fn main() {
    let mut it = Some(9).into_iter();
    println!("opt back={:?}", it.next_back());
    println!("opt back={:?}", it.next_back());
    let mut front = Some(7).into_iter();
    println!("opt front={:?}", front.next());
    println!("opt back={:?}", front.next_back());
}
"#;

#[test]
fn run_mode_next_back_option_into_iter() {
    run_ok(
        "next-back-option",
        NEXT_BACK_OPTION_SRC,
        "opt back=Some(9)\nopt back=None\nopt front=Some(7)\nopt back=None\n",
    );
}

const NEXT_BACK_USER_SRC: &str = r#"
struct Countdown(u32);
impl Iterator for Countdown {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.0 == 0 { None } else { self.0 -= 1; Some(self.0) }
    }
}
impl DoubleEndedIterator for Countdown {
    fn next_back(&mut self) -> Option<u32> {
        if self.0 == 0 { None } else { let v = self.0; self.0 -= 1; Some(v) }
    }
}
fn main() {
    let mut it = Countdown(3);
    println!("back={:?}", it.next_back());
    println!("front={:?}", it.next());
    println!("back={:?}", it.next_back());
    println!("back={:?}", it.next_back());
}
"#;

#[test]
fn run_mode_next_back_user_defined_iterator() {
    // A user-defined `DoubleEndedIterator` keeps its collected
    // body as the primary execution: the builtin only serves
    // the modelled iterator types.
    run_ok(
        "next-back-user",
        NEXT_BACK_USER_SRC,
        "back=Some(3)\nfront=Some(1)\nback=Some(1)\nback=None\n",
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

// The map is held behind a trait-object box, the shape of
// rustc's issue-6128 test: a plain local `HashMap` value's drop
// glue is a separate, still-unmodelled conversion gap.
const HASHMAP_NEW_SRC: &str = r#"
use std::collections::HashMap;
trait G { fn g(&self); }
impl G for HashMap<i32, i32> { fn g(&self) {} }
fn main() {
    let g: Box<HashMap<i32, i32>> = Box::new(HashMap::new());
    let _g2: Box<dyn G> = g as Box<dyn G>;
    println!("ok");
}
"#;

#[test]
fn run_mode_hashmap_new_uses_modelled_random_keys() {
    let dir = temp_dir("hashmap-new");
    let src = fixture(&dir, "hashmap_new.rs", HASHMAP_NEW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const HASHSET_FLOOD_FILL_SRC: &str = r#"
use std::collections::HashSet;

#[derive(Clone, PartialEq, Eq, Hash)]
struct XYZ {
    x: isize,
    y: isize,
    z: isize,
}

fn inside(x: isize, y: isize, z: isize) -> bool {
    x * x + 2 * y * y + 3 * z * z <= 60
}

fn main() {
    let mut connected = HashSet::new();
    let mut todo = HashSet::new();
    let origin = XYZ { x: 0, y: 0, z: 0 };
    connected.insert(origin.clone());
    todo.insert(origin);
    while connected.len() < 120 {
        let current = todo.iter().next().unwrap().clone();
        todo.remove(&current);
        let deltas = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ];
        for d in deltas {
            let next = XYZ {
                x: current.x + d.0,
                y: current.y + d.1,
                z: current.z + d.2,
            };
            if inside(next.x, next.y, next.z) && !connected.contains(&next) {
                connected.insert(next.clone());
                todo.insert(next);
            }
        }
    }
    assert!(connected.len() >= 120);
    assert!(connected.contains(&XYZ { x: 0, y: 0, z: 0 }));
    println!("ok");
}
"#;

#[test]
fn run_mode_hashset_flood_fill() {
    // A flood fill over two growing `HashSet`s: insert /
    // contains / remove / iterate through the collected
    // hashbrown bodies, reading bucket arrays that live in
    // allocations. Guards the borrowing read projection the
    // sets' per-operation cost now rests on: a read that
    // projected the wrong bucket would fail the membership
    // assertions in the guest.
    let dir = temp_dir("hashset-flood-fill");
    let src = fixture(&dir, "hashset_flood_fill.rs", HASHSET_FLOOD_FILL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const RANDOM_STATE_DEFAULT_SRC: &str = r#"
use std::collections::HashMap;
use std::hash::RandomState;
trait G { fn g(&self); }
impl G for HashMap<i32, i32> { fn g(&self) {} }
fn main() {
    let h: Box<HashMap<i32, i32>> = Box::new(HashMap::with_hasher(RandomState::default()));
    let _h2: Box<dyn G> = h as Box<dyn G>;
    println!("ok");
}
"#;

#[test]
fn run_mode_random_state_default_builds_hasher() {
    let dir = temp_dir("random-state-default");
    let src = fixture(&dir, "random_state_default.rs", RANDOM_STATE_DEFAULT_SRC);
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
}

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

// ----- HashMap / hashbrown raw-table execution (Lane HX) -----

const HASHMAP_BASIC_SRC: &str = r#"
use std::collections::HashMap;
fn main() {
    let mut m: HashMap<i32, i32> = HashMap::new();
    m.insert(1, 2);
    println!("{}", m[&1]);
}
"#;

#[test]
fn run_mode_hashmap_insert_and_index() {
    let dir = temp_dir("hashmap-basic");
    let src = fixture(&dir, "hashmap_basic.rs", HASHMAP_BASIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "2\n");
}

const HASHMAP_RESIZE_SRC: &str = r#"
use std::collections::HashMap;
fn main() {
    let mut m: HashMap<i32, i32> = HashMap::new();
    for i in 0..10 {
        m.insert(i, i * 10);
    }
    println!("len {}", m.len());
    for i in 0..10 {
        assert_eq!(m[&i], i * 10);
    }
    println!("resize ok");
}
"#;

#[test]
fn run_mode_hashmap_resize_rehashes_entries() {
    let dir = temp_dir("hashmap-resize");
    let src = fixture(&dir, "hashmap_resize.rs", HASHMAP_RESIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len 10\nresize ok\n");
}

const HASHMAP_STRING_KEYS_SRC: &str = r#"
use std::collections::HashMap;
fn main() {
    let mut m: HashMap<String, String> = HashMap::new();
    for (k, v) in [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")] {
        m.insert(k.to_string(), v.to_string());
    }
    println!("len {}", m.len());
    println!("got {}", &m["c"]);
}
"#;

#[test]
fn run_mode_hashmap_string_keys_across_resize() {
    let dir = temp_dir("hashmap-string-keys");
    let src = fixture(&dir, "hashmap_string_keys.rs", HASHMAP_STRING_KEYS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len 4\ngot 3\n");
}

const HASHMAP_COW_KEYS_SRC: &str = r#"
use std::borrow::Cow;
use std::collections::HashMap;
fn main() {
    let mut m: HashMap<Cow<str>, i32> = HashMap::new();
    m.insert(Cow::Borrowed("foo"), 42);
    assert_eq!(m.get(&Cow::Owned("foo".to_string())), Some(&42));
    assert_eq!(m.insert(Cow::Owned("foo".to_string()), 43), Some(42));
    println!("cow ok {}", m.len());
}
"#;

#[test]
fn run_mode_hashmap_cow_keys_borrowed_owned_agree() {
    let dir = temp_dir("hashmap-cow-keys");
    let src = fixture(&dir, "hashmap_cow_keys.rs", HASHMAP_COW_KEYS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "cow ok 1\n");
}

const HASHMAP_ITER_SRC: &str = r#"
use std::collections::HashMap;
fn main() {
    let mut m: HashMap<i32, i32> = HashMap::new();
    m.insert(1, 10);
    m.insert(2, 20);
    m.insert(3, 30);
    let mut sum = 0;
    for (k, v) in &m {
        sum += k + v;
    }
    assert_eq!(sum, 66);
    let mut it = m.iter();
    let mut seen = 0;
    for _ in it.by_ref() {
        seen += 1;
        break;
    }
    for _ in it {
        seen += 1;
    }
    assert_eq!(seen, 3);
    println!("iter ok");
}
"#;

#[test]
fn run_mode_hashmap_iter_and_by_ref_restart() {
    let dir = temp_dir("hashmap-iter");
    let src = fixture(&dir, "hashmap_iter.rs", HASHMAP_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "iter ok\n");
}

const HASHSET_ITER_SRC: &str = r#"
use std::collections::HashSet;
fn main() {
    let mut s: HashSet<i32> = HashSet::new();
    s.insert(1);
    s.insert(2);
    let mut n = 0;
    for x in &s {
        n += x;
    }
    assert_eq!(n, 3);
    println!("set ok");
}
"#;

#[test]
fn run_mode_hashset_iter() {
    let dir = temp_dir("hashset-iter");
    let src = fixture(&dir, "hashset_iter.rs", HASHSET_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "set ok\n");
}

const HASHMAP_WITH_CAPACITY_SRC: &str = r#"
use std::collections::HashMap;
fn main() {
    let mut m: HashMap<u8, u8> = HashMap::with_capacity(100);
    m.insert(7, 9);
    println!("{}", m[&7]);
}
"#;

#[test]
fn run_mode_hashmap_with_capacity() {
    let dir = temp_dir("hashmap-capacity");
    let src = fixture(&dir, "hashmap_capacity.rs", HASHMAP_WITH_CAPACITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "9\n");
}

const NONNULL_CAST_DENOMINATION_SRC: &str = r#"
use std::ptr::NonNull;
struct Inner { ctrl: NonNull<u8>, mask: usize }
struct Bucket<T> { ptr: NonNull<T> }
impl Inner {
    fn data_end<T>(&self) -> NonNull<T> { self.ctrl.cast() }
    unsafe fn bucket<T>(&self, index: usize) -> Bucket<T> {
        unsafe { Bucket::from_base_index(self.data_end(), index) }
    }
}
impl<T> Bucket<T> {
    unsafe fn from_base_index(base: NonNull<T>, index: usize) -> Self {
        let ptr = unsafe { base.as_ptr().sub(index) };
        Bucket { ptr: unsafe { NonNull::new_unchecked(ptr) } }
    }
    unsafe fn write(&self, value: T) {
        unsafe { self.ptr.as_ptr().sub(1).write(value) }
    }
}
fn main() {
    unsafe {
        let layout = std::alloc::Layout::from_size_align(64, 8).unwrap();
        let base = std::alloc::alloc(layout);
        let inner = Inner { ctrl: NonNull::new_unchecked(base.add(32)), mask: 3 };
        let b: Bucket<(i32, i32)> = inner.bucket(0);
        b.write((7i32, 8i32));
        let back = base.add(24) as *const (i32, i32);
        let v = back.read();
        assert_eq!(v, (7, 8));
        std::alloc::dealloc(base, layout);
    }
    println!("denomination ok");
}
"#;

#[test]
fn run_mode_nonnull_cast_denominates_pointer_units() {
    let dir = temp_dir("nonnull-cast-units");
    let src = fixture(&dir, "nonnull_cast_units.rs", NONNULL_CAST_DENOMINATION_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "denomination ok\n");
}

const SWAP_STRUCT_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct TableLike { mask: usize, ctrl: u64, growth: usize, items: usize }
fn main() {
    let mut a = TableLike { mask: 3, ctrl: 11, growth: 5, items: 2 };
    let mut b = TableLike { mask: 7, ctrl: 22, growth: 1, items: 6 };
    std::mem::swap(&mut a, &mut b);
    assert_eq!(a, TableLike { mask: 7, ctrl: 22, growth: 1, items: 6 });
    assert_eq!(b, TableLike { mask: 3, ctrl: 11, growth: 5, items: 2 });
    println!("swap ok");
}
"#;

#[test]
fn run_mode_swap_nonoverlapping_whole_struct() {
    let dir = temp_dir("swap-struct");
    let src = fixture(&dir, "swap_struct.rs", SWAP_STRUCT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "swap ok\n");
}
const BTREE_ITER_WRAPPER_PTR_SRC: &str = r#"
use std::borrow::Cow;
use std::collections::BTreeMap;
fn main() {
    let mut m: BTreeMap<i32, i32> = BTreeMap::new();
    m.insert(1, 10);
    m.insert(2, 20);
    println!("count={}", m.iter().count());
    let mut c: BTreeMap<Cow<'static, str>, usize> = BTreeMap::new();
    assert!(c.insert(Cow::Borrowed("foo"), 42).is_none());
    assert!(c.insert(Cow::Owned("foo".to_string()), 43).is_some());
    println!("len={} count={}", c.len(), c.iter().count());
    println!("get={:?}", c.get("foo"));
}
"#;

#[test]
fn run_mode_btree_map_iter_reads_leaf_nodes() {
    // Iterating a BTreeMap walks its leaf nodes through the
    // node pointer stored in the root's NodeRef: the pointer
    // enters the node's `MaybeUninit` storage layer with its
    // field-0 step, and the stored element at that address is
    // the leaf aggregate itself ([parent, parent index,
    // length, keys, values]). The read must land on the
    // aggregate the pointer addresses, not descend one field
    // further into the leaf's parent slot.
    run_ok(
        "btree-iter-wrapper-ptr",
        BTREE_ITER_WRAPPER_PTR_SRC,
        "count=2\nlen=1 count=1\nget=Some(43)\n",
    );
}

const BTREE_ZST_ITER_SRC: &str = r#"
use std::collections::BTreeMap;
#[derive(Eq, Hash, Debug, Ord, PartialEq, PartialOrd)]
struct Zst;
fn main() {
    let mut m: BTreeMap<Zst, ()> = BTreeMap::new();
    m.insert(Zst, ());
    m.insert(Zst, ());
    println!("len={} count={}", m.len(), m.iter().count());
    println!("get={}", m.get(&Zst).is_some());
}
"#;

#[test]
fn run_mode_btree_map_zst_iter_count() {
    // A zero-sized key/value BTreeMap: the leaf walk reads the
    // node through the same wrapper-entering node pointer as
    // the sized case, and the elided ZST slots fill out to the
    // declared length.
    run_ok("btree-zst-iter", BTREE_ZST_ITER_SRC, "len=1 count=1\nget=true\n");
}

// ----- HashSet / HashMap collect (FromIterator) execution (#350) -----
//
// `Iterator::collect` / `FromIterator::from_iter` into a hash
// collection cannot be served by the consumer builtin (it builds
// only the target shapes it models natively), so the collected
// instance bodies -- `with_hasher` + `extend` over the raw-table
// memory model -- execute instead. These tests pin that path for
// the modelled iterator sources (array / slice / range / vec),
// which previously failed with "collect into std::collections::
// HashSet<...> is not runtime-supported".

const HASHSET_COLLECT_ARRAY_SRC: &str = r#"
use std::collections::HashSet;
fn main() {
    let s: HashSet<i32> = [1, 2, 2, 3].into_iter().collect();
    println!("len={} has2={} sum={}", s.len(), s.contains(&2), s.iter().sum::<i32>());
}
"#;

#[test]
fn run_mode_hashset_collect_from_array() {
    let dir = temp_dir("hashset-collect-array");
    let src = fixture(&dir, "hashset_collect_array.rs", HASHSET_COLLECT_ARRAY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len=3 has2=true sum=6\n");
}

const HASHSET_COLLECT_RANGE_SRC: &str = r#"
use std::collections::HashSet;
fn main() {
    let s: HashSet<u64> = (0..100).collect();
    println!("len={} has99={} sum={}", s.len(), s.contains(&99), s.iter().sum::<u64>());
}
"#;

#[test]
fn run_mode_hashset_collect_from_range_grows_table() {
    let dir = temp_dir("hashset-collect-range");
    let src = fixture(&dir, "hashset_collect_range.rs", HASHSET_COLLECT_RANGE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len=100 has99=true sum=4950\n"
    );
}

const HASHSET_COLLECT_VEC_STRING_SRC: &str = r#"
use std::collections::HashSet;
fn main() {
    let v = vec!["x".to_string(), "y".to_string(), "x".to_string()];
    let s: HashSet<String> = v.into_iter().collect();
    println!("len={} hasy={}", s.len(), s.contains("y"));
}
"#;

#[test]
fn run_mode_hashset_collect_from_vec_strings() {
    let dir = temp_dir("hashset-collect-vec-string");
    let src = fixture(&dir, "hashset_collect_vec_string.rs", HASHSET_COLLECT_VEC_STRING_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len=2 hasy=true\n");
}

const HASHSET_COLLECT_COPIED_SRC: &str = r#"
use std::collections::HashSet;
fn main() {
    let a = [1, 2, 3, 3];
    let s: HashSet<i32> = a.iter().copied().collect();
    println!("len={} sum={}", s.len(), s.iter().sum::<i32>());
}
"#;

#[test]
fn run_mode_hashset_collect_copied() {
    let dir = temp_dir("hashset-collect-copied");
    let src = fixture(&dir, "hashset_collect_copied.rs", HASHSET_COLLECT_COPIED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len=3 sum=6\n");
}

const HASHSET_FROM_ITER_SRC: &str = r#"
use std::collections::HashSet;
fn main() {
    let s: HashSet<i32> = HashSet::from_iter([7, 8, 8]);
    println!("len={} has7={}", s.len(), s.contains(&7));
}
"#;

#[test]
fn run_mode_hashset_from_iter_direct() {
    let dir = temp_dir("hashset-from-iter");
    let src = fixture(&dir, "hashset_from_iter.rs", HASHSET_FROM_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len=2 has7=true\n");
}

const HASHMAP_COLLECT_ARRAY_SRC: &str = r#"
use std::collections::HashMap;
fn main() {
    let m: HashMap<&str, i32> = [("a", 1), ("b", 2), ("a", 3)].into_iter().collect();
    println!("len={} a={:?} b={:?}", m.len(), m.get("a"), m.get("b"));
}
"#;

#[test]
fn run_mode_hashmap_collect_from_array() {
    let dir = temp_dir("hashmap-collect-array");
    let src = fixture(&dir, "hashmap_collect_array.rs", HASHMAP_COLLECT_ARRAY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len=2 a=Some(3) b=Some(2)\n"
    );
}

const HASHMAP_COLLECT_VEC_PAIRS_SRC: &str = r#"
use std::collections::HashMap;
fn main() {
    let mut v = Vec::new();
    for i in 0..100u64 {
        v.push((i, i * 2));
    }
    let m: HashMap<u64, u64> = v.into_iter().collect();
    println!("len={} m77={:?}", m.len(), m.get(&77));
}
"#;

#[test]
fn run_mode_hashmap_collect_from_vec_pairs_grows_table() {
    let dir = temp_dir("hashmap-collect-vec-pairs");
    let src = fixture(&dir, "hashmap_collect_vec_pairs.rs", HASHMAP_COLLECT_VEC_PAIRS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len=100 m77=Some(154)\n");
}

const HASHSET_RECOLLECT_SRC: &str = r#"
use std::collections::HashSet;
fn main() {
    let s: HashSet<i32> = [4, 5, 6].into_iter().collect();
    let t: HashSet<i32> = s.into_iter().collect();
    println!("len={} sum={}", t.len(), t.iter().sum::<i32>());
}
"#;

#[test]
fn run_mode_hashset_recollect_from_own_into_iter() {
    let dir = temp_dir("hashset-recollect");
    let src = fixture(&dir, "hashset_recollect.rs", HASHSET_RECOLLECT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len=3 sum=15\n");
}

const CHUNKS_ITERATION_SRC: &str = r#"
fn main() {
    let v = [1, 2, 3, 4, 5];
    let mut c = v.chunks(2);
    println!("{:?}", c.next());
    println!("{:?}", c.next_back());
    println!("{:?}", c.next());
    println!("{:?}", c.next());
    let r: Vec<&[i32]> = v.rchunks(2).collect();
    println!("{:?}", r);
    let ce: Vec<&[i32]> = v.chunks_exact(2).collect();
    println!("{:?}", ce);
    println!("rem {:?}", v.chunks_exact(2).remainder());
    let mut rc = v.rchunks_exact(2);
    println!("{:?}", rc.next());
    println!("{:?}", rc.next_back());
    println!("{:?}", rc.next());
    println!("count {}", v.chunks(3).count());
    let s: i32 = v.chunks(2).map(|c| c.iter().sum::<i32>()).sum();
    println!("sum {}", s);
}
"#;

#[test]
fn run_mode_chunks_family_iteration() {
    // The slice chunk family (`Chunks` / `RChunks` and their
    // `Exact` variants) had a modelled `size_hint` but no
    // modelled step: every `next` on their real-layout state
    // ([remaining slice, chunk size], the `Exact` variants
    // adding a remainder slice) died as not runtime-supported.
    // The state machine now steps them, including the short
    // chunk each family yields last in its primary direction.
    // The outputs are native rustc 1.100.0's.
    let dir = temp_dir("chunks-iteration");
    let src = fixture(&dir, "chunks_iteration.rs", CHUNKS_ITERATION_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "chunks iteration must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "Some([1, 2])\nSome([5])\nSome([3, 4])\nNone\n[[4, 5], [2, 3], [1]]\n[[1, 2], [3, 4]]\nrem [5]\nSome([4, 5])\nSome([2, 3])\nNone\ncount 2\nsum 15\n"
    );
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

const HASH_SET_COLLECT_SRC: &str = r#"
use std::collections::HashSet;
fn main() {
    let mut s = HashSet::new();
    s.insert(3u32); s.insert(1); s.insert(2);
    let mut v: Vec<u32> = s.into_iter().collect();
    v.sort_unstable();
    println!("{:?}", v);
    let mut s2 = HashSet::new();
    s2.insert(10u32); s2.insert(20);
    println!("count {}", s2.into_iter().count());
}
"#;

#[test]
fn run_mode_hash_set_collect_into_vec() {
    // `collect::<Vec>()` over a hash-set iterator: the
    // nested specialization entry (`SpecFromIterNested`)
    // was pre-empted by the builtin for every iterator, and
    // the builtin's drain drives the modelled `next` state
    // machine, which cannot walk a raw table -- the collect
    // died on the first step even though the direct `next`
    // body chain worked. For a body-served iterator the
    // entry's collected body chain is the execution. The
    // outputs are native rustc 1.100.0's.
    let dir = temp_dir("hash-set-collect");
    let src = fixture(&dir, "hash_set_collect.rs", HASH_SET_COLLECT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "hash set collect must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 2, 3]\ncount 2\n"
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

const HASH_ITER_CONSUMERS_SRC: &str = r#"
use std::collections::{HashMap, HashSet};
fn main() {
    let s: HashSet<i32> = [1, 2, 3, 4].into_iter().collect();
    println!("all={}", s.iter().all(|x| *x > 0));
    println!("all2={}", s.iter().all(|x| *x > 2));
    println!("any={}", s.iter().any(|x| *x == 3));
    println!("any2={}", s.iter().any(|x| *x == 9));
    println!("find={:?}", s.iter().find(|x| **x == 3));
    println!("position_some={}", s.iter().position(|x| *x == 3).is_some());
    let mut it = s.iter();
    println!("nth_some={} rest={}", it.nth(1).is_some(), it.count());
    println!("count={} fold={}", s.iter().count(), s.iter().fold(0, |a, x| a + *x));
    let m: HashMap<i32, i32> = [(1, 10), (2, 20), (3, 30)].into_iter().collect();
    println!("mapall={}", m.iter().all(|(k, v)| *v == *k * 10));
    println!("mapany={}", m.iter().any(|(k, _)| *k == 2));
    println!("mapvalues={}", m.values().sum::<i32>());
}
"#;

#[test]
fn run_mode_hash_iter_consumers() {
    // The consumers that reach the iterator through
    // `try_fold` (`all` / `any` / `find` / `position`) and
    // `nth` over a hash-collection iterator: the builtin
    // consumers drive the modelled `next` state machine,
    // which cannot walk a raw table, so the collected
    // instance bodies are the execution -- the same rule the
    // older consumers (`count` / `fold`) already had. Every
    // printed value is iteration-order independent (a hash
    // table's order is not part of the contract).
    let dir = temp_dir("hash_iter_consumers");
    let src = fixture(&dir, "hash_iter_consumers.rs", HASH_ITER_CONSUMERS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "try_fold / nth consumers over hash iterators must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "all=true\nall2=false\nany=true\nany2=false\nfind=Some(3)\nposition_some=true\nnth_some=true rest=2\ncount=4 fold=10\nmapall=true\nmapany=true\nmapvalues=60\n"
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

const ADAPTER_OVER_HASH_COLLECT_SRC: &str = r#"
use std::collections::HashSet;
fn main() {
    let s: HashSet<i32> = [1, 2, 3, 4, 5].into_iter().collect();
    let mapped: Vec<i32> = s.iter().map(|x| *x + 1).collect();
    println!("mapcollect={}", mapped.iter().sum::<i32>());
    let filtered: Vec<i32> = s.iter().filter(|x| **x % 2 == 0).cloned().collect();
    println!("filtercollect={} {}", filtered.len(), filtered.iter().sum::<i32>());
    let cloned: Vec<i32> = s.iter().cloned().collect();
    println!("clonedcollect={}", cloned.len());
    let skipped: Vec<i32> = s.iter().skip(2).cloned().collect();
    println!("skipcollect={}", skipped.len());
    let taken: Vec<i32> = s.iter().take(2).cloned().collect();
    println!("takecollect={}", taken.len());
}
"#;

#[test]
fn run_mode_adapter_over_hash_iter_collect() {
    // Collecting an adapter over a hash iterator into a
    // `Vec`: the nested / in-place collect entries are
    // pre-empted by their builtin only for iterators the
    // state machine drains, and an adapter's step composes
    // its inner iterator's step, so an adapter over a
    // body-served iterator is itself body-served -- the
    // collected entry chain, stepping the adapter's own
    // collected `next` body, is the execution.
    let dir = temp_dir("adapter_hash_collect");
    let src = fixture(&dir, "adapter_hash_collect.rs", ADAPTER_OVER_HASH_COLLECT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "collect of adapters over a hash iterator must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "mapcollect=20\nfiltercollect=2 6\nclonedcollect=5\nskipcollect=3\ntakecollect=2\n"
    );
}

const LINKED_LIST_ITER_CONSUMERS_SRC: &str = r#"
use std::collections::LinkedList;
fn main() {
    let mut l: LinkedList<i32> = LinkedList::new();
    l.push_back(1);
    l.push_back(2);
    l.push_back(3);
    println!("llall={}", l.iter().all(|x| *x > 0));
    println!("llany={}", l.iter().any(|x| *x == 2));
    println!("llcount={}", l.iter().count());
    let mut it = l.iter();
    println!("llnth={:?}", it.nth(1));
}
"#;

#[test]
fn run_mode_linked_list_iter_consumers() {
    // The concrete collection iterators share the hash
    // iterators' arrangement (their `next` bodies are the
    // faithful execution), so the `try_fold` / `nth`
    // consumers over them follow the same rule.
    let dir = temp_dir("linked_list_iter_consumers");
    let src = fixture(&dir, "linked_list_iter_consumers.rs", LINKED_LIST_ITER_CONSUMERS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "try_fold / nth consumers over a LinkedList iterator must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "llall=true\nllany=true\nllcount=3\nllnth=Some(2)\n"
    );
}


const LINKED_LIST_COLLECT_SRC: &str = r#"
use std::collections::LinkedList;
fn main() {
    let mut l: LinkedList<i32> = LinkedList::new();
    l.push_back(1); l.push_back(2); l.push_back(3);
    let v: Vec<&i32> = l.iter().collect();
    println!("bare={:?}", v);
    for x in l.iter_mut() { *x += 10; }
    let v: Vec<&i32> = l.iter().collect();
    println!("mutated={:?}", v);
    let mut big: LinkedList<i32> = LinkedList::new();
    for i in 0..40 { big.push_back(i * 3); }
    let w: Vec<&i32> = big.iter().collect();
    println!("big={} {}", w.len(), w[39]);
    let v: Vec<i32> = big.into_iter().collect();
    println!("owned={} {}", v.len(), v.iter().sum::<i32>());
    let e: LinkedList<i32> = LinkedList::new();
    let v: Vec<&i32> = e.iter().collect();
    println!("empty={:?}", v);
}
"#;

#[test]
fn run_mode_linked_list_collect_into_vec() {
    // Collecting a `LinkedList` iterator into a `Vec` runs
    // the collected `TrustedLen` chain: the nested
    // `Vec::from_iter` body reads `size_hint`'s upper bound
    // as an exact capacity (and panics with `capacity
    // overflow` when it is `None`), then the trusted-extend
    // body drains the source. Both calls must come from the
    // collected bodies for a body-served iterator -- the
    // `size_hint` builtin answers the trait default
    // `(0, None)` for a real-layout state it does not read,
    // and the trusted-extend builtin drains through the
    // modelled `next` machine, which cannot step one.
    let dir = temp_dir("linkedlistcollect");
    let src = fixture(&dir, "linkedlistcollect.rs", LINKED_LIST_COLLECT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "collecting LinkedList iterators into a Vec must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "bare=[1, 2, 3]\nmutated=[11, 12, 13]\nbig=40 117\nowned=40 2340\nempty=[]\n"
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


const VEC_DEQUE_ITER_CONSUMERS_SRC: &str = r#"
use std::collections::VecDeque;
fn main() {
    let mut d = VecDeque::new();
    d.push_back(1); d.push_back(2); d.push_back(3);
    println!("count {}", d.iter().count());
    let v: Vec<&i32> = d.iter().collect();
    println!("collect {:?}", v);
    let mut seen = 0;
    let n = d.iter().fold(0, |a, x| { seen += 1; a + x });
    println!("fold {} {}", n, seen);
    let mut w = VecDeque::new();
    for i in 1..=6 { w.push_back(i); }
    for _ in 0..4 { w.pop_front(); }
    w.push_back(7); w.push_back(8);
    println!("wrapped {}", w.iter().count());
    let mut seen2 = 0;
    let s = w.iter().fold(0, |a, x| { seen2 += 1; a + x });
    println!("wrapped fold {} {}", s, seen2);
    let wv: Vec<&i32> = w.iter().collect();
    println!("wrapped collect {:?}", wv);
}
"#;

#[test]
fn run_mode_vec_deque_iter_consumers() {
    // `vec_deque::Iter`'s own `fold` / `try_fold` bodies
    // forward to the two inner slice iterators with `&mut f`:
    // the modelled consumer receives a reference to the
    // closure as its callable operand and must call through
    // it (the blanket `FnMut` impl for `&mut F`), with the
    // closure's capture mutations persisting across both
    // inner folds and back to the caller. The wrapped deque
    // exercises both slice halves.
    let dir = temp_dir("vecdequeiterconsumers");
    let src = fixture(&dir, "vecdequeiterconsumers.rs", VEC_DEQUE_ITER_CONSUMERS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "VecDeque iterator consumers must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "count 3\ncollect [1, 2, 3]\nfold 6 3\nwrapped 4\nwrapped fold 26 4\nwrapped collect [5, 6, 7, 8]\n"
    );
}

const VEC_DEQUE_INTO_ITER_FOLD_SRC: &str = r#"
use std::collections::VecDeque;
fn main() {
    let mut d: VecDeque<i32> = VecDeque::new();
    d.push_back(1); d.push_back(2); d.push_back(3);
    println!("fold {}", d.into_iter().fold(0, |a, b| a + b));
    let mut d: VecDeque<i32> = VecDeque::new();
    for i in 0..6 { d.push_back(i); }
    d.pop_front(); d.pop_front();
    d.push_back(6); d.push_back(7); d.push_back(8); d.push_back(9);
    println!("wrapped {}", d.into_iter().fold(0, |a, b| a + b));
    let mut d: VecDeque<i32> = VecDeque::new();
    d.push_back(1); d.push_back(2); d.push_back(3);
    println!("try_fold {:?}", d.into_iter().try_fold(0, |a, b| Some(a + b)));
    let mut d: VecDeque<i32> = VecDeque::new();
    d.push_back(1); d.push_back(2); d.push_back(3);
    let v: Vec<i32> = d.into_iter().collect();
    println!("collect {:?}", v);
    let mut d: VecDeque<String> = VecDeque::new();
    d.push_back("a".to_string()); d.push_back("b".to_string()); d.push_back("c".to_string());
    println!("strings {}", d.into_iter().fold(String::new(), |mut a, b| { a.push_str(&b); a }));
}
"#;

#[test]
fn run_mode_vecdeque_into_iter_fold() {
    // `IntoIter::fold` wraps the fold closure in a
    // `Result`-returning closure and forwards it through the
    // slice iterators' `Map` adapter: the wrapper closure is
    // called through a `&mut` stored in the adapter closure's
    // own capture field, so the `FnMut` receiver chain carries
    // an extra reference indirection. The environment
    // reference the wrapper body is entered with must denote
    // the wrapper closure itself, not the capture field
    // holding the reference to it - entered through the field,
    // the body's first capture read yields the stored `&mut`
    // (a reference to the wrapper closure), the call recurses
    // into the wrapper, and its `Result` return lands in the
    // inner call's integer destination.
    let dir = temp_dir("vecdequeintoiterfold");
    let src = fixture(&dir, "vecdequeintoiterfold.rs", VEC_DEQUE_INTO_ITER_FOLD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "VecDeque into_iter fold consumers must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "fold 6\nwrapped 44\ntry_fold Some(6)\ncollect [1, 2, 3]\nstrings abc\n"
    );
}

const LINKED_LIST_ADAPTER_COLLECT_SRC: &str = r#"
use std::collections::LinkedList;
fn main() {
    let mut l: LinkedList<i32> = LinkedList::new();
    l.push_back(1); l.push_back(2); l.push_back(3);
    let v: Vec<i32> = l.iter().cloned().collect();
    println!("cloned={:?}", v);
    println!("chint {:?}", l.iter().cloned().size_hint());
    let v: Vec<i32> = l.iter().copied().collect();
    println!("copied={:?}", v);
    let v: Vec<i32> = l.iter().map(|x| x + 1).collect();
    println!("mapped={:?}", v);
    let mut big: LinkedList<i32> = LinkedList::new();
    for i in 0..40 { big.push_back(i * 3); }
    let w: Vec<i32> = big.iter().cloned().collect();
    println!("big={} {}", w.len(), w[39]);
    let v: Vec<i32> = big.into_iter().map(|x| x + 1).collect();
    println!("ownedmapped={} {}", v.len(), v.iter().sum::<i32>());
}
"#;

#[test]
fn run_mode_linked_list_adapter_collect_into_vec() {
    // Adapter-wrapped `LinkedList` iterators collect
    // through the same body-served chain as the bare
    // source: the `size_hint` / `extend_trusted`
    // judgements recurse into the adapter's generic
    // arguments, so the collected bodies -- not the
    // builtins, which cannot step a real-layout iterator
    // state -- serve the calls.
    let dir = temp_dir("linkedlistadaptercollect");
    let src = fixture(
        &dir,
        "linkedlistadaptercollect.rs",
        LINKED_LIST_ADAPTER_COLLECT_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "collecting adapter-wrapped LinkedList iterators into a Vec must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "cloned=[1, 2, 3]\nchint (3, Some(3))\ncopied=[1, 2, 3]\nmapped=[2, 3, 4]\nbig=40 117\nownedmapped=40 2380\n"
    );
}
