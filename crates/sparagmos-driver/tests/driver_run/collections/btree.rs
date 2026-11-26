// BTreeMap / BTreeSet tests.



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

const BTREE_MAP_DEBUG_SRC: &str = r#"
use std::collections::{BTreeMap, BTreeSet};
fn main() {
    let empty: BTreeMap<i32, i32> = BTreeMap::new();
    println!("empty {:?}", empty);
    let mut m: BTreeMap<i32, i32> = BTreeMap::new();
    m.insert(2, 20);
    m.insert(1, 10);
    println!("map {:?}", m);
    let mut s: BTreeSet<i32> = BTreeSet::new();
    s.insert(3);
    s.insert(1);
    println!("set {:?}", s);
    println!("pretty {:#?}", m);
}
"#;

#[test]
fn run_mode_btree_map_debug_format() {
    // `{:?}` of a `BTreeMap` / `BTreeSet` is the same
    // `debug_map` / `debug_set` form as the hash collections',
    // composed from the pairs the B-tree node walk reads back
    // (in tree order, so the expected text is exact); the
    // storage form's uninitialised node slots are not part of
    // the rendering.
    let dir = temp_dir("btreemapdebug");
    let src = fixture(&dir, "btreemapdebug.rs", BTREE_MAP_DEBUG_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "formatting a BTreeMap with {{:?}} must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "empty {}\nmap {1: 10, 2: 20}\nset {1, 3}\npretty {\n    1: 10,\n    2: 20,\n}\n"
    );
}