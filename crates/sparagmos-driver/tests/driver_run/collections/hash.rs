// HashMap / HashSet tests.


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

const HASH_MAP_DEBUG_SRC: &str = r#"
use std::collections::{HashMap, HashSet};
fn main() {
    let empty: HashMap<i32, i32> = HashMap::new();
    println!("empty {:?}", empty);
    let mut one: HashMap<i32, i32> = HashMap::new();
    one.insert(1, 10);
    println!("one {:?}", one);
    let mut m: HashMap<String, i32> = HashMap::new();
    m.insert("a".to_string(), 1);
    println!("strkey {:?}", m);
    let mut s: HashSet<i32> = HashSet::new();
    s.insert(7);
    println!("set {:?}", s);
    let mut v: HashMap<i32, Vec<i32>> = HashMap::new();
    v.insert(1, vec![1, 2]);
    println!("vecval {:?}", v);
    println!("pretty {:#?}", one);
    let d: HashMap<String, Vec<i32>> = HashMap::default();
    println!("default {:?}", d);
}
"#;

#[test]
fn run_mode_hash_map_debug_format() {
    // `{:?}` of a `HashMap` / `HashSet` is std's handwritten
    // `debug_map` / `debug_set` form (`{k: v, ...}` / `{a, ...}`),
    // not the derived form of the storage aggregate (`HashMap
    // { base: HashMap { hash_builder: ..., table: RawTable
    // { ... } } }`): the fmt engine composes it from the entries
    // the raw-table walk reads back. Single-element maps keep
    // the expected text exact (a multi-element hash map's entry
    // order is the table's bucket order, which std leaves to
    // the hasher). The empty map is the `{}` form, not an error
    // over the empty table's control-byte block.
    let dir = temp_dir("hashmapdebug");
    let src = fixture(&dir, "hashmapdebug.rs", HASH_MAP_DEBUG_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "formatting a HashMap with {{:?}} must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "empty {}\none {1: 10}\nstrkey {\"a\": 1}\nset {7}\nvecval {1: [1, 2]}\npretty {\n    1: 10,\n}\ndefault {}\n"
    );
}
const HASHSET_LARGE_CLONE_SRC: &str = r#"
use std::collections::HashSet;

fn main() {
    let mut s: HashSet<i32> = HashSet::new();
    for i in 0..1200 {
        s.insert(i);
    }
    let t = s.clone();
    println!("len={} sum={}", t.len(), t.iter().sum::<i32>());
    println!("has0={} has1199={} has1200={}", t.contains(&0), t.contains(&1199), t.contains(&1200));
    let mut b: Box<[i32]> = vec![0; 1200].into_boxed_slice();
    for i in 0..1200 {
        b[i] = (i * 2) as i32;
    }
    println!("boxed sum={}", b.iter().sum::<i32>());
}
"#;

#[test]
fn run_mode_hashset_large_clone_and_boxed_writes() {
    // Regression for #399: filling and copying a large hash
    // table writes single elements into a large buffer one by
    // one; the pointer-crossing probe on each projected write
    // must not re-read (clone) the whole buffer per element,
    // which made such programs cost superlinearly per step.
    // The clone and the element-by-element boxed-slice fill
    // exercise exactly that write shape and must stay correct.
    run_ok(
        "hashset-large-clone",
        HASHSET_LARGE_CLONE_SRC,
        "len=1200 sum=719400\nhas0=true has1199=true has1200=false\nboxed sum=1438800\n",
    );
}
