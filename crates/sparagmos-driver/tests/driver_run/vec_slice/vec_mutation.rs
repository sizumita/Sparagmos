// Vec mutation tests (insert, remove, drain, nested Vec).

const VEC_REMOVE_ADT_SRC: &str = r#"
struct E(i32);
impl Drop for E {
    fn drop(&mut self) {
        println!("drop E({})", self.0);
    }
}
fn main() {
    // `Vec::remove` shifts the tail with one `ptr::copy` of ADT
    // elements: every shifted element must land whole, in order.
    let mut v: Vec<E> = (0..6).map(E).collect();
    let x = v.remove(1);
    println!("removed {:?}", x.0);
    std::mem::forget(x);
    println!("v={:?}", v.iter().map(|e| e.0).collect::<Vec<_>>());
    // A shorter tail (a two-element shift) and a two-field
    // element take the same copy.
    let mut w: Vec<E> = (0..4).map(E).collect();
    let y = w.remove(1);
    std::mem::forget(y);
    println!("w={:?}", w.iter().map(|e| e.0).collect::<Vec<_>>());
}
"#;

#[test]
fn run_mode_vec_remove_shifts_adt_elements_whole() {
    let dir = temp_dir("vec-remove-adt");
    let src = fixture(&dir, "vec_remove_adt.rs", VEC_REMOVE_ADT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::remove over ADT elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "removed 1\nv=[0, 2, 3, 4, 5]\nw=[0, 2, 3]\ndrop E(0)\ndrop E(2)\ndrop E(3)\ndrop E(0)\ndrop E(2)\ndrop E(3)\ndrop E(4)\ndrop E(5)\n"
    );
}

const NESTED_VEC_SRC: &str = r#"
fn main() {
    let mut v: Vec<Vec<u32>> = vec![vec![1, 2, 3, 4], vec![5, 6]];
    assert_eq!(v[0][1], 2);
    assert_eq!(v[1][0], 5);
    v[0].reverse();
    v[1].push(7);
    assert_eq!(v[0], vec![4, 3, 2, 1]);
    assert_eq!(v[1][2], 7);
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_nested_vec_index_and_reverse() {
    // A `Vec` value is the aggregate of its elements, so an
    // index step into it is element access and a following index
    // step ranges inside it: composing the two steps into one
    // index over the outer vector (the flat reading a data
    // pointer's position takes) reads the sibling vector whole
    // instead of the element. Nested reads, writes through
    // `v[0].reverse()`'s swaps, and growth of an inner vector
    // must all observe the inner elements.
    let dir = temp_dir("nestedvec");
    let src = fixture(&dir, "nestedvec.rs", NESTED_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested vec index/reverse must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[[4, 3, 2, 1], [5, 6, 7]]\n",
        "the inner vectors reverse and grow independently, exactly as natively"
    );
}

const VEC_POP_AFTER_SHRINK_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::with_capacity(16);
    v.push(1);
    v.push(2);
    v.shrink_to_fit();
    println!("shrunk len={}", v.len());
    match v.pop() {
        Some(x) => println!("popped {}", x),
        None => println!("none"),
    }
    println!("after len={}", v.len());
    match v.pop() {
        Some(x) => println!("popped {}", x),
        None => println!("none"),
    }
    println!("final len={} empty={}", v.len(), v.is_empty());
}
"#;

#[test]
fn run_mode_vec_pop_after_shrink() {
    // `pop`'s collected body reads the length slot, writes the
    // decremented length back, reads the capacity slot for its
    // `assert_unchecked(len < capacity)`, and reads the element
    // at the new length through the buffer pointer: the length
    // write must not destroy the element it is about to read,
    // and the capacity answer must stay the storage extent
    // across the write (see the vec length/storage rule in
    // `vec_with_len`).
    run_ok(
        "vec-pop-after-shrink",
        VEC_POP_AFTER_SHRINK_SRC,
        "shrunk len=2\npopped 2\nafter len=1\npopped 1\nfinal len=0 empty=true\n",
    );
}

const VEC_APPEND_TAIL_SRC: &str = r#"
fn main() {
    let mut a = vec![1, 2];
    let mut b = vec![3, 4, 5];
    a.append(&mut b);
    println!("a={:?} len={} cap={}", a, a.len(), a.capacity());
    println!("b={:?} len={}", b, b.len());
    println!("a[0]={} a[3]={} a[4]={}", a[0], a[3], a[4]);
}
"#;

#[test]
fn run_mode_vec_append_tail() {
    // `append`'s collected body copies the source elements
    // through the destination's buffer pointer at the old
    // length and only then writes the length slot: the raw
    // writes extend the storage without moving the declared
    // length, and the final length write lands on the extended
    // storage.
    run_ok(
        "vec-append-tail",
        VEC_APPEND_TAIL_SRC,
        "a=[1, 2, 3, 4, 5] len=5 cap=5\nb=[] len=0\na[0]=1 a[3]=4 a[4]=5\n",
    );
}

const VEC_SPLIT_OFF_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3, 4, 5, 6];
    v.truncate(3);
    println!("trunc v={:?} len={}", v, v.len());
    let mut w = vec![1, 2, 3, 4, 5];
    let tail = w.split_off(2);
    println!("w={:?} tail={:?}", w, tail);
}
"#;

#[test]
fn run_mode_vec_split_off() {
    // `split_off` builds its result in the real aggregate form
    // inside its body (an inlined `with_capacity_in`, `set_len`,
    // and a raw copy into the buffer allocation): the value
    // translates to the element form as it returns to the
    // caller (see `vec_from_real_form`).
    run_ok(
        "vec-split-off",
        VEC_SPLIT_OFF_SRC,
        "trunc v=[1, 2, 3] len=3\nw=[1, 2] tail=[3, 4, 5]\n",
    );
}

const VEC_DRAIN_COLLECT_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3, 4, 5];
    let taken: Vec<i32> = v.drain(1..3).collect();
    println!("taken={:?}", taken);
    let mut u = vec![10, 20, 30];
    let all: Vec<i32> = u.drain(..).collect();
    println!("all={:?}", all);
}
"#;

#[test]
fn run_mode_vec_drain_collect() {
    // A drain's iteration steps its slice iterator through the
    // source vector's storage (see `drain_next_state`); the
    // collected elements are the drained range.
    run_ok(
        "vec-drain-collect",
        VEC_DRAIN_COLLECT_SRC,
        "taken=[2, 3]\nall=[10, 20, 30]\n",
    );
}

const VEC_INSERT_GROW_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3];
    v.insert(3, 4);
    println!("{:?} len={} cap={}", v, v.len(), v.capacity());
    v.insert(1, 99);
    println!("{:?}", v);
    v.insert(0, 0);
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_vec_insert_positions_with_growth() {
    // `Vec::insert` runs its collected body: the capacity
    // check and the regrowth walk the buffer wrapper chain
    // (`RawVec` / `RawVecInner`) through references handed
    // across calls, and the chain's bookkeeping slots are the
    // vector's own storage extent and declared length in this
    // model. End, middle and start inserts over a growing
    // vector land exactly as natively, capacity included.
    let dir = temp_dir("vecinsertgrow");
    let src = fixture(&dir, "vecinsertgrow.rs", VEC_INSERT_GROW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::insert with growth must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 2, 3, 4] len=4 cap=6\n[1, 99, 2, 3, 4]\n[0, 1, 99, 2, 3, 4]\n"
    );
}

const VEC_INSERT_RESERVED_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::with_capacity(16);
    v.push(10);
    v.push(20);
    v.push(30);
    v.insert(1, 15);
    println!("{:?} cap={}", v, v.capacity());
    assert_eq!(v, vec![10, 15, 20, 30]);
    let mut s = vec![String::from("a"), String::from("c")];
    s.insert(1, String::from("b"));
    assert_eq!(s, vec![String::from("a"), String::from("b"), String::from("c")]);
    println!("ok");
}
"#;

#[test]
fn run_mode_vec_insert_within_capacity_and_equality() {
    // An insert inside the reserved capacity does not grow:
    // `with_capacity` materialises the requested storage
    // extent, so the capacity stays 16 and the elements shift
    // in place. Whole-vector equality after a mutation
    // compares the logical element sequence: the sparse form
    // a growth or a shrink leaves behind (declared length
    // below the stored elements) equals the plain aggregate
    // form of the same elements.
    let dir = temp_dir("vecinsertreserved");
    let src = fixture(&dir, "vecinsertreserved.rs", VEC_INSERT_RESERVED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::insert within capacity must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[10, 15, 20, 30] cap=16\nok\n"
    );
}

const VEC_INSERT_NESTED_SRC: &str = r#"
fn main() {
    let mut v = vec![vec![1, 2], vec![4]];
    v.insert(1, vec![3]);
    println!("{:?}", v);
    let mut t = (vec![1, 3], 7);
    t.0.insert(1, 2);
    println!("{:?}", t);
}
"#;

#[test]
fn run_mode_vec_insert_nested_and_tuple_field() {
    // The inserted element may itself be a vector, and the
    // mutated vector may live inside another aggregate: the
    // chain re-entry judges the place the buffer pointer
    // denotes by the value it currently holds, not by the
    // declared type of the local that happens to contain it.
    let dir = temp_dir("vecinsertnested");
    let src = fixture(&dir, "vecinsertnested.rs", VEC_INSERT_NESTED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::insert on nested/tuple-field vectors must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[[1, 2], [3], [4]]\n([1, 2, 3], 7)\n"
    );
}

const VEC_INSERT_SHRINK_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::with_capacity(16);
    v.push(1);
    v.push(2);
    v.insert(1, 9);
    println!("{:?} cap={}", v, v.capacity());
    v.shrink_to_fit();
    println!("{:?} cap={}", v, v.capacity());
    v.insert(3, 5);
    println!("{:?} cap={}", v, v.capacity());
}
"#;

#[test]
fn run_mode_vec_insert_after_shrink_to_fit() {
    // After a shrink the storage extent is the length itself,
    // so the next insert grows again from there: the capacity
    // slot writes of both paths (the shrink's truncation and
    // the regrowth's amortised capacity) rebuild the vector's
    // storage extent exactly.
    let dir = temp_dir("vecinsertshrink");
    let src = fixture(&dir, "vecinsertshrink.rs", VEC_INSERT_SHRINK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::insert after shrink_to_fit must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 9, 2] cap=16\n[1, 9, 2] cap=3\n[1, 9, 2, 5] cap=6\n"
    );
}

const BOX_VEC_GROW_SRC: &str = r#"
struct D(&'static str);
impl Drop for D {
    fn drop(&mut self) {
        println!("drop {}", self.0);
    }
}
fn main() {
    let mut b: Box<Vec<D>> = Box::new(Vec::with_capacity(2));
    b.push(D("a"));
    b.push(D("b"));
    b.insert(1, D("x"));
    b.reserve(16);
    for d in b.iter() {
        println!("elem {}", d.0);
    }
    drop(b);
    println!("done");
}
"#;

#[test]
fn run_mode_box_vec_insert_and_reserve_with_growth() {
    // Growing a `Vec` that is the whole content of a `Box`
    // allocation through the collected `insert` / `reserve`
    // bodies: the vector's buffer pointer is derived from the
    // vector place inside the box's allocation, and the
    // reallocation must not free or move the box's storage --
    // the elements and their drop order land exactly as
    // natively and the box stays live throughout.
    let dir = temp_dir("boxvecgrow");
    let src = fixture(&dir, "boxvecgrow.rs", BOX_VEC_GROW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<Vec> growth must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "elem a\nelem x\nelem b\ndrop a\ndrop x\ndrop b\ndone\n"
    );
}

const BOX_BOX_VEC_GROW_SRC: &str = r#"
fn main() {
    let mut b: Box<Box<Vec<i32>>> = Box::new(Box::new(Vec::with_capacity(2)));
    b.push(1);
    b.push(2);
    b.push(3);
    b.insert(1, 9);
    println!("{:?}", b);
}
"#;

#[test]
fn run_mode_box_box_vec_insert_with_growth() {
    // The same growth through two owning wrappers: the inner
    // vector's buffer pointer denotes a position inside the
    // inner box's stored value, never a wrapper allocation's
    // base.
    let dir = temp_dir("boxboxvecgrow");
    let src = fixture(&dir, "boxboxvecgrow.rs", BOX_BOX_VEC_GROW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<Box<Vec>> growth must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "[1, 9, 2, 3]\n");
}

const NESTED_VEC_POP_SRC: &str = r#"
fn main() {
    let mut v = vec![vec![1, 2], vec![4]];
    let x = v[0].pop();
    println!("{:?} {:?}", v, x);
}
"#;

#[test]
fn run_mode_nested_vec_pop_after_shrink() {
    // A collected method (`pop`) on a vector element reached
    // through an index projection: `pop` shrinks the element
    // in place, so the element takes the sparse form, and the
    // body's own reads of the element's elements (through the
    // buffer pointer the method forms) must range inside the
    // element, not compose into positions of the outer vector.
    run_ok(
        "nested-vec-pop",
        NESTED_VEC_POP_SRC,
        "[[1], [4]] Some(2)\n",
    );
}

const NESTED_VEC_INSERT_SRC: &str = r#"
fn main() {
    let mut v = vec![vec![1, 2], vec![4]];
    v.insert(1, vec![3]);
    println!("{:?}", v);
    v[0].insert(1, 9);
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_nested_vec_insert_after_outer_insert() {
    // After the outer vector's own collected `insert` leaves
    // it in the sparse form, a collected `insert` on one of its
    // elements forms its mutable receiver through `IndexMut`
    // over the outer place: the sparse form is the same element
    // aggregate as the plain form and is addressable the same
    // way.
    run_ok(
        "nested-vec-insert",
        NESTED_VEC_INSERT_SRC,
        "[[1, 2], [3], [4]]\n[[1, 9, 2], [3], [4]]\n",
    );
}

const NESTED_VEC_SET_LEN_READS_SRC: &str = r#"
fn main() {
    let mut v = vec![vec![1, 2], vec![4]];
    unsafe { v[0].set_len(1); }
    println!("len={}", v[0].len());
    println!("e={}", v[0][0]);
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_nested_vec_reads_after_set_len() {
    // `set_len` rewrites the element's declared length in
    // place, leaving the sparse form behind; element reads
    // through the index projection then denote the element's
    // own elements.
    run_ok(
        "nested-vec-set-len",
        NESTED_VEC_SET_LEN_READS_SRC,
        "len=1\ne=1\n[[1], [4]]\n",
    );
}

const NESTED_VEC_PUSH_TRUNC_SRC: &str = r#"
fn main() {
    let mut w = vec![vec![vec![7, 8]], vec![vec![3]]];
    w.truncate(1);
    w[0][0].push(10);
    println!("{:?}", w);
}
"#;

#[test]
fn run_mode_nested_vec_push_after_outer_truncate() {
    // With the outer vector in the sparse form (after
    // `truncate`), a method on a doubly-indexed element forms
    // its receiver through the outer element position: the
    // steps nest into the element rather than composing into
    // the outer vector's positions.
    run_ok(
        "nested-vec-push-trunc",
        NESTED_VEC_PUSH_TRUNC_SRC,
        "[[[7, 8, 10]]]\n",
    );
}
