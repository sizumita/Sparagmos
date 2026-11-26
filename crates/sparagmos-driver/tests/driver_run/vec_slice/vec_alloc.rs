// Vec allocation, capacity and resize tests.

// ----- Vec allocation paths (Lane BO) -----

const VEC_DROPPABLE_SRC: &str = r#"
struct D(i32);
impl Drop for D {
    fn drop(&mut self) {
        println!("drop {}", self.0);
    }
}
fn main() {
    let v = vec![D(3), D(4)];
    println!("len {}", v.len());
    drop(v);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_macro_of_droppables() {
    // The `vec![...]` list form builds the vector through
    // `Box::new_uninit` + `box_assume_init_into_vec_unsafe`, whose
    // real bodies bottom out in the runtime allocator
    // (`Global::alloc_impl_runtime` has no body). The interpreter
    // models those entry points, so a Vec of droppable elements
    // runs end to end and the element destructors run in order,
    // byte for byte as natively.
    let dir = temp_dir("vecdroppable");
    let src = fixture(&dir, "vecdroppable.rs", VEC_DROPPABLE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec! of droppables must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 2\ndrop 3\ndrop 4\ndone\n",
        "vec! elements drop in order, exactly as natively"
    );
}

const VEC_ZST_ENUM_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) {} }
enum CS<Y, R> { Yielded(Y), Complete(R) }
enum Never {}
fn read_it(v: Vec<CS<Never, DropMe>>) {
    match &v[0] { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
    match &v[1] { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
    println!("done");
}
fn main() {
    read_it(vec![CS::Complete(DropMe), CS::Complete(DropMe)]);
}
"#;

#[test]
fn run_mode_vec_of_zero_sized_enum_elements_read_by_index() {
    // A `Vec` of a zero-sized enum (one variant uninhabited, the
    // other's payload zero-sized) built by `vec!`: MIR elides the
    // element store entirely, so the elements reach execution as
    // uninitialized slots, and an index read of an element denotes
    // the element type's single value (the canonical-value rescue
    // follows the projection's index step). Both elements read
    // back as the `Complete` variant, exactly as natively.
    let dir = temp_dir("veczstenum");
    let src = fixture(&dir, "veczstenum.rs", VEC_ZST_ENUM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec! of zero-sized enum elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\nc\ndone\n",
        "zero-sized enum elements read by index denote their single value"
    );
}

const VEC_ZST_ENUM_GENERIC_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) {} }
enum CS<Y, R> { Yielded(Y), Complete(R) }
trait Tr { type Out; }
struct S;
impl Tr for S { type Out = DropMe; }
enum Never {}
fn read_it<T: Tr>(v: Vec<CS<Never, T::Out>>) {
    match &v[0] { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
    println!("done");
}
fn main() {
    read_it::<S>(vec![CS::Complete(DropMe)]);
}
"#;

#[test]
fn run_mode_vec_of_zero_sized_enum_elements_generic_projection() {
    // The same element read with the payload type arriving
    // through a trait projection (`T::Out`): the element type is
    // still single-valued after instantiation, so the index read
    // is rescued the same way.
    let dir = temp_dir("veczstenumgeneric");
    let src = fixture(&dir, "veczstenumgeneric.rs", VEC_ZST_ENUM_GENERIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "generic vec! of zero-sized enum elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\ndone\n",
        "projected zero-sized enum elements read by index denote their single value"
    );
}

const VEC_GROWTH_SRC: &str = r#"
struct D(i32);
impl Drop for D {
    fn drop(&mut self) {
        println!("drop {}", self.0);
    }
}
fn main() {
    let mut v: Vec<D> = Vec::with_capacity(4);
    v.reserve(8);
    v.push(D(1));
    v.push(D(2));
    println!("len {}", v.len());
}
"#;

#[test]
fn run_mode_vec_with_capacity_push_and_reserve() {
    // `with_capacity` and `reserve` are allocation entry points:
    // the abstract Vec model does not represent capacity, so both
    // are modelled (an empty vector / a no-op) and pushes onto the
    // reserved vector behave as natively, including the scope-end
    // drop of the elements.
    let dir = temp_dir("vecgrowth");
    let src = fixture(&dir, "vecgrowth.rs", VEC_GROWTH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "with_capacity/reserve/push must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 2\ndrop 1\ndrop 2\n",
        "reserved Vec pushes and drops exactly as natively"
    );
}

const VEC_RESIZE_GROW_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3];
    v.resize(6, 9);
    println!("{:?} len={} cap={}", v, v.len(), v.capacity());
    let mut w: Vec<i32> = Vec::new();
    w.resize(4, 7);
    println!("{:?} len={} cap={}", w, w.len(), w.capacity());
}
"#;

#[test]
fn run_mode_vec_resize_grows_with_fill() {
    // `Vec::resize` growth runs through `extend_with`, whose
    // `SetLenOnDrop` guard holds a `&mut` to the vector's length
    // slot and writes the filled length back through it on drop:
    // the write must land on the vector itself (it used to fail
    // with "expected mutable place, found constant aggregate"
    // because the length-slot reference denoted a detached value).
    let dir = temp_dir("vecresizegrow");
    let src = fixture(&dir, "vecresizegrow.rs", VEC_RESIZE_GROW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec resize growth must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 2, 3, 9, 9, 9] len=6 cap=6\n[7, 7, 7, 7] len=4 cap=4\n",
        "grown vectors hold the fill values with native len/cap"
    );
}

const VEC_RESIZE_REGROW_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3, 4, 5];
    v.resize(2, 0);
    println!("{:?} len={} cap={}", v, v.len(), v.capacity());
    v.resize(3, 42);
    println!("{:?} len={} cap={}", v, v.len(), v.capacity());
}
"#;

#[test]
fn run_mode_vec_resize_regrows_within_capacity() {
    // Shrinking then growing back inside the reserved capacity
    // takes the same length-slot write path with no reservation
    // in between: the regrown slot is filled, the length and the
    // (unchanged) capacity answer natively.
    let dir = temp_dir("vecresizeregrow");
    let src = fixture(&dir, "vecresizeregrow.rs", VEC_RESIZE_REGROW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "shrink-then-regrow must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 2] len=2 cap=5\n[1, 2, 42] len=3 cap=5\n",
        "regrowth within capacity fills and reports natively"
    );
}

const VEC_RESIZE_PAYLOAD_SRC: &str = r#"
#[derive(Debug, Clone)]
struct P {
    a: i32,
    b: i32,
}
fn main() {
    let mut s = vec!["a".to_string()];
    s.resize(4, "x".to_string());
    println!("{:?} len={}", s, s.len());
    let mut p = vec![P { a: 1, b: 2 }];
    p.resize(3, P { a: 7, b: 8 });
    println!("{:?} len={}", p, p.len());
}
"#;

#[test]
fn run_mode_vec_resize_grows_clone_payloads() {
    // The fill value is cloned per new slot: heap (`String`) and
    // aggregate (struct) payloads grow through the same path and
    // read back exactly.
    let dir = temp_dir("vecresizepayload");
    let src = fixture(&dir, "vecresizepayload.rs", VEC_RESIZE_PAYLOAD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "resize of payload vectors must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[\"a\", \"x\", \"x\", \"x\"] len=4\n[P { a: 1, b: 2 }, P { a: 7, b: 8 }, P { a: 7, b: 8 }] len=3\n",
        "cloned fill payloads read back exactly"
    );
}

const VEC_RESIZE_SPARE_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::with_capacity(100);
    v.push(1);
    v.push(2);
    v.resize(50, 7);
    println!("len={} cap={} sum={}", v.len(), v.capacity(), v.iter().sum::<i32>());
    v.resize(10, 0);
    println!("len={} cap={}", v.len(), v.capacity());
    v.resize(60, 3);
    println!("len={} cap={} sum={}", v.len(), v.capacity(), v.iter().sum::<i32>());
}
"#;

#[test]
fn run_mode_vec_resize_uses_spare_capacity() {
    // Growth into a large spare capacity, a shrink, and a
    // regrowth past the previous length all stay inside the
    // original allocation: capacity never moves and the sums
    // cover exactly the declared elements.
    let dir = temp_dir("vecresizespare");
    let src = fixture(&dir, "vecresizespare.rs", VEC_RESIZE_SPARE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "resize within spare capacity must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len=50 cap=100 sum=339\nlen=10 cap=100\nlen=60 cap=100 sum=209\n",
        "spare-capacity growth/shrink/regrowth answers natively"
    );
}

const VEC_HUGE_CAPACITY_SRC: &str = r#"
fn main() {
    let n: usize = (1 << 22) + 5;
    let v: Vec<std::mem::MaybeUninit<u8>> = Vec::with_capacity(n);
    println!("len {} cap {}", v.len(), v.capacity());
    println!("done");
}
"#;

#[test]
fn run_mode_vec_with_capacity_past_materialisation_limit() {
    // A `with_capacity` request past the materialisation limit
    // stores no elements, but the requested capacity is the
    // vector's storage and stays in the value's account: the
    // reported capacity is the requested one, as natively —
    // not the zero stored elements.
    let dir = temp_dir("vechugecapacity");
    let src = fixture(&dir, "vechugecapacity.rs", VEC_HUGE_CAPACITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "huge with_capacity must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 0 cap 4194309\ndone\n",
        "the requested capacity past the limit is reported as natively"
    );
}

const VEC_HUGE_SET_LEN_POP_SRC: &str = r#"
fn main() {
    let n: usize = (1 << 22) + 5;
    let mut v: Vec<std::mem::MaybeUninit<u8>> = Vec::with_capacity(n);
    unsafe {
        v.set_len(n);
        match v.pop() {
            Some(b) => {
                std::mem::forget(b);
                println!("popped len {}", v.len());
            }
            None => println!("none"),
        }
    }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_huge_set_len_pop() {
    // `Vec::pop` decrements the length and then asserts the
    // new length is below the capacity. On a vector grown by
    // `set_len` to the requested capacity, the accounted
    // capacity is the full request, so the assertion holds as
    // natively; a capacity collapsed onto the declared length
    // would make the decremented length equal it and trip the
    // assertion.
    let dir = temp_dir("vechugesetlenpop");
    let src = fixture(&dir, "vechugesetlenpop.rs", VEC_HUGE_SET_LEN_POP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "pop on a huge set_len-grown vector must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "popped len 4194308\ndone\n",
        "pop returns the tail slot and the decremented length, as natively"
    );
}

const STRING_WITH_CAPACITY_SRC: &str = r#"
fn main() {
    let mut s = String::with_capacity(16);
    s.push_str("hi");
    println!("{} {}", s, s.len());
}
"#;

#[test]
fn run_mode_string_with_capacity_push_str_and_len() {
    // `String::with_capacity` is a direct `RawVecInner` caller:
    // its collected body inlines down to
    // `RawVecInner::with_capacity_in`, which has no converted
    // body, so the constructor is modelled as the empty string
    // (the requested capacity is not represented, as for
    // `Vec::with_capacity`).
    let dir = temp_dir("stringwithcapacity");
    let src = fixture(&dir, "stringwithcapacity.rs", STRING_WITH_CAPACITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "String::with_capacity must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hi 2\n");
}

const VEC_TRY_RESERVE_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3];
    println!("small {}", v.try_reserve(10).is_ok());
    println!("huge {}", v.try_reserve(usize::MAX).is_ok());
    v.push(4);
    println!("len {}", v.len());
}
"#;

#[test]
fn run_mode_vec_try_reserve_verdicts() {
    // Allocation never fails in the abstract model, so the only
    // error `try_reserve` can produce is capacity overflow:
    // 3 + 10 elements fit easily (Ok), while `usize::MAX` more
    // elements of 4 bytes exceed the `isize::MAX`-byte limit
    // (Err), exactly as natively.
    let dir = temp_dir("vectryreserve");
    let src = fixture(&dir, "vectryreserve.rs", VEC_TRY_RESERVE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::try_reserve must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "small true\nhuge false\nlen 4\n"
    );
}

const VEC_SHRINK_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3];
    v.shrink_to_fit();
    v.shrink_to(2);
    println!("len {}", v.len());
    v.push(4);
    println!("len {}", v.len());
}
"#;

#[test]
fn run_mode_vec_shrink_to_fit_and_shrink_to() {
    // Shrinking moves only the capacity, which the abstract Vec
    // model does not represent: length, elements and later
    // pushes are exactly as natively.
    let dir = temp_dir("vecshrink");
    let src = fixture(&dir, "vecshrink.rs", VEC_SHRINK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::shrink_to_fit/shrink_to must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len 3\nlen 4\n");
}
