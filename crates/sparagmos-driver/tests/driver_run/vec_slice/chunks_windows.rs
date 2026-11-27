// Slice chunks and windows tests.

const WINDOWS_DISPLACED_SRC: &str = r#"
pub trait Iter {
    type Item<'a> where Self: 'a;
    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>>;
    fn for_each<F>(mut self, mut f: F)
    where Self: Sized, F: for<'a> FnMut(Self::Item<'a>)
    {
        while let Some(item) = self.next() {
            f(item);
        }
    }
}
pub struct Windows<T> {
    items: Vec<T>,
    start: usize,
    len: usize,
}
impl<T> Windows<T> {
    pub fn new(items: Vec<T>, len: usize) -> Self {
        Self { items, start: 0, len }
    }
}
impl<T> Iter for Windows<T> {
    type Item<'a> = &'a mut [T] where T: 'a;
    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>> {
        let slice = self.items.get_mut(self.start..self.start + self.len)?;
        self.start += 1;
        Some(slice)
    }
}
fn main() {
    Windows::new(vec![1, 2, 3, 4, 5], 3)
        .for_each(|slice| {
            println!("{:?}", slice);
            assert_eq!(slice.len(), 3);
            let mut sum = 0;
            for x in slice.iter() {
                sum += x;
            }
            println!("sum={}", sum);
        });
    println!("ok");
}
"#;

#[test]
fn run_mode_windows_displaced_slice_reads_elements() {
    // A slice formed by displacing a container pointer by a whole
    // number of elements (the `windows` shape: a range index into
    // the middle of a Vec) denotes the elements from that position
    // on: its length, its iterated elements and its Debug rendering
    // all see the displaced window, not just the first element.
    let dir = temp_dir("windows-displaced");
    let src = fixture(&dir, "windows_displaced.rs", WINDOWS_DISPLACED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "displaced window slices must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 2, 3]\nsum=6\n[2, 3, 4]\nsum=9\n[3, 4, 5]\nsum=12\nok\n"
    );
}

const WINDOWS_NESTED_VEC_SRC: &str = r#"
fn main() {
    let v: Vec<Vec<usize>> = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];
    for w in v.windows(2) {
        println!("w {} {} {} {}", w[0][0], w[1][0], w[0].len(), w[1].len());
    }
    let u: Vec<Vec<usize>> = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let w = u.windows(2).next().unwrap();
    println!("c {} {}", w[0][1], w[1][0]);
    let a: Vec<[u8; 2]> = vec![[1, 2], [3, 4], [5, 6]];
    let w = a.windows(2).next().unwrap();
    println!("a {}", w[0][1]);
}
"#;

#[test]
fn run_mode_windows_over_vec_of_vecs_index_reads() {
    // A window the `Windows` step yields is a snapshot over
    // the elements it materialised: indexing the window reads
    // the window's elements themselves. The step used to
    // denominate the snapshot's base as an element position
    // inside its own element tuple, so an index step through
    // the window composed onto that position and landed
    // inside the first element whenever the element was
    // itself a sequence: `w[0]` over a `Vec<Vec<usize>>`
    // handed back the inner vector's first element (and
    // `w[0].len()` died on the integer it found), for every
    // inner-length shape -- the ones whose length equals the
    // window size included. The outputs below are native
    // rustc 1.100.0's.
    run_ok(
        "windows-nested-vec",
        WINDOWS_NESTED_VEC_SRC,
        "w 1 4 3 2\nw 4 6 2 4\nc 2 3\na 2\n",
    );
}

const CHUNKS_NESTED_VEC_SRC: &str = r#"
fn main() {
    let v: Vec<Vec<usize>> = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9], vec![10]];
    for c in v.chunks(2) {
        println!("chunk {} {}", c[0][0], c[1][0]);
    }
    for c in v.rchunks(2) {
        println!("rchunk {} {}", c[0][0], c[1][0]);
    }
}
"#;

#[test]
fn run_mode_chunks_over_vec_of_vecs_index_reads() {
    // The chunk family's yielded sub-slices are snapshots
    // denominated exactly as the `Windows` step's windows
    // are (see `run_mode_windows_over_vec_of_vecs_index_reads`):
    // indexing a chunk of a `Vec<Vec<usize>>` reads the
    // chunk's elements, not the inside of its first element.
    run_ok(
        "chunks-nested-vec",
        CHUNKS_NESTED_VEC_SRC,
        "chunk 1 4\nchunk 6 10\nrchunk 6 10\nrchunk 1 4\n",
    );
}

const CHUNKS_WINDOWS_LEN_SRC: &str = r#"
use std::iter::ExactSizeIterator;
fn main() {
    let v = [1, 2, 3, 4, 5];
    println!("c_rem {}", v.chunks(2).len());
    println!("c_exact {}", v.chunks(5).len());
    println!("c_big {}", v.chunks(9).len());
    println!("c_one {}", v.chunks(1).len());
    println!("w_rem {}", v.windows(3).len());
    println!("w_exact {}", v.windows(5).len());
    println!("w_big {}", v.windows(6).len());
    println!("w_one {}", v.windows(1).len());
    let e: [i32; 0] = [];
    println!("c_empty {}", e.chunks(2).len());
    println!("w_empty {}", e.windows(1).len());
    println!("ce_empty {}", e.chunks_exact(2).len());
    println!("hint_c {:?}", v.chunks(2).size_hint());
    println!("hint_w {:?}", v.windows(3).size_hint());
    println!("hint_ce {:?}", v.chunks_exact(2).size_hint());
    let mut w = v.windows(2);
    let _ = w.next();
    println!("w_after1 {}", w.len());
    let _ = w.next();
    let _ = w.next();
    let _ = w.next();
    println!("w_after4 {}", w.len());
    let mut c = v.chunks(2);
    let rc = &mut c;
    println!("refmut_len {}", rc.len());
    let c2 = v.chunks(3);
    let r2 = &c2;
    println!("ref_len {}", r2.len());
    println!("ce_rem {:?}", v.chunks_exact(2).remainder());
    println!("rce_rem {:?}", v.rchunks_exact(2).remainder());
    let vec = vec![10, 20, 30, 40, 50, 60, 70];
    println!("vec_c {}", vec.chunks(3).len());
    println!("vec_w {}", vec.windows(4).len());
    let z = [(), (), (), ()];
    println!("zst_c {}", z.chunks(3).len());
    println!("zst_w {}", z.windows(2).len());
    let s = "hello".as_bytes();
    println!("bytes_c {}", s.chunks(2).len());
}
"#;

#[test]
fn run_mode_chunks_windows_len() {
    // `ExactSizeIterator::len` for the slice chunk/window
    // family: the trait default body asserts `size_hint`'s
    // upper bound equals its lower bound, and the modelled
    // `size_hint` answered the trait default `(0, None)` for
    // these types, so every `len()` call died in core's
    // `assert_eq!`. The modelled `size_hint` now computes
    // std's exact counts (ceiling division by the chunk size,
    // floor for the exact variants, `remaining - size + 1`
    // for windows), and the outputs below are native
    // rustc 1.100.0's.
    let dir = temp_dir("chunks-windows-len");
    let src = fixture(&dir, "chunks_windows_len.rs", CHUNKS_WINDOWS_LEN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "chunks/windows len must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c_rem 3\nc_exact 1\nc_big 1\nc_one 5\nw_rem 3\nw_exact 1\nw_big 0\nw_one 5\nc_empty 0\nw_empty 0\nce_empty 0\nhint_c (3, Some(3))\nhint_w (3, Some(3))\nhint_ce (2, Some(2))\nw_after1 3\nw_after4 0\nrefmut_len 3\nref_len 2\nce_rem [5]\nrce_rem [1]\nvec_c 3\nvec_w 4\nzst_c 2\nzst_w 3\nbytes_c 3\n"
    );
}

const WINDOWS_COLLECT_NESTED_SRC: &str = r#"
fn main() {
    let v: Vec<Vec<usize>> = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let w: Vec<&[Vec<usize>]> = v.windows(2).collect();
    println!("len {}", w.len());
    println!("w0 {:?}", w[0]);
    println!("w1 {:?}", w[1]);
    println!("first {}", w[0][0][0]);
    println!("last {}", w[1][1][1]);
}
"#;

#[test]
fn run_mode_windows_collect_nested_vec_and_drop() {
    // Collecting `windows` into a `Vec` and dropping the
    // result stopped at an unsupported `Drop` terminator:
    // the glue conversion attached to the consuming
    // `collect` call planned the `Windows` iterator's fields
    // from its native two-field shape -- the second field is
    // a `NonZero<usize>`, whose inner type is an associated
    // type the evidence tables cannot resolve, so its plan
    // stayed a one-field `Untyped` plan and never collapsed
    // -- and then tried to run that plan against the modelled
    // state, a [slice, window size, next index] snapshot with
    // three fields. `NonZero` is glue-free (a primitive
    // integer in a niche wrapper) and the borrowing slice
    // iterators own nothing, so both plans are trivial now.
    // The collected windows also read through the window
    // slices the step machine manufactures: those denoted
    // their snapshot tuple at an element position, which the
    // slice reader mistook for the first element's own
    // elements when that element was itself a sequence of
    // the window's length (here a `Vec` of two numbers), so
    // `w[0]` printed the first inner vector instead of the
    // window. The manufactured slices now denote the
    // snapshot itself.
    run_ok(
        "windows-collect-nested",
        WINDOWS_COLLECT_NESTED_SRC,
        "len 2\nw0 [[1, 2], [3, 4]]\nw1 [[3, 4], [5, 6]]\nfirst 1\nlast 6\n",
    );
}

const WINDOWS_COLLECT_SCALAR_SRC: &str = r#"
fn main() {
    let v = vec![1usize, 2, 3, 4];
    let w: Vec<&[usize]> = v.windows(2).collect();
    println!("len {}", w.len());
    println!("w0 {:?}", w[0]);
    println!("w2 {:?}", w[2]);
    let nested: Vec<Vec<usize>> = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let c: Vec<&[Vec<usize>]> = nested.chunks(2).collect();
    println!("chunks {} {:?} {:?}", c.len(), c[0][0], c[1][0]);
}
"#;

#[test]
fn run_mode_windows_collect_scalar_and_chunks_collect_nested() {
    // The scalar-element shape of the same collect-and-drop
    // path (the `Drop` failure did not depend on the element
    // type), and the chunk family's manufactured slices over
    // nested vectors, which had the same element-position
    // denomination as the window slices.
    run_ok(
        "windows-collect-scalar",
        WINDOWS_COLLECT_SCALAR_SRC,
        "len 3\nw0 [1, 2]\nw2 [3, 4]\nchunks 2 [1, 2] [5, 6]\n",
    );
}

const CHUNKS_MUT_LEN_SRC: &str = r#"
use std::iter::ExactSizeIterator;
fn main() {
    let v = [1, 2, 3, 4, 5];
    println!("ce {}", v.chunks_exact(2).len());
    println!("rce {}", v.rchunks_exact(2).len());
    println!("rc {}", v.rchunks(2).len());
    let mut m = [1, 2, 3, 4, 5];
    println!("cm {}", m.chunks_mut(2).len());
    let mut m2 = [1, 2, 3, 4, 5];
    println!("cem {}", m2.chunks_exact_mut(2).len());
    let mut m3 = [1, 2, 3, 4, 5];
    println!("rcm {}", m3.rchunks_mut(2).len());
    let mut m4 = [1, 2, 3, 4, 5];
    println!("rcem {}", m4.rchunks_exact_mut(2).len());
}
"#;

#[test]
fn run_mode_chunks_mut_and_exact_len() {
    // The mutable and exact variants of the chunk family
    // store the remaining slice differently (a raw fat
    // pointer plus a marker field for the `Mut` shapes), but
    // `len` is the same count family: ceiling division for
    // the plain variants, floor division for the exact ones.
    let dir = temp_dir("chunks-mut-len");
    let src = fixture(&dir, "chunks_mut_len.rs", CHUNKS_MUT_LEN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "chunks mut/exact len must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "ce 2\nrce 2\nrc 3\ncm 3\ncem 2\nrcm 3\nrcem 2\n"
    );
}

const CHUNKS_MUT_WRITE_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3, 4, 5];
    for c in v.chunks_mut(2) {
        for x in c.iter_mut() { *x *= 10; }
    }
    println!("a {:?}", v);
    let mut w = vec![1, 2, 3, 4, 5];
    let mut it = w.chunks_mut(2);
    let first = it.next().unwrap();
    first[0] = 99;
    println!("b {} {}", first[1], it.next().unwrap()[0]);
    let mut u = vec![1, 2, 3, 4, 5, 6];
    for c in u.rchunks_mut(2) {
        c[0] += 100;
    }
    println!("c {:?}", u);
    let mut t = vec![1, 2, 3, 4, 5];
    for c in t.chunks_exact_mut(2) {
        c[1] += 7;
    }
    println!("d {:?}", t);
    let mut v2 = vec![1, 2, 3, 4, 5];
    let mut it2 = v2.chunks_mut(2);
    let last = it2.next_back().unwrap();
    last[0] = 55;
    let head = it2.next().unwrap();
    head[1] = 22;
    println!("nb {:?}", v2);
    let mut w2 = vec![1, 2, 3, 4, 5, 6];
    let mut rit = w2.rchunks_mut(4);
    let tail = rit.next_back().unwrap();
    tail[0] = 11;
    println!("rnb {:?}", w2);
    let mut e = vec![1, 2, 3, 4, 5];
    let mut eit = e.chunks_exact_mut(2);
    eit.next().unwrap()[0] = 10;
    let rem = eit.into_remainder();
    rem[0] = 50;
    println!("rem {:?}", e);
    let mut n = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    for c in n.chunks_mut(2) {
        c[0][0] += 100;
    }
    println!("nest {} {} {}", n[0][0], n[1][0], n[2][0]);
    let mut arr = [1, 2, 3, 4];
    for c in arr.chunks_mut(3) {
        for x in c { *x += 1; }
    }
    println!("arr {:?}", arr);
    let mut c0 = vec![1, 2, 3, 4];
    let parts: Vec<&mut [i32]> = c0.chunks_mut(2).collect();
    println!("collected {}", parts.len());
    let mut s = vec![1, 2, 3, 4, 5];
    let sit = s.chunks_mut(2);
    println!("hint {:?}", sit.size_hint());
    let mut z = vec![(), (), ()];
    let mut count = 0;
    for c in z.chunks_mut(2) { count += c.len(); }
    println!("zst {count}");
}
"#;

#[test]
fn run_mode_chunks_mut_family_writes_alias_source() {
    // The `Mut` chunk family as an iterator: `into_iter`
    // is the identity, the step denominates each yielded
    // chunk as a window over the source storage (so writes
    // through a chunk land in the source, in both step
    // directions), and the trailing marker field of the
    // `Mut` states rides through the step untouched. The
    // outputs below are native rustc 1.100.0's.
    run_ok(
        "chunks-mut-write",
        CHUNKS_MUT_WRITE_SRC,
        "a [10, 20, 30, 40, 50]\nb 2 3\nc [101, 2, 103, 4, 105, 6]\nd [1, 9, 3, 11, 5]\nnb [1, 22, 3, 4, 55]\nrnb [11, 2, 3, 4, 5, 6]\nrem [10, 2, 3, 4, 50]\nnest 101 3 105\narr [2, 3, 4, 5]\ncollected 2\nhint (3, Some(3))\nzst 3\n",
    );
}

const CHUNKS_EXACT_NESTED_VEC_SRC: &str = r#"
fn main() {
    let v: Vec<Vec<usize>> = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9], vec![10]];
    let mut it = v.chunks_exact(3);
    let c = it.next().unwrap();
    println!("c {} {} {} {}", c[0][0], c[1][0], c[2][0], it.remainder().len());
    let mut rit = v.rchunks_exact(3);
    let rc = rit.next().unwrap();
    println!("r {} {} {}", rc[0][0], rc[1][0], rc[2][0]);
    let mut sum = 0;
    for c in v.chunks_exact(2) { sum += c[0][0] + c[1][0]; }
    println!("sum {sum}");
}
"#;

#[test]
fn run_mode_chunks_exact_over_vec_of_vecs_index_reads() {
    // The `Exact` chunk variants' state is built by core's
    // converted body: the stored whole-chunk slice is a
    // `from_raw_parts` fat reference over the container's
    // element pointer, a base value-identical to the one a
    // deref formation produces for a slice over the first
    // element's own elements. The step read the stored slice
    // by the element reading, so a chunk whose length equals
    // the first inner vector's length (3 and 3 here) yielded
    // the inner vector's elements as the chunk's elements and
    // `c[0][0]` died in slice elements. The raw-parts
    // formation now denominates its base as the container
    // window it ranges over (see `raw_parts_base`), and the
    // chunk step's snapshots are denominated by their element
    // tuple. The outputs below are native rustc 1.100.0's.
    run_ok(
        "chunks-exact-nested-vec",
        CHUNKS_EXACT_NESTED_VEC_SRC,
        "c 1 4 6 1\nr 4 6 10\nsum 21\n",
    );
}

const CHUNK_ARRAY_PTR_SRC: &str = r#"
struct Chunks<'a> { v: &'a [i32] }
impl<'a> Iterator for Chunks<'a> {
    type Item = &'a [i32; 3];
    fn next(&mut self) -> Option<Self::Item> {
        if self.v.len() < 3 {
            None
        } else {
            let (fst, snd) = self.v.split_at(3);
            self.v = snd;
            let ptr = fst.as_ptr() as *const [i32; 3];
            Some(unsafe { &*ptr })
        }
    }
}
fn main() {
    let slice: &[i32] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mut chunks = Chunks { v: slice };
    let mut iter = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].iter();
    while let Some(a) = chunks.next() {
        assert_eq!(a, iter.next().unwrap());
    }
    println!("chunk array ptr ok");
}
"#;

#[test]
fn run_mode_chunk_array_pointer_windows() {
    // A sub-slice's `as_ptr` cast to an array pointer
    // denotes the chunk at the sub-slice's offset: the
    // first chunk reads from the allocation's start, the
    // later chunks from their element positions -- neither
    // the whole allocation nor the single element at the
    // position (the rustc suite's
    // `const-generics/type-dependent/issue-70507` shape).
    run_ok("chunk-array-ptr", CHUNK_ARRAY_PTR_SRC, "chunk array ptr ok\n");
}
