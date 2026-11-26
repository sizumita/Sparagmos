// iter_mut tests.


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