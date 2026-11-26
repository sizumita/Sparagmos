// VecDeque tests.



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