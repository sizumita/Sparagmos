// BinaryHeap tests.


const BINARY_HEAP_POP_SRC: &str = r#"
use std::collections::BinaryHeap;
fn main() {
    let mut h = BinaryHeap::new();
    h.push(3); h.push(1); h.push(4); h.push(1); h.push(5);
    println!("{:?}", h.pop());
    println!("{:?}", h.pop());
    let mut sum = 0;
    while let Some(x) = h.pop() { sum += x; }
    println!("sum {}", sum);
    println!("{:?}", h.pop());
}
"#;

const BINARY_HEAP_FROM_VEC_SRC: &str = r#"
use std::collections::BinaryHeap;
fn main() {
    let h = BinaryHeap::from(vec![1, 2, 3]);
    println!("len {} peek {:?}", h.len(), h.peek());
    let mut sum = 0;
    for x in &h { sum += x; }
    println!("iter sum {sum}");
    let e: BinaryHeap<i32> = BinaryHeap::from(Vec::new());
    println!("empty {} {:?}", e.len(), e.peek());
}
"#;

#[test]
fn run_mode_binary_heap_pop() {
    // `BinaryHeap::pop` in its collected std body inlines
    // `Vec::pop`, whose length decrement writes the place
    // `(*self).data.len`: the length slot of a `Vec` reached
    // through the owner's field step. The write path's
    // length-slot pre-pass recognised the slot only when the
    // place's prefix crossed references alone, so the write
    // fell through to place resolution, which answers the
    // slot as a detached value, and the run stopped with
    // "expected mutable place, found constant aggregate".
    // The shape judgment now tracks the declared type across
    // field steps as well. The outputs are native rustc
    // 1.100.0's.
    let dir = temp_dir("binary-heap-pop");
    let src = fixture(&dir, "binary_heap_pop.rs", BINARY_HEAP_POP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "binary heap pop must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "Some(5)\nSome(4)\nsum 5\nNone\n"
    );
}

#[test]
fn run_mode_binary_heap_from_vec() {
    // `BinaryHeap::from(vec)`: the conversion moves the
    // vector's elements into the heap's `data` field in
    // heap order (the root is the maximum), the shape the
    // converted std bodies then read.
    let dir = temp_dir("binary-heap-from-vec");
    let src = fixture(&dir, "binaryheapfrom.rs", BINARY_HEAP_FROM_VEC_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "len 3 peek Some(3)\niter sum 6\nempty 0 None\n", "stdout={stdout}");
}