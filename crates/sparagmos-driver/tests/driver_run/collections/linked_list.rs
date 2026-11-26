// LinkedList tests.


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