use super::{fixture, run_driver, run_ok, temp_dir};

// Regression coverage for the String mutator audit: every
// `&mut self` mutator must work on a string in whichever
// storage form it is held -- the abstract text form and the
// real aggregate (Vec) layout a string keeps when it was
// built by a byte-level producer (`from_utf8_unchecked`, a
// `Vec<u8>` transmute, `chars().collect()`, `+`). Before the
// fix, the converted std bodies of insert / insert_str /
// remove / truncate / pop / clear / retain / replace_range /
// split_off walked the aggregate layout (or, for the abstract
// form, dereferenced the receiver into a constant aggregate)
// and failed; `String + &str` inlined `push_str` into `add`'s
// body and failed the same way; `String::drain` had neither
// an iterator step nor a drop plan.

const PRODUCERS_SRC: &str = r#"
fn exercise(mut s: String) -> String {
    s.insert(1, 'X');
    s.insert_str(0, ">");
    let c = s.remove(2);
    assert_eq!(c, 'X');
    s.truncate(4);
    s.push('p');
    let p = s.pop();
    assert_eq!(p, Some('p'));
    s.replace_range(1..3, "ipp");
    let t = s.split_off(3);
    format!("{}|{}", s, t)
}
fn main() {
    println!("{}", exercise(String::from("hello")));
    println!("{}", exercise(unsafe {
        String::from_utf8_unchecked(vec![104, 101, 108, 108, 111])
    }));
    println!("{}", exercise(unsafe {
        std::mem::transmute::<Vec<u8>, String>(vec![104, 101, 108, 108, 111])
    }));
    println!("{}", exercise("hello".chars().collect()));
    println!("{}", exercise(String::from("hel") + "lo"));
    let rhs = String::from("lo");
    println!("{}", String::from("hel") + &rhs);
    let mut s = String::from("hello");
    s.clear();
    println!("[{}] {}", s, s.len());
}
"#;

#[test]
fn run_mode_string_mutators_on_every_storage_form() {
    run_ok(
        "strmutforms",
        PRODUCERS_SRC,
        ">ip|pl\n>ip|pl\n>ip|pl\n>ip|pl\n>ip|pl\nhello\n[] 0\n",
    );
}

const RETAIN_SRC: &str = r#"
fn main() {
    let mut a = String::from("hello");
    a.retain(|c| c != 'l');
    println!("{}", a);
    let mut kept = String::new();
    let mut b = String::from("hello");
    b.retain(|c| {
        if c != 'l' {
            kept.push(c);
        }
        c != 'l'
    });
    println!("{} {}", b, kept);
    // A predicate panic before any rejection leaves the
    // string untouched (std's fast phase only reads).
    let mut c1 = String::from("hello");
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        c1.retain(|c| {
            if c == 'e' {
                panic!("boom");
            }
            true
        });
    }));
    println!("{:?} [{}]", r.is_err(), c1);
    // A predicate panic after a rejection commits the
    // retained prefix (std's PanicGuard write-back).
    let mut c2 = String::from("hello");
    let r2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        c2.retain(|c| {
            if c == 'o' {
                panic!("boom");
            }
            c != 'l'
        });
    }));
    println!("{:?} [{}]", r2.is_err(), c2);
}
"#;

#[test]
fn run_mode_string_retain_phases_and_panics() {
    run_ok(
        "strmutretain",
        RETAIN_SRC,
        "heo\nheo heo\ntrue [hello]\ntrue [he]\n",
    );
}

const RANGE_AND_DRAIN_SRC: &str = r#"
fn main() {
    let mut s = String::from("hello");
    s.replace_range(1..3, "Y");
    println!("{}", s);
    let mut s = String::from("hello");
    s.replace_range(3.., "Y");
    println!("{}", s);
    let mut s = String::from("hello");
    s.replace_range(..2, "Y");
    println!("{}", s);
    let mut s = String::from("hello");
    s.replace_range(1..=2, "Y");
    println!("{}", s);
    let mut s = String::from("hello");
    s.replace_range(.., "Y");
    println!("{}", s);
    let mut s = String::from("hello");
    let t: String = s.drain(1..3).collect();
    println!("{} [{}]", t, s);
    let mut s = String::from("hello");
    {
        let mut d = s.drain(1..4);
        println!("{:?}", d.next());
    }
    println!("[{}]", s);
    let mut s = String::from("hello");
    {
        let mut d = s.drain(1..3);
        println!("{:?}", d.next_back());
    }
    println!("[{}]", s);
    let mut s = String::from("hello");
    {
        let _d = s.drain(1..3);
    }
    println!("[{}]", s);
    let mut s = unsafe { String::from_utf8_unchecked(vec![104, 101, 108, 108, 111]) };
    let t: String = s.drain(1..3).collect();
    println!("{} [{}]", t, s);
    let mut s = String::from("hello");
    s.reserve(10);
    s.reserve_exact(20);
    println!(
        "{:?} {:?}",
        s.try_reserve(5).is_ok(),
        s.try_reserve_exact(5).is_ok()
    );
    println!("{}", s);
}
"#;

#[test]
fn run_mode_string_replace_range_forms_and_drain() {
    run_ok(
        "strmutdrain",
        RANGE_AND_DRAIN_SRC,
        "hYlo\nhelY\nYllo\nhYlo\nY\nel [hlo]\nSome('e')\n[ho]\nSome('l')\n[hlo]\n[hlo]\nel [hlo]\ntrue true\nhello\n",
    );
}

fn run_panic(tag: &str, src: &str, message: &str) {
    let dir = temp_dir(tag);
    let src_path = fixture(&dir, "case.rs", src);
    let out = run_driver(&src_path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "case {tag} must end in a guest panic; stderr={stderr}"
    );
    assert!(
        stderr.contains(message),
        "case {tag} panic must read {message:?}; stderr={stderr}"
    );
}

#[test]
fn run_mode_string_mutator_boundary_panics() {
    run_panic(
        "strmutpanicinsert",
        "fn main() { let mut s = String::from(\"h\u{e9}llo\"); s.insert(2, 'X'); }",
        "assertion failed: self.is_char_boundary(idx)",
    );
    run_panic(
        "strmutpanicremove",
        "fn main() { let mut s = String::from(\"h\u{e9}llo\"); s.remove(2); }",
        "start byte index 2 is not a char boundary; it is inside '\u{e9}' (bytes 1..3 of string)",
    );
    run_panic(
        "strmutpanicremoveend",
        "fn main() { let mut s = String::from(\"h\u{e9}llo\"); s.remove(6); }",
        "cannot remove a char from the end of a string",
    );
    run_panic(
        "strmutpanicrange",
        "fn main() { let mut s = String::from(\"h\u{e9}llo\"); s.replace_range(3..99, \"Y\"); }",
        "range end index 99 out of range for slice of length 6",
    );
    run_panic(
        "strmutpanicsplit",
        "fn main() { let mut s = String::from(\"h\u{e9}llo\"); s.split_off(2); }",
        "assertion failed: self.is_char_boundary(at)",
    );
}

// Regression coverage for the byte-view mutators: `String::
// as_mut_vec`, `String::leak` and `str::as_bytes_mut`. The
// abstract string has no byte slots, so each of these calls
// rewrites the denoted storage into the byte element form
// and the vector / byte operations mutate that form in
// place; the string readers (text coercions, formatting,
// equality, ordering) read the form back as the string it
// denotes. Before the fix, `Vec::push` through the view
// stopped on the string value, `leak` stopped in conversion
// on its body's pointer transmute, and a byte write through
// `as_bytes_mut` stopped on the string storage.

const AS_MUT_VEC_SRC: &str = r#"
fn main() {
    let mut s = String::from("hello");
    {
        let v = unsafe { s.as_mut_vec() };
        v.push(b'!');
        v[0] = b'H';
        v.extend_from_slice(b"?");
        let p = v.pop();
        assert_eq!(p, Some(b'?'));
        assert_eq!(v.len(), 6);
    }
    assert_eq!(s, "Hello!");
    assert!(s == String::from("Hello!"));
    assert!(s > String::from("Hello"));
    println!("{} {:?} {}", s, s, s.len());
    let mut e = String::new();
    unsafe { e.as_mut_vec() }.extend_from_slice(b"abc");
    println!("{}", e);
    let mut c: String = "hello".chars().collect();
    { let v = unsafe { c.as_mut_vec() }; v[4] = b'O'; }
    let mut u = unsafe { String::from_utf8_unchecked(vec![104, 105]) };
    unsafe { u.as_mut_vec() }.push(b'!');
    println!("{} {}", c, u);
    struct Holder { s: String }
    let mut h = Holder { s: String::from("ab") };
    unsafe { h.s.as_mut_vec() }.push(b'c');
    let mut vs = vec![String::from("x")];
    unsafe { vs[0].as_mut_vec() }.push(b'y');
    println!("{} {}", h.s, vs[0]);
    let mut t = String::from("hello");
    {
        let v = unsafe { t.as_mut_vec() };
        v.truncate(2);
        v.insert(1, b'E');
        let r = v.remove(0);
        assert_eq!(r, b'h');
        let sum: u32 = v.iter().map(|b| *b as u32).sum();
        assert_eq!(sum, 69 + 101);
        v.clear();
        assert!(v.is_empty());
    }
    assert_eq!(t, "");
    s.push_str("?");
    println!("[{}]", s);
}
"#;

#[test]
fn run_mode_string_as_mut_vec_writes_through() {
    run_ok(
        "strmutvec",
        AS_MUT_VEC_SRC,
        "Hello! \"Hello!\" 6\nabc\nhellO hi!\nabc xy\n[Hello!?]\n",
    );
}

const LEAK_AND_BYTES_MUT_SRC: &str = r#"
fn leak_it(s: String) -> &'static mut str { s.leak() }
fn main() {
    let m = leak_it(String::from("rust"));
    assert_eq!(m, "rust");
    m.make_ascii_uppercase();
    { let b = unsafe { m.as_bytes_mut() }; b[0] = b'R'; }
    assert_eq!(&*m, "RUST");
    println!("{:?} {}", m, m.len());
    let m2 = leak_it("ab".chars().collect());
    println!("{}", m2);
    let mut s2 = String::from("cat");
    let r: &mut str = &mut s2;
    { let b = unsafe { r.as_bytes_mut() }; b[0] = b'b'; }
    println!("{}", s2);
    let mut s3 = String::from("hello");
    { let b = unsafe { s3.as_bytes_mut() }; b[0] = b'H'; }
    println!("{}", s3);
}
"#;

#[test]
fn run_mode_string_leak_and_as_bytes_mut() {
    run_ok(
        "strleak",
        LEAK_AND_BYTES_MUT_SRC,
        "\"RUST\" 4\nab\nbat\nHello\n",
    );
}
