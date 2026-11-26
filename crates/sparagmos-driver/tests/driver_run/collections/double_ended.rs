// Double-ended iteration (next_back) tests.


const NEXT_BACK_SLICE_SRC: &str = r#"
fn main() {
    let v = vec![10, 20, 30, 40];
    let mut it = v.iter();
    println!("back={:?}", it.next_back());
    println!("front={:?}", it.next());
    println!("back={:?}", it.next_back());
    println!("back={:?}", it.next_back());
    println!("back={:?}", it.next_back());
}
"#;

#[test]
fn run_mode_next_back_slice_iter() {
    run_ok(
        "next-back-slice",
        NEXT_BACK_SLICE_SRC,
        "back=Some(40)\nfront=Some(10)\nback=Some(30)\nback=Some(20)\nback=None\n",
    );
}

const NEXT_BACK_VEC_INTO_SRC: &str = r#"
fn main() {
    let mut it = vec![1, 2, 3].into_iter();
    println!("back={:?}", it.next_back());
    println!("front={:?}", it.next());
    println!("back={:?}", it.next_back());
    println!("back={:?}", it.next_back());
}
"#;

#[test]
fn run_mode_next_back_vec_into_iter() {
    run_ok(
        "next-back-vec-into",
        NEXT_BACK_VEC_INTO_SRC,
        "back=Some(3)\nfront=Some(1)\nback=Some(2)\nback=None\n",
    );
}

const NEXT_BACK_RANGE_SRC: &str = r#"
fn main() {
    let mut it = 1..5;
    println!("back={:?}", it.next_back());
    println!("front={:?}", it.next());
    println!("back={:?}", it.next_back());
    let mut incl = 1..=3;
    println!("incl back={:?}", incl.next_back());
    println!("incl front={:?}", incl.next());
    println!("incl back={:?}", incl.next_back());
    println!("incl back={:?}", incl.next_back());
}
"#;

#[test]
fn run_mode_next_back_ranges() {
    run_ok(
        "next-back-ranges",
        NEXT_BACK_RANGE_SRC,
        "back=Some(4)\nfront=Some(1)\nback=Some(3)\nincl back=Some(3)\nincl front=Some(1)\nincl back=Some(2)\nincl back=None\n",
    );
}

const NEXT_BACK_ADAPTERS_SRC: &str = r#"
fn main() {
    let v = vec![1, 2, 3];
    let mut copied = v.iter().copied();
    println!("copied back={:?}", copied.next_back());
    let mut mapped = (1..=3).map(|x| x * 10);
    println!("map back={:?}", mapped.next_back());
    println!("map front={:?}", mapped.next());
    let mut enumerated = v.iter().enumerate();
    println!("enum back={:?}", enumerated.next_back());
    println!("enum front={:?}", enumerated.next());
    let mut reversed = (1..=3).rev();
    println!("rev back={:?}", reversed.next_back());
    println!("rev front={:?}", reversed.next());
    let mut once = std::iter::once(42);
    println!("once back={:?}", once.next_back());
    println!("once back={:?}", once.next_back());
}
"#;

#[test]
fn run_mode_next_back_adapters() {
    run_ok(
        "next-back-adapters",
        NEXT_BACK_ADAPTERS_SRC,
        "copied back=Some(3)\nmap back=Some(30)\nmap front=Some(10)\nenum back=Some((2, 3))\nenum front=Some((0, 1))\nrev back=Some(1)\nrev front=Some(3)\nonce back=Some(42)\nonce back=None\n",
    );
}

const NEXT_BACK_OPTION_SRC: &str = r#"
fn main() {
    let mut it = Some(9).into_iter();
    println!("opt back={:?}", it.next_back());
    println!("opt back={:?}", it.next_back());
    let mut front = Some(7).into_iter();
    println!("opt front={:?}", front.next());
    println!("opt back={:?}", front.next_back());
}
"#;

#[test]
fn run_mode_next_back_option_into_iter() {
    run_ok(
        "next-back-option",
        NEXT_BACK_OPTION_SRC,
        "opt back=Some(9)\nopt back=None\nopt front=Some(7)\nopt back=None\n",
    );
}

const NEXT_BACK_USER_SRC: &str = r#"
struct Countdown(u32);
impl Iterator for Countdown {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.0 == 0 { None } else { self.0 -= 1; Some(self.0) }
    }
}
impl DoubleEndedIterator for Countdown {
    fn next_back(&mut self) -> Option<u32> {
        if self.0 == 0 { None } else { let v = self.0; self.0 -= 1; Some(v) }
    }
}
fn main() {
    let mut it = Countdown(3);
    println!("back={:?}", it.next_back());
    println!("front={:?}", it.next());
    println!("back={:?}", it.next_back());
    println!("back={:?}", it.next_back());
}
"#;

#[test]
fn run_mode_next_back_user_defined_iterator() {
    // A user-defined `DoubleEndedIterator` keeps its collected
    // body as the primary execution: the builtin only serves
    // the modelled iterator types.
    run_ok(
        "next-back-user",
        NEXT_BACK_USER_SRC,
        "back=Some(3)\nfront=Some(1)\nback=Some(1)\nback=None\n",
    );
}