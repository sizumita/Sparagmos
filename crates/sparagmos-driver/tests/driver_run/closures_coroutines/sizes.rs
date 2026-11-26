// Size and layout observations over closures, coroutines and packed structs.

const PACKED_DERIVE_SRC: &str = r#"
use std::mem;

#[derive(Copy, Clone)]
struct Aligned(usize);

#[inline(never)]
fn check_align(ptr: *const Aligned) {
    assert_eq!(ptr as usize % mem::align_of::<Aligned>(), 0);
}

impl PartialEq for Aligned {
    fn eq(&self, other: &Self) -> bool {
        check_align(self);
        check_align(other);
        self.0 == other.0
    }
}

#[repr(packed)]
#[derive(Copy, Clone, PartialEq)]
struct Packed(Aligned, Aligned);

#[derive(PartialEq)]
#[repr(C)]
struct Dealigned<T>(u8, T);

fn main() {
    let d1 = Dealigned(0, Packed(Aligned(1), Aligned(2)));
    let ck = d1 == d1;
    assert!(ck);
    println!("eq-ok");
}
"#;

// ----- coroutine/closure layouts (Lane HN, ex-DW) -----

const COROUTINE_CLOSURE_SIZE_SRC: &str = r#"
async fn f(x: u8) -> u8 { x + 1 }
fn main() {
    let fut = f(3);
    println!("fut {}", std::mem::size_of_val(&fut));
    let a = async { 1u8 };
    let b = async { 2u16 };
    println!("blocks {} {}", std::mem::size_of_val(&a), std::mem::size_of_val(&b));
    let arr = [1u64, 2u64];
    let big = async move { arr[0] };
    println!("big {}", std::mem::size_of_val(&big));
    let x = 5u32;
    let c1 = move || x;
    let c2 = move || x as u64;
    let y = 7u64;
    let c3 = move || y;
    println!("closures {} {} {}", std::mem::size_of_val(&c1), std::mem::size_of_val(&c2), std::mem::size_of_val(&c3));
    println!("tysize {}", std::mem::size_of_val(&c1) + std::mem::size_of::<u32>());
    println!("call {}", c1());
}
"#;

#[test]
fn run_mode_size_of_val_coroutine_and_closure() {
    // The values are the compiler's own layouts (verified against a
    // native 1.100.0 build of the same source): the driver records
    // closure and coroutine layouts per monomorphic type, keyed by
    // the definition name plus generic arguments, and the mem
    // builtins answer from that table. `big` (24) differs from the
    // two unit-capture blocks (1, 1), so distinct coroutine types in
    // one function do not share an entry; likewise the u64-capturing
    // closure (8) differs from the two u32-capturing ones (4, 4), so
    // distinct closure types do not share one either.
    let dir = temp_dir("coroutine-closure-size");
    let src = fixture(&dir, "sizes.rs", COROUTINE_CLOSURE_SIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "size_of_val of coroutine/closure values runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "fut 2\nblocks 1 1\nbig 24\nclosures 4 4 8\ntysize 8\ncall 5\n",
        "sizes match the native ones"
    );
}

#[test]
fn run_mode_derived_eq_on_packed_struct_uses_align_of() {
    let dir = temp_dir("packed-derive");
    let src = fixture(&dir, "packed_derive.rs", PACKED_DERIVE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "derived eq over a packed struct runs (align_of::<Aligned> resolves); stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "eq-ok\n");
}
