use super::{ANON_CONST_GENERIC_SRC, Command, TAIL_CALL_SRC, fixture, run_driver, run_driver_extra, run_ok, run_ok_edition, temp_dir};


// ----- by-reference constants (Lane R) ---------------------------------------

const PROMOTED_REF_SRC: &str = r#"
fn main() {
    let x = 10;
    let y = x;
    assert_eq!(y, 10);
    let z: u8 = 255;
    assert_eq!(z, 255);
}
"#;

const CONST_SHAPES_SRC: &str = r#"
struct S { a: i32, b: u8 }
const S1: S = S { a: -2, b: 7 };
const ARR: [i32; 3] = [10, 20, 30];
const R: &i32 = &ARR[1];
static A: usize = 42;
static B: usize = A;
const NAMES: [&str; 2] = ["x", "yy"];
const OPT: Option<&i32> = Some(&5);
fn main() {
    println!("{} {} {} {} {} {}", S1.a, S1.b, *R, B, NAMES[1],
             match OPT { Some(r) => *r, None => 0 });
}
"#;

#[test]
fn run_mode_promoted_ref_constant() {
    // `assert_eq!` against a literal builds a promoted `&i32`
    // constant; it must execute, not stop at the constant.
    let dir = temp_dir("promoted");
    let src = fixture(&dir, "promoted.rs", PROMOTED_REF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "promoted reference constants must run; stderr={stderr}"
    );
}

/// Build a guest source holding a 5,000-element byte table
/// (past the driver's element-wise constant bound of 4,096):
/// `decl` declares the table (with `ELEMS` as the element list)
/// and a `read_at` accessor over it; `main` sums the table and
/// prints the sum plus two spot reads.
fn large_byte_table_src(decl: &str) -> String {
    let elems: Vec<String> = (0..5000).map(|i| (i % 251).to_string()).collect();
    let mut src = decl.replace("ELEMS", &elems.join(", "));
    src.push_str("fn main() { let mut sum: u32 = 0;");
    src.push_str(" for i in 0..5000 { sum += read_at(i) as u32; }");
    src.push_str(" println!(\"{} {} {}\", sum, read_at(0), read_at(4999)); }");
    src
}

#[test]
fn run_mode_large_byte_table_static() {
    // A static byte table past the element-wise constant bound
    // materialises as the raw bytes and reads back exactly.
    let src = large_byte_table_src(
        "static TABLE: [u8; 5000] = [ELEMS]; fn read_at(i: usize) -> u8 { TABLE[i] }",
    );
    let dir = temp_dir("large-byte-table");
    let src = fixture(&dir, "large_byte_table.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "large byte table must run; stderr={stderr}"
    );
    let expect_sum: u32 = (0..5000).map(|i| i % 251).sum();
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("{expect_sum} 0 {}\n", 4999 % 251),
        "decoded table values must match the source program"
    );
}

#[test]
fn run_mode_large_byte_table_wrapped_static() {
    // The same table inside an alignment-wrapper newtype -- the
    // shape of the `unicode_ident` XID tables, whose reference
    // constants reached the converter as `ByRef` identities
    // before byte arrays past the bound materialised.
    let src = large_byte_table_src(
        "#[repr(C, align(64))] struct Align64<T>(T); static LEAF: Align64<[u8; 5000]> = Align64([ELEMS]); fn read_at(i: usize) -> u8 { LEAF.0[i] }",
    );
    let dir = temp_dir("large-byte-table-wrapped");
    let src = fixture(&dir, "large_byte_table_wrapped.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "wrapped large byte table must run; stderr={stderr}"
    );
    let expect_sum: u32 = (0..5000).map(|i| i % 251).sum();
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("{expect_sum} 0 {}\n", 4999 % 251),
        "decoded table values must match the source program"
    );
}

/// Build a guest source holding a 5,000-element `u32` table
/// (past the driver's element-wise constant bound of 4,096):
/// `decl` declares the table (with `ELEMS` as the element list)
/// and `main` is the caller's entry point.
fn large_u32_table_src(decl: &str, main: &str) -> String {
    let elems: Vec<String> = (0..5000)
        .map(|i| ((i * 3 + 1) % 1009).to_string())
        .collect();
    let mut src = decl.replace("ELEMS", &elems.join(", "));
    src.push_str(main);
    src
}

#[test]
fn run_mode_large_u32_table_static() {
    // A static `u32` table past the element-wise constant
    // bound: its elements are present in the allocation, so it
    // decodes element by element and reads back exactly, both
    // by value and through a reference. Before the presence
    // check, the constant kept its `ByRef` identity and the
    // interpreter rejected the whole body.
    let src = large_u32_table_src(
        "static TABLE: [u32; 5000] = [ELEMS];",
        "fn main() { let mut sum: u64 = 0; for i in 0..5000 { sum += TABLE[i] as u64; } let r: &[u32; 5000] = &TABLE; println!(\"{} {} {}\", sum, r[1234], TABLE[4999]); }",
    );
    let dir = temp_dir("large-u32-table");
    let src = fixture(&dir, "large_u32_table.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "large u32 table must run; stderr={stderr}"
    );
    let expect_sum: u64 = (0..5000).map(|i| ((i * 3 + 1) % 1009) as u64).sum();
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!(
            "{} {} {}\n",
            expect_sum,
            (1234 * 3 + 1) % 1009,
            (4999 * 3 + 1) % 1009
        ),
        "decoded table values must match the source program"
    );
}

#[test]
fn run_mode_large_u32_table_const() {
    // The same table as a `const` item: the constant operand's
    // allocation decodes by the same presence rule.
    let src = large_u32_table_src(
        "const TABLE: [u32; 5000] = [ELEMS];",
        "fn main() { println!(\"{} {}\", TABLE[42], TABLE[4999]); }",
    );
    let dir = temp_dir("large-u32-table-const");
    let src = fixture(&dir, "large_u32_table_const.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "large u32 const table must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("{} {}\n", (42 * 3 + 1) % 1009, (4999 * 3 + 1) % 1009),
        "decoded table values must match the source program"
    );
}

#[test]
fn run_mode_nested_zst_table_stays_undecoded() {
    // A nonzero-stride element type can contain zero-sized
    // arrays whose declared lengths multiply without occupying
    // allocation bytes: this static is 5,000 bytes of storage
    // whose element-wise expansion would produce tens of
    // billions of values. The shared expansion budget keeps
    // the constant undecoded, so the run stops at it -- the
    // same outcome as before the presence check existed --
    // instead of exhausting host memory in the decode.
    let dir = temp_dir("nested-zst-table");
    let src = fixture(
        &dir,
        "nested_zst_table.rs",
        "static TABLE: [([[(); 4096]; 4096], bool); 5000] = [([[(); 4096]; 4096], true); 5000]; fn main() { println!(\"{} {}\", TABLE.len(), TABLE[0].1); }",
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "nested zero-sized expansion must stay undecoded; stderr={stderr}"
    );
}

#[test]
fn run_mode_nested_zst_elements_under_budget() {
    // The same nesting shape at a size whose total expansion
    // fits the shared budget still materialises element by
    // element and reads back exactly.
    let dir = temp_dir("nested-zst-under-budget");
    let src = fixture(
        &dir,
        "nested_zst_under_budget.rs",
        "static TABLE: [([[(); 10]; 10], u32); 5000] = [([[(); 10]; 10], 7); 5000]; fn main() { println!(\"{} {} {}\", TABLE.len(), TABLE[0].1, TABLE[4999].1); }",
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "moderate nested expansion must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "5000 7 7\n",
        "decoded table values must match the source program"
    );
}

#[test]
fn run_mode_shared_byte_block_refs_stay_undecoded() {
    // An array of references to one shared byte block: each
    // element's decode copies the block's bytes, so counting
    // decode calls alone would materialise gigabytes here
    // while spending only ~500,000 budget units. Byte-copying
    // arms charge the copied length against the shared budget,
    // so the constant stays undecoded and the run stops at it.
    let dir = temp_dir("shared-byte-block-refs");
    let src = fixture(
        &dir,
        "shared_byte_block_refs.rs",
        "static TABLE: [&[u8; 4096]; 500000] = [&[7; 4096]; 500000]; fn main() { println!(\"{}\", TABLE[0][0]); }",
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "repeated copies of one byte block must stay undecoded; stderr={stderr}"
    );
}

#[test]
fn run_mode_shared_byte_block_refs_under_budget() {
    // The same shape at a count whose total copied bytes fit
    // the shared budget still materialises and reads back
    // exactly.
    let dir = temp_dir("shared-byte-block-refs-under-budget");
    let src = fixture(
        &dir,
        "shared_byte_block_refs_under_budget.rs",
        "static TABLE: [&[u8; 4096]; 100] = [&[7; 4096]; 100]; fn main() { println!(\"{} {} {}\", TABLE.len(), TABLE[0][0], TABLE[99][4095]); }",
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "moderate shared byte block refs must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "100 7 7\n",
        "decoded table values must match the source program"
    );
}

#[test]
fn run_mode_over_budget_fat_pointer_prefix_stays_undecoded() {
    // A fat-pointer constant whose recorded length is a prefix
    // of a larger backing allocation, at a size the shared
    // expansion budget cannot pay for: the constant must stay
    // undecoded. Reading the whole backing allocation instead
    // (the plain-pointee shortcut) would silently replace the
    // recorded length 1,048,577 with the allocation's 1,048,578.
    let lit = "a".repeat(1048578);
    let src = format!(
        "const RAW: &[u8; 1048578] = b\"{lit}\"; const fn prefix(arr: &'static [u8; 1048578]) -> &'static [u8] {{ unsafe {{ std::slice::from_raw_parts(arr.as_ptr(), 1048577) }} }} const S: &[u8] = prefix(RAW); fn main() {{ println!(\"{{}}\", S.len()); }}"
    );
    let dir = temp_dir("over-budget-fat-prefix");
    let src = fixture(&dir, "over_budget_fat_prefix.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an over-budget fat-pointer prefix must stay undecoded; stderr={stderr}"
    );
}

#[test]
fn run_mode_over_budget_str_prefix_stays_undecoded() {
    // The `&str` form of the same shape: the recorded length
    // is a prefix of the backing allocation and over the
    // budget, so the constant stays undecoded rather than
    // materialising with the allocation's length.
    let lit = "a".repeat(1048578);
    let src = format!(
        "const RAW: &[u8; 1048578] = b\"{lit}\"; const fn prefix_str(arr: &'static [u8; 1048578]) -> &'static str {{ unsafe {{ std::str::from_utf8_unchecked(std::slice::from_raw_parts(arr.as_ptr(), 1048577)) }} }} const T: &str = prefix_str(RAW); fn main() {{ println!(\"{{}}\", T.len()); }}"
    );
    let dir = temp_dir("over-budget-str-prefix");
    let src = fixture(&dir, "over_budget_str_prefix.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an over-budget str prefix must stay undecoded; stderr={stderr}"
    );
}

#[test]
fn run_mode_fat_pointer_prefix_under_budget_reads_back() {
    // The same prefix shape at a size the budget covers still
    // materialises with its recorded length, not the backing
    // allocation's.
    let lit = "a".repeat(600000);
    let src = format!(
        "const RAW: &[u8; 600000] = b\"{lit}\"; const fn prefix(arr: &'static [u8; 600000]) -> &'static [u8] {{ unsafe {{ std::slice::from_raw_parts(arr.as_ptr(), 500000) }} }} const S: &[u8] = prefix(RAW); fn main() {{ println!(\"{{}} {{}}\", S.len(), S[499999]); }}"
    );
    let dir = temp_dir("fat-prefix-under-budget");
    let src = fixture(&dir, "fat_prefix_under_budget.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an under-budget fat-pointer prefix must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "500000 97\n",
        "the recorded prefix length must be preserved"
    );
}

#[test]
fn run_mode_shared_tuple_byte_array_over_budget_stays_undecoded() {
    // A table of references to one shared aggregate holding a
    // 1 MiB byte array: each element's decode reaches the inline
    // byte-array copy in `decode_const` through the tuple, so
    // the copies are charged against the shared expansion
    // budget and the table stays undecoded instead of
    // materialising one 1 MiB copy per element.
    let src = "static TABLE: [&([u8; 1048576], bool); 4] = [&([7; 1048576], true); 4]; fn main() { println!(\"{} {}\", TABLE[0].0[0], TABLE[3].1); }";
    let dir = temp_dir("shared-tuple-byte-array-over-budget");
    let src = fixture(&dir, "shared_tuple_byte_array_over_budget.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an over-budget shared byte-array table must stay undecoded; stderr={stderr}"
    );
}

#[test]
fn run_mode_shared_tuple_byte_array_under_budget_reads_back() {
    // The same tuple-wrapped shape at a size the budget covers
    // still materialises: an over-element-limit byte array
    // reached through a shared reference table decodes as bytes.
    let src = "static TABLE: [&([u8; 5000], bool); 3] = [&([9; 5000], true); 3]; fn main() { println!(\"{} {}\", TABLE[2].0[4999], TABLE[0].1); }";
    let dir = temp_dir("shared-tuple-byte-array-under-budget");
    let src = fixture(&dir, "shared_tuple_byte_array_under_budget.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an under-budget shared byte-array table must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "9 true\n",
        "decoded table values must match the source program"
    );
}

#[test]
fn run_mode_str_newtype_table_over_budget_stays_undecoded() {
    // A table of references to one shared newtype over `str`:
    // each element's decode copies the trailing `str` field's
    // bytes, so the copies are charged against the shared
    // expansion budget and the table stays undecoded.
    let lit = "a".repeat(1048576);
    let src = format!(
        "struct MyStr(str); static S: &str = \"{lit}\"; static BIG: &MyStr = unsafe {{ std::mem::transmute::<&str, &MyStr>(S) }}; static TABLE: [&MyStr; 5000] = [BIG; 5000]; fn main() {{ println!(\"{{}} {{}}\", TABLE.len(), TABLE[0].0.len()); }}"
    );
    let dir = temp_dir("str-newtype-table-over-budget");
    let src = fixture(&dir, "str_newtype_table_over_budget.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an over-budget shared str-newtype table must stay undecoded; stderr={stderr}"
    );
}

#[test]
fn run_mode_bytes_newtype_table_over_budget_stays_undecoded() {
    // A table of references to one shared newtype over `[u8]`:
    // each element's decode copies the trailing slice field's
    // bytes, so the copies are charged against the shared
    // expansion budget and the table stays undecoded.
    let src = "struct MyBytes([u8]); static SLICE: &[u8] = &[7u8; 1048576]; static BIG: &MyBytes = unsafe { std::mem::transmute::<&[u8], &MyBytes>(SLICE) }; static TABLE: [&MyBytes; 5000] = [BIG; 5000]; fn main() { println!(\"{} {}\", TABLE.len(), TABLE[0].0.len()); }";
    let dir = temp_dir("bytes-newtype-table-over-budget");
    let src = fixture(&dir, "bytes_newtype_table_over_budget.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an over-budget shared bytes-newtype table must stay undecoded; stderr={stderr}"
    );
}

#[test]
fn run_mode_several_large_literals_each_decode() {
    // Several string literals in one body, each comfortably
    // under the expansion budget on its own: every root
    // constant starts with a full budget, so all of them
    // decode -- the third must not be rejected for what the
    // first two spent.
    let a = "a".repeat(400000);
    let b = "a".repeat(400001);
    let c = "a".repeat(400002);
    let src = format!(
        "fn main() {{ let s0 = \"{a}\"; let s1 = \"{b}\"; let s2 = \"{c}\"; println!(\"{{}} {{}} {{}}\", s0.len(), s1.len(), s2.len()); }}"
    );
    let dir = temp_dir("several-large-literals");
    let src = fixture(&dir, "several_large_literals.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "each under-budget literal must decode; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "400000 400001 400002\n",
        "decoded literal lengths must match the source program"
    );
}

#[test]
fn run_mode_erased_pointer_pair_over_budget_stays_undecoded() {
    // Two copies of one type-erased pointer (a `*const ()`
    // denoting a 600,000-byte allocation) in one constant:
    // each erased decode charges the allocation's bytes
    // against the shared expansion budget, so the second
    // copy does not fit. The constant must stay undecoded:
    // falling back to the declared type's decode for the
    // second copy would give it a reference to a unit value,
    // so the two identical pointers would compare unequal
    // (the run printed `eq=false` before this fix) and the
    // second pointer's backing bytes would be lost.
    let src = "static BIG: [u8; 600000] = [7; 600000]; struct Pair(*const (), *const ()); unsafe impl Sync for Pair {} static PAIR: Pair = Pair(BIG.as_ptr() as *const (), BIG.as_ptr() as *const ()); fn main() { println!(\"eq={}\", PAIR.0 == PAIR.1); println!(\"b={}\", unsafe { *(PAIR.1 as *const u8) }); }";
    let dir = temp_dir("erased-pointer-pair-over-budget");
    let src = fixture(&dir, "erased_pointer_pair_over_budget.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an over-budget erased pointer pair must stay undecoded; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "",
        "no wrong value may be printed for an undecoded constant"
    );
}

#[test]
fn run_mode_erased_pointer_pair_under_budget_reads_back() {
    // The same erased-pointer pair at a size whose two
    // charges fit the shared budget: both copies decode as
    // erased blocks, compare equal, and read back exactly.
    let src = "static BIG: [u8; 400000] = [7; 400000]; struct Pair(*const (), *const ()); unsafe impl Sync for Pair {} static PAIR: Pair = Pair(BIG.as_ptr() as *const (), BIG.as_ptr() as *const ()); fn main() { println!(\"eq={}\", PAIR.0 == PAIR.1); println!(\"b={}\", unsafe { *(PAIR.1 as *const u8) }); }";
    let dir = temp_dir("erased-pointer-pair-under-budget");
    let src = fixture(&dir, "erased_pointer_pair_under_budget.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an under-budget erased pointer pair must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "eq=true\nb=7\n",
        "decoded erased pointers must match the source program"
    );
}

#[test]
fn run_mode_static_ref_pair_over_budget_stays_undecoded() {
    // Two `*mut u32` pointers to one field of a large static
    // in one constant: building the static-reference seed for
    // each pointer charges the whole 600,000-byte static
    // against the shared expansion budget, so the second seed
    // does not fit. The constant must stay undecoded: falling
    // back to a snapshot of the pointee for the second pointer
    // would detach it from the static's storage, so a write
    // through the first pointer was not observed through the
    // second (the run printed `v=1` before this fix; the
    // native program prints `v=99`).
    let src = "static mut S: ([u8; 600000], u32) = ([7; 600000], 1); struct Pair(*mut u32, *mut u32); unsafe impl Sync for Pair {} static PAIR: Pair = Pair(unsafe { core::ptr::addr_of_mut!(S.1) }, unsafe { core::ptr::addr_of_mut!(S.1) }); fn main() { unsafe { *PAIR.0 = 99; println!(\"v={}\", *PAIR.1); } }";
    let dir = temp_dir("static-ref-pair-over-budget");
    let src = fixture(&dir, "static_ref_pair_over_budget.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an over-budget static-ref pair must stay undecoded; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "",
        "no wrong value may be printed for an undecoded constant"
    );
}

#[test]
fn run_mode_static_ref_pair_under_budget_shares_storage() {
    // The same static-field pointer pair at a size whose two
    // seed charges fit the shared budget: both pointers decode
    // as static references to the one storage, and a write
    // through the first is observed through the second,
    // exactly as in the native program.
    let src = "static mut S: ([u8; 400000], u32) = ([7; 400000], 1); struct Pair(*mut u32, *mut u32); unsafe impl Sync for Pair {} static PAIR: Pair = Pair(unsafe { core::ptr::addr_of_mut!(S.1) }, unsafe { core::ptr::addr_of_mut!(S.1) }); fn main() { unsafe { *PAIR.0 = 99; println!(\"v={}\", *PAIR.1); } }";
    let dir = temp_dir("static-ref-pair-under-budget");
    let src = fixture(&dir, "static_ref_pair_under_budget.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an under-budget static-ref pair must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "v=99\n",
        "both pointers must denote the static's shared storage"
    );
}

#[test]
fn run_mode_uninit_scan_table_over_budget_stays_undecoded() {
    // A 300-element table of references to one struct whose
    // 16 MiB `MaybeUninit` payload is uninitialised and whose
    // `bool` field is set: decoding each element's static seed
    // scans the payload in full twice (once at the struct,
    // whose scan runs to the one initialised field, and once
    // at the payload itself), while the expansion budget
    // spends only a few units per element. The scan budget
    // (1 GiB of examinations per root constant) runs out
    // after about thirty elements; the decode then loses the
    // `Uninit` shortcut, the payload's structural decode
    // charges its 16 MiB against the expansion budget, and
    // the whole table stays undecoded -- in seconds, where
    // the uncharged scan needed about 0.2 s per element
    // (a 100,000-element table scanned terabytes in
    // aggregate) and the table decoded.
    let src = "#[repr(C)] struct S { payload: std::mem::MaybeUninit<[u8; 16777216]>, flag: bool } static S_VAL: S = S { payload: std::mem::MaybeUninit::uninit(), flag: true }; static TABLE: [&S; 300] = [&S_VAL; 300]; fn main() { println!(\"len={} flag={}\", TABLE.len(), TABLE[0].flag); }";
    let dir = temp_dir("uninit-scan-table-over-budget");
    let src = fixture(&dir, "uninit_scan_table_over_budget.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a reference table whose repeated uninit scans exceed the scan budget must stay undecoded; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "",
        "no wrong value may be printed for an undecoded constant"
    );
}

#[test]
fn run_mode_uninit_scan_table_under_budget_reads_back() {
    // The same table at four elements: the repeated scans
    // total about 128 MiB of examinations, well within the
    // scan budget, so every element's seed decodes exactly as
    // before and the run reads the flag back through the
    // references.
    let src = "#[repr(C)] struct S { payload: std::mem::MaybeUninit<[u8; 16777216]>, flag: bool } static S_VAL: S = S { payload: std::mem::MaybeUninit::uninit(), flag: true }; static TABLE: [&S; 4] = [&S_VAL; 4]; fn main() { println!(\"len={} flag={}\", TABLE.len(), TABLE[3].flag); }";
    let dir = temp_dir("uninit-scan-table-under-budget");
    let src = fixture(&dir, "uninit_scan_table_under_budget.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an under-budget uninit-scan table must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len=4 flag=true\n",
        "decoded table values must match the source program"
    );
}

#[test]
fn run_mode_static_ref_table_scan_exhaustion_stays_undecoded() {
    // A 70,000-element table of pointers to the `flag` field
    // of one static whose other two fields are 4 KiB
    // uninitialised payloads. Building each element's static
    // seed scans about 16 KiB (the struct's scan runs through
    // both payloads to the one initialised field, and each
    // payload is scanned again on its own), so the scan
    // budget (1 GiB of examinations per root constant) runs
    // out after about 65,000 elements, while the expansion
    // budget spends only a few units per element (the
    // payloads decode through the `Uninit` shortcut). Past
    // exhaustion the payload's structural decode fails on an
    // uninitialised byte -- an ordinary failure that set no
    // budget flag -- and the static-reference fallback then
    // substituted a snapshot of the flag for the remaining
    // elements: those pointers no longer denoted the static's
    // storage, and a write through the first was not observed
    // through the last (the native program prints
    // `last=true`). With scan exhaustion recorded alongside
    // expansion exhaustion, the constant stays undecoded.
    // Two payloads halve the element count at which
    // exhaustion bites, keeping this test near ten seconds;
    // the single-payload shape at 140,000 entries behaves
    // identically.
    let src = "#[repr(C)] struct S { a: std::mem::MaybeUninit<[u8; 4096]>, b: std::mem::MaybeUninit<[u8; 4096]>, flag: std::cell::Cell<bool> } unsafe impl Sync for S {} static S_VAL: S = S { a: std::mem::MaybeUninit::uninit(), b: std::mem::MaybeUninit::uninit(), flag: std::cell::Cell::new(false) }; struct Table([*const std::cell::Cell<bool>; 70000]); unsafe impl Sync for Table {} static TABLE: Table = Table([core::ptr::addr_of!(S_VAL.flag); 70000]); fn main() { unsafe { (*TABLE.0[0]).set(true); println!(\"last={}\", (*TABLE.0[69999]).get()); } }";
    let dir = temp_dir("static-ref-table-scan-exhaustion");
    let src = fixture(&dir, "static_ref_table_scan_exhaustion.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a static-ref table whose seed scans exhaust the scan budget must stay undecoded; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "",
        "no wrong value may be printed for an undecoded constant"
    );
}

#[test]
fn run_mode_static_ref_table_scan_within_budget_shares_storage() {
    // The same flag-pointer table at sixteen elements: the
    // repeated seed scans total about 256 KiB of
    // examinations, well within the scan budget, so every
    // element decodes as a static reference to the one
    // storage, and a write through the first is observed
    // through the last, exactly as in the native program.
    let src = "#[repr(C)] struct S { a: std::mem::MaybeUninit<[u8; 4096]>, b: std::mem::MaybeUninit<[u8; 4096]>, flag: std::cell::Cell<bool> } unsafe impl Sync for S {} static S_VAL: S = S { a: std::mem::MaybeUninit::uninit(), b: std::mem::MaybeUninit::uninit(), flag: std::cell::Cell::new(false) }; struct Table([*const std::cell::Cell<bool>; 16]); unsafe impl Sync for Table {} static TABLE: Table = Table([core::ptr::addr_of!(S_VAL.flag); 16]); fn main() { unsafe { (*TABLE.0[0]).set(true); println!(\"first={} last={}\", (*TABLE.0[0]).get(), (*TABLE.0[15]).get()); } }";
    let dir = temp_dir("static-ref-table-scan-within-budget");
    let src = fixture(&dir, "static_ref_table_scan_within_budget.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an under-budget static-ref table must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "first=true last=true\n",
        "both pointers must denote the static's shared storage"
    );
}

#[test]
fn run_mode_static_ref_table_large_uninit_payload_stays_bounded() {
    // A 2,000-element table of pointers to the `flag` field of
    // one static whose other two fields are 4 KiB uninitialised
    // payloads. Every element's `StaticRef` constant used to
    // carry its own converted copy of the static's initializer
    // seed, the payloads expanded to their declared skeleton:
    // megabytes of retained constants per element, and a
    // 2,000-element table peaked over 21 GB once elements were
    // accessed (about 9 GB when only the length was read), so
    // such records landed in the runner's `memory limit`
    // bucket instead of being judged. The seed now lives once
    // in the program's seed table, so the run stays near the
    // driver's baseline footprint -- and the elements still
    // denote the one storage: a write through the first is
    // observed through the last, exactly as in the native
    // program. The RSS bound is asserted on Linux only, from
    // the child's VmHWM; elsewhere the run's output is the
    // check.
    let src = "#[repr(C)] struct S { a: std::mem::MaybeUninit<[u8; 4096]>, b: std::mem::MaybeUninit<[u8; 4096]>, flag: std::cell::Cell<bool> } unsafe impl Sync for S {} static S_VAL: S = S { a: std::mem::MaybeUninit::uninit(), b: std::mem::MaybeUninit::uninit(), flag: std::cell::Cell::new(false) }; struct Table([*const std::cell::Cell<bool>; 2000]); unsafe impl Sync for Table {} static TABLE: Table = Table([core::ptr::addr_of!(S_VAL.flag); 2000]); fn main() { unsafe { (*TABLE.0[0]).set(true); println!(\"first={} last={}\", (*TABLE.0[0]).get(), (*TABLE.0[1999]).get()); } }";
    let dir = temp_dir("static-ref-table-large-uninit-payload");
    let src_path = fixture(&dir, "static_ref_table_large_uninit_payload.rs", src);
    // Spawn with output redirected to files so the child's
    // peak RSS can be polled from /proc while it runs.
    let out_path = dir.join("stdout.txt");
    let err_path = dir.join("stderr.txt");
    let mut child = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src_path)
        .stdout(std::fs::File::create(&out_path).unwrap())
        .stderr(std::fs::File::create(&err_path).unwrap())
        .spawn()
        .expect("failed to run sparagmos-driver");
    let pid = child.id();
    let mut peak_kib: u64 = 0;
    let status = loop {
        #[cfg(target_os = "linux")]
        if let Ok(proc_status) = std::fs::read_to_string(format!("/proc/{pid}/status")) {
            for line in proc_status.lines() {
                if let Some(rest) = line.strip_prefix("VmHWM:") {
                    if let Ok(kib) = rest.trim().trim_end_matches(" kB").trim().parse::<u64>() {
                        peak_kib = peak_kib.max(kib);
                    }
                }
            }
        }
        match child.try_wait().expect("failed to poll sparagmos-driver") {
            Some(status) => break status,
            None => std::thread::sleep(std::time::Duration::from_millis(20)),
        }
    };
    let stdout = std::fs::read_to_string(&out_path).unwrap();
    let stderr = std::fs::read_to_string(&err_path).unwrap();
    assert_eq!(
        status.code(),
        Some(0),
        "a large static-ref table over an uninit payload must run; stderr={stderr}"
    );
    assert_eq!(
        stdout, "first=true last=true\n",
        "both pointers must denote the static's shared storage"
    );
    #[cfg(target_os = "linux")]
    assert!(
        peak_kib < 1024 * 1024,
        "peak RSS {peak_kib} KiB must stay bounded: the static's seed \
         is recorded once in the program, not once per table element"
    );
}

#[test]
fn run_mode_slice_table_across_large_statics_stays_within_cache_cap() {
    // One root constant -- a 21-element table of one-byte
    // slices -- whose elements alternate between ten distinct
    // 128 MiB statics and two 100 MiB promoted arrays, in an
    // order that keeps evicting the promoted allocations from
    // the backing-allocation caches. The two caches (promoted
    // allocations, static initialisers) share one byte counter
    // capped at 256 MiB; resetting that counter while dropping
    // only the promoted-allocation cache left every cached
    // static initialiser retained but unaccounted, and the
    // driver's peak RSS grew to about 5.7 GiB. With both
    // caches dropped together the peak stays near 3.4 GiB
    // (the compiler itself holds one copy of each backing
    // allocation). The decoded values must stay exact either
    // way. The RSS bound is asserted on Linux only, from the
    // child's VmHWM; elsewhere the run's output is the check.
    let mut src = String::new();
    for i in 1..=10 {
        src.push_str(&format!(
            "static S{i}: [u8; 134217728] = [{}; 134217728]; ",
            10 + i
        ));
    }
    src.push_str("const BIG1: [u8; 104857600] = [21; 104857600]; ");
    src.push_str("const BIG2: [u8; 104857600] = [22; 104857600]; ");
    let sl = |name: &str| format!("unsafe {{ std::slice::from_raw_parts({name}.as_ptr(), 1) }}");
    let mut elems = vec![sl("S1"), sl("BIG1"), sl("BIG2")];
    let mut m = 1;
    for i in 2..=10 {
        elems.push(sl(&format!("S{i}")));
        elems.push(sl(&format!("BIG{m}")));
        m = 3 - m;
    }
    src.push_str(&format!("static TABLE: [&[u8]; 21] = [{}]; ", elems.join(", ")));
    src.push_str("fn main() { println!(\"{} {} {} {} {} {} {}\", TABLE[0][0], TABLE[3][0], TABLE[1][0], TABLE[2][0], TABLE[19][0], TABLE[20][0], TABLE.len()); }");
    let dir = temp_dir("slice-table-cache-cap");
    let src_path = fixture(&dir, "slice_table_cache_cap.rs", &src);
    // Spawn with output redirected to files so the child's
    // peak RSS can be polled from /proc while it runs.
    let out_path = dir.join("stdout.txt");
    let err_path = dir.join("stderr.txt");
    let mut child = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src_path)
        .stdout(std::fs::File::create(&out_path).unwrap())
        .stderr(std::fs::File::create(&err_path).unwrap())
        .spawn()
        .expect("failed to run sparagmos-driver");
    let pid = child.id();
    let mut peak_kib: u64 = 0;
    let status = loop {
        #[cfg(target_os = "linux")]
        if let Ok(proc_status) = std::fs::read_to_string(format!("/proc/{pid}/status")) {
            for line in proc_status.lines() {
                if let Some(rest) = line.strip_prefix("VmHWM:") {
                    if let Ok(kib) = rest.trim().trim_end_matches(" kB").trim().parse::<u64>() {
                        peak_kib = peak_kib.max(kib);
                    }
                }
            }
        }
        match child.try_wait().expect("failed to poll sparagmos-driver") {
            Some(status) => break status,
            None => std::thread::sleep(std::time::Duration::from_millis(20)),
        }
    };
    let stdout = std::fs::read_to_string(&out_path).unwrap();
    let stderr = std::fs::read_to_string(&err_path).unwrap();
    assert_eq!(
        status.code(),
        Some(0),
        "the slice table across large statics must run; stderr={stderr}"
    );
    assert_eq!(
        stdout, "11 12 21 22 20 21 21\n",
        "decoded slices must match the source program"
    );
    #[cfg(target_os = "linux")]
    assert!(
        peak_kib < 4_500 * 1024,
        "peak RSS {peak_kib} KiB must stay bounded by the shared cache cap: \
         cached static initialisers must not outlive their accounting"
    );
}

#[test]
fn run_mode_alternating_large_statics_table_over_work_budget_stays_undecoded() {
    // A 40-element table of one-byte slices alternating between
    // two 129 MiB statics. Together the two backings exceed the
    // 256 MiB allocation-cache cap, so every element evicts the
    // other static's initialiser and `eval_initializer`
    // re-materialises it in full: the retained-bytes cap bounds
    // what the caches hold, not the cumulative work, and the
    // expansion budget sees only a few thousand units for the
    // whole table. The materialisation-work budget charges each
    // evaluation at its byte size, so the cumulative charge
    // crosses it at about the thirty-second element and the
    // root constant stays undecoded. (At 5,000 elements the
    // same shape copies over 600 GiB; the within-budget
    // alternating shape is covered by
    // `run_mode_slice_table_across_large_statics_stays_within_cache_cap`.)
    let mut src = String::new();
    src.push_str("static A: [u8; 135266304] = [7; 135266304]; ");
    src.push_str("static B: [u8; 135266304] = [9; 135266304]; ");
    let mut elems = Vec::new();
    for i in 0..40 {
        let name = if i % 2 == 0 { "A" } else { "B" };
        elems.push(format!(
            "unsafe {{ std::slice::from_raw_parts({name}.as_ptr(), 1) }}"
        ));
    }
    src.push_str(&format!("static TABLE: [&[u8]; 40] = [{}]; ", elems.join(", ")));
    src.push_str(
        "fn main() { println!(\"{} {} {} {}\", TABLE[0][0], TABLE[1][0], TABLE[39][0], TABLE.len()); }",
    );
    let dir = temp_dir("alternating-statics-work-budget");
    let src = fixture(&dir, "alternating_statics_work_budget.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a slice table whose alternating statics exhaust the materialisation-work budget must stay undecoded; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "",
        "no wrong value may be printed for an undecoded constant"
    );
}

#[test]
fn run_mode_large_reference_table_reads_back() {
    // A 400,000-element table of references to one promoted
    // value: the table's allocation carries 400,000
    // provenance entries, and each element's decode looks
    // its entry up in the offset-sorted vector by binary
    // search. A linear scan made this decode quadratic in
    // the table length (on the order of 80 billion
    // comparisons) and the run did not finish in reasonable
    // time.
    let src = "static TABLE: [&u32; 400000] = [&7; 400000]; fn main() { println!(\"{} {}\", TABLE[0], TABLE[399999]); }";
    let dir = temp_dir("large-reference-table");
    let src = fixture(&dir, "large_reference_table.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a large reference table must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "7 7\n",
        "decoded table values must match the source program"
    );
}

#[test]
fn run_mode_repeated_static_ref_table_converts_and_reads_back() {
    // A static initializer that builds a 300-element table of
    // one-byte slices alternating between two 1 MiB statics.
    // Every element's slice expression mentions its static
    // through a distinct reference constant, and each such
    // constant's decode materialises the static's whole
    // backing, so the initializer body's conversion otherwise
    // pays the backing size per element (the 5,000-element
    // shape of Issue #492 took ~500 s to convert before the
    // root-constant memo and the pre-sized byte copy in
    // `alloc_bytes_in`). The decoded values must stay exact:
    // each element reads back the byte of the static it
    // denotes, so the memo must not conflate the two targets.
    let mut src = String::new();
    src.push_str("static A: [u8; 1048576] = [7; 1048576]; ");
    src.push_str("static B: [u8; 1048576] = [9; 1048576]; ");
    let mut elems = Vec::new();
    for i in 0..300 {
        let name = if i % 2 == 0 { "A" } else { "B" };
        elems.push(format!(
            "unsafe {{ std::slice::from_raw_parts({name}.as_ptr(), 1) }}"
        ));
    }
    src.push_str(&format!(
        "static TABLE: [&[u8]; 300] = [{}]; ",
        elems.join(", ")
    ));
    src.push_str(
        "fn main() { println!(\"{} {} {} {} {}\", TABLE[0][0], TABLE[1][0], TABLE[150][0], TABLE[299][0], TABLE.len()); }",
    );
    let dir = temp_dir("repeated-static-ref-table");
    let src = fixture(&dir, "repeated_static_ref_table.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a table of repeated static references must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "7 9 7 9 300\n",
        "each element must read back the byte of the static it denotes"
    );
}

#[test]
fn run_mode_slice_table_into_shared_array_reads_requested_ranges() {
    // Slices at different offsets into one shared backing
    // array decode exactly the requested byte ranges: the
    // fat-pointer decode reads `target.bytes` over the
    // element's range only, so values at nonzero offsets and
    // a slice ending at the backing's last byte come back
    // byte-exact.
    let src = "static BIG: [u8; 4096] = { let mut a = [0u8; 4096]; let mut i = 0; while i < 4096 { a[i] = (i % 251) as u8; i += 1; } a }; static TABLE: [&[u8]; 3] = unsafe { [std::slice::from_raw_parts(BIG.as_ptr(), 1), std::slice::from_raw_parts(BIG.as_ptr().add(100), 3), std::slice::from_raw_parts(BIG.as_ptr().add(4000), 96)] }; fn main() { println!(\"{} {} {} {}\", TABLE[0][0], TABLE[1][2], TABLE[2][95], TABLE[2].len()); }";
    let dir = temp_dir("slice-table-ranges");
    let src = fixture(&dir, "slice_table_ranges.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a slice table into a shared array must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "0 102 79 96\n",
        "slices must read back the requested ranges of the backing array"
    );
}

#[test]
fn run_mode_large_slice_table_into_shared_static_reads_back() {
    // A 400,000-element table of one-byte slices into one
    // promoted 16 MiB static: the expansion budget charges
    // only the requested byte per element, but decoding used
    // to materialise the whole backing allocation for every
    // element (over 6 TB of transient copies in total), so
    // the run could not finish. The static's initialiser is
    // now evaluated once per root decode and each element
    // reads only its requested range.
    let src = "static BIG: [u8; 16777216] = [7; 16777216]; static TABLE: [&[u8]; 400000] = [unsafe { std::slice::from_raw_parts(BIG.as_ptr(), 1) }; 400000]; fn main() { println!(\"{} {} {}\", TABLE[0][0], TABLE[399999][0], TABLE.len()); }";
    let dir = temp_dir("large-slice-table-static");
    let src = fixture(&dir, "large_slice_table_static.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a large slice table into a shared static must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "7 7 400000\n",
        "decoded slice values must match the source program"
    );
}

#[test]
fn run_mode_large_slice_table_into_shared_const_reads_back() {
    // The same table shape with a `const` backing array: the
    // elements resolve through the memory-allocation arm,
    // whose materialisation is cached per root decode, so
    // 1,000 elements into one 16 MiB allocation decode
    // without re-materialising the backing per element.
    let src = "const BIGC: [u8; 16777216] = [7; 16777216]; static TABLE: [&[u8]; 1000] = [unsafe { std::slice::from_raw_parts(BIGC.as_ptr(), 1) }; 1000]; fn main() { println!(\"{} {} {}\", TABLE[0][0], TABLE[999][0], TABLE.len()); }";
    let dir = temp_dir("large-slice-table-const");
    let src = fixture(&dir, "large_slice_table_const.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a large slice table into a shared const must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "7 7 1000\n",
        "decoded slice values must match the source program"
    );
}

#[test]
fn run_mode_large_pair_table_static() {
    // Elements with a stride larger than the element payloads
    // (a `(u16, u16)` pair): the presence check runs on the
    // strided byte count, and each pair decodes structurally.
    let elems: Vec<String> = (0..5000).map(|i| format!("({i}, {})", i * 2)).collect();
    let src = format!(
        "static TABLE: [(u16, u16); 5000] = [{}]; fn main() {{ println!(\"{{}} {{}} {{}} {{}}\", TABLE[0].0, TABLE[0].1, TABLE[4999].0, TABLE[4999].1); }}",
        elems.join(", ")
    );
    let dir = temp_dir("large-pair-table");
    let src = fixture(&dir, "large_pair_table.rs", &src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "large pair table must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "0 0 4999 9998\n",
        "decoded table values must match the source program"
    );
}

#[test]
fn run_mode_large_f64_table_static() {
    // A static `f64` table past the bound, computed in a const
    // block; the spot values are exact binary fractions, so the
    // printed forms are stable.
    let src = "static TABLE: [f64; 5000] = { let mut t = [0.0f64; 5000]; let mut i = 0usize; while i < 5000 { t[i] = (i as f64) * 0.5 - 3.0; i += 1; } t }; fn main() { println!(\"{} {} {}\", TABLE[0], TABLE[2500], TABLE[4999]); }";
    let dir = temp_dir("large-f64-table");
    let src = fixture(&dir, "large_f64_table.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "large f64 table must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "-3 1247 2496.5\n",
        "decoded table values must match the source program"
    );
}

#[test]
fn run_mode_large_bool_table_static() {
    // A 9,000-element `bool` table: one byte per element, so
    // the allocation-presence check is what admits it past the
    // element bound.
    let src = "static TABLE: [bool; 9000] = { let mut t = [false; 9000]; let mut i = 0usize; while i < 9000 { t[i] = i % 3 == 0; i += 1; } t }; fn main() { let mut n = 0u32; for b in TABLE.iter() { if *b { n += 1; } } println!(\"{} {} {}\", n, TABLE[3], TABLE[4]); }";
    let dir = temp_dir("large-bool-table");
    let src = fixture(&dir, "large_bool_table.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "large bool table must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "3000 true false\n",
        "decoded table values must match the source program"
    );
}

#[test]
fn run_mode_large_u32_table_nested_static() {
    // The same table as a field of a struct static, behind a
    // scalar field: the decode runs at a nonzero offset into
    // the allocation, and the presence check accounts for it.
    let src = "struct Wrap { tag: u32, table: [u32; 5000] } static W: Wrap = Wrap { tag: 77, table: { let mut t = [0u32; 5000]; let mut i = 0usize; while i < 5000 { t[i] = (i as u32) * 5 + 2; i += 1; } t } }; fn main() { println!(\"{} {} {}\", W.tag, W.table[0], W.table[4999]); }";
    let dir = temp_dir("large-u32-table-nested");
    let src = fixture(&dir, "large_u32_table_nested.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested large u32 table must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "77 2 24997\n",
        "decoded table values must match the source program"
    );
}

#[test]
fn run_mode_large_str_ref_table_static() {
    // A 5,000-element table of references: each element
    // decodes through its provenance entry, past the bound.
    let src = "static NAMES: [&str; 5000] = [\"xy\"; 5000]; fn main() { let mut total = 0usize; for s in NAMES.iter() { total += s.len(); } println!(\"{} {}\", total, NAMES[4999]); }";
    let dir = temp_dir("large-str-ref-table");
    let src = fixture(&dir, "large_str_ref_table.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "large str ref table must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "10000 xy\n",
        "decoded table values must match the source program"
    );
}

#[test]
fn run_mode_materialised_constant_shapes() {
    // Struct constants, a reference into the middle of an array
    // constant, statics read by value, a `[&str]` constant and a
    // niche-encoded `Option<&i32>` constant, all printed.
    let dir = temp_dir("const-shapes");
    let src = fixture(&dir, "const_shapes.rs", CONST_SHAPES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "materialised constants must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "-2 7 20 42 yy 5\n",
        "decoded constant values must match the source program"
    );
}

const ASYNC_CLOSURE_SRC: &str = r#"
fn main() {
    let f = async move |x: i32| x + 1;
    let _ = &f;
}
"#;

const ASYNC_CLOSURE_CALL_SRC: &str = r#"
fn main() {
    let f = async move |x: i32| x + 1;
    let _fut = f(1);
}
"#;

const COMPLETED_FUTURE_DROP_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

fn poll_once<F: Future>(fut: F) -> Poll<F::Output> {
    let mut fut = Box::pin(fut);
    let mut cx = Context::from_waker(Waker::noop());
    Pin::as_mut(&mut fut).poll(&mut cx)
}

struct Guard(&'static str);
impl Drop for Guard {
    fn drop(&mut self) {
        println!("dropped {}", self.0);
    }
}

fn main() {
    {
        let g = Guard("block-capture");
        let fut = async move {
            let _ = &g;
            5
        };
        drop(fut);
        println!("block future dropped");
    }
    let g = Guard("block-done");
    let fut = async move {
        let _ = &g;
        6
    };
    match poll_once(fut) {
        Poll::Ready(v) => println!("ready: {}", v),
        Poll::Pending => println!("pending"),
    }
    println!("end of main");
}
"#;


#[test]
fn run_mode_tail_call_body_is_skipped_and_call_exits_two() {
    // Converting a `become` (explicit tail call) makes rustc_public
    // panic (`unimplemented!()` in Stable MIR's terminator
    // conversion). The driver contains that panic at the compiler
    // boundary and skips the body with the named tail-call reason,
    // so conversion completes; the call in `main` then reports the
    // unconverted body under the usual run-mode contract — exit 2 —
    // instead of the whole run dying with the panic (exit 101 and
    // a default panic dump on stderr).
    let dir = temp_dir("tailcall");
    let src = fixture(&dir, "tailcall.rs", TAIL_CALL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "calling an unconverted body is unsupported (exit 2), not a process panic; stderr={stderr}"
    );
    assert!(
        stderr.contains("no converted body"),
        "the reason names the unconverted body: {stderr}"
    );
    assert!(
        stderr.contains("tailcall::countdown"),
        "the reason names the skipped function: {stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "the conversion panic is contained, not reported: {stderr}"
    );
    assert!(
        !stderr.contains("RUST_BACKTRACE"),
        "the default panic dump is suppressed, the message is reported as the reason: {stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "run mode keeps stdout clean: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}


#[test]
fn run_mode_async_closure_call_runs() {
    // Calling an async closure goes through the `AsyncFn` trait
    // dispatch: the call creates the closure's coroutine (the
    // value the compiler's shim body would construct) and the
    // program runs to completion - the created future here is
    // simply never polled. (Before Lane MJ this call was the
    // unsupported case this test pinned; the end-to-end behaviour
    // of the dispatch is covered by the driver_async_closure
    // tests.)
    let dir = temp_dir("asyncclosure");
    let src = fixture(&dir, "asyncclosure.rs", ASYNC_CLOSURE_CALL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "calling an async closure runs; stderr={stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "the program prints nothing: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}


#[test]
fn run_mode_completed_future_drops_capture_once() {
    // A completed async block's capture is dropped by the
    // coroutine body itself when it completes (the body's own
    // `Drop` terminator on the state field), and dropping the
    // completed future afterwards must not drop it a second
    // time: the body's drop leaves the state cell dead, and the
    // future's glue finds nothing left there. An unpolled future
    // still drops its capture exactly once, through its glue.
    let dir = temp_dir("completed_future_drop");
    let src = fixture(&dir, "completed_future_drop.rs", COMPLETED_FUTURE_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program runs to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "dropped block-capture\nblock future dropped\ndropped block-done\nready: 6\nend of main\n",
        "each capture drops exactly once"
    );
}


#[test]
fn run_mode_uncalled_async_closure_runs() {
    // Defining an async closure and borrowing the value never
    // calls it: the closure value is a zero-sized value whose
    // creation the converted body represents without the
    // closure's type (which Stable MIR cannot hand over), so
    // the program runs to completion.
    let dir = temp_dir("asyncclosure_uncalled");
    let src = fixture(&dir, "asyncclosure.rs", ASYNC_CLOSURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an uncalled async closure runs; stderr={stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "run mode keeps stdout clean: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

const EXPLICIT_DISCRIMINANT_SRC: &str = r#"
#[repr(i32)]
enum E { A = 3, B = 7, C = -2 }
#[repr(i32)]
enum P { X(i32) = -5, Y(u32) = 42, Z = 0 }
#[repr(u8)]
enum U { Lo = 3, Hi = 200 }
fn name(e: E) -> i32 { match e { E::A => 100, E::B => 200, E::C => 300 } }
fn p_val(p: P) -> i32 { match p { P::X(v) => v, P::Y(v) => v as i32 + 1, P::Z => -1 } }
fn u_val(u: U) -> u32 { match u { U::Lo => 1, U::Hi => 2 } }
const PC: P = P::Y(9);
fn main() {
    println!("{}", name(E::A));
    println!("{}", name(E::B));
    println!("{}", name(E::C));
    println!("{}", p_val(P::X(-7)));
    println!("{}", p_val(PC));
    println!("{}", u_val(U::Hi));
    println!("{}", E::C as i32);
}
"#;

#[test]
fn run_mode_explicit_discriminant_match() {
    // Enums with explicit discriminants: `Discriminant` must yield
    // the discriminant values (3, 7, -2, ...), not the variant
    // indices, or every match falls through. The expected output
    // is what the native build of the same program prints.
    let dir = temp_dir("explicit-discriminant");
    let src = fixture(&dir, "explicit_discriminant.rs", EXPLICIT_DISCRIMINANT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "explicit-discriminant enums must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "100\n200\n300\n-7\n10\n2\n-2\n",
        "matches on explicit discriminants must pick the native branches"
    );
}

const NEGATIVE_INT_MATCH_SRC: &str = r#"
fn f(x: i32) -> i32 { match x { -2 => 300, 3 => 100, _ => 0 } }
fn main() { println!("{}", f(-2)); println!("{}", f(3)); println!("{}", f(9)); }
"#;

#[test]
fn run_mode_negative_int_match() {
    // SwitchInt branch keys are stored as zero-extended bit
    // patterns; a signed match must still hit its negative arms.
    let dir = temp_dir("negative-int-match");
    let src = fixture(&dir, "negative_int_match.rs", NEGATIVE_INT_MATCH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "signed int match must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "300\n100\n0\n",
        "the -2 arm must be taken for -2"
    );
}

#[test]
fn run_mode_generic_context_anon_const_executes() {
    let dir = temp_dir("anon-const-generic-run");
    let src = fixture(&dir, "anon_const_generic.rs", ANON_CONST_GENERIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "4\n");
}

// ----- allocated constants: f16, null and fn pointers, uninit (Lane CK) -----

const F16_SRC: &str = r#"#![feature(f16)]
const A: f16 = 1.5;
fn main() {
    let b: f16 = A + 0.25;
    println!("{}", b as f32);
    println!("{}", (A * 2.0) as f64);
}
"#;

const NULLPTR_SRC: &str = r#"fn target() {}
const FN_AS_PTR: *const () = target as *const ();
const NULLP: *const u8 = core::ptr::null();
static WORD: u32 = 9;
const STATIC_PTR: *const u32 = &raw const WORD;
fn main() {
    println!("{:?}", NULLP.is_null());
    println!("{:?}", FN_AS_PTR.is_null());
    println!("{:?}", unsafe { *STATIC_PTR });
}
"#;

#[test]
fn run_mode_f16_constants_execute() {
    let dir = temp_dir("f16const");
    let src = fixture(&dir, "f16const.rs", F16_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "f16 program must run to completion; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "1.75\n3\n");
}

const F16_NAN_SRC: &str = r##"#![feature(f16)]
fn emit(x: f16) {
    println!("{}", x.to_bits());
    println!("{}", x.floor().to_bits());
    println!("{}", x.abs().to_bits());
    println!("{}", (x as f32).to_bits());
    println!("{}", (-x).to_bits());
}
fn main() {
    emit(f16::from_bits(0x7c01));
    emit(f16::from_bits(0x7e01));
    emit(f16::from_bits(0xfc01));
    emit(f16::from_bits(0xfe00));
    emit(f16::from_bits(0x7dff));
    let s = f32::from_bits(0x7f800001);
    println!("{}", s.to_bits());
    println!("{}", (s as f16).to_bits());
}
"##;

#[test]
fn run_mode_f16_nan_transport() {
    // NaN bit patterns survive f16 transport: `from_bits` /
    // `to_bits` and place copies are the identity, the bit
    // operations (`abs`, negation) preserve a signaling NaN, the
    // arithmetic operations (`floor`) quiet it with its payload
    // kept, and the widening cast to `f32` preserves it (the
    // native values below were measured on the pinned toolchain).
    let dir = temp_dir("f16nan");
    let src = fixture(&dir, "f16nan.rs", F16_NAN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "f16 NaN program must run to completion; stderr={stderr}"
    );
    let expected = "\
31745
32257
31745
2139103232
64513
32257
32257
32257
2143297536
65025
64513
65025
31745
4286586880
31745
65024
65024
32256
4290772992
32256
32255
32767
32255
2143281152
65023
2139095041
32256
";
    assert_eq!(String::from_utf8_lossy(&out.stdout), expected);
}

const F16_COPYSIGN_SRC: &str = r##"#![feature(f16)]
fn emit(m: u16, s: u16) {
    println!("{}", f16::from_bits(m).copysign(f16::from_bits(s)).to_bits());
}
fn main() {
    emit(0x3c00, 0x0000);
    emit(0x3c00, 0x8000);
    emit(0x0000, 0x8000);
    emit(0x8000, 0x3c00);
    emit(0x7c01, 0x8000);
    emit(0xfc01, 0x0000);
    emit(0x7e00, 0xbc00);
    emit(0x7c00, 0x8000);
    println!("{}", 1.5f32.copysign(-0.0f32).to_bits());
    println!("{}", f32::from_bits(0x7f800001).copysign(-1.0f32).to_bits());
    println!("{}", (-2.5f32).copysign(0.0f32).to_bits());
    println!("{}", 1.5f64.copysign(-0.0f64).to_bits());
    println!("{}", f64::from_bits(0x7ff0000000000001).copysign(-1.0f64).to_bits());
    println!("{}", (-2.5f64).copysign(0.0f64).to_bits());
}
"##;

#[test]
fn run_mode_f16_copysign() {
    // `copysign` takes the sign from the sign argument and keeps
    // the magnitude's bits exactly — NaN payloads (signaling
    // included) pass through unquieted, signed zeros and
    // infinities follow the sign argument. All widths are served
    // by the float-method builtin; the values below were measured
    // natively on the pinned toolchain.
    let dir = temp_dir("f16copysign");
    let src = fixture(&dir, "f16copysign.rs", F16_COPYSIGN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "copysign program must run to completion; stderr={stderr}"
    );
    let expected = "\
15360
48128
32768
0
64513
31745
65024
64512
3217031168
4286578689
1075838976
13832806255468478464
18442240474082181121
4612811918334230528
";
    assert_eq!(String::from_utf8_lossy(&out.stdout), expected);
}

const F128_COPYSIGN_SRC: &str = r##"#![feature(f128)]
fn emit(m: u128, s: u128) {
    println!("{}", f128::from_bits(m).copysign(f128::from_bits(s)).to_bits());
}
fn main() {
    emit(0x3fff0000000000000000000000000000, 0x00000000000000000000000000000000);
    emit(0x3fff0000000000000000000000000000, 0x80000000000000000000000000000000);
    emit(0x00000000000000000000000000000000, 0x80000000000000000000000000000000);
    emit(0x80000000000000000000000000000000, 0x3fff0000000000000000000000000000);
    emit(0x7fff0000000000000000000000000001, 0x80000000000000000000000000000000);
    emit(0xffff0000000000000000000000000001, 0x00000000000000000000000000000000);
    emit(0x7f800000000000000000000000000000, 0xbfff0000000000000000000000000000);
    emit(0x7fff0000000000000000000000000000, 0x80000000000000000000000000000000);
    println!("{}", 1.5f128.copysign(-0.0f128).to_bits());
    println!("{}", (-2.5f128).copysign(0.0f128).to_bits());
    println!("{}", unsafe { std::mem::transmute::<f128, u128>(2.5f128) });
    println!("{}", unsafe { std::mem::transmute::<u128, f128>(0x40004000000000000000000000000000u128) }.to_bits());
}
"##;

#[test]
fn run_mode_f128_copysign_and_transmute() {
    // `f128::copysign` runs the collected `copysignf128` body,
    // whose `to_bits` / `from_bits` are `Cast Transmute`
    // rvalues between `f128` and `u128`. The scalar-shape
    // reinterpretation covers the 128-bit float shape (an
    // `f128` value is stored as its exact bit pattern), so
    // both directions are exact: NaN payloads, signaling
    // included, pass through unquieted, and signed zeros and
    // infinities follow the sign argument. The values below
    // were measured natively on the pinned toolchain.
    let dir = temp_dir("f128copysign");
    let src = fixture(&dir, "f128copysign.rs", F128_COPYSIGN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "f128 copysign program must run to completion; stderr={stderr}"
    );
    let expected = "\
85065399433376081038215121361612832768
255206582893845312769902425077496938496
170141183460469231731687303715884105728
0
340277174624079928635746076935438991361
170135991163610696904058773219554885633
339617752923046005526922703901628039168
340277174624079928635746076935438991360
255209179042274580183716690325661548544
85071889804449249572750784482024357888
85071889804449249572750784482024357888
85071889804449249572750784482024357888
";
    assert_eq!(String::from_utf8_lossy(&out.stdout), expected);
}

#[test]
fn run_mode_null_and_fn_pointer_constants_execute() {
    let dir = temp_dir("nullptr");
    let src = fixture(&dir, "nullptr.rs", NULLPTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "null/fn pointer constant program must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "true\nfalse\n9\n"
    );
}

// ----- statics storage model (Lane CQ) -----

const DROP_15858_SRC: &str = r#"
#![allow(static_mut_refs)]
static mut DROP_RAN: bool = false;

trait Bar {
    fn do_something(&mut self);
}

struct BarImpl;

impl Bar for BarImpl {
    fn do_something(&mut self) {}
}

struct Foo<B: Bar>(B);

impl<B: Bar> Drop for Foo<B> {
    fn drop(&mut self) {
        unsafe {
            DROP_RAN = true;
        }
    }
}

fn main() {
    {
        let _x: Foo<BarImpl> = Foo(BarImpl);
    }
    unsafe {
        assert_eq!(DROP_RAN, true);
        println!("drop ran");
    }
}
"#;

const STATIC_INDEPENDENT_SRC: &str = r#"
static mut A: i32 = 1;
static mut B: i32 = 2;

fn main() {
    unsafe {
        A += 10;
        println!("a={} b={}", A, B);
    }
}
"#;

const STATIC_INTERIOR_PTR_SRC: &str = r#"
use std::cell::UnsafeCell;

#[repr(transparent)]
struct SyncUnsafeCellLike(UnsafeCell<u32>);
unsafe impl Sync for SyncUnsafeCellLike {}

#[repr(C)]
struct SyncPtr {
    foo: *mut u32,
}
unsafe impl Sync for SyncPtr {}

static mut STATIC: u32 = 42;
static PTR: SyncPtr = SyncPtr { foo: std::ptr::addr_of_mut!(STATIC) };

static WRAPPED: SyncUnsafeCellLike = SyncUnsafeCellLike(UnsafeCell::new(7));
static PTR2: SyncPtr =
    SyncPtr { foo: std::ptr::addr_of!(WRAPPED) as *mut u32 };

fn main() {
    unsafe {
        assert_eq!(*PTR.foo, 42);
        *PTR.foo = 0;
        assert_eq!(*PTR.foo, 0);
        assert_eq!(*PTR2.foo, 7);
        *PTR2.foo = 9;
        assert_eq!(*PTR2.foo, 9);
        println!("done");
    }
}
"#;

const STATIC_INTERIOR_REF_SRC: &str = r#"
struct S { a: usize }

static A: S = S { a: 3 };
static B: &'static usize = &A.a;
static D: [usize; 1] = [1];
static F: &'static usize = &D[0];

fn main() {
    assert_eq!(*B, A.a);
    assert_eq!(*F, D[0]);
    println!("b={} f={}", *B, *F);
}
"#;

const STATIC_INTERIOR_SRC: &str = r#"
use std::cell::RefCell;

static mut CELL: RefCell<i32> = RefCell::new(10);

fn main() {
    unsafe {
        *CELL.borrow_mut() = 5;
        let v = *CELL.borrow();
        *CELL.borrow_mut() = v + 2;
        println!("cell={}", *CELL.borrow());
    }
}
"#;

const STATIC_MUT_COUNTER_SRC: &str = r#"
static mut COUNT: i32 = 0;

fn bump() {
    unsafe { COUNT += 1; }
}

fn main() {
    bump();
    bump();
    bump();
    unsafe { println!("count={}", COUNT); }
}
"#;

#[test]
fn run_mode_generic_drop_trait_bound_15858() {
    let dir = temp_dir("drop-15858");
    let src = fixture(&dir, "drop_15858.rs", DROP_15858_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the 15858 destructor must run and write the static; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "drop ran\n",
        "main observes the static write the destructor made"
    );
}

#[test]
fn run_mode_static_interior_mutability() {
    let dir = temp_dir("static-interior");
    let src = fixture(&dir, "interior.rs", STATIC_INTERIOR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a RefCell living in a static must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "cell=7\n",
        "guard writes to a static are stored back and read by later borrows"
    );
}

#[test]
fn run_mode_static_interior_pointer_writes_through() {
    let dir = temp_dir("static-interior-ptr");
    let src = fixture(&dir, "interior_ptr.rs", STATIC_INTERIOR_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a raw pointer to a static (or into its wrapper) must write the store; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "done\n",
        "writes through interior pointers are observed by later reads"
    );
}

#[test]
fn run_mode_static_interior_reference_reads_the_field() {
    let dir = temp_dir("static-interior-ref");
    let src = fixture(&dir, "interior_ref.rs", STATIC_INTERIOR_REF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a reference into a static must read the field it names; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "b=3 f=1\n",
        "the store seed is the full initializer; the reference carries the projection"
    );
}

#[test]
fn run_mode_static_mut_counter_across_calls() {
    let dir = temp_dir("static-counter");
    let src = fixture(&dir, "counter.rs", STATIC_MUT_COUNTER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a static mut counter must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "count=3\n",
        "writes to a static persist across calls"
    );
}

#[test]
fn run_mode_two_statics_of_same_type_stay_independent() {
    let dir = temp_dir("static-independent");
    let src = fixture(&dir, "independent.rs", STATIC_INDEPENDENT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "two statics must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "a=11 b=2\n",
        "statics are keyed by identity, not by type or initializer"
    );
}

const ADT_CONST_SRC: &str = r#"
#![feature(test)]
extern crate test;
use test::black_box as b;
fn main() {
    assert_eq!(b(1u128).checked_shl(b(127)), Some(1 << 127));
    println!("ok");
}
"#;

#[test]
fn run_mode_adt_constant_fields_use_their_field_types() {
    // The `Some(1 << 127)` constant's payload decodes under the
    // field's own `u128` type, not by a sign-bit guess.
    run_ok("adt-const", ADT_CONST_SRC, "ok\n");
}


// ----- statics holding slice references (Issue #140) ---------------------------

const STATIC_MUT_SLICE_SRC: &str = r#"
#![allow(static_mut_refs)]

static mut TEST: &'static mut [isize] = &mut [1];
static mut EMPTY: &'static mut [isize] = &mut [];
static mut INT: &'static mut isize = &mut 1;
static mut TEST_RAW: *mut [isize] = &mut [1isize] as *mut _;
static mut INT_RAW: *mut isize = &mut 1isize as *mut _;

pub fn main() {
    unsafe {
        TEST[0] += 1;
        assert_eq!(TEST[0], 2);
        assert_eq!(EMPTY.len(), 0);
        *INT += 1;
        assert_eq!(*INT, 2);
        (*TEST_RAW)[0] += 1;
        assert_eq!((*TEST_RAW)[0], 2);
        *INT_RAW += 1;
        assert_eq!(*INT_RAW, 2);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_static_mut_slice_references_write_through() {
    // A mutable slice reference stored in a static denotes the
    // promoted array's own storage (a fat reference over the
    // static reference), so a write through it is observed by
    // later reads of the static instead of landing in a detached
    // snapshot of the elements.
    let dir = temp_dir("static-mut-slice");
    let src = fixture(&dir, "static_mut_slice.rs", STATIC_MUT_SLICE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "writes through a static's slice reference must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const STATIC_SLICE_ALIAS_SRC: &str = r#"
use std::sync::atomic::{AtomicIsize, Ordering};

static A: [AtomicIsize; 1] = [AtomicIsize::new(7)];
static TEST: &[AtomicIsize] = &A;

pub fn main() {
    TEST[0].store(9, Ordering::SeqCst);
    assert_eq!(A[0].load(Ordering::SeqCst), 9);
    println!("ok");
}
"#;

#[test]
fn run_mode_static_slice_alias_observes_the_named_static() {
    // A slice reference to a whole named array static is the
    // same storage as the static itself: an interior write
    // through the alias is read back through the static.
    let dir = temp_dir("static-slice-alias");
    let src = fixture(&dir, "static_slice_alias.rs", STATIC_SLICE_ALIAS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a slice alias of an array static must share its storage; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

// ----- uninitialized array constants under optimized MIR (Lane JS) ---------
// At opt-level 2 `[MaybeUninit::uninit(); N]` const-propagates into a
// single `<uninit>` constant of the array type, and an element written
// with `MaybeUninit::new(..)` arrives as a decoded union constant
// carrying both union slots. The converted values must carry the
// array's and the union's shapes, so a dynamic-index write followed by
// a read of the written element observes the payload -- while a
// never-written element's payload stays an error, exactly as in the
// unoptimized form. Verification is by exit code: under -O a plain
// `println!` reaches unrelated fmt machinery gaps on this tree.

const UNINIT_ARRAY_O2_SRC: &str = r#"
use std::mem::MaybeUninit;
fn idx() -> usize { 2 }
fn main() {
    let mut arr: [MaybeUninit<u32>; 4] = [MaybeUninit::uninit(); 4];
    let i = idx();
    arr[i] = MaybeUninit::new(42);
    unsafe {
        let v = arr[i].assume_init();
        std::process::exit(v as i32);
    }
}
"#;

#[test]
fn run_mode_uninit_array_o2_dynamic_index_write_then_read() {
    let dir = temp_dir("uninit-array-o2");
    let src = fixture(&dir, "uninit_array_o2.rs", UNINIT_ARRAY_O2_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(42),
        "the written element's payload must read back; stderr={stderr}"
    );
}

const UNINIT_ARRAY_O2_LOOP_SRC: &str = r#"
use std::mem::MaybeUninit;
fn main() {
    let mut arr: [MaybeUninit<u32>; 4] = [MaybeUninit::uninit(); 4];
    for i in 0..4 {
        arr[i].write(i as u32 * 10);
    }
    unsafe {
        let sum = arr[0].assume_init() + arr[1].assume_init()
            + arr[2].assume_init() + arr[3].assume_init();
        std::process::exit(sum as i32);
    }
}
"#;

#[test]
fn run_mode_uninit_array_o2_loop_write_then_read_all() {
    let dir = temp_dir("uninit-array-o2-loop");
    let src = fixture(&dir, "uninit_array_o2_loop.rs", UNINIT_ARRAY_O2_LOOP_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(60),
        "every written element must read back; stderr={stderr}"
    );
}

const UNINIT_SCALAR_CONST_O2_SRC: &str = r#"
use std::mem::MaybeUninit;
#[inline(never)]
fn read(x: MaybeUninit<u32>) -> u32 { unsafe { x.assume_init() } }
fn main() {
    let x: MaybeUninit<u32> = MaybeUninit::new(42);
    std::process::exit(read(x) as i32);
}
"#;

#[test]
fn run_mode_maybeuninit_scalar_const_o2_reads_payload() {
    let dir = temp_dir("uninit-scalar-o2");
    let src = fixture(&dir, "uninit_scalar_o2.rs", UNINIT_SCALAR_CONST_O2_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(42),
        "a decoded MaybeUninit constant must hold its payload at field 0; stderr={stderr}"
    );
}

const UNINIT_ARRAY_O2_UNWRITTEN_SRC: &str = r#"
use std::mem::MaybeUninit;
fn idx() -> usize { 2 }
fn main() {
    let mut arr: [MaybeUninit<u32>; 4] = [MaybeUninit::uninit(); 4];
    let i = idx();
    arr[i] = MaybeUninit::new(42);
    unsafe {
        let v = arr[1].assume_init();
        std::process::exit(v as i32);
    }
}
"#;

#[test]
fn run_mode_uninit_array_o2_unwritten_element_stays_an_error() {
    let dir = temp_dir("uninit-array-o2-unwritten");
    let src = fixture(&dir, "uninit_array_o2_unwritten.rs", UNINIT_ARRAY_O2_UNWRITTEN_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "reading a never-written element's payload must stay a named error; stderr={stderr}"
    );
}

const TRANSMUTE_OPTION_FIELDLESS_SRC: &str = r#"
#[repr(u8)]
pub enum Size { One = 1, Two = 2, Three = 3 }
fn main() {
    let n: u8 = unsafe { std::mem::transmute::<Option<Size>, u8>(None) };
    let s: u8 = unsafe { std::mem::transmute::<Option<Size>, u8>(Some(Size::Two)) };
    let back: Option<Size> = unsafe { std::mem::transmute::<u8, Option<Size>>(3) };
    println!("{} {} {}", n, s, matches!(back, Some(Size::Three)));
}
"#;

#[test]
fn run_mode_transmute_option_fieldless_enum_niche() {
    run_ok(
        "transmute-option-fieldless",
        TRANSMUTE_OPTION_FIELDLESS_SRC,
        "0 2 true\n",
    );
}

const TRANSMUTE_FN_PTR_RETURN_SRC: &str = r#"
use std::mem;
#[inline(never)]
extern "C" fn foo(x: &isize) -> Option<&isize> { Some(x) }
static FOO: isize = 0xDEADBEE;
pub fn main() {
    unsafe {
        let f: extern "C" fn(&isize) -> &isize =
            mem::transmute(foo as extern "C" fn(&isize) -> Option<&isize>);
        println!("{}", *f(&FOO));
    }
}
"#;

#[test]
fn run_mode_transmute_fn_ptr_return_niche_to_ref() {
    run_ok(
        "transmute-fn-ptr-return",
        TRANSMUTE_FN_PTR_RETURN_SRC,
        "233495534\n",
    );
}

const UNIQUE_DANGLING_SRC: &str = r#"
#![feature(ptr_internals)]
use std::ptr::Unique;
const PTR: *mut u32 = Unique::dangling().as_ptr();
fn main() {
    let p = Unique::<u32>::dangling().as_ptr();
    println!("{}", PTR == p);
}
"#;

#[test]
fn run_mode_unique_dangling_const_and_runtime() {
    run_ok("unique-dangling", UNIQUE_DANGLING_SRC, "true\n");
}

const NONNULL_DANGLING_SRC: &str = r#"
use std::ptr::NonNull;
use std::num::NonZero;
#[inline(never)]
fn dangling() -> NonNull<u32> { NonNull::dangling() }
fn main() {
    let a = dangling();
    let b = NonNull::<u32>::without_provenance(NonZero::new(4usize).unwrap());
    println!("{} {}", a.as_ptr() as usize, b.as_ptr() as usize);
}
"#;

#[test]
fn run_mode_nonnull_dangling_without_provenance() {
    run_ok("nonnull-dangling", NONNULL_DANGLING_SRC, "4 4\n");
}


const SLICE_TO_VEC_SRC: &str = r#"
fn bar(v: &mut [usize]) -> Vec<usize> {
    v.to_vec()
}

fn bip(v: &[usize]) -> Vec<usize> {
    v.to_vec()
}

fn main() {
    // A slice over an array literal.
    let s: &[usize] = &[1, 2, 3, 100];
    let r = s.to_vec();
    assert_eq!(r.len(), 4);
    assert_eq!(r[0], 1);
    assert_eq!(r[3], 100);
    assert_eq!(r, vec![1, 2, 3, 100]);
    // A slice reached by deref coercion of a Vec reference at a
    // call argument, in both the shared and mutable forms.
    let mut the_vec = vec![1, 2, 3, 100];
    assert_eq!(the_vec.clone(), bar(&mut the_vec));
    assert_eq!(the_vec.clone(), bip(&the_vec));
    // Elements are cloned: a slice of owned strings copies.
    let words: &[String] = &[String::from("a"), String::from("bc")];
    let owned = words.to_vec();
    assert_eq!(owned.len(), 2);
    assert_eq!(owned[1], "bc");
    println!("slice-to-vec-ok");
}
"#;

#[test]
fn run_mode_slice_to_vec() {
    let dir = temp_dir("slice-to-vec");
    let src = fixture(&dir, "slice_to_vec.rs", SLICE_TO_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "slice-to-vec-ok\n");
}

const DEREF_COPY_INTO_ALLOC_SRC: &str = r#"
use std::alloc::{alloc, dealloc, Layout};

fn main() {
    let layout = Layout::array::<usize>(4).unwrap();
    let p = unsafe { alloc(layout) as *mut usize };
    assert!(!p.is_null());
    // The destination pointer reaches the copy through a
    // dereference of the fresh block's base pointer, so it
    // denotes the block's first element position while the
    // block still holds no written elements.
    let r: &mut usize = unsafe { &mut *p };
    let q = r as *mut usize;
    unsafe {
        std::ptr::copy_nonoverlapping([10usize, 20, 30].as_ptr(), q, 3);
        assert_eq!(*q, 10);
        assert_eq!(*q.add(1), 20);
        assert_eq!(*q.add(2), 30);
        dealloc(p as *mut u8, layout);
    }
    println!("deref-copy-ok");
}
"#;

#[test]
fn run_mode_copy_nonoverlapping_through_deref_into_fresh_alloc() {
    let dir = temp_dir("deref-copy-alloc");
    let src = fixture(
        &dir,
        "deref_copy_alloc.rs",
        DEREF_COPY_INTO_ALLOC_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "deref-copy-ok\n");
}

const ARC_WAKE_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

static WAKES: AtomicUsize = AtomicUsize::new(0);

trait ArcWake {
    fn wake(self: Arc<Self>);
    fn wake_by_ref(arc_self: &Arc<Self>) { arc_self.clone().wake() }
    fn into_waker(wake: Arc<Self>) -> Waker where Self: Sized {
        let ptr = Arc::into_raw(wake) as *const ();
        unsafe { Waker::from_raw(RawWaker::new(ptr, vtable::<Self>())) }
    }
}

fn vtable<T: ArcWake>() -> &'static RawWakerVTable {
    &RawWakerVTable::new(clone_arc_raw::<T>, wake_arc_raw::<T>, wake_by_ref_arc_raw::<T>, drop_arc_raw::<T>)
}
unsafe fn clone_arc_raw<T: ArcWake>(data: *const ()) -> RawWaker {
    let arc: Arc<T> = Arc::from_raw(data as *const T);
    let c = arc.clone();
    let _ = Arc::into_raw(arc);
    let _ = Arc::into_raw(c);
    RawWaker::new(data, vtable::<T>())
}
unsafe fn wake_arc_raw<T: ArcWake>(data: *const ()) {
    let arc: Arc<T> = Arc::from_raw(data as *const T);
    ArcWake::wake(arc);
}
unsafe fn wake_by_ref_arc_raw<T: ArcWake>(data: *const ()) {
    let arc: Arc<T> = Arc::from_raw(data as *const T);
    ArcWake::wake_by_ref(&arc);
    let _ = Arc::into_raw(arc);
}
unsafe fn drop_arc_raw<T: ArcWake>(data: *const ()) {
    let _: Arc<T> = Arc::from_raw(data as *const T);
}

struct Counter;
impl ArcWake for Counter {
    fn wake(self: Arc<Self>) { WAKES.fetch_add(1, Ordering::SeqCst); }
}

struct YieldOnce(bool);
impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0 { Poll::Ready(()) } else { cx.waker().wake_by_ref(); self.0 = true; Poll::Pending }
    }
}

fn main() {
    let counter = Arc::new(Counter);
    let waker = ArcWake::into_waker(counter);
    let mut cx = Context::from_waker(&waker);
    let mut fut = YieldOnce(false);
    let mut pinned = unsafe { Pin::new_unchecked(&mut fut) };
    assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Pending);
    assert_eq!(WAKES.load(Ordering::SeqCst), 1);
    assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Ready(()));
    // Dropping the waker runs the vtable drop entry; cloning it
    // first runs the clone entry as well.
    let waker2 = waker.clone();
    drop(waker2);
    drop(waker);
    println!("arc-wake-ok");
}
"#;

#[test]
fn run_mode_arc_wake_vtable_waker() {
    // The whole RawWaker vtable surface as function-pointer
    // calls: the vtable entries live inside a constant
    // allocation, the waker is dropped through the vtable drop
    // entry, and the entries rebuild their `Arc` with
    // `Arc::from_raw`.
    run_ok_edition("arc-wake", ARC_WAKE_SRC, "arc-wake-ok\n");
}

const DYN_FUTURE_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

fn raw_waker() -> RawWaker {
    unsafe fn clone(_: *const ()) -> RawWaker { raw_waker() }
    unsafe fn no_op(_: *const ()) {}
    RawWaker::new(std::ptr::null(), &RawWakerVTable::new(clone, no_op, no_op, no_op))
}

async fn me() -> i32 { 41 }

fn main() {
    let waker = unsafe { Waker::from_raw(raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    let mut fut: Pin<Box<dyn Future<Output = i32>>> = Box::pin(me());
    match fut.as_mut().poll(&mut cx) {
        Poll::Ready(v) => assert_eq!(v, 41),
        Poll::Pending => panic!("pending"),
    }
    println!("dyn-future-ok");
}
"#;

#[test]
fn run_mode_dyn_future_poll() {
    // `Future::poll` through a `Pin<Box<dyn Future>>`: the
    // dispatch finds the coroutine body (the compiler's own
    // poll implementation for an async fn) through the vtable
    // demand, with the receiver rebuilt in the callee's
    // `Pin<&mut Self>` shape.
    run_ok_edition("dyn-future", DYN_FUTURE_SRC, "dyn-future-ok\n");
}

const ARC_FROM_RAW_SRC: &str = r#"
use std::sync::Arc;
fn main() {
    let arc = Arc::new(41u32);
    let raw = Arc::into_raw(arc);
    let p2 = unsafe { raw.byte_add(8) };
    let p3 = unsafe { p2.byte_sub(8) };
    let arc2 = unsafe { Arc::from_raw(p3) };
    assert_eq!(*arc2, 41);
    assert_eq!(Arc::strong_count(&arc2), 1);
    println!("from-raw-ok");
}
"#;

#[test]
fn run_mode_arc_from_raw_roundtrip() {
    run_ok_edition("arc-from-raw", ARC_FROM_RAW_SRC, "from-raw-ok\n");
}

const ENUM_WRAPPER_TRANSMUTE_SRC: &str = r#"
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(usize)]
enum AlignEnum { A1 = 1, A2 = 2, A4 = 4, A8 = 8 }
#[repr(transparent)]
#[derive(Clone, Copy)]
struct Wrap(AlignEnum);
fn main() {
    let w: Wrap = unsafe { std::mem::transmute(AlignEnum::A8) };
    let back: AlignEnum = unsafe { std::mem::transmute(w) };
    assert_eq!(back, AlignEnum::A8);
    let n: usize = unsafe { std::mem::transmute(w) };
    assert_eq!(n, 8);
    let w2: Wrap = unsafe { std::mem::transmute(4usize) };
    let back2: AlignEnum = unsafe { std::mem::transmute(w2) };
    assert_eq!(back2, AlignEnum::A4);
    println!("enum-wrapper-ok");
}
"#;

#[test]
fn run_mode_enum_wrapper_transmutes() {
    // The `mem::Alignment` shapes: a fieldless enum and the
    // one-field transparent struct wrapping it transmute in
    // both directions, and an integer transmutes to the
    // wrapper through the wrapped enum's discriminant table
    // (the transmute `Arc::from_raw`'s alignment path uses).
    run_ok_edition(
        "enum-wrapper-transmute",
        ENUM_WRAPPER_TRANSMUTE_SRC,
        "enum-wrapper-ok\n",
    );
}

// ---------------------------------------------------------------------
// Async block state storage (Lane JY, Issue #186): a coroutine's
// upvar storage and its per-state saved storage are distinct even
// where their field indices coincide. An async block keeps its
// captures in the upvar storage and reads them back after a
// suspension whose saved field 0 is the awaitee; an async fn's
// transform copies the capture into the state instead. Both shapes
// (and nesting, and dropping a suspended block future) must run.

const ASYNC_BLOCK_STATE_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

struct Nop;
impl Wake for Nop {
    fn wake(self: Arc<Self>) {}
}

struct YieldOnce(bool);
impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.0 = true;
            Poll::Pending
        }
    }
}

fn async_block(x: u8) -> impl Future<Output = u8> {
    async move {
        YieldOnce(false).await;
        x
    }
}

fn two_awaits(a: u8, b: u8) -> impl Future<Output = u8> {
    async move {
        YieldOnce(false).await;
        let c = a + 1;
        YieldOnce(false).await;
        c + b
    }
}

fn nested(x: u8) -> impl Future<Output = u8> {
    async move {
        let inner = async move {
            YieldOnce(false).await;
            x + 1
        };
        inner.await
    }
}

fn string_upvar() -> impl Future<Output = usize> {
    let s = String::from("hello");
    async move {
        YieldOnce(false).await;
        s.len()
    }
}

fn main() {
    let waker = Waker::from(Arc::new(Nop));
    let mut cx = Context::from_waker(&waker);

    let mut fut = Box::pin(async_block(9));
    assert_eq!(Poll::Pending, fut.as_mut().poll(&mut cx));
    assert_eq!(Poll::Ready(9), fut.as_mut().poll(&mut cx));

    let mut fut = Box::pin(two_awaits(9, 3));
    assert_eq!(Poll::Pending, fut.as_mut().poll(&mut cx));
    assert_eq!(Poll::Pending, fut.as_mut().poll(&mut cx));
    assert_eq!(Poll::Ready(13), fut.as_mut().poll(&mut cx));

    let mut fut = Box::pin(nested(10));
    assert_eq!(Poll::Pending, fut.as_mut().poll(&mut cx));
    assert_eq!(Poll::Ready(11), fut.as_mut().poll(&mut cx));

    // A suspended block future holding a non-Copy capture drops
    // its saved state and the capture.
    let mut fut = Box::pin(string_upvar());
    assert_eq!(Poll::Pending, fut.as_mut().poll(&mut cx));
    drop(fut);
    let mut fut = Box::pin(string_upvar());
    assert_eq!(Poll::Pending, fut.as_mut().poll(&mut cx));
    assert_eq!(Poll::Ready(5), fut.as_mut().poll(&mut cx));

    println!("async-block-state-ok");
}
"#;

#[test]
fn run_mode_async_block_state_storage() {
    run_ok_edition(
        "async-block-state",
        ASYNC_BLOCK_STATE_SRC,
        "async-block-state-ok\n",
    );
}

// ----- uninitialized statics (Lane JZ) --------------------------------------
// A static whose initializer is (partly) uninitialized decodes to a
// bare `Uninit` seed: the storage holds no trace of the declared
// type's structure, so an element write lands in a shapeless value
// and every element read fails as an uninitialized-place read. The
// converted seed carries the declared type's skeleton (the same
// expansion the thread-local table applies to its seeds), so writes
// and reads through the static's storage observe the array's shape.
// Verification is by exit code, the sum of the payloads read back.

const STATIC_UNINIT_ARRAY_SRC: &str = r#"
use std::mem::MaybeUninit;
static mut ARR: [MaybeUninit<u32>; 4] = [MaybeUninit::uninit(); 4];
fn main() {
    unsafe {
        ARR[1] = MaybeUninit::new(42);
        let v = ARR[1].assume_init();
        std::process::exit(v as i32);
    }
}
"#;

#[test]
fn run_mode_static_uninit_array_write_then_read() {
    let dir = temp_dir("static-uninit-array");
    let src = fixture(&dir, "static_uninit_array.rs", STATIC_UNINIT_ARRAY_SRC);
    let out = run_driver_extra(&src, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(42),
        "the written element's payload must read back; stderr={stderr}"
    );
}

const STATIC_UNINIT_ARRAY_LOOP_SRC: &str = r#"
use std::mem::MaybeUninit;
static mut ARR: [MaybeUninit<u32>; 4] = [MaybeUninit::uninit(); 4];
fn main() {
    unsafe {
        let mut i = 0usize;
        while i < 4 {
            ARR[i] = MaybeUninit::new((i * 10) as u32);
            i += 1;
        }
        let mut s = 0u32;
        let mut j = 0usize;
        while j < 4 {
            s += ARR[j].assume_init();
            j += 1;
        }
        std::process::exit(s as i32);
    }
}
"#;

#[test]
fn run_mode_static_uninit_array_dynamic_index_loop() {
    let dir = temp_dir("static-uninit-array-loop");
    let src = fixture(
        &dir,
        "static_uninit_array_loop.rs",
        STATIC_UNINIT_ARRAY_LOOP_SRC,
    );
    let out = run_driver_extra(&src, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(60),
        "every written element must read back; stderr={stderr}"
    );
}

const STATIC_UNINIT_SLICE_SRC: &str = r#"
use std::mem::MaybeUninit;
static mut S: &mut [MaybeUninit<u32>] = &mut [MaybeUninit::uninit(); 3];
fn main() {
    unsafe {
        S[0] = MaybeUninit::new(5);
        S[2] = MaybeUninit::new(7);
        let s = S[0].assume_init() + S[2].assume_init();
        std::process::exit(s as i32);
    }
}
"#;

#[test]
fn run_mode_static_uninit_slice_write_then_read() {
    let dir = temp_dir("static-uninit-slice");
    let src = fixture(&dir, "static_uninit_slice.rs", STATIC_UNINIT_SLICE_SRC);
    let out = run_driver_extra(&src, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(12),
        "elements written through the slice static must read back; stderr={stderr}"
    );
}

const STATIC_UNINIT_SCALAR_AND_FIELD_SRC: &str = r#"
use std::mem::MaybeUninit;
static mut MU: MaybeUninit<u32> = MaybeUninit::uninit();
struct Holder {
    tag: u32,
    buf: [MaybeUninit<u16>; 2],
}
static mut H: Holder = Holder {
    tag: 9,
    buf: [MaybeUninit::uninit(); 2],
};
fn main() {
    unsafe {
        MU.as_mut_ptr().write(77);
        H.buf[1] = MaybeUninit::new(55u16);
        let s = MU.assume_init() + H.tag + H.buf[1].assume_init() as u32;
        std::process::exit(s as i32);
    }
}
"#;

#[test]
fn run_mode_static_uninit_scalar_and_struct_field() {
    let dir = temp_dir("static-uninit-scalar-field");
    let src = fixture(
        &dir,
        "static_uninit_scalar_field.rs",
        STATIC_UNINIT_SCALAR_AND_FIELD_SRC,
    );
    let out = run_driver_extra(&src, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(141),
        "the scalar static and the struct field array must both read back; stderr={stderr}"
    );
}

const RAW_PARTS_REBUILD_SRC: &str = r#"
fn main() {
    let s: &[u8] = &[7, 8, 9];
    let sp = s.as_ptr();
    let sr = unsafe { std::slice::from_raw_parts(sp, 3) };
    println!("rebuild: {} {} {}", sr[0], sr[1], sr[2]);
    let sr2 = unsafe { std::slice::from_raw_parts(sp.add(1), 2) };
    println!("rebuild-off: {} {}", sr2[0], sr2[1]);
    let a: [u8; 4] = [10, 20, 30, 40];
    let ap = a.as_ptr();
    let ar = unsafe { std::slice::from_raw_parts(ap, 4) };
    println!("arr rebuild: {} {} {} {}", ar[0], ar[1], ar[2], ar[3]);
    let mut b: [u8; 4] = [0, 0, 0, 0];
    {
        let p = b.as_mut_ptr();
        let w = unsafe { std::slice::from_raw_parts_mut(p.add(1), 2) };
        w[0] = 7;
        w[1] = 9;
    }
    println!("wback {} {} {} {}", b[0], b[1], b[2], b[3]);
    // A row of a matrix reached through an element pointer is
    // the element itself, not a window over the outer array.
    let m: [[u8; 3]; 2] = [[1, 2, 3], [4, 5, 6]];
    let row: &[u8] = &m[1];
    println!("row {} {} {} len {}", row[0], row[1], row[2], row.len());
}
"#;

#[test]
fn run_mode_raw_parts_rebuilt_slice_reads() {
    run_ok(
        "raw-parts-rebuild",
        RAW_PARTS_REBUILD_SRC,
        "rebuild: 7 8 9\nrebuild-off: 8 9\narr rebuild: 10 20 30 40\nwback 0 7 9 0\nrow 4 5 6 len 3\n",
    );
}

const FMT_GENERIC_AS_ARRAY_SRC: &str = r#"
use std::fmt;
struct S<T, const N: usize>(pub [T; N]);
impl<T, const N: usize> S<T, N> {
    pub const fn as_array(&self) -> &[T; N] {
        let p: *const Self = self;
        unsafe { &*p.cast::<[T; N]>() }
    }
}
impl<T: fmt::Debug, const N: usize> fmt::Debug for S<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <[T; N] as fmt::Debug>::fmt(self.as_array(), f)
    }
}
fn main() {
    println!("{:?}", S::<i16, 3>([10, -3, -9]));
}
"#;

#[test]
fn run_mode_generic_single_field_pointer_cast_enters_the_field() {
    let dir = temp_dir("fmt-generic-as-array");
    let src = fixture(&dir, "generic_as_array.rs", FMT_GENERIC_AS_ARRAY_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "must run; stdout={stdout}");
    assert_eq!(
        stdout, "[10, -3, -9]\n",
        "a raw-pointer cast from a single-field struct to its \
         field type denotes the field, also from generic code"
    );
}

const FMT_SIMD_CAST_SIGN_SRC: &str = r#"
#![feature(repr_simd, core_intrinsics)]
#[repr(simd)]
#[derive(Copy, Clone)]
struct Simd<T, const N: usize>([T; N]);
impl<T, const N: usize> Simd<T, N> {
    fn as_array(&self) -> &[T; N] {
        let p: *const Self = self;
        unsafe { &*p.cast::<[T; N]>() }
    }
}
fn main() {
    let a = Simd::<i8, 3>([10, -3, -9]);
    let b: Simd<i16, 3> = unsafe { std::intrinsics::simd::simd_cast(a) };
    println!("{:?}", b.as_array());
}
"#;

#[test]
fn run_mode_simd_cast_sign_extends_signed_lanes() {
    let dir = temp_dir("fmt-simd-cast-sign");
    let src = fixture(&dir, "simd_cast_sign.rs", FMT_SIMD_CAST_SIGN_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "must run; stdout={stdout}");
    assert_eq!(
        stdout, "[10, -3, -9]\n",
        "simd_cast from a signed element type sign-extends its lanes"
    );
}

const COUNTING_ALLOCATOR_SRC: &str = r#"#![feature(slice_ptr_get)]

use std::alloc::{self, Allocator, Global, Layout, System};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

static HITS: AtomicUsize = AtomicUsize::new(0);

struct A;

unsafe impl alloc::GlobalAlloc for A {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        HITS.fetch_add(1, Ordering::SeqCst);
        alloc::GlobalAlloc::alloc(&alloc::System, layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        HITS.fetch_add(1, Ordering::SeqCst);
        alloc::GlobalAlloc::dealloc(&alloc::System, ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: A = A;

fn hits() -> usize {
    HITS.load(Ordering::SeqCst)
}

fn main() {
    println!("hello!");
    // std's lazy stdout buffer goes through the global
    // allocator once.
    let n = hits();
    assert!(n >= 1, "stdout buffer allocation not observed: {n}");

    unsafe {
        let layout = Layout::new::<u32>();
        // The trait entry points count; the System allocator
        // behind them does not.
        let memory = Global.allocate(layout).unwrap();
        assert_eq!(hits(), n + 1);
        Global.deallocate(NonNull::new(memory.as_non_null_ptr().as_ptr() as *mut u8).unwrap(), layout);
        assert_eq!(hits(), n + 2);

        let s = String::with_capacity(10);
        assert_eq!(hits(), n + 3);
        drop(s);
        assert_eq!(hits(), n + 4);

        let memory = System.allocate(layout).unwrap();
        assert_eq!(hits(), n + 4);
        System.deallocate(NonNull::new(memory.as_non_null_ptr().as_ptr() as *mut u8).unwrap(), layout);
        assert_eq!(hits(), n + 4);

        let b = Box::new(42u32);
        assert_eq!(hits(), n + 5);
        assert_eq!(*b, 42);
        drop(b);
        assert_eq!(hits(), n + 6);
    }
    println!("counting ok");
}
"#;

#[test]
fn run_mode_guest_global_allocator_observes_allocations() {
    // A guest `#[global_allocator]` must observe the
    // allocations the abstract std model performs: the lazy
    // stdout buffer, `String::with_capacity` and its drop,
    // `Box::new` and its drop, and the `Global` trait entry
    // points — while direct `System` allocations stay
    // invisible, exactly as in the ui allocator tests.
    let dir = temp_dir("counting-allocator");
    let src = fixture(&dir, "counting_allocator.rs", COUNTING_ALLOCATOR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "guest allocator must observe modelled allocations; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hello!\ncounting ok\n"
    );
}
const NESTED_MAYBE_UNINIT_BUFFER_SRC: &str = r#"
use std::mem::MaybeUninit;
fn main() {
    let mut buf: MaybeUninit<MaybeUninit<[u32; 2]>> = MaybeUninit::uninit();
    let ptr = buf.as_mut_ptr() as *mut u32;
    unsafe {
        ptr.write(11);
        ptr.add(1).write(22);
        let inner = buf.assume_init();
        let arr = inner.assume_init();
        println!("nested: {} {}", arr[0], arr[1]);
    }
}
"#;
#[test]
fn run_mode_nested_maybe_uninit_buffer_cast() {
    // `buf.as_mut_ptr() as *mut u32` over a nested
    // `MaybeUninit<MaybeUninit<[u32; 2]>>` crosses both wrapper
    // layers and the array layer in a single cast; the resulting
    // pointer denotes the first array element, so element writes
    // through it fill the array's slots in order.
    let dir = temp_dir("nested-maybe-uninit-buffer");
    let src = fixture(&dir, "nested_maybe_uninit.rs", NESTED_MAYBE_UNINIT_BUFFER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "nested: 11 22\n");
}

const MAYBE_UNINIT_COPY_MOVE_SRC: &str = r#"
fn main() {
    let m: std::mem::MaybeUninit<u8> = std::mem::MaybeUninit::uninit();
    unsafe {
        let mut b: std::mem::MaybeUninit<u8> = std::mem::MaybeUninit::uninit();
        std::ptr::copy_nonoverlapping(&m, &mut b, 1);
        let c = b;
        std::mem::forget(c);
    }
    println!("done");
}
"#;

#[test]
fn run_mode_maybe_uninit_copy_then_move_uninit() {
    // A `MaybeUninit` value whose payload is uninitialized is
    // the type's ordinary content: copying it element-wise and
    // moving the destination place carries the union value
    // (only a read of the payload itself is an uninitialized
    // read).
    run_ok("maybe-uninit-copy-move", MAYBE_UNINIT_COPY_MOVE_SRC, "done\n");
}

const MAYBE_UNINIT_COPY_DEFINED_SRC: &str = r#"
fn main() {
    let m: std::mem::MaybeUninit<u8> = std::mem::MaybeUninit::new(7);
    unsafe {
        let mut b: std::mem::MaybeUninit<u8> = std::mem::MaybeUninit::uninit();
        std::ptr::copy_nonoverlapping(&m, &mut b, 1);
        let c = b;
        println!("val={}", c.assume_init());
    }
}
"#;

#[test]
fn run_mode_maybe_uninit_copy_defined_assume_init() {
    // The element a `MaybeUninit`-typed copy moves is the union
    // value, not the bare payload: the destination place keeps
    // the union form its own `assume_init` reads.
    run_ok(
        "maybe-uninit-copy-defined",
        MAYBE_UNINIT_COPY_DEFINED_SRC,
        "val=7\n",
    );
}

const MAYBE_UNINIT_ALLOC_READ_SRC: &str = r#"
fn main() {
    unsafe {
        let layout = std::alloc::Layout::new::<std::mem::MaybeUninit<u8>>();
        let p = std::alloc::alloc(layout) as *mut std::mem::MaybeUninit<u8>;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let b: std::mem::MaybeUninit<u8> = std::ptr::read(p);
        std::mem::forget(b);
        std::alloc::dealloc(p as *mut u8, layout);
    }
    println!("done");
}
"#;

#[test]
fn run_mode_maybe_uninit_ptr_read_unwritten_alloc() {
    // A `MaybeUninit` place inside a raw allocation no write has
    // ever touched holds the union's uninitialized form, the
    // type's ordinary content: `ptr::read` of it succeeds (only
    // reading the payload itself is an uninitialized read).
    run_ok(
        "maybe-uninit-alloc-read",
        MAYBE_UNINIT_ALLOC_READ_SRC,
        "done\n",
    );
}

const MAYBE_UNINIT_ALLOC_READ_WRITTEN_SRC: &str = r#"
fn main() {
    unsafe {
        let layout = std::alloc::Layout::new::<std::mem::MaybeUninit<u8>>();
        let p = std::alloc::alloc(layout) as *mut std::mem::MaybeUninit<u8>;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        std::ptr::write(p, std::mem::MaybeUninit::new(41));
        let b: std::mem::MaybeUninit<u8> = std::ptr::read(p);
        println!("val={}", b.assume_init());
        std::alloc::dealloc(p as *mut u8, layout);
    }
}
"#;

#[test]
fn run_mode_maybe_uninit_ptr_read_written_alloc() {
    // The same read over storage a write did fill assembles the
    // payload the write stored.
    run_ok(
        "maybe-uninit-alloc-read-written",
        MAYBE_UNINIT_ALLOC_READ_WRITTEN_SRC,
        "val=41\n",
    );
}

const PLAIN_ALLOC_READ_UNWRITTEN_SRC: &str = r#"
fn main() {
    unsafe {
        let layout = std::alloc::Layout::new::<u8>();
        let p = std::alloc::alloc(layout) as *mut u8;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let b: u8 = std::ptr::read(p);
        println!("val={}", b);
        std::alloc::dealloc(p, layout);
    }
}
"#;

#[test]
fn run_mode_ptr_read_unwritten_alloc_plain_payload_errors() {
    // The unwritten-allocation allowance is the `MaybeUninit`
    // type's, not the read's: a plain payload type read from
    // never-written storage stays an uninitialized read.
    let dir = temp_dir("plain-alloc-read-unwritten");
    let src = fixture(&dir, "case.rs", PLAIN_ALLOC_READ_UNWRITTEN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("uninitialized"),
        "plain payload read must be an uninitialised read; stderr={stderr}"
    );
}

const MAYBE_UNINIT_ALLOC_READ_ASSUME_SRC: &str = r#"
fn main() {
    unsafe {
        let layout = std::alloc::Layout::new::<std::mem::MaybeUninit<u8>>();
        let p = std::alloc::alloc(layout) as *mut std::mem::MaybeUninit<u8>;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let b: std::mem::MaybeUninit<u8> = std::ptr::read(p);
        println!("val={}", b.assume_init());
        std::alloc::dealloc(p as *mut u8, layout);
    }
}
"#;

#[test]
fn run_mode_maybe_uninit_ptr_read_then_assume_init_errors() {
    // Reading the union out of unwritten storage succeeds, but
    // the value is still uninitialized: using its payload through
    // `assume_init` fails instead of fabricating a byte.
    let dir = temp_dir("maybe-uninit-alloc-read-assume");
    let src = fixture(&dir, "case.rs", MAYBE_UNINIT_ALLOC_READ_ASSUME_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
}
