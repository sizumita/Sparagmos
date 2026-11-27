//! Run-mode tests for the libc process / descriptor /
//! dynamic-loading models (Issue #513, census group
//! G3): `sysconf`, `getuid`, `getgid`, `getpid`,
//! `prctl`, `clearenv`, `dlopen` / `dlsym` over the
//! program's own exports, the descriptor calls (`dup`,
//! `dup2`, `close`, `pipe`, `fcntl`) on the descriptor
//! routing, and the helpers reached only under renamed
//! or versioned link symbols (`strlen`, the
//! `rust_dbg_extern_identity_u64` identity, `nftw` in
//! its 2.2.5 and 2.3.3 bindings). The `CommandExt`
//! credential setters the same records use are covered
//! too.
//!
//! The guest programs use libc through an auxiliary
//! crate (the driver's own sysroot pinning serves the
//! aux build, as in the driver_signals tests); the
//! renamed-symbol declarations live in the main crate,
//! mirroring the target records' shapes. On the base
//! tree every program here stops at its first modelled
//! call (no converted body, no model).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture(dir: &Path, name: &str, src: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, src).unwrap();
    path
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-libc-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run_driver_args(src: &Path, aux: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .arg("--edition")
        .arg("2021")
        .arg("--aux")
        .arg(format!("libcaux={}", aux.display()))
        .args(extra)
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to run sparagmos-driver")
}

fn run_ok(tag: &str, aux_src: &str, main_src: &str, expected_stdout: &str) {
    run_ok_args(tag, aux_src, main_src, &[], expected_stdout);
}

fn run_ok_args(
    tag: &str,
    aux_src: &str,
    main_src: &str,
    extra: &[&str],
    expected_stdout: &str,
) {
    let dir = temp_dir(tag);
    let aux = fixture(&dir, "libcaux.rs", aux_src);
    let src = fixture(&dir, "case.rs", main_src);
    let out = run_driver_args(&src, &aux, extra);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "case {tag} must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        expected_stdout,
        "case {tag} stdout mismatch; stderr={stderr}"
    );
}

/// An aux crate that only re-exports libc, for the
/// main-crate declaration tests.
const EMPTY_AUX: &str = r#"
#![feature(rustc_private)]
extern crate libc;
"#;

/// sysconf answers the host configuration, the identity
/// calls the host process's own credentials and id, and
/// prctl sets and reads back the dumpable flag on the
/// process the guest shares.
#[test]
fn libc_sysconf_identity_prctl() {
    let aux = r#"
#![feature(rustc_private)]
extern crate libc;

pub fn page_size() -> i64 {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as i64 }
}

pub fn ids() -> (u32, u32, i32) {
    unsafe { (libc::getuid(), libc::getgid(), libc::getpid()) }
}

pub fn prctl_roundtrip() -> (i32, i32, i32) {
    unsafe {
        let set = libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0);
        let got = libc::prctl(libc::PR_GET_DUMPABLE, 0, 0, 0, 0);
        let back = libc::prctl(libc::PR_SET_DUMPABLE, 1, 0, 0, 0);
        (set, got, back)
    }
}
"#;
    let main = r#"
extern crate libcaux;

fn main() {
    let page = libcaux::page_size();
    assert!(page >= 4096 && (page & (page - 1)) == 0, "page size {page}");
    let (uid, gid, pid) = libcaux::ids();
    assert!(pid > 0);
    let (uid2, gid2, pid2) = libcaux::ids();
    assert_eq!((uid, gid, pid), (uid2, gid2, pid2));
    assert_eq!(libcaux::prctl_roundtrip(), (0, 0, 0));
    println!("ids ok");
}
"#;
    run_ok("ids", aux, main, "ids ok\n");
}

/// clearenv empties the environment the guest reads:
/// the model's environment is the host environment
/// itself, so the clear is observed by `std::env`.
#[test]
fn libc_clearenv() {
    let aux = r#"
#![feature(rustc_private)]
extern crate libc;

pub fn clear() -> i32 {
    unsafe { libc::clearenv() }
}
"#;
    let main = r#"
extern crate libcaux;

fn main() {
    std::env::set_var("SPARAGMOS_LIBC_MARKER", "1");
    assert_eq!(std::env::var("SPARAGMOS_LIBC_MARKER").unwrap(), "1");
    assert_eq!(libcaux::clear(), 0);
    assert_eq!(std::env::vars().count(), 0);
    println!("clearenv ok");
}
"#;
    run_ok("clearenv", aux, main, "clearenv ok\n");
}

/// dlopen of the main program plus dlsym over the
/// program's own exports: the guest's `#[no_mangle]`
/// function is found and callable through the answered
/// pointer; a missing symbol answers null.
#[test]
fn libc_dlopen_dlsym() {
    let aux = r#"
#![feature(rustc_private)]
extern crate libc;

use std::ffi::c_void;

pub fn main_handle() -> *mut c_void {
    unsafe { libc::dlopen(std::ptr::null(), libc::RTLD_NOW) }
}

pub fn sym(handle: *mut c_void, name: &std::ffi::CStr) -> *mut c_void {
    unsafe { libc::dlsym(handle, name.as_ptr()) }
}
"#;
    let main = r#"
extern crate libcaux;

#[unsafe(no_mangle)]
pub extern "C" fn hack() -> u64 {
    998244353
}

fn main() {
    let handle = libcaux::main_handle();
    assert!(!handle.is_null());
    let found = libcaux::sym(handle, c"hack");
    assert!(!found.is_null());
    let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(found) };
    assert_eq!(f(), 998244353);
    let missing = libcaux::sym(handle, c"no_such_symbol_here");
    assert!(missing.is_null());
    println!("dlsym ok");
}
"#;
    run_ok("dlsym", aux, main, "dlsym ok\n");
}

/// The descriptor calls over a pipe: dup duplicates the
/// read end, the original closes, the duplicate reads
/// what the write end wrote; a second close and an
/// fcntl on the closed number fail with EBADF, and
/// dup2 installs the write end at a chosen number.
#[test]
fn libc_descriptor_calls() {
    let aux = r#"
#![feature(rustc_private)]
extern crate libc;

use std::ffi::c_void;

pub fn pipe() -> (i32, i32) {
    let mut fds = [0; 2];
    assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
    (fds[0], fds[1])
}

pub fn write(fd: i32, bytes: &[u8]) -> isize {
    unsafe {
        libc::write(fd, bytes.as_ptr() as *const c_void, bytes.len())
    }
}

pub fn read(fd: i32, buf: &mut [u8]) -> isize {
    unsafe {
        libc::read(fd, buf.as_mut_ptr() as *mut c_void, buf.len())
    }
}

pub fn dup(fd: i32) -> i32 {
    unsafe { libc::dup(fd) }
}

pub fn dup2(old: i32, new: i32) -> i32 {
    unsafe { libc::dup2(old, new) }
}

pub fn close(fd: i32) -> i32 {
    unsafe { libc::close(fd) }
}

pub fn fcntl_getfd(fd: i32) -> i32 {
    unsafe { libc::fcntl(fd, libc::F_GETFD) }
}

pub fn errno() -> i32 {
    unsafe { *libc::__errno_location() }
}
"#;
    let main = r#"
extern crate libcaux;

fn main() {
    let (reader, writer) = libcaux::pipe();
    assert_eq!(libcaux::write(writer, b"abc"), 3);
    // dup2 installs the write end at a chosen number;
    // writing through it lands in the same pipe.
    assert_eq!(libcaux::dup2(writer, 100), 100);
    assert_eq!(libcaux::write(100, b"z"), 1);
    assert_eq!(libcaux::close(100), 0);
    let duped = libcaux::dup(reader);
    assert!(duped >= 0 && duped != reader);
    assert_eq!(libcaux::close(reader), 0);
    let mut buf = [0u8; 4];
    assert_eq!(libcaux::read(duped, &mut buf), 4);
    assert_eq!(&buf, b"abcz");
    assert_eq!(libcaux::close(duped), 0);
    assert_eq!(libcaux::close(duped), -1);
    assert_eq!(libcaux::errno(), libc::EBADF_PLACEHOLDER);
    assert_eq!(libcaux::fcntl_getfd(duped), -1);
    assert_eq!(libcaux::fcntl_getfd(writer), 0);
    assert_eq!(libcaux::close(writer), 0);
    println!("descriptors ok");
}
"#;
    // The guest has no libc of its own; the EBADF value
    // is the host's 9, checked through the aux instead.
    let main = main.replace("libc::EBADF_PLACEHOLDER", "9");
    run_ok("descriptors", aux, &main, "descriptors ok\n");
}

/// `strlen` reached through a `#[link_name]` declaration
/// in the guest's own crate (the foreign-fn-linkname
/// shape): the symbol, not the Rust path, identifies
/// the model.
#[test]
fn strlen_via_link_name() {
    let main = r#"
extern crate libcaux;

extern "C" {
    #[link_name = "strlen"]
    fn my_strlen(s: *const std::ffi::c_char) -> usize;
}

fn main() {
    let s = std::ffi::CString::new("hello world").unwrap();
    let n = unsafe { my_strlen(s.as_ptr()) };
    assert_eq!(n, 11);
    println!("strlen ok");
}
"#;
    run_ok("strlen", EMPTY_AUX, main, "strlen ok\n");
}

/// The identity helper reached through its link symbol
/// by a declaration named after std's `black_box` (the
/// stack-probes shape): the call runs and returns.
#[test]
fn black_box_via_link_symbol() {
    let main = r#"
extern crate libcaux;

extern "C" {
    #[link_name = "rust_dbg_extern_identity_u64"]
    fn black_box(u: u64);
}

fn main() {
    unsafe { black_box(42) };
    println!("black box ok");
}
"#;
    run_ok("blackbox", EMPTY_AUX, main, "black box ok\n");
}

/// `nftw` under its versioned symbols: the 2.2.5
/// binding walks with an arbitrary flag word, the
/// 2.3.3 binding rejects the same word with EINVAL and
/// walks with a valid one. The walk visits the root
/// directory, a subdirectory and two files.
#[test]
fn nftw_via_versioned_symbols() {
    let main = r#"
extern crate libcaux;

use std::ffi::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

#[repr(C)]
struct Ftw {
    base: c_int,
    level: c_int,
}

extern "C" {
    #[link_name = "nftw@GLIBC_2.2.5"]
    fn nftw_2_2_5(
        dirpath: *const c_char,
        cb: extern "C" fn(*const c_char, *const c_void, c_int, *mut Ftw) -> c_int,
        nopenfd: c_int,
        flags: c_int,
    ) -> c_int;
    #[link_name = "nftw@GLIBC_2.3.3"]
    fn nftw_2_3_3(
        dirpath: *const c_char,
        cb: extern "C" fn(*const c_char, *const c_void, c_int, *mut Ftw) -> c_int,
        nopenfd: c_int,
        flags: c_int,
    ) -> c_int;
}

static COUNT: AtomicUsize = AtomicUsize::new(0);
static ROOT_KIND: AtomicUsize = AtomicUsize::new(99);

extern "C" fn count(
    _path: *const c_char,
    _sb: *const c_void,
    tflag: c_int,
    ftwbuf: *mut Ftw,
) -> c_int {
    let n = COUNT.fetch_add(1, Ordering::SeqCst);
    if n == 0 {
        ROOT_KIND.store(tflag as usize, Ordering::SeqCst);
        assert_eq!(unsafe { (*ftwbuf).level }, 0);
    }
    0
}

fn main() {
    let root = std::env::temp_dir().join(format!(
        "sparagmos-nftw-{}",
        std::process::id()
    ));
    let sub = root.join("sub");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(root.join("a.txt"), b"a").unwrap();
    std::fs::write(sub.join("b.txt"), b"b").unwrap();
    let path = std::ffi::CString::new(root.to_str().unwrap()).unwrap();

    COUNT.store(0, Ordering::SeqCst);
    let legacy = unsafe { nftw_2_2_5(path.as_ptr(), count, 20, 1 << 30) };
    assert_eq!(legacy, 0);
    assert_eq!(COUNT.load(Ordering::SeqCst), 4);
    assert_eq!(ROOT_KIND.load(Ordering::SeqCst), 1);

    COUNT.store(0, Ordering::SeqCst);
    let rejected = unsafe { nftw_2_3_3(path.as_ptr(), count, 20, 1 << 30) };
    assert_eq!(rejected, -1);
    assert_eq!(COUNT.load(Ordering::SeqCst), 0);

    let modern = unsafe { nftw_2_3_3(path.as_ptr(), count, 20, 0) };
    assert_eq!(modern, 0);
    assert_eq!(COUNT.load(Ordering::SeqCst), 4);

    std::fs::remove_dir_all(&root).unwrap();
    println!("nftw ok");
}
"#;
    run_ok("nftw", EMPTY_AUX, main, "nftw ok\n");
}

/// The `CommandExt` credential setters: a spawn with
/// the caller's own uid/gid succeeds; a spawn with
/// 65,537 supplementary groups fails the way the native
/// spawn fails; a spawn as uid 0 fails for a non-root
/// caller.
#[test]
fn command_ext_credentials() {
    let aux = r#"
#![feature(rustc_private)]
extern crate libc;

pub fn uid() -> u32 {
    unsafe { libc::getuid() }
}

pub fn gid() -> u32 {
    unsafe { libc::getgid() }
}
"#;
    let main = r#"
#![feature(setgroups)]

extern crate libcaux;

use std::os::unix::process::CommandExt;
use std::process::Command;

fn main() {
    let uid = libcaux::uid();
    let gid = libcaux::gid();
    let status = Command::new("/bin/true")
        .uid(uid)
        .gid(gid)
        .status()
        .expect("spawn with own credentials");
    assert!(status.success());

    let many: Vec<u32> = (0..65537).collect();
    let too_many = Command::new("/bin/true").groups(&many).spawn();
    assert!(too_many.is_err(), "65537 groups must fail the spawn");

    if uid != 0 {
        let as_root = Command::new("/bin/true").uid(0).spawn();
        assert!(as_root.is_err(), "uid 0 spawn must fail for non-root");
    }
    println!("credentials ok");
}
"#;
    run_ok_args(
        "credentials",
        aux,
        main,
        &["--guest-process-host"],
        "credentials ok\n",
    );
}
