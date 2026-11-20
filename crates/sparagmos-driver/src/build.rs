// Capture the sysroot of the rustc that builds this driver (the pinned
// 1.100.0 toolchain) so the resulting binary can (a) link against that
// sysroot's librustc_driver via an rpath and (b) pass the correct
// `--sysroot` to the compiler it drives, regardless of the caller's PATH.
use std::process::Command;

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let out = Command::new(&rustc)
        .args(["--print", "sysroot"])
        .output()
        .expect("failed to run rustc --print sysroot");
    assert!(out.status.success(), "rustc --print sysroot failed");
    let sysroot = String::from_utf8(out.stdout).unwrap().trim().to_string();
    println!("cargo:rustc-env=SPARAGMOS_BUILD_SYSROOT={sysroot}");
    let vout = Command::new(&rustc)
        .args(["--version"])
        .output()
        .expect("failed to run rustc --version");
    let version = String::from_utf8(vout.stdout).unwrap().trim().to_string();
    println!("cargo:rustc-env=SPARAGMOS_BUILD_RUSTC_VERSION={version}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{sysroot}/lib");
    // librustc_driver needs libLLVM, which lives in <sysroot>/lib.
    println!("cargo:rustc-link-search=native={sysroot}/lib");
    println!("cargo:rerun-if-changed=build.rs");
}
