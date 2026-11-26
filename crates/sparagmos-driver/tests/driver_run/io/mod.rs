use super::{Command, fixture, run_driver, run_ok, temp_dir};

// The io run-mode tests are split by sub-feature into the
// files included below. `include!` keeps every test in this
// module, so test paths stay `io::<name>`.

include!("printing.rs");
include!("error.rs");
include!("write.rs");
include!("stdin.rs");
include!("misc.rs");
