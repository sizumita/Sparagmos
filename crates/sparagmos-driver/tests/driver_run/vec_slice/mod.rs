use super::{fixture, run_driver, run_driver_extra, run_driver_with_aux, run_ok, summarize_driver, temp_dir};

// The vec and slice run-mode tests are split by
// sub-feature into the files included below. `include!` keeps
// every test in this module, so test paths stay
// `vec_slice::<name>`.

include!("smallvec.rs");
include!("vec_alloc.rs");
include!("boxed_slice.rs");
include!("range_views.rs");
include!("vec_mutation.rs");
include!("iterators.rs");
include!("scalar_backed.rs");
include!("chunks_windows.rs");
include!("sort.rs");
include!("o3.rs");
include!("subslice.rs");
include!("unsize.rs");
include!("uninit.rs");
include!("arrays.rs");
include!("misc.rs");
