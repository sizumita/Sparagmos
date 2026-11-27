use super::{
    CLOSURE_GENERIC_CONTEXT_SRC, fixture, run_driver, run_driver_extra, run_ok_edition,
    summarize_driver, temp_dir,
};

// The closure and coroutine run-mode tests are split by
// sub-feature into the files included below. `include!` keeps
// every test in this module, so test paths stay
// `closures_coroutines::<name>`.

include!("closures.rs");
include!("generic_closures.rs");
include!("sizes.rs");
include!("closure_clone.rs");
include!("convert_limits.rs");
include!("wakers.rs");
include!("coroutine_drop.rs");
include!("async_futures.rs");
include!("precise_captures.rs");
include!("drop_order.rs");
include!("misc.rs");
