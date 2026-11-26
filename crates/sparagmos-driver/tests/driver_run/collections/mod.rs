use super::{fixture, run_driver, run_driver_extra, run_ok, temp_dir};

// The collection and iterator run-mode tests are split by
// sub-feature into the files included below. `include!` keeps
// every test in this module, so test paths stay
// `collections::<name>`.

include!("btree.rs");
include!("misc.rs");
include!("iterators.rs");
include!("iter_mut.rs");
include!("hash.rs");
include!("defaults.rs");
include!("double_ended.rs");
include!("ranges.rs");
include!("vec_and_array.rs");
include!("linked_list.rs");
include!("vec_deque.rs");
include!("binary_heap.rs");
include!("from_impls.rs");