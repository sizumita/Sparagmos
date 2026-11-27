//! Host libc functions backing the libc builtins.

use super::*;





extern "C" {
    /// The host C library's `atol`, backing
    /// [`Builtin::LibcAtol`]: the guest's extern declaration
    /// names this symbol, so the host function itself is the
    /// faithful execution (leading whitespace, sign, digit
    /// prefix and the implementation's overflow behaviour
    /// included).
    #[link_name = "atol"]
    pub(super) fn host_atol(s: *const std::ffi::c_char) -> std::ffi::c_long;
    /// The host C library's `atoll`, backing
    /// [`Builtin::LibcAtoll`].
    #[link_name = "atoll"]
    pub(super) fn host_atoll(s: *const std::ffi::c_char) -> std::ffi::c_longlong;
    /// The host C library's `atoi`, backing
    /// [`Builtin::LibcAtoi`].
    #[link_name = "atoi"]
    pub(super) fn host_atoi(s: *const std::ffi::c_char) -> std::ffi::c_int;
    /// The host C library's `strtol`, backing
    /// [`Builtin::LibcStrtol`]; only ever called with a NULL
    /// `endptr` (the builtin rejects a non-NULL one).
    #[link_name = "strtol"]
    pub(super) fn host_strtol(
        s: *const std::ffi::c_char,
        endptr: *mut *mut std::ffi::c_char,
        base: std::ffi::c_int,
    ) -> std::ffi::c_long;
    /// The host C library's `strtoll`, backing
    /// [`Builtin::LibcStrtoll`]; NULL `endptr` only, as
    /// `host_strtol`.
    #[link_name = "strtoll"]
    pub(super) fn host_strtoll(
        s: *const std::ffi::c_char,
        endptr: *mut *mut std::ffi::c_char,
        base: std::ffi::c_int,
    ) -> std::ffi::c_longlong;
    /// The host C library's `strtoul`, backing
    /// [`Builtin::LibcStrtoul`]; NULL `endptr` only, as
    /// `host_strtol`.
    #[link_name = "strtoul"]
    pub(super) fn host_strtoul(
        s: *const std::ffi::c_char,
        endptr: *mut *mut std::ffi::c_char,
        base: std::ffi::c_int,
    ) -> std::ffi::c_ulong;
    /// The host C library's `strtoull`, backing
    /// [`Builtin::LibcStrtoull`]; NULL `endptr` only, as
    /// `host_strtol`.
    #[link_name = "strtoull"]
    pub(super) fn host_strtoull(
        s: *const std::ffi::c_char,
        endptr: *mut *mut std::ffi::c_char,
        base: std::ffi::c_int,
    ) -> std::ffi::c_ulonglong;
    /// The host C library's `lgamma_r`, backing
    /// [`Builtin::LgammaR`]: the guest's extern declaration
    /// names this symbol, so the host function itself is the
    /// faithful execution.
    #[link_name = "lgamma_r"]
    pub(super) fn host_lgamma_r(x: f64, signp: *mut std::ffi::c_int) -> f64;
    /// The host C library's `sysconf`, backing
    /// [`Builtin::LibcSysconf`]: the guest runs
    /// in-process, so the host's configuration values
    /// (page size, group maximum, ...) are the values the
    /// guest observes.
    #[link_name = "sysconf"]
    pub(super) fn host_sysconf(name: std::ffi::c_int) -> std::ffi::c_long;
    /// The host C library's `getuid`, backing
    /// [`Builtin::LibcGetuid`]: this process's real user
    /// id, the identity the in-process guest runs with.
    #[link_name = "getuid"]
    pub(super) fn host_getuid() -> std::ffi::c_uint;
    /// The host C library's `getgid`, backing
    /// [`Builtin::LibcGetgid`]: this process's real group
    /// id.
    #[link_name = "getgid"]
    pub(super) fn host_getgid() -> std::ffi::c_uint;
}

/// Registration fragment of the builtin name registry: the libc allocator surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn libc_alloc_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "libc::unix::malloc" => Builtin::LibcMalloc,
        "libc::unix::calloc" => Builtin::LibcCalloc,
        "libc::unix::free" => Builtin::LibcFree,
        "libc::unix::realloc" => Builtin::LibcRealloc,
        "libc::unix::posix_memalign" => Builtin::LibcPosixMemalign,
        _ => return None,
    })
}
