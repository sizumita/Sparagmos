//! The interpreter's debugger interface: the machinery the
//! debuginfo test harness drives instead of a real debugger.
//!
//! A real debugger reads DWARF — type and line information out of
//! a compiled binary's debug sections — which this runtime never
//! produces (it executes MIR directly; there is no binary). What a
//! debugger session actually needs from the execution engine is
//! small, though, and the interpreter can provide it directly:
//! line breakpoints, run/continue, step/next/finish, and reading
//! the current frame's locals rendered the way the debugger would
//! print them. That is the interface here.
//!
//! A [`DebugSession`] wraps one converted program. The caller sets
//! breakpoints (by source file and line, from the test file's
//! `#break` markers, as compiletest does) and hands over the
//! session's source text (for rendering stop lines), then runs a
//! script of [`DebugCommand`]s. Execution is a single ordinary
//! interpreter run with a hook installed: the drive loop consults
//! the hook at every source-line change (see
//! [`crate::interp::Interpreter::run`]'s loop and `drive_hook`
//! below), and the hook consumes script commands at each stop —
//! printing values into the transcript — until a resuming command
//! sets the next stop plan. With no session installed the hook is
//! absent and ordinary runs are untouched.
//!
//! The transcript is the debugger's output, in the shape the
//! harness's check lines expect: a stop emits the source line it
//! stopped on; a `print` emits `$N = <value>` (gdb form) or
//! `= <value>` (lldb form, whose check lines are loose fragments).
//! Values render as the dialect prints them: scalars (integers,
//! booleans, floats, chars, unit) and aggregates — tuples, arrays
//! and ADT values, the last through the program's debug-shape
//! table — reached through locals, dereferences and field/index
//! projections.

mod desc;
mod evaluate;
mod parse;
mod printers;
mod queries;
mod render_gdb;
mod render_lldb;
mod session;

// ---------------------------------------------------------------------------
// Render budget (Issue #480)
//
// The renderers' depth cap (see `render_gdb::MAX_RENDER_DEPTH`)
// bounds how *deep* one print unfolds, but not how much work it
// does in total: a value whose rendering branches -- a union
// read back through every declared field, a tuple tree that
// doubles at every level, a collection provider walking its
// storage -- multiplies the number of rendered nodes at each
// level, so a shallow value can still expand without bound
// (the debugger-session form of the balloon in Issue #480:
// the process grows while a single print is being rendered).
// A print therefore runs under a total node budget: every node
// the gdb / lldb / pretty-printer renderers unfold spends one
// unit, and a print that spends the whole budget fails as
// unrenderable -- the session reports it with its ordinary
// `<error: ... could not be rendered>` line -- instead of
// ballooning, and instead of emitting a truncated rendering
// that a check line could mistake for the value. The budget
// is sized far past any rendering the debuginfo suite's check
// lines expect (their prints are bounded by gdb's own
// `print elements` in pretty mode and stay in the thousands of
// nodes in raw mode), so legitimate prints never reach it.
// ---------------------------------------------------------------------------

/// The most value nodes one print may unfold across all of
/// its renderers (see the module note above).
pub(super) const MAX_RENDER_NODES: usize = 100_000;

thread_local! {
    /// Nodes left in the active print's render budget;
    /// meaningful only while `RENDER_BUDGET_ACTIVE` is set.
    static RENDER_NODES_LEFT: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    /// Whether a render budget is active on this thread.
    static RENDER_BUDGET_ACTIVE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Activate a fresh render budget for one print; the returned
/// guard restores the previous state when it drops, so a
/// print rendered while no session print is in progress (a
/// direct renderer call in a unit test, say) stays unbudgeted.
pub(super) fn begin_render_budget() -> RenderBudgetGuard {
    let prev_active = RENDER_BUDGET_ACTIVE.with(|a| a.replace(true));
    let prev_left = RENDER_NODES_LEFT.with(|n| n.replace(MAX_RENDER_NODES));
    RenderBudgetGuard {
        prev_active,
        prev_left,
    }
}

/// The RAII handle of [`begin_render_budget`].
pub(super) struct RenderBudgetGuard {
    prev_active: bool,
    prev_left: usize,
}

impl Drop for RenderBudgetGuard {
    fn drop(&mut self) {
        RENDER_BUDGET_ACTIVE.with(|a| a.set(self.prev_active));
        RENDER_NODES_LEFT.with(|n| n.set(self.prev_left));
    }
}

/// Spend one node of the active render budget: `false` when
/// the budget is spent, which the renderers treat exactly
/// like exceeding the depth cap (the value is unrenderable,
/// never truncated). With no budget active every node is
/// free.
pub(super) fn spend_render_node() -> bool {
    if !RENDER_BUDGET_ACTIVE.with(|a| a.get()) {
        return true;
    }
    RENDER_NODES_LEFT.with(|n| {
        let left = n.get();
        if left == 0 {
            false
        } else {
            n.set(left - 1);
            true
        }
    })
}


pub use self::session::{
    DebugCommand, DebugDialect, DebugPrint, DebugRun, DebugSession, LldbFormats, LldbNumFormat,
    PrintFormat,
};
pub(crate) use self::session::{drive_hook, DebugControl};
