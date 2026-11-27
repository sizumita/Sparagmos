//! Process spawning, child processes and stdio configuration.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    Frame, Interpreter,
};
use crate::value::Value;

use super::*;

impl<'a> Interpreter<'a> {
    /// The command id a process-command receiver denotes: the
    /// receiver is a reference to the handle value the
    /// `Command::new` builtin returned (possibly reached through
    /// the public `Command`'s `inner` field projection).
    pub(super) fn command_id_of(
        &self,
        frames: &[Frame],
        value: &Value,
        name: &str,
    ) -> InterpResult<u64> {
        let resolved = self.resolve_value(frames, value)?;
        command_handle_id(&resolved).ok_or_else(|| {
            InterpError::type_mismatch(
                format!("builtin {name}"),
                "a process command handle",
                resolved.kind_name(),
            )
        })
    }

    /// The child-table id of the `Child` handle a builtin
    /// call received (as receiver, by reference or by value).
    pub(super) fn child_id_of(
        &self,
        frames: &[Frame],
        value: &Value,
        name: &str,
    ) -> InterpResult<u64> {
        let resolved = self.resolve_value(frames, value)?;
        child_handle_id(&resolved).ok_or_else(|| {
            InterpError::type_mismatch(
                format!("builtin {name}"),
                "a process child handle",
                resolved.kind_name(),
            )
        })
    }

    /// The barrier-table id a `Barrier` handle value
    /// denotes: resolve the reference and read the id out of
    /// the handle (see `barrier_value`).
    pub(super) fn barrier_id_of(
        &self,
        frames: &[Frame],
        value: &Value,
        name: &str,
    ) -> InterpResult<u64> {
        let target = self.deref_value(frames, value)?;
        let resolved = self.read_resolved(frames, &target)?;
        barrier_handle_id(&resolved).ok_or_else(|| {
            InterpError::type_mismatch(
                format!("builtin {name}"),
                "a barrier handle",
                resolved.kind_name(),
            )
        })
    }

    /// Run this command to completion through the process
    /// host, in the shape of a tool outcome (see
    /// `tool_outcome`). `None` when no host is configured (the
    /// caller keeps the standing answer); `Some(Err(value))`
    /// when starting the child failed the way std reports it
    /// (the value is the guest's `io::Error`).
    pub(super) fn host_outcome(
        &self,
        state: &crate::interp::CommandState,
        output_captured: bool,
    ) -> InterpResult<Option<Result<crate::interp::ToolRunOutcome, Value>>> {
        let Some(host) = self.process_host() else {
            return Ok(None);
        };
        check_host_stdio(state)?;
        let files = self.stdio_files(state)?;
        let request =
            process_spawn_request(state, output_captured, files, self.child_sigpipe());
        match host.0.run(request) {
            Ok(outcome) => {
                self.close_wired_stdin(state);
                Ok(Some(Ok(crate::interp::ToolRunOutcome {
                    exit_code: outcome.exit_code,
                    stdout: outcome.stdout,
                    stderr: outcome.stderr,
                })))
            }
            Err(crate::interp::ProcessHostError::Io(e)) => {
                Ok(Some(Err(host_io_error(&e))))
            }
            Err(crate::interp::ProcessHostError::Failed(reason)) => {
                Err(unsupported(reason))
            }
        }
    }

    /// The outcome of running this command to completion,
    /// for the entry points that collect one (`status` /
    /// `output`): the tool interception answers first -- a
    /// named tool program through its runner, any other
    /// program through the fallback (the artifacts this run
    /// produced) -- and a program the fallback cannot carry
    /// is the process host's to serve when the run
    /// configured one, the same consult-interception-first,
    /// host-for-the-rest order `spawn` uses (which never
    /// consults the interception at all). A named program's
    /// runner error is authoritative (the tool ran and
    /// reported it cannot be carried out); the fallback's
    /// rejection of a program it does not know is not --
    /// with no host configured it stands as the run's
    /// unsupported error, exactly as before.
    pub(super) fn tool_or_host_outcome(
        &self,
        state: &crate::interp::CommandState,
        output_captured: bool,
    ) -> InterpResult<Option<Result<crate::interp::ToolRunOutcome, Value>>> {
        match tool_outcome(self.tool_interception(), state, output_captured) {
            Ok(Some(outcome)) => Ok(Some(Ok(outcome))),
            Ok(None) => self.host_outcome(state, output_captured),
            Err(e) => {
                let named = self.tool_interception().is_some_and(
                    |interception| {
                        interception
                            .programs
                            .iter()
                            .any(|p| p == &state.program)
                    },
                );
                if !named && self.process_host().is_some() {
                    match self.host_outcome(state, output_captured)? {
                        Some(outcome) => Ok(Some(outcome)),
                        None => Err(e),
                    }
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Start this command's program through the process host
    /// and register the child, returning the guest's
    /// `io::Result<Child>` value. `None` when no host is
    /// configured (the caller keeps the standing answer).
    pub(super) fn host_spawn(
        &self,
        state: &crate::interp::CommandState,
    ) -> InterpResult<Option<Value>> {
        let Some(host) = self.process_host() else {
            return Ok(None);
        };
        check_host_stdio(state)?;
        let files = self.stdio_files(state)?;
        let request =
            process_spawn_request(state, false, files, self.child_sigpipe());
        match host.0.spawn(request) {
            Ok(spawned) => {
                self.close_wired_stdin(state);
                // The child's pipe ends join the file table;
                // the guest's `Child` value carries a handle
                // value for each stream it configured piped
                // (the only slots std fills), while the entry
                // keeps every end for collection and relay.
                let stdin_file =
                    spawned.stdin.map(|file| self.register_file(file));
                let stdout_file =
                    spawned.stdout.map(|file| self.register_file(file));
                let stderr_file =
                    spawned.stderr.map(|file| self.register_file(file));
                let stdin_slot = (state.stdin_stdio
                    == Some(crate::interp::StdioConfig::Piped))
                .then_some(stdin_file)
                .flatten();
                let stdout_slot = (state.stdout_stdio
                    == Some(crate::interp::StdioConfig::Piped))
                .then_some(stdout_file)
                .flatten();
                let stderr_slot = (state.stderr_stdio
                    == Some(crate::interp::StdioConfig::Piped))
                .then_some(stderr_file)
                .flatten();
                let id = self.new_child(crate::interp::ChildState {
                    host_id: spawned.host_id,
                    pid: spawned.pid,
                    stdout_stdio: state.stdout_stdio,
                    stderr_stdio: state.stderr_stdio,
                    stdin_file,
                    stdout_file,
                    stderr_file,
                    finished: None,
                    relayed: false,
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    drains: Vec::new(),
                });
                Ok(Some(ok_result(child_value(
                    id,
                    stdin_slot,
                    stdout_slot,
                    stderr_slot,
                ))))
            }
            Err(crate::interp::ProcessHostError::Io(e)) => {
                Ok(Some(err_result(host_io_error(&e))))
            }
            Err(crate::interp::ProcessHostError::Failed(reason)) => {
                Err(unsupported(reason))
            }
        }
    }

    /// Resolve the host files a command's file-form stdio
    /// configurations name: a duplicate of each named
    /// file-table entry, in stdin/stdout/stderr order, for
    /// the spawn request to wire into the child (see
    /// `ProcessStdio::Fd`). A configuration naming a file
    /// the run does not hold -- a handle closed already, or
    /// a fabricated descriptor number that names no entry --
    /// makes the invocation unsupported: running it with a
    /// guessed wiring would wire the wrong stream.
    fn stdio_files(
        &self,
        state: &crate::interp::CommandState,
    ) -> InterpResult<[Option<std::fs::File>; 3]> {
        let mut files = [None, None, None];
        for (slot, config) in files.iter_mut().zip([
            state.stdin_stdio,
            state.stdout_stdio,
            state.stderr_stdio,
        ]) {
            if let Some(crate::interp::StdioConfig::File(id)) = config {
                // The configuration names a guest
                // descriptor: what it denotes is the
                // descriptor routing's current answer (a
                // `from_raw_fd` handle over a descriptor
                // the guest rerouted with `dup2` denotes
                // the reroute's file, not the number's
                // default).
                *slot = Some(self.desc_file_clone(id).ok_or_else(
                    || {
                        unsupported(
                            "a stdio configuration naming a file this run does not hold"
                                .to_string(),
                        )
                    },
                )?);
            }
        }
        Ok(files)
    }

    /// Close the file-table entry a command's stdin
    /// configuration named, once a spawn has taken the
    /// wiring: the child holds its own duplicate of the
    /// descriptor, and the builder's `Stdio` -- the
    /// parent-side owner, whose drop the model does not
    /// observe -- must not keep the parent's end open. This
    /// is what delivers EPIPE to a producer whose stdout
    /// handle the guest handed to a consumer's stdin: once
    /// the consumer is spawned and later exits, no reader
    /// remains and the producer's writes meet a broken
    /// pipe, as they do natively once the builder is gone.
    fn close_wired_stdin(&self, state: &crate::interp::CommandState) {
        if let Some(crate::interp::StdioConfig::File(id)) =
            state.stdin_stdio
        {
            if let Some(entry) = self.desc_entry_id(id) {
                self.file_handle_closed(entry);
            }
        }
    }

    /// Write bytes a child's stream produced into the file
    /// its stdio configuration names (see `StdioRoute::File`):
    /// the stream was redirected to that file, so its bytes
    /// land there and nowhere else. A file the run no longer
    /// holds, or a write the host refuses, is not the
    /// parent's error: a native child's redirected stream
    /// fails on the child's side, never in the parent's
    /// wait.
    pub(super) fn write_stdio_file(
        &self,
        config: Option<crate::interp::StdioConfig>,
        bytes: &[u8],
    ) {
        if let Some(crate::interp::StdioConfig::File(id)) = config {
            if let Some(entry) = self.desc_entry_id(id) {
                self.with_file(entry, |file| {
                    use std::io::Write as _;
                    let _ = file.write_all(bytes);
                });
            }
        }
    }

    /// Which of the guest `Child` value's stdio slots still
    /// hold their handle (a `Some` slot the guest has not
    /// taken out). The blocking child operations act on the
    /// handles the `Child` itself still holds -- std drops
    /// its stdin handle before waiting, and `wait_with_output`
    /// collects the streams it still holds -- so they consult
    /// the receiver value's slots.
    pub(super) fn child_slots_of(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<(bool, bool, bool)> {
        let resolved = self.resolve_value(frames, value)?;
        let present =
            |slot: &Value| matches!(slot, Value::Enum { variant: 1, .. });
        Ok(match &resolved {
            Value::Enum { fields, .. } if fields.len() == 4 => (
                present(&fields[1]),
                present(&fields[2]),
                present(&fields[3]),
            ),
            _ => (false, false, false),
        })
    }

    /// Start collecting the child's stream bytes on drain
    /// threads for the streams this observation needs: an
    /// inherited (relayed) stream, and a piped stream whose
    /// handle the `Child` still holds (a taken stream belongs
    /// to the guest, which reads it itself). Collection runs
    /// alongside the child, so a child that fills its pipes
    /// is never starved by a parent blocked in `wait`, as
    /// std's own collection runs alongside.
    ///
    /// A stream whose source is a regular file -- a
    /// re-executed child's channelled stderr -- is collected
    /// only once the child is finished (`child_finished`):
    /// a regular file does not hold a reader at its current
    /// end the way a pipe does, so a drain started while the
    /// writer still runs would gather only the prefix
    /// written so far and report the stream drained. Once
    /// the child is finished the file's content is complete.
    fn child_start_drains(
        &self,
        id: u64,
        slots: (bool, bool, bool),
        child_finished: bool,
    ) {
        let Some((out_file, err_file, out_cfg, err_cfg, started)) =
            self.with_child(id, |state| {
                (
                    state.stdout_file,
                    state.stderr_file,
                    state.stdout_stdio,
                    state.stderr_stdio,
                    (
                        state.drains.iter().any(|(is_err, _)| !*is_err),
                        state.drains.iter().any(|(is_err, _)| *is_err),
                    ),
                )
            })
        else {
            return;
        };
        let mut drains = Vec::new();
        for (is_err, file, cfg, slot, already) in [
            (false, out_file, out_cfg, slots.1, started.0),
            (true, err_file, err_cfg, slots.2, started.1),
        ] {
            let route = stdio_route(cfg, false);
            let wanted = route == StdioRoute::Relay
                || route == StdioRoute::File
                || (cfg == Some(crate::interp::StdioConfig::Piped) && slot);
            if already || !wanted {
                continue;
            }
            let Some(file_id) = file else { continue };
            let regular_file = self
                .with_file(file_id, |file| {
                    file.metadata()
                        .map(|meta| meta.file_type().is_file())
                        .unwrap_or(false)
                })
                .unwrap_or(false);
            if regular_file && !child_finished {
                continue;
            }
            let Some(Ok(mut clone)) =
                self.with_file(file_id, |file| file.try_clone())
            else {
                continue;
            };
            drains.push((
                is_err,
                std::thread::spawn(move || {
                    let mut bytes = Vec::new();
                    let _ =
                        std::io::Read::read_to_end(&mut clone, &mut bytes);
                    bytes
                }),
            ));
        }
        if !drains.is_empty() {
            self.with_child(id, |state| state.drains.extend(drains));
        }
    }

    /// Move finished drain threads' bytes into the child's
    /// collected buffers; with `join_all`, wait for every
    /// drain (a blocking call's collection completes there).
    /// Returns whether no drain remains in flight.
    fn child_pump_drains(&self, id: u64, join_all: bool) -> bool {
        let mut handles = Vec::new();
        let remaining = self
            .with_child(id, |state| {
                let mut kept = Vec::new();
                for drain in std::mem::take(&mut state.drains) {
                    if join_all || drain.1.is_finished() {
                        handles.push(drain);
                    } else {
                        kept.push(drain);
                    }
                }
                let remaining = !kept.is_empty();
                state.drains = kept;
                remaining
            })
            .unwrap_or(false);
        for (is_err, handle) in handles {
            let bytes = handle.join().unwrap_or_default();
            self.with_child(id, |state| {
                if is_err {
                    state.stderr.extend_from_slice(&bytes);
                } else {
                    state.stdout.extend_from_slice(&bytes);
                }
            });
        }
        !remaining
    }

    /// Relay the child's inherited streams into the guest's
    /// own streams, once their collected bytes are complete:
    /// the end observed and every drain collected.
    fn child_relay(&mut self, id: u64) {
        let Some((finished, relayed, drained, out_cfg, err_cfg)) =
            self.with_child(id, |state| {
                (
                    state.finished.is_some(),
                    state.relayed,
                    state.drains.is_empty(),
                    state.stdout_stdio,
                    state.stderr_stdio,
                )
            })
        else {
            return;
        };
        if !finished || relayed || !drained {
            return;
        }
        let (stdout, stderr) = self
            .with_child(id, |state| {
                (state.stdout.clone(), state.stderr.clone())
            })
            .expect("child entry present");
        match stdio_route(out_cfg, false) {
            StdioRoute::Relay => {
                self.push_output(&String::from_utf8_lossy(&stdout));
            }
            StdioRoute::File => self.write_stdio_file(out_cfg, &stdout),
            _ => {}
        }
        match stdio_route(err_cfg, false) {
            StdioRoute::Relay => {
                self.push_error_output(&String::from_utf8_lossy(&stderr));
            }
            StdioRoute::File => self.write_stdio_file(err_cfg, &stderr),
            _ => {}
        }
        self.with_child(id, |state| state.relayed = true);
    }

    /// Observe the end of the child with the given id,
    /// waiting for it when `block` is set. A blocking call
    /// first closes the stdin end the `Child` still holds
    /// (std's `wait` / `wait_with_output` drop that handle
    /// before waiting, so a child reading its stdin to the
    /// end sees EOF) and starts collecting the streams the
    /// `Child` still holds; on the first observation of the
    /// end, the inherited streams are relayed into the
    /// guest's own streams once their bytes are collected,
    /// and the exit code is cached, so every later
    /// observation answers with the same status, as std's
    /// cached status does.
    pub(super) fn child_finish(
        &mut self,
        id: u64,
        block: bool,
        slots: (bool, bool, bool),
    ) -> InterpResult<ChildWait> {
        let Some((host_id, finished)) = self.with_child(id, |state| {
            (state.host_id, state.finished)
        }) else {
            return Err(unsupported(
                "call on a stale child handle".to_string(),
            ));
        };
        if block && slots.0 {
            if let Some(stdin_file) = self
                .with_child(id, |state| state.stdin_file.take())
                .flatten()
            {
                self.file_handle_closed(stdin_file);
            }
        }
        if let Some(code) = finished {
            self.child_start_drains(id, slots, true);
            self.child_pump_drains(id, block);
            self.child_relay(id);
            return Ok(ChildWait::Finished(code));
        }
        let host = self.process_host().cloned().ok_or_else(|| {
            unsupported("child process without a process host".to_string())
        })?;
        if block {
            self.child_start_drains(id, slots, false);
        }
        let outcome = if block {
            host.0.wait(host_id).map(Some)
        } else {
            host.0.try_wait(host_id)
        };
        let code = match outcome {
            Ok(Some(code)) => code,
            Ok(None) => return Ok(ChildWait::Running),
            Err(crate::interp::ProcessHostError::Io(e)) => {
                return Ok(ChildWait::IoError(host_io_error(&e)));
            }
            Err(crate::interp::ProcessHostError::Failed(reason)) => {
                return Err(unsupported(reason));
            }
        };
        self.with_child(id, |state| state.finished = Some(code));
        self.child_start_drains(id, slots, true);
        self.child_pump_drains(id, block);
        self.child_relay(id);
        Ok(ChildWait::Finished(code))
    }
}

/// `Ok(value)` in the value model.
/// The initial builder state for a command over `program`.
pub(super) fn command_state_new(program: Vec<u8>) -> crate::interp::CommandState {
    crate::interp::CommandState {
        saw_nul: program.contains(&0),
        program,
        args: Vec::new(),
        argv0: None,
        envs: Vec::new(),
        env_clear: false,
        cwd: None,
        stdout_stdio: None,
        stderr_stdio: None,
        stdin_stdio: None,
        uid: None,
        gid: None,
        groups: None,
    }
}

/// The value a sys `Command` has in the interpreter: an
/// aggregate in the real layout's first three field positions.
/// Field 0 (the `program` slot) carries the command id -- the
/// program's bytes live in the command table ([`CommandState`]),
/// not in the value. Field 1 (the `args` slot) is an empty
/// placeholder; the arguments live in the table too. Field 2
/// (the `env` slot) carries the environment-editor handle for
/// the same command: std bodies inlined into the caller form
/// the `&mut command.env` place directly (`Command::envs`
/// reaches `CommandEnv::set` through `env_mut()`, whose body is
/// `&mut self.env`), and the `CommandEnv` builtins resolve that
/// place's value back to the command -- so the slot must hold
/// an initialised value denoting the command, exactly what
/// `ProcessCommandEnvMut` fabricates for the non-inlined path.
/// The remaining fields are never materialised: no modelled
/// path projects them, and the one-field handle this replaces
/// was already short of the real layout in the same way.
pub(super) fn command_value(id: u64) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Uint(u128::from(id)), Value::Unit, id_handle(id)],
    }
}

/// The command id a handle value carries, at whichever nesting
/// the value arrives with (the bare handle, the sys `Command`
/// aggregate, or the aggregate inside the public `Command`).
fn command_handle_id(value: &Value) -> Option<u64> {
    let mut cur = value;
    loop {
        match cur {
            Value::Uint(id) => return Some(*id as u64),
            Value::Enum { fields, .. } if fields.len() == 1 => {
                cur = &fields[0];
            }
            // The sys `Command` aggregate (`command_value`): its
            // first field carries the id directly.
            Value::Enum { fields, .. }
                if matches!(fields.first(), Some(Value::Uint(_))) =>
            {
                cur = &fields[0];
            }
            _ => return None,
        }
    }
}

pub(super) const ERROR_KIND_INVALID_INPUT: u32 = 20;
/// The `errno` for a missing file or directory.
const ERRNO_NOT_FOUND: i32 = 2;
/// The `errno` for a permission failure.
const ERRNO_PERMISSION_DENIED: i32 = 13;
/// The `errno` for a path component that is not a directory.
const ERRNO_NOT_A_DIRECTORY: i32 = 20;

/// Whether `candidate` could be executed as a program: it must
/// exist, not be a directory, and carry an execute bit. The
/// error is the `errno` the native spawn reports for it.
fn check_executable(candidate: &std::path::Path) -> Result<(), i32> {
    match std::fs::metadata(candidate) {
        Ok(m) if m.is_dir() => Err(ERRNO_PERMISSION_DENIED),
        Ok(m) if !host_executable(&m) => Err(ERRNO_PERMISSION_DENIED),
        Ok(_) => Ok(()),
        Err(_) => Err(ERRNO_NOT_FOUND),
    }
}

/// Whether the host metadata carries an execute
/// permission: the Unix mode's execute bits. Windows has
/// no execute bit — a non-directory file is executable as
/// far as this gate can honestly tell (the gate only
/// vetoes spawns that would fail natively; a program that
/// would start is reported as a named unsupported error
/// by the caller).
#[cfg(unix)]
fn host_executable(md: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt as _;
    md.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn host_executable(_md: &std::fs::Metadata) -> bool {
    true
}

/// Resolve a command's program the way `execvp` does at spawn:
/// a program containing a path separator is used as-is
/// (relative to the child's working directory when one is set);
/// any other program is searched along the effective `PATH`
/// (the command's own `PATH` entry wins over the inherited one,
/// as in std's environment capture). `Ok` is the path the
/// search settled on -- the program the invocation denotes,
/// which a bare name under a command-specific `PATH` owes to
/// the search, not to the working directory; `Err` is the
/// `errno` the failed spawn reports.
fn resolve_program(state: &crate::interp::CommandState) -> Result<std::path::PathBuf, i32> {
    // A program name with no host string form on this
    // platform names no file the host could execute: the
    // resolution fails as not-found.
    let Some(program) = crate::platform::os_str_from_bytes(&state.program)
    else {
        return Err(ERRNO_NOT_FOUND);
    };
    if state.program.contains(&b'/') {
        let candidate = if state.program.starts_with(b"/") {
            std::path::PathBuf::from(program)
        } else {
            match &state.cwd {
                Some(cwd) => {
                    let Some(cwd_os) =
                        crate::platform::os_str_from_bytes(cwd)
                    else {
                        return Err(ERRNO_NOT_FOUND);
                    };
                    std::path::Path::new(cwd_os).join(program)
                }
                None => std::path::PathBuf::from(program),
            }
        };
        return check_executable(&candidate).map(|()| candidate);
    }
    // The effective PATH: the inherited environment with the
    // command's entries applied in order.
    let mut path_var: Option<Vec<u8>> = std::env::var_os("PATH")
        .and_then(|v| {
            crate::platform::os_str_bytes(&v).map(|b| b.into_owned())
        });
    if state.env_clear {
        path_var = None;
    }
    for (key, value) in &state.envs {
        if key == b"PATH" {
            path_var = value.clone();
        }
    }
    let search: Vec<u8> = path_var.unwrap_or_else(|| b"/bin:/usr/bin".to_vec());
    let mut denied = false;
    let Some(search_os) = crate::platform::os_str_from_bytes(&search)
    else {
        return Err(ERRNO_NOT_FOUND);
    };
    for dir in std::env::split_paths(search_os) {
        let candidate = dir.join(program);
        match check_executable(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(ERRNO_PERMISSION_DENIED) => denied = true,
            Err(_) => {}
        }
    }
    Err(if denied {
        ERRNO_PERMISSION_DENIED
    } else {
        ERRNO_NOT_FOUND
    })
}

/// The `io::Error` a spawn of this command fails with, if it
/// fails before any program starts: a NUL byte in the program
/// or an argument is std's `InvalidInput` const error, a bad
/// working directory fails the spawn's `chdir`, and an
/// unresolvable program fails the `exec`. `None` means the
/// program would start — which the process model does not do,
/// so the caller reports it as a named unsupported error.
/// The spawn gate for a command: the `io::Error` the spawn
/// fails with, if it fails at all. A program the run's tool
/// interception claims by name is exempt from the filesystem
/// resolution `process_spawn_error` performs: the host side
/// has taken responsibility for that program, so whether it
/// can start is the host's call. The claimed name in a plain
/// run is the guest's own program -- a source path, which the
/// filesystem check would refuse as not executable even
/// though the host runs it by re-interpreting the source.
pub(super) fn spawn_gate_error(
    interception: Option<&crate::interp::ToolInterception>,
    state: &crate::interp::CommandState,
) -> Option<Value> {
    if let Some(interception) = interception {
        if interception.programs.iter().any(|p| p == &state.program) {
            return None;
        }
    }
    process_spawn_error(state)
}

fn process_spawn_error(state: &crate::interp::CommandState) -> Option<Value> {
    if state.saw_nul {
        return Some(io_simple_error_value(ERROR_KIND_INVALID_INPUT));
    }
    if let Some(cwd) = &state.cwd {
        let Some(cwd_os) = crate::platform::os_str_from_bytes(cwd)
        else {
            return Some(io_os_error_value(ERRNO_NOT_FOUND));
        };
        let dir = std::path::Path::new(cwd_os);
        match std::fs::metadata(dir) {
            Ok(m) if m.is_dir() => {}
            Ok(_) => return Some(io_os_error_value(ERRNO_NOT_A_DIRECTORY)),
            Err(_) => return Some(io_os_error_value(ERRNO_NOT_FOUND)),
        }
    }
    match resolve_program(state) {
        Ok(_) => None,
        Err(code) => Some(io_os_error_value(code)),
    }
}

/// The OS error function table the model answers for
/// `get_os_functions`: std's unix entries, by name (all three
/// entries are the builtins above).
pub(super) fn os_functions_table() -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::FnPtr("std::sys::io::error::unix::format_error".to_string()),
            Value::FnPtr("std::sys::io::error::unix::decode_error_kind".to_string()),
            Value::FnPtr("std::sys::io::error::unix::is_interrupted".to_string()),
        ],
    }
}

/// std's unix `decode_error_kind`: the `ErrorKind` discriminant
/// for an `errno` (Linux values). Mirrors the table in
/// `std::sys::io::error::unix` entry for entry; anything not
/// listed is `Uncategorized` (43).
pub(super) fn decode_error_kind_unix(errno: i32) -> u32 {
    match errno {
        7 => 34,   // E2BIG -> ArgumentListTooLong
        98 => 8,   // EADDRINUSE -> AddrInUse
        99 => 9,   // EADDRNOTAVAIL -> AddrNotAvailable
        16 => 28,  // EBUSY -> ResourceBusy
        103 => 6,  // ECONNABORTED -> ConnectionAborted
        111 => 2,  // ECONNREFUSED -> ConnectionRefused
        104 => 3,  // ECONNRESET -> ConnectionReset
        35 => 30,  // EDEADLK -> Deadlock
        122 => 26, // EDQUOT -> QuotaExceeded
        17 => 12,  // EEXIST -> AlreadyExists
        27 => 27,  // EFBIG -> FileTooLarge
        113 => 4,  // EHOSTUNREACH -> HostUnreachable
        4 => 35,   // EINTR -> Interrupted
        22 => 20,  // EINVAL -> InvalidInput
        21 => 15,  // EISDIR -> IsADirectory
        40 => 18,  // ELOOP -> FilesystemLoop
        2 => 0,    // ENOENT -> NotFound
        12 => 38,  // ENOMEM -> OutOfMemory
        28 => 24,  // ENOSPC -> StorageFull
        31 => 32,  // EMLINK -> TooManyLinks
        36 => 33,  // ENAMETOOLONG -> InvalidFilename
        100 => 10, // ENETDOWN -> NetworkDown
        101 => 5,  // ENETUNREACH -> NetworkUnreachable
        107 => 7,  // ENOTCONN -> NotConnected
        20 => 14,  // ENOTDIR -> NotADirectory
        39 => 16,  // ENOTEMPTY -> DirectoryNotEmpty
        32 => 11,  // EPIPE -> BrokenPipe
        30 => 17,  // EROFS -> ReadOnlyFilesystem
        29 => 25,  // ESPIPE -> NotSeekable
        116 => 19, // ESTALE -> StaleNetworkFileHandle
        110 => 22, // ETIMEDOUT -> TimedOut
        26 => 29,  // ETXTBSY -> ExecutableFileBusy
        18 => 31,  // EXDEV -> CrossesDevices
        115 => 39, // EINPROGRESS -> InProgress
        24 | 23 => 40, // EMFILE | ENFILE -> TooManyOpenFiles
        5 => 41,   // EIO -> InputOutputError
        13 | 1 => 1, // EACCES | EPERM -> PermissionDenied
        38 => 36,  // ENOSYS -> Unsupported
        95 => 36,  // EOPNOTSUPP / ENOTSUP -> Unsupported
        11 => 13,  // EAGAIN / EWOULDBLOCK -> WouldBlock
        _ => 43,   // Uncategorized
    }
}

pub(super) fn ok_result(value: Value) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![value],
    }
}

/// `Err(value)` in the value model.
pub(super) fn err_result(value: Value) -> Value {
    Value::Enum {
        variant: 1,
        fields: vec![value],
    }
}

/// The `str::Utf8Error` value for a host validation failure:
/// the struct's `valid_up_to` and `error_len` fields, in the
/// model's aggregate shape.
pub(super) fn utf8_error_value(error: &std::str::Utf8Error) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::Uint(u128::from(error.valid_up_to() as u64)),
            match error.error_len() {
                Some(n) => Value::Enum {
                    variant: 1,
                    fields: vec![Value::Uint(n as u128)],
                },
                None => Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                },
            },
        ],
    }
}

/// A one-field handle value carrying an id: the shape of the
/// thread model's `JoinHandle` / `Thread` / `ThreadId` values and
/// of the mpsc `Sender` / `Receiver` handle values.
pub(super) fn id_handle(id: u64) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Uint(u128::from(id))],
    }
}

/// A join handle value: `JoinHandle { inner: JoinInner }`, the
/// inner value being the one-field handle of the thread id.
/// Optimizing MIR moves field 0 (the `JoinInner`) out of the
/// handle and calls `JoinInner::join` on it directly, so the
/// nesting is observable: the inner value must have the
/// `JoinInner` type's shape, not the bare id.
pub(super) fn join_handle_value(id: u64) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![id_handle(id)],
    }
}

/// The thread id a join handle value carries, at whichever
/// nesting the value arrives with: the `JoinHandle` aggregate,
/// the `JoinInner` moved out of it, or the bare id.
pub(super) fn join_handle_id(value: &Value) -> Option<u64> {
    let mut cur = value;
    loop {
        match cur {
            Value::Uint(id) => return Some(*id as u64),
            Value::Enum { fields, .. } if fields.len() == 1 => {
                cur = &fields[0];
            }
            _ => return None,
        }
    }
}

/// The thread name a `std::thread::Builder` value carries (its
/// first field, an `Option<String>` in the real field shape), as
/// host text. `None` for an unnamed builder or a value of an
/// unexpected shape.
/// The stack size a `std::thread::Builder` value carries, in
/// bytes: the builder's second field (the `stack_size` setting
/// `Builtin::ThreadBuilderStackSize` recorded), when it is set.
/// Read at `Builder::spawn` time, where the setting becomes the
/// spawned thread's stack budget.
pub(super) fn builder_thread_stack_size(builder: &Value) -> Option<u64> {
    let Value::Enum { variant: 0, fields } = builder else {
        return None;
    };
    let size_field = fields.get(1)?;
    let Value::Enum {
        variant: 1,
        fields: size_fields,
    } = size_field
    else {
        return None;
    };
    match size_fields.first()? {
        Value::Int(v) if *v >= 0 => Some(*v as u64),
        Value::Uint(v) => Some(*v as u64),
        _ => None,
    }
}

pub(super) fn builder_thread_name(builder: &Value) -> Option<String> {
    let Value::Enum { variant: 0, fields } = builder else {
        return None;
    };
    let name_field = fields.first()?;
    let Value::Enum {
        variant: 1,
        fields: name_fields,
    } = name_field
    else {
        return None;
    };
    let bytes = string_aggregate_bytes(name_fields.first()?)?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

/// Run the configured tool runner for this command, when the
/// command's program is one of the intercepted tool programs --
/// or, for any other program, the interception's fallback
/// runner when one is configured (the host side answers for
/// artifacts the guest itself produced earlier in the run).
/// `None` when no interception applies, in which case the
/// caller keeps the process model's standing answer. A runner
/// error becomes the run's unsupported error: the tool ran the
/// invocation and reported it cannot be carried out.
pub(super) fn tool_outcome(
    interception: Option<&crate::interp::ToolInterception>,
    state: &crate::interp::CommandState,
    output_captured: bool,
) -> InterpResult<Option<crate::interp::ToolRunOutcome>> {
    let Some(interception) = interception else {
        return Ok(None);
    };
    let runner = if interception.programs.iter().any(|p| p == &state.program) {
        &interception.runner
    } else if let Some(fallback) = &interception.fallback {
        fallback
    } else {
        return Ok(None);
    };
    // The path the spawn resolution settled on rides along:
    // the host side looks its answer up by it, so a bare
    // program name invoked under a command-specific `PATH`
    // reaches the artifact that search found, not a
    // namesake in the working directory (see
    // `ToolRunRequest::resolved_program`). Both callers run
    // this only after the spawn gate has accepted the
    // command, so the resolution is the one that succeeded.
    let resolved_program = resolve_program(state).ok().and_then(|path| {
        crate::platform::os_str_bytes(path.as_os_str())
            .map(|b| b.into_owned())
    });
    let request = crate::interp::ToolRunRequest {
        program: state.program.clone(),
        args: state.args.clone(),
        argv0: state.argv0.clone(),
        envs: state.envs.clone(),
        env_clear: state.env_clear,
        cwd: state.cwd.clone(),
        resolved_program,
        output_captured,
    };
    match (runner.0)(&request) {
        Ok(outcome) => Ok(Some(outcome)),
        Err(reason) => Err(unsupported(reason)),
    }
}

/// The result of observing a child's end (see
/// `Interpreter::child_finish`).
pub(super) enum ChildWait {
    Running,
    Finished(i32),
    IoError(Value),
}

/// The spawn request a command denotes, for the process
/// host: the recorded builder state, the interpreter's own
/// resolution of the program, and the stdio wiring the entry
/// point settles on -- `output` pipes the streams it
/// collects and nulls the child's stdin, `status` / `spawn`
/// inherit what the guest did not configure.
fn process_spawn_request(
    state: &crate::interp::CommandState,
    output_captured: bool,
    stdio_files: [Option<std::fs::File>; 3],
    child_sigpipe: u64,
) -> crate::interp::ProcessSpawnRequest {
    use crate::interp::ProcessStdio;
    let [stdin_file, stdout_file, stderr_file] = stdio_files;
    let wire = |config: Option<crate::interp::StdioConfig>,
                default: ProcessStdio,
                file: Option<std::fs::File>|
     -> ProcessStdio {
        match config {
            Some(crate::interp::StdioConfig::Inherit) => {
                ProcessStdio::Inherit
            }
            Some(crate::interp::StdioConfig::Null) => ProcessStdio::Null,
            Some(crate::interp::StdioConfig::Piped) => ProcessStdio::Piped,
            Some(crate::interp::StdioConfig::File(_)) => match file {
                Some(file) => ProcessStdio::Fd(file),
                // The caller resolves the file of every
                // file-form configuration before the request
                // is built and rejects a missing one; inherit
                // is the neutral wire for one that is absent.
                None => ProcessStdio::Inherit,
            },
            // Rejected by the caller's stdio check before the
            // request is built; inherit is the neutral wire.
            Some(crate::interp::StdioConfig::Unmodelled) => {
                ProcessStdio::Inherit
            }
            None => default,
        }
    };
    crate::interp::ProcessSpawnRequest {
        program: state.program.clone(),
        args: state.args.clone(),
        argv0: state.argv0.clone(),
        envs: state.envs.clone(),
        env_clear: state.env_clear,
        cwd: state.cwd.clone(),
        resolved_program: resolve_program(state).ok().and_then(|path| {
            crate::platform::os_str_bytes(path.as_os_str())
                .map(|b| b.into_owned())
        }),
        stdin: wire(
            state.stdin_stdio,
            if output_captured {
                ProcessStdio::Null
            } else {
                ProcessStdio::Inherit
            },
            stdin_file,
        ),
        stdout: wire(
            state.stdout_stdio,
            if output_captured {
                ProcessStdio::Piped
            } else {
                ProcessStdio::Inherit
            },
            stdout_file,
        ),
        stderr: wire(
            state.stderr_stdio,
            if output_captured {
                ProcessStdio::Piped
            } else {
                ProcessStdio::Inherit
            },
            stderr_file,
        ),
        child_sigpipe,
        uid: state.uid,
        gid: state.gid,
        groups: state.groups.clone(),
    }
}

/// Reject an invocation the process host cannot wire: a
/// stdio configuration the model does not decode (see
/// `check_stdio_modelled`, which covers the output streams
/// for the tool model; the host also wires the child's
/// stdin, so its configuration is checked here too).
fn check_host_stdio(
    state: &crate::interp::CommandState,
) -> InterpResult<()> {
    for config in [
        state.stdin_stdio,
        state.stdout_stdio,
        state.stderr_stdio,
    ] {
        if config == Some(crate::interp::StdioConfig::Unmodelled) {
            return Err(unsupported(
                "a stdio configuration this run does not model (an unrecognized form)"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

/// A `std::process::Child` value denoting child-table id
/// `id`: the handle -- the sys `Process` value, the id in
/// its first slot and the not-yet-observed status slot
/// empty -- and the three stdio handle slots. A stream the
/// guest configured piped carries its handle value: the
/// pipe end's file-table id under the handle wrapper chain
/// (see `file_value`), so the guest reads and writes the
/// pipe through the ordinary file model, and dropping the
/// handle closes its end. A stream wired any other way
/// leaves its slot empty, as std leaves it.
fn child_value(
    id: u64,
    stdin: Option<u64>,
    stdout: Option<u64>,
    stderr: Option<u64>,
) -> Value {
    let slot = |file: Option<u64>| match file {
        Some(file_id) => some_value(file_value(file_id)),
        None => none_value(),
    };
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::Enum {
                variant: 0,
                fields: vec![Value::Uint(u128::from(id)), none_value()],
            },
            slot(stdin),
            slot(stdout),
            slot(stderr),
        ],
    }
}

/// A `Cow<str>` value over the given bytes: borrowed when
/// they are valid UTF-8, owned with replacement characters
/// otherwise (the host `from_utf8_lossy` semantics); in this
/// model both variants carry the string data itself.
pub(super) fn cow_str_value(bytes: &[u8]) -> Value {
    match std::str::from_utf8(bytes) {
        Ok(text) => Value::Enum {
            variant: 0,
            fields: vec![Value::Str(text.to_string())],
        },
        Err(_) => Value::Enum {
            variant: 1,
            fields: vec![Value::Str(
                String::from_utf8_lossy(bytes).into_owned(),
            )],
        },
    }
}

/// A `Vec<u8>` value: the element-list aggregate the
/// interpreter uses for vectors.
pub(super) fn byte_vec_value(bytes: &[u8]) -> Value {
    Value::Enum {
        variant: 0,
        fields: bytes.iter().map(|b| Value::Uint(u128::from(*b))).collect(),
    }
}

/// An `OsString` value in std's shape: `OsString { inner: Buf {
/// inner: Vec<u8> } }`, the same nesting `pathbuf_value` builds
/// for the `OsString` inside a `PathBuf`.
pub(super) fn os_string_value(bytes: &[u8]) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![byte_vec_value(bytes)],
        }],
    }
}

/// A `std::process::ExitStatus` value: the platform wait
/// status in std's own layout (the outer struct wraps the sys
/// status struct, which wraps the raw `c_int`). An exited
/// process's raw status carries its code in the high byte; a
/// process ended by a signal carries the signal number in the
/// low byte, as `waitpid` reports it. The guest's own
/// `ExitStatus::code` body decodes it from there.
pub(super) fn exit_status_value(exit_code: i32) -> Value {
    let raw: i32 = if exit_code >= 0 {
        exit_code << 8
    } else {
        -exit_code
    };
    Value::Enum {
        variant: 0,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![Value::Int(i128::from(raw))],
        }],
    }
}

/// A `std::process::Output` value: the exit status and the two
/// captured byte vectors, in declaration order.
/// The `Stdio` configuration a stdio argument value
/// denotes. The public `std::process::Stdio` is a one-field
/// wrapper over the sys enum (`Inherit` 0, `Null` 1,
/// `MakePipe` 2, `Fd` 3, `StaticFd` 4); the sys-level setter
/// receives the sys value unwrapped. The `Fd` form denotes
/// the file its payload handle carries: the payload is a
/// file handle value (a `File`, a child pipe handle), whose
/// wrapper chain bottoms out at the file-table id
/// (`file_handle_id`). The value the conversion hands the
/// setter may also arrive as that bare chain under the
/// public wrapper, with no enum layer in between; both
/// shapes decode to the same configuration. A shape the
/// model does not decode is
/// [`StdioConfig::Unmodelled`](crate::interp::StdioConfig::Unmodelled).
fn stdio_config_of(value: &Value) -> crate::interp::StdioConfig {
    use crate::interp::StdioConfig;
    let inner = match value {
        Value::Enum { fields, .. } if fields.len() == 1 => &fields[0],
        other => other,
    };
    match inner {
        Value::Enum { variant: 0, fields } if fields.is_empty() => StdioConfig::Inherit,
        Value::Enum { variant: 1, fields } if fields.is_empty() => StdioConfig::Null,
        Value::Enum { variant: 2, fields } if fields.is_empty() => StdioConfig::Piped,
        Value::Enum { variant: 3, fields } if fields.len() == 1 => {
            match crate::interp::file_handle_id(&fields[0]) {
                Some(id) => StdioConfig::File(id),
                None => StdioConfig::Unmodelled,
            }
        }
        // The bare handle chain (see the doc above): a
        // one-field wrapper chain bottoming out at the
        // file-table id.
        Value::Enum { variant: 0, fields } if fields.len() == 1 => {
            match crate::interp::file_handle_id(inner) {
                Some(id) => StdioConfig::File(id),
                None => StdioConfig::Unmodelled,
            }
        }
        _ => StdioConfig::Unmodelled,
    }
}

/// Record a stdio setter's configuration on the command it
/// names: the builtin's own name says which stream it
/// configures. The stdin configuration is recorded for the
/// process host's wiring (the tool model never feeds a
/// child's stdin and ignores it).
pub(super) fn record_stdio_config(state: &mut crate::interp::CommandState, name: &str, value: &Value) {
    let config = stdio_config_of(value);
    if name.ends_with("::stdout") {
        state.stdout_stdio = Some(config);
    } else if name.ends_with("::stderr") {
        state.stderr_stdio = Some(config);
    } else if name.ends_with("::stdin") {
        state.stdin_stdio = Some(config);
    }
}

/// Reject an invocation whose stdio configuration the
/// model does not route (see `StdioConfig::Unmodelled`):
/// running it with a guessed routing would expose output
/// the guest configured away, or hide output it kept.
pub(super) fn check_stdio_modelled(state: &crate::interp::CommandState) -> InterpResult<()> {
    for config in [state.stdout_stdio, state.stderr_stdio] {
        if config == Some(crate::interp::StdioConfig::Unmodelled) {
            return Err(unsupported(
                "a stdio configuration this run does not model (an unrecognized form)"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

/// Where one stream's bytes go for an invocation, under
/// its configuration: `output` collects the streams it
/// pipes, `status` relays the streams it inherits, and a
/// stream redirected to null -- or, for `status`, to a
/// pipe nobody reads -- goes nowhere observable. An
/// unmodelled configuration never reaches the routing
/// (`check_stdio_modelled` rejects it first); it is
/// grouped with the discards defensively.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum StdioRoute {
    Collect,
    Relay,
    Discard,
    /// The stream's bytes go to the file the configuration
    /// names (see `StdioConfig::File`): the invocation's
    /// collected bytes are written into that file-table
    /// entry, and the stream contributes nothing to the
    /// guest's own streams or to its `Output`.
    File,
}

pub(super) fn stdio_route(
    config: Option<crate::interp::StdioConfig>,
    output_captured: bool,
) -> StdioRoute {
    use crate::interp::StdioConfig;
    match (config, output_captured) {
        (Some(StdioConfig::File(_)), _) => StdioRoute::File,
        (Some(StdioConfig::Null), _) | (Some(StdioConfig::Unmodelled), _) => {
            StdioRoute::Discard
        }
        (Some(StdioConfig::Piped), false) => StdioRoute::Discard,
        (Some(StdioConfig::Piped), true) | (None, true) => StdioRoute::Collect,
        (Some(StdioConfig::Inherit), _) | (None, false) => StdioRoute::Relay,
    }
}

pub(super) fn process_output_value(exit_code: i32, stdout: &[u8], stderr: &[u8]) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            exit_status_value(exit_code),
            byte_vec_value(stdout),
            byte_vec_value(stderr),
        ],
    }
}

/// Registration fragment of the builtin name registry: the process environment surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn env_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::env::args" => Builtin::EnvArgs,
        "std::env::args_os" => Builtin::EnvArgsOs,
        "std::env::vars" => Builtin::EnvVars,
        "std::env::vars_os" => Builtin::EnvVarsOs,
        "std::env::current_exe" => Builtin::EnvCurrentExe,
        "std::env::var::inner" => Builtin::EnvVarInner,
        "std::env::split_paths" => Builtin::EnvSplitPaths,
        "std::env::join_paths" => Builtin::EnvJoinPaths,
        "std::sys::env::unix::getenv" => Builtin::EnvGetenv,
        "std::sys::env::unix::setenv" => Builtin::EnvSetenv,
        "std::sys::env::unix::unsetenv" => Builtin::EnvUnsetenv,
        "std::env::current_dir" => Builtin::EnvCurrentDir,
        "std::sys::paths::unix::getcwd" => Builtin::EnvCurrentDir,
        "std::env::temp_dir" => Builtin::EnvTempDir,
        "std::sys::paths::unix::temp_dir" => Builtin::EnvTempDir,
        "std::env::home_dir" => Builtin::EnvHomeDir,
        "std::sys::paths::unix::home_dir" => Builtin::EnvHomeDir,
        "std::sys::paths::unix::chdir" => Builtin::SysChdir,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: process exit, abort and unreachable.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn process_exit_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::process::exit" => Builtin::ProcessExit,
        "std::process::abort" => Builtin::IntrinsicsAbort,
        "std::intrinsics::abort" | "core::intrinsics::abort" => {
            Builtin::IntrinsicsAbort
        }
        "std::intrinsics::unreachable" | "core::intrinsics::unreachable" => {
            Builtin::IntrinsicsUnreachable
        }
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the `Command` builder surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn command_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::sys::process::unix::common::Command::new" => Builtin::ProcessCommandNew,
        "std::sys::process::unix::common::Command::arg" => Builtin::ProcessCommandArg,
        "std::sys::process::unix::common::Command::cwd" => Builtin::ProcessCommandCwd,
        "std::sys::process::unix::common::Command::env_mut" => Builtin::ProcessCommandEnvMut,
        "std::sys::process::env::CommandEnv::set" => Builtin::ProcessCommandEnvSet,
        "std::sys::process::env::CommandEnv::remove" => Builtin::ProcessCommandEnvRemove,
        "std::sys::process::env::CommandEnv::clear" => Builtin::ProcessCommandEnvClear,
        "std::sys::process::unix::common::Command::stdin" => Builtin::ProcessCommandStdio,
        "std::sys::process::unix::common::Command::stdout" => Builtin::ProcessCommandStdio,
        "std::sys::process::unix::common::Command::stderr" => Builtin::ProcessCommandStdio,
        "std::sys::process::unix::unix::Command::spawn" => Builtin::ProcessSpawn,
        "std::env::current_exe" => Builtin::CurrentExe,
        "std::process::id" => Builtin::ProcessId,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the `std::process` public surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn command_api_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::process::Command::new" => Builtin::ProcessCommandNewPublic,
        "std::process::Command::arg" => Builtin::ProcessCommandArgPublic,
        "std::process::Command::current_dir" => Builtin::ProcessCommandCwdPublic,
        "std::process::Command::env" => Builtin::ProcessCommandEnvPublic,
        "std::process::Command::env_remove" => Builtin::ProcessCommandEnvRemovePublic,
        "std::process::Command::env_clear" => Builtin::ProcessCommandEnvClearPublic,
        "std::process::Command::stdin" => Builtin::ProcessCommandStdioPublic,
        "std::process::Command::stdout" => Builtin::ProcessCommandStdioPublic,
        "std::process::Command::stderr" => Builtin::ProcessCommandStdioPublic,
        "std::os::unix::process::CommandExt::uid" => Builtin::ProcessCommandUid,
        "std::os::unix::process::CommandExt::gid" => Builtin::ProcessCommandGid,
        "std::os::unix::process::CommandExt::groups" => Builtin::ProcessCommandGroups,
        "std::process::Command::spawn" => Builtin::ProcessSpawn,
        "std::process::Command::output" => Builtin::ProcessOutput,
        "std::process::Command::status" => Builtin::ProcessSpawn,
        "std::process::Child::wait" => Builtin::ProcessChildWait,
        "std::process::Child::try_wait" => Builtin::ProcessChildTryWait,
        "std::process::Child::kill" => Builtin::ProcessChildKill,
        "std::process::Child::wait_with_output" => {
            Builtin::ProcessChildWaitWithOutput
        }
        "std::process::Child::id" => Builtin::ProcessChildId,
        "std::sys::process::output" => Builtin::ProcessOutput,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry:
/// the raw-descriptor constructor trait.
///
/// One contiguous slice of the table that [`lookup`]
/// chains in registry order; the fragment lives with
/// the family that serves these builtins, so a new name
/// is registered here, in the family's own module, and
/// nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn raw_fd_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::os::fd::FromRawFd::from_raw_fd"
        | "std::os::unix::io::FromRawFd::from_raw_fd" => Builtin::FromRawFd,
        _ => return None,
    })
}

impl<'a> Interpreter<'a> {
    /// `FromRawFd::from_raw_fd(fd) -> Self`: build the
    /// handle value of the call's `Self` type around a
    /// guest descriptor number. In the descriptor model
    /// the number the guest holds is the handle's
    /// identity (a file handle's table id, a descriptor
    /// the routing rerouted), so a `File` is the
    /// file-handle chain over the number and a `Stdio`
    /// is the file-form configuration wrapping that
    /// chain — the shapes the file and stdio models
    /// already read back. A socket `Self` has a
    /// different value shape and is declined.
    pub(super) fn exec_from_raw_fd(
        &mut self,
        _frames: &mut Vec<Frame>,
        name: &str,
        generic_args: &[String],
        vals: &[Value],
    ) -> InterpResult<Value> {
        let fd = as_i128(&vals[0], name)?;
        if fd < 0 {
            return Err(unsupported(format!(
                "call to {name}: a negative descriptor number"
            )));
        }
        let self_ty = generic_args.first().map(|t| t.as_str()).unwrap_or("");
        if self_ty == "std::process::Stdio" {
            return Ok(Value::Enum {
                variant: 0,
                fields: vec![Value::Enum {
                    variant: 3,
                    fields: vec![file_value(fd as u64)],
                }],
            });
        }
        if self_ty == "std::fs::File" || self_ty.is_empty() {
            return Ok(file_value(fd as u64));
        }
        Err(unsupported(format!(
            "call to {name}: from_raw_fd for {self_ty} is not modelled"
        )))
    }
}
