//! The driver-side process host: real child processes behind the guest process model.


/// The process host of a run that opted into real child
/// processes (`--guest-process-host`): the interpreter's
/// `Child` model performed against live host children. A
/// child whose program is this driver executable itself --
/// or a byte-identical copy of it, the shape a guest that
/// copies its own `current_exe` and runs the copy produces --
/// is the guest program re-executing: the child process is
/// this driver running the same source with the child's
/// arguments, environment and working directory, which is
/// what a native re-execution of the test binary does. Any
/// other program starts as itself on the host.
pub(crate) struct DriverProcessHost {
    pub(crate) exe: std::path::PathBuf,
    /// Absolute path of the source this run interprets (a
    /// self-child's input).
    pub(crate) input: String,
    pub(crate) edition: String,
    pub(crate) rustc_args: Vec<String>,
    pub(crate) aux_specs: Vec<String>,
    pub(crate) aux_lib_specs: Vec<String>,
    pub(crate) children: std::sync::Mutex<
        std::collections::HashMap<u64, HostChild>,
    >,
    pub(crate) next: std::sync::Mutex<u64>,
    /// (length, content hash) of this executable, computed
    /// once for the copy-of-self check.
    pub(crate) exe_fingerprint: std::sync::OnceLock<Option<(u64, u64)>>,
}

/// One live host child started for the interpreter's
/// `Child` model (see [`DriverProcessHost`]): the streams
/// themselves are with the interpreter (see
/// `ProcessSpawned`); what stays here is what observing the
/// child's end needs.
pub(crate) struct HostChild {
    child: std::process::Child,
    /// A self-child's stderr drain: the nested driver's own
    /// diagnostic lines, kept for the exit-2 disambiguation
    /// when the child's end is observed. A plain host child's
    /// stderr is the guest's own stream, handed to the
    /// interpreter at spawn, so there is nothing to drain.
    diag: Option<std::thread::JoinHandle<Vec<u8>>>,
    /// A self-child's guest-stderr relay: the thread
    /// carrying the guest-stderr stream into the file the
    /// guest wired the child's stderr to, when it wired one
    /// (see `ProcessHostOps::spawn`). It owns the relay's
    /// duplicate of the wired end and drops it when the
    /// stream ends, which is what delivers end-of-file to a
    /// guest reading the other end of a pipe it wired.
    relay: Option<std::thread::JoinHandle<()>>,
    /// A self-child's guest-stderr channel file (the guest
    /// reads the stream fed from it through the interpreter;
    /// see `guest_stderr_stream`); removed once the child's
    /// end has been observed.
    stderr_file: Option<std::path::PathBuf>,
    self_run: bool,
    /// The exit code once the end has been observed, so
    /// repeated waits answer the same, as std's cached
    /// status does.
    exit: Option<i32>,
}

/// The pipe end a spawned child hands over, as the file the
/// interpreter's file table holds: the descriptor the pipe
/// value owns transfers to the file.
#[cfg(unix)]
fn pipe_file<P: std::os::unix::io::IntoRawFd>(
    pipe: P,
) -> std::fs::File {
    use std::os::unix::io::FromRawFd as _;
    // SAFETY: `pipe` owns its descriptor; `into_raw_fd`
    // relinquishes it and the file takes ownership, so the
    // descriptor is closed exactly once, by the file.
    unsafe { std::fs::File::from_raw_fd(pipe.into_raw_fd()) }
}

/// The pipe end a spawned child hands over, as the file the
/// interpreter's file table holds: the handle the pipe
/// value owns transfers to the file.
#[cfg(windows)]
fn pipe_file<P: std::os::windows::io::IntoRawHandle>(
    pipe: P,
) -> std::fs::File {
    use std::os::windows::io::FromRawHandle as _;
    // SAFETY: `pipe` owns its handle; `into_raw_handle`
    // relinquishes it and the file takes ownership, so the
    // handle is closed exactly once, by the file.
    unsafe { std::fs::File::from_raw_handle(pipe.into_raw_handle()) }
}

/// Whether a self-child's end is the nested driver's own
/// failure rather than the child program's exit: exit 2 with
/// the driver's own lines on the diagnostic stream. Then
/// `Some(reason)` -- the driver's first named reason when it
/// gave one -- and the invocation is the host's failure to
/// carry out, not an exit code (the same disambiguation
/// `artifact_run` makes).
fn self_run_failure(exit_code: i32, drained: &[u8]) -> Option<String> {
    if exit_code != 2 {
        return None;
    }
    let text = String::from_utf8_lossy(drained);
    let reason = text.lines().find_map(|line| {
        line.strip_prefix("sparagmos-driver run: unsupported: ")
            .or_else(|| {
                line.strip_prefix(
                    "sparagmos-driver: compilation failed: ",
                )
            })
            .map(|rest| rest.to_string())
    });
    let driver_failed = reason.is_some()
        || text
            .lines()
            .any(|line| line.starts_with("sparagmos-driver"));
    if !driver_failed {
        return None;
    }
    Some(reason.unwrap_or_else(|| {
        "the guest program could not be run as a child process"
            .to_string()
    }))
}

/// A host string from a byte string the interpreter's
/// model carries (a program name, an argument, an
/// environment name or value, a working directory). On
/// Unix the bytes are the host string itself. On other
/// hosts the byte strings this process host carries are
/// UTF-8 text -- the form the spawn plumbing produces
/// there -- and the host string is that text.
fn os_string(bytes: &[u8]) -> std::ffi::OsString {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt as _;
        std::ffi::OsStr::from_bytes(bytes).to_os_string()
    }
    #[cfg(not(unix))]
    {
        std::ffi::OsString::from(
            String::from_utf8_lossy(bytes).into_owned(),
        )
    }
}

fn drain_pipe<R: std::io::Read>(mut pipe: R) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = pipe.read_to_end(&mut buf);
    buf
}

/// Carry one stream into a file until the stream ends:
/// the relay a file-wired stream of a spawned self-child
/// runs on (see `ProcessHostOps::spawn`). The target is
/// the relay's own duplicate of the wired end; dropping it
/// when the source ends is what delivers end-of-file to a
/// reader of the other end of a pipe the guest wired the
/// stream to, so no duplicated writer end may outlive the
/// relay.
fn relay_stream(
    mut source: std::fs::File,
    mut target: std::fs::File,
) {
    use std::io::Write as _;
    let _ = std::io::copy(&mut source, &mut target);
    let _ = target.flush();
}

/// The guest-visible stderr stream of a spawned self-child:
/// a pipe fed by a thread tailing the guest-stderr channel
/// file. The nested driver writes the guest's stderr to the
/// file (its own stderr is the diagnostic stream), but the
/// file itself cannot be the guest's stream: a plain file's
/// reads answer end-of-file as soon as they reach the bytes
/// written so far, so a guest reading its child's stderr
/// while the child still runs observes an empty stream. The
/// pipe restores the native shape -- reads wait for the
/// child's writes, and the stream ends once the child has
/// ended (signalled by the diagnostic drain's completion)
/// and the file's remaining bytes have been carried over.
fn guest_stderr_stream(
    file: &std::path::Path,
    diag_done: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Option<std::fs::File> {
    use std::io::{Read as _, Write as _};
    let mut source = std::fs::File::open(file).ok()?;
    let (reader, mut writer) = std::io::pipe().ok()?;
    std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        // Set once the child's end is observed: the stream
        // finishes after one more drain pass, so bytes the
        // child wrote just before ending are not left behind.
        let mut final_pass = false;
        loop {
            match source.read(&mut buf) {
                Ok(0) => {
                    if final_pass {
                        break;
                    }
                    if diag_done
                        .load(std::sync::atomic::Ordering::SeqCst)
                    {
                        final_pass = true;
                    }
                    std::thread::sleep(
                        std::time::Duration::from_millis(5),
                    );
                }
                Ok(n) => {
                    if writer.write_all(&buf[..n]).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    Some(pipe_file(reader))
}

/// One host child being run to completion for a
/// `status` / `output` call (see `ProcessHostOps::run`):
/// both streams drained on threads while it runs, as std's
/// collection does.
struct RunChild {
    stdout: Option<std::thread::JoinHandle<Vec<u8>>>,
    stderr: Option<std::thread::JoinHandle<Vec<u8>>>,
    stderr_file: Option<std::path::PathBuf>,
    self_run: bool,
}

/// The exit code a finished host child reports in the
/// interpreter's convention: the exit code, or the negated
/// signal number for a child ended by a signal.
fn host_exit_code(status: std::process::ExitStatus) -> i32 {
    match status.code() {
        Some(code) => code,
        None => {
            // A status without a code is a child ended by a
            // signal, which only Unix reports; hosts without
            // signals always carry a code, so this arm is
            // unreachable there and answers 0.
            #[cfg(unix)]
            {
                use std::os::unix::process::ExitStatusExt as _;
                -status.signal().unwrap_or(0)
            }
            #[cfg(not(unix))]
            {
                0
            }
        }
    }
}

impl DriverProcessHost {
    fn fingerprint(path: &std::path::Path) -> Option<(u64, u64)> {
        use std::io::Read as _;
        let mut file = std::fs::File::open(path).ok()?;
        let len = file.metadata().ok()?.len();
        let mut hasher =
            std::collections::hash_map::DefaultHasher::new();
        let mut buf = [0u8; 65536];
        loop {
            let n = file.read(&mut buf).ok()?;
            if n == 0 {
                break;
            }
            std::hash::Hasher::write(&mut hasher, &buf[..n]);
        }
        Some((len, std::hash::Hasher::finish(&hasher)))
    }

    /// Whether `path` is this executable or a byte-identical
    /// copy of it.
    fn is_self_program(&self, path: &std::path::Path) -> bool {
        if path == self.exe {
            return true;
        }
        if self.is_input_program(path) {
            return true;
        }
        let mine = self
            .exe_fingerprint
            .get_or_init(|| Self::fingerprint(&self.exe));
        match (mine, Self::fingerprint(path)) {
            (Some(a), Some(b)) => *a == b,
            _ => false,
        }
    }

    /// Whether `path` denotes the source this run interprets.
    /// A spawn naming it is the guest re-executing itself under
    /// the name it knows: the guest's argv[0] is the source
    /// path, which is not an executable file, so the filesystem
    /// resolution refuses it -- but the invocation is a
    /// self-child all the same, run by re-interpreting the
    /// source (the `is_self_program` treatment).
    fn is_input_program(&self, path: &std::path::Path) -> bool {
        let input = std::path::Path::new(&self.input);
        if path == input {
            return true;
        }
        match (
            std::fs::canonicalize(path),
            std::fs::canonicalize(input),
        ) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    }

    /// Collect a finished self/host child's outcome from its
    /// status and drained streams. For a self-child, the
    /// guest's stderr is read back from its channel file, and
    /// a nested driver failure (exit 2 with the driver's own
    /// lines on the diagnostic stream) is the host's failure
    /// to carry the invocation out, not the child program's
    /// exit -- the same disambiguation `artifact_run` makes.
    fn finalize(
        &self,
        entry: &mut RunChild,
        status: std::process::ExitStatus,
    ) -> Result<sparagmos_interp::ProcessChildOutcome, sparagmos_interp::ProcessHostError> {
        use sparagmos_interp::ProcessHostError as HostErr;
        let stdout = entry
            .stdout
            .take()
            .map(|t| t.join().unwrap_or_default())
            .unwrap_or_default();
        let drained = entry
            .stderr
            .take()
            .map(|t| t.join().unwrap_or_default())
            .unwrap_or_default();
        let exit_code = host_exit_code(status);
        if !entry.self_run {
            return Ok(sparagmos_interp::ProcessChildOutcome {
                exit_code,
                stdout,
                stderr: drained,
            });
        }
        let stderr = entry
            .stderr_file
            .as_ref()
            .map(|f| std::fs::read(f).unwrap_or_default())
            .unwrap_or_default();
        if let Some(file) = &entry.stderr_file {
            let _ = std::fs::remove_file(file);
        }
        if let Some(reason) = self_run_failure(exit_code, &drained) {
            return Err(HostErr::Failed(reason));
        }
        Ok(sparagmos_interp::ProcessChildOutcome {
            exit_code,
            stdout,
            stderr,
        })
    }
}

fn wire_stdio(
    stdio: sparagmos_interp::ProcessStdio,
) -> std::process::Stdio {
    match stdio {
        sparagmos_interp::ProcessStdio::Inherit => {
            std::process::Stdio::inherit()
        }
        sparagmos_interp::ProcessStdio::Null => {
            std::process::Stdio::null()
        }
        sparagmos_interp::ProcessStdio::Piped => {
            std::process::Stdio::piped()
        }
        sparagmos_interp::ProcessStdio::Fd(file) => {
            std::process::Stdio::from(file)
        }
    }
}

impl DriverProcessHost {
    /// Assemble the host command a spawn request denotes:
    /// a self invocation becomes this driver re-interpreting
    /// the same source with the child's arguments (see
    /// [`DriverProcessHost`]); any other program starts as
    /// itself. Also settles whether the invocation is a
    /// self-run and, for one, the guest-stderr channel file.
    /// The caller wires the stdio and starts the command.
    fn assemble(
        &self,
        request: &sparagmos_interp::ProcessSpawnRequest,
    ) -> Result<
        (
            std::process::Command,
            bool,
            Option<std::path::PathBuf>,
        ),
        sparagmos_interp::ProcessHostError,
    > {
        use sparagmos_interp::ProcessHostError as HostErr;
        let os = os_string;
        // The program the invocation denotes, for the self
        // check: the interpreter's own resolution when it
        // resolved; else the spelled program resolved against
        // the child's working directory, where a native spawn
        // resolves it.
        let spelled =
            std::path::Path::new(&os(&request.program)).to_path_buf();
        let candidate = match &request.resolved_program {
            Some(resolved) => {
                Some(std::path::PathBuf::from(os(resolved)))
            }
            None => {
                let base = match &request.cwd {
                    Some(cwd) => std::env::current_dir()
                        .map(|dir| dir.join(os(cwd))),
                    None => std::env::current_dir(),
                };
                base.map(|dir| dir.join(&spelled)).ok()
            }
        };
        let self_run = candidate
            .as_deref()
            .map(|path| self.is_self_program(path))
            .unwrap_or(false);
        let mut cmd = if self_run {
            std::process::Command::new(&self.exe)
        } else {
            std::process::Command::new(os(&request.program))
        };
        {
            use std::os::unix::process::CommandExt as _;
            if let Some(uid) = request.uid {
                cmd.uid(uid);
            }
            if let Some(gid) = request.gid {
                cmd.gid(gid);
            }
        }
        let mut stderr_file = None;
        if self_run {
            // The nested run carries its arguments as
            // command-line strings, and the interpreter's
            // argument model holds text: an argument that is
            // not valid UTF-8 cannot be carried without
            // changing its bytes, so the invocation is
            // unsupported rather than silently lossy (the
            // artifact run's standing answer).
            for arg in &request.args {
                if std::str::from_utf8(arg).is_err() {
                    return Err(HostErr::Failed(
                        "spawning the guest program as a child: a guest argument is not valid UTF-8, which the run cannot carry unchanged"
                            .to_string(),
                    ));
                }
            }
            static NEXT_FILE: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let file = std::env::temp_dir().join(format!(
                "sparagmos-process-child-stderr-{}-{}",
                std::process::id(),
                NEXT_FILE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            cmd.arg("run")
                .arg(&self.input)
                .arg("--edition")
                .arg(&self.edition);
            for spec in &self.aux_specs {
                cmd.arg("--aux").arg(spec);
            }
            for spec in &self.aux_lib_specs {
                cmd.arg("--aux-lib").arg(spec);
            }
            for arg in &self.rustc_args {
                cmd.arg("--rustc-arg").arg(arg);
            }
            cmd.arg("--guest-process-host");
            cmd.arg("--guest-stderr-file").arg(&file);
            cmd.arg("--guest-stdout-passthrough");
            let program_name = request
                .argv0
                .as_deref()
                .unwrap_or(&request.program);
            cmd.arg("--guest-program-name")
                .arg(String::from_utf8_lossy(program_name).into_owned());
            for arg in &request.args {
                cmd.arg("--guest-arg")
                    .arg(String::from_utf8_lossy(arg).into_owned());
            }
            stderr_file = Some(file);
        } else {
            if let Some(argv0) = &request.argv0 {
                // std offers an argv[0] override only on
                // Unix; elsewhere the spawned program's own
                // name stands as its argv[0], the native
                // convention there.
                #[cfg(unix)]
                {
                    use std::os::unix::process::CommandExt as _;
                    cmd.arg0(os(argv0));
                }
                #[cfg(not(unix))]
                {
                    let _ = argv0;
                }
            }
            for arg in &request.args {
                cmd.arg(os(arg));
            }
        }
        if request.env_clear {
            cmd.env_clear();
        }
        for (key, value) in &request.envs {
            match value {
                Some(v) => {
                    cmd.env(os(key), os(v));
                }
                None => {
                    cmd.env_remove(os(key));
                }
            }
        }
        if let Some(cwd) = &request.cwd {
            cmd.current_dir(os(cwd));
        }
        // The child's supplementary groups the guest set
        // through `CommandExt::groups`: std's own setter
        // for them is not stable, so a `pre_exec` hook
        // makes the `setgroups` call in the child before
        // exec — the call std's implementation makes —
        // and its failure (too many groups, no privilege)
        // fails the spawn with the host error, as the
        // native spawn fails.
        if let Some(groups) = &request.groups {
            use std::os::unix::process::CommandExt as _;
            let groups = groups.clone();
            unsafe {
                cmd.pre_exec(move || {
                    if host_setgroups(groups.len(), groups.as_ptr()) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
        // The child starts with the `SIGPIPE`
        // disposition the guest's spawn semantics name
        // (see `ProcessSpawnRequest::child_sigpipe`):
        // this process's own spawn applies the driver
        // binary's runtime rule (the default reset to
        // `SIG_DFL`), which is the guest's rule only
        // under the default setup, so a `pre_exec` hook
        // — which std runs in the child after its own
        // reset, before exec — installs the guest's
        // disposition. Hosts without signals have no
        // SIGPIPE disposition to install: the guest's rule
        // names a Unix disposition, so there is nothing to
        // carry out there.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt as _;
            let disposition = request.child_sigpipe as usize;
            unsafe {
                cmd.pre_exec(move || {
                    const SIGPIPE: i32 = 13;
                    const SIG_ERR: usize = usize::MAX;
                    if host_signal(SIGPIPE, disposition) == SIG_ERR {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
        Ok((cmd, self_run, stderr_file))
    }

    /// Settle a spawned child's observed end: the exit code
    /// in the interpreter's convention, cached on the entry.
    /// A self-child's end first faces the exit-2
    /// disambiguation against its drained diagnostic stream
    /// (see `self_run_failure`): a nested driver failure is
    /// the host's failure, not an exit code.
    fn observe_end(
        &self,
        entry: &mut HostChild,
        status: std::process::ExitStatus,
        join_relay: bool,
    ) -> Result<i32, sparagmos_interp::ProcessHostError> {
        use sparagmos_interp::ProcessHostError as HostErr;
        let exit_code = host_exit_code(status);
        if entry.self_run {
            let drained = entry
                .diag
                .take()
                .map(|t| t.join().unwrap_or_default())
                .unwrap_or_default();
            if let Some(file) = &entry.stderr_file {
                let _ = std::fs::remove_file(file);
            }
            if join_relay {
                // A blocking wait leaves no relay in flight:
                // the wired stream's last bytes have reached
                // the guest's file and the relay's duplicate
                // of its end is closed, as a native wait
                // leaves the child's ends closed with it.
                // (The relay ends once the diagnostic drain
                // has completed, which the join above has
                // already observed.)
                if let Some(relay) = entry.relay.take() {
                    let _ = relay.join();
                }
            }
            if let Some(reason) =
                self_run_failure(exit_code, &drained)
            {
                return Err(HostErr::Failed(reason));
            }
        }
        entry.exit = Some(exit_code);
        Ok(exit_code)
    }
}

/// `signal(2)` from the libc the driver links, named
/// directly: the process host needs this one call in a
/// spawned child (see `assemble`), and the driver takes
/// no libc dependency for it.
#[cfg(unix)]
extern "C" {
    #[link_name = "signal"]
    fn host_signal(signum: i32, handler: usize) -> usize;
    #[link_name = "setgroups"]
    fn host_setgroups(count: usize, groups: *const u32) -> i32;
}

impl sparagmos_interp::ProcessHostOps for DriverProcessHost {
    fn spawn(
        &self,
        request: sparagmos_interp::ProcessSpawnRequest,
    ) -> Result<
        sparagmos_interp::ProcessSpawned,
        sparagmos_interp::ProcessHostError,
    > {
        use sparagmos_interp::ProcessHostError as HostErr;
        let (mut cmd, self_run, stderr_file) =
            self.assemble(&request)?;
        cmd.stdin(wire_stdio(request.stdin));
        // A stream the guest wired to a file is wired into
        // the child directly, at spawn: its bytes move
        // through the guest's file while the child runs --
        // a guest reading the other end of a pipe it wired
        // observes them, and end-of-file, as the child
        // produces them, not a collection delivered when
        // the child's end is observed. `Stdio::from` takes
        // the request's duplicate of the wired end; the
        // spawn dups it into the child and this process
        // keeps no copy of it, so no duplicated writer end
        // outlives the child. Any other stream is piped and
        // its end handed to the interpreter, as before.
        let mut stdout_wired = false;
        match request.stdout {
            sparagmos_interp::ProcessStdio::Fd(file) => {
                cmd.stdout(std::process::Stdio::from(file));
                stdout_wired = true;
            }
            _ => {
                cmd.stdout(std::process::Stdio::piped());
            }
        }
        // A self-child's stderr is the exception: the
        // nested process's own stderr is the diagnostic
        // pipe, and its guest stderr arrives through the
        // channel file, so a wired file is served by a
        // relay carrying that stream into the file (see
        // below) rather than by the child's descriptor.
        let mut stderr_target = None;
        let mut stderr_wired = false;
        match request.stderr {
            sparagmos_interp::ProcessStdio::Fd(file) => {
                if self_run {
                    stderr_target = Some(file);
                    cmd.stderr(std::process::Stdio::piped());
                } else {
                    cmd.stderr(std::process::Stdio::from(file));
                    stderr_wired = true;
                }
            }
            _ => {
                cmd.stderr(std::process::Stdio::piped());
            }
        }
        if self_run {
            // The guest-stderr channel file exists before the
            // child starts, so the stream feeding the guest
            // can open it now; the nested driver writes
            // through its own handle on the same file.
            if let Some(file) = &stderr_file {
                let _ = std::fs::File::create(file);
            }
        }
        let mut child = cmd.spawn().map_err(HostErr::Io)?;
        let pid = child.id();
        let stdin = child.stdin.take().map(pipe_file);
        let stdout = if stdout_wired {
            None
        } else {
            child.stdout.take().map(pipe_file)
        };
        let (stderr, diag, relay) = if self_run {
            let pipe = child.stderr.take().expect("piped stderr");
            // The diagnostic stream closes when the nested
            // process ends; the drain's completion is what
            // ends the guest's stderr stream (see
            // `guest_stderr_stream`).
            let diag_done = std::sync::Arc::new(
                std::sync::atomic::AtomicBool::new(false),
            );
            let flag = diag_done.clone();
            let drain = std::thread::spawn(move || {
                let drained = drain_pipe(pipe);
                flag.store(
                    true,
                    std::sync::atomic::Ordering::SeqCst,
                );
                drained
            });
            let guest = stderr_file.as_ref().and_then(|file| {
                guest_stderr_stream(file, diag_done)
            });
            match (guest, stderr_target) {
                (Some(tail), Some(target)) => {
                    // The guest wired this child's stderr to
                    // a file: the guest-stderr stream is
                    // relayed into it while the child runs,
                    // on a thread owning the request's
                    // duplicate of the wired end, dropped
                    // when the stream ends (see
                    // `relay_stream`).
                    let relay = std::thread::spawn(move || {
                        relay_stream(tail, target)
                    });
                    (None, Some(drain), Some(relay))
                }
                (guest, _) => (guest, Some(drain), None),
            }
        } else if stderr_wired {
            (None, None, None)
        } else {
            (child.stderr.take().map(pipe_file), None, None)
        };
        let id = {
            let mut next = self.next.lock().unwrap();
            let id = *next;
            *next += 1;
            id
        };
        self.children.lock().unwrap().insert(
            id,
            HostChild {
                child,
                diag,
                relay,
                stderr_file,
                self_run,
                exit: None,
            },
        );
        Ok(sparagmos_interp::ProcessSpawned {
            host_id: id,
            pid,
            stdin,
            stdout,
            stderr,
        })
    }

    fn run(
        &self,
        request: sparagmos_interp::ProcessSpawnRequest,
    ) -> Result<
        sparagmos_interp::ProcessChildOutcome,
        sparagmos_interp::ProcessHostError,
    > {
        use sparagmos_interp::ProcessHostError as HostErr;
        let (mut cmd, self_run, stderr_file) =
            self.assemble(&request)?;
        cmd.stdin(wire_stdio(request.stdin));
        // A stream the guest wired to a file is wired into
        // the child directly, as at spawn (see `spawn`): its
        // bytes move through the guest's file while the
        // child runs and the collected half of the outcome
        // is empty, as std's collected output for a wired
        // stream is. A self-child's stderr keeps the
        // channel-file path (see `finalize`): the nested
        // process's own stderr is the diagnostic pipe.
        let mut stdout_wired = false;
        match request.stdout {
            sparagmos_interp::ProcessStdio::Fd(file) => {
                cmd.stdout(std::process::Stdio::from(file));
                stdout_wired = true;
            }
            _ => {
                cmd.stdout(std::process::Stdio::piped());
            }
        }
        let mut stderr_wired = false;
        match request.stderr {
            sparagmos_interp::ProcessStdio::Fd(file)
                if !self_run =>
            {
                cmd.stderr(std::process::Stdio::from(file));
                stderr_wired = true;
            }
            _ => {
                cmd.stderr(std::process::Stdio::piped());
            }
        }
        let mut child = cmd.spawn().map_err(HostErr::Io)?;
        let stdout = if stdout_wired {
            None
        } else {
            let out_pipe =
                child.stdout.take().expect("piped stdout");
            Some(std::thread::spawn(move || drain_pipe(out_pipe)))
        };
        let stderr = if stderr_wired {
            None
        } else {
            let err_pipe =
                child.stderr.take().expect("piped stderr");
            Some(std::thread::spawn(move || drain_pipe(err_pipe)))
        };
        let status = child.wait().map_err(HostErr::Io)?;
        let mut entry = RunChild {
            stdout,
            stderr,
            stderr_file,
            self_run,
        };
        self.finalize(&mut entry, status)
    }

    fn wait(
        &self,
        child: u64,
    ) -> Result<i32, sparagmos_interp::ProcessHostError> {
        use sparagmos_interp::ProcessHostError as HostErr;
        let mut children = self.children.lock().unwrap();
        let entry = children
            .get_mut(&child)
            .ok_or_else(|| HostErr::Failed("stale host child".into()))?;
        if let Some(code) = entry.exit {
            return Ok(code);
        }
        let status = entry.child.wait().map_err(HostErr::Io)?;
        self.observe_end(entry, status, true)
    }

    fn try_wait(
        &self,
        child: u64,
    ) -> Result<Option<i32>, sparagmos_interp::ProcessHostError> {
        use sparagmos_interp::ProcessHostError as HostErr;
        let mut children = self.children.lock().unwrap();
        let entry = children
            .get_mut(&child)
            .ok_or_else(|| HostErr::Failed("stale host child".into()))?;
        if let Some(code) = entry.exit {
            return Ok(Some(code));
        }
        match entry.child.try_wait().map_err(HostErr::Io)? {
            None => Ok(None),
            Some(status) => {
                self.observe_end(entry, status, false).map(Some)
            }
        }
    }

    fn kill(
        &self,
        child: u64,
    ) -> Result<(), sparagmos_interp::ProcessHostError> {
        use sparagmos_interp::ProcessHostError as HostErr;
        let mut children = self.children.lock().unwrap();
        let entry = children
            .get_mut(&child)
            .ok_or_else(|| HostErr::Failed("stale host child".into()))?;
        entry.child.kill().map_err(HostErr::Io)
    }

    fn release(&self, child: u64) {
        // Dropping the host `Child` leaves the process itself
        // running, as a dropped native `Child` does.
        self.children.lock().unwrap().remove(&child);
    }
}
