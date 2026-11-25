//! Host-facing interfaces: tool interception and the process host.



/// A request to run a configured tool program, handed from the
/// interpreter to the host-side tool runner (see
/// [`InterpConfig::tool_interception`]). The fields mirror the
/// interpreter's command-table entry for the invocation: the
/// program and its arguments as raw byte strings, the environment
/// entries the guest set on the command (a `None` value is a
/// removal), whether the inherited environment was cleared, and
/// the child's working directory if the guest set one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRunRequest {
    pub program: Vec<u8>,
    pub args: Vec<Vec<u8>>,
    pub argv0: Option<Vec<u8>>,
    pub envs: Vec<(Vec<u8>, Option<Vec<u8>>)>,
    pub env_clear: bool,
    pub cwd: Option<Vec<u8>>,
    /// The program path the interpreter's own spawn
    /// resolution (the process model's `execvp` search)
    /// settled on for this invocation, when it resolved:
    /// the path the host side must look its answer up by.
    /// For a bare program name invoked under a
    /// command-specific `PATH` this differs from the
    /// spelled `program` -- the artifact the search finds
    /// is the one the invocation denotes, not a namesake
    /// in the working directory. `program` itself stays
    /// as spelled: it is what the program reads as its
    /// argv[0].
    pub resolved_program: Option<Vec<u8>>,
    /// Whether the invocation collects the program's output
    /// (`Command::output`: the streams are piped, and the
    /// outcome's bytes are the caller's to keep) or inherits
    /// them (`Command::status`: the streams are the guest's
    /// own, and the interpreter relays the outcome's bytes
    /// into them before it reports the status). The runner
    /// produces the outcome's bytes either way; the mode is
    /// part of the request so the host side answers for the
    /// invocation the guest actually made.
    pub output_captured: bool,
}

/// The outcome of one tool invocation, from which the fabricated
/// `std::process::Output` / `ExitStatus` values are built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRunOutcome {
    /// The child's exit code; negative when the child ended on a
    /// signal (the fabricated wait status encodes it as such).
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// The host-side tool runner: invoked by the interpreter when the
/// guest runs one of the configured tool programs through
/// `Command::output` / `Command::status`. An `Err` return becomes
/// the run's unsupported error, with the returned text as the
/// reason.
#[derive(Clone)]
pub struct ToolRunner(
    pub std::sync::Arc<
        dyn Fn(&ToolRunRequest) -> Result<ToolRunOutcome, String> + Send + Sync,
    >,
);

impl std::fmt::Debug for ToolRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ToolRunner(..)")
    }
}

impl PartialEq for ToolRunner {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ToolRunner {}

/// Tool interception configured for a run: the exact program byte
/// strings the interpreter routes to `runner` instead of answering
/// with the process model's named unsupported error. This is how a
/// test recipe's compiler invocations reach the driver's own
/// conversion machinery: the program named here is the compiler,
/// and running it means converting the guest's input with the
/// driver, not starting a process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolInterception {
    pub programs: Vec<Vec<u8>>,
    pub runner: ToolRunner,
    /// The runner for programs that are NOT in `programs`, when
    /// the host side can still answer for some of them (the
    /// driver answers for the artifacts a recipe produced
    /// earlier in the same run: running one means interpreting
    /// its recorded source). A fallback `Err` becomes the run's
    /// unsupported error with the returned text as the reason,
    /// exactly as a `runner` error does; with no fallback
    /// configured, an unlisted program keeps the process model's
    /// standing unsupported answer.
    pub fallback: Option<ToolRunner>,
}

/// How one stdio stream of a spawned child is wired, as the
/// command's recorded configuration prescribes (see
/// [`ProcessSpawnRequest`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessStdio {
    Inherit,
    Null,
    Piped,
}

/// A request to start one child process, handed from the
/// interpreter to the process host (see
/// [`InterpConfig::process_host`]). The fields mirror the
/// interpreter's command-table entry for the invocation, in the
/// same shape as [`ToolRunRequest`], plus the stdio wiring the
/// entry point's semantics and the command's recorded
/// configuration settle on. The host starts the program the
/// invocation denotes: `resolved_program` (the interpreter's
/// own `execvp` search result) when it resolved, else the
/// spelled `program`, resolved by the host under the child's
/// working directory and environment, as a native spawn
/// resolves it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessSpawnRequest {
    pub program: Vec<u8>,
    pub args: Vec<Vec<u8>>,
    pub argv0: Option<Vec<u8>>,
    pub envs: Vec<(Vec<u8>, Option<Vec<u8>>)>,
    pub env_clear: bool,
    pub cwd: Option<Vec<u8>>,
    pub resolved_program: Option<Vec<u8>>,
    pub stdin: ProcessStdio,
    pub stdout: ProcessStdio,
    pub stderr: ProcessStdio,
}

/// The outcome of one finished child process: its exit code
/// (negative when the child ended on a signal) and the bytes
/// its stdout and stderr produced. The host always captures
/// both streams; the interpreter routes them by the command's
/// stdio configuration (collected into the guest's `Output`,
/// relayed into the guest's own streams, or discarded).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessChildOutcome {
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// Why a process-host operation produced no outcome: the
/// host's own `io` failure, which the guest observes as the
/// `io::Error` a native spawn/wait would report, or a failure
/// the run cannot carry out, which becomes the run's
/// unsupported error with the text as the reason.
#[derive(Debug)]
pub enum ProcessHostError {
    Io(std::io::Error),
    Failed(String),
}

/// What starting one child hands back across the process-host
/// boundary (see [`ProcessHostOps::spawn`]): the host's child
/// id, the child's process id, and the pipe ends the guest's
/// stdio handles denote -- one host file per stream the wiring
/// piped (the child's stdin write end, its stdout and stderr
/// read ends). The interpreter registers the files in its file
/// table, so the guest's `ChildStdin` / `ChildStdout` /
/// `ChildStderr` handles are ordinary file handles: reads and
/// writes go through the file model, and dropping a handle
/// closes its end of the pipe, as a native handle drop does.
/// A stream the wiring did not pipe (an inherited or nulled
/// stdin) hands back no file.
pub struct ProcessSpawned {
    pub host_id: u64,
    pub pid: u32,
    pub stdin: Option<std::fs::File>,
    pub stdout: Option<std::fs::File>,
    pub stderr: Option<std::fs::File>,
}

/// The host side of real child processes: the operations the
/// interpreter's `Child` model performs against live children
/// the host started. Child ids are the host's own, handed back
/// at spawn; the interpreter's child table maps its guest
/// handles onto them.
pub trait ProcessHostOps: Send + Sync {
    /// Start the child the request describes and hand back
    /// its pipe ends (see [`ProcessSpawned`]): the child keeps
    /// running; the interpreter drives it through the other
    /// operations and reads and writes its streams through
    /// the returned files.
    fn spawn(
        &self,
        request: &ProcessSpawnRequest,
    ) -> Result<ProcessSpawned, ProcessHostError>;
    /// Run the child the request describes to completion and
    /// collect its outcome: the spawn-and-collect path behind
    /// `Command::status` / `Command::output`, where no guest
    /// `Child` handle exists and the host drains the streams
    /// while the child runs, as std's collection does.
    fn run(
        &self,
        request: &ProcessSpawnRequest,
    ) -> Result<ProcessChildOutcome, ProcessHostError>;
    /// Wait for the child to end; returns its exit code (the
    /// [`ProcessChildOutcome`] convention: negative when the
    /// child ended on a signal). The streams are not collected
    /// here: their bytes move through the files `spawn`
    /// returned.
    fn wait(&self, child: u64) -> Result<i32, ProcessHostError>;
    /// The child's exit code if it has already ended; `None`
    /// while it is still running.
    fn try_wait(&self, child: u64)
        -> Result<Option<i32>, ProcessHostError>;
    /// Kill the child (SIGKILL semantics, as `Child::kill`).
    fn kill(&self, child: u64) -> Result<(), ProcessHostError>;
    /// Forget the host child without waiting for it (the
    /// guest's `Child` value was dropped, or its outcome was
    /// collected): the child itself is left alone, exactly as
    /// a dropped native `Child` leaves it running.
    fn release(&self, child: u64);
}

/// The process host configured for a run: an
/// [`InterpConfig::process_host`] value the interpreter calls
/// when the guest starts a child process no tool interception
/// claims.
#[derive(Clone)]
pub struct ProcessHost(pub std::sync::Arc<dyn ProcessHostOps>);

impl std::fmt::Debug for ProcessHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ProcessHost(..)")
    }
}

impl PartialEq for ProcessHost {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ProcessHost {}

/// The write-through target for the guest's standard output:
/// the file behind a run's own stdout (see
/// [`InterpConfig::stdout_sink`]). Writes go through
/// `&File`, so the interpreter's shared borrow of its
/// configuration is enough.
#[derive(Clone)]
pub struct StdoutSink(pub std::sync::Arc<std::fs::File>);

impl std::fmt::Debug for StdoutSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StdoutSink(..)")
    }
}

impl PartialEq for StdoutSink {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for StdoutSink {}

/// The file-table id of the guest's standard input (see
/// `Interpreter::stdin_file_id`): the table hands out ids
/// from 3 up for files the guest opens (the descriptor
/// numbers a native run reserves), so 0 names the duplicated
/// stdin without colliding.
pub(crate) const STDIN_FILE_ID: u64 = 0;
