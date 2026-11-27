//! External tool programs: the registry, subprocess conversion runs, and guest-environment plumbing.

use super::*;

/// One artifact a tool invocation produced, as the registry
/// remembers it: the name it was built under, the source it was
/// converted from, and the attributes the producing invocation
/// carried -- which is what a later `--extern` naming the
/// artifact binds through `--aux`.
struct ToolArtifact {
    name: String,
    source: std::path::PathBuf,
    /// The source tree as it stood when the producing
    /// invocation converted it (see `SourceTree`): a recipe
    /// may rewrite or delete the source files after
    /// building, and a later run of the artifact must still
    /// execute the program its build produced (see
    /// `ToolRegistry::effective_source`).
    source_tree: SourceTree,
    /// The working directory of the producing invocation:
    /// the directory its relative compilation paths (a
    /// forwarded `--extern` path, a `-L` directory) were
    /// resolved against when it converted. A later run of
    /// the artifact re-converts under this directory, not
    /// the execution's, so those paths denote the same
    /// files at both times (see `artifact_run`).
    build_cwd: std::path::PathBuf,
    edition: Option<String>,
    crate_type: Option<String>,
    /// The compilation environment the producing invocation
    /// ran under (its request's environment assignments and
    /// clear flag): what a later run of the artifact (see
    /// `artifact_run`) spawns its re-conversion with, so
    /// compile-time environment reads (`option_env!`, `env!`)
    /// see the values the build saw, while the interpreted
    /// program itself runs under the execution request's
    /// environment.
    compile_env_clear: bool,
    compile_envs: Vec<(Vec<u8>, Option<Vec<u8>>)>,
    /// The `--aux` specs the producing invocation itself
    /// carried (its `--extern` bindings, already translated):
    /// what a later run of the artifact (see `artifact_run`)
    /// hands to the run subprocess so the artifact's source
    /// converts against the same libraries its build used.
    aux_specs: Vec<String>,
    /// The dependency version each entry of `aux_specs`
    /// bound when this artifact's build linked it, in the
    /// same order: the dependency artifact's source path
    /// and the source tree its build converted. A later run of
    /// this artifact resolves its `--aux` specs against
    /// these pins, not against the registry as it then
    /// stands: the registry replaces a rebuilt artifact's
    /// entry and keys nothing by version, so resolving by
    /// source path at execution time would hand the run
    /// whatever build of the dependency happens to be
    /// current then -- a library rebuilt after this
    /// artifact was, or another output built from the
    /// same source.
    aux_pins: Vec<(std::path::PathBuf, SourceTree)>,
    /// The compilation-shaping options the producing
    /// invocation carried (its forwarded rustc arguments --
    /// `--cfg`, `--check-cfg`, codegen and lint options, and
    /// the rest of that family): what a later run of the
    /// artifact (see `artifact_run`) replays through
    /// `--rustc-arg` so the recorded source converts under
    /// the same configuration its build used, not merely
    /// against the same libraries.
    rustc_args: Vec<String>,
    /// The identity (device and inode) of the placeholder
    /// file the producing invocation left at its output
    /// path, as it stood when the build recorded it: how a
    /// later execution lookup tells this artifact's file
    /// from another's after the recipe renames one over the
    /// other (see `ToolRegistry::get_for_run`).
    file_id: Option<(u64, u64)>,
}

/// The artifact registry of one run (see `execute`): artifact
/// paths in the order the guest produced them (a recipe builds
/// a crate before naming it as an `--extern` input, so build
/// order is dependency order).
#[derive(Default)]
pub(crate) struct ToolRegistry {
    entries: std::sync::Mutex<Vec<(std::path::PathBuf, ToolArtifact)>>,
    /// Source trees materialised to directories, keyed by
    /// the artifact's recorded source path, with the tree's
    /// content hash: a rebuilt artifact at the same path
    /// with a different tree gets a different directory.
    /// The value is the materialised root file's path.
    snapshot_files:
        std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, (u64, std::path::PathBuf)>>,
}

/// A detached copy of one recorded artifact: the registry
/// hands these out, so a caller can hold an artifact while
/// the registry keeps accepting records.
fn cloned_artifact(a: &ToolArtifact) -> ToolArtifact {
    ToolArtifact {
        name: a.name.clone(),
        source: a.source.clone(),
        source_tree: a.source_tree.clone(),
        build_cwd: a.build_cwd.clone(),
        edition: a.edition.clone(),
        crate_type: a.crate_type.clone(),
        compile_env_clear: a.compile_env_clear,
        compile_envs: a.compile_envs.clone(),
        aux_specs: a.aux_specs.clone(),
        aux_pins: a.aux_pins.clone(),
        rustc_args: a.rustc_args.clone(),
        file_id: a.file_id,
    }
}

/// A host string from a byte string the interpreter's
/// model carries (an environment name or value, a working
/// directory). On Unix the bytes are the host string
/// itself. On other hosts the byte strings this driver
/// carries are UTF-8 text -- the form the guest
/// environment file and the tool plumbing produce there --
/// and the host string is that text.
fn os_string_from_bytes(bytes: &[u8]) -> std::ffi::OsString {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt as _;
        std::ffi::OsStr::from_bytes(bytes).to_os_string()
    }
    #[cfg(not(unix))]
    {
        std::ffi::OsString::from(String::from_utf8_lossy(bytes).into_owned())
    }
}

/// The identity (device and inode numbers) of the file at
/// `path`: what distinguishes one artifact's placeholder
/// file from another's after a recipe moves files around,
/// given every placeholder holds the same (empty) bytes.
/// `None` when the file cannot be stat'ed, or when the
/// host exposes no such identity for a file.
fn file_identity(path: &std::path::Path) -> Option<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        let md = std::fs::metadata(path).ok()?;
        Some((md.dev(), md.ino()))
    }
    #[cfg(not(unix))]
    {
        // The caller falls back to the path record alone,
        // as it does for a file that cannot be stat'ed.
        let _ = path;
        None
    }
}

impl ToolRegistry {
    fn get(&self, path: &std::path::Path) -> Option<ToolArtifact> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, a)| cloned_artifact(a))
    }

    /// The artifact an execution of the file at `path` must
    /// run. The registry's records are keyed by the path a
    /// build produced its output at, but a recipe can move
    /// files after building them -- `std::fs::rename` of one
    /// artifact over another's path is the plain case -- and
    /// the interpreter's filesystem builtins perform such
    /// moves directly on the host, with no word to this
    /// registry. So the lookup follows the file itself: when
    /// the file now at `path` carries an identity a build
    /// recorded, that build's artifact is the program the
    /// path now denotes, wherever it was recorded under.
    /// When the path's recorded artifact is no longer the
    /// file there and no recorded identity matches, what
    /// stands there is not an artifact this run produced,
    /// and the execution is refused rather than answered
    /// with a stale program. When the file cannot be
    /// stat'ed (it is gone), the path record alone decides,
    /// as before.
    fn get_for_run(&self, path: &std::path::Path) -> Result<Option<ToolArtifact>, String> {
        let entries = self.entries.lock().unwrap();
        if let Some(id) = file_identity(path) {
            if let Some((_, a)) = entries.iter().find(|(_, a)| a.file_id == Some(id)) {
                return Ok(Some(cloned_artifact(a)));
            }
            if entries.iter().any(|(p, _)| p == path) {
                return Err(format!(
                    "tool interception: the file at `{}` was replaced after it was built; what stands there is not an artifact this run produced",
                    path.display()
                ));
            }
            return Ok(None);
        }
        Ok(entries
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, a)| cloned_artifact(a)))
    }

    /// The source a conversion of `artifact` must read: the
    /// recorded path itself while every file of the recorded
    /// tree still holds the bytes the build converted (the
    /// common case, and the one whose `file!` spellings stay
    /// the build's), and a materialised snapshot of the
    /// recorded tree -- relative layout preserved, so the
    /// root's `mod` and `include!` inputs stand beside it --
    /// once the recipe has rewritten or deleted any of it.
    fn effective_source(&self, artifact: &ToolArtifact) -> std::path::PathBuf {
        self.effective_source_of(&artifact.source, &artifact.source_tree)
    }

    /// `effective_source` for a bare (source path, tree)
    /// pair: an artifact's own source, or one recorded
    /// dependency pin (see `ToolArtifact::aux_pins`).
    fn effective_source_of(
        &self,
        source: &std::path::Path,
        tree: &SourceTree,
    ) -> std::path::PathBuf {
        if source_tree_intact(source, tree) {
            return source.to_path_buf();
        }
        use std::hash::Hasher as _;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for (rel, bytes) in tree {
            hasher.write(rel.as_os_str().as_encoded_bytes());
            hasher.write(bytes);
        }
        let hash = hasher.finish();
        let mut files = self.snapshot_files.lock().unwrap();
        if let Some((h, path)) = files.get(source) {
            if *h == hash {
                return path.clone();
            }
        }
        let root_name = source
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "source.rs".to_string());
        let base =
            std::env::temp_dir().join(format!("sparagmos-artifact-sources-{}", std::process::id()));
        // The tree is read only by this process and by the
        // child driver processes it waits for, so it is dead
        // once this process exits: register the base for
        // removal on every exit path (#537).
        register_process_scratch(&base);
        let dir = base.join(format!("{hash:016x}"));
        for (rel, bytes) in tree {
            let target = dir.join(rel);
            if let Some(parent) = target.parent() {
                // The materialised tree copies every recorded
                // neighbouring file verbatim -- a private file
                // beside the crate root among them -- so it
                // must not stand in the shared temporary
                // directory under the default permissions:
                // keep every directory of it owner-only. The
                // mode is set at creation time, not tightened
                // afterwards: a create-then-tighten sequence
                // leaves a window in which a concurrent
                // observer sees the tree at the default mode
                // (a parallel test run caught exactly that in
                // CI). The restriction below is still applied
                // on each visit, so a directory an older run
                // left at this pid's name is tightened too.
                let mut builder = std::fs::DirBuilder::new();
                builder.recursive(true);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::DirBuilderExt as _;
                    builder.mode(0o700);
                }
                let _ = builder.create(parent);
                for ancestor in parent.ancestors() {
                    if !ancestor.starts_with(&base) {
                        break;
                    }
                    restrict_to_owner(ancestor, 0o700);
                    if ancestor == base {
                        break;
                    }
                }
            }
            // Owner-only from the moment the file exists,
            // and tightened again after the write in case a
            // file of that name already stood there.
            use std::io::Write as _;
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create(true).truncate(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt as _;
                options.mode(0o600);
            }
            let _ = options
                .open(&target)
                .and_then(|mut f| f.write_all(bytes));
            restrict_to_owner(&target, 0o600);
        }
        let path = dir.join(root_name);
        files.insert(source.to_path_buf(), (hash, path.clone()));
        path
    }

    /// The path-prefix remap a conversion reading the
    /// source at `source` from its snapshot needs, so the
    /// converted program's source-location spellings
    /// (`file!` and its kin) stay the build's: when the
    /// recorded tree no longer stands intact, the
    /// conversion reads the materialised snapshot under
    /// the returned scratch directory, while the build
    /// read the same relative layout under the recorded
    /// source's own directory -- the pair to hand rustc as
    /// `--remap-path-prefix <from>=<to>`. `None` when the
    /// recorded path itself is read (the intact case,
    /// whose spellings already are the build's).
    fn snapshot_remap(
        &self,
        source: &std::path::Path,
        tree: &SourceTree,
    ) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        if source_tree_intact(source, tree) {
            return None;
        }
        let materialised = self.effective_source_of(source, tree);
        let from = materialised.parent()?.to_path_buf();
        let to = source.parent()?.to_path_buf();
        Some((from, to))
    }

    /// One `--aux` spec with its source path routed
    /// through a recorded dependency pin rather than the
    /// registry's current state: the conversion the spec
    /// feeds reads the tree the pinned build converted,
    /// even if the dependency has since been rebuilt (at
    /// the same output path, or as another output from
    /// the same source).
    fn pinned_aux_spec(&self, spec: &str, pin: &(std::path::PathBuf, SourceTree)) -> String {
        let Some((name, rest)) = spec.split_once('=') else {
            return spec.to_string();
        };
        let attrs = match rest.split_once(';') {
            Some((_, a)) => format!(";{a}"),
            None => String::new(),
        };
        format!(
            "{}={}{}",
            name,
            self.effective_source_of(&pin.0, &pin.1).display(),
            attrs
        )
    }

    /// `specs` with each entry routed through its
    /// recorded pin (see `pinned_aux_spec`). Specs and
    /// pins are recorded together by `tool_run`, so a
    /// spec without a pin does not arise for a recorded
    /// artifact; should one appear anyway it stands as
    /// recorded -- no resolution against the registry's
    /// current state happens anywhere on this path.
    fn pinned_aux_specs(
        &self,
        specs: &[String],
        pins: &[(std::path::PathBuf, SourceTree)],
    ) -> Vec<String> {
        specs
            .iter()
            .enumerate()
            .map(|(i, s)| match pins.get(i) {
                Some(pin) => self.pinned_aux_spec(s, pin),
                None => s.clone(),
            })
            .collect()
    }

    /// The `--aux` specs a run of `artifact` hands its
    /// subprocess: the specs its build recorded, each
    /// resolved against the dependency version that
    /// build bound (see `ToolArtifact::aux_pins`).
    fn artifact_aux_specs(&self, artifact: &ToolArtifact) -> Vec<String> {
        self.pinned_aux_specs(&artifact.aux_specs, &artifact.aux_pins)
    }

    /// The recorded artifact a bound `--aux` spec denotes:
    /// the artifact whose build the spec was taken from,
    /// identified by its pinned source and by the spec the
    /// artifact itself renders under the spec's binding name
    /// (a rebuild of the same source renders the same spec;
    /// the first match stands).
    fn artifact_for_spec(
        &self,
        spec: &str,
        pin: &(std::path::PathBuf, SourceTree),
    ) -> Option<ToolArtifact> {
        let name = spec.split('=').next().unwrap_or("");
        self.entries
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, a)| a.source == pin.0)
            .find(|(_, a)| aux_spec_for(name, a) == spec)
            .map(|(_, a)| cloned_artifact(a))
    }

    fn record(&self, path: std::path::PathBuf, artifact: ToolArtifact) {
        let mut entries = self.entries.lock().unwrap();
        if let Some(slot) = entries.iter_mut().find(|(p, _)| *p == path) {
            slot.1 = artifact;
        } else {
            entries.push((path, artifact));
        }
    }

    fn all(&self) -> Vec<(std::path::PathBuf, ToolArtifact)> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .map(|(p, a)| (p.clone(), cloned_artifact(a)))
            .collect()
    }
}

/// The per-invocation cap on one conversion subprocess: a
/// conversion that has not finished by then is not going to,
/// and the recipe's own harness applies its (larger) budget to
/// the whole run.
const TOOL_RUN_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(180);

/// An absolute form of `path`, resolved against `cwd` when the
/// guest spelled it relative. Lexical only: the registry keys
/// are built the same way, so the spellings a recipe uses for
/// an artifact it produced meet the keys it looks up later.
fn absolutize(cwd: &std::path::Path, path: &str) -> std::path::PathBuf {
    let p = std::path::PathBuf::from(path);
    if p.is_absolute() {
        p
    } else {
        cwd.join(p)
    }
}

/// The file name a real rustc gives the artifact of one crate
/// type: the crate name with the build's `-C extra-filename`
/// suffix verbatim between it and the extension ("" when the
/// invocation set none).
fn artifact_filename(crate_name: &str, crate_type: &str, extra_filename: &str) -> String {
    match crate_type {
        "bin" => format!("{crate_name}{extra_filename}"),
        "dylib" | "cdylib" | "proc-macro" => {
            format!("lib{crate_name}{extra_filename}.so")
        }
        "staticlib" => format!("lib{crate_name}{extra_filename}.a"),
        "rmeta" => format!("lib{crate_name}{extra_filename}.rmeta"),
        _ => format!("lib{crate_name}{extra_filename}.rlib"),
    }
}

/// The `--aux` spec binding one registered artifact.
fn aux_spec_for(name: &str, artifact: &ToolArtifact) -> String {
    let mut spec = format!("{}={}", name, artifact.source.display());
    if let Some(edition) = &artifact.edition {
        spec += &format!(";edition={edition}");
    }
    if let Some(crate_type) = &artifact.crate_type {
        spec += &format!(";crate-type={crate_type}");
    }
    if carried_force_unstable(&artifact.rustc_args) {
        spec += ";force-unstable=yes";
    }
    spec
}

/// Whether a forwarded rustc argument list carries
/// `-Zforce-unstable-if-unmarked`, in either its glued
/// one-token spelling or the split `-Z` + name pair.
fn carried_force_unstable(args: &[String]) -> bool {
    let mut prev_dash_z = false;
    for arg in args {
        if arg == "-Zforce-unstable-if-unmarked"
            || (prev_dash_z && arg == "force-unstable-if-unmarked")
        {
            return true;
        }
        prev_dash_z = arg == "-Z";
    }
    false
}

/// Bind one registered artifact into a conversion's `aux`
/// list under `name`, preceded by the bindings its own build
/// recorded (see `ToolArtifact::aux_specs`): the artifact's
/// auxiliary build inside the conversion declares those
/// dependencies by name, and they must stand among the
/// conversion's auxiliary crates, in dependency order, for
/// the build to find them -- the chain a real compiler
/// resolves through the artifacts' own metadata and the
/// `-L` directory. The expansion is recursive (a
/// dependency's own dependencies precede it) and each
/// distinct spec is bound once.
fn bind_artifact_with_deps(
    registry: &ToolRegistry,
    name: &str,
    artifact: &ToolArtifact,
    aux: &mut Vec<String>,
    aux_pins: &mut Vec<(std::path::PathBuf, SourceTree)>,
) {
    fn push_spec(
        registry: &ToolRegistry,
        spec: &str,
        pin: &(std::path::PathBuf, SourceTree),
        owner: Option<&ToolArtifact>,
        aux: &mut Vec<String>,
        aux_pins: &mut Vec<(std::path::PathBuf, SourceTree)>,
        depth: usize,
    ) {
        if depth > 32 || aux.iter().any(|s| s == spec) {
            return;
        }
        if let Some(owner) = owner {
            let deps: Vec<(String, (std::path::PathBuf, SourceTree))> = owner
                .aux_specs
                .iter()
                .cloned()
                .zip(owner.aux_pins.iter().cloned())
                .collect();
            for (dep_spec, dep_pin) in &deps {
                let dep_owner = registry.artifact_for_spec(dep_spec, dep_pin);
                push_spec(
                    registry,
                    dep_spec,
                    dep_pin,
                    dep_owner.as_ref(),
                    aux,
                    aux_pins,
                    depth + 1,
                );
            }
        }
        aux.push(spec.to_string());
        aux_pins.push(pin.clone());
    }
    let spec = aux_spec_for(name, artifact);
    let pin = (artifact.source.clone(), artifact.source_tree.clone());
    push_spec(registry, &spec, &pin, Some(artifact), aux, aux_pins, 0);
}

/// The subprocess's stderr with the driver's own report lines
/// removed: what remains is the compiler's diagnostics (and,
/// for `--print` invocations, the printed output, which the
/// session discipline routes to stderr), which is what a real
/// compiler child would have produced.
fn child_stream(stderr: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(stderr);
    let mut out = String::new();
    for line in text.lines() {
        if !line.starts_with("sparagmos-driver:") {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.into_bytes()
}

/// Run this driver binary as a subprocess (convert mode for a
/// tool invocation, run mode for an artifact invocation) and
/// collect (exit code, stdout, stderr). The guest command's
/// environment is applied over the inherited one, as a real
/// spawn would apply it.
fn run_convert_subprocess(
    exe: &std::path::Path,
    tail: &[String],
    cwd: &std::path::Path,
    env_clear: bool,
    envs: &[(Vec<u8>, Option<Vec<u8>>)],
) -> Result<(i32, Vec<u8>, Vec<u8>), String> {
    use std::io::Read as _;
    let mut cmd = std::process::Command::new(exe);
    cmd.args(tail)
        .current_dir(cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if env_clear {
        cmd.env_clear();
    }
    for (key, value) in envs {
        match value {
            Some(v) => {
                cmd.env(os_string_from_bytes(key), os_string_from_bytes(v));
            }
            None => {
                cmd.env_remove(os_string_from_bytes(key));
            }
        }
    }
    let mut child = cmd.spawn().map_err(|e| {
        format!("tool interception: cannot start the tool subprocess: {e}")
    })?;
    let mut out_pipe = child.stdout.take().expect("piped stdout");
    let mut err_pipe = child.stderr.take().expect("piped stderr");
    let out_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = out_pipe.read_to_end(&mut buf);
        buf
    });
    let err_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = err_pipe.read_to_end(&mut buf);
        buf
    });
    let deadline = std::time::Instant::now() + TOOL_RUN_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if std::time::Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(
                        "tool interception: the tool subprocess timed out"
                            .to_string(),
                    );
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(e) => {
                return Err(format!(
                    "tool interception: waiting on the tool subprocess failed: {e}"
                ));
            }
        }
    };
    let stdout = out_thread.join().unwrap_or_default();
    let stderr = err_thread.join().unwrap_or_default();
    Ok((status.code().unwrap_or(101), stdout, stderr))
}

/// Create a fresh file in the shared temporary directory
/// that no other account can read or predict: the name
/// carries the process id, a per-process sequence, and the
/// wall-clock nanosecond stamp; the open is `create_new`,
/// so an existing namesake is never truncated (the name is
/// retried instead); and the permissions are owner-only
/// from the moment the file exists. For files whose
/// contents are worth keeping from other accounts on a
/// shared host -- the guest environment file is the one
/// here (see `write_guest_env_file`).
/// Set `path`'s permissions to the owner-only `mode`,
/// best-effort. For the directories and files of a
/// materialised source snapshot (see
/// `ToolRegistry::effective_source_of`): the snapshot
/// copies recorded neighbouring files verbatim, so its
/// scratch tree in the shared temporary directory must
/// not be traversable or readable by other accounts.
pub(crate) fn restrict_to_owner(path: &std::path::Path, mode: u32) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode));
    }
    #[cfg(not(unix))]
    {
        // The owner-only mode model does not transfer to
        // this host, whose temporary directories are
        // already per-account; there is nothing to tighten.
        let _ = (path, mode);
    }
}

fn create_private_temp_file(tag: &str) -> std::io::Result<(std::path::PathBuf, std::fs::File)> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    for _ in 0..100 {
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "sparagmos-{tag}-{}-{seq}-{nanos:x}",
            std::process::id()
        ));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        match options.open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "no free private temporary file name",
    ))
}

/// Write one complete environment assignment (the environment
/// an interpreted program is to run under, as `artifact_run`
/// computed it from the execution request) to a fresh private
/// temporary file (see `create_private_temp_file`), in the
/// length-prefixed binary form `read_guest_env_file` reads:
/// a sequence of key/value records, each two u32-LE-length-
/// prefixed byte strings. The assignment is complete, not a
/// delta: the reader replaces its whole environment with it,
/// so no variable from the conversion's environment can leak
/// into the program's. The file is private because the
/// assignment is this process's whole environment --
/// inherited credentials and tokens included -- and it
/// stands in the shared temporary directory for the run's
/// whole duration. Returns the file's path; the caller
/// removes it once the run is over.
fn write_guest_env_file(env: &[(Vec<u8>, Vec<u8>)]) -> std::io::Result<std::path::PathBuf> {
    use std::io::Write as _;
    let mut buf = Vec::new();
    for (key, value) in env {
        buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
        buf.extend_from_slice(value);
    }
    let (path, mut file) = create_private_temp_file("artifact-env")?;
    if let Err(e) = file.write_all(&buf) {
        let _ = std::fs::remove_file(&path);
        return Err(e);
    }
    Ok(path)
}

/// Read back a file `write_guest_env_file` wrote.
pub(crate) fn read_guest_env_file(path: &std::path::Path) -> std::io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let buf = std::fs::read(path)?;
    let mut env = Vec::new();
    let mut at = 0usize;
    let take = |buf: &[u8], at: &mut usize| -> std::io::Result<Vec<u8>> {
        let len = |buf: &[u8], at: &mut usize| -> std::io::Result<usize> {
            let bytes = buf.get(*at..*at + 4).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "truncated env file")
            })?;
            *at += 4;
            Ok(u32::from_le_bytes(bytes.try_into().unwrap()) as usize)
        };
        let n = len(buf, at)?;
        let bytes = buf.get(*at..*at + n).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "truncated env file")
        })?;
        *at += n;
        Ok(bytes.to_vec())
    };
    while at < buf.len() {
        let key = take(&buf, &mut at)?;
        let value = take(&buf, &mut at)?;
        env.push((key, value));
    }
    Ok(env)
}

/// Replace this process's environment with the complete
/// assignment `env`: the run entry point applies the guest
/// environment file with this after the conversion is done,
/// so the interpreted program's environment reads (which the
/// interpreter answers from the host environment) see the
/// environment the program was invoked with, not the one its
/// source was converted under.
pub(crate) fn apply_guest_env(env: &[(Vec<u8>, Vec<u8>)]) {
    for (key, _) in std::env::vars_os() {
        std::env::remove_var(&key);
    }
    for (key, value) in env {
        std::env::set_var(os_string_from_bytes(key), os_string_from_bytes(value));
    }
}

/// The literal `include!` paths a source text names, in
/// order: `include!` followed by a string literal, the
/// only spelling rustc accepts for the path.
fn literal_include_paths(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut paths = Vec::new();
    let mut i = 0;
    while let Some(at) = text[i..].find("include!") {
        let mut j = i + at + "include!".len();
        i = j;
        while j < bytes.len() && (bytes[j] as char).is_whitespace() {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'(' {
            continue;
        }
        j += 1;
        while j < bytes.len() && (bytes[j] as char).is_whitespace() {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'"' {
            continue;
        }
        j += 1;
        let mut path = String::new();
        while j < bytes.len() && bytes[j] != b'"' {
            if bytes[j] == b'\\' && j + 1 < bytes.len() {
                j += 1;
            }
            path.push(bytes[j] as char);
            j += 1;
        }
        if j < bytes.len() {
            paths.push(path);
            i = j;
        }
    }
    paths
}

/// Whether the crate rooted at `root` declares a `main`
/// function: the root's own text, or the text of a file
/// the root splices in through a literal `include!`
/// (transitively, each include resolved against the
/// including file's directory). A `mod` file cannot hold
/// the crate's entry point, but an included file is root
/// text by other means, and a root that only includes
/// its `main` still builds a binary under a real rustc --
/// the root-only reading would record such a build as a
/// library, and the artifact could never be run.
fn crate_root_has_main(root: &std::path::Path, root_text: &str) -> bool {
    if root_text.contains("fn main") {
        return true;
    }
    let base = root.parent().unwrap_or(root);
    let mut frontier: Vec<std::path::PathBuf> = literal_include_paths(root_text)
        .iter()
        .map(|rel| base.join(rel))
        .collect();
    let mut seen: std::collections::HashSet<std::path::PathBuf> =
        frontier.iter().cloned().collect();
    let mut steps = 0;
    while let Some(path) = frontier.pop() {
        steps += 1;
        if steps > 64 {
            break;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if text.contains("fn main") {
            return true;
        }
        let base = path.parent().unwrap_or(&path).to_path_buf();
        for rel in literal_include_paths(&text) {
            let next = base.join(rel);
            if seen.insert(next.clone()) {
                frontier.push(next);
            }
        }
    }
    false
}

/// Write one `dep-info` emit's file: the Makefile-shaped
/// dependency text naming the invocation's primary output
/// (or the crate name, when the invocation recorded none)
/// and the sources it read -- the input and the aux sources
/// it bound. Shared by the success path's emit loop and the
/// dep-info verdict fallback in `tool_run`, the two places an
/// invocation is judged to have produced dep-info.
fn write_dep_info_emit(
    path: &Option<String>,
    cwd: &std::path::Path,
    out_base: &std::path::Path,
    crate_name: &str,
    input_abs: &std::path::Path,
    aux: &[String],
    primary: Option<&std::path::Path>,
) {
    let p = match path {
        Some(p) => absolutize(cwd, p),
        None => out_base.join(format!("{crate_name}.d")),
    };
    if let Some(parent) = p.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut deps = vec![input_abs.to_path_buf()];
    for spec in aux {
        if let Some((_, src)) = spec.split_once('=') {
            let src = src.split(';').next().unwrap_or(src);
            deps.push(std::path::PathBuf::from(src));
        }
    }
    let target = primary
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| crate_name.to_string());
    let mut text = format!(
        "{}: {}",
        target,
        deps.iter()
            .map(|d| d.display().to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    for d in &deps {
        text += &format!("\n{}:", d.display());
    }
    text.push('\n');
    let _ = std::fs::write(&p, text);
}

/// Answer one intercepted tool invocation (a run-make recipe's
/// compiler call) with the driver's own conversion machinery:
/// the invocation is re-run as this binary in convert mode over
/// the guest's input, and the fabricated process outcome is the
/// conversion's verdict -- exit 0 only when the target crate
/// converted with nothing skipped, with the compiler's
/// diagnostics as the child's stderr. Artifacts a real
/// compiler would have produced are recorded in the registry
/// (and created as placeholders, with real text for dep-info
/// and MIR output) so later invocations can name them through
/// `--extern`, which then binds the original source through
/// `--aux`.
///
/// One invocation class is not judged by conversion: when the
/// emit set is dep-info alone, the real compiler answers at
/// its dep-info phase boundary -- success iff the crate
/// parses, expands and resolves, however its type checking
/// would fare -- so a failed conversion verdict for such an
/// invocation is re-taken at that boundary (see the verdict
/// fallback below). Callers asking for the conversion verdict
/// itself (the cargo shim's verdict channel) pass
/// `allow_dep_info_verdict = false` and never take it.
pub(crate) fn tool_run(
    request: &sparagmos_interp::ToolRunRequest,
    registry: &ToolRegistry,
    exe: &std::path::Path,
    allow_dep_info_verdict: bool,
) -> Result<sparagmos_interp::ToolRunOutcome, String> {
    let cwd: std::path::PathBuf = match &request.cwd {
        Some(dir) => std::path::PathBuf::from(os_string_from_bytes(dir)),
        None => std::env::current_dir()
            .map_err(|e| format!("tool interception: no working directory: {e}"))?,
    };
    // A guest-spelled directory is relative to this
    // process's own working directory (the parent the
    // child would have inherited it from); make it stand
    // on its own, so the paths resolved against it -- the
    // recorded build directory among them -- keep their
    // meaning wherever they are read back.
    let cwd = if cwd.is_absolute() {
        cwd
    } else {
        std::env::current_dir()
            .map_err(|e| format!("tool interception: no working directory: {e}"))?
            .join(cwd)
    };
    let args: Vec<String> = request
        .args
        .iter()
        .map(|a| String::from_utf8_lossy(a).into_owned())
        .collect();

    // Translate the guest's rustc arguments.
    let mut input: Option<String> = None;
    let mut edition: Option<String> = None;
    let mut crate_name: Option<String> = None;
    let mut crate_types: Vec<String> = Vec::new();
    let mut test_crate = false;
    let mut forward: Vec<String> = Vec::new();
    let mut aux: Vec<String> = Vec::new();
    // The dependency version each `aux` entry binds, in
    // the same order (see `ToolArtifact::aux_pins`): the
    // specs are resolved through these pins, so a build
    // -- and the artifact it records -- keeps the
    // dependency builds it actually named.
    let mut aux_pins: Vec<(std::path::PathBuf, SourceTree)> = Vec::new();
    let mut emits: Vec<(String, Option<String>)> = Vec::new();
    let mut out_file: Option<String> = None;
    let mut out_dir: Option<String> = None;
    let mut informational = false;
    let mut positional_only = false;
    let mut extra_filename: Option<String> = None;
    // The crate names this invocation binds through an
    // explicit `--extern`, whether the path resolved to a
    // registered artifact or was forwarded: the `extern
    // crate` scan below must not add a second binding for a
    // name the invocation itself bound. Under rustc the
    // explicit `--extern` IS the binding for its name; a
    // scan-added binding for the same name would shadow a
    // forwarded path the session must judge on its own (a
    // nonexistent or non-crate file among them).
    let mut explicit_extern_names: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].clone();
        i += 1;
        if positional_only {
            if input.is_none() {
                input = Some(arg);
            } else {
                forward.push(arg);
            }
            continue;
        }
        if arg == "--" {
            positional_only = true;
            continue;
        }
        // Split a `--flag=value` spelling; short flags keep their
        // value attached (`-Copt-level=2` is one token already).
        let (flag, inline): (&str, Option<String>) = if arg.starts_with("--") {
            match arg.split_once('=') {
                Some((f, v)) => (f, Some(v.to_string())),
                None => (arg.as_str(), None),
            }
        } else {
            (arg.as_str(), None)
        };
        let mut take_value = |inline: Option<String>, i: &mut usize| -> Option<String> {
            match inline {
                Some(v) => Some(v),
                None => {
                    let v = args.get(*i).cloned();
                    if v.is_some() {
                        *i += 1;
                    }
                    v
                }
            }
        };
        match flag {
            "--test" => {
                test_crate = true;
                forward.push(arg.clone());
            }
            "--edition" => edition = take_value(inline, &mut i),
            "--crate-name" => crate_name = take_value(inline, &mut i),
            "--crate-type" => {
                if let Some(v) = take_value(inline, &mut i) {
                    crate_types.push(v.clone());
                    forward.push(format!("--crate-type={v}"));
                }
            }
            "--emit" => {
                if let Some(v) = take_value(inline, &mut i) {
                    for part in v.split(',') {
                        let (kind, path) = match part.split_once('=') {
                            Some((k, p)) => (k.to_string(), Some(p.to_string())),
                            None => (part.to_string(), None),
                        };
                        emits.push((kind, path));
                    }
                }
            }
            "-o" => out_file = take_value(inline, &mut i),
            "--out-dir" => out_dir = take_value(inline, &mut i),
            "--extern" => {
                if let Some(v) = take_value(inline, &mut i) {
                    let (name, path) = match v.split_once('=') {
                        Some((n, p)) => (n.to_string(), Some(p.to_string())),
                        None => (v.clone(), None),
                    };
                    explicit_extern_names.insert(name.clone());
                    let bound = path.as_ref().and_then(|p| {
                        let abs = absolutize(&cwd, p);
                        registry.get(&abs).map(|art| (name.clone(), art))
                    });
                    match bound {
                        Some((name, art)) => {
                            bind_artifact_with_deps(
                                registry,
                                &name,
                                &art,
                                &mut aux,
                                &mut aux_pins,
                            );
                        }
                        None => {
                            // A real file (a sysroot or prebuilt
                            // rlib) or a name the session must
                            // resolve itself: forward verbatim.
                            forward.push("--extern".to_string());
                            forward.push(v);
                        }
                    }
                }
            }
            "--print" => {
                informational = true;
                forward.push("--print".to_string());
                if let Some(v) = take_value(inline, &mut i) {
                    forward.push(v);
                }
            }
            "--version" | "-V" | "--explain" => {
                informational = true;
                forward.push(arg.clone());
            }
            "-C" => {
                forward.push(flag.to_string());
                if let Some(v) = take_value(inline, &mut i) {
                    // The output-name suffix is the one codegen
                    // option the artifact recording below must
                    // honour: it is part of the file name a
                    // real rustc gives the artifact.
                    if let Some(suffix) = v.strip_prefix("extra-filename=") {
                        extra_filename = Some(suffix.to_string());
                    }
                    forward.push(v);
                }
            }
            "--cfg" | "--check-cfg" | "-Z" | "-L" | "--target"
            | "--error-format" | "--cap-lints" | "--remap-path-prefix"
            | "--color" | "--diagnostic-width" | "-A" | "-W" | "-D" | "-F" => {
                forward.push(flag.to_string());
                if let Some(v) = take_value(inline, &mut i) {
                    forward.push(v);
                }
            }
            _ if arg.starts_with("-C") && arg.len() > 2 => {
                // A glued codegen option (the spelling
                // run-make-support emits, `-Cextra-filename=2`):
                // forward as-is, but note the output-name
                // suffix for the artifact recording below.
                if let Some(suffix) = arg[2..].strip_prefix("extra-filename=") {
                    extra_filename = Some(suffix.to_string());
                }
                forward.push(arg.clone());
            }
            _ if flag.starts_with('-') && flag.len() > 1 => {
                // Any other flag spelling (single-token forms
                // like `-g`, `--pretty=expanded`): forward as-is.
                forward.push(arg.clone());
            }
            _ => {
                if input.is_none() {
                    input = Some(arg);
                } else {
                    forward.push(arg);
                }
            }
        }
    }

    if informational {
        // `--print` / `--version` / `--explain`: the convert
        // session answers these itself (the compiler skips
        // conversion and reports success); the printed text is
        // the child's stdout. An invocation without an input
        // runs over an empty scratch crate.
        let input = match input {
            Some(path) => path,
            None => {
                let scratch = std::env::temp_dir().join(format!(
                    "sparagmos-tool-informational-{}.rs",
                    std::process::id()
                ));
                register_process_scratch(&scratch);
                if !scratch.exists() {
                    let _ = std::fs::write(&scratch, "");
                }
                scratch.display().to_string()
            }
        };
        let mut tail = vec![input];
        for spec in registry.pinned_aux_specs(&aux, &aux_pins) {
            tail.push("--aux".to_string());
            tail.push(spec);
        }
        for tok in &forward {
            tail.push("--rustc-arg".to_string());
            tail.push(tok.clone());
        }
        let (code, _stdout, stderr) =
            run_convert_subprocess(exe, &tail, &cwd, request.env_clear, &request.envs)?;
        let text = child_stream(&stderr);
        return Ok(if code == 0 {
            sparagmos_interp::ToolRunOutcome {
                exit_code: 0,
                stdout: text,
                stderr: Vec::new(),
            }
        } else {
            sparagmos_interp::ToolRunOutcome {
                exit_code: code,
                stdout: Vec::new(),
                stderr: text,
            }
        });
    }

    let Some(input) = input else {
        return Err(
            "tool interception: the invocation names no input file".to_string(),
        );
    };
    let input_abs = absolutize(&cwd, &input);
    let edition = edition.unwrap_or_else(|| "2015".to_string());
    let source_text = std::fs::read_to_string(&input_abs).unwrap_or_default();
    // The crate's name: the invocation's `--crate-name`,
    // else the source's own `#![crate_name]` declaration
    // (the name the compiler itself gives the crate, and
    // the one the recipe looks its artifacts up by), else
    // the file stem.
    let crate_name = crate_name
        .or_else(|| declared_crate_name(&source_text))
        .unwrap_or_else(|| {
            input_abs
                .file_stem()
                .map(|s| s.to_string_lossy().replace('-', "_"))
                .unwrap_or_else(|| "crate".to_string())
        });
    // The source tree this build converts, snapshotted
    // into every artifact it records: the recipe may
    // rewrite or delete the sources once the build returns,
    // and the artifact must still denote the program this
    // invocation produced.
    let source_tree = snapshot_source_tree(&input_abs);

    // Registered artifacts the input declares through
    // `extern crate` items bind too (the driver's session
    // binding counts them as declarations, as for `--aux`).
    // A name the invocation bound through an explicit
    // `--extern` is not scanned for: the explicit binding is
    // the name's binding, as under rustc, and a scan-added
    // second binding would shadow it in the session.
    for (_, art) in registry.all() {
        if explicit_extern_names.contains(&art.name) {
            continue;
        }
        let needle = format!("extern crate {}", art.name);
        if source_text.contains(&needle) {
            bind_artifact_with_deps(
                registry,
                &art.name,
                &art,
                &mut aux,
                &mut aux_pins,
            );
        }
    }

    // A MIR emit is served by the same subprocess through the
    // driver's `--dump`: decide its path before running.
    let out_base = match &out_dir {
        Some(dir) => absolutize(&cwd, dir),
        None => cwd.clone(),
    };
    let mir_dump: Option<std::path::PathBuf> = emits
        .iter()
        .find(|(kind, _)| kind == "mir")
        .map(|(_, path)| match path {
            Some(p) => absolutize(&cwd, p),
            None => out_base.join(format!("{crate_name}.mir")),
        });

    let mut tail = vec![
        input_abs.display().to_string(),
        "--edition".to_string(),
        edition.clone(),
    ];
    for spec in registry.pinned_aux_specs(&aux, &aux_pins) {
        tail.push("--aux".to_string());
        tail.push(spec);
    }
    for tok in &forward {
        tail.push("--rustc-arg".to_string());
        tail.push(tok.clone());
    }
    if let Some(dump) = &mir_dump {
        tail.push("--dump".to_string());
        tail.push(dump.display().to_string());
    }
    // Whether the invocation's emit set is dep-info alone:
    // such an invocation is judged at the real compiler's
    // dep-info phase boundary, not by conversion (see the
    // verdict fallback at the failure return below).
    let dep_info_only = !emits.is_empty()
        && emits.iter().all(|(kind, _)| kind == "dep-info");

    let (code, stdout, stderr) =
        run_convert_subprocess(exe, &tail, &cwd, request.env_clear, &request.envs)?;

    // The verdict: exit 0 only when the target crate converted
    // with nothing skipped. The report's target-crate line is
    // the authority (a plain exit 0 also covers informational
    // invocations and demand-driven partial conversions).
    let report = String::from_utf8_lossy(&stdout);
    let skipped = report.lines().find_map(|line| {
        let at = line.find("target-crate converted:")?;
        let rest = &line[at..];
        let at = rest.find("skipped:")?;
        rest[at + "skipped:".len()..]
            .trim()
            .split_whitespace()
            .next()?
            .parse::<usize>()
            .ok()
    });
    let child_stderr = child_stream(&stderr);
    if code != 0 || skipped != Some(0) {
        // The dep-info verdict fallback (see the doc comment
        // above): conversion subsumes resolution -- a crate
        // that fails parsing, expansion or resolution cannot
        // convert -- so when the crate resolves cleanly, the
        // conversion failure lies in a phase dep-info skips
        // and the invocation succeeds the way the real
        // compiler's does. The dep-info file is its only
        // artifact (the real compiler produces no primary
        // output for it, and none is recorded), and the
        // verdict run's stderr -- the diagnostics through
        // resolution -- is what the guest sees.
        if allow_dep_info_verdict && dep_info_only {
            let mut verdict_tail = tail.clone();
            verdict_tail.push("--dep-info-verdict".to_string());
            if let Ok((verdict_code, _, verdict_stderr)) = run_convert_subprocess(
                exe,
                &verdict_tail,
                &cwd,
                request.env_clear,
                &request.envs,
            ) {
                if verdict_code == 0 {
                    for (kind, path) in &emits {
                        if kind == "dep-info" {
                            write_dep_info_emit(
                                path,
                                &cwd,
                                &out_base,
                                &crate_name,
                                &input_abs,
                                &aux,
                                None,
                            );
                        }
                    }
                    return Ok(sparagmos_interp::ToolRunOutcome {
                        exit_code: 0,
                        stdout: Vec::new(),
                        stderr: child_stream(&verdict_stderr),
                    });
                }
            }
        }
        return Ok(sparagmos_interp::ToolRunOutcome {
            exit_code: if code == 101 { 101 } else { 1 },
            stdout: Vec::new(),
            stderr: child_stderr,
        });
    }

    // Success: record and create the artifacts a real compiler
    // would have produced, so later invocations and the
    // recipe's own file checks find them.
    // The crate types the invocation builds: its own
    // `--crate-type` flags, else the source's `#![crate_type]`
    // declaration, else the default (a test crate or a root
    // with `main` builds a binary, anything else a library).
    // A metadata emit is not a crate type: the emit loop
    // below records it under the build's first type, the
    // type a later `--extern` binding rebuilds the source
    // as when it binds the artifact as an auxiliary crate.
    let types: Vec<String> = if crate_types.is_empty() {
        let has_main = crate_root_has_main(&input_abs, &source_text);
        if test_crate {
            // A `--test` build's primary output is the test
            // binary, named after the crate, whether or not
            // the root declares a `main` (the harness supplies
            // it); a metadata emit the invocation also carries
            // is recorded by the emit loop below, as rustc
            // writes both.
            vec!["bin".to_string()]
        } else if let Some(declared) = declared_crate_type(&source_text) {
            vec![declared]
        } else if has_main {
            vec!["bin".to_string()]
        } else {
            vec!["rlib".to_string()]
        }
    } else {
        crate_types.clone()
    };
    // The output-name suffix `-C extra-filename` gives every
    // artifact of the build ("" when the invocation set
    // none: `-C metadata` alone does not suffix the name).
    let extra_suffix = extra_filename.as_deref().unwrap_or("");
    // Whether the invocation produces its crate-type outputs
    // at all: an `--emit` list replaces the default emit
    // set, and only a `link` emit among it (or no `--emit`
    // at all) writes them -- `--emit metadata` alone writes
    // no rlib beside the rmeta.
    let link_output =
        emits.is_empty() || emits.iter().any(|(k, _)| k == "link");
    let mut primary: Option<std::path::PathBuf> = None;
    let mut record = |path: std::path::PathBuf,
                      crate_type: Option<String>,
                      registry: &ToolRegistry,
                      primary: &mut Option<std::path::PathBuf>| {
        if primary.is_none() {
            *primary = Some(path.clone());
        }
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&path, b"");
        if crate_type.as_deref() == Some("bin") {
            // A real compiler's binary output is executable:
            // the recipe's spawn of it (and its own executable
            // checks) must find it so.
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755));
            }
        }
        let file_id = file_identity(&path);
        registry.record(
            path,
            ToolArtifact {
                name: crate_name.clone(),
                source: input_abs.clone(),
                source_tree: source_tree.clone(),
                build_cwd: cwd.clone(),
                edition: Some(edition.clone()),
                crate_type,
                compile_env_clear: request.env_clear,
                compile_envs: request.envs.clone(),
                aux_specs: aux.clone(),
                aux_pins: aux_pins.clone(),
                rustc_args: forward.clone(),
                file_id,
            },
        );
    };
    if let Some(out) = &out_file {
        // `-o` names the invocation's one output (a real rustc
        // refuses `-o` with several crate types), so the type is
        // known exactly when the invocation carried one.
        record(
            absolutize(&cwd, out),
            (types.len() == 1).then(|| types[0].clone()),
            registry,
            &mut primary,
        );
    } else if link_output {
        for ty in &types {
            record(
                out_base.join(artifact_filename(&crate_name, ty, extra_suffix)),
                Some(ty.clone()),
                registry,
                &mut primary,
            );
        }
    }
    for (kind, path) in &emits {
        match kind.as_str() {
            "metadata" => {
                let p = match path {
                    Some(p) => absolutize(&cwd, p),
                    None => out_base.join(artifact_filename(
                        &crate_name,
                        "rmeta",
                        extra_suffix,
                    )),
                };
                // The recorded type is the build's own crate
                // type, not the emit kind: a later `--extern`
                // binding rebuilds the source as an auxiliary
                // crate under this type, and `rmeta` is not a
                // crate type a compiler accepts.
                record(p, Some(types[0].clone()), registry, &mut primary);
            }
            "dep-info" => {
                write_dep_info_emit(
                    path,
                    &cwd,
                    &out_base,
                    &crate_name,
                    &input_abs,
                    &aux,
                    primary.as_deref(),
                );
            }
            "llvm-ir" => {
                let p = match path {
                    Some(p) => absolutize(&cwd, p),
                    None => out_base.join(format!("{crate_name}.ll")),
                };
                if let Some(parent) = p.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(&p, b"");
            }
            "asm" => {
                let p = match path {
                    Some(p) => absolutize(&cwd, p),
                    None => out_base.join(format!("{crate_name}.s")),
                };
                if let Some(parent) = p.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(&p, b"");
            }
            _ => {}
        }
    }

    Ok(sparagmos_interp::ToolRunOutcome {
        exit_code: 0,
        stdout: Vec::new(),
        stderr: child_stderr,
    })
}

/// The environment variable naming the cargo shim's verdict
/// channel: the stage-3 runner sets it to a file path, and
/// every shim invocation appends one line per compilation
/// unit it converts (see `rustc_shim`).
pub(crate) const CARGO_SHIM_VERDICTS_ENV: &str =
    "SPARAGMOS_CARGO_SHIM_VERDICTS";

/// Act as a cargo `RUSTC_WRAPPER`: cargo invokes the wrapper
/// as `<wrapper> <rustc> <args...>` for every compilation
/// unit of a build. Each unit is first converted with this
/// driver's own machinery -- the same conversion `tool_run`
/// performs for an intercepted invocation, over a fresh
/// registry: a shim build's artifacts are the real files the
/// delegation below produces, never registry placeholders a
/// later invocation must resolve -- and the unit's verdict
/// is appended to the side channel the runner named in
/// `SPARAGMOS_CARGO_SHIM_VERDICTS`. The invocation is then
/// delegated to the real rustc with the same arguments, and
/// the wrapper exits with the real compiler's exit code.
/// The two verdicts are kept apart on purpose: the
/// conversion verdict rides the side channel, the build's
/// own verdict is cargo's exit status as the recipe observes
/// it -- the stage-3 cargo proposition reads both (issue
/// #359). Informational invocations (`--print`, `--version`
/// and their kin: cargo's probes, not compilation units)
/// are delegated without a conversion and record no verdict.
pub(crate) fn rustc_shim(args: &[String]) -> i32 {
    let Some((rustc, rest)) = args.split_first() else {
        eprintln!("sparagmos-driver rustc-shim: no rustc program named");
        return 1;
    };
    let informational = rest.iter().any(|a| {
        a == "--print"
            || a.starts_with("--print=")
            || a == "--version"
            || a == "-V"
            || a == "-vV"
            || a == "--explain"
    });
    if !informational {
        record_shim_verdict(rustc, rest);
    }
    // Delegate: the real compiler produces the real
    // artifacts, over this process's own streams.
    match std::process::Command::new(rustc).args(rest).status() {
        Ok(status) => status.code().unwrap_or(101),
        Err(e) => {
            eprintln!(
                "sparagmos-driver rustc-shim: cannot start {rustc}: {e}"
            );
            1
        }
    }
}

/// Convert one shim unit and append its verdict line to the
/// side channel: `<verdict>\t<crate name>\t<input>`, where
/// the verdict is `converted` (the unit converted with
/// nothing skipped), `failed` (the conversion ran and did
/// not succeed), or `unsupported` (the machinery could not
/// carry the invocation out). The crate name and input are
/// diagnostic; the runner's judging reads the verdict word.
/// An invocation naming no input file is a probe the
/// informational scan did not recognise, not a compilation
/// unit: it records nothing.
fn record_shim_verdict(rustc: &str, rest: &[String]) {
    let mut crate_name = String::new();
    let mut input = String::new();
    let mut i = 0;
    while i < rest.len() {
        let arg = &rest[i];
        if arg == "--crate-name" {
            if let Some(v) = rest.get(i + 1) {
                crate_name = v.clone();
                i += 1;
            }
        } else if let Some(v) = arg.strip_prefix("--crate-name=") {
            crate_name = v.to_string();
        } else if input.is_empty()
            && arg.ends_with(".rs")
            && !arg.contains('=')
        {
            input = arg.clone();
        }
        i += 1;
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let request = sparagmos_interp::ToolRunRequest {
        program: rustc.as_bytes().to_vec(),
        args: rest.iter().map(|a| a.as_bytes().to_vec()).collect(),
        argv0: None,
        envs: Vec::new(),
        env_clear: false,
        cwd: Some(cwd.as_os_str().as_encoded_bytes().to_vec()),
        resolved_program: None,
        output_captured: false,
    };
    let registry = ToolRegistry::default();
    let exe = std::env::current_exe().unwrap_or_default();
    let verdict = match tool_run(&request, &registry, &exe, false) {
        Ok(outcome) if outcome.exit_code == 0 => "converted",
        Ok(_) => "failed",
        Err(reason) if reason.contains("names no input file") => return,
        Err(_) => "unsupported",
    };
    let Some(path) = std::env::var_os(CARGO_SHIM_VERDICTS_ENV) else {
        return;
    };
    use std::io::Write as _;
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = file.write_all(
            format!("{verdict}\t{crate_name}\t{input}\n").as_bytes(),
        );
    }
}

/// Answer one invocation of a program the guest itself
/// produced earlier in this run (a run-make recipe running a
/// binary it just built): the artifact registry knows the
/// path, so the run is this driver binary in run mode over
/// the recorded source, with the `--aux` bindings and the
/// compilation-shaping options the producing invocation
/// carried and the guest's arguments as the program's own.
/// This is the stage-3 harness's reading of "execute the
/// built binary": the program's input/output and exit code
/// are the interpreter's, not a native binary's (see
/// docs/CI.md). Programs the registry does not know, and
/// artifacts that are not binaries, stay unsupported -- as
/// does a recorded program the driver cannot itself run, so
/// the harness keeps sorting those recipes as unsupported
/// rather than failed.
///
/// The run is reconstructed from what the producing
/// invocation carried, not from the file system as it now
/// stands: the source it converts is the recorded path while
/// every file of its tree still holds the build's bytes, and
/// the recorded snapshot once the recipe has rewritten or
/// deleted any of it (see `ToolRegistry::effective_source`),
/// and its `--aux` bindings
/// likewise resolve against the dependency versions its build
/// bound, not the registry's current entries (see
/// `ToolArtifact::aux_pins`); the program's argv[0]
/// is the name the recipe invoked it under, not the source
/// path; and the two environments are kept apart -- the
/// subprocess is spawned under the producing invocation's
/// compilation environment, so compile-time environment
/// reads see the build's values, and the execution request's
/// environment is handed over as a complete assignment the
/// subprocess applies to itself once the conversion is done
/// (see `--guest-env-file`), so the program's own environment
/// reads see the values it was invoked with and nothing from
/// the build leaks into them.
///
/// The two working directories are kept apart the same
/// way: the subprocess is spawned in the producing
/// invocation's working directory (see
/// `ToolArtifact::build_cwd`), so the compilation-shaping
/// arguments it replays -- a forwarded relative `--extern`
/// path, a `-L` directory -- resolve to the files they
/// denoted at build time, and the requested execution
/// directory is handed over as `--guest-cwd`, which the
/// subprocess enters once its conversion is done, so the
/// program itself runs where the recipe invoked it.
///
/// The lookup path the interpreter's spawn resolution
/// settled on is read against the directory that
/// resolution ran in -- this process's own working
/// directory -- not the execution directory: the
/// resolution already applied the child's directory to a
/// spelled path (and a relative `PATH` entry is relative
/// to the searching process), so standing it on the
/// execution directory would apply the child's directory
/// twice. The guest arguments travel to the subprocess as
/// command-line strings, so an argument that is not valid
/// UTF-8 cannot be carried unchanged and makes the
/// invocation unsupported instead of lossy.
///
/// The subprocess keeps the two stderr channels apart: the
/// recorded program's own stderr travels through the file
/// named by `--guest-stderr-file` and is relayed byte for
/// byte, while the subprocess's stderr carries only the
/// driver's own lines. The channel file is allocated
/// owner-only under an unpredictable name before the
/// subprocess starts (see `create_private_temp_file`), as
/// the environment file is: the program's stderr can carry
/// the recipe's own diagnostics and secrets, so it is no
/// more exposed to other accounts than the environment is.
/// That separation is also what tells a
/// driver failure from the program's own exit: the run
/// subprocess exits 2 both when it cannot run the recorded
/// program and when the program itself exits 2, but only
/// the driver failure reports a `sparagmos-driver` line on
/// its stderr.
/// The environment a program run for a tool request runs
/// under, as a complete assignment: this process's
/// environment (the guest's own, which the interpreter
/// answers its environment reads from) with the request's
/// assignments applied -- or, when the request clears the
/// environment, exactly the request's assignments.
fn request_guest_env(
    request: &sparagmos_interp::ToolRunRequest,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut guest_env: Vec<(Vec<u8>, Vec<u8>)> = if request.env_clear {
        Vec::new()
    } else {
        std::env::vars_os()
            .map(|(k, v)| (k.as_encoded_bytes().to_vec(), v.as_encoded_bytes().to_vec()))
            .collect()
    };
    for (key, value) in &request.envs {
        match value {
            Some(v) => {
                if let Some(slot) = guest_env.iter_mut().find(|(k, _)| k == key) {
                    slot.1 = v.clone();
                } else {
                    guest_env.push((key.clone(), v.clone()));
                }
            }
            None => guest_env.retain(|(k, _)| k != key),
        }
    }
    guest_env
}

pub(crate) fn artifact_run(
    request: &sparagmos_interp::ToolRunRequest,
    registry: &ToolRegistry,
    exe: &std::path::Path,
) -> Result<sparagmos_interp::ToolRunOutcome, String> {
    let cwd: std::path::PathBuf = match &request.cwd {
        Some(dir) => std::path::PathBuf::from(os_string_from_bytes(dir)),
        None => std::env::current_dir()
            .map_err(|e| format!("tool interception: no working directory: {e}"))?,
    };
    // As in `tool_run`: a guest-spelled directory is
    // relative to this process's own, and the execution
    // directory is handed to a subprocess that starts in
    // the build's directory, so it must stand on its own.
    let cwd = if cwd.is_absolute() {
        cwd
    } else {
        std::env::current_dir()
            .map_err(|e| format!("tool interception: no working directory: {e}"))?
            .join(cwd)
    };
    let program = String::from_utf8_lossy(&request.program).into_owned();
    // The lookup follows the path the interpreter's spawn
    // resolution settled on, not the spelled name: a bare
    // name invoked under a command-specific `PATH` denotes
    // the artifact that search found, not a namesake in
    // the working directory (see
    // `ToolRunRequest::resolved_program`). The spelled
    // name remains what the run presents as argv[0].
    let path = match &request.resolved_program {
        Some(resolved) => {
            let resolved = std::path::PathBuf::from(os_string_from_bytes(resolved));
            if resolved.is_absolute() {
                resolved
            } else {
                // The resolution settled this path against
                // this process's own working directory (the
                // directory the interpreter's spawn search
                // ran in), not the execution directory: a
                // spelling like `sub/./prog` already carries
                // the child's directory, and standing it on
                // the execution directory would apply that
                // directory twice.
                std::env::current_dir()
                    .map_err(|e| format!("tool interception: no working directory: {e}"))?
                    .join(resolved)
            }
        }
        None => absolutize(&cwd, &program),
    };
    let artifact = match registry.get_for_run(&path) {
        Ok(Some(artifact)) => artifact,
        Ok(None) => {
            return Err(format!(
                "tool interception: program `{program}` is not an artifact this run produced"
            ));
        }
        Err(e) => return Err(e),
    };
    if artifact.crate_type.as_deref() != Some("bin") {
        return Err(format!(
            "tool interception: artifact `{program}` is not a binary this run produced"
        ));
    }
    // The recorded program's stderr gets its own channel,
    // so nothing of it is filtered or rewritten on its way
    // to the recipe. The channel is a private temporary
    // file (see `create_private_temp_file`), allocated
    // before the subprocess starts: it stands owner-only
    // under an unpredictable name for the whole run, so
    // the guest's stderr is never readable by another
    // account while it stands, and the subprocess writes
    // into the file that is already there, keeping its
    // permissions.
    let (stderr_path, stderr_file) =
        create_private_temp_file("artifact-stderr").map_err(|e| {
            format!("tool interception: cannot allocate the guest stderr channel: {e}")
        })?;
    drop(stderr_file);
    let guest_env = request_guest_env(request);
    let env_path = write_guest_env_file(&guest_env).map_err(|e| {
        format!("tool interception: cannot record the guest environment: {e}")
    })?;
    // The name the program reads as its argv[0]: the explicit
    // argv0 the invocation carried, else the program name as
    // the recipe spelled it -- never the source path the run
    // is reconstructed from.
    let program_name = match &request.argv0 {
        Some(argv0) => String::from_utf8_lossy(argv0).into_owned(),
        None => program.clone(),
    };
    let mut tail = vec![
        "run".to_string(),
        registry.effective_source(&artifact).display().to_string(),
    ];
    if let Some(edition) = &artifact.edition {
        tail.push("--edition".to_string());
        tail.push(edition.clone());
    }
    for spec in registry.artifact_aux_specs(&artifact) {
        tail.push("--aux".to_string());
        tail.push(spec);
    }
    for tok in &artifact.rustc_args {
        tail.push("--rustc-arg".to_string());
        tail.push(tok.clone());
    }
    // A recorded source tree that no longer stands intact
    // is read from its materialised snapshot under a
    // scratch directory; remap that directory back to the
    // recorded source's own directory, so the program's
    // source-location spellings stay the build's (see
    // `ToolRegistry::snapshot_remap`). The artifact's own
    // tree and every pinned dependency's tree take part.
    let mut remap_trees: Vec<(&std::path::Path, &SourceTree)> =
        vec![(&artifact.source, &artifact.source_tree)];
    for pin in &artifact.aux_pins {
        remap_trees.push((&pin.0, &pin.1));
    }
    for (source, tree) in remap_trees {
        if let Some((from, to)) = registry.snapshot_remap(source, tree) {
            tail.push("--rustc-arg".to_string());
            tail.push("--remap-path-prefix".to_string());
            tail.push("--rustc-arg".to_string());
            tail.push(format!("{}={}", from.display(), to.display()));
        }
    }
    tail.push("--guest-cwd".to_string());
    tail.push(cwd.display().to_string());
    tail.push("--guest-stderr-file".to_string());
    tail.push(stderr_path.display().to_string());
    tail.push("--guest-env-file".to_string());
    tail.push(env_path.display().to_string());
    tail.push("--guest-program-name".to_string());
    tail.push(program_name);
    // The arguments travel as command-line strings, and
    // the interpreter's argument model holds text: an
    // argument that is not valid UTF-8 cannot be carried
    // without changing its bytes, so the invocation is
    // unsupported rather than silently lossy.
    for arg in &request.args {
        if std::str::from_utf8(arg).is_err() {
            let _ = std::fs::remove_file(&stderr_path);
            let _ = std::fs::remove_file(&env_path);
            return Err(format!(
                "tool interception: running artifact `{program}`: a guest argument is not valid UTF-8, which the run cannot carry unchanged"
            ));
        }
        tail.push("--guest-arg".to_string());
        tail.push(String::from_utf8_lossy(arg).into_owned());
    }
    let outcome = run_convert_subprocess(
        exe,
        &tail,
        &artifact.build_cwd,
        artifact.compile_env_clear,
        &artifact.compile_envs,
    );
    let guest_stderr = std::fs::read(&stderr_path).unwrap_or_default();
    let _ = std::fs::remove_file(&stderr_path);
    let _ = std::fs::remove_file(&env_path);
    let (code, stdout, stderr) = outcome?;
    if code == 2 {
        // Exit 2 is both the subprocess's driver-failure code
        // and a code the recorded program itself may exit
        // with; the driver's own stderr channel tells them
        // apart (see the doc comment above). A driver failure
        // makes the recipe's invocation unsupported, with the
        // subprocess's own named reason when it gave one; no
        // driver line means the program ran and chose 2.
        let text = String::from_utf8_lossy(&stderr);
        let reason = text.lines().find_map(|line| {
            line.strip_prefix("sparagmos-driver run: unsupported: ")
                .or_else(|| line.strip_prefix("sparagmos-driver: compilation failed: "))
                .map(|rest| rest.to_string())
        });
        let driver_failed = reason.is_some()
            || text.lines().any(|line| line.starts_with("sparagmos-driver"));
        if driver_failed {
            let reason =
                reason.unwrap_or_else(|| "the recorded program could not be run".to_string());
            return Err(format!(
                "tool interception: running artifact `{program}`: {reason}"
            ));
        }
    }
    Ok(sparagmos_interp::ToolRunOutcome {
        exit_code: code,
        stdout,
        stderr: guest_stderr,
    })
}

/// The invocation context of a plain run, replayed when the
/// guest spawns the program it is itself running (see
/// [`execute`]): this run's conversion inputs, so the child
/// run converts the same source the same way.
#[derive(Clone)]
pub(crate) struct SelfRunContext {
    pub(crate) input: String,
    pub(crate) edition: String,
    pub(crate) aux: Vec<String>,
    pub(crate) rustc_args: Vec<String>,
}

/// Run the guest program itself for a spawn the interpreter
/// routed here: the guest invoked `Command` on its own
/// program (the name this run presents as argv[0]), which is
/// the standard shape of the test suite's `needs-subprocess`
/// tests -- a parent run checking a child run's exit status
/// or collected output. The answer is a subprocess of this
/// driver binary reconstructing this run under the request's
/// invocation (arguments, environment, working directory),
/// exactly as `artifact_run` reconstructs a recorded
/// program's run; the two differ only in what is replayed
/// (this run's own conversion inputs, not a registry
/// artifact's).
pub(crate) fn self_run(
    request: &sparagmos_interp::ToolRunRequest,
    ctx: &SelfRunContext,
    exe: &std::path::Path,
) -> Result<sparagmos_interp::ToolRunOutcome, String> {
    let cwd: std::path::PathBuf = match &request.cwd {
        Some(dir) => std::path::PathBuf::from(os_string_from_bytes(dir)),
        None => std::env::current_dir()
            .map_err(|e| format!("self spawn: no working directory: {e}"))?,
    };
    // As in `artifact_run`: a guest-spelled directory is
    // relative to this process's own working directory (the
    // parent the child would have inherited it from), and the
    // execution directory is handed to a subprocess that
    // starts in this process's directory, so it must stand on
    // its own.
    let cwd = if cwd.is_absolute() {
        cwd
    } else {
        std::env::current_dir()
            .map_err(|e| format!("self spawn: no working directory: {e}"))?
            .join(cwd)
    };
    let program = String::from_utf8_lossy(&request.program).into_owned();
    // The recorded program's stderr gets its own channel, as
    // in `artifact_run` (see there): the guest's stderr is
    // the outcome's, and this process's stderr carries only
    // the driver's own lines.
    let (stderr_path, stderr_file) =
        create_private_temp_file("self-stderr").map_err(|e| {
            format!("self spawn: cannot allocate the guest stderr channel: {e}")
        })?;
    drop(stderr_file);
    let guest_env = request_guest_env(request);
    let env_path = write_guest_env_file(&guest_env).map_err(|e| {
        format!("self spawn: cannot record the guest environment: {e}")
    })?;
    // The name the child reads as its argv[0]: the explicit
    // argv0 the invocation carried, else the program name as
    // the guest spelled it -- the same name this run presents,
    // so the child's own self-spawns name it again.
    let program_name = match &request.argv0 {
        Some(argv0) => String::from_utf8_lossy(argv0).into_owned(),
        None => program.clone(),
    };
    let mut tail = vec!["run".to_string(), ctx.input.clone()];
    tail.push("--edition".to_string());
    tail.push(ctx.edition.clone());
    for spec in &ctx.aux {
        tail.push("--aux".to_string());
        tail.push(spec.clone());
    }
    for tok in &ctx.rustc_args {
        tail.push("--rustc-arg".to_string());
        tail.push(tok.clone());
    }
    tail.push("--guest-cwd".to_string());
    tail.push(cwd.display().to_string());
    tail.push("--guest-stderr-file".to_string());
    tail.push(stderr_path.display().to_string());
    tail.push("--guest-env-file".to_string());
    tail.push(env_path.display().to_string());
    tail.push("--guest-program-name".to_string());
    tail.push(program_name);
    // The child's guest stdout writes through to the stream
    // this spawn wired as its stdout, as a native child's
    // would (see `InterpConfig::stdout_sink`).
    tail.push("--guest-stdout-passthrough".to_string());
    // The arguments travel as command-line strings, as in
    // `artifact_run`: an argument that is not valid UTF-8
    // cannot be carried without changing its bytes, so the
    // invocation is unsupported rather than silently lossy.
    for arg in &request.args {
        if std::str::from_utf8(arg).is_err() {
            let _ = std::fs::remove_file(&stderr_path);
            let _ = std::fs::remove_file(&env_path);
            return Err(
                "self spawn: a guest argument is not valid UTF-8, which the run cannot carry unchanged"
                    .to_string(),
            );
        }
        tail.push("--guest-arg".to_string());
        tail.push(String::from_utf8_lossy(arg).into_owned());
    }
    let start_dir = std::env::current_dir()
        .map_err(|e| format!("self spawn: no working directory: {e}"))?;
    let outcome = run_convert_subprocess(exe, &tail, &start_dir, false, &[]);
    let guest_stderr = std::fs::read(&stderr_path).unwrap_or_default();
    let _ = std::fs::remove_file(&stderr_path);
    let _ = std::fs::remove_file(&env_path);
    let (code, stdout, stderr) = outcome?;
    if code == 2 {
        // As in `artifact_run`: exit 2 is both the
        // subprocess's driver-failure code and a code the
        // program itself may exit with; the driver's own
        // stderr channel tells them apart.
        let text = String::from_utf8_lossy(&stderr);
        let reason = text.lines().find_map(|line| {
            line.strip_prefix("sparagmos-driver run: unsupported: ")
                .or_else(|| line.strip_prefix("sparagmos-driver: compilation failed: "))
                .map(|rest| rest.to_string())
        });
        let driver_failed = reason.is_some()
            || text.lines().any(|line| line.starts_with("sparagmos-driver"));
        if driver_failed {
            let reason =
                reason.unwrap_or_else(|| "the program could not be run".to_string());
            return Err(format!("self spawn: running `{program}`: {reason}"));
        }
    }
    // A guest abort (a panic under `panic=abort`, a panic
    // escaping a cleanup, an explicit abort) ends the child
    // driver with the run's abort exit code, 134, which only
    // an abort produces (an unwound guest panic exits 101);
    // natively the child died on SIGABRT, and the outcome's
    // negative-code convention carries exactly that (see
    // `exit_status_value` in the interpreter).
    let exit_code = if code == 134 {
        -6 // SIGABRT
    } else {
        code
    };
    Ok(sparagmos_interp::ToolRunOutcome {
        exit_code,
        stdout,
        stderr: guest_stderr,
    })
}

/// Render an `--aux` spec back to its command-line spelling
/// (`<name>=<file.rs>` with its `;crate-type=` / `;edition=`
/// attributes), for replays of this run's invocation (see
/// [`SelfRunContext`]).
pub(crate) fn render_aux_spec(spec: &AuxSpec) -> String {
    let mut out = format!("{}={}", spec.name, spec.path);
    if let Some(crate_type) = &spec.crate_type {
        out.push_str(";crate-type=");
        out.push_str(crate_type);
    }
    if let Some(edition) = &spec.edition {
        out.push_str(";edition=");
        out.push_str(edition);
    }
    if spec.force_unstable {
        out.push_str(";force-unstable=yes");
    }
    out
}
