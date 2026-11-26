//! Auxiliary crates: scanning source declarations and compiling aux sources.

use super::Mode;


/// The crate type an auxiliary source declares for itself
/// (`#![crate_type = "..."]`), if any. When a source declares one,
/// the declaration decides the artifact kind and the driver passes
/// no `--crate-type` of its own.
fn declared_crate_type(src: &str) -> Option<String> {
    declared_inner_attr(src, "crate_type")
}

/// The crate name an auxiliary source declares for itself
/// (`#![crate_name = "..."]`), if any. When a source declares one,
/// the declaration is the crate's name (compiletest links the crate
/// under it, and the test refers to it by it), so the driver passes
/// no `--crate-name` of its own — rustc rejects a `--crate-name`
/// that differs from the declaration.
fn declared_crate_name(src: &str) -> Option<String> {
    declared_inner_attr(src, "crate_name")
}

/// Read an inner-attribute string value (`#![<attr> = "..."]`) from
/// a source file. Only a real inner attribute at the top of the
/// file counts — one written inside a comment does not (the
/// open-ns auxiliary `open-ns-my_api_core.rs` carries
/// `// #![crate_name = "my_api::core"]` as a comment, and treating
/// it as a declaration made the driver look for an artifact under
/// a namespaced name the build never produced).
fn declared_inner_attr(src: &str, attr: &str) -> Option<String> {
    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        if !line.starts_with("#![") {
            break;
        }
        let Some(rest) = line[3..].trim_start().strip_prefix(attr) else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix('=') else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix('"') else {
            continue;
        };
        let end = rest.find('"')?;
        return Some(rest[..end].to_string());
    }
    None
}

/// First `error` line of a compiler's stderr, for failure reasons.
fn first_error_line(stderr: &str) -> String {
    for line in stderr.lines() {
        let line = line.trim();
        if line.starts_with("error") {
            return line.chars().take(200).collect();
        }
    }
    stderr
        .lines()
        .next()
        .unwrap_or("<no compiler output>")
        .chars()
        .take(200)
        .collect()
}

/// Index just past the balanced `open`..`close` group at the
/// start of `s` (`s` must start with `open`), or `None` when the
/// group never closes. Nesting of the same bracket pair counts;
/// string literals inside are not lexed (attribute arguments in
/// test sources do not hide brackets inside strings).
fn skip_balanced(s: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0usize;
    for (i, c) in s.char_indices() {
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(i + c.len_utf8());
            }
        }
    }
    None
}

/// The crate name an `extern crate <name>` item on this line
/// declares, if the line is one. Attributes (`#[no_link]`,
/// `#[macro_use]`, possibly several in a row, possibly carrying
/// arguments) and a visibility (`pub`, `pub(crate)`) may precede
/// the item — auxiliary sources write e.g.
/// `#[macro_use] #[no_link] extern crate empty_crate_1 as t1;`,
/// and missing those declarations left the dependency unbound
/// (E0463 in the dependent's build). An `as` rename does not
/// change the declared crate name, and `extern crate self as
/// ...` declares no external crate.
fn extern_crate_item_name(line: &str) -> Option<String> {
    let mut rest = line.trim_start();
    loop {
        if rest.starts_with("#[") {
            let brackets = &rest[1..];
            rest = brackets[skip_balanced(brackets, '[', ']')?..].trim_start();
        } else if let Some(after) = rest.strip_prefix("pub") {
            if after.starts_with('(') {
                rest = after[skip_balanced(after, '(', ')')?..].trim_start();
            } else if after.starts_with(|c: char| c.is_whitespace()) {
                rest = after.trim_start();
            } else {
                break;
            }
        } else {
            break;
        }
    }
    let rest = rest.strip_prefix("extern crate ")?;
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() || name == "self" {
        None
    } else {
        Some(name)
    }
}

/// The crate names a flag list declares through `--extern`:
/// both the valued form (`--extern name=path` /
/// `--extern=name=path`) and the locator form (`--extern name` /
/// `--extern=name`), where rustc is told the name and locates
/// the crate itself. A test that locates an auxiliary crate this
/// way (tests/ui/resolve/extern-prelude.rs passes `--extern Vec`
/// for an auxiliary crate whose source declares
/// `#![crate_name = "Vec"]`) declares the name exactly as much
/// as an `extern crate` item does, so binding decisions count
/// it. The name is the part before the first `=`, verbatim
/// (modifiers included, as in `declared_aux_binding_names`).
pub(crate) fn locator_extern_names(flags: &[String]) -> Vec<String> {
    let mut names = Vec::new();
    let mut iter = flags.iter();
    while let Some(flag) = iter.next() {
        let value = if flag == "--extern" {
            iter.next().map(|v| v.as_str())
        } else {
            flag.strip_prefix("--extern=")
        };
        if let Some(value) = value {
            let name = value.split('=').next().unwrap_or("");
            if !name.is_empty() {
                names.push(name.to_string());
            }
        }
    }
    names
}

/// The subset of a source's declarations that are true
/// `--extern` binding declarations: the names of its
/// `//@ aux-crate:` directives, verbatim (modifiers and
/// namespaced paths included). Unlike an `aux-build:` stem or
/// an `extern crate` item, an `aux-crate:` name is the binding
/// itself: when it carries modifiers (`priv,noprelude:somedep`)
/// those modifiers are the semantics (noprelude keeps the crate
/// out of the extern prelude), so a consumer that declared one
/// must not also receive the crate under its unmodified build
/// name — that second binding silently undoes the modifiers
/// (tests/ui/extern-flag/multiple-opts.rs relies on the prelude
/// NOT containing the crate).
pub(crate) fn declared_binding_names(src_text: &str) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    for line in src_text.lines() {
        let Some(at) = line.find("//@") else { continue };
        let rest = line[at + 3..].trim_start();
        let rest = if rest.starts_with('[') {
            match rest.find(']') {
                Some(end) => rest[end + 1..].trim_start(),
                None => continue,
            }
        } else {
            rest
        };
        if let Some(value) = rest.strip_prefix("aux-crate:") {
            if let Some((name, _)) = value.trim().split_once('=') {
                names.insert(name.trim().to_string());
            }
        }
    }
    names
}

/// The `--extern` binding names a source file declares as its
/// dependencies: for `//@ aux-crate:` the name before the `=`
/// verbatim (modifiers and namespaced paths included), for
/// `//@ aux-build:` / `//@ proc-macro:` the file stem with dashes
/// as underscores, and for an `extern crate <name>;` item the
/// crate name. Scoped directives (`//@[rev] ...`) count too: the
/// union over revisions is what a caller that does not track
/// revisions can bind, and it is still exactly the set the file
/// declares somewhere — never unrelated crates.
pub(crate) fn declared_aux_binding_names(src_text: &str) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    for line in src_text.lines() {
        if let Some(name) = extern_crate_item_name(line) {
            names.insert(name);
        }
        let Some(at) = line.find("//@") else { continue };
        let rest = line[at + 3..].trim_start();
        let rest = if rest.starts_with('[') {
            match rest.find(']') {
                Some(end) => rest[end + 1..].trim_start(),
                None => continue,
            }
        } else {
            rest
        };
        if let Some(value) = rest.strip_prefix("aux-crate:") {
            if let Some((name, _)) = value.trim().split_once('=') {
                names.insert(name.trim().to_string());
            }
        } else {
            for directive in ["aux-build:", "proc-macro:"] {
                if let Some(value) = rest.strip_prefix(directive) {
                    let fname = value.trim().split_whitespace().next().unwrap_or("");
                    if let Some(stem) = std::path::Path::new(fname).file_stem() {
                        names.insert(stem.to_string_lossy().replace('-', "_"));
                    }
                }
            }
        }
    }
    names
}

/// The flags a source file's own `//@ compile-flags:` directives
/// declare (whitespace-split; scoped directives included, as in
/// [`declared_aux_binding_names`]). compiletest compiles each
/// auxiliary crate with its own file's flags; the main
/// compilation's flags come from the caller instead.
fn declared_compile_flags(src_text: &str) -> Vec<String> {
    let mut flags = Vec::new();
    for line in src_text.lines() {
        let Some(at) = line.find("//@") else { continue };
        let rest = line[at + 3..].trim_start();
        let rest = if rest.starts_with('[') {
            match rest.find(']') {
                Some(end) => rest[end + 1..].trim_start(),
                None => continue,
            }
        } else {
            rest
        };
        if let Some(value) = rest.strip_prefix("compile-flags:") {
            flags.extend(value.split_whitespace().map(str::to_string));
        }
    }
    flags
}

/// Directive scan shared by the auxiliary-header readers
/// below: yields each `//@` directive's text with any
/// `//@[revision]` scope prefix stripped, the approximation
/// declared_compile_flags already applies.
fn aux_directive_texts(src_text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in src_text.lines() {
        let Some(at) = line.find("//@") else { continue };
        let rest = line[at + 3..].trim_start();
        let rest = if rest.starts_with('[') {
            match rest.find(']') {
                Some(end) => rest[end + 1..].trim_start(),
                None => continue,
            }
        } else {
            rest
        };
        out.push(rest.to_string());
    }
    out
}

/// Whether an auxiliary source opts out of the prefer-dynamic
/// auxiliary build (`//@ no-prefer-dynamic`).
fn declared_no_prefer_dynamic(src_text: &str) -> bool {
    aux_directive_texts(src_text)
        .iter()
        .any(|d| d.starts_with("no-prefer-dynamic"))
}

/// An auxiliary source's own `//@ rustc-env:` pairs and
/// `//@ unset-rustc-env:` names.
fn declared_rustc_env(src_text: &str) -> (Vec<(String, String)>, Vec<String>) {
    let mut pairs = Vec::new();
    let mut unset = Vec::new();
    for d in aux_directive_texts(src_text) {
        if let Some(value) = d.strip_prefix("rustc-env:") {
            if let Some((name, value)) = value.trim().split_once('=') {
                pairs.push((name.trim().to_string(), value.trim().to_string()));
            }
        } else if let Some(value) = d.strip_prefix("unset-rustc-env:") {
            unset.push(value.trim().to_string());
        }
    }
    (pairs, unset)
}

/// The sysroot crates a source's `extern crate` items name that
/// the sysroot's target lib dir offers ambiguously, as
/// (name, rmeta, rlib) pins.
///
/// A rustup sysroot carrying the rustc-dev component holds two
/// rmetas for crates such as libc in its target lib dir: the
/// rust-std build's, paired with an rlib, and the compiler's
/// own build's, metadata only. An `extern crate` lookup then
/// fails with E0464 (multiple candidates), where the bootstrap
/// sysroot compiletest runs against holds a single candidate
/// per crate. The pin names the one candidate whose hash also
/// names an rlib -- the only linkable one -- in both its forms,
/// the pairing rustc takes as one crate (an rmeta-only pin
/// fails links that need the code; an rlib-only pin fails with
/// "only metadata stub found"). Mirrors the subset runner's
/// ambiguous_sysroot_crates for its native compilations.
/// The host target triple of a sysroot's rustc (`rustc -vV`'s
/// `host:` line), asked once per sysroot per process.
fn sysroot_host_target(sysroot: &str) -> Option<String> {
    static CACHE: std::sync::Mutex<Vec<(String, String)>> =
        std::sync::Mutex::new(Vec::new());
    if let Ok(cache) = CACHE.lock() {
        if let Some((_, host)) = cache.iter().find(|(s, _)| s == sysroot) {
            return Some(host.clone());
        }
    }
    let rustc = std::path::Path::new(sysroot).join("bin").join("rustc");
    let out = std::process::Command::new(&rustc)
        .arg("-vV")
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let host = text
        .lines()
        .find_map(|l| l.strip_prefix("host: "))
        .map(|h| h.trim().to_string())?;
    if let Ok(mut cache) = CACHE.lock() {
        cache.push((sysroot.to_string(), host.clone()));
    }
    Some(host)
}

fn sysroot_extern_pins(
    sysroot: &str,
    src_text: &str,
) -> Vec<(String, std::path::PathBuf, std::path::PathBuf)> {
    let mut declared: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for line in src_text.lines() {
        if let Some(name) = extern_crate_item_name(line) {
            declared.insert(name);
        }
    }
    if declared.is_empty() {
        return Vec::new();
    }
    // crate name -> hash -> (lib dir, has rmeta, has rlib)
    let mut by_name: std::collections::HashMap<
        String,
        std::collections::HashMap<String, (std::path::PathBuf, bool, bool)>,
    > = std::collections::HashMap::new();
    // Only the host target's lib dir is scanned: a toolchain
    // carrying std for many targets offers a complete
    // candidate for the name under most of them, and the
    // ambiguity this disambiguates is the host dir's own
    // (the rustc-dev component's second rmeta beside the
    // std component's rlib+rmeta pair). The host triple is
    // the sysroot rustc's own, asked once.
    let rustlib = std::path::Path::new(sysroot).join("lib").join("rustlib");
    let libdirs: Vec<std::path::PathBuf> = match sysroot_host_target(sysroot) {
        Some(host) => vec![rustlib.join(host).join("lib")],
        None => match std::fs::read_dir(&rustlib) {
            Ok(targets) => targets
                .flatten()
                .map(|t| t.path().join("lib"))
                .collect(),
            Err(_) => return Vec::new(),
        },
    };
    for libdir in libdirs {
        let Ok(entries) = std::fs::read_dir(&libdir) else {
            continue;
        };
        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().into_owned();
            let Some(rest) = fname.strip_prefix("lib") else {
                continue;
            };
            let Some((stem, ext)) = rest.rsplit_once('.') else {
                continue;
            };
            if ext != "rmeta" && ext != "rlib" {
                continue;
            }
            let Some((name, hash)) = stem.rsplit_once('-') else {
                continue;
            };
            if hash.len() != 16 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
                continue;
            }
            if !declared.contains(name) {
                continue;
            }
            let slot = by_name
                .entry(name.to_string())
                .or_default()
                .entry(hash.to_string())
                .or_insert_with(|| (libdir.clone(), false, false));
            if ext == "rmeta" {
                slot.1 = true;
            } else {
                slot.2 = true;
            }
        }
    }
    let mut out = Vec::new();
    for (name, hashes) in by_name {
        if hashes.len() < 2 {
            continue;
        }
        let linkable: Vec<(&String, &(std::path::PathBuf, bool, bool))> = hashes
            .iter()
            .filter(|(_, (_, rmeta, rlib))| *rmeta && *rlib)
            .collect();
        if linkable.len() == 1 {
            let (hash, (dir, _, _)) = linkable[0];
            out.push((
                name.clone(),
                dir.join(format!("lib{name}-{hash}.rmeta")),
                dir.join(format!("lib{name}-{hash}.rlib")),
            ));
        }
    }
    out
}

/// Expand compiletest's `{{...}}` template variables in an
/// auxiliary crate's declared compile flags. compiletest expands
/// every directive value before compiling an auxiliary crate with
/// the crate's own flags; passing `{{src-base}}` verbatim makes a
/// flag like `--remap-path-prefix={{src-base}}=remapped` a no-op,
/// so paths baked into the auxiliary crate's metadata (a macro's
/// `file!()`) stay unremapped. `{{src-base}}` is the auxiliary
/// file's suite root (the nearest ancestor directory whose parent
/// is `tests`, as in the subset runner) and `{{cwd}}` the driver's
/// working directory. Variables that name directories only
/// compiletest owns (`{{build-base}}` and friends) are left as-is:
/// no auxiliary crate's flags use them.
fn expand_aux_flag_variables(flag: &str, src_path: &str) -> String {
    if !flag.contains("{{") {
        return flag.to_string();
    }
    let src = std::path::Path::new(src_path);
    let mut src_base = src.parent().map(|p| p.to_path_buf());
    for ancestor in src.ancestors() {
        let is_suite_root = ancestor
            .parent()
            .is_some_and(|p| p.file_name().is_some_and(|n| n == "tests"));
        if is_suite_root {
            src_base = Some(ancestor.to_path_buf());
            break;
        }
    }
    let mut out = flag.to_string();
    if let Some(base) = &src_base {
        out = out.replace("{{src-base}}", &base.to_string_lossy());
    }
    if let Ok(cwd) = std::env::current_dir() {
        out = out.replace("{{cwd}}", &cwd.to_string_lossy());
    }
    out
}

/// One `--aux` specification: the crate name and source path from
/// `<name>=<file.rs>`, plus the optional per-crate attributes that
/// followed the path (`;crate-type=<type>`, `;edition=<ed>`,
/// `;cfg=<name>`).
pub(crate) struct AuxSpec {
    pub(crate) name: String,
    pub(crate) path: String,
    pub(crate) crate_type: Option<String>,
    pub(crate) edition: Option<String>,
    cfgs: Vec<String>,
}

/// Parse one `--aux` value: `<name>=<file.rs>` optionally followed
/// by `;crate-type=<type>`, `;edition=<ed>` and/or `;cfg=<name>`
/// attributes. Anything else is a usage error.
pub(crate) fn parse_aux_spec(spec: &str) -> Result<AuxSpec, String> {
    let (name, rest) = spec
        .split_once('=')
        .filter(|(n, r)| !n.is_empty() && !r.is_empty())
        .ok_or_else(|| format!("--aux needs a <name>=<file.rs> value, got: {spec}"))?;
    let mut parts = rest.split(';');
    let path = parts.next().unwrap_or("");
    if path.is_empty() {
        return Err(format!("--aux needs a <name>=<file.rs> value, got: {spec}"));
    }
    let mut parsed = AuxSpec {
        name: name.to_string(),
        path: path.to_string(),
        crate_type: None,
        edition: None,
        cfgs: Vec::new(),
    };
    for attr in parts {
        let (key, value) = attr
            .split_once('=')
            .filter(|(_, v)| !v.is_empty())
            .ok_or_else(|| format!("--aux attribute needs <key>=<value>, got: {attr}"))?;
        match key {
            "crate-type" => parsed.crate_type = Some(value.to_string()),
            "edition" => parsed.edition = Some(value.to_string()),
            "cfg" => parsed.cfgs.push(value.to_string()),
            _ => return Err(format!("unknown --aux attribute: {key}")),
        }
    }
    Ok(parsed)
}

/// Parse one `--aux-lib` value: `<name>=<path>` optionally followed
/// by `;collect=no`. A prebuilt auxiliary binds into the main
/// compilation and into the auxiliary builds that declare it,
/// exactly like a compiled one; whether its functions join the
/// converted program is the caller's `collect` choice (default
/// yes, a compiled auxiliary's treatment). A caller whose artifact
/// must not be a conversion source -- the minicore stub, whose
/// rlib carries no encoded MIR for its non-generic functions --
/// passes `collect=no`: the crate still binds everywhere it is
/// declared, it is simply never collected.
pub(crate) fn parse_aux_lib_spec(
    spec: &str,
) -> Result<(String, std::path::PathBuf, bool), String> {
    let (name, rest) = spec
        .split_once('=')
        .filter(|(n, r)| !n.is_empty() && !r.is_empty())
        .ok_or_else(|| format!("--aux-lib needs a <name>=<path> value, got: {spec}"))?;
    let mut parts = rest.split(';');
    let path = parts.next().unwrap_or("");
    if path.is_empty() {
        return Err(format!("--aux-lib needs a <name>=<path> value, got: {spec}"));
    }
    let mut collect = true;
    for attr in parts {
        let (key, value) = attr
            .split_once('=')
            .filter(|(_, v)| !v.is_empty())
            .ok_or_else(|| format!("--aux-lib attribute needs <key>=<value>, got: {attr}"))?;
        match (key, value) {
            ("collect", "no") => collect = false,
            ("collect", "yes") => collect = true,
            _ => return Err(format!("unknown --aux-lib attribute: {attr}")),
        }
    }
    Ok((name.to_string(), std::path::PathBuf::from(path), collect))
}

/// One compiled auxiliary crate.
#[derive(Clone)]
pub(crate) struct AuxBuilt {
    /// The name the crate was compiled under: its source's own
    /// `#![crate_name]` when it declares one, else the source file
    /// stem with dashes as underscores — the name compiletest's
    /// auxiliary builds carry, and the name the crate's metadata
    /// (and so the Stable MIR session) knows it by.
    pub(crate) name: String,
    /// The name dependents and the main compilation bind the crate
    /// under with `--extern`: the `--aux` spec's name verbatim. It
    /// may carry `--extern` modifiers (`priv:foo`,
    /// `priv,noprelude:foo`) or a namespaced-crates path
    /// (`my_api::utils`), neither of which is a valid `--crate-name`
    /// for the auxiliary build itself.
    pub(crate) extern_name: String,
    /// Artifact to link with `--extern`; `None` for crate types
    /// that expose no Rust API (cdylib and friends).
    pub(crate) artifact: Option<std::path::PathBuf>,
    /// Whether the crate's functions are guest code to convert
    /// (false for proc-macro and cdylib crates).
    pub(crate) collect: bool,
    /// The crate's resolved overflow-checks setting: what its
    /// own build flags resolve to under the session rule (see
    /// `session_overflow_checks`), the setting native codegen
    /// compiled this crate's instantiations of
    /// `#[rustc_inherit_overflow_checks]` bodies with. For a
    /// prebuilt library (--aux-lib) there is no build to read;
    /// `main` records the main compilation's setting, which is
    /// also what the interpreter fell back to for every crate
    /// before this table existed.
    pub(crate) overflow_checks: bool,
}

/// The names under which a consumer may declare a built
/// auxiliary crate: the `--extern` binding name (the `--aux`
/// spec's name verbatim) and the build name (the source's
/// declared `#![crate_name]`, else the file stem). Which one a
/// declaration uses depends on the declaring source: an
/// `aux-crate:` directive names the binding, while an
/// `extern crate` item that refers to a crate by the name its
/// own source declares (`#![crate_name = "a"]` in the svh
/// crates, `issue6919_3`, `Vec`) uses the build name. Matching
/// only the binding name left those dependencies unbound
/// (E0463 in the dependent's or the main compilation).
fn binding_names_for(built: &AuxBuilt) -> [&str; 2] {
    [built.extern_name.as_str(), built.name.as_str()]
}

/// The names to bind `built` under for a consumer declaring
/// `declared`: every name of `binding_names_for(built)` the
/// consumer declared (binding name first), or none. A consumer
/// can declare both names of the same crate — a test file
/// carries `//@ aux-build:iss-6919.rs` (the binding name) and
/// `extern crate issue6919_3;` (the build name) — and its
/// `extern crate` item looks the crate up by the name it
/// writes, so binding only one of the two leaves the other
/// reference unresolved (E0463). When two built crates share a
/// declared name (two svh `a` crates, the duplicate-rlib `foo`
/// crates), each is bound in build order and rustc's own
/// `--extern` precedence decides, as under compiletest.
///
/// One exception: when the consumer declared the binding name
/// through a true binding declaration (`binding_declared`: an
/// `aux-crate:` directive or a locator `--extern`), that
/// binding alone is emitted. Its modifiers are load-bearing —
/// with `priv,noprelude:somedep` bound, an additional
/// unmodified `--extern somedep=` would put the crate in the
/// extern prelude the modifier exists to keep it out of, and
/// `extern crate somedep;` items still resolve through the
/// modified binding (rustc matches them by crate name).
pub(crate) fn bind_names_for<'a>(
    built: &'a AuxBuilt,
    declared: &std::collections::HashSet<String>,
    binding_declared: &std::collections::HashSet<String>,
) -> Vec<&'a str> {
    if binding_declared.contains(&built.extern_name) {
        return vec![built.extern_name.as_str()];
    }
    binding_names_for(built)
        .into_iter()
        .filter(|name| declared.contains(*name))
        .collect()
}

/// Compile one auxiliary source file with the pinned rustc into
/// `aux_dir`, under the source's declared `#![crate_name]` or the
/// file-stem name compiletest's auxiliary builds use (never the
/// `--aux` spec's binding name: that string may carry `--extern`
/// modifiers or a namespaced-crates path, which are binding syntax,
/// not crate names). Previously built
/// auxiliary crates are visible through `--extern`, so callers pass
/// the crates in dependency order. The crate is built with
/// `spec.edition` when the caller attached one (compiletest builds
/// each auxiliary crate with its own edition) and with
/// `default_edition` otherwise; its crate type is the source's own
/// `#![crate_type]` declaration when present, else the caller's
/// `spec.crate_type`, else `rlib`. The build carries every
/// `spec.cfgs` entry as a `--cfg` (with a matching `--check-cfg`
/// declaration, so the name is a known cfg): compiletest compiles
/// a revised test's auxiliary crates with the test's compile
/// arguments, revision `--cfg` included, and an auxiliary crate
/// can select per-revision definitions through it. On failure,
/// returns the first compiler error line as the reason.
pub(crate) fn compile_aux(
    sysroot: &str,
    default_edition: &str,
    aux_dir: &std::path::Path,
    spec: &AuxSpec,
    previous: &[AuxBuilt],
    native_lib_dirs: &[std::path::PathBuf],
    env_scrub: &[String],
    mode: Mode,
) -> Result<AuxBuilt, String> {
    let name = spec.name.as_str();
    let src_path = spec.path.as_str();
    let edition = spec.edition.as_deref().unwrap_or(default_edition);
    let src_text = std::fs::read_to_string(src_path)
        .map_err(|e| format!("cannot read auxiliary source {src_path}: {e}"))?;
    let declared = declared_crate_type(&src_text);
    let effective_type = declared
        .as_deref()
        .or(spec.crate_type.as_deref())
        .unwrap_or("rlib");
    let declared_name = declared_crate_name(&src_text);
    // The build name: the declared `#![crate_name]`, else the file
    // stem with dashes as underscores (compiletest's
    // path_to_crate_name). The spec's binding name is used only for
    // `--extern`, never as `--crate-name`.
    let stem_name = std::path::Path::new(src_path)
        .file_stem()
        .map(|stem| stem.to_string_lossy().replace('-', "_"));
    let effective_name = declared_name.as_deref().or(stem_name.as_deref()).unwrap_or(name);
    let rustc = std::path::Path::new(sysroot).join("bin").join("rustc");
    let mut cmd = std::process::Command::new(&rustc);
    // `-Zalways-encode-mir` makes the compiler encode every
    // function's MIR into the crate metadata; without it, the
    // metadata of a library crate only carries MIR for generic and
    // inline functions, and the main compilation's Stable MIR
    // session would find no bodies for the auxiliary functions.
    // The flag is unstable, so the auxiliary compilation runs with
    // RUSTC_BOOTSTRAP=1 — the same condition this driver itself is
    // built under, and the condition compiletest runs under (a
    // development compiler).
    cmd.env("RUSTC_BOOTSTRAP", "1");
    cmd.arg(src_path)
        .arg("-Zalways-encode-mir")
        .arg(format!("--edition={edition}"));
    // In the coverage modes, compiletest builds auxiliary
    // crates with the mode's instrumentation flags as well
    // (its make_compile_args applies them to every compilation
    // of the mode, and coverage mappings in auxiliary libraries
    // are merged into the final executable): the auxiliary
    // functions' coverage info must be encoded in their MIR
    // for the main capture to read.
    // The mode's opt level is not applied to auxiliary builds:
    // at opt-level 2 an auxiliary body's encoded MIR inlines
    // deeply enough that converting the main test then demands
    // bodies that have no encoded MIR at all (core's
    // panic_nounwind and friends). Instrumentation alone gives
    // the auxiliary functions their coverage info; the records
    // of generic auxiliary functions are produced by the main
    // compilation (at its own opt level) anyway.
    match mode {
        Mode::CoverageMap => {
            cmd.arg("-C").arg("instrument-coverage")
                .arg("-Zno-profiler-runtime");
        }
        Mode::CoverageRun => {
            cmd.arg("-C").arg("instrument-coverage");
        }
        _ => {}
    }
    for cfg in &spec.cfgs {
        cmd.arg("--cfg").arg(cfg);
        cmd.arg("--check-cfg").arg(format!("cfg({cfg})"));
    }
    cmd.arg(format!("--out-dir={}", aux_dir.display()))
        .arg(format!("--sysroot={sysroot}"))
        .arg("-L")
        .arg(format!("dependency={}", aux_dir.display()));
    // The harness's native support library (librust_test_helpers
    // .a, built once by the subset runner from the test tree's
    // rust_test_helpers.c): auxiliary crates that name it in a
    // `#[link]` attribute fail with "could not find native
    // static library" when their build cannot see the directory
    // the native auxiliary builds are handed.
    for dir in native_lib_dirs {
        cmd.arg("-L").arg(format!("native={}", dir.display()));
    }
    if declared.is_none() {
        cmd.arg(format!("--crate-type={effective_type}"));
    }
    if declared_name.is_none() {
        cmd.arg(format!("--crate-name={effective_name}"));
    }
    // Bind only the auxiliary crates this source declares as its
    // own dependencies (compiletest binds an auxiliary crate's
    // declared deps, not every crate built so far): binding an
    // unrelated crate can change the outcome — a `priv:`-modified
    // or namespaced binding is invalid without the flags that
    // accompany its declaration, and a crate visible as a direct
    // dependency is not a private one. Undeclared transitive deps
    // still resolve through the `-L dependency` directory above.
    let mut deps = declared_aux_binding_names(&src_text);
    let own_flags = declared_compile_flags(&src_text);
    deps.extend(locator_extern_names(&own_flags));
    let mut binding_declared = declared_binding_names(&src_text);
    binding_declared.extend(locator_extern_names(&own_flags));
    for prev in previous {
        if let Some(artifact) = &prev.artifact {
            for bind in bind_names_for(prev, &deps, &binding_declared) {
                cmd.arg("--extern")
                    .arg(format!("{}={}", bind, artifact.display()));
            }
        }
    }
    // Pin the sysroot crates this source declares that the
    // sysroot offers ambiguously (sysroot_extern_pins): the
    // same disambiguation the subset runner applies to its
    // native builds, without which an auxiliary source's
    // `extern crate libc;` fails with E0464 in a rustup sysroot
    // carrying the rustc-dev component. A name the source binds
    // to an auxiliary crate built above keeps that binding.
    for (name, rmeta, rlib) in sysroot_extern_pins(sysroot, &src_text) {
        if binding_declared.contains(&name) {
            continue;
        }
        cmd.arg("--extern").arg(format!("{name}={}", rmeta.display()));
        cmd.arg("--extern").arg(format!("{name}={}", rlib.display()));
    }
    // compiletest compiles an auxiliary crate with
    // `-C prefer-dynamic` unless the crate opts out with
    // `//@ no-prefer-dynamic` or sets the flag itself (the
    // subset runner's native auxiliary builds do the same): a
    // dylib auxiliary built without it cannot satisfy the
    // "std only shows up once" requirement against the
    // prefer-dynamic sysroot std (linkage-attr/issue-12133-3).
    if !declared_no_prefer_dynamic(&src_text)
        && !own_flags.iter().any(|f| f.contains("prefer-dynamic"))
    {
        cmd.arg("-C").arg("prefer-dynamic");
    }
    // The auxiliary crate's own `//@ rustc-env:` /
    // `//@ unset-rustc-env:`: compiletest configures each
    // auxiliary build's environment from the auxiliary file's
    // own directives, separately from the test's (the
    // compiletest-self-test env.rs auxiliary reads
    // COMPILETEST_BAR through env!() and does not compile
    // without it).
    // The test's own rustc-env names are scrubbed first:
    // compiletest configures an auxiliary build's environment
    // separately from the test's, so a variable the test's
    // rustc-env set for the main compilation (which the
    // subset runner hands this process as its environment)
    // must not leak into the auxiliary build, whose own
    // directives are applied after the scrub and win.
    for name in env_scrub {
        cmd.env_remove(name);
    }
    let (aux_env, aux_unset) = declared_rustc_env(&src_text);
    for name in &aux_unset {
        cmd.env_remove(name);
    }
    for (name, value) in &aux_env {
        cmd.env(name, expand_aux_flag_variables(value, src_path));
    }
    // The auxiliary crate's own `//@ compile-flags:` (compiletest
    // compiles each auxiliary crate with its own file's flags,
    // with the `{{...}}` template variables expanded).
    for flag in declared_compile_flags(&src_text) {
        cmd.arg(expand_aux_flag_variables(&flag, src_path));
    }
    // The crate's own overflow-checks setting, resolved from
    // the very command line it is about to be built with (its
    // own `//@ compile-flags:` included above): the setting its
    // instantiations of checked library bodies were compiled
    // under, recorded for the interpreter's program.
    let cmd_args: Vec<String> = cmd
        .get_args()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    let overflow_checks = super::session_overflow_checks(&cmd_args);
    let out = cmd
        .output()
        .map_err(|e| format!("failed to run rustc for auxiliary crate `{name}`: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "auxiliary crate `{name}` failed to compile: {}",
            first_error_line(&stderr)
        ));
    }
    // Locate the produced artifact: lib<name>.rlib / .rmeta /
    // .so. A build carrying `-C extra-filename=<suffix>` (an
    // auxiliary source's own compile-flags can set one — the
    // crateresolve crates use `-C extra-filename=-1`) names its
    // artifact lib<name><suffix>.rlib instead, with the suffix
    // between the crate name and the extension; accept that
    // shape too, ranked below the plain lib<name>. artifact. A
    // `-` after the name can only be that separator: crate names
    // never contain one (declared names forbid it, and file
    // stems had theirs replaced with underscores).
    let mut artifact: Option<std::path::PathBuf> = None;
    let mut best_rank = (u8::MAX, u8::MAX);
    if let Ok(entries) = std::fs::read_dir(aux_dir) {
        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().into_owned();
            let group = if fname.starts_with(&format!("lib{effective_name}.")) {
                0
            } else if fname.starts_with(&format!("lib{effective_name}-")) {
                1
            } else {
                continue;
            };
            let ext = match fname.rsplit('.').next() {
                Some("rlib") => 0,
                Some("rmeta") => 1,
                Some("so") | Some("dylib") | Some("dll") => 2,
                _ => continue,
            };
            if (group, ext) < best_rank {
                best_rank = (group, ext);
                artifact = Some(entry.path());
            }
        }
    }
    let (link, collect) = match effective_type {
        "rlib" | "lib" | "dylib" => (true, true),
        "proc-macro" => (true, false),
        _ => (false, false),
    };
    if link && artifact.is_none() {
        return Err(format!(
            "auxiliary crate `{name}` produced no linkable artifact in {}",
            aux_dir.display()
        ));
    }
    Ok(AuxBuilt {
        name: effective_name.to_string(),
        extern_name: name.to_string(),
        artifact: if link { artifact } else { None },
        collect,
        overflow_checks,
    })
}
