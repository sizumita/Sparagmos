//! A bounded snapshot of the source tree, used to detect sources changing under a tool run.


/// The source tree one build converted: every file under
/// the crate root's directory at the moment of the build,
/// as (path relative to that directory, bytes) pairs, the
/// root itself among them under its file name. A crate is
/// more than its root file -- `mod` files and `include!`
/// inputs live beside it -- so an artifact denoting "the
/// program its build produced" must denote the tree, not
/// the root alone: rewriting only a module file after the
/// build must not change what an earlier artifact runs,
/// and rewriting or deleting the root must not orphan the
/// artifact while its modules survive in the snapshot.
pub(crate) type SourceTree = Vec<(std::path::PathBuf, Vec<u8>)>;

/// The bounds past which a directory is not read as a
/// crate's source tree: past them the snapshot keeps the
/// root alone (the root-only behaviour) rather than
/// copying a tree the build could not have read whole.
const SOURCE_TREE_MAX_FILES: usize = 4096;
const SOURCE_TREE_MAX_BYTES: usize = 64 << 20;
const SOURCE_TREE_MAX_DEPTH: usize = 32;

/// Read `path` holding at most `limit` bytes: `None` when
/// the file cannot be opened or read, or when it holds
/// more than `limit` bytes. The size stands checked from
/// the metadata before any byte is read, and the read
/// itself is capped at `limit + 1` bytes, so a file that
/// grows between the check and the read is caught by the
/// extra byte rather than allocated whole: a snapshot
/// read must never take in more than its bound, however
/// large a file in the tree stands on disk.
fn read_bounded(path: &std::path::Path, limit: usize) -> Option<Vec<u8>> {
    use std::io::Read;
    if std::fs::metadata(path).ok()?.len() > limit as u64 {
        return None;
    }
    let file = std::fs::File::open(path).ok()?;
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1).read_to_end(&mut bytes).ok()?;
    if bytes.len() > limit {
        return None;
    }
    Some(bytes)
}

/// Snapshot the source tree rooted at `root` (see
/// `SourceTree`): the root's own bytes and, recursively,
/// every regular file under the root's directory, in
/// deterministic (sorted) order.
pub(crate) fn snapshot_source_tree(root: &std::path::Path) -> SourceTree {
    let root_name = root
        .file_name()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("source.rs"));
    let root_bytes = read_bounded(root, SOURCE_TREE_MAX_BYTES).unwrap_or_default();
    let mut tree: SourceTree = vec![(root_name, root_bytes)];
    if let Some(base) = root.parent() {
        let mut budget = (SOURCE_TREE_MAX_FILES, SOURCE_TREE_MAX_BYTES);
        if !collect_source_tree(base, base, 0, &mut tree, &mut budget) {
            // Over the bounds: keep the root alone.
            tree.truncate(1);
            return tree;
        }
    }
    let mut seen = std::collections::HashSet::new();
    tree.retain(|(rel, _)| seen.insert(rel.clone()));
    tree.sort();
    tree
}

/// The recursive walk under `snapshot_source_tree`; `false`
/// when the tree runs past the snapshot bounds.
fn collect_source_tree(
    base: &std::path::Path,
    dir: &std::path::Path,
    depth: usize,
    tree: &mut SourceTree,
    budget: &mut (usize, usize),
) -> bool {
    if depth > SOURCE_TREE_MAX_DEPTH {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        // An unreadable directory contributes nothing.
        return true;
    };
    let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            if !collect_source_tree(base, &path, depth + 1, tree, budget) {
                return false;
            }
        } else if path.is_file() {
            if budget.0 == 0 {
                return false;
            }
            // The size stands checked before any read: a
            // file past the remaining byte budget ends the
            // walk without its bytes ever being allocated,
            // however large it stands on disk.
            match std::fs::metadata(&path) {
                Ok(meta) if meta.len() > budget.1 as u64 => return false,
                Ok(_) => {}
                Err(_) => continue,
            }
            // The read itself stays bounded by the same
            // budget, so a file that grows between the
            // check and the read ends the walk the same
            // way instead of being allocated whole; a file
            // that cannot be read at all leaves the walk
            // the same way, since a tree this walk cannot
            // read whole is not one it can snapshot.
            let Some(bytes) = read_bounded(&path, budget.1) else {
                return false;
            };
            budget.0 -= 1;
            budget.1 -= bytes.len();
            let Ok(rel) = path.strip_prefix(base) else {
                continue;
            };
            tree.push((rel.to_path_buf(), bytes));
        }
    }
    true
}

/// Whether the tree standing at `root`'s directory still
/// denotes the conversion the build made: every file of
/// `tree` holds the bytes the build converted, and no
/// source file has been added beside them. An addition
/// can change what a re-conversion reads even when every
/// recorded file stands unchanged -- a `mod helper;` the
/// build resolved to the recorded `helper.rs` gains a
/// second candidate when `helper/mod.rs` appears beside
/// it, and converting the live tree then fails with
/// E0761 although the program the build produced is
/// unchanged -- so an added `.rs` file anywhere in the
/// tree ends the intact reading too, and the run falls
/// back to the recorded snapshot (see
/// `tree_has_added_source_file`).
pub(crate) fn source_tree_intact(root: &std::path::Path, tree: &SourceTree) -> bool {
    let Some(base) = root.parent() else {
        return false;
    };
    if !tree.iter().all(|(rel, bytes)| {
        read_bounded(&base.join(rel), bytes.len()).is_some_and(|now| now == *bytes)
    }) {
        return false;
    }
    let recorded: std::collections::HashSet<&std::path::Path> =
        tree.iter().map(|(rel, _)| rel.as_path()).collect();
    let mut budget = SOURCE_TREE_MAX_FILES;
    !tree_has_added_source_file(base, base, 0, &recorded, &mut budget)
}

/// Whether a `.rs` file the recorded tree does not hold
/// has appeared anywhere under `dir` (recursing from
/// `base`): the addition scan behind `source_tree_intact`.
/// Only `.rs` files take part -- they are the files whose
/// presence steers source resolution (`mod` candidates
/// and `#[path]` targets). An added file of any other
/// kind cannot change what a conversion reads: a file
/// the build's conversion did read (an `include!` input,
/// say) stood in the tree at build time, and so stands
/// recorded. The scan gives up -- reports no addition --
/// once it has looked at more entries than the snapshot
/// bounds allow: a tree that large was never recorded
/// whole (see `snapshot_source_tree`), and its runs keep
/// the recorded-files-only reading rather than falling
/// back to a snapshot that does not hold the tree.
fn tree_has_added_source_file(
    base: &std::path::Path,
    dir: &std::path::Path,
    depth: usize,
    recorded: &std::collections::HashSet<&std::path::Path>,
    budget: &mut usize,
) -> bool {
    if depth > SOURCE_TREE_MAX_DEPTH {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        // An unreadable directory contributes nothing.
        return false;
    };
    let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        let path = entry.path();
        if path.is_dir() {
            if tree_has_added_source_file(base, &path, depth + 1, recorded, budget) {
                return true;
            }
        } else if path.is_file() && path.extension().is_some_and(|ext| ext == "rs") {
            let Ok(rel) = path.strip_prefix(base) else {
                continue;
            };
            if !recorded.contains(rel) {
                return true;
            }
        }
    }
    false
}
