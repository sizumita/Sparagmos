//! Path value manipulation for the path builtins.

use std::cmp::Ordering;


use crate::error::{InterpError, InterpResult};
use crate::interp::{
    Frame, Interpreter,
};
use crate::value::{RefProj, RefTarget, Value};

use super::*;

impl<'a> Interpreter<'a> {
    /// The fat reference `value` shortened to its first
    /// `len` elements, the sub-slice view
    /// `CStr::from_bytes_until_nul` returns: a snapshot of
    /// the prefix elements under the shortened length.
    pub(super) fn fat_ref_prefix(
        &self,
        frames: &[Frame],
        value: &Value,
        len: usize,
    ) -> InterpResult<Value> {
        // The prefix is materialised as a snapshot of its
        // elements: a fat reference that keeps the source's
        // base would have its length re-derived from that
        // base (the full slice) when the value is written
        // to a typed place, losing the shortening.
        let (items, _) = self.slice_elements(frames, value)?;
        if items.len() < len {
            return Err(InterpError::type_mismatch(
                "sub-slice prefix",
                "a slice of at least the requested length",
                format!("{} elements", items.len()),
            ));
        }
        Ok(Value::FatRef {
            base: Box::new(Value::Ref(RefTarget::Value(
                Box::new(Value::Tuple(items.into_iter().take(len).collect())),
                Vec::new(),
            ))),
            len: len as u128,
        })
    }

    /// Whether a comparison call compares paths: the
    /// call's generic arguments name a path type
    /// (`path_cmp_subject`), or — for a call the converter
    /// kept in generic form, which carries no type tags —
    /// both operands are owned `PathBuf` values by shape
    /// (`is_pathbuf_value`).
    pub(super) fn comparison_is_path(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        vals: &[Value],
    ) -> bool {
        if path_cmp_subject(generic_args) {
            return true;
        }
        if !generic_args.is_empty() || vals.len() != 2 {
            return false;
        }
        if vals.iter().all(|v| {
            self.resolve_value(frames, v)
                .is_ok_and(|resolved| is_pathbuf_value(&resolved))
        }) {
            return true;
        }
        // The reference form of the same signal: a `&Path`
        // operand (the shape a generic adapter inside a
        // library body passes, e.g. the less-than closure a
        // slice sort derives from `Ord`) is a fat pointer
        // whose base target was reached through the path
        // wrapper chain — the two leading `Field(0)` steps
        // (`PathBuf`'s `OsString` field, then its buffer
        // field) the `as_path` formation appends. A byte
        // slice's fat pointer is formed from a vector place
        // or a constant directly and carries no such descent.
        vals.iter().all(|v| self.operand_is_path_ref(frames, v))
    }

    /// Whether a comparison operand denotes a path through
    /// references: unwrapping reference layers reaches a fat
    /// pointer whose base target's projection opens with the
    /// two `Field(0)` steps of the path wrapper chain (see
    /// [`Self::comparison_is_path`]).
    fn operand_is_path_ref(&self, frames: &[Frame], value: &Value) -> bool {
        let mut cur = value.clone();
        for _ in 0..4 {
            match cur {
                Value::Ref(_) => {
                    let Ok(target) = self.deref_value(frames, &cur) else {
                        return false;
                    };
                    let Ok(stored) = self.read_resolved(frames, &target) else {
                        return false;
                    };
                    cur = stored;
                }
                Value::FatRef { base, .. } => {
                    let Value::Ref(target) = &*base else {
                        return false;
                    };
                    let projection = match target {
                        RefTarget::Place { projection, .. } => projection,
                        RefTarget::Value(_, projection) => projection,
                        RefTarget::Static { projection, .. } => projection,
                        RefTarget::Alloc { projection, .. } => projection,
                        _ => return false,
                    };
                    return matches!(
                        projection.first(),
                        Some(RefProj::Field(0))
                    ) && matches!(
                        projection.get(1),
                        Some(RefProj::Field(0))
                    );
                }
                _ => return false,
            }
        }
        false
    }

    /// The bytes of a path argument: a path value is a fat
    /// pointer over the path's bytes (or a reference to one).
    /// The byte string a `PartialEq` operand for a path type
    /// denotes: the fat-pointer form reads through
    /// [`Self::path_bytes`]; an owned `PathBuf` value (a
    /// one-field aggregate chain around the string) reads
    /// through [`Self::os_str_bytes`].
    pub(super) fn path_cmp_bytes(&self, frames: &[Frame], value: &Value) -> InterpResult<Vec<u8>> {
        if let Ok(bytes) = self.path_bytes(frames, value) {
            return Ok(bytes);
        }
        // An owned path value may be stored in the aggregate
        // shapes `string_aggregate_bytes` recognises (a wrapper
        // chain around a byte-element aggregate, as the
        // `current_exe` model's `PathBuf` is stored).
        if let Ok(resolved) = self.resolve_value(frames, value) {
            if let Some(bytes) = string_aggregate_bytes(&resolved) {
                return Ok(bytes);
            }
        }
        self.os_str_bytes(frames, value)
    }

    pub(super) fn path_bytes(&self, frames: &[Frame], value: &Value) -> InterpResult<Vec<u8>> {
        Ok(self.path_bytes_of(frames, value)?.0)
    }
}

impl<'a> Interpreter<'a> {
    pub(super) fn path_bytes_of(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<(Vec<u8>, bool)> {
        match value {
            Value::Str(text) => Ok((text.clone().into_bytes(), true)),
            Value::Bytes(b) => Ok((b.clone(), false)),
            Value::FatRef { .. } => {
                let (items, _) = self.slice_elements(frames, value)?;
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        Value::Uint(b) => out.push(b as u8),
                        other => {
                            return Err(InterpError::type_mismatch(
                                "path argument",
                                "byte elements",
                                other.kind_name(),
                            ))
                        }
                    }
                }
                Ok((out, false))
            }
            Value::Ref(_) => {
                let target = self.deref_value(frames, value)?;
                let stored = self.read_resolved(frames, &target)?;
                self.path_bytes_of(frames, &stored)
            }
            other => Err(InterpError::type_mismatch(
                "path argument",
                "path byte string",
                other.kind_name(),
            )),
        }
    }
}

/// The byte string at the bottom of an owned string-like
/// aggregate: one-field wrappers (a `PathBuf` around an
/// `OsString` around its buffer) peel to their field, and the
/// innermost element list (a `Vec<u8>` value) is the byte
/// string itself. `None` for any other shape.
/// One parsed path component, in std's `components` parsing
/// on Unix: a root directory, a leading current-directory
/// marker, a parent reference, or a normal name.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum PathComponent<'a> {
    RootDir,
    CurDir,
    ParentDir,
    Normal(&'a [u8]),
}

/// Parse path bytes into the component sequence std's
/// `Path::components` yields (and its `PartialEq` compares):
/// a leading run of separators is one root component,
/// repeated and trailing separators are dropped, an interior
/// `.` component is dropped, and a leading `.` component of
/// a relative path is the current-directory component.
pub(super) fn path_components(bytes: &[u8]) -> Vec<PathComponent<'_>> {
    let mut comps = Vec::new();
    let mut rest = bytes;
    if rest.first() == Some(&b'/') {
        comps.push(PathComponent::RootDir);
        let run = rest.iter().take_while(|b| **b == b'/').count();
        rest = &rest[run..];
    }
    let has_root = !comps.is_empty();
    let mut first = true;
    for piece in rest.split(|b| *b == b'/') {
        match piece {
            [] => {}
            [b'.'] if first && !has_root => comps.push(PathComponent::CurDir),
            [b'.'] => {}
            [b'.', b'.'] => comps.push(PathComponent::ParentDir),
            name => comps.push(PathComponent::Normal(name)),
        }
        first = false;
    }
    comps
}

/// Whether a comparison call's generic arguments name a
/// path type (`Path` / `PathBuf`) as the `Self` or `Rhs`
/// type. The callee name carries no impl subject in this
/// model, so the path comparisons are recognised by the
/// call's own generic arguments (the type tags, with any
/// reference marker stripped).
fn path_cmp_subject(generic_args: &[String]) -> bool {
    generic_args.iter().take(2).any(|tag| {
        let tag = tag
            .strip_prefix("&mut ")
            .or_else(|| tag.strip_prefix('&'))
            .unwrap_or(tag);
        matches!(tag, "std::path::Path" | "std::path::PathBuf")
    })
}

/// Whether a comparison call compares paths. The primary
/// signal is the call's own generic arguments
/// (`path_cmp_subject`). A call the converter kept in
/// generic form (a comparison inside a generic library
/// body, reached through the run-time name fallback)
/// carries no type tags at all; there the owned operands
/// themselves are the only signal, and an owned `PathBuf`
/// value is recognised by its exact shape — three
/// one-field wrappers (`PathBuf` over `OsString` over its
/// buffer) around the byte-element list. No other modelled
/// string-like type has that shape: an `OsString` has one
/// wrapper fewer, a `String` two fewer, and a `Vec<u8>`
/// none, so their byte-order semantics are untouched.
fn is_pathbuf_value(value: &Value) -> bool {
    let mut cur = value;
    let mut wrappers = 0usize;
    loop {
        match cur {
            Value::Enum { fields, .. }
                if fields.len() == 1
                    && matches!(fields[0], Value::Enum { .. }) =>
            {
                wrappers += 1;
                cur = &fields[0];
            }
            Value::Enum { fields, .. } => {
                return wrappers == 3
                    && fields
                        .iter()
                        .all(|f| matches!(f, Value::Uint(_) | Value::Int(_)));
            }
            _ => return false,
        }
    }
}

/// std's ordering of two paths: `Path::cmp` compares the
/// parsed component sequences lexicographically, and a
/// component orders first by kind — `Component`'s
/// declaration order (root directory, current directory,
/// parent directory, then normal names) — and only then by
/// the name's bytes. This differs from the raw byte order
/// wherever a name contains a byte below '/' (notably '.'):
/// "a/c" < "a.b/c" here, the opposite bytewise.
pub(super) fn path_ordering(lhs: &[u8], rhs: &[u8]) -> Ordering {
    fn key<'a>(component: &PathComponent<'a>) -> (u8, &'a [u8]) {
        match component {
            PathComponent::RootDir => (1, &[]),
            PathComponent::CurDir => (2, &[]),
            PathComponent::ParentDir => (3, &[]),
            PathComponent::Normal(name) => (4, name),
        }
    }
    path_components(lhs)
        .iter()
        .map(key)
        .cmp(path_components(rhs).iter().map(key))
}

pub(super) fn string_aggregate_bytes(value: &Value) -> Option<Vec<u8>> {
    match value {
        Value::Str(s) => Some(s.clone().into_bytes()),
        Value::Bytes(b) => Some(b.clone()),
        // The sparse element form (the shape a growth of the
        // bare element form leaves in a string's storage, see
        // `String::as_mut_vec`): the bytes are the first `len`
        // elements; any slots past them are spare capacity.
        Value::SparseVec { elems, len, .. } => {
            let len = usize::try_from(*len).ok()?;
            if elems.len() < len {
                return None;
            }
            let mut out = Vec::with_capacity(len);
            for f in &elems[..len] {
                match f {
                    Value::Uint(b) => out.push(*b as u8),
                    Value::Int(b) if (0..=255).contains(b) => out.push(*b as u8),
                    _ => return None,
                }
            }
            Some(out)
        }
        Value::Enum { fields, .. } => {
            if fields
                .iter()
                .all(|f| matches!(f, Value::Uint(_) | Value::Int(_)))
            {
                // The element list of the byte vector (an empty
                // list is the empty string).
                let mut out = Vec::with_capacity(fields.len());
                for f in fields {
                    match f {
                        Value::Uint(b) => out.push(*b as u8),
                        Value::Int(b) if (0..=255).contains(b) => {
                            out.push(*b as u8)
                        }
                        _ => return None,
                    }
                }
                Some(out)
            } else if fields.len() == 1 {
                string_aggregate_bytes(&fields[0])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// A `&Path` / `&OsStr` value over the given bytes. When the
/// bytes are UTF-8 the value is the string data itself, which
/// is what a path over string data denotes everywhere else in
/// this model (a `Path::new` over a `&str` keeps the string
/// value); non-UTF-8 bytes become a fat pointer to a fresh
/// aggregate of the byte elements, the shape the interpreter's
/// own slice operations produce for a materialised sub-slice.
pub(super) fn sub_path_value(bytes: &[u8]) -> Value {
    match std::str::from_utf8(bytes) {
        Ok(text) => Value::Str(text.to_string()),
        Err(_) => Value::FatRef {
            base: Box::new(Value::Tuple(
                bytes.iter().map(|b| Value::Uint(u128::from(*b))).collect(),
            )),
            len: bytes.len() as u128,
        },
    }
}

/// The byte length of a path's parent, following std's
/// components-based `Path::parent` on unix separators: trailing
/// separators and trailing `.` components take no part (as
/// `Components` normalizes them away), the last remaining
/// component is removed, and a root or empty path has no
/// parent. The parent is always a prefix of the path's bytes.
pub(super) fn path_parent_len(bytes: &[u8]) -> Option<usize> {
    let mut end = bytes.len();
    loop {
        if end > 1 && bytes[end - 1] == b'/' {
            end -= 1;
        } else if end >= 2 && bytes[end - 2] == b'/' && bytes[end - 1] == b'.' {
            end -= 2;
        } else {
            break;
        }
    }
    if end == 0 || bytes[..end].iter().all(|b| *b == b'/') {
        return None;
    }
    match bytes[..end].iter().rposition(|b| *b == b'/') {
        None => Some(0),
        Some(0) => Some(1),
        Some(mut i) => {
            while i > 1 && bytes[i - 1] == b'/' {
                i -= 1;
            }
            Some(i)
        }
    }
}

/// The byte range of a path's file name, following std's
/// `Path::file_name`: the final component when it is a normal
/// one (a trailing `.` component names the previous one, as
/// `Components` drops it; `..` and the root have no file name).
pub(super) fn path_file_name_range(bytes: &[u8]) -> Option<(usize, usize)> {
    let mut end = bytes.len();
    while end > 1 && bytes[end - 1] == b'/' {
        end -= 1;
    }
    if end == 0 || bytes[..end].iter().all(|b| *b == b'/') {
        return None;
    }
    let start = bytes[..end]
        .iter()
        .rposition(|b| *b == b'/')
        .map_or(0, |i| i + 1);
    let seg = &bytes[start..end];
    if seg == b".." {
        return None;
    }
    if seg == b"." {
        if start == 0 {
            return None;
        }
        return path_file_name_range(&bytes[..start - 1]);
    }
    Some((start, end))
}

/// One path component in std's `Components` normalization
/// (unix separators): a leading `.` is its own component, `.`
/// elsewhere and empty segments are dropped, and a leading
/// run of separators is the root.
/// A path's components, in `Components` order.
#[derive(PartialEq, Eq)]
enum PathComp<'a> {
    Root,
    Cur,
    Parent,
    Normal(&'a [u8]),
}

/// `PathBuf::push` on raw bytes (unix): an absolute argument
/// replaces the base; otherwise the argument is appended, with
/// a separator inserted unless the base is empty or already
/// ends with one.
pub(super) fn path_push_bytes(base: &[u8], path: &[u8]) -> Vec<u8> {
    if path.first() == Some(&b'/') {
        return path.to_vec();
    }
    if base.is_empty() || path.is_empty() {
        let mut out = base.to_vec();
        out.extend_from_slice(path);
        return out;
    }
    let mut out = base.to_vec();
    if !out.ends_with(b"/") {
        out.push(b'/');
    }
    out.extend_from_slice(path);
    out
}

fn path_comps(bytes: &[u8]) -> Vec<PathComp<'_>> {
    let mut comps = Vec::new();
    let mut rest = bytes;
    if rest.first() == Some(&b'/') {
        comps.push(PathComp::Root);
        while rest.first() == Some(&b'/') {
            rest = &rest[1..];
        }
    } else if rest == b"." || rest.starts_with(b"./") {
        comps.push(PathComp::Cur);
        rest = if rest.len() > 1 { &rest[2..] } else { &rest[1..] };
        while rest.first() == Some(&b'/') {
            rest = &rest[1..];
        }
    }
    for seg in rest.split(|b| *b == b'/') {
        match seg {
            b"" | b"." => {}
            b".." => comps.push(PathComp::Parent),
            s => comps.push(PathComp::Normal(s)),
        }
    }
    comps
}

/// std's `Path::starts_with`: the child's components are a
/// prefix of the path's components.
pub(super) fn path_starts_with(path: &[u8], child: &[u8]) -> bool {
    let comps = path_comps(path);
    let child = path_comps(child);
    comps.len() >= child.len() && comps[..child.len()] == child[..]
}

/// std's `Path::ends_with`: the child's components are a
/// suffix of the path's components.
pub(super) fn path_ends_with(path: &[u8], child: &[u8]) -> bool {
    let comps = path_comps(path);
    let child = path_comps(child);
    comps.len() >= child.len() && comps[comps.len() - child.len()..] == child[..]
}

/// std's `rsplit_file_at_dot` on a file name's bytes: the
/// split at the last dot, as a (before, after) pair, with std's
/// special cases (a `..` name splits as itself with no
/// extension side; a dot in the first position does not start
/// an extension).
pub(super) fn rsplit_file_at_dot(name: &[u8]) -> (Option<&[u8]>, Option<&[u8]>) {
    if name == b".." {
        return (Some(name), None);
    }
    match name.iter().rposition(|b| *b == b'.') {
        None => (None, Some(name)),
        Some(0) => (Some(name), None),
        Some(i) => (Some(&name[..i]), Some(&name[i + 1..])),
    }
}

/// The bytes of an `OsString` value, in either of the model's
/// two shapes: the string form (the inner data is the string
/// itself) and the buffer form (an `OsString` wrapping a `Buf`
/// wrapping the byte vector).
pub(super) fn os_string_bytes(value: &Value) -> Option<Vec<u8>> {
    let Value::Enum { fields, .. } = value else {
        return None;
    };
    match fields.as_slice() {
        [Value::Str(text)] => Some(text.clone().into_bytes()),
        [buf] => {
            let Value::Enum { fields, .. } = buf else {
                return None;
            };
            let [vec] = fields.as_slice() else {
                return None;
            };
            byte_list_value(vec)
        }
        _ => None,
    }
}

/// The bytes of a byte-vector value (the element-list
/// aggregate): `None` when any element is not a byte.
fn byte_list_value(value: &Value) -> Option<Vec<u8>> {
    let Value::Enum { fields, .. } = value else {
        return None;
    };
    let mut out = Vec::with_capacity(fields.len());
    for field in fields {
        match field {
            Value::Uint(b) => out.push(*b as u8),
            _ => return None,
        }
    }
    Some(out)
}

/// The bytes of a `PathBuf` value: its one field is the
/// buffer's `OsString`.
pub(super) fn path_buf_bytes(value: &Value) -> Option<Vec<u8>> {
    let Value::Enum { fields, .. } = value else {
        return None;
    };
    let [os_string] = fields.as_slice() else {
        return None;
    };
    os_string_bytes(os_string)
}

/// The sys-level `std::fs::ReadDir` value: the struct whose
/// fields are the owned-iterator state ([entry results, next
/// index]) the fs sys layer's snapshot builds. The public
/// `std::fs::read_dir` body wraps it in the public one-field
/// `ReadDir` aggregate itself.
pub(super) fn read_dir_value(items: Vec<Value>) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Tuple(items), Value::Uint(0)],
    }
}

/// A `std::fs::DirEntry` value: the public one-field wrapper
/// around the pair of the entry's full path (as a `PathBuf`)
/// and its file name (as an `OsString`) — the two facts the
/// entry's modelled methods read.
pub(super) fn dir_entry_value(entry: &std::fs::DirEntry) -> InterpResult<Value> {
    let file_name = entry.file_name();
    let name_bytes =
        crate::platform::os_str_bytes(&file_name).ok_or_else(|| {
            unsupported(
                "DirEntry::file_name: the host file name has no byte form on this platform",
            )
        })?;
    Ok(Value::Enum {
        variant: 0,
        fields: vec![Value::Tuple(vec![
            pathbuf_value(&host_path_bytes(&entry.path())?),
            os_string_value(&name_bytes),
        ])],
    })
}

/// The (full path, file name) values of a modelled `DirEntry`.
pub(super) fn dir_entry_parts(value: &Value) -> Option<(Value, Value)> {
    let Value::Enum { fields, .. } = value else {
        return None;
    };
    let [inner] = fields.as_slice() else {
        return None;
    };
    let Value::Tuple(pair) = inner else {
        return None;
    };
    let [path, name] = pair.as_slice() else {
        return None;
    };
    Some((path.clone(), name.clone()))
}

/// A host path from raw path bytes (guest paths are byte
/// strings, not necessarily UTF-8). On a host whose
/// strings are not byte strings, bytes that are not valid
/// UTF-8 denote no path the host can express: a named
/// unsupported error, never a substituted path.
pub(super) fn host_path(bytes: &[u8]) -> InterpResult<std::path::PathBuf> {
    crate::platform::path_from_bytes(bytes).ok_or_else(|| {
        unsupported(
            "host path: the path bytes have no host string form on this platform",
        )
    })
}

/// The bytes of a host path: on a host whose strings are
/// not byte strings, a path that is not valid UTF-8 has
/// no byte form (see [`host_path`]).
pub(super) fn host_path_bytes(path: &std::path::Path) -> InterpResult<Vec<u8>> {
    crate::platform::path_bytes(path).ok_or_else(|| {
        unsupported("host path: the host path has no byte form on this platform")
    })
}

/// One `Iterator::next` / `DoubleEndedIterator::next_back`
/// step of a `std::path::Components` state (see
/// `Builtin::PathComponents`): rebuild the host iterator
/// over the state's byte string, replay the components
/// already consumed from each end, and take one more, so
/// the parsing rules (root, `.` and `..` handling,
/// repeated separators) are the host's own. The counts in
/// the returned state advance on the side the step came
/// from; an exhausted iterator returns its state
/// unchanged, so it stays exhausted.
pub(super) fn components_step_state(
    state: &Value,
    from_back: bool,
) -> InterpResult<(Value, Option<Value>)> {
    let fields = iterator_fields(state, "std::path::Components")?;
    if fields.len() != 3 {
        return Err(InterpError::type_mismatch(
            "Components state",
            "[path bytes, front count, back count]",
            format!("{} field(s)", fields.len()),
        ));
    }
    let bytes = match &fields[0] {
        Value::Bytes(b) => b.clone(),
        Value::Str(s) => s.clone().into_bytes(),
        other => {
            return Err(InterpError::type_mismatch(
                "Components state",
                "path byte string",
                other.kind_name(),
            ))
        }
    };
    let (front, back) = match (&fields[1], &fields[2]) {
        (Value::Uint(f), Value::Uint(b)) => (*f, *b),
        _ => {
            return Err(InterpError::type_mismatch(
                "Components state",
                "front and back counts",
                "non-integer count(s)",
            ))
        }
    };
    let path = std::path::Path::new(
        crate::platform::os_str_from_bytes(&bytes).ok_or_else(|| {
            unsupported(
                "Components: the path bytes have no host string form on this platform",
            )
        })?,
    );
    let mut iter = path.components();
    for _ in 0..front {
        let _ = iter.next();
    }
    for _ in 0..back {
        let _ = iter.next_back();
    }
    let item = if from_back { iter.next_back() } else { iter.next() };
    match item {
        Some(component) => {
            let Some(value) = component_value(component) else {
                return Err(unsupported(
                    "Components: a host path component has no byte form on this platform",
                ));
            };
            let (front, back) = if from_back {
                (front, back + 1)
            } else {
                (front + 1, back)
            };
            Ok((components_state(&bytes, front, back), Some(value)))
        }
        None => Ok((state.clone(), None)),
    }
}

/// The `std::path::Components` iterator state in this
/// model: the path's byte string and the numbers of
/// components consumed from the front and from the back.
/// The parse itself is replayed from this state with the
/// host's `std::path` on every step (see `components_step`),
/// so no parse state has to be represented here.
pub(super) fn components_state(bytes: &[u8], front: u128, back: u128) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::Bytes(bytes.to_vec()),
            Value::Uint(front),
            Value::Uint(back),
        ],
    }
}

/// A `std::path::Component` value in std's shape: the
/// variants in declaration order (`Prefix`, `RootDir`,
/// `CurDir`, `ParentDir`, `Normal`); the `Normal` payload is
/// the component's byte string (the model's `&OsStr`, the
/// same form `Builtin::PathPart` returns for `&Path` /
/// `&OsStr` results).
fn component_value(component: std::path::Component) -> Option<Value> {
    match component {
        std::path::Component::Prefix(prefix) => Some(Value::Enum {
            variant: 0,
            fields: vec![prefix_component_value(prefix)?],
        }),
        std::path::Component::RootDir => Some(Value::Enum {
            variant: 1,
            fields: Vec::new(),
        }),
        std::path::Component::CurDir => Some(Value::Enum {
            variant: 2,
            fields: Vec::new(),
        }),
        std::path::Component::ParentDir => Some(Value::Enum {
            variant: 3,
            fields: Vec::new(),
        }),
        std::path::Component::Normal(part) => Some(Value::Enum {
            variant: 4,
            fields: vec![Value::Bytes(
                crate::platform::os_str_bytes(part)?.into_owned(),
            )],
        }),
    }
}

/// A `std::path::PrefixComponent` value in std's shape,
/// `{ kind, raw }`: the `Prefix` enum in declaration order
/// (`Verbatim`, `VerbatimUNC`, `VerbatimDisk`, `DeviceNS`,
/// `UNC`, `Disk`) and the raw prefix string. Prefixes do
/// not occur on Unix; the shape is built for completeness.
fn prefix_component_value(prefix: std::path::PrefixComponent) -> Option<Value> {
    let os = |s: &std::ffi::OsStr| {
        crate::platform::os_str_bytes(s).map(|b| Value::Bytes(b.into_owned()))
    };
    let kind = match prefix.kind() {
        std::path::Prefix::Verbatim(s) => Value::Enum {
            variant: 0,
            fields: vec![os(s)?],
        },
        std::path::Prefix::VerbatimUNC(a, b) => Value::Enum {
            variant: 1,
            fields: vec![os(a)?, os(b)?],
        },
        std::path::Prefix::VerbatimDisk(d) => Value::Enum {
            variant: 2,
            fields: vec![Value::Uint(d as u128)],
        },
        std::path::Prefix::DeviceNS(s) => Value::Enum {
            variant: 3,
            fields: vec![os(s)?],
        },
        std::path::Prefix::UNC(a, b) => Value::Enum {
            variant: 4,
            fields: vec![os(a)?, os(b)?],
        },
        std::path::Prefix::Disk(d) => Value::Enum {
            variant: 5,
            fields: vec![Value::Uint(d as u128)],
        },
    };
    Some(Value::Enum {
        variant: 0,
        fields: vec![kind, os(prefix.as_os_str())?],
    })
}

/// A `PathBuf` value in std's shape: `PathBuf { inner:
/// OsString { inner: Buf { inner: Vec<u8> } } }`, the Vec being
/// the element-list value.
pub(super) fn pathbuf_value(bytes: &[u8]) -> Value {
    let vec = Value::Enum {
        variant: 0,
        fields: bytes.iter().map(|b| Value::Uint(*b as u128)).collect(),
    };
    let buf = Value::Enum {
        variant: 0,
        fields: vec![vec],
    };
    let os_string = Value::Enum {
        variant: 0,
        fields: vec![buf],
    };
    Value::Enum {
        variant: 0,
        fields: vec![os_string],
    }
}

/// Registration fragment of the builtin name registry: the path component surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn path_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::path::Path::parent" => Builtin::PathPart(PathPart::Parent),
        "std::path::Path::file_name" => Builtin::PathPart(PathPart::FileName),
        "std::path::Path::file_stem" => Builtin::PathPart(PathPart::FileStem),
        "std::path::Path::extension" => Builtin::PathPart(PathPart::Extension),
        "std::path::Path::components" => Builtin::PathComponents,
        "std::path::Path::to_path_buf" => Builtin::PathToPathBuf,
        "std::path::PathBuf::_push" => Builtin::PathBufPush,
        "std::path::PathBuf::pop" => Builtin::PathBufPop,
        "std::path::PathBuf::_set_extension" => Builtin::PathBufSetExtension,
        "std::path::Path::_with_extension" => Builtin::PathWithExtension,
        "std::path::PathBuf::new" => Builtin::PathBufNew,
        "std::path::PathBuf::with_capacity" => Builtin::PathBufNew,
        _ => return None,
    })
}
