//! BTreeMap/BTreeSet storage walks shared by the collection builtins.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    Frame, Interpreter, Resolved,
};
use crate::value::{RefProj, RefTarget, Value};

use super::*;

impl<'a> Interpreter<'a> {
    /// The elements a slice-like value denotes, with its length: a
    /// fat reference contributes its base's elements truncated to the
    /// carried length; any other value contributes its own elements.
    /// The elements a borrowed `BTreeSet` iteration yields, in
    /// iteration order, read from the set value behind a reference.
    pub(super) fn btree_set_elements_of_ref(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Vec<Value>> {
        let target = self.deref_value(frames, value)?;
        let set = self.read_resolved(frames, &target)?;
        self.btree_set_elements(frames, &set)
    }

    /// The elements a borrowed `BTreeMap` iteration yields — one
    /// `(key, value)` tuple per entry — read from the map value
    /// behind a reference.
    pub(super) fn btree_map_pair_elements_of_ref(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Vec<Value>> {
        let pairs = self.btree_map_pairs_of_ref(frames, value)?;
        Ok(pairs
            .into_iter()
            .map(|(key, value)| Value::Tuple(vec![key, value]))
            .collect())
    }

    /// The `(key, value)` pairs of the `BTreeMap` behind a
    /// reference, in iteration order.
    pub(super) fn btree_map_pairs_of_ref(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Vec<(Value, Value)>> {
        let target = self.deref_value(frames, value)?;
        let map = self.read_resolved(frames, &target)?;
        self.btree_map_pairs(frames, &map)
    }

    /// The elements of an owned `BTreeSet` value: its keys, in
    /// iteration order (the set's values are the `SetValZST`
    /// marker and carry no data).
    pub(super) fn btree_set_elements(
        &self,
        frames: &[Frame],
        set: &Value,
    ) -> InterpResult<Vec<Value>> {
        let fields = match set {
            Value::Enum { fields, .. } if fields.len() == 1 => fields,
            other => {
                return Err(InterpError::type_mismatch(
                    "BTreeSet value",
                    "a one-field aggregate holding its BTreeMap",
                    other.kind_name(),
                ))
            }
        };
        Ok(self
            .btree_map_pairs(frames, &fields[0])?
            .into_iter()
            .map(|(key, _)| key)
            .collect())
    }

    /// The `(key, value)` tuple elements of an owned `BTreeMap`
    /// value, in iteration order.
    pub(super) fn btree_map_pair_elements(
        &self,
        frames: &[Frame],
        map: &Value,
    ) -> InterpResult<Vec<Value>> {
        Ok(self
            .btree_map_pairs(frames, map)?
            .into_iter()
            .map(|(key, value)| Value::Tuple(vec![key, value]))
            .collect())
    }

    /// The `(key, value)` pairs of a `BTreeMap` value in iteration
    /// order, walked from the tree's node storage. The walk is
    /// strict: every shape it relies on — the map's
    /// `[root, length, ..]` fields, the `NodeRef` layout, the leaf
    /// and internal node layouts, the declared length — is
    /// checked, and anything unexpected is an explicit unsupported
    /// error rather than a guessed element. A node pointer that
    /// does not name an allocation is such an error: the elements
    /// behind it are unreachable in this model, never fabricated.
    pub(super) fn btree_map_pairs(
        &self,
        frames: &[Frame],
        map: &Value,
    ) -> InterpResult<Vec<(Value, Value)>> {
        let fields = match map {
            Value::Enum { fields, .. } if fields.len() >= 2 => fields,
            other => {
                return Err(InterpError::type_mismatch(
                    "BTreeMap value",
                    "an aggregate of [root, length, ..] fields",
                    other.kind_name(),
                ))
            }
        };
        let declared = btree_count(&fields[1], "BTreeMap length")?;
        let root_fields: &[Value] = match &fields[0] {
            Value::Enum { fields, .. } => fields,
            _ => &[],
        };
        let mut out = Vec::new();
        match root_fields.len() {
            // `Option::None`: no payload fields.
            0 => {}
            // `Option::Some(root)`: the NodeRef is the payload.
            1 => self.btree_walk_noderef(frames, &root_fields[0], &mut out, 0)?,
            // A niche-packed `Option<Root>` reads as the NodeRef
            // itself; a null node pointer there is the empty tree.
            _ => {
                let (_, ptr) = btree_noderef_parts(&fields[0])?;
                if ptr.is_some() {
                    self.btree_walk_noderef(frames, &fields[0], &mut out, 0)?;
                }
            }
        }
        if out.len() != declared {
            return Err(unsupported(format!(
                "BTreeMap node walk found {} entries but the map declares {declared}",
                out.len()
            )));
        }
        Ok(out)
    }

    /// Append the pairs under one `NodeRef` (the root reference or
    /// an internal node's edge) to `out`, in iteration order.
    fn btree_walk_noderef(
        &self,
        frames: &[Frame],
        noderef: &Value,
        out: &mut Vec<(Value, Value)>,
        depth: usize,
    ) -> InterpResult<()> {
        if depth > 64 {
            return Err(unsupported(
                "BTreeMap node walk exceeded 64 levels".to_string(),
            ));
        }
        let (height, ptr) = btree_noderef_parts(noderef)?;
        let ptr = btree_internal_target(ptr, height > 0);
        let node = self.btree_node_value(frames, ptr, "BTree node reference")?;
        self.btree_walk_node(frames, &node, height, out, depth)
    }

    /// The node value a tree pointer names. Only an allocation
    /// pointer reaches storage; anything else (an address-only or
    /// null pointer) leaves the elements unreachable, which is an
    /// explicit unsupported error.
    fn btree_node_value(
        &self,
        frames: &[Frame],
        ptr: Option<RefTarget>,
        what: &str,
    ) -> InterpResult<Value> {
        match ptr {
            Some(RefTarget::Alloc { id, projection }) => {
                self.read_resolved(frames, &Resolved::Alloc { id, projection })
            }
            _ => Err(unsupported(format!(
                "{what}: the node pointer does not name an allocation in this model, so the tree's elements are unreachable"
            ))),
        }
    }

    /// Append the pairs stored in one node value to `out`. A leaf
    /// (height 0) holds its pairs directly; an internal node
    /// interleaves its edge subtrees with its own pairs.
    fn btree_walk_node(
        &self,
        frames: &[Frame],
        node: &Value,
        height: usize,
        out: &mut Vec<(Value, Value)>,
        depth: usize,
    ) -> InterpResult<()> {
        let fields = match node {
            Value::Enum { fields, .. } => fields,
            other => {
                return Err(InterpError::type_mismatch(
                    "BTree node value",
                    "a node aggregate",
                    other.kind_name(),
                ))
            }
        };
        if height == 0 {
            if fields.len() != 5 {
                return Err(InterpError::type_mismatch(
                    "BTree leaf node value",
                    "[parent, parent index, length, keys, values]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let (len, keys, values) = btree_leaf_parts(fields)?;
            for i in 0..len {
                out.push((btree_maybe_uninit(&keys[i])?, btree_maybe_uninit(&values[i])?));
            }
            return Ok(());
        }
        if fields.len() != 2 {
            return Err(InterpError::type_mismatch(
                "BTree internal node value",
                "[leaf data, edges]",
                format!("{} field(s)", fields.len()),
            ));
        }
        let data_fields = match &fields[0] {
            Value::Enum { fields, .. } => fields,
            other => {
                return Err(InterpError::type_mismatch(
                    "BTree internal node data",
                    "a leaf node aggregate",
                    other.kind_name(),
                ))
            }
        };
        if data_fields.len() != 5 {
            return Err(InterpError::type_mismatch(
                "BTree internal node data",
                "[parent, parent index, length, keys, values]",
                format!("{} field(s)", data_fields.len()),
            ));
        }
        let (len, keys, values) = btree_leaf_parts(data_fields)?;
        let edges = match &fields[1] {
            Value::Tuple(items) => items,
            other => {
                return Err(InterpError::type_mismatch(
                    "BTree internal node edges",
                    "an array of child node pointers",
                    other.kind_name(),
                ))
            }
        };
        if edges.len() < len + 1 {
            return Err(InterpError::type_mismatch(
                "BTree internal node edges",
                "at least length + 1 edges",
                format!("{} edge(s) for length {len}", edges.len()),
            ));
        }
        for i in 0..=len {
            let edge = btree_maybe_uninit(&edges[i])?;
            let ptr = btree_ptr_of(&edge)?;
            let ptr = btree_internal_target(ptr, height > 1);
            let child = self.btree_node_value(frames, ptr, "BTree internal edge")?;
            self.btree_walk_node(frames, &child, height - 1, out, depth + 1)?;
            if i < len {
                out.push((btree_maybe_uninit(&keys[i])?, btree_maybe_uninit(&values[i])?));
            }
        }
        Ok(())
    }
}

/// The `slice::Iter` value over an element snapshot, in the
/// struct's real field layout: `ptr` and `end_or_len` are element
/// pointers into the snapshot — value targets carrying an
/// `Index` projection — and the third field is the
/// `PhantomData` marker. The layout matters beyond the builtins:
/// core's converted bodies for the iterator's other methods
/// (`fold`, `count`, `nth`, `size_hint`, ...) project these very
/// fields, so a fabricated (snapshot, cursor) pair in their place
/// reads back as a bogus pointer pair and the run dies on the
/// first field copy.
/// A non-negative integer field of a B-tree value (a length, a
/// node height), unwrapping one-field aggregates.
fn btree_count(value: &Value, what: &str) -> InterpResult<usize> {
    match value {
        Value::Uint(v) => usize::try_from(*v).map_err(|_| {
            InterpError::type_mismatch(what, "a count that fits in usize", format!("{v}"))
        }),
        Value::Int(v) if *v >= 0 => usize::try_from(*v).map_err(|_| {
            InterpError::type_mismatch(what, "a count that fits in usize", format!("{v}"))
        }),
        Value::Enum { fields, .. } if fields.len() == 1 => btree_count(&fields[0], what),
        other => Err(InterpError::type_mismatch(
            what,
            "an unsigned integer",
            other.kind_name(),
        )),
    }
}

/// The node value target for a node known to be internal
/// (`height > 0` at the reference that names it). A pointer to
/// an internal node denotes the node's leaf data -- the first
/// field of the internal node value, which is what the
/// pointer's cast into the leaf type names -- while the walk
/// needs the internal node value itself, edges included: step
/// one field-0 step up to the enclosing value. A pointer that
/// does not end in that step, and every leaf node's pointer,
/// names the node value directly.
fn btree_internal_target(
    ptr: Option<RefTarget>,
    internal: bool,
) -> Option<RefTarget> {
    if !internal {
        return ptr;
    }
    match ptr {
        Some(RefTarget::Alloc { id, mut projection })
            if matches!(projection.last(), Some(RefProj::Field(0))) =>
        {
            projection.pop();
            Some(RefTarget::Alloc { id, projection })
        }
        other => other,
    }
}

/// The `(height, node pointer)` of a `NodeRef` value:
/// `[height, node, marker]`. The pointer is `None` when the node
/// slot holds the null pointer.
fn btree_noderef_parts(noderef: &Value) -> InterpResult<(usize, Option<RefTarget>)> {
    let fields = match noderef {
        Value::Enum { fields, .. } if fields.len() == 3 => fields,
        other => {
            return Err(InterpError::type_mismatch(
                "BTree NodeRef value",
                "[height, node, marker]",
                other.kind_name(),
            ))
        }
    };
    let height = btree_count(&fields[0], "NodeRef height")?;
    let ptr = btree_ptr_of(&fields[1])?;
    Ok((height, ptr))
}

/// The pointer inside a `NonNull`-shaped value, unwrapping
/// one-field aggregates. `None` for the null pointer.
fn btree_ptr_of(value: &Value) -> InterpResult<Option<RefTarget>> {
    match value {
        Value::Ref(RefTarget::Null) => Ok(None),
        Value::Ref(target) => Ok(Some(target.clone())),
        Value::Enum { fields, .. } if fields.len() == 1 => btree_ptr_of(&fields[0]),
        other => Err(InterpError::type_mismatch(
            "BTree node pointer",
            "a NonNull-shaped aggregate holding a pointer",
            other.kind_name(),
        )),
    }
}

/// The `(length, keys, values)` of a leaf node's fields:
/// `[parent, parent index, length, keys, values]`, where the keys
/// and values are the node's fixed-capacity arrays.
fn btree_leaf_parts(fields: &[Value]) -> InterpResult<(usize, Vec<Value>, Vec<Value>)> {
    let len = btree_count(&fields[2], "BTree leaf length")?;
    let keys = match &fields[3] {
        Value::Tuple(items) => items.clone(),
        other => {
            return Err(InterpError::type_mismatch(
                "BTree leaf keys",
                "an array of key slots",
                other.kind_name(),
            ))
        }
    };
    let values = match &fields[4] {
        Value::Tuple(items) => items.clone(),
        other => {
            return Err(InterpError::type_mismatch(
                "BTree leaf values",
                "an array of value slots",
                other.kind_name(),
            ))
        }
    };
    let keys = btree_fill_elided(keys, len)?;
    let values = btree_fill_elided(values, len)?;
    Ok((len, keys, values))
}

/// Whether a stored slot's value is zero-sized in form: a unit,
/// an empty aggregate, or a wrapper over one. A zero-sized
/// element type has exactly one value, and every stored slot
/// of it has this shape.
fn btree_zst_shaped(value: &Value) -> bool {
    match value {
        Value::Unit => true,
        Value::Tuple(items) => items.iter().all(btree_zst_shaped),
        Value::Enum { fields, .. } => fields.iter().all(btree_zst_shaped),
        _ => false,
    }
}

/// Fill a B-tree node's key/value array out to the declared
/// length. A slot within the length that was never stored is
/// an elided zero-sized element: the write machinery stores
/// nothing for a value with no leaves, so a zero-sized element
/// type (a set's marker value, a unit key) leaves the array
/// shorter than the length, and every present slot holds the
/// type's sole value in the same form the missing slots hold.
/// A short array whose present slots are not all zero-sized
/// is a genuine hole in the node and stays an error.
fn btree_fill_elided(mut slots: Vec<Value>, len: usize) -> InterpResult<Vec<Value>> {
    if slots.len() >= len {
        return Ok(slots);
    }
    let sole = slots.first().cloned();
    match sole {
        Some(first) if slots.iter().all(btree_zst_shaped) => {
            slots.resize(len, first);
            Ok(slots)
        }
        _ => Err(InterpError::type_mismatch(
            "BTree leaf node",
            "key and value arrays covering the declared length",
            format!(
                "length {len}, {} slot(s) present",
                slots.len()
            ),
        )),
    }
}

/// The payload of a `MaybeUninit` slot in a B-tree node: the
/// value model stores it in the field-0 payload form; a slot
/// within the declared length that is still uninitialized is an
/// explicit unsupported error.
fn btree_maybe_uninit(value: &Value) -> InterpResult<Value> {
    match value {
        Value::Uninit => Err(unsupported(
            "a BTree node slot within the declared length is uninitialized".to_string(),
        )),
        Value::Enum {
            variant: 0,
            fields,
        } if fields.len() == 1 => Ok(fields[0].clone()),
        other => Ok(other.clone()),
    }
}

/// The item of a `btree_map::Iter` step: the snapshot element
/// is the stored (key, value) pair, and the iterator's item is
/// the pair of references to the pair's two fields, not a
/// reference to the pair itself. Every other iterator sharing
/// the slice-state step (a plain slice iterator over elements,
/// the key-only and value-only btree iterators, whose snapshots
/// hold the yielded elements directly) hands out the element
/// reference as it stands.
pub(super) fn btree_pair_item(self_ty: &str, item: Value) -> Value {
    if !self_ty.starts_with("std::collections::btree_map::Iter<") {
        return item;
    }
    let with_field = |target: &RefTarget, idx: u32| -> Value {
        let mut target = target.clone();
        match &mut target {
            RefTarget::Place { projection, .. }
            | RefTarget::Static { projection, .. }
            | RefTarget::Alloc { projection, .. }
            | RefTarget::Value(_, projection) => {
                projection.push(RefProj::Field(idx as usize));
            }
            RefTarget::Address(_) | RefTarget::Null => {}
        }
        Value::Ref(target)
    };
    match &item {
        Value::Ref(target) => {
            Value::Tuple(vec![with_field(target, 0), with_field(target, 1)])
        }
        _ => item,
    }
}
