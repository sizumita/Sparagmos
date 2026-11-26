//! HashMap/HashSet storage walks shared by the collection builtins.



use crate::interp::Interpreter;
use crate::value::{RefProj, RefTarget, Value};

impl<'a> Interpreter<'a> {
    /// The `(key, value)` entries of a hash collection value (a
    /// `HashMap`, or a `HashSet` through its inner map), in bucket
    /// order. The walk reads the raw table the way hashbrown lays
    /// it out in this model: one allocation holds the buckets and
    /// the control bytes; the control pointer's projection names
    /// the slot where the control bytes begin, and the buckets sit
    /// in the slots before it, addressed from the control pointer
    /// the way the table's own pointer arithmetic addresses them:
    /// bucket `i`'s pair ends one bucket width further from the
    /// control bytes than bucket `i + 1`'s. The region is padded
    /// before its first bucket up to the control region's
    /// alignment, so the control offset only bounds the bucket
    /// width: the width is the region's even split when there is
    /// no padding, and otherwise one of the narrower widths whose
    /// padded region ends at the same offset. A candidate width is
    /// accepted only when every pair-shaped slot in the region
    /// sits at one of its bucket starts, so a deleted bucket's
    /// stale pair cannot be read as a live entry at a wider
    /// candidate's offset. A bucket is live when its slot holds a
    /// pair and its control byte is a hash byte (below the
    /// empty/deleted sentinels). `None` when the value does not
    /// read as a hash collection in this model, so the caller
    /// keeps its existing behaviour. (The debugger's pretty
    /// printers walk the same layout in `debug::printers`.)
    pub(super) fn hash_map_entries(
        &self,
        value: &Value,
        is_set: bool,
    ) -> Option<Vec<(Value, Value)>> {
        let base = hash_agg_fields(value)?.first()?;
        let map = if is_set {
            hash_agg_fields(base)?.first()?
        } else {
            base
        };
        let table = hash_agg_fields(map)?.get(1)?;
        let inner = hash_agg_fields(table)?.first()?;
        let inner_fields = hash_agg_fields(inner)?;
        if inner_fields.len() != 4 {
            return None;
        }
        let mask = hash_agg_uint(inner_fields.first()?)?;
        let size = hash_agg_uint(inner_fields.get(3)?)?;
        if size == 0 {
            return Some(Vec::new());
        }
        let (id, projection) = hash_wrapped_alloc_ref(inner_fields.get(1)?)?;
        let ctrl_index = projection.iter().rev().find_map(|step| match step {
            RefProj::Index(i) => Some(*i as usize),
            _ => None,
        })?;
        let num_buckets = usize::try_from(mask + 1).ok()?;
        if num_buckets == 0 {
            return None;
        }
        let stored = self.alloc_base(id).ok()?;
        let slots = match &stored {
            Value::Tuple(items) => items,
            _ => return None,
        };
        // One walk at a candidate bucket width: the entries in
        // bucket order, when every live control byte's bucket
        // holds a pair and the live count is the item count.
        let walk = |width: usize| -> Option<Vec<(Value, Value)>> {
            let mut entries = Vec::new();
            for i in 0..num_buckets {
                let ctrl = slots.get(ctrl_index + i)?;
                let ctrl_byte = match hash_unwrap_element(ctrl) {
                    Value::Uint(b) => b,
                    _ => return None,
                };
                if ctrl_byte >= 0x80 {
                    continue;
                }
                let pos = ctrl_index.checked_sub((i + 1) * width)?;
                let pair = match slots.get(pos)? {
                    Value::Tuple(items) if items.len() == 2 => items,
                    _ => return None,
                };
                entries.push((pair[0].clone(), pair[1].clone()));
            }
            if entries.len() as u128 != size {
                return None;
            }
            // The width must also account for every pair-shaped
            // slot in the region: a pair this partition does not
            // place at a bucket start belongs to another width's
            // buckets — a deleted bucket's stale pair sits at its
            // own bucket start, and a wider candidate reads it as
            // a nearer bucket's entry, passing the shape and
            // count checks above with a removed entry in place of
            // a live one. Only the true width places every pair
            // in the region at one of its bucket starts.
            let region_start = ctrl_index - num_buckets * width;
            for j in region_start..ctrl_index {
                let is_pair = matches!(
                    slots.get(j),
                    Some(Value::Tuple(items)) if items.len() == 2
                );
                if is_pair && (ctrl_index - j) % width != 0 {
                    return None;
                }
            }
            Some(entries)
        };
        // The even split first (exact for unpadded tables), then
        // the remaining widths whose region — padded before its
        // first bucket by less than the control alignment, at
        // most 64 slots — ends at the same control offset,
        // widest first.
        let mut widths: Vec<usize> = Vec::new();
        if ctrl_index % num_buckets == 0 && ctrl_index / num_buckets > 0 {
            widths.push(ctrl_index / num_buckets);
        }
        let widest = ctrl_index / num_buckets;
        let narrowest = if ctrl_index >= 64 {
            (ctrl_index - 64) / num_buckets + 1
        } else {
            1
        };
        for w in (narrowest..=widest).rev() {
            if !widths.contains(&w) {
                widths.push(w);
            }
        }
        for width in widths {
            if let Some(entries) = walk(width) {
                return Some(entries);
            }
        }
        None
    }
}

/// The fields of an aggregate value (an ADT value in this
/// model), when it is one.
fn hash_agg_fields(value: &Value) -> Option<&Vec<Value>> {
    match value {
        Value::Enum { fields, .. } => Some(fields),
        _ => None,
    }
}

/// The integer an aggregate field holds, when it is one.
fn hash_agg_uint(value: &Value) -> Option<u128> {
    match value {
        Value::Uint(n) => Some(*n),
        Value::Int(n) => Some(*n as u128),
        _ => None,
    }
}

/// The value under one single-field aggregate layer (the
/// `MaybeUninit` shell a collection element is stored in);
/// any other value stands as it is.
fn hash_unwrap_element(value: &Value) -> Value {
    match value {
        Value::Enum { fields, .. } if fields.len() == 1 => fields[0].clone(),
        _ => value.clone(),
    }
}

/// The allocation a pointer-shaped aggregate denotes: the
/// chain of single-field wrappers (`NonNull`, `Unique`, a
/// box shell) unwrapped down to the reference they hold.
/// Returns the allocation id and the reference's projection.
fn hash_wrapped_alloc_ref(value: &Value) -> Option<(u64, Vec<RefProj>)> {
    let mut cur = value;
    loop {
        match cur {
            Value::Ref(RefTarget::Alloc { id, projection }) => {
                return Some((*id, projection.clone()));
            }
            Value::Enum { fields, .. } if fields.len() == 1 => {
                cur = &fields[0];
            }
            _ => return None,
        }
    }
}
