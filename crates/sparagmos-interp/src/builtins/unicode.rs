//! Core's generated unicode property tables: the runtime
//! model of the `core::unicode::unicode_data` skiplist
//! lookups.
//!
//! Each property module's `lookup` is `#[inline]`, so its
//! body converts, but the `lookup_slow` it defers to is
//! `#[inline(never)]` and non-generic: the shipped core's
//! rmeta carries no MIR body for it, so no conversion can
//! serve the call. The model replays core's own
//! `skip_search` over the module's own table statics
//! (`SHORT_OFFSET_RUNS` and `OFFSETS`), whose initializer
//! seeds the driver records in the program precisely
//! because no converted reference ever names them (see
//! `mir::StaticSeedDef`). The answers are therefore core's
//! own data and core's own algorithm, at the guest's
//! toolchain's Unicode version — nothing is retyped or
//! approximated here.

use crate::error::{InterpError, InterpResult};
use crate::interp::Interpreter;
use crate::value::Value;

use super::*;

impl<'a> Interpreter<'a> {
    /// `core::unicode::unicode_data::<module>::lookup_slow(c)`:
    /// whether the char `c` is a member of the property, by
    /// core's skiplist search over the module's tables.
    pub(super) fn exec_unicode_lookup_slow(
        &mut self,
        property: UnicodeProperty,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let needle = match vals.first() {
            Some(Value::Uint(v)) => *v as u32,
            Some(other) => {
                return Err(InterpError::type_mismatch(
                    format!("{}::lookup_slow", property.module_path()),
                    "a char argument",
                    other.kind_name(),
                ))
            }
            None => {
                return Err(InterpError::type_mismatch(
                    format!("{}::lookup_slow", property.module_path()),
                    "a char argument",
                    "no argument",
                ))
            }
        };
        let module = property.module_path();
        let runs = self.unicode_table_headers(&format!("{module}::SHORT_OFFSET_RUNS"))?;
        let offsets = self.unicode_table_offsets(&format!("{module}::OFFSETS"))?;
        Ok(Value::Bool(skip_search(needle, &runs, &offsets)))
    }

    /// The packed header words of a property module's
    /// `SHORT_OFFSET_RUNS` static: the static's seeded value,
    /// read as the array of one-field header values it is.
    fn unicode_table_headers(&self, name: &str) -> InterpResult<Vec<u32>> {
        let value = self.unicode_table_static(name)?;
        let mut words = Vec::new();
        for element in table_elements(&value, name)? {
            words.push(header_word(element, name)?);
        }
        Ok(words)
    }

    /// The bytes of a property module's `OFFSETS` static.
    fn unicode_table_offsets(&self, name: &str) -> InterpResult<Vec<u8>> {
        let value = self.unicode_table_static(name)?;
        if let Value::Bytes(bytes) = &value {
            return Ok(bytes.clone());
        }
        let mut out = Vec::new();
        for element in table_elements(&value, name)? {
            out.push(header_word(element, name)? as u8);
        }
        Ok(out)
    }

    /// The seeded value of one of the unicode table statics,
    /// seeding its storage from the program's seed table on
    /// first touch, like any other static's.
    fn unicode_table_static(&self, name: &str) -> InterpResult<Value> {
        self.ensure_static_seeded(name)?;
        self.static_base(name)
    }
}

/// The elements of a table static's array value, in order.
/// An array constant materialises as a tuple of its elements
/// (or, for a byte array, as raw bytes — handled by the
/// caller); anything else is not a table shape.
fn table_elements<'v>(value: &'v Value, name: &str) -> InterpResult<&'v [Value]> {
    match value {
        Value::Tuple(elements) => Ok(elements),
        other => Err(InterpError::type_mismatch(
            format!("unicode table static `{name}`"),
            "an array value",
            other.kind_name(),
        )),
    }
}

/// The word a table element holds: the integer itself, or
/// the single field of the one-field aggregate the value
/// model wraps it in (a `ShortOffsetRunHeader` is a
/// one-field tuple struct over its packed `u32`).
fn header_word(value: &Value, name: &str) -> InterpResult<u32> {
    match value {
        Value::Uint(v) => Ok(*v as u32),
        Value::Enum { fields, .. } if fields.len() == 1 => header_word(&fields[0], name),
        Value::Tuple(fields) if fields.len() == 1 => header_word(&fields[0], name),
        other => Err(InterpError::type_mismatch(
            format!("unicode table static `{name}`"),
            "an integer table element",
            other.kind_name(),
        )),
    }
}

/// The index a header packs in its high bits (the offsets
/// index at which its run starts).
fn header_start_index(header: u32) -> usize {
    (header >> 21) as usize
}

/// The code-point prefix sum a header packs in its low bits.
fn header_prefix_sum(header: u32) -> u32 {
    header & ((1 << 21) - 1)
}

/// Core's `skip_search` (library/core/src/unicode/
/// unicode_data.rs), transcribed: a binary search over the
/// run headers for the run that could contain `needle`,
/// then a walk of that run's gap offsets; the needle is a
/// member exactly when the walk stops at an odd offset
/// index (the runs alternate gap/member spans). The final
/// header is a sentinel past `char::MAX`, so the binary
/// search always lands inside the table.
fn skip_search(needle: u32, short_offset_runs: &[u32], offsets: &[u8]) -> bool {
    let last_idx =
        match short_offset_runs.binary_search_by_key(&(needle << 11), |header| header << 11) {
            Ok(idx) => idx + 1,
            Err(idx) => idx,
        };
    let mut offset_idx = header_start_index(short_offset_runs[last_idx]);
    let length = if let Some(next) = short_offset_runs.get(last_idx + 1) {
        header_start_index(*next) - offset_idx
    } else {
        offsets.len() - offset_idx
    };
    let prev = last_idx
        .checked_sub(1)
        .map(|prev| header_prefix_sum(short_offset_runs[prev]))
        .unwrap_or(0);
    let total = needle - prev;
    let mut prefix_sum = 0u32;
    for _ in 0..length.saturating_sub(1) {
        let offset = offsets[offset_idx];
        prefix_sum += offset as u32;
        if prefix_sum > total {
            break;
        }
        offset_idx += 1;
    }
    offset_idx % 2 == 1
}
