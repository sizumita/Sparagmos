use std::collections::HashMap;

use crate::program::{StaticBlob};

// ---------------------------------------------------------------------------
// Message table
// ---------------------------------------------------------------------------

#[derive(Default)]
pub(super) struct MsgTable {
    msgs: Vec<String>,
    map: HashMap<String, u32>,
}

impl MsgTable {
    pub(super) fn intern(&mut self, s: &str) -> u32 {
        if let Some(i) = self.map.get(s) {
            return *i;
        }
        let i = self.msgs.len() as u32;
        self.msgs.push(s.to_string());
        self.map.insert(s.to_string(), i);
        i
    }
    pub(super) fn into_vec(self) -> Vec<String> {
        self.msgs
    }
}

// ---------------------------------------------------------------------------
// Statics table
// ---------------------------------------------------------------------------

/// The program's static blobs: the pointees of pointer-shaped
/// constants, interned by content like the message tables. A blob
/// is either raw bytes (a string / byte-string constant's contents)
/// or a constant value's leaf words (a reference constant's
/// pointee), with address words left symbolic
/// ([`StaticWord::AddrOf`]) for the program layer to resolve once
/// every blob's address is known.
#[derive(Default)]
pub(super) struct StaticsTable {
    blobs: Vec<StaticBlob>,
    map: HashMap<StaticBlob, u32>,
}

impl StaticsTable {
    pub(super) fn intern(&mut self, b: StaticBlob) -> u32 {
        if let Some(i) = self.map.get(&b) {
            return *i;
        }
        let i = self.blobs.len() as u32;
        self.map.insert(b.clone(), i);
        self.blobs.push(b);
        i
    }
    pub(super) fn into_vec(self) -> Vec<StaticBlob> {
        self.blobs
    }
}
