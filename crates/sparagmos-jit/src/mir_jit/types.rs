use std::collections::HashMap;
use std::fmt;

use sparagmos_mir as mir;
use sparagmos_runtime::error::PanicRecord;

use crate::compiler::CompileError;
use crate::ir::{Function as IrFunction};
use crate::builtins::{BuiltinTables};
use crate::program::{compile_program_with_builtins, CompiledProgram};

use super::adt::AdtLayouts;
use super::lower::FnLower;
use super::tables::{MsgTable, StaticsTable};
use super::ty_helpers::flatten;

// ---------------------------------------------------------------------------
// Public input / output types
// ---------------------------------------------------------------------------

/// One function of a program to compile: a name (used for lookup and
/// diagnostics), the `DefId` other functions call it by (if any), and
/// its MIR body. This mirrors the shape the driver's conversion
/// produces (`MirProgram` in `sparagmos-interp`), kept local to this
/// crate so the JIT does not depend on the interpreter.
#[derive(Debug, Clone)]
pub struct MirFunctionInput {
    pub name: String,
    pub def_id: Option<mir::DefId>,
    pub body: mir::Body,
}

/// A whole program: the functions, the driver's definition-name
/// table (`DefId` -> path), used to name callees in diagnostics, and
/// the ADT discriminant table, which gives `Discriminant` reads and
/// `SetDiscriminant` writes the compiler's discriminant values.
#[derive(Debug, Clone, Default)]
pub struct MirProgramInput {
    pub functions: Vec<MirFunctionInput>,
    pub names: HashMap<mir::DefId, String>,
    pub adt_discriminants: mir::AdtDiscriminants,
    /// The program-wide source-file table the bodies'
    /// `source_lines` index into: a panic site's recorded
    /// location is resolved through it. Empty for inputs built
    /// without source information (their panic records then
    /// carry no location).
    pub source_files: Vec<String>,
}

/// Why a program did not compile on the JIT path.
#[derive(Debug)]
pub enum JitError {
    /// A construct outside the JIT's supported range. The string
    /// names the function and the construct.
    Unsupported(String),
    /// The backend rejected the lowered IR.
    Backend(CompileError),
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JitError::Unsupported(what) => write!(f, "unsupported: {what}"),
            JitError::Backend(e) => write!(f, "backend error: {e}"),
        }
    }
}
impl std::error::Error for JitError {}
impl From<CompileError> for JitError {
    fn from(e: CompileError) -> Self {
        JitError::Backend(e)
    }
}

/// Why a call into a compiled program did not produce a value.
#[derive(Debug)]
pub enum JitCallError {
    /// The guest panicked; the record carries the panic message.
    Panic(PanicRecord),
    /// No function with this name is part of the program.
    UnknownFunction(String),
    /// The argument count differs from the function's parameter count.
    WrongArgCount {
        name: String,
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for JitCallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JitCallError::Panic(p) => write!(f, "guest panic: {}", p.message),
            JitCallError::UnknownFunction(n) => write!(f, "unknown function `{n}`"),
            JitCallError::WrongArgCount { name, expected, got } => {
                write!(f, "function `{name}` takes {expected} arguments, got {got}")
            }
        }
    }
}
impl std::error::Error for JitCallError {}

/// A MIR program compiled to executable code.
pub struct JitProgram {
    inner: CompiledProgram,
    index_by_name: HashMap<String, usize>,
    /// Parameter word count (IR parameters) per function: one
    /// word per scalar leaf of each MIR argument.
    param_counts: Vec<usize>,
    /// Return leaf count per function. A count above two means
    /// the function takes the hidden sret argument word (see
    /// `FnLower::run`); the call API supplies it itself.
    ret_counts: Vec<usize>,
    names: Vec<String>,
}

impl fmt::Debug for JitProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JitProgram")
            .field("functions", &self.names)
            .finish_non_exhaustive()
    }
}

impl JitProgram {
    /// Lower and compile the whole program. Fails with
    /// [`JitError::Unsupported`] if any function uses a construct
    /// outside the supported range — nothing is executed partially.
    pub fn compile(input: &MirProgramInput) -> Result<JitProgram, JitError> {
        // The backend lowers x86_64 machine code and its entry
        // trampoline is x86_64 assembly, so JIT compilation is
        // refused outright on other hosts. The interpreter path is
        // unaffected: the driver still builds and runs there.
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = input;
            return Err(JitError::Unsupported(format!(
                "JIT compilation on {} (the JIT targets x86_64 only)",
                std::env::consts::ARCH
            )));
        }
        let mut def_index: HashMap<mir::DefId, usize> = HashMap::new();
        for (i, f) in input.functions.iter().enumerate() {
            if let Some(def) = f.def_id {
                def_index.insert(def, i);
            }
        }
        // Parameter types of every function, for typing call arguments.
        let param_tys: Vec<Vec<mir::Ty>> = input
            .functions
            .iter()
            .map(|f| {
                f.body.local_decls[1..=f.body.arg_count]
                    .iter()
                    .map(|d| d.ty.clone())
                    .collect()
            })
            .collect();
        let layouts = AdtLayouts::build(&input.functions, &input.adt_discriminants, &input.names);
        let mut messages = MsgTable::default();
        let mut locations = MsgTable::default();
        let mut places = MsgTable::default();
        let mut statics = StaticsTable::default();
        let mut tables = BuiltinTables::default();
        let mut ir_funcs: Vec<IrFunction> = Vec::with_capacity(input.functions.len());
        for f in &input.functions {
            let lower = FnLower::new(
                f,
                &def_index,
                &input.names,
                &param_tys,
                &input.adt_discriminants,
                &layouts,
                &mut messages,
                &mut locations,
                &input.source_files,
                &mut places,
                &mut statics,
                &mut tables,
            );
            ir_funcs.push(lower.run()?);
        }
        let param_counts: Vec<usize> = ir_funcs.iter().map(|f| f.param_count).collect();
        let mut ret_counts: Vec<usize> = Vec::with_capacity(input.functions.len());
        for f in &input.functions {
            ret_counts.push(flatten(
                &f.body.local_decls[0].ty,
                &f.name,
                &layouts,
            )?);
        }
        let inner = compile_program_with_builtins(
            &ir_funcs,
            messages.into_vec(),
            places.into_vec(),
            locations.into_vec(),
            statics.into_vec(),
            tables,
        )?;
        Ok(JitProgram {
            inner,
            index_by_name: input
                .functions
                .iter()
                .enumerate()
                .map(|(i, f)| (f.name.clone(), i))
                .collect(),
            param_counts,
            ret_counts,
            names: input.functions.iter().map(|f| f.name.clone()).collect(),
        })
    }

    /// Names of the program's functions, in input order.
    pub fn function_names(&self) -> Vec<&str> {
        self.names.iter().map(|s| s.as_str()).collect()
    }

    /// The guest output the builtin layer captured during the
    /// current (or last) call: (stdout, stderr). A program that
    /// never prints yields two empty strings.
    pub fn guest_output(&self) -> (String, String) {
        self.inner.guest_output()
    }

    /// Call the function called `name` with the function's
    /// argument words in layout order (see [`arg_word_layout`]):
    /// one word per scalar leaf of each MIR parameter, in order
    /// (unit parameters are passed as a dummy 0 word and ignored; a
    /// fat pointer parameter is two words: the address, then the
    /// length; a two-word parameter that would straddle the
    /// register/stack boundary is preceded by a pad 0 word, so the
    /// slice length is the function's parameter word count, pads
    /// included). Words past the sixth travel on the stack.
    /// Arguments are the parameters' canonical 64-bit representations
    /// (sign-extended for signed types). The result is the return
    /// value's canonical representation (0 for a unit return).
    /// A function returning more than two words stores them
    /// through a hidden sret argument this API supplies itself;
    /// `call` then yields the result's first word (and
    /// [`JitProgram::call_pair`] its first two).
    pub fn call(&self, name: &str, args: &[i64]) -> Result<i64, JitCallError> {
        let idx = self
            .index_by_name
            .get(name)
            .copied()
            .ok_or_else(|| JitCallError::UnknownFunction(name.to_string()))?;
        self.call_index(idx, args)
    }

    /// Call the function at index `idx` (input order); see
    /// [`JitProgram::call`] for the argument contract.
    pub fn call_index(&self, idx: usize, args: &[i64]) -> Result<i64, JitCallError> {
        let (full, buf) = self.prepare_call(idx, args)?;
        let r = self
            .inner
            .call(idx, &full)
            .map_err(JitCallError::Panic)?;
        Ok(match &buf {
            // A wide result's first word is in the sret buffer
            // (its last cell: the leaves lie at descending
            // addresses from the base the callee was given).
            Some(b) => b[b.len() - 1],
            None => r,
        })
    }

    /// Validate the public argument words of a call to `idx` and
    /// build the full IR argument sequence: for a function with a
    /// wide (sret) result, the hidden leading word — the address
    /// of a fresh result buffer, its last cell — is prepended, and
    /// the buffer is returned alongside so the caller can read the
    /// result words back (leaf `k` at cell `len - 1 - k`).
    fn prepare_call(
        &self,
        idx: usize,
        args: &[i64],
    ) -> Result<(Vec<i64>, Option<Vec<i64>>), JitCallError> {
        let wide = self.ret_counts[idx] > 2;
        let expected = self.param_counts[idx] - usize::from(wide);
        if args.len() != expected {
            return Err(JitCallError::WrongArgCount {
                name: self.names[idx].clone(),
                expected,
                got: args.len(),
            });
        }
        if !wide {
            return Ok((args.to_vec(), None));
        }
        let mut buf = vec![0i64; self.ret_counts[idx]];
        let base = buf.as_mut_ptr() as i64 + ((buf.len() - 1) * 8) as i64;
        let mut full = Vec::with_capacity(args.len() + 1);
        full.push(base);
        full.extend_from_slice(args);
        Ok((full, Some(buf)))
    }

    /// Call the function called `name`; see [`JitProgram::call`]
    /// for the argument contract. Returns both words of a two-word
    /// result (a fat pointer: the address, then the length; a pair:
    /// its two fields, in order). The second word is meaningful
    /// only when the function returns two words; a guest panic
    /// yields no words at all.
    pub fn call_pair(&self, name: &str, args: &[i64]) -> Result<(i64, i64), JitCallError> {
        let idx = self
            .index_by_name
            .get(name)
            .copied()
            .ok_or_else(|| JitCallError::UnknownFunction(name.to_string()))?;
        self.call_index_pair(idx, args)
    }

    /// Call the function at index `idx` (input order); see
    /// [`JitProgram::call_pair`] for the contract.
    pub fn call_index_pair(
        &self,
        idx: usize,
        args: &[i64],
    ) -> Result<(i64, i64), JitCallError> {
        let (full, buf) = self.prepare_call(idx, args)?;
        let r = self
            .inner
            .call_pair(idx, &full)
            .map_err(JitCallError::Panic)?;
        Ok(match &buf {
            Some(b) => (b[b.len() - 1], b[b.len() - 2]),
            None => r,
        })
    }
}
