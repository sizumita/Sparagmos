//! The MIR front end: compiles `sparagmos-mir` bodies to executable
//! code through the crate's existing backend IR.
//!
//! # Design: lowering, not a second code generator
//!
//! MIR bodies are *lowered* to the backend IR ([`crate::ir`]) and the
//! result is compiled by the existing baseline compiler as one
//! program ([`crate::program`]). The backend was always shaped for
//! this: it is a locals-in-stack-slots machine, and MIR locals are
//! mutable slots, so each MIR local maps onto one IR value slot per
//! scalar leaf (a tuple local occupies consecutive slots, one per
//! leaf). Writing a second, MIR-direct code generator instead would
//! duplicate the emitter, the W^X memory handling and the calling
//! convention for no semantic gain.
//!
//! # Value representation
//!
//! Every scalar leaf is one 64-bit word in *canonical form* for its
//! MIR type: sign-extended for signed integer types, zero-extended
//! (masked to the type's width) for unsigned types, `bool` and
//! `char`. The lowering re-establishes the invariant after every
//! operation whose raw 64-bit result can leave the type's range
//! (plain arithmetic, shifts, casts), so the backend's plain 64-bit
//! operations are exact for every integer type of up to 64 bits, and
//! comparisons only need the operand type's signedness.
//!
//! A float leaf holds its IEEE 754 bit pattern instead: all 64 bits
//! for `f64`, the low 32 bits (upper half zero) for `f32`. The
//! backend computes on the patterns in SSE registers, so float
//! arithmetic, comparisons and casts follow the hardware's IEEE
//! semantics — which are exactly Rust's — with no normalisation
//! step in the lowering.
//!
//! An ADT value is its discriminant tag followed by its payload:
//! one tag leaf (for a multi-variant ADT) holding the discriminant
//! value from the program's discriminant table, exactly as the
//! interpreter computes `Discriminant` results, then the payload
//! leaves — for an enum, the variants share one payload area, each
//! variant's fields laid out in order from its start; a struct
//! (a single-variant ADT) has no tag leaf and is just its fields.
//! The layout of each ADT instantiation is recovered from the
//! program itself before lowering: every `Field` projection in
//! MIR carries the field's declared type, and every aggregate
//! construction lists its variant's field count, so a scan of all
//! bodies (the interpreter's own recovery method) pins down each
//! variant's field types. An ADT the scan says nothing about keeps
//! the historical tag-only shape. Shapes the scan cannot pin down
//! — a field whose type no projection or construction reveals, a
//! value wider than the layout bound, a recursive layout — fail
//! with an explicit unsupported error, as do unions and payload
//! shapes outside the layout (see the scope list).
//!
//! A thin reference or raw pointer is one word: the address of
//! the pointee's first leaf. Taking a reference to a frame place
//! emits the backend's slot-address instruction; leaf `k` of the
//! pointee lies `8 * k` bytes below that address, matching the
//! backend's descending slot layout, so a place reached through
//! dereferences and field/index projections resolves to an
//! address expression — a base pointer (a slot's value, a slot's
//! address, or a pointer loaded from memory), a constant leaf
//! offset, and at most one dynamic index term — and reads and
//! writes through it land on the original place, the interpreter's
//! model of a reference. This holds across a call, too: a caller's
//! frame outlives the call, so a callee writing through a pointer
//! argument writes the caller's place. A second dereference loads
//! the inner pointer from memory and continues the same way.
//!
//! A fat reference — to a slice or to `str` — is two words: the
//! address above plus the length, carried in the next slot
//! wherever the reference lives (a local, an argument, a call
//! argument). Taking a reference to a slice/str place yields the
//! place's address and its length (a constant for arrays, the
//! fat pair's own length word for a reborrowed slice); the
//! `Unsize` coercion from `&[T; N]` builds the pair from the
//! array's address and the constant `N`. `Len` and `PtrMetadata`
//! read the length word. Indexing a slice (dynamically or with a
//! literal) checks the index against the length word and panics
//! with the interpreter's index message, exactly like indexing an
//! array local. References to trait objects stay unsupported:
//! their metadata word is a vtable pointer, which the slot model
//! has no representation for.
//!
//! # Panics
//!
//! A guest panic (failed `Assert`, checked-arithmetic overflow via
//! the `Assert` rustc emits, division by zero) is compiled to the
//! backend's panic instruction: the message is recorded through
//! `sparagmos-runtime`'s panic path and the program's panic flag is
//! set, after which the panic propagates out of the JIT call stack
//! and [`JitProgram::call`] returns it as a [`PanicRecord`]. Division
//! and remainder get explicit divisor guards at lowering time because
//! the hardware instructions trap on a zero divisor (and on
//! `MIN / -1` in the 64-bit signed case). An out-of-bounds index
//! is a guest panic with the interpreter's message — `index out of
//! bounds: the len is {len} but the index is {index}`, formatted
//! at runtime from the actual length and index values. A negative
//! signed index is the interpreter's negative-index panic —
//! `index out of bounds: negative index {index} used for
//! {place}`, naming the indexed place — and a sub-slice range
//! that does not fit its base is one of the interpreter's range
//! panics (`range end index {end} out of range for slice of
//! length {len}`, or `slice index starts at {from} but ends at
//! {end}` for an inverted range), likewise formatted at runtime.
//! Reaching a MIR `Unreachable`
//! block is a guest panic too, with the message `entered unreachable
//! code` — the interpreter's model — lowered to a jump into a panic
//! block like a failed `Assert`, never to a hardware trap.
//! Cleanup blocks are never entered either — the panic model
//! propagates through the panic flag, not through unwind edges —
//! so their closing `Resume` is sealed the same way.
//!
//! The *plain* (unchecked) operators wrap, in both engines: the
//! overflow panic belongs to the `Assert` rustc emits after a
//! checked operation, never to the plain operation itself. Plain
//! `Add` / `Sub` / `Mul` are compiled through the checked backend
//! operation with its flag unused, plain division wraps the
//! `MIN / -1` case (rewritten so the hardware instruction cannot
//! trap on it), plain shifts take the amount modulo the type
//! width, and plain `Neg` wraps the type's minimum to itself. The
//! checked forms (`CheckedBinaryOp`, `*WithOverflow`) yield the
//! flag as a value instead, as in the interpreter.
//!
//! # Scope
//!
//! Supported: integer / `bool` / `char` / `f32` / `f64` /
//! tuple-of-supported and array-of-supported locals (arrays are
//! their elements' leaves in order, read and written through
//! constant and dynamic index projections — signed indices
//! included — and sub-slice projections, each with the
//! interpreter's checks and panics), ADT locals (structs
//! and enums with scalar, tuple, array, reference and nested-ADT
//! payload fields, within the layout bound), thin references and
//! raw pointers, fat
//! references to slices and `str` (as address+length pairs), and
//! projections through dereferences — fields, indices, further
//! dereferences, `Discriminant` / `SetDiscriminant` of an ADT
//! in memory, and references taken of memory places;
//! `Use`, `CopyForDeref`, plain and checked `BinaryOp`, `UnaryOp`
//! (including `PtrMetadata`), `IntToInt` / `FloatToInt` /
//! `IntToFloat` / `FloatToFloat` casts, the address-preserving
//! pointer casts and the array-to-slice `Unsize` coercion, `Len`,
//! tuple aggregates, array aggregates, ADT aggregates (structs and enums,
//! including payloads), `Repeat`,
//! `Discriminant`, `SetDiscriminant`; `Goto`, `SwitchInt`, `Call`
//! (within the program, any number of argument words — the first
//! six in the System V registers, the rest on the stack; see
//! [`arg_word_layout`] for the exact contract — and results of up
//! to two words — a fat pointer or a pair — returned in RAX/RDX,
//! the System V pair-return convention; wider results are stored
//! through a hidden leading sret argument word),
//! `Assert`, `Drop`, `Return`, `Unreachable` (as a guest panic);
//! by-value iteration over arrays: `IntoIterator::into_iter` and
//! `Iterator::next` for `std::array::IntoIter` are not part of
//! the program (the driver leaves them to the interpreter's
//! builtins) and are lowered inline, the iterator value holding
//! the array's elements followed by the `alive` range (see
//! [`ArrayIterShape`]).
//! Anything else — `f16` / `f128`, references
//! to trait objects, unions, ADT values wider than the layout
//! bound or with a field type the program never reveals, a second dynamic
//! index within one place,
//! a literal index from the end of a slice,
//! aggregates other than tuples, ADTs and arrays, calls to functions
//! outside the program, and
//! so on — fails the whole function's compilation with
//! [`JitError::Unsupported`] naming the construct. There is
//! deliberately no partial or fallback execution: a program
//! either compiles completely on the JIT path or not at all.

#[allow(unused_imports)]
use adt::ArrayIterShape;
#[allow(unused_imports)]
use sparagmos_runtime::error::PanicRecord;
#[allow(unused_imports)]
use ty_helpers::arg_word_layout;

mod adt;
mod lower;
mod tables;
mod ty_helpers;
mod types;

pub use types::{JitCallError, JitError, JitProgram, MirFunctionInput, MirProgramInput};
