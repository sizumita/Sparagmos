//! Dual-engine tests for payload-carrying ADTs in the JIT
//! (Lane CB): payload structs and enums are laid out as a tag
//! leaf (multi-variant ADTs only) plus a payload area shared by
//! all variants, each variant's fields in order from the payload
//! start. Every test runs the same hand-written MIR body through
//! the JIT and the interpreter and requires both to produce the
//! hand-computed answer.

use std::collections::HashMap;

use sparagmos_interp::{InterpError, MirRunError, Value};
use sparagmos_jit::{JitCallError, JitError, JitProgram, MirFunctionInput, MirProgramInput};
use sparagmos_mir as mir;

// ----- small builders (same conventions as mir_jit.rs) ---------------------

fn def(index: u32) -> mir::DefId {
    mir::DefId { krate: 0, index }
}

fn bb(stmts: Vec<mir::StatementKind>, term: mir::TerminatorKind) -> mir::BasicBlockData {
    mir::BasicBlockData {
        statements: stmts.into_iter().map(mir::Statement::new).collect(),
        terminator: Some(mir::Terminator::new(term)),
        is_cleanup: false,
    }
}

fn assign(place: mir::Place, rv: mir::Rvalue) -> mir::StatementKind {
    mir::StatementKind::Assign(Box::new((place, rv)))
}

fn pl(n: u32) -> mir::Place {
    mir::Place::from_local(mir::Local(n))
}

fn cp(p: mir::Place) -> mir::Operand {
    mir::Operand::Copy(p)
}

fn mv(p: mir::Place) -> mir::Operand {
    mir::Operand::Move(p)
}

fn i32t() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}

fn iconst(data: u128, size: u8) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int { data, size },
    )))
}

fn i32c(v: i32) -> mir::Operand {
    iconst(v as u32 as u128, 4)
}

fn fndef(d: u32) -> mir::Operand {
    mir::Operand::constant(mir::Constant {
        literal: mir::ConstKind::Ty(mir::Ty::FnDef(def(d), vec![])),
        user_ty: None,
        ty: None,
    })
}

fn call_term(
    func: mir::Operand,
    args: Vec<mir::Operand>,
    dest: u32,
    target: u32,
) -> mir::TerminatorKind {
    mir::TerminatorKind::Call {
        func,
        args,
        destination: pl(dest),
        target: Some(mir::BasicBlock(target)),
        unwind: mir::UnwindAction::Continue,
        call_source: mir::CallSource::Normal,
    }
}

fn locals(tys: &[mir::Ty]) -> Vec<mir::LocalDecl> {
    tys.iter().cloned().map(mir::LocalDecl::new).collect()
}

// ----- dual-engine harness (same shape as mir_jit.rs) ------------------------

struct FnSpec {
    name: &'static str,
    def: u32,
    body: mir::Body,
}

fn spec(
    name: &'static str,
    def: u32,
    arg_count: usize,
    tys: &[mir::Ty],
    blocks: Vec<mir::BasicBlockData>,
) -> FnSpec {
    FnSpec {
        name,
        def,
        body: mir::Body::new(locals(tys), arg_count, blocks),
    }
}

fn build_with_table(
    specs: &[FnSpec],
    table: mir::AdtDiscriminants,
) -> (JitProgram, sparagmos_interp::MirProgram) {
    let names: HashMap<mir::DefId, String> = specs
        .iter()
        .map(|s| (def(s.def), s.name.to_string()))
        .collect();
    let jit_in = MirProgramInput {
        functions: specs
            .iter()
            .map(|s| MirFunctionInput {
                name: s.name.to_string(),
                def_id: Some(def(s.def)),
                body: s.body.clone(),
            })
            .collect(),
        names: names.clone(),
        adt_discriminants: table.clone(),
    };
    let interp_in = sparagmos_interp::MirProgram {
        functions: specs
            .iter()
            .map(|s| sparagmos_interp::MirFunction {
                name: s.name.to_string(),
                def_id: Some(def(s.def)),
                body: s.body.clone(),
            })
            .collect(),
        fn_names: names,
        adt_discriminants: table,
        adt_layouts: Default::default(),
        coroutine_layouts: Default::default(),
        adt_shapes: Default::default(),
        adt_debug_shapes: Default::default(),
        adt_field_layouts: Default::default(),
        env_debug_shapes: Default::default(),
        thread_locals: Default::default(),
        static_tys: Default::default(),
        adt_instance_shapes: Default::default(),
        adt_niches: Default::default(),
        source_files: Default::default(),
        drop_trivial_adts: Default::default(),
        drop_no_impl_adts: Default::default(),
        dtor_bodies: Default::default(),

        track_caller_defs: Default::default(),
        fn_shim_locations: Default::default(),
    };
    let jit = JitProgram::compile(&jit_in).expect("JIT compile failed");
    (jit, interp_in)
}

fn jit_call(jit: &JitProgram, name: &str, args: &[i64]) -> Result<i64, String> {
    jit.call(name, args).map_err(|e| match e {
        JitCallError::Panic(rec) => rec.message,
        other => other.to_string(),
    })
}

fn interp_call(
    prog: &sparagmos_interp::MirProgram,
    name: &str,
    args: Vec<Value>,
) -> Result<i64, String> {
    match sparagmos_interp::interpret_mir(prog, name, args) {
        Ok(v) => Ok(match v {
            Value::Int(x) => x as i64,
            Value::Uint(x) => x as i64,
            Value::Bool(b) => b as i64,
            Value::Unit => 0,
            other => panic!("unexpected interp result {other:?}"),
        }),
        Err(MirRunError::Interp(InterpError::Panic { message, .. })) => Err(message),
        Err(other) => panic!("unexpected interp error: {other}"),
    }
}

fn check(
    jit: &JitProgram,
    prog: &sparagmos_interp::MirProgram,
    name: &str,
    expected: i64,
) {
    assert_eq!(jit_call(jit, name, &[]), Ok(expected), "JIT result for {name}");
    assert_eq!(
        interp_call(prog, name, vec![]),
        Ok(expected),
        "interp result for {name}"
    );
}

// ----- the ADT types under test ----------------------------------------------

/// `struct Pair { a: i32, b: i32 }` — DefId 50, no tag leaf.
fn pair_ty() -> mir::Ty {
    mir::Ty::Adt(def(50), vec![])
}

/// `enum Shape { None, One(i32), Two(i32, i32) }` — DefId 51,
/// discriminants [5, 9, 13] (recorded values, not indices).
fn shape_ty() -> mir::Ty {
    mir::Ty::Adt(def(51), vec![])
}

/// `enum Opt { None, Some(i32) }` — DefId 52, two leaves total.
fn opt_ty() -> mir::Ty {
    mir::Ty::Adt(def(52), vec![])
}

/// `struct Outer { p: Pair, n: i32 }` — DefId 53.
fn outer_ty() -> mir::Ty {
    mir::Ty::Adt(def(53), vec![])
}

fn tables() -> mir::AdtDiscriminants {
    let mut t = mir::AdtDiscriminants::new();
    t.insert(def(51), vec![5, 9, 13]);
    t
}

fn adt_agg_on(
    n: u32,
    adt_def: u32,
    variant_index: u32,
    ops: Vec<mir::Operand>,
) -> mir::StatementKind {
    assign(
        pl(n),
        mir::Rvalue::Aggregate(
            Box::new(mir::AggregateKind::Adt {
                def: def(adt_def),
                variant_index,
                generic_args: vec![],
                user_ty: None,
                field_index: None,
            }),
            ops,
        ),
    )
}

fn pair_agg(n: u32, a: mir::Operand, b: mir::Operand) -> mir::StatementKind {
    adt_agg_on(n, 50, 0, vec![a, b])
}

fn binop(op: mir::BinOp, l: mir::Operand, r: mir::Operand) -> mir::Rvalue {
    mir::Rvalue::BinaryOp(op, Box::new((l, r)))
}

// ----- payload structs --------------------------------------------------------

#[test]
fn payload_struct_construct_and_project() {
    // fn f() -> i32 { _1 = Pair { a: 3, b: 4 }; _0 = _1.a + _1.b; return }
    let f = spec(
        "f",
        0,
        0,
        &[i32t(), pair_ty()],
        vec![bb(
            vec![
                pair_agg(1, i32c(3), i32c(4)),
                assign(
                    pl(0),
                    binop(
                        mir::BinOp::Add,
                        cp(pl(1).field(0, i32t())),
                        cp(pl(1).field(1, i32t())),
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build_with_table(&[f], tables());
    check(&jit, &prog, "f", 7);
}

#[test]
fn payload_struct_mutate_field() {
    // fn f() -> i32 { _1 = Pair { a: 1, b: 2 }; _1.b = 40;
    //                 _0 = _1.a + _1.b; return }
    let f = spec(
        "f",
        0,
        0,
        &[i32t(), pair_ty()],
        vec![bb(
            vec![
                pair_agg(1, i32c(1), i32c(2)),
                assign(pl(1).field(1, i32t()), mir::Rvalue::Use(i32c(40))),
                assign(
                    pl(0),
                    binop(
                        mir::BinOp::Add,
                        cp(pl(1).field(0, i32t())),
                        cp(pl(1).field(1, i32t())),
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build_with_table(&[f], tables());
    check(&jit, &prog, "f", 41);
}

#[test]
fn payload_struct_as_argument_and_return() {
    // fn get_b(p: Pair) -> i32 { return p.b }
    // fn mkpair(x: i32) -> Pair { _2 = x + 1; return Pair { a: x, b: _2 } }
    // fn main() -> i32 { _1 = mkpair(7); _0 = get_b(_1); return }
    let get_b = spec(
        "get_b",
        1,
        1,
        &[i32t(), pair_ty()],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::Use(cp(pl(1).field(1, i32t()))),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    let mkpair = spec(
        "mkpair",
        2,
        1,
        &[pair_ty(), i32t(), i32t()],
        vec![bb(
            vec![
                assign(pl(2), binop(mir::BinOp::Add, cp(pl(1)), i32c(1))),
                pair_agg(0, cp(pl(1)), cp(pl(2))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let main = FnSpec {
        name: "main",
        def: 0,
        body: mir::Body::new(
            locals(&[i32t(), pair_ty()]),
            0,
            vec![
                bb(vec![], call_term(fndef(2), vec![i32c(7)], 1, 1)),
                bb(vec![], call_term(fndef(1), vec![mv(pl(1))], 0, 2)),
                bb(vec![], mir::TerminatorKind::Return),
            ],
        ),
    };
    let (jit, prog) = build_with_table(&[get_b, mkpair, main], tables());
    check(&jit, &prog, "main", 8);
}

#[test]
fn nested_payload_struct_field() {
    // fn f() -> i32 { _1 = Pair { a: 1, b: 2 }; _2 = Outer { p: _1, n: 10 };
    //                 _0 = _2.p.b + _2.n; return }
    let f = spec(
        "f",
        0,
        0,
        &[i32t(), pair_ty(), outer_ty()],
        vec![bb(
            vec![
                pair_agg(1, i32c(1), i32c(2)),
                adt_agg_on(2, 53, 0, vec![mv(pl(1)), i32c(10)]),
                assign(
                    pl(0),
                    binop(
                        mir::BinOp::Add,
                        cp(pl(2).field(0, pair_ty()).field(1, i32t())),
                        cp(pl(2).field(1, i32t())),
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build_with_table(&[f], tables());
    check(&jit, &prog, "f", 12);
}

// ----- payload enums ------------------------------------------------------------

#[test]
fn payload_enum_discriminant_switch_and_fields() {
    // fn f() -> i32 {
    //   _1 = Shape::Two(7, 100); _2 = Discriminant(_1);
    //   SwitchInt(_2) { 13 => bb1, otherwise => bb2 }
    //   bb1: _0 = (_1 as Two).0 + (_1 as Two).1; return
    //   bb2: _0 = -1; return
    // }
    let f = spec(
        "f",
        0,
        0,
        &[i32t(), shape_ty(), i32t()],
        vec![
            bb(
                vec![
                    adt_agg_on(1, 51, 2, vec![i32c(7), i32c(100)]),
                    assign(pl(2), mir::Rvalue::Discriminant(pl(1))),
                ],
                mir::TerminatorKind::SwitchInt {
                    discr: cp(pl(2)),
                    targets: mir::SwitchTargets::new(
                        vec![13],
                        vec![mir::BasicBlock(1)],
                        mir::BasicBlock(2),
                    ),
                },
            ),
            bb(
                vec![assign(
                    pl(0),
                    binop(
                        mir::BinOp::Add,
                        cp(pl(1).downcast(None, 2).field(0, i32t())),
                        cp(pl(1).downcast(None, 2).field(1, i32t())),
                    ),
                )],
                mir::TerminatorKind::Return,
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(i32c(-1)))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build_with_table(&[f], tables());
    check(&jit, &prog, "f", 107);
}

#[test]
fn payload_enum_set_discriminant_to_fieldless_variant() {
    // fn f() -> i32 {
    //   _1 = Shape::One(1); SetDiscriminant(_1, 0);
    //   _0 = Discriminant(_1); return   // 5, the table value of None
    // }
    // SetDiscriminant writes only the tag leaf of a payload
    // enum. (The companion shape — SetDiscriminant followed by
    // field writes through the new variant's downcast, the
    // in-place reinitialisation rustc emits — is JIT-supported
    // but cannot be dual-tested here: the interpreter clears the
    // payload on SetDiscriminant and its field-write navigation
    // does not regrow the field vector, so it rejects the write.
    // Recorded as an interpreter gap in the lane report.)
    let f = spec(
        "f",
        0,
        0,
        &[i32t(), shape_ty()],
        vec![bb(
            vec![
                adt_agg_on(1, 51, 1, vec![i32c(1)]),
                mir::StatementKind::SetDiscriminant {
                    place: Box::new(pl(1)),
                    variant_index: 0,
                },
                assign(pl(0), mir::Rvalue::Discriminant(pl(1))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build_with_table(&[f], tables());
    check(&jit, &prog, "f", 5);
}

#[test]
fn payload_enum_as_argument_and_return() {
    // fn bump(e: Opt) -> Opt {
    //   _2 = Discriminant(e); SwitchInt(_2) { 1 => bb1, otherwise => bb2 }
    //   bb1: _3 = (e as Some).0 + 1; _0 = Opt::Some(_3); return
    //   bb2: _0 = Opt::None; return
    // }
    // fn main() -> i32 { _1 = Opt::Some(10); _2 = bump(_1);
    //                    _0 = (_2 as Some).0; return }
    // fn main_none() -> i32 { _1 = Opt::None; _2 = bump(_1);
    //                         _0 = Discriminant(_2); return }
    let bump = spec(
        "bump",
        1,
        1,
        &[opt_ty(), opt_ty(), i32t(), i32t()],
        vec![
            bb(
                vec![assign(pl(2), mir::Rvalue::Discriminant(pl(1)))],
                mir::TerminatorKind::SwitchInt {
                    discr: cp(pl(2)),
                    targets: mir::SwitchTargets::new(
                        vec![1],
                        vec![mir::BasicBlock(1)],
                        mir::BasicBlock(2),
                    ),
                },
            ),
            bb(
                vec![
                    assign(
                        pl(3),
                        binop(
                            mir::BinOp::Add,
                            cp(pl(1).downcast(None, 1).field(0, i32t())),
                            i32c(1),
                        ),
                    ),
                    adt_agg_on(0, 52, 1, vec![cp(pl(3))]),
                ],
                mir::TerminatorKind::Return,
            ),
            bb(vec![adt_agg_on(0, 52, 0, vec![])], mir::TerminatorKind::Return),
        ],
    );
    let mk_main = |name: &'static str, mdef: u32, some: bool| FnSpec {
        name,
        def: mdef,
        body: mir::Body::new(
            locals(&[i32t(), opt_ty(), opt_ty()]),
            0,
            vec![
                bb(
                    vec![if some {
                        adt_agg_on(1, 52, 1, vec![i32c(10)])
                    } else {
                        adt_agg_on(1, 52, 0, vec![])
                    }],
                    call_term(fndef(1), vec![mv(pl(1))], 2, 1),
                ),
                bb(
                    vec![assign(
                        pl(0),
                        if some {
                            mir::Rvalue::Use(cp(pl(2).downcast(None, 1).field(0, i32t())))
                        } else {
                            mir::Rvalue::Discriminant(pl(2))
                        },
                    )],
                    mir::TerminatorKind::Return,
                ),
            ],
        ),
    };
    let (jit, prog) = build_with_table(
        &[bump, mk_main("main", 0, true), mk_main("main_none", 3, false)],
        tables(),
    );
    check(&jit, &prog, "main", 11);
    check(&jit, &prog, "main_none", 0);
}

#[test]
fn payload_adt_constant_with_fields() {
    // fn f() -> i32 { _1 = const Shape::Two(7, 8);
    //                 _0 = (_1 as Two).0 + (_1 as Two).1; return }
    let konst = mir::Operand::constant(mir::Constant {
        literal: mir::ConstKind::Value(mir::ConstValue::Adt {
            variant: 2,
            fields: vec![
                mir::ConstValue::Scalar(mir::Scalar::Int { data: 7, size: 4 }),
                mir::ConstValue::Scalar(mir::Scalar::Int { data: 8, size: 4 }),
            ],
        }),
        user_ty: None,
        ty: None,
    });
    let f = spec(
        "f",
        0,
        0,
        &[i32t(), shape_ty()],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(konst)),
                assign(
                    pl(0),
                    binop(
                        mir::BinOp::Add,
                        cp(pl(1).downcast(None, 2).field(0, i32t())),
                        cp(pl(1).downcast(None, 2).field(1, i32t())),
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build_with_table(&[f], tables());
    check(&jit, &prog, "f", 15);
}

// ----- payload ADTs through references -------------------------------------------

fn refmty(inner: mir::Ty) -> mir::Ty {
    mir::Ty::Ref(mir::Region::Erased, Box::new(inner), mir::Mutability::Mut)
}

#[test]
fn payload_struct_field_write_through_reference() {
    // fn f() -> i32 { _1 = Pair { a: 3, b: 4 }; _2 = &mut _1; (*_2).b = 40;
    //                 _0 = _1.a + _1.b; return }
    let f = spec(
        "f",
        0,
        0,
        &[i32t(), pair_ty(), refmty(pair_ty())],
        vec![bb(
            vec![
                pair_agg(1, i32c(3), i32c(4)),
                assign(
                    pl(2),
                    mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::mutable(), pl(1)),
                ),
                assign(pl(2).deref().field(1, i32t()), mir::Rvalue::Use(i32c(40))),
                assign(
                    pl(0),
                    binop(
                        mir::BinOp::Add,
                        cp(pl(1).field(0, i32t())),
                        cp(pl(1).field(1, i32t())),
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build_with_table(&[f], tables());
    check(&jit, &prog, "f", 43);
}

#[test]
fn payload_struct_copy_for_deref() {
    // fn f() -> i32 { _1 = Pair { a: 3, b: 4 }; _2 = &_1;
    //                 _3 = CopyForDeref(_2); _0 = _3.a + _3.b; return }
    let refty = mir::Ty::Ref(mir::Region::Erased, Box::new(pair_ty()), mir::Mutability::Not);
    let f = spec(
        "f",
        0,
        0,
        &[i32t(), pair_ty(), refty, pair_ty()],
        vec![bb(
            vec![
                pair_agg(1, i32c(3), i32c(4)),
                assign(
                    pl(2),
                    mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, pl(1)),
                ),
                assign(pl(3), mir::Rvalue::CopyForDeref(pl(2))),
                assign(
                    pl(0),
                    binop(
                        mir::BinOp::Add,
                        cp(pl(3).field(0, i32t())),
                        cp(pl(3).field(1, i32t())),
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build_with_table(&[f], tables());
    check(&jit, &prog, "f", 7);
}

#[test]
fn payload_struct_aggregate_through_reference() {
    // fn f() -> i32 { _1 = Pair { a: 0, b: 0 }; _2 = &mut _1;
    //                 *_2 = Pair { a: 5, b: 6 }; _0 = _1.a + _1.b; return }
    let f = spec(
        "f",
        0,
        0,
        &[i32t(), pair_ty(), refmty(pair_ty())],
        vec![bb(
            vec![
                pair_agg(1, i32c(0), i32c(0)),
                assign(
                    pl(2),
                    mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::mutable(), pl(1)),
                ),
                assign(
                    pl(2).deref(),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Adt {
                            def: def(50),
                            variant_index: 0,
                            generic_args: vec![],
                            user_ty: None,
                            field_index: None,
                        }),
                        vec![i32c(5), i32c(6)],
                    ),
                ),
                assign(
                    pl(0),
                    binop(
                        mir::BinOp::Add,
                        cp(pl(1).field(0, i32t())),
                        cp(pl(1).field(1, i32t())),
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build_with_table(&[f], tables());
    check(&jit, &prog, "f", 11);
}

#[test]
fn payload_enum_discriminant_through_reference() {
    // fn f() -> i32 { _1 = Shape::One(9); _2 = &mut _1;
    //                 SetDiscriminant(*_2, 2); _0 = Discriminant(_1); return }
    let f = spec(
        "f",
        0,
        0,
        &[i32t(), shape_ty(), refmty(shape_ty()), shape_ty(), i32t()],
        vec![bb(
            vec![
                // A dead Two construction (its discriminant is
                // read into _4) so the program reveals every
                // variant's field types, as a match would.
                assign(pl(4), mir::Rvalue::Use(i32c(9))),
                adt_agg_on(3, 51, 2, vec![cp(pl(4)), cp(pl(4))]),
                assign(pl(4), mir::Rvalue::Discriminant(pl(3))),
                adt_agg_on(1, 51, 1, vec![i32c(9)]),
                assign(
                    pl(2),
                    mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::mutable(), pl(1)),
                ),
                mir::StatementKind::SetDiscriminant {
                    place: Box::new(pl(2).deref()),
                    variant_index: 2,
                },
                assign(pl(0), mir::Rvalue::Discriminant(pl(1))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build_with_table(&[f], tables());
    check(&jit, &prog, "f", 13);
}

// ----- shapes that stay unsupported ------------------------------------------------

fn expect_unsupported(f: FnSpec, substr: &str) {
    let names: HashMap<mir::DefId, String> =
        [(def(f.def), f.name.to_string())].into_iter().collect();
    let err = JitProgram::compile(&MirProgramInput {
        functions: vec![MirFunctionInput {
            name: f.name.to_string(),
            def_id: Some(def(f.def)),
            body: f.body,
        }],
        names,
        adt_discriminants: tables(),
    })
    .expect_err("compile must fail");
    match err {
        JitError::Unsupported(what) => {
            assert!(what.contains(substr), "{what:?} lacks {substr:?}")
        }
        other => panic!("expected Unsupported, got {other}"),
    }
}

#[test]
fn union_aggregate_is_unsupported() {
    // A union aggregate carries an active-field index; unions are
    // outside the layout model.
    let f = spec(
        "f",
        0,
        0,
        &[i32t(), pair_ty()],
        vec![bb(
            vec![assign(
                pl(1),
                mir::Rvalue::Aggregate(
                    Box::new(mir::AggregateKind::Adt {
                        def: def(50),
                        variant_index: 0,
                        generic_args: vec![],
                        user_ty: None,
                        field_index: Some(0),
                    }),
                    vec![i32c(1)],
                ),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    expect_unsupported(f, "union ADT");
}

#[test]
fn unrevealed_field_type_is_unsupported() {
    // The variant is constructed with one field, but the field's
    // type is revealed nowhere: the operand is a function-item
    // constant (whose type the layout scan does not derive) and
    // no projection ever names the field. No layout can be
    // computed, so compilation fails explicitly. (A variant that
    // is never constructed or projected anywhere is simply
    // fieldless in the layout — nothing can observe its payload.)
    let f = spec(
        "f",
        0,
        0,
        &[i32t(), shape_ty()],
        vec![bb(
            vec![
                adt_agg_on(1, 51, 1, vec![fndef(7)]),
                assign(pl(0), mir::Rvalue::Discriminant(pl(1))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    expect_unsupported(f, "field type not recoverable");
}

#[test]
fn oversized_adt_is_unsupported() {
    // A struct with 65 scalar fields exceeds the layout leaf bound.
    let big_ty = mir::Ty::Adt(def(54), vec![]);
    let mut tys = vec![i32t(), big_ty];
    for _ in 0..65 {
        tys.push(i32t());
    }
    let mut stmts = vec![adt_agg_on(1, 54, 0, (0..65).map(|_| i32c(1)).collect())];
    for k in 0..65u32 {
        stmts.push(assign(
            pl(2 + k),
            mir::Rvalue::Use(cp(pl(1).field(k, i32t()))),
        ));
    }
    stmts.push(assign(pl(0), mir::Rvalue::Use(cp(pl(2)))));
    let f = spec("f", 0, 0, &tys, vec![bb(stmts, mir::TerminatorKind::Return)]);
    expect_unsupported(f, "layout bound");
}
