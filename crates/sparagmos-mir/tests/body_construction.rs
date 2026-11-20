//! Hand-build bodies shaped like real 1.100.0 `--emit=mir` output and verify structure.
use sparagmos_mir::*;

/// Mirrors the real 1.100.0 MIR for `fn add(_1: i32, _2: i32) -> i32`:
/// bb0: `_3 = AddWithOverflow(copy _1, copy _2); assert(!move (_3.1)) -> bb1`
/// bb1: `_0 = move (_3.0); return;`
#[test]
fn add_body_like_real_mir() {
    let i32t = Ty::Int(IntTy::I32);
    let tuple_ty = Ty::Tuple(vec![i32t.clone(), Ty::Bool]);
    let locals = vec![
        LocalDecl::new(i32t.clone()),     // _0 return
        LocalDecl::mutable(i32t.clone()), // _1 a
        LocalDecl::mutable(i32t.clone()), // _2 b
        LocalDecl::new(tuple_ty),         // _3 (i32, bool)
    ];
    let p = |n: u32| Place::from_local(Local(n));
    let bb0 = BasicBlockData {
        is_cleanup: false,
        statements: vec![Statement::assign(
            p(3),
            Rvalue::CheckedBinaryOp(
                BinOp::Add,
                Box::new((Operand::copy(p(1)), Operand::copy(p(2)))),
            ),
        )],
        terminator: Some(Terminator::new(TerminatorKind::Assert {
            cond: Operand::move_(p(3).field(1, Ty::Bool)),
            expected: false,
            msg: Box::new(AssertMessage::Overflow(
                BinOp::Add,
                Box::new(Operand::copy(p(1))),
                Box::new(Operand::copy(p(2))),
            )),
            target: BasicBlock(1),
            unwind: UnwindAction::Continue,
        })),
    };
    let bb1 = BasicBlockData {
        is_cleanup: false,
        statements: vec![Statement::assign(
            p(0),
            Rvalue::Use(Operand::move_(p(3).field(0, i32t.clone()))),
        )],
        terminator: Some(Terminator::return_()),
    };
    let body = Body::new(locals, 2, vec![bb0, bb1]);
    assert_eq!(body.arg_count, 2);
    assert_eq!(body.arg_locals(), vec![Local(1), Local(2)]);
    assert_eq!(body.return_ty(), Some(&i32t));
    assert_eq!(body.block_count(), 2);
    assert_eq!(body.statement_count(), 2);
    assert_eq!(body.reachable_blocks(), vec![BasicBlock(0), BasicBlock(1)]);
    assert!(!body.is_diverging());
}

/// Mirrors 1.100.0 MIR for `fn diverge() -> !` / a caller of it:
/// the call to a `!` function has no return target.
#[test]
fn never_body_like_real_mir() {
    let locals = vec![
        LocalDecl::new(Ty::Int(IntTy::I32)), // _0
        LocalDecl::mutable(Ty::Bool),        // _1 x
        LocalDecl::new(Ty::Never),           // _2 result of diverge()
    ];
    let p = |n: u32| Place::from_local(Local(n));
    let bb0 = BasicBlockData::new(Terminator::new(TerminatorKind::SwitchInt {
        discr: Operand::copy(p(1)),
        targets: SwitchTargets::new(vec![0], vec![BasicBlock(2)], BasicBlock(1)),
    }));
    let bb1 = BasicBlockData {
        is_cleanup: false,
        statements: vec![Statement::assign(
            p(0),
            Rvalue::Use(Operand::constant(Constant::int(1))),
        )],
        terminator: Some(Terminator::return_()),
    };
    // bb2: `_2 = diverge() -> unwind continue;` (no return block)
    let bb2 = BasicBlockData::new(Terminator::new(TerminatorKind::Call {
        func: Operand::constant(Constant {
            literal: ConstKind::Ty(Ty::FnDef(DefId::new(0, 1), vec![])),
            user_ty: None,
        }),
        args: vec![],
        destination: p(2),
        target: None,
        unwind: UnwindAction::Continue,
        call_source: CallSource::Normal,
    }));
    let body = Body::new(locals, 1, vec![bb0, bb1, bb2]);
    assert_eq!(body.local_ty(Local(2)), Some(&Ty::Never));
    assert_eq!(body.basic_blocks[2].successors(), Vec::<BasicBlock>::new());
    assert_eq!(body.reachable_blocks().len(), 3);
}

/// SwitchInt / Discriminant / Drop / Resume shapes from loops & enums samples.
#[test]
fn control_flow_shapes() {
    let usizet = Ty::Uint(UintTy::Usize);
    let locals = vec![
        LocalDecl::new(usizet.clone()),
        LocalDecl::new(Ty::Adt(DefId::new(0, 2), vec![usizet.clone()])),
    ];
    let p = |n: u32| Place::from_local(Local(n));
    let bb0 = BasicBlockData {
        is_cleanup: false,
        statements: vec![
            Statement::new(StatementKind::StorageLive(Local(1))),
            Statement::assign(p(0), Rvalue::Discriminant(p(1))),
            Statement::new(StatementKind::StorageDead(Local(1))),
        ],
        terminator: Some(Terminator::new(TerminatorKind::Drop {
            place: p(1),
            target: BasicBlock(1),
            unwind: UnwindAction::Cleanup(BasicBlock(2)),
            replace: false,
        })),
    };
    let bb1 = BasicBlockData::new(Terminator::return_());
    let bb2 = BasicBlockData::cleanup(Terminator::new(TerminatorKind::Resume));
    let body = Body::new(locals, 0, vec![bb0, bb1, bb2]);
    assert_eq!(body.statement_count(), 3);
    assert_eq!(
        body.basic_blocks[0].successors(),
        vec![BasicBlock(1), BasicBlock(2)]
    );
    assert!(body.basic_blocks[2].is_cleanup);
    assert_eq!(
        coverage::model_variant_total(),
        coverage::model_tables().iter().map(|(_, n)| n).sum::<usize>()
    );
    // The metric-1 denominator is a separate, fixed set: the 40 variants
    // of rustc 1.100.0's three main MIR enums (docs/COVERAGE.md).
    assert_eq!(coverage::metric1_variant_count(), 40);
}
