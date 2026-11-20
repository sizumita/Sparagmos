//! Anonymous `const _` bodies: the driver names every anonymous
//! const in a crate by the same item path (`{crate}::_`), so one
//! program can hold several distinct bodies under a single name.
//! Conversion disambiguates them by `DefId` (see `program_keys` in
//! mir_convert); a reference — which names a `DefId`, never the
//! shared display name — must evaluate its own const's body, not a
//! sibling's.

use sparagmos_interp::mir_convert::{
    convert_program, interpret_mir, ConvertError, MirProgram, MirRunError,
};
use sparagmos_interp::{InterpError, Value};
use sparagmos_mir as mir;

// ----- small builders over sparagmos_mir (mirroring body_isolation_tests) ---

fn u32_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U32)
}

fn unit_ty() -> mir::Ty {
    mir::Ty::Tuple(Vec::new())
}

/// A `dyn Trait` type: outside the interpreter's type model, so
/// any body holding one as a local fails conversion.
fn dyn_ty() -> mir::Ty {
    mir::Ty::Dynamic(
        vec!["std::fmt::Debug".to_string()],
        Box::new(unit_ty()),
        mir::Region::Erased,
    )
}

fn decls(tys: &[mir::Ty]) -> Vec<mir::LocalDecl> {
    tys.iter().cloned().map(mir::LocalDecl::new).collect()
}

fn place(idx: u32) -> mir::Place {
    mir::Place::from_local(mir::Local(idx))
}

fn const_u32(v: u32) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int {
            data: v as u128,
            size: 4,
        },
    )))
}

fn block(stmts: Vec<mir::Statement>, kind: mir::TerminatorKind) -> mir::BasicBlockData {
    let mut bb = mir::BasicBlockData::new(mir::Terminator::new(kind));
    bb.statements = stmts;
    bb
}

fn assign(place: mir::Place, rv: mir::Rvalue) -> mir::Statement {
    mir::Statement::assign(place, rv)
}

fn ret() -> mir::TerminatorKind {
    mir::TerminatorKind::Return
}

// ----- the anonymous consts --------------------------------------------------

/// The shared display name of both anonymous consts, as the driver
/// reports it for a crate's `const _` items.
const ANON_NAME: &str = "test_crate::_";
const CONST_A: mir::DefId = mir::DefId::new(0, 10);
const CONST_B: mir::DefId = mir::DefId::new(0, 11);

/// `const _: u32 = <v>;` — a zero-argument body returning a constant.
fn const_body(v: u32) -> mir::Body {
    mir::Body::new(
        decls(&[u32_ty()]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_u32(v)))],
            ret(),
        )],
    )
}

/// A const body that cannot convert (a trait-object local).
fn broken_const_body() -> mir::Body {
    mir::Body::new(
        decls(&[u32_ty(), dyn_ty()]),
        0,
        vec![block(vec![], ret())],
    )
}

fn unevaluated_operand(def: mir::DefId) -> mir::Operand {
    mir::Operand::constant(mir::Constant {
        literal: mir::ConstKind::Unevaluated {
            def,
            args: Vec::new(),
        },
        user_ty: None,
    })
}

fn call_const(def: mir::DefId, dest: u32, target: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Call {
        func: unevaluated_operand(def),
        args: vec![],
        destination: place(dest),
        target: Some(mir::BasicBlock(target)),
        unwind: mir::UnwindAction::Continue,
        call_source: mir::CallSource::Normal,
    }
}

/// `fn main() -> u32 { _1 = CONST_A; _2 = CONST_B; _0 = _1 + _2; return }`
/// where each const is referenced by its `DefId`, as MIR names an
/// unevaluated constant operand.
fn main_body() -> mir::Body {
    mir::Body::new(
        decls(&[u32_ty(), u32_ty(), u32_ty()]),
        0,
        vec![
            block(vec![], call_const(CONST_A, 1, 1)),
            block(vec![], call_const(CONST_B, 2, 2)),
            block(
                vec![assign(
                    place(0),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((
                            mir::Operand::Copy(place(1)),
                            mir::Operand::Copy(place(2)),
                        )),
                    ),
                )],
                ret(),
            ),
        ],
    )
}

/// `fn main() -> u32 { _0 = CONST_A; return }`
fn main_using_only_a() -> mir::Body {
    mir::Body::new(
        decls(&[u32_ty()]),
        0,
        vec![
            block(vec![], call_const(CONST_A, 0, 1)),
            block(vec![], ret()),
        ],
    )
}

fn program_with(main: mir::Body, b_body: mir::Body) -> MirProgram {
    let mut prog = MirProgram::new();
    prog.add("main", main);
    prog.add_with_def_id(ANON_NAME, CONST_A, const_body(40));
    prog.add_with_def_id(ANON_NAME, CONST_B, b_body);
    // The driver's full name table spells both DefIds with the
    // shared display name, exactly as for real anonymous consts.
    prog.fn_names.insert(CONST_A, ANON_NAME.to_string());
    prog.fn_names.insert(CONST_B, ANON_NAME.to_string());
    prog
}

// ----- the naming contract ---------------------------------------------------

#[test]
fn two_anonymous_consts_share_a_name_but_evaluate_independently() {
    let prog = program_with(main_body(), const_body(2));

    let converted = convert_program(&prog).expect("same-named const bodies must convert");
    // Each body is stored under its own DefId-disambiguated key.
    assert!(
        converted.get("test_crate::_#DefId(0:10)").is_some(),
        "const A has its own key"
    );
    assert!(
        converted.get("test_crate::_#DefId(0:11)").is_some(),
        "const B has its own key"
    );
    assert!(
        converted.get(ANON_NAME).is_none(),
        "the shared display name itself keys no body"
    );

    // 40 + 2: had either reference resolved to the sibling's body,
    // the sum would be 80 or 4 instead.
    assert_eq!(
        interpret_mir(&prog, "main", vec![]).expect("main runs"),
        Value::Uint(42)
    );
}

#[test]
fn unconvertible_anonymous_const_is_isolated_under_its_own_key() {
    let prog = program_with(main_using_only_a(), broken_const_body());

    let converted = convert_program(&prog).expect("conversion is isolated per body");
    assert!(converted.get("test_crate::_#DefId(0:10)").is_some());
    let reason = converted
        .unconverted_reason("test_crate::_#DefId(0:11)")
        .expect("const B's failure is recorded under its own key");
    assert!(
        reason.contains("Dynamic"),
        "the recorded reason names the unconvertible type: {reason}"
    );

    assert_eq!(
        interpret_mir(&prog, "main", vec![]).expect("main only uses const A"),
        Value::Uint(40)
    );
}

#[test]
fn calling_an_unconvertible_anonymous_const_names_its_key() {
    let prog = program_with(main_body(), broken_const_body());

    let err = interpret_mir(&prog, "main", vec![]).expect_err("the call must fail");
    match err {
        MirRunError::Interp(InterpError::UnconvertedFunction { name, .. }) => {
            assert_eq!(name, "test_crate::_#DefId(0:11)");
        }
        other => panic!("expected UnconvertedFunction for const B, got {other}"),
    }
}

#[test]
fn same_name_and_same_def_id_stays_a_duplicate_error() {
    let mut prog = MirProgram::new();
    prog.add("main", main_body());
    prog.add_with_def_id(ANON_NAME, CONST_A, const_body(40));
    prog.add_with_def_id(ANON_NAME, CONST_A, const_body(2));

    let err = convert_program(&prog).expect_err("indistinguishable bodies must fail");
    assert!(
        matches!(err, ConvertError::DuplicateFunction { .. }),
        "expected DuplicateFunction, got {err}"
    );
}
