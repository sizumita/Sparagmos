//! Thread and mpsc model tests: the cooperative thread model and
//! the interpreter-held mpsc channel state — spawn captures a
//! callable without running it, a thread runs to completion when
//! the running thread joins it (or reaches another scheduling
//! point), a thread's panic is its join `Err`, channel values move
//! through the queue in FIFO order, and dropping the last handle
//! of a side disconnects the channel (the `ChannelHandleDrop`
//! glue). The programs are hand-built `sparagmos-mir` bodies
//! shaped after the driver's output for the corresponding Rust
//! sources, following the `refcell_tests` fixture style.

use std::collections::HashMap;

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram};
use sparagmos_interp::Value;
use sparagmos_mir as mir;

// ----- shared builders over sparagmos_mir -----------------------------------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}

fn unit_ty() -> mir::Ty {
    mir::Ty::Tuple(vec![])
}

fn u64_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U64)
}

fn decls(tys: &[mir::Ty]) -> Vec<mir::LocalDecl> {
    tys.iter().cloned().map(mir::LocalDecl::new).collect()
}

fn place(idx: u32) -> mir::Place {
    mir::Place::from_local(mir::Local(idx))
}

fn copy(idx: u32) -> mir::Operand {
    mir::Operand::Copy(place(idx))
}

fn move_(idx: u32) -> mir::Operand {
    mir::Operand::Move(place(idx))
}

fn const_i32(v: i32) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int {
            data: v as u32 as u128,
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

fn drop(place_: mir::Place, next: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Drop {
        place: place_,
        target: mir::BasicBlock(next),
        unwind: mir::UnwindAction::Continue,
        replace: false,
    }
}

fn fndef(def: mir::DefId, generic_args: Vec<mir::Ty>) -> mir::Operand {
    mir::Operand::constant(mir::Constant {
        literal: mir::ConstKind::Ty(mir::Ty::FnDef(def, generic_args)),
        user_ty: None,
    })
}

fn call(
    func: mir::Operand,
    args: Vec<mir::Operand>,
    dest: u32,
    next: u32,
) -> mir::TerminatorKind {
    mir::TerminatorKind::Call {
        func,
        args,
        destination: place(dest),
        target: Some(mir::BasicBlock(next)),
        unwind: mir::UnwindAction::Continue,
        call_source: mir::CallSource::Normal,
    }
}

fn ref_rvalue(idx: u32) -> mir::Rvalue {
    mir::Rvalue::Ref(
        mir::Region::Erased,
        mir::BorrowKind::Shared,
        place(idx),
    )
}

fn ref_of(p: mir::Place) -> mir::Rvalue {
    mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, p)
}

fn ref_ty(inner: mir::Ty) -> mir::Ty {
    mir::Ty::Ref(mir::Region::Erased, Box::new(inner), mir::Mutability::Not)
}

fn closure_ty(def: mir::DefId, params: Vec<mir::Ty>, upvars: Vec<mir::Ty>) -> mir::Ty {
    mir::Ty::Closure(
        def,
        vec![
            mir::Ty::Int(mir::IntTy::I8),
            mir::Ty::FnPtr(Box::new(mir::FnSig::new(
                vec![mir::Ty::Tuple(params)],
                i32_ty(),
            ))),
            mir::Ty::Tuple(upvars),
        ],
    )
}

fn closure_aggregate(def: mir::DefId, clo_ty: &mir::Ty, operands: Vec<mir::Operand>) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Closure {
            def,
            generic_args: match clo_ty {
                mir::Ty::Closure(_, args) => args.clone(),
                _ => unreachable!(),
            },
        }),
        operands,
    )
}

// ----- the items the tests name ----------------------------------------------

const SENDER_TY: mir::DefId = mir::DefId::new(0, 400);
const RECEIVER_TY: mir::DefId = mir::DefId::new(0, 401);
const JOINHANDLE_TY: mir::DefId = mir::DefId::new(0, 402);
const OPTION_TY: mir::DefId = mir::DefId::new(0, 403);
const RESULT_TY: mir::DefId = mir::DefId::new(0, 404);
const TRYRECVERROR_TY: mir::DefId = mir::DefId::new(0, 405);
const SCOPE_TY: mir::DefId = mir::DefId::new(0, 406);

const CHANNEL_FN: mir::DefId = mir::DefId::new(0, 410);
const SEND_FN: mir::DefId = mir::DefId::new(0, 411);
const RECV_FN: mir::DefId = mir::DefId::new(0, 412);
const TRY_RECV_FN: mir::DefId = mir::DefId::new(0, 413);
const SPAWN_FN: mir::DefId = mir::DefId::new(0, 414);
const JOIN_FN: mir::DefId = mir::DefId::new(0, 415);
const CLONE_FN: mir::DefId = mir::DefId::new(0, 416);
const OPTION_UNWRAP_FN: mir::DefId = mir::DefId::new(0, 417);
const SCOPE_FN: mir::DefId = mir::DefId::new(0, 418);

const CLO_VALUE: mir::DefId = mir::DefId::new(0, 420);
const CLO_PANIC: mir::DefId = mir::DefId::new(0, 421);
const CLO_SEND: mir::DefId = mir::DefId::new(0, 422);
const CLO_SCOPE: mir::DefId = mir::DefId::new(0, 423);

fn sender_ty() -> mir::Ty {
    mir::Ty::Adt(SENDER_TY, vec![i32_ty()])
}

fn receiver_ty() -> mir::Ty {
    mir::Ty::Adt(RECEIVER_TY, vec![i32_ty()])
}

fn handle_ty() -> mir::Ty {
    mir::Ty::Adt(JOINHANDLE_TY, vec![i32_ty()])
}

fn result_ty(ok: mir::Ty, err: mir::Ty) -> mir::Ty {
    mir::Ty::Adt(RESULT_TY, vec![ok, err])
}

fn std_names() -> HashMap<mir::DefId, String> {
    HashMap::from([
        (SENDER_TY, "std::sync::mpsc::Sender".to_string()),
        (RECEIVER_TY, "std::sync::mpsc::Receiver".to_string()),
        (JOINHANDLE_TY, "std::thread::JoinHandle".to_string()),
        (OPTION_TY, "std::option::Option".to_string()),
        (RESULT_TY, "std::result::Result".to_string()),
        (TRYRECVERROR_TY, "std::sync::mpsc::TryRecvError".to_string()),
        (SCOPE_TY, "std::thread::Scope".to_string()),
        (CHANNEL_FN, "std::sync::mpsc::channel".to_string()),
        (SEND_FN, "std::sync::mpsc::Sender::<T>::send".to_string()),
        (RECV_FN, "std::sync::mpsc::Receiver::<T>::recv".to_string()),
        (
            TRY_RECV_FN,
            "std::sync::mpsc::Receiver::<T>::try_recv".to_string(),
        ),
        (SPAWN_FN, "std::thread::spawn".to_string()),
        (JOIN_FN, "std::thread::JoinHandle::<T>::join".to_string()),
        (CLONE_FN, "std::clone::Clone::clone".to_string()),
        (
            OPTION_UNWRAP_FN,
            "std::option::Option::<T>::unwrap".to_string(),
        ),
        (SCOPE_FN, "std::thread::scope".to_string()),
        (CLO_VALUE, "main::{closure#0}".to_string()),
        (CLO_PANIC, "main::{closure#1}".to_string()),
        (CLO_SEND, "main::{closure#2}".to_string()),
        (CLO_SCOPE, "main::{closure#3}".to_string()),
    ])
}

fn program(main: mir::Body, closures: &[(mir::DefId, &str, mir::Body)]) -> MirProgram {
    let mut prog = MirProgram::new();
    prog.add("main", main);
    for (def, name, body) in closures {
        prog.add_with_def_id(*name, *def, body.clone());
    }
    prog.fn_names = std_names();
    prog
}

fn run(main: mir::Body, closures: &[(mir::DefId, &str, mir::Body)]) -> Value {
    interpret_mir(&program(main, closures), "main", vec![]).expect("run")
}

// ----- spawn / join -----------------------------------------------------------

#[test]
fn spawn_join_returns_the_closure_value() {
    // closure: captures 40, returns capture + 2.
    let clo_ty = closure_ty(CLO_VALUE, vec![], vec![i32_ty()]);
    let clo_body = mir::Body::new(
        decls(&[i32_ty(), clo_ty.clone(), i32_ty()]),
        1,
        vec![block(
            vec![
                assign(
                    place(2),
                    mir::Rvalue::Use(mir::Operand::Move(place(1).field(0, i32_ty()))),
                ),
                assign(
                    place(0),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((copy(2), const_i32(2))),
                    ),
                ),
            ],
            ret(),
        )],
    );
    // main: _1 = Closure(40); _2 = spawn(move _1); _0 = join(move _2)
    let main = mir::Body::new(
        decls(&[
            result_ty(i32_ty(), unit_ty()),
            clo_ty.clone(),
            handle_ty(),
        ]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    closure_aggregate(CLO_VALUE, &clo_ty, vec![const_i32(40)]),
                )],
                call(
                    fndef(SPAWN_FN, vec![clo_ty.clone(), i32_ty()]),
                    vec![move_(1)],
                    2,
                    1,
                ),
            ),
            block(
                vec![],
                call(fndef(JOIN_FN, vec![i32_ty()]), vec![move_(2)], 0, 2),
            ),
            block(vec![], ret()),
        ],
    );
    let value = run(main, &[(CLO_VALUE, "main::{closure#0}", clo_body)]);
    assert_eq!(
        value,
        Value::Enum {
            variant: 0,
            fields: vec![Value::Int(42)],
        }
    );
}

#[test]
fn spawned_thread_panic_becomes_the_join_error() {
    // closure: unwraps a None, panicking inside the thread.
    let clo_ty = closure_ty(CLO_PANIC, vec![], vec![]);
    let option_ty = mir::Ty::Adt(OPTION_TY, vec![i32_ty()]);
    let clo_body = mir::Body::new(
        decls(&[i32_ty(), clo_ty.clone(), option_ty.clone(), i32_ty()]),
        1,
        vec![
            block(
                vec![assign(
                    place(2),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Adt {
                            def: OPTION_TY,
                            variant_index: 0,
                            generic_args: vec![i32_ty()],
                            user_ty: None,
                            field_index: None,
                        }),
                        vec![],
                    ),
                )],
                call(
                    fndef(OPTION_UNWRAP_FN, vec![i32_ty()]),
                    vec![move_(2)],
                    3,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(move_(3)))], ret()),
        ],
    );
    let main = mir::Body::new(
        decls(&[
            result_ty(i32_ty(), unit_ty()),
            clo_ty.clone(),
            handle_ty(),
        ]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    closure_aggregate(CLO_PANIC, &clo_ty, vec![]),
                )],
                call(
                    fndef(SPAWN_FN, vec![clo_ty.clone(), i32_ty()]),
                    vec![move_(1)],
                    2,
                    1,
                ),
            ),
            block(
                vec![],
                call(fndef(JOIN_FN, vec![i32_ty()]), vec![move_(2)], 0, 2),
            ),
            block(vec![], ret()),
        ],
    );
    let value = run(main, &[(CLO_PANIC, "main::{closure#1}", clo_body)]);
    match value {
        Value::Enum { variant: 1, fields } => match &fields[0] {
            // The join error is the panic payload as a
            // `Box<dyn Any>`: a `DynRef` whose base is the
            // payload value (here the `&str` message).
            Value::DynRef { base, concrete, .. } => {
                assert_eq!(concrete, "&str");
                match &**base {
                    Value::Str(message) => {
                        assert!(message.contains("Option::unwrap"), "{message}")
                    }
                    other => panic!("expected the panic message string, got {other:?}"),
                }
            }
            other => panic!("expected the payload DynRef, got {other:?}"),
        },
        other => panic!("expected the join Err, got {other:?}"),
    }
}

#[test]
fn thread_runs_at_join_not_at_spawn() {
    // The child sends 7 on a channel whose sender it captured.
    // main: try_recv before join must see Empty (the child has not
    // run); after join, recv must see 7.
    let clo_ty = closure_ty(CLO_SEND, vec![], vec![sender_ty()]);
    let clo_body = mir::Body::new(
        decls(&[
            unit_ty(),
            clo_ty.clone(),
            ref_ty(sender_ty()),
            i32_ty(),
            result_ty(unit_ty(), unit_ty()),
        ]),
        1,
        vec![
            block(
                vec![
                    assign(
                        place(2),
                        ref_of(place(1).field(0, sender_ty())),
                    ),
                    assign(place(3), mir::Rvalue::Use(const_i32(7))),
                ],
                call(
                    fndef(SEND_FN, vec![i32_ty()]),
                    vec![copy(2), move_(3)],
                    4,
                    1,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let pair_ty = mir::Ty::Tuple(vec![sender_ty(), receiver_ty()]);
    let try_result_ty = result_ty(i32_ty(), mir::Ty::Adt(TRYRECVERROR_TY, vec![]));
    // main locals:
    // _0 (u64, i32) out, _1 channel pair, _2 tx, _3 rx, _4 closure,
    // _5 handle, _6 &rx, _7 try result, _8 join result,
    // _9 recv result, _10 try payload discr, _11 recv payload
    let main = mir::Body::new(
        decls(&[
            mir::Ty::Tuple(vec![u64_ty(), i32_ty()]),
            pair_ty,
            sender_ty(),
            receiver_ty(),
            clo_ty.clone(),
            handle_ty(),
            ref_ty(receiver_ty()),
            try_result_ty.clone(),
            result_ty(unit_ty(), unit_ty()),
            result_ty(i32_ty(), unit_ty()),
            u64_ty(),
            i32_ty(),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(CHANNEL_FN, vec![i32_ty()]), vec![], 1, 1),
            ),
            block(
                vec![
                    assign(
                        place(2),
                        mir::Rvalue::Use(mir::Operand::Move(place(1).field(0, sender_ty()))),
                    ),
                    assign(
                        place(3),
                        mir::Rvalue::Use(mir::Operand::Move(place(1).field(1, receiver_ty()))),
                    ),
                    assign(
                        place(4),
                        closure_aggregate(CLO_SEND, &clo_ty, vec![move_(2)]),
                    ),
                ],
                call(
                    fndef(SPAWN_FN, vec![clo_ty.clone(), unit_ty()]),
                    vec![move_(4)],
                    5,
                    2,
                ),
            ),
            block(
                vec![assign(place(6), ref_rvalue(3))],
                call(
                    fndef(TRY_RECV_FN, vec![i32_ty()]),
                    vec![copy(6)],
                    7,
                    3,
                ),
            ),
            block(
                vec![],
                call(fndef(JOIN_FN, vec![unit_ty()]), vec![move_(5)], 8, 4),
            ),
            block(
                vec![assign(place(6), ref_rvalue(3))],
                call(
                    fndef(RECV_FN, vec![i32_ty()]),
                    vec![copy(6)],
                    9,
                    5,
                ),
            ),
            block(
                vec![
                    assign(
                        place(10),
                        mir::Rvalue::Discriminant(place(7).field(0, try_result_ty)),
                    ),
                    assign(
                        place(11),
                        mir::Rvalue::Use(mir::Operand::Move(place(9).field(0, i32_ty()))),
                    ),
                    assign(
                        place(0),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Tuple),
                            vec![move_(10), move_(11)],
                        ),
                    ),
                ],
                ret(),
            ),
        ],
    );
    let value = run(main, &[(CLO_SEND, "main::{closure#2}", clo_body)]);
    // (TryRecvError::Empty = 0, received 7)
    assert_eq!(value, Value::Tuple(vec![Value::Uint(0), Value::Int(7)]));
}

#[test]
fn scope_returns_the_closure_value() {
    // scope closure: returns 5.
    let clo_ty = closure_ty(CLO_SCOPE, vec![ref_ty(mir::Ty::Adt(SCOPE_TY, vec![]))], vec![]);
    let clo_body = mir::Body::new(
        decls(&[
            i32_ty(),
            clo_ty.clone(),
            ref_ty(mir::Ty::Adt(SCOPE_TY, vec![])),
        ]),
        2,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_i32(5)))],
            ret(),
        )],
    );
    let main = mir::Body::new(
        decls(&[i32_ty(), clo_ty.clone()]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    closure_aggregate(CLO_SCOPE, &clo_ty, vec![]),
                )],
                call(
                    fndef(SCOPE_FN, vec![clo_ty.clone(), i32_ty()]),
                    vec![move_(1)],
                    0,
                    1,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let value = run(main, &[(CLO_SCOPE, "main::{closure#3}", clo_body)]);
    assert_eq!(value, Value::Int(5));
}

// ----- channels ---------------------------------------------------------------

#[test]
fn channel_send_recv_roundtrip_in_one_thread() {
    // main: (tx, rx) = channel(); tx.send(41); _0 = rx.recv()
    let pair_ty = mir::Ty::Tuple(vec![sender_ty(), receiver_ty()]);
    let main = mir::Body::new(
        decls(&[
            result_ty(i32_ty(), unit_ty()),
            pair_ty,
            sender_ty(),
            receiver_ty(),
            ref_ty(sender_ty()),
            i32_ty(),
            result_ty(unit_ty(), unit_ty()),
            ref_ty(receiver_ty()),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(CHANNEL_FN, vec![i32_ty()]), vec![], 1, 1),
            ),
            block(
                vec![
                    assign(
                        place(2),
                        mir::Rvalue::Use(mir::Operand::Move(place(1).field(0, sender_ty()))),
                    ),
                    assign(
                        place(3),
                        mir::Rvalue::Use(mir::Operand::Move(place(1).field(1, receiver_ty()))),
                    ),
                    assign(place(4), ref_rvalue(2)),
                    assign(place(5), mir::Rvalue::Use(const_i32(41))),
                ],
                call(
                    fndef(SEND_FN, vec![i32_ty()]),
                    vec![copy(4), move_(5)],
                    6,
                    2,
                ),
            ),
            block(
                vec![assign(place(7), ref_rvalue(3))],
                call(
                    fndef(RECV_FN, vec![i32_ty()]),
                    vec![copy(7)],
                    0,
                    3,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let value = run(main, &[]);
    assert_eq!(
        value,
        Value::Enum {
            variant: 0,
            fields: vec![Value::Int(41)],
        }
    );
}

#[test]
fn recv_disconnects_when_the_last_sender_drops() {
    // main: (tx, rx) = channel(); drop(tx); _0 = rx.recv() // Err
    let pair_ty = mir::Ty::Tuple(vec![sender_ty(), receiver_ty()]);
    let main = mir::Body::new(
        decls(&[
            result_ty(i32_ty(), unit_ty()),
            pair_ty,
            sender_ty(),
            receiver_ty(),
            ref_ty(receiver_ty()),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(CHANNEL_FN, vec![i32_ty()]), vec![], 1, 1),
            ),
            block(
                vec![
                    assign(
                        place(2),
                        mir::Rvalue::Use(mir::Operand::Move(place(1).field(0, sender_ty()))),
                    ),
                    assign(
                        place(3),
                        mir::Rvalue::Use(mir::Operand::Move(place(1).field(1, receiver_ty()))),
                    ),
                ],
                drop(place(2), 2),
            ),
            block(
                vec![assign(place(4), ref_rvalue(3))],
                call(
                    fndef(RECV_FN, vec![i32_ty()]),
                    vec![copy(4)],
                    0,
                    3,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let value = run(main, &[]);
    assert_eq!(
        value,
        Value::Enum {
            variant: 1,
            fields: vec![Value::Enum {
                variant: 0,
                fields: vec![],
            }],
        }
    );
}

#[test]
fn cloned_sender_keeps_the_channel_open() {
    // main: (tx, rx) = channel(); tx2 = tx.clone(); drop(tx);
    //       r1 = rx.try_recv() // Empty: tx2 still lives
    //       drop(tx2); r2 = rx.try_recv() // Disconnected
    // returns (discr of r1's payload, discr of r2's payload) = (0, 1)
    let pair_ty = mir::Ty::Tuple(vec![sender_ty(), receiver_ty()]);
    let try_result_ty = result_ty(i32_ty(), mir::Ty::Adt(TRYRECVERROR_TY, vec![]));
    let main = mir::Body::new(
        decls(&[
            mir::Ty::Tuple(vec![u64_ty(), u64_ty()]),
            pair_ty,
            sender_ty(),
            receiver_ty(),
            ref_ty(sender_ty()),
            sender_ty(),
            ref_ty(receiver_ty()),
            try_result_ty.clone(),
            try_result_ty.clone(),
            u64_ty(),
            u64_ty(),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(CHANNEL_FN, vec![i32_ty()]), vec![], 1, 1),
            ),
            block(
                vec![
                    assign(
                        place(2),
                        mir::Rvalue::Use(mir::Operand::Move(place(1).field(0, sender_ty()))),
                    ),
                    assign(
                        place(3),
                        mir::Rvalue::Use(mir::Operand::Move(place(1).field(1, receiver_ty()))),
                    ),
                    assign(place(4), ref_rvalue(2)),
                ],
                call(
                    fndef(CLONE_FN, vec![sender_ty()]),
                    vec![copy(4)],
                    5,
                    2,
                ),
            ),
            block(vec![], drop(place(2), 3)),
            block(
                vec![assign(place(6), ref_rvalue(3))],
                call(
                    fndef(TRY_RECV_FN, vec![i32_ty()]),
                    vec![copy(6)],
                    7,
                    4,
                ),
            ),
            block(vec![], drop(place(5), 5)),
            block(
                vec![assign(place(6), ref_rvalue(3))],
                call(
                    fndef(TRY_RECV_FN, vec![i32_ty()]),
                    vec![copy(6)],
                    8,
                    6,
                ),
            ),
            block(
                vec![
                    assign(
                        place(9),
                        mir::Rvalue::Discriminant(place(7).field(0, try_result_ty.clone())),
                    ),
                    assign(
                        place(10),
                        mir::Rvalue::Discriminant(place(8).field(0, try_result_ty)),
                    ),
                    assign(
                        place(0),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Tuple),
                            vec![move_(9), move_(10)],
                        ),
                    ),
                ],
                ret(),
            ),
        ],
    );
    let value = run(main, &[]);
    assert_eq!(value, Value::Tuple(vec![Value::Uint(0), Value::Uint(1)]));
}
