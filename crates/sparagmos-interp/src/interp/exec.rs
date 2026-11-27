//! The execution loop: block driving, statements and rvalues.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{RefProj, RefTarget, Value};

impl<'a> Interpreter<'a> {
    /// Run `entry` with `args` and return the value of its return place.
    pub fn run(&mut self, entry: &str, args: Vec<Value>) -> InterpResult<Value> {
        self.run_with_report(entry, args, |_, _, _| Ok(None))
    }

    /// Like [`Interpreter::run`], but when the entry returns
    /// normally, `report` is applied to the returned value before
    /// the entry thread's TLS destructors run -- the native
    /// order, in which `lang_start` applies a `Termination`
    /// report first and the thread's TLS teardown follows (and
    /// the only point at which a report rendering that itself
    /// touches TLS still sees that TLS alive). The text the
    /// report returns joins the guest's stderr at that point, so
    /// it precedes whatever the destructors print. A guest
    /// outcome the report itself produces -- a panic in a
    /// `Debug` body it executes, a `std::process::exit` --
    /// replaces the entry's result, exactly as it replaces the
    /// report's own outcome natively; the destructors still run,
    /// as they do for the entry's own outcomes.
    pub fn run_with_report(
        &mut self,
        entry: &str,
        args: Vec<Value>,
        report: impl FnOnce(&mut Self, &mut Vec<Frame>, &Value) -> InterpResult<Option<String>>,
    ) -> InterpResult<Value> {
        self.steps = 0;
        self.next_frame_id = 0;
        self.output.clear();
        self.error_output.clear();
        let body = self
            .program
            .get(entry)
            .ok_or_else(|| self.missing_body_error(entry))?;
        let args = self.pack_variadic_args(entry, body, args);
        if args.len() != body.arg_count {
            return Err(InterpError::type_mismatch(
                format!("call to `{entry}`"),
                format!("{} argument(s)", body.arg_count),
                format!("{} argument(s)", args.len()),
            ));
        }
        // The program's pre-main constructors run before the
        // entry point, in section order, as the platform runs
        // the initialiser sections before `main` (see
        // [`ir::Program::init_constructors`]).
        self.run_init_constructors()?;

        let mut locals = vec![Value::Uninit; body.local_count()];
        for (i, arg) in args.into_iter().enumerate() {
            locals[i + 1] = arg;
        }
        let entry_id = self.fresh_frame_id();
        self.entry_frame_id.set(Some(entry_id));
        self.register_frame_base(&[], entry_id, entry);
        let stack_bytes = self.charge_stack(&[], self.frame_stack_bytes(body))?;
        let mut frames: Vec<Frame> = vec![Frame {
            id: entry_id,
            func: entry.to_string(),
            locals,
            stack_bytes,
            block: 0,
            stmt: 0,
            return_dest: None,
            return_block: None,
            unwind_payload: None,
            in_cleanup: false,
            call_once_env_drop: None,
            incoming_asm: None,
        }];
        // The entry thread's scheduler loop: drive the entry
        // segment one quantum at a time; when its quantum expires
        // with other threads runnable, give each of them a
        // quantum on this stack (the entry's frames stay resident
        // below theirs, so every frame a spawned segment may name
        // is live) and resume the entry segment. With no other
        // thread runnable the drive never yields and this is a
        // single drive, exactly as before.
        let result = loop {
            let previous_segment = self.segment.replace(SegmentMark {
                depth: self.drive_depth + 1,
                thread: 1,
                steps0: self.steps,
            });
            let driven = self.drive_frames(&mut frames, 0);
            self.segment = previous_segment;
            match driven {
                Err(InterpError::Yield) => {
                    self.drain_threads(&mut frames)?;
                    continue;
                }
                other => break other,
            }
        };
        // An uncaught panic out of the entry is `lang_start`'s
        // catch: the catch consumes the panic (a panic the
        // payload's disposal raises is a fresh one, counted
        // and reported as its own), and before the report, the
        // exit handlers or the TLS teardown run, the payload
        // is disposed of (see `dispose_uncaught_payload`) --
        // natively that drop precedes all of them, and its
        // own panic aborts the run in place of the original
        // panic's exit.
        let result = match result {
            Err(InterpError::Panic { message, payload }) => {
                self.note_panic_consumed();
                let outcome = match &payload {
                    Some(payload) => {
                        self.dispose_uncaught_payload(&mut frames, payload)
                    }
                    None => Ok(()),
                };
                match outcome {
                    Ok(()) => Err(InterpError::Panic { message, payload }),
                    Err(e) => Err(e),
                }
            }
            other => other,
        };
        // The termination report applies here: after the entry
        // has returned, before its thread's TLS teardown (see
        // [`Interpreter::run_with_report`]).
        let result = match result {
            Ok(value) => match report(self, &mut frames, &value) {
                Ok(Some(text)) => {
                    self.push_error_output(&text);
                    Ok(value)
                }
                Ok(None) => Ok(value),
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        };
        // Program exit first runs the guest's registered
        // `atexit` handlers (see `run_exit_handlers`): the
        // platform runs them as `main` returns, ahead of the
        // thread's TLS teardown. A handler's failure replaces
        // the entry's outcome, as a destructor's does below.
        let result = match self.run_exit_handlers(&mut frames) {
            Ok(()) => result,
            Err(e) => Err(e),
        };
        // Program exit runs the entry thread's registered TLS
        // destructors (the registrations wrap
        // `__cxa_thread_atexit`, whose entries run at exit),
        // whatever the entry's outcome was.
        let result = match self.run_tls_dtors(&mut frames, 1) {
            Ok(()) => result,
            Err(InterpError::Panic { .. }) => {
                // A panic in an entry-thread TLS destructor: as
                // on any thread, std reports it as a fatal
                // runtime error and aborts (see `run_thread`).
                self.push_error_output(
                    "fatal runtime error: thread local panicked on drop, aborting\n",
                );
                Err(InterpError::Abort {
                    message: "thread local panicked on drop".to_string(),
                })
            }
            Err(e) => Err(e),
        };
        self.entry_frame_id.set(None);
        result
    }

    /// Drive the frame stack until the frame at depth `base`
    /// returns, yielding its return value. The caller's stack is
    /// always restored to the depth this drive started from: on
    /// success the loop stops exactly there; on an error — above
    /// all a guest panic, which a thread's join must observe as
    /// the thread's outcome rather than as a failure of the whole
    /// run — the frames above the base are dead (a panic has
    /// already unwound them, running their cleanups; see
    /// [`Interpreter::drive`]) and are discarded.
    pub(crate) fn drive_frames(&mut self, frames_out: &mut Vec<Frame>, base: usize) -> InterpResult<Value> {
        let mut frames: Vec<Frame> = std::mem::take(frames_out);
        let result = self.drive(&mut frames, base);
        // A scheduler yield leaves the driven segment's frames
        // live above `base`: the segment's driver parks them and
        // the segment later resumes from exactly this stack. A
        // suspension ([`InterpError::Suspended`]) leaves them
        // whole for the same reason: they are the suspended
        // thread's live state, which `run_thread` splits off and
        // saves. Every other outcome (a value, a panic, a hard
        // error) ends the region, so its frames are truncated
        // away as documented above.
        if !matches!(
            result,
            Err(InterpError::Yield) | Err(InterpError::Suspended)
        ) {
            frames.truncate(base);
        }
        *frames_out = frames;
        result
    }

    /// Put back the argument values a call's `move` operands
    /// consumed out of plain locals, after the call ended in a
    /// suspension (see [`InterpError::Suspended`]): the resumed
    /// thread re-executes the call terminator, so its arguments
    /// must be in their places again, exactly as if the call
    /// had never been entered — a `move` leaves its source
    /// place uninitialized, and the blocking builtins' decisive
    /// argument (the `&self` temporary of a channel operation
    /// above all) is such a move. Only plain-local moves are
    /// restored; a move out of a projection stays consumed. The
    /// restore applies only when the caller frame is again the
    /// top frame: a call that pushed a callee frame which then
    /// suspended leaves that frame on the stack, and the
    /// caller's terminator is not re-executed.
    pub(crate) fn restore_moved_args(
        frames: &mut Vec<Frame>,
        caller_id: u64,
        moved: Vec<(usize, Value)>,
    ) {
        if frames.last().map(|f| f.id) != Some(caller_id) {
            return;
        }
        let frame = frames.last_mut().expect("frame");
        for (local, value) in moved {
            if let Some(slot) = frame.locals.get_mut(local) {
                *slot = value;
            }
        }
    }

    /// Drive the frame stack with unwinding: run the raw loop
    /// ([`Interpreter::drive_loop`]); when it reports a guest
    /// panic, unwind the stack frame by frame down to `stop_depth`
    /// before reporting the panic. Each frame the panic passes
    /// through runs the cleanup its current block prescribes (see
    /// [`crate::ir::Body::unwind_action`]): a cleanup block chain
    /// (the `Drop` terminators of the locals live at the panic
    /// point, in MIR's own order) runs with the panic stashed in
    /// the frame, and its closing `Resume` pops the frame and
    /// continues the unwind in the caller at its call site. A
    /// frame with no cleanup is simply discarded. A panic that may
    /// not unwind — one raised directly in a cleanup frame, or one
    /// whose block prescribes termination — aborts the run
    /// ([`InterpError::Abort`]), as a native double panic does.
    /// Non-panic errors never unwind: they are oracle failures,
    /// not guest control flow.
    pub(crate) fn drive(
        &mut self,
        frames: &mut Vec<Frame>,
        stop_depth: usize,
    ) -> InterpResult<Value> {
        // Every drive region counts toward the depth the
        // scheduler's segment marks name, whichever entry built
        // it (`drive_frames`, `call_to_completion`).
        self.drive_depth += 1;
        let result = self.drive_inner(frames, stop_depth);
        self.drive_depth -= 1;
        result
    }

    /// Run the program's pre-main constructors once, in the
    /// recorded order: the function bodies the initialiser
    /// sections (`.init_array` and its platform spellings)
    /// name, which the platform calls before `main`. Each
    /// constructor is called to completion on an empty frame
    /// stack with the platform's argument triple -- `argc`,
    /// `argv`, `envp` -- fabricated from the run's own
    /// arguments and environment, truncated to the body's
    /// declared argument count (a constructor declaring fewer
    /// parameters is unaffected by the rest, as under the C
    /// ABI). A constructor whose body the program does not
    /// hold is skipped, the policy the exit handlers follow.
    /// A panic may not unwind out of an `extern "C"`
    /// constructor -- natively the process aborts, its report
    /// already printed by the panic machinery -- so a
    /// constructor's panic becomes the run's abort.
    fn run_init_constructors(&mut self) -> InterpResult<()> {
        if self.program.init_constructors.is_empty() {
            return Ok(());
        }
        let ctors = self.program.init_constructors.clone();
        let mut frames: Vec<Frame> = Vec::new();
        for name in &ctors {
            let Some(body) = self.program.get(name) else {
                continue;
            };
            let args = self.init_ctor_args(body.arg_count);
            let callable = Value::FnPtr(name.clone());
            match self.call_callable_to_completion(&mut frames, &callable, args) {
                Ok(_) => {}
                Err(InterpError::Panic { message, .. }) => {
                    return Err(InterpError::Abort {
                        message: format!(
                            "pre-main constructor `{name}` panicked: {message}"
                        ),
                    });
                }
                Err(err) => return Err(err),
            }
        }
        Ok(())
    }

    /// The argument vector of one pre-main constructor call:
    /// the platform's `(argc, argv, envp)` triple, truncated
    /// to the constructor's declared argument count and
    /// padded with null pointers past the triple (a
    /// constructor declaring more parameters reads past the
    /// platform's own arguments, which the platform leaves
    /// undefined). `argv` and `envp` are fabricated C string
    /// arrays -- the run's program path and guest arguments,
    /// and the inherited environment the `env` builtins
    /// themselves read -- held in fresh allocations the
    /// constructor reads through ordinary memory operations.
    fn init_ctor_args(&mut self, arg_count: usize) -> Vec<Value> {
        if arg_count == 0 {
            return Vec::new();
        }
        let mut argv_strings = vec![self.program_path().as_bytes().to_vec()];
        argv_strings.extend(
            self.guest_args()
                .iter()
                .map(|arg| arg.as_bytes().to_vec()),
        );
        let argc = argv_strings.len() as i128;
        let argv = self.c_string_array(argv_strings);
        let env_strings: Vec<Vec<u8>> = std::env::vars_os()
            .map(|(key, value)| {
                let mut entry = os_str_bytes(&key);
                entry.push(b'=');
                entry.extend_from_slice(&os_str_bytes(&value));
                entry
            })
            .collect();
        let envp = self.c_string_array(env_strings);
        let mut args = vec![Value::Int(argc), argv, envp];
        args.truncate(arg_count);
        while args.len() < arg_count {
            args.push(Value::Ref(RefTarget::Null));
        }
        args
    }

    /// Allocate a NUL-terminated array of pointers to
    /// NUL-terminated byte strings, the shape of `argv` and
    /// `envp` at process start, and return a pointer to it.
    fn c_string_array(&mut self, strings: Vec<Vec<u8>>) -> Value {
        let mut pointers: Vec<Value> = strings
            .into_iter()
            .map(|mut bytes| {
                bytes.push(0);
                let id = self.alloc_insert(Value::Bytes(bytes));
                Value::Ref(RefTarget::Alloc {
                    id,
                    projection: Vec::new(),
                })
            })
            .collect();
        pointers.push(Value::Ref(RefTarget::Null));
        let id = self.alloc_insert(Value::Tuple(pointers));
        Value::Ref(RefTarget::Alloc {
            id,
            projection: Vec::new(),
        })
    }

    fn drive_inner(
        &mut self,
        frames: &mut Vec<Frame>,
        stop_depth: usize,
    ) -> InterpResult<Value> {
        let program: &'a Program = self.program;
        loop {
            match self.drive_loop(frames, stop_depth) {
                Ok(value) => return Ok(value),
                Err(e) if !matches!(e, InterpError::Panic { .. }) => {
                    return Err(e);
                }
                Err(InterpError::Panic { message, payload }) => {
                    let resumed = std::mem::take(&mut self.resume_signaled);
                    if !resumed {
                        if self.always_abort.get() {
                            // `panic::always_abort` is in
                            // effect: the panic prints its abort
                            // line and the run aborts, without
                            // the hook running (std's
                            // MustAbort::AlwaysAbort). The
                            // payload prints through its Display,
                            // which is the panic's message.
                            if !self.panic_report_suppressed.get() {
                                let out = match self.panic_location_in(frames) {
                                    Some((file, line, col)) => format!(
                                        "aborting due to panic at {file}:{line}:{col}:\n{message}\n"
                                    ),
                                    None => {
                                        format!("aborting due to panic:\n{message}\n")
                                    }
                                };
                                self.push_error_output(&out);
                            }
                            return Err(InterpError::Abort {
                                message: format!(
                                    "panic `{message}` after `panic::always_abort`: \
                                     the run aborts"
                                ),
                            });
                        }
                        self.fire_panic_hook(frames, stop_depth, &message, &payload);
                        if self.hook_abort.get() {
                            return Err(InterpError::Abort {
                                message:
                                    "a panic raised while the panic hook was running \
                                     aborts the run"
                                        .to_string(),
                            });
                        }
                    }
                    let mut panic = InterpError::Panic { message, payload };
                    // Whether the panic was raised directly in the
                    // top frame (as opposed to resuming into it from
                    // a callee this unwinder already popped).
                    let mut direct = true;
                    // Unwind one frame at a time until the panic
                    // leaves this drive region or a cleanup block
                    // takes over execution.
                    loop {
                        if frames.len() <= stop_depth {
                            return Err(panic);
                        }
                        let (func, block, in_cleanup) = {
                            let frame = frames.last().expect("frame");
                            (frame.func.clone(), frame.block, frame.in_cleanup)
                        };
                        if direct && in_cleanup && !resumed {
                            // A fresh panic raised by the cleanup
                            // frame itself (not one resuming into
                            // it from a callee): panicking while
                            // unwinding aborts, natively. The
                            // fresh panic's own report has just
                            // been printed by the hook; std then
                            // raises `panic_in_cleanup` (a fixed
                            // location in core — the
                            // `panic_nounwind_nobacktrace` call
                            // at core/src/panicking.rs:233:5 in
                            // the pinned 1.100.0 build — a `&str`
                            // payload, no backtrace), whose
                            // report and the non-unwinding line
                            // precede the abort.
                            let InterpError::Panic { message, .. } = &panic else {
                                unreachable!("panic is a Panic")
                            };
                            self.fire_nounwind_abort_panic(
                                frames,
                                (
                                    "/rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/core/src/panicking.rs"
                                        .to_string(),
                                    233,
                                    5,
                                ),
                                "panic in a destructor during cleanup",
                                true,
                            );
                            return Err(InterpError::Abort {
                                message: format!(
                                    "panic `{message}` raised in a cleanup of `{func}`: \
                                     panicking while unwinding aborts the run"
                                ),
                            });
                        }
                        let body = program.get(&func);
                        let action = body
                            .map(|b| b.unwind_action(block))
                            .unwrap_or(crate::ir::UnwindAction::Continue);
                        match action {
                            crate::ir::UnwindAction::Cleanup(target) => {
                                let Some(body) = body else {
                                    unreachable!("cleanup action implies a body")
                                };
                                let InterpError::Panic { message, payload } = panic
                                else {
                                    unreachable!("panic is a Panic")
                                };
                                let frame = frames.last_mut().expect("frame");
                                frame.unwind_payload = Some((message, payload));
                                frame.in_cleanup = true;
                                // The frame's own cleanup blocks own
                                // its drops now: a `call_once` shim
                                // environment drop recorded for this
                                // frame must not run as well.
                                frame.call_once_env_drop = None;
                                Self::jump(frames, body, target)?;
                                break;
                            }
                            crate::ir::UnwindAction::Continue => {
                                // A destructor frame unwound out of
                                // with glue still suspended under it:
                                // the remaining glue is the cleanup
                                // of the native `drop_in_place` whose
                                // destructor panicked, so it runs
                                // before the panic moves on (the
                                // destructor frame itself is popped
                                // first; the glue works on the owner
                                // frame below it).
                                let cont_pos = frames.last().and_then(|top| {
                                    self.drop_continuations
                                        .iter()
                                        .rposition(|c| c.callee_frame == top.id)
                                });
                                let env_drop = frames
                                    .last()
                                    .and_then(|top| top.call_once_env_drop.clone());
                                if let Some((root, glue)) = env_drop {
                                    // A `call_once` shim environment
                                    // drop owed for the frame being
                                    // unwound out of: the shim drops
                                    // the receiver value as the panic
                                    // passes through the call, before
                                    // the caller's call-site action
                                    // decides (see
                                    // `Frame::call_once_env_drop`).
                                    let callee_id =
                                        frames.last().expect("frame").id;
                                    self.pop_unwind_frame(frames);
                                    let cont = DropContinuation {
                                        callee_frame: callee_id,
                                        session: GlueSession {
                                            root,
                                            work: vec![(
                                                Vec::new(),
                                                glue.root.clone(),
                                            )],
                                            adts: glue.adts.clone(),
                                            completion:
                                                GlueCompletion::UnwindCleanup,
                                        },
                                        sessions: Vec::new(),
                                    };
                                    match self.run_glue_cleanup(frames, cont, panic)? {
                                        GlueCleanupFlow::Completed(p) => {
                                            panic = p;
                                        }
                                        GlueCleanupFlow::Suspended => break,
                                    }
                                } else if let Some(pos) = cont_pos {
                                    let cont = self.drop_continuations.remove(pos);
                                    self.pop_unwind_frame(frames);
                                    match self.run_glue_cleanup(frames, cont, panic)? {
                                        GlueCleanupFlow::Completed(p) => {
                                            panic = p;
                                        }
                                        GlueCleanupFlow::Suspended => break,
                                    }
                                } else {
                                    self.pop_unwind_frame(frames);
                                }
                                // The caller frame is now on top and
                                // the panic resumes into it: that is
                                // a resumption, not a fresh panic in
                                // the caller, even when the caller is
                                // itself running cleanups, so the
                                // direct-panic guard no longer
                                // applies; the caller's own call-site
                                // action decides.
                                direct = false;
                                continue;
                            }
                            crate::ir::UnwindAction::Terminate(_)
                            | crate::ir::UnwindAction::Unreachable => {
                                let InterpError::Panic { message, .. } = &panic else {
                                    unreachable!("panic is a Panic")
                                };
                                // Which of core's nounwind abort
                                // panics codegen raises here is a
                                // property of the block (the MIR
                                // `UnwindTerminateReason`): a
                                // `Terminate` in a cleanup block is
                                // a panic escaping the cleanup of
                                // an earlier panic, answered by
                                // `panic_in_cleanup` — the
                                // `panic_nounwind_nobacktrace` call
                                // at core/src/panicking.rs:233:5 in
                                // the pinned 1.100.0 build, a `&str`
                                // payload with no backtrace — while
                                // a `Terminate` in an ordinary block
                                // is a panic crossing a nounwind
                                // boundary, answered by
                                // `panic_cannot_unwind` — the
                                // `panic_nounwind_fmt` call inside
                                // `panic_nounwind` at
                                // core/src/panicking.rs:225:5, a
                                // `&str` payload whose report may
                                // carry a backtrace. In both cases
                                // the abort panic's report and the
                                // non-unwinding line precede the
                                // abort. Under `-C panic=abort` the
                                // panic's own report is all that
                                // prints: the strategy aborts in
                                // the personality, with no second
                                // panic.
                                let in_cleanup = matches!(
                                    action,
                                    crate::ir::UnwindAction::Terminate(
                                        crate::ir::UnwindTerminateReason::InCleanup
                                    )
                                );
                                if !self.config.panic_abort {
                                    if in_cleanup {
                                        self.fire_nounwind_abort_panic(
                                            frames,
                                            (
                                                "/rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/core/src/panicking.rs"
                                                    .to_string(),
                                                233,
                                                5,
                                            ),
                                            "panic in a destructor during cleanup",
                                            true,
                                        );
                                    } else {
                                        self.fire_nounwind_abort_panic(
                                            frames,
                                            (
                                                "/rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/core/src/panicking.rs"
                                                    .to_string(),
                                                225,
                                                5,
                                            ),
                                            "panic in a function that cannot unwind",
                                            false,
                                        );
                                    }
                                }
                                return Err(InterpError::Abort {
                                    message: if in_cleanup {
                                        format!(
                                            "panic `{message}` raised in a cleanup of `{func}`: \
                                             panicking while unwinding aborts the run"
                                        )
                                    } else {
                                        format!(
                                            "panic `{message}` may not unwind past `{func}`: \
                                             the run aborts"
                                        )
                                    },
                                });
                            }
                        }
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Pop one frame during unwinding, discarding any suspended
    /// drop glue that belonged to it. The unwinder runs a popped
    /// destructor frame's glue as the panic's cleanup before
    /// calling this (see `drive`); what remains here is glue whose
    /// session can no longer run (its destructors never resume:
    /// the frame they would have resumed into is gone).
    fn pop_unwind_frame(&mut self, frames: &mut Vec<Frame>) {
        let Some(frame) = frames.pop() else {
            return;
        };
        self.drop_continuations
            .retain(|cont| cont.callee_frame != frame.id);
    }

    /// The switch bits of a discriminant held in its byte form
    /// (a scalar read out of byte-form storage, which
    /// `Value::switch_bits` cannot read on its own: the byte
    /// string carries neither its signedness nor its width --
    /// it may be the scalar's whole storage or an allocation
    /// snapshot from the scalar on). The discriminant place's
    /// own type supplies both: the scalar is the string's
    /// first `width` bytes, read unsigned for an unsigned type
    /// (bool included; char is a `Uint` in this IR) and
    /// sign-extended from the width for a signed type --
    /// exactly the canonical forms conversion normalises the
    /// switch targets to. `None` when the operand names no
    /// typed place or the type is no switch scalar, keeping
    /// the caller's mismatch verdict.
    fn switch_bits_of_byte_form(
        &self,
        body: &Body,
        discr: &Operand,
        value: &Value,
    ) -> Option<u128> {
        if !matches!(value, Value::Bytes(_)) {
            return None;
        }
        let place = match discr {
            Operand::Copy(place) | Operand::Move(place) => place,
            Operand::Constant(_) => return None,
        };
        match self.place_result_ty(body, place)? {
            Ty::Uint(ut) => value.bytes_prefix_as_uint(ut.bits() as usize / 8),
            Ty::Bool => value.bytes_prefix_as_uint(1),
            Ty::Int(it) => value
                .bytes_prefix_as_int(it.bits() as usize / 8)
                .map(|v| v as u128),
            _ => None,
        }
    }

    /// The run loop: execute until the frame stack shrinks to
    /// `stop_depth`, then return the value the frame that was on
    /// top returned. The top-level [`Interpreter::run`] drives with
    /// `stop_depth == 0` (the entry frame's return ends the run);
    /// runtime support that calls back into guest code (the
    /// `catch_unwind` intrinsic) drives a nested region with the
    /// current depth as its stop, so the nested call's return value
    /// comes back here instead of being delivered to a caller
    /// place. Frames pushed for a nested region carry no return
    /// destination (see `start_detached_call`); stack discipline
    /// guarantees such a frame returns exactly when the stack is
    /// back at `stop_depth + 1`.
    fn drive_loop(
        &mut self,
        mut frames: &mut Vec<Frame>,
        stop_depth: usize,
    ) -> InterpResult<Value> {
        // The current frame's body, borrowed from the program for
        // the whole step: bodies never change during a run, so the
        // loop resolves the top frame's body once per frame switch
        // (keyed by the frame's unique id; a frame's function never
        // changes over its life) and borrows the current statement
        // and terminator from it, instead of cloning the body — and
        // with it every statement, type and operand it holds — on
        // every step.
        let program: &'a Program = self.program;
        let mut cached: Option<(u64, &Body, std::rc::Rc<crate::decode::DecodedBody>)> =
            None;
        let mut func_name = String::new();
        'drive: loop {
            if self.yield_due(frames, stop_depth) {
                return Err(InterpError::Yield);
            }
            let (block_idx, stmt_idx) = {
                let frame = frames
                    .last()
                    .expect("frame stack is non-empty while running");
                if cached.as_ref().map(|(id, ..)| *id) != Some(frame.id) {
                    func_name = frame.func.clone();
                    let resolved = program.get(&func_name).ok_or_else(|| {
                        InterpError::UnknownFunction {
                            name: func_name.clone(),
                        }
                    })?;
                    let decoded = self.decoded_body(resolved);
                    cached = Some((frame.id, resolved, decoded));
                }
                (frame.block, frame.stmt)
            };
            let body = cached
                .as_ref()
                .expect("current frame body resolved")
                .1;
            let block = body
                .blocks
                .get(block_idx)
                .ok_or_else(|| InterpError::UnknownBlock {
                    func: func_name.clone(),
                    block: block_idx,
                })?;
            // The decoded mirror of `block` (same block and
            // statement layout by construction).
            let dblock = &cached
                .as_ref()
                .expect("current frame body resolved")
                .2
                .blocks[block_idx];

            if self.debug_ctl.is_some() {
                crate::debug::drive_hook(self, frames, body, block_idx, stmt_idx);
            }
            if stmt_idx < block.statements.len() {
                let dstmt = &dblock.stmts[stmt_idx];
                // Advance the program counter before executing, so a call
                // (not possible in a statement) or an error leaves sane state.
                frames.last_mut().expect("frame").stmt += 1;
                self.tick()?;
                self.exec_dstmt(frames, body, dstmt)?;
                continue;
            }

            // Terminator.
            let terminator = &block.terminator;
            self.tick()?;
            match terminator {
                TerminatorKind::Goto { target } => {
                    Self::jump(frames, body, *target)?;
                }
                TerminatorKind::SwitchInt {
                    discr,
                    targets,
                    otherwise,
                } => {
                    let value = self.eval_operand(frames, body, discr)?;
                    let bits = match value.switch_bits() {
                        Some(bits) => bits,
                        None => self
                            .switch_bits_of_byte_form(body, discr, &value)
                            .ok_or_else(|| {
                                InterpError::type_mismatch(
                                    "SwitchInt discriminant",
                                    "integer or bool",
                                    value.kind_name(),
                                )
                            })?,
                    };
                    let target = match &dblock.switch {
                        // Sorted table with all-unique patterns:
                        // binary search finds the same target the
                        // linear scan would.
                        Some(table) => table
                            .binary_search_by_key(&bits, |(pattern, _)| *pattern)
                            .map(|idx| table[idx].1)
                            .unwrap_or(*otherwise),
                        None => targets
                            .iter()
                            .find(|(pattern, _)| *pattern == bits)
                            .map(|(_, bb)| *bb)
                            .unwrap_or(*otherwise),
                    };
                    Self::jump(frames, body, target)?;
                }
                TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    target,
                } => {
                    let caller_id = frames.last().expect("frame").id;
                    let mut arg_values = Vec::with_capacity(args.len());
                    for arg in args {
                        arg_values.push(self.eval_operand(frames, body, arg)?);
                    }
                    // The values the call's `move` operands
                    // consumed out of plain locals, snapshotted
                    // before the call: if the call ends in a
                    // suspension, they are put back (see
                    // `restore_moved_args`).
                    let moved_args: Vec<(usize, Value)> = args
                        .iter()
                        .zip(arg_values.iter())
                        .filter_map(|(arg, value)| match arg {
                            Operand::Move(place) if place.projection.is_empty() => {
                                Some((place.local.0, value.clone()))
                            }
                            _ => None,
                        })
                        .collect();
                    // A call naming a trait method by its plain
                    // trait path, in a body the converter kept in
                    // generic form, has no body under that name:
                    // when it is not a vtable dispatch, the
                    // builtin of the same name may serve it (see
                    // `builtin_name_fallback`). The vtable check
                    // comes first, exactly as in `start_call`. A
                    // vtable dispatch whose concrete impl body is
                    // not in the program may still be served, by
                    // the file-io model, when the method is one
                    // it serves for the receiver's concrete type
                    // (see `dyn_file_io_target`).
                    // A by-value `Self` vtable call reaches here
                    // with the receiver already loaded out of the
                    // fat pointer it was called through (see
                    // `recover_deref_dyn_receiver`): restore the
                    // fat pointer so the vtable check below can
                    // dispatch through it.
                    self.recover_deref_dyn_receiver(
                        frames, body, func, args, &mut arg_values,
                    );
                    let vt = self.virtual_call_target(func, &arg_values);
                    let fallback = if vt.is_none() {
                        self.builtin_name_fallback(func)
                    } else {
                        None
                    };
                    let dyn_io = match &vt {
                        Some(Err(InterpError::UnsupportedDynamicDispatch { .. })) => {
                            self.dyn_file_io_target(func, &arg_values)
                        }
                        _ => None,
                    };
                    let outcome = if let Some((name, generic_args, receiver)) = dyn_io {
                        self.complete_dyn_file_io(
                            frames,
                            body,
                            &name,
                            &generic_args,
                            receiver,
                            arg_values,
                            destination,
                            *target,
                        )
                    } else if let Some((canonical, kind, generic_args)) = fallback {
                        self.complete_builtin_name_fallback(
                            frames,
                            body,
                            &canonical,
                            kind,
                            &generic_args,
                            func,
                            arg_values,
                            destination,
                            *target,
                        )
                    } else {
                        self.start_call(frames, func, arg_values, destination, *target)
                    };
                    if matches!(outcome, Err(InterpError::Suspended)) {
                        Self::restore_moved_args(frames, caller_id, moved_args);
                    }
                    outcome?;
                }
                TerminatorKind::CallIndirect {
                    func,
                    args,
                    destination,
                    target,
                } => {
                    // The callee value comes from a place: resolve it
                    // together with the environment reference a
                    // closure body expects (see `eval_callable`). A
                    // body that takes its environment by value (an
                    // `FnOnce` closure's own body) gets the value
                    // itself instead (see `consumed_env_arg`).
                    let (callee, env_ref) = self.eval_callable(&frames, body, func)?;
                    let env_ref = match self.consumed_env_arg(frames, body, func, &callee)? {
                        Some(by_value) => by_value,
                        None => env_ref,
                    };
                    let mut arg_values = Vec::with_capacity(args.len());
                    for arg in args {
                        arg_values.push(self.eval_operand(frames, body, arg)?);
                    }
                    match callee {
                        Value::FnPtr(name) => {
                            // A function value naming a trait
                            // method by its plain trait path has
                            // no body under that name; the builtin
                            // of the same name may serve it (see
                            // `builtin_name_fallback`). A vtable
                            // dispatch with no impl body in the
                            // program may be served by the file-io
                            // model instead (see
                            // `dyn_file_io_target`).
                            // As in the direct-call arm: a
                            // by-value `Self` vtable call through
                            // a function value may reach here with
                            // the receiver already loaded out of
                            // its fat pointer (see
                            // `recover_deref_dyn_receiver`).
                            self.recover_deref_dyn_receiver(
                                frames, body, &name, args, &mut arg_values,
                            );
                            let vt = self.virtual_call_target(&name, &arg_values);
                            let fallback = if vt.is_none() {
                                self.builtin_name_fallback(&name)
                            } else {
                                None
                            };
                            let dyn_io = match &vt {
                                Some(Err(
                                    InterpError::UnsupportedDynamicDispatch { .. },
                                )) => self.dyn_file_io_target(&name, &arg_values),
                                _ => None,
                            };
                            if let Some((io_name, generic_args, receiver)) = dyn_io {
                                self.complete_dyn_file_io(
                                    frames,
                                    body,
                                    &io_name,
                                    &generic_args,
                                    receiver,
                                    arg_values,
                                    destination,
                                    *target,
                                )?;
                            } else if let Some((canonical, kind, generic_args)) = fallback {
                                self.complete_builtin_name_fallback(
                                    frames,
                                    body,
                                    &canonical,
                                    kind,
                                    &generic_args,
                                    &name,
                                    arg_values,
                                    destination,
                                    *target,
                                )?;
                            } else {
                                self.start_call(
                                    frames,
                                    &name,
                                    arg_values,
                                    destination,
                                    *target,
                                )?;
                            }
                        }
                        Value::Closure { def, .. } => {
                            let mut full = vec![env_ref];
                            full.extend(arg_values);
                            self.start_call(frames, &def, full, destination, *target)?;
                        }
                        other => {
                            return Err(InterpError::type_mismatch(
                                "indirect call",
                                "callable value",
                                other.kind_name(),
                            ));
                        }
                    }
                }
                TerminatorKind::CallTrait {
                    kind,
                    receiver,
                    args,
                    stamp,
                    receiver_glue,
                    async_captures,
                    destination,
                    target,
                } => {
                    // An `Fn`-family call whose receiver is an
                    // async closure value is served as the
                    // corresponding `AsyncFn` call below: the
                    // closure's own body is not in the program
                    // (it mentions the coroutine-closure type,
                    // which Stable MIR cannot convert), and the
                    // call's meaning is the `AsyncFn` one - it
                    // constructs the closure's coroutine from
                    // the call arguments and the environment.
                    let async_kind;
                    let kind = match kind {
                        FnCallKind::CallOnce
                            if self.fn_call_receiver_is_async_closure(
                                frames, body, receiver, true,
                            ) =>
                        {
                            async_kind = FnCallKind::AsyncCallOnce;
                            &async_kind
                        }
                        FnCallKind::CallMut
                            if self.fn_call_receiver_is_async_closure(
                                frames, body, receiver, false,
                            ) =>
                        {
                            async_kind = FnCallKind::AsyncCallMut;
                            &async_kind
                        }
                        FnCallKind::Call
                            if self.fn_call_receiver_is_async_closure(
                                frames, body, receiver, false,
                            ) =>
                        {
                            async_kind = FnCallKind::AsyncCall;
                            &async_kind
                        }
                        other => other,
                    };
                    // The `AsyncFn` family: the call runs no body.
                    // It creates the async closure's coroutine -
                    // the value the compiler's shim body for the
                    // call constructs - with the coroutine's
                    // state-0 fields in the shim's own order: the
                    // call arguments first, then the captures. The
                    // captures' shape is the call kind: references
                    // into the closure's environment for
                    // `async_call` / `async_call_mut` (the future
                    // borrows the closure), the environment's
                    // fields themselves for `async_call_once` (the
                    // future owns them). The coroutine body the
                    // future's `poll` runs is found through the
                    // future's own type at the poll site, exactly
                    // as for an async block's coroutine.
                    if matches!(
                        kind,
                        FnCallKind::AsyncCall
                            | FnCallKind::AsyncCallMut
                            | FnCallKind::AsyncCallOnce
                    ) {
                        let args_value = self.eval_operand(frames, body, &args)?;
                        let mut fields = match args_value {
                            Value::Tuple(items) => items,
                            Value::Unit => Vec::new(),
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "AsyncFn trait call arguments",
                                    "argument tuple",
                                    other.kind_name(),
                                ));
                            }
                        };
                        match kind {
                            FnCallKind::AsyncCallOnce => {
                                let value =
                                    self.eval_operand(frames, body, receiver)?;
                                match value {
                                    Value::Closure { env, .. } => {
                                        fields.extend(env);
                                    }
                                    // A zero-sized async closure
                                    // is a value whose local is
                                    // never assigned; it
                                    // materialises as the unit
                                    // value (its declared type is
                                    // the placeholder). Its
                                    // captures are all zero-sized
                                    // too: the environment the
                                    // coroutine takes is one unit
                                    // field per capture the layout
                                    // records.
                                    Value::Unit => {
                                        let n = async_captures.unwrap_or(0);
                                        fields.extend(
                                            std::iter::repeat_n(Value::Unit, n),
                                        );
                                    }
                                    other => {
                                        return Err(InterpError::type_mismatch(
                                            "AsyncFnOnce::async_call_once receiver",
                                            "async closure value",
                                            other.kind_name(),
                                        ));
                                    }
                                }
                            }
                            _ => {
                                let recv =
                                    self.eval_operand(frames, body, receiver)?;
                                let resolved = self.deref_value(&frames, &recv)?;
                                let value = self.read_resolved(&frames, &resolved)?;
                                match value {
                                    // A zero-sized async closure
                                    // behind the reference, as in
                                    // the `async_call_once` arm
                                    // above: each capture's field
                                    // is a reference to the
                                    // zero-sized capture value,
                                    // which materialises as the
                                    // unit value.
                                    Value::Unit => {
                                        let n = async_captures.unwrap_or(0);
                                        for _ in 0..n {
                                            fields.push(Value::Ref(
                                                RefTarget::value(Value::Unit),
                                            ));
                                        }
                                    }
                                    Value::Closure { def, env } => {
                                        for (i, field) in env.iter().enumerate() {
                                            // A capture taken into
                                            // the closure by
                                            // reference: the
                                            // environment field is
                                            // the reference itself,
                                            // and the coroutine's
                                            // field is that
                                            // reference reborrowed -
                                            // the field's own
                                            // value. Any other
                                            // capture: the field is
                                            // a reference to the
                                            // environment field.
                                            let by_ref = self
                                                .program
                                                .async_closure_captures
                                                .get(&def)
                                                .and_then(|flags| {
                                                    flags.get(i).copied()
                                                });
                                            let field_ref = match by_ref {
                                                Some(true) => Some(field.clone()),
                                                Some(false) => {
                                                    ref_to_field(&recv, i)
                                                }
                                                None => {
                                                    if matches!(
                                                        field,
                                                        Value::Ref(_)
                                                            | Value::FatRef { .. }
                                                    ) {
                                                        Some(field.clone())
                                                    } else {
                                                        ref_to_field(&recv, i)
                                                    }
                                                }
                                            };
                                            let Some(field_ref) = field_ref
                                            else {
                                                return Err(InterpError::type_mismatch(
                                                    "AsyncFn::async_call receiver",
                                                    "reference to an async closure value",
                                                    recv.kind_name(),
                                                ));
                                            };
                                            fields.push(field_ref);
                                        }
                                    }
                                    other => {
                                        return Err(InterpError::type_mismatch(
                                            "AsyncFn::async_call receiver",
                                            "async closure value",
                                            other.kind_name(),
                                        ));
                                    }
                                }
                            }
                        }
                        let future = Value::Coroutine {
                            variant: 0,
                            fields,
                            saved: Vec::new(),
                        };
                        let Some(bb) = target else {
                            return Err(InterpError::MissingContinuation {
                                func: func_name.clone(),
                            });
                        };
                        self.write_place(frames, body, destination, future)?;
                        Self::jump(frames, body, *bb)?;
                        continue 'drive;
                    }
                    // The caller-location stamp the conversion
                    // carried on this edge (a track-caller
                    // closure called directly or through a
                    // generic `Fn` bound), evaluated in the
                    // caller frame like any other argument.
                    let edge_stamp = match &stamp {
                        Some(op) => {
                            Some(self.eval_operand(frames, body, op)?)
                        }
                        None => None,
                    };
                    let args_value = self.eval_operand(frames, body, &args)?;
                    let arg_values = match args_value {
                        Value::Tuple(items) => items,
                        Value::Unit => Vec::new(),
                        other => {
                            return Err(InterpError::type_mismatch(
                                "Fn trait call arguments",
                                "argument tuple",
                                other.kind_name(),
                            ));
                        }
                    };
                    // A call on a panic-hook marker value (the
                    // default hook, or an `update_hook` chain
                    // link): the `prev` invocation of an update
                    // closure. No guest body serves it; the
                    // hook machinery runs it synchronously, as
                    // it runs the hook itself.
                    if let Some(hook) =
                        self.peek_panic_hook_receiver(frames, body, receiver)
                    {
                        let info_ref =
                            arg_values.first().cloned().unwrap_or(Value::Unit);
                        self.invoke_panic_hook(frames, &hook, info_ref);
                        let Some(bb) = target else {
                            return Err(InterpError::MissingContinuation {
                                func: func_name.clone(),
                            });
                        };
                        self.write_place(frames, body, destination, Value::Unit)?;
                        Self::jump(frames, body, *bb)?;
                        continue 'drive;
                    }
                    // The callable value, and the environment
                    // argument a closure body expects as its first
                    // argument. For `call` / `call_mut` the receiver
                    // already is the reference the body takes. For
                    // `call_once` the receiver is the value itself:
                    // a body that takes its environment by value (an
                    // `FnOnce` closure's own body — no shim exists in
                    // the program) receives the value, consuming the
                    // receiver place; any other body receives a
                    // reference to the value in place, which is what
                    // the real `call_once` shim passes when it
                    // consumes an `Fn` / `FnMut` closure (aliasing
                    // the moved-from place is equivalent because that
                    // place is dead after the call).
                    // The concrete type behind a dyn receiver, when
                    // the receiver is one: the dispatch for an ADT
                    // concrete value (a hand-written Fn impl) keys
                    // on it, and `dyn_callable` erases it.
                    let mut dyn_concrete: Option<String> = None;
                    let mut env_consumed = false;
                    let mut env_unwrapped = false;
                    let (callable, env_ref) = match kind {
                        // Every `AsyncFn` call completed in the
                        // branch at the top of this arm (it creates
                        // the coroutine and jumps); none reaches the
                        // body-call dispatch below.
                        FnCallKind::AsyncCall
                        | FnCallKind::AsyncCallMut
                        | FnCallKind::AsyncCallOnce => {
                            unreachable!("AsyncFn calls complete above")
                        }
                        FnCallKind::CallOnce => {
                            let (value, env_ref) = self.eval_callable(&frames, body, receiver)?;
                            let (value, env_ref) =
                                if matches!(value, Value::DynRef { .. }) {
                                    if let Value::DynRef { concrete, .. } = &value {
                                        dyn_concrete = Some(concrete.clone());
                                    }
                                    self.dyn_callable(&frames, &value)?
                                } else {
                                    (value, env_ref)
                                };
                            // A reference receiver names the
                            // blanket `FnOnce` impl for `&F` /
                            // `&mut F` (which requires `F: FnMut`):
                            // the pointee is the callable, called
                            // without being consumed -- the shape
                            // of a generic consumer's `call_once`
                            // on a borrowed closure, such as an
                            // `Option::map` handed `&mut self.f`.
                            // Follow the reference to the callable
                            // as the `Call` / `CallMut` arm's deref
                            // chain and the callable-operand
                            // completion path both do; a reference
                            // whose chain ends elsewhere keeps the
                            // value (and its error) unchanged.
                            let (value, env_ref) =
                                if matches!(value, Value::Ref(_) | Value::FatRef { .. }) {
                                    self.callable_through_refs(frames, value, env_ref)?
                                } else {
                                    (value, env_ref)
                                };
                            // A by-value receiver may still be a
                            // transparent one-field wrapper around
                            // the callable (an `FnOnce` wrapper
                            // struct): unwrap it, but only when the
                            // wrapped value is itself callable.
                            let (value, unwrapped) = unwrap_callable_wrapper(value);
                            env_unwrapped = unwrapped;
                            let env_ref = if unwrapped {
                                Value::Ref(RefTarget::value(value.clone()))
                            } else {
                                env_ref
                            };
                            let env_ref = match self.consumed_env_arg(
                                frames,
                                body,
                                receiver,
                                &value,
                            )? {
                                Some(by_value) => {
                                    env_consumed = true;
                                    by_value
                                }
                                None => env_ref,
                            };
                            (value, env_ref)
                        }
                        FnCallKind::Call | FnCallKind::CallMut => {
                            let recv = self.eval_operand(frames, body, receiver)?;
                            match recv {
                                Value::Ref(_) | Value::FatRef { .. } => {
                                    let resolved = self.deref_value(&frames, &recv)?;
                                    let mut value =
                                        self.read_resolved(&frames, &resolved)?;
                                    // Blanket `Fn` impls make
                                    // references and one-field
                                    // smart-pointer wrappers (an
                                    // `&F`, a `Box<dyn Fn>`) callable
                                    // themselves: follow the deref
                                    // chain until the callable value
                                    // itself shows.
                                    //
                                    // `direct_ref` tracks the
                                    // reference that directly denotes
                                    // the value the chain currently
                                    // shows: the receiver itself at
                                    // first, then each reference the
                                    // chain follows. The environment
                                    // reference a closure body
                                    // receives must be that reference
                                    // - the body's capture reads
                                    // project through it onto the
                                    // callable value itself. An outer
                                    // reference of the chain denotes a
                                    // place holding the next reference
                                    // (a `&mut F` stored in another
                                    // closure's capture field is the
                                    // common shape), and a body
                                    // entered through it reads that
                                    // stored reference where its first
                                    // capture belongs.
                                    let mut direct_ref = recv.clone();
                                    let mut unwrapped = false;
                                    for _ in 0..8 {
                                        let followed_ref = matches!(
                                            value,
                                            Value::Ref(_) | Value::FatRef { .. }
                                        );
                                        let next = match &value {
                                            Value::Ref(_) | Value::FatRef { .. } => {
                                                let Ok(t) =
                                                    self.deref_value(&frames, &value)
                                                else {
                                                    break;
                                                };
                                                let Ok(v) =
                                                    self.read_resolved(&frames, &t)
                                                else {
                                                    break;
                                                };
                                                v
                                            }
                                            Value::Enum { fields, .. }
                                                if fields.len() == 1 =>
                                            {
                                                unwrapped = true;
                                                fields[0].clone()
                                            }
                                            Value::Tuple(items)
                                                if items.len() == 1 =>
                                            {
                                                unwrapped = true;
                                                items[0].clone()
                                            }
                                            _ => break,
                                        };
                                        if followed_ref {
                                            direct_ref = value.clone();
                                        }
                                        value = next;
                                    }
                                    if matches!(value, Value::DynRef { .. }) {
                                        if let Value::DynRef { concrete, .. } = &value {
                                            dyn_concrete = Some(concrete.clone());
                                        }
                                        self.dyn_callable(&frames, &value)?
                                    } else if unwrapped {
                                        // The callable was reached by
                                        // unwrapping aggregate values
                                        // (smart pointers), so no place
                                        // reference names it: its
                                        // environment reference is a
                                        // value reference to the
                                        // callable itself, which is
                                        // what capture reads resolve
                                        // through.
                                        let env =
                                            Value::Ref(RefTarget::value(value.clone()));
                                        (value, env)
                                    } else {
                                        (value, direct_ref)
                                    }
                                }
                                Value::DynRef { .. } => {
                                    if let Value::DynRef { concrete, .. } = &recv {
                                        dyn_concrete = Some(concrete.clone());
                                    }
                                    self.dyn_callable(&frames, &recv)?
                                }
                                value => {
                                    // A by-value receiver may be a
                                    // transparent one-field wrapper
                                    // around the callable: unwrap it
                                    // when the wrapped value is
                                    // callable, as in the `CallOnce`
                                    // arm.
                                    let (value, unwrapped) =
                                        unwrap_callable_wrapper(value);
                                    let env = if unwrapped {
                                        Value::Ref(RefTarget::value(
                                            value.clone(),
                                        ))
                                    } else {
                                        self.callable_env_ref(
                                            &frames,
                                            body,
                                            receiver,
                                            &value,
                                        )?
                                    };
                                    (value, env)
                                }
                            }
                        }
                    };
                    // The `call_once` shim's environment drop (see
                    // `Frame::call_once_env_drop`): owed when the
                    // closure body took its environment by
                    // reference — the environment was not consumed
                    // into the body, the receiver was not a `dyn`
                    // dispatch or an unwrapped wrapper — and the
                    // body does not drop the environment itself.
                    let shim_drop = if matches!(kind, FnCallKind::CallOnce)
                        && !env_consumed
                        && !env_unwrapped
                        && dyn_concrete.is_none()
                    {
                        match (&callable, receiver_glue) {
                            (Value::Closure { def, .. }, Some(glue))
                                if !self.closure_body_drops_env(def) =>
                            {
                                self.call_once_receiver_root(
                                    frames, body, receiver, &callable,
                                )
                                .ok()
                                .map(|root| (root, glue.clone()))
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    // A `dyn` receiver's call passes through
                    // the `Fn` vtable shim, which presents its
                    // own location (the trait method's definition
                    // in core) to a track-caller callee: with no
                    // edge stamp, that shim location is the fixed
                    // stamp for this entry.
                    let shim_stamp = if edge_stamp.is_none()
                        && dyn_concrete.is_some()
                    {
                        let idx = match kind {
                            FnCallKind::Call | FnCallKind::AsyncCall => 0,
                            FnCallKind::CallMut | FnCallKind::AsyncCallMut => 1,
                            FnCallKind::CallOnce | FnCallKind::AsyncCallOnce => 2,
                        };
                        match &self.program.fn_shim_locations[idx] {
                            Some(c) => Some(self.eval_constant(c)?),
                            None => None,
                        }
                    } else {
                        None
                    };
                    match callable {
                        Value::Closure { def, .. } => {
                            let mut full = vec![env_ref];
                            full.extend(arg_values);
                            self.start_call_stamped(
                                frames,
                                &def,
                                full,
                                edge_stamp,
                                shim_stamp,
                                &destination,
                                *target,
                            )?;
                            if let Some(drop) = shim_drop {
                                if let Some(frame) = frames.last_mut() {
                                    frame.call_once_env_drop = Some(drop);
                                }
                            }
                        }
                        Value::FnPtr(name) => {
                            // As in the indirect-call arm: a
                            // trait method named by its plain
                            // trait path may be served by the
                            // builtin of the same name.
                            let fallback =
                                if self.virtual_call_target(&name, &arg_values).is_none() {
                                    self.builtin_name_fallback(&name)
                                } else {
                                    None
                                };
                            if let Some((canonical, kind, generic_args)) = fallback {
                                self.complete_builtin_name_fallback(
                                    frames,
                                    body,
                                    &canonical,
                                    kind,
                                    &generic_args,
                                    &name,
                                    arg_values,
                                    destination,
                                    *target,
                                )?;
                            } else {
                                self.start_call_stamped(
                                    frames,
                                    &name,
                                    arg_values,
                                    edge_stamp,
                                    shim_stamp,
                                    &destination,
                                    *target,
                                )?;
                            }
                        }
                        other => {
                            // A dyn receiver whose concrete value
                            // is an ADT value: a hand-written Fn
                            // impl on that type, when the program
                            // has its body, is the callable.
                            if let Some(concrete) = dyn_concrete.clone() {
                                if let Some(result) = self.dyn_adt_fn_call(
                                    frames,
                                    *kind,
                                    &concrete,
                                    other.clone(),
                                    env_ref,
                                    arg_values,
                                    destination,
                                    *target,
                                ) {
                                    result?;
                                } else {
                                    return Err(InterpError::type_mismatch(
                                        "Fn trait call",
                                        "callable value",
                                        other.kind_name(),
                                    ));
                                }
                            } else {
                                return Err(InterpError::type_mismatch(
                                    "Fn trait call",
                                    "callable value",
                                    other.kind_name(),
                                ));
                            }
                        }
                    }
                }
                TerminatorKind::CallBuiltin {
                    name,
                    generic_args,
                    generic_type_ids,
                    args,
                    destination,
                    target,
                    drop_glue,
                    consumed_glue,
                    skipped_glue,
                    elem_glue,
                    receiver_glue,
                    cmp_discriminants,
                    result_unsize,
                    unique_elem,
                    zeroed_value,
                } => {
                    if let Some(glue) = drop_glue {
                        // `std::mem::drop` with the dropped type's glue
                        // attached by conversion: run it against the
                        // argument (a place, or a constant's value),
                        // exactly like a `Drop`
                        // terminator — a destructor call suspends the
                        // remaining glue on a `DropContinuation` and
                        // resumes it when the destructor returns. The
                        // call's own (unit) result is written first;
                        // the destination is a different local from
                        // the dropped argument place.
                        let Some(bb) = target else {
                            return Err(InterpError::MissingContinuation {
                                func: func_name.clone(),
                            });
                        };
                        let Some(operand) = args.first() else {
                            return Err(InterpError::type_mismatch(
                                "std::mem::drop",
                                "one argument",
                                "no arguments",
                            ));
                        };
                        // The glue runs against the argument place,
                        // or — for a constant argument, whose value
                        // conversion accounted for when it attached
                        // the glue — against the constant's own value.
                        let root = match operand {
                            Operand::Copy(place) | Operand::Move(place) => {
                                let place = place.clone();
                                match self.resolve_place(&frames, body, &place)? {
                                    Resolved::Place {
                                        frame_pos,
                                        local,
                                        projection,
                                    } => RefTarget::place(
                                        frames[frame_pos].id,
                                        local,
                                        projection,
                                    ),
                                    Resolved::Static { name, projection } => {
                                        RefTarget::Static { name, projection }
                                    }
                                    Resolved::Alloc { id, projection } => {
                                        RefTarget::Alloc { id, projection }
                                    }
                                    Resolved::Value(..) => {
                                        return Err(InterpError::type_mismatch(
                                            "std::mem::drop",
                                            "a storage place",
                                            "a constant aggregate",
                                        ));
                                    }
                                }
                            }
                            Operand::Constant(_) => {
                                let value =
                                    self.eval_operand(&mut frames, body, operand)?;
                                RefTarget::value(value)
                            }
                        };
                        self.write_place(frames, body, destination, Value::Unit)?;
                        let work = vec![(Vec::new(), glue.root.clone())];
                        self.run_drop_glue(frames, root, work, &glue.adts, *bb)?;
                    } else {
                        // A `mem::drop` of a string whose glue is
                        // trivial runs no glue session: the recorded
                        // buffer's deallocation is observed here,
                        // against the argument place.
                        if name == "std::mem::drop" && args.len() == 1 {
                            if let Operand::Copy(place)
                            | Operand::Move(place) = &args[0]
                            {
                                if let Ok(Resolved::Place {
                                    frame_pos,
                                    local,
                                    projection,
                                }) = self.resolve_place(frames, body, place)
                                {
                                    if projection.is_empty() {
                                        let root = RefTarget::place(
                                            frames[frame_pos].id,
                                            local,
                                            projection,
                                        );
                                        self.note_string_buffer_free_inner(
                                            frames, &root, true,
                                        );
                                    }
                                }
                            }
                        }
                        // The values the call's `move` operands
                        // are about to consume out of plain
                        // locals, read without consuming: if the
                        // builtin ends in a suspension, they are
                        // put back (see `restore_moved_args`),
                        // because the resumed thread re-executes
                        // this terminator.
                        let caller_id = frames.last().expect("frame").id;
                        let moved_args: Vec<(usize, Value)> = args
                            .iter()
                            .filter_map(|arg| match arg {
                                Operand::Move(place) if place.projection.is_empty() => self
                                    .read_place(frames, body, place)
                                    .ok()
                                    .map(|value| (place.local.0, value)),
                                _ => None,
                            })
                            .collect();
                        *self.consumed_iter_final.borrow_mut() = None;
                        let builtin_outcome = self.exec_builtin(
                            frames,
                            body,
                            name,
                            generic_args,
                            generic_type_ids.as_deref(),
                            args,
                            cmp_discriminants.as_deref(),
                            unique_elem.as_ref(),
                            skipped_glue.as_ref(),
                            elem_glue.as_ref(),
                            receiver_glue.as_ref(),
                            zeroed_value.as_ref(),
                        );
                        if matches!(builtin_outcome, Err(InterpError::Suspended)) {
                            Self::restore_moved_args(frames, caller_id, moved_args);
                        }
                        let consumed_final =
                            self.consumed_iter_final.borrow_mut().take();
                        let mut result = match builtin_outcome {
                            Ok(result) => result,
                            Err(err) => {
                                // The builtin consumed an owned
                                // value and handed its final state
                                // back for the glue attached to
                                // the call, but the call is
                                // unwinding (a guest panic inside
                                // the builtin): the callee's
                                // cleanup would drop the value on
                                // this path too, so run the glue
                                // synchronously against the handed-
                                // back state before the error
                                // propagates, or its destructors
                                // never run. The cleanup's own
                                // outcome is not discarded: a
                                // destructor that panics here is a
                                // double panic in the guest, which
                                // natively aborts the run (it is
                                // not a catchable panic, so it must
                                // not be masked by the panic already
                                // propagating), and any other
                                // failure of the cleanup -- an abort
                                // it raised itself, or a non-panic
                                // interpreter failure -- propagates
                                // as itself rather than vanishing.
                                if let (Some(glue), Some(final_state)) =
                                    (&consumed_glue, consumed_final)
                                {
                                    match self.drop_value_sync(frames, final_state, glue) {
                                        Ok(()) => {}
                                        Err(InterpError::Panic { message, .. }) => {
                                            self.fire_nounwind_abort_panic(
                                                frames,
                                                (
                                                    "/rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/core/src/panicking.rs"
                                                        .to_string(),
                                                    233,
                                                    5,
                                                ),
                                                "panic in a destructor during cleanup",
                                                true,
                                            );
                                            return Err(InterpError::Abort {
                                                message: format!(
                                                    "panic `{message}` raised while dropping \
                                                     a value consumed by `{name}` as the call \
                                                     unwound: panicking while unwinding aborts \
                                                     the run"
                                                ),
                                            });
                                        }
                                        Err(other) => return Err(other),
                                    }
                                }
                                return Err(err);
                            }
                        };
                        // A `From` / `Into` conversion into a
                        // `Box<dyn Trait>` returns the plain box
                        // aggregate from the modelled conversion;
                        // conversion baked the unsizing the
                        // destination type implies, so the
                        // destination holds the same `DynRef` a
                        // coercion cast would have produced.
                        if let Some(meta) = result_unsize {
                            result = self.eval_unsize(frames, result, meta)?;
                        }
                        match target {
                            Some(bb) => {
                                self.write_place(frames, body, destination, result)?;
                                if name == "std::string::String::with_capacity" {
                                    self.record_string_buffer(frames, body, destination);
                                }
                                // A modelled consumer of an owned
                                // iterator drops the iterator at
                                // the end of the callee's real
                                // body: run the glue conversion
                                // attached to the call against the
                                // final state the builtin handed
                                // back, exactly like a `Drop`
                                // terminator for that value (the
                                // glue run jumps to the target, or
                                // suspends and resumes there).
                                match (consumed_glue, consumed_final) {
                                    (Some(glue), Some(final_state)) => {
                                        // The final state is parked
                                        // in the allocation store
                                        // first: a destructor body
                                        // writes back into its
                                        // receiver (a `mem::take` of
                                        // a field is the common
                                        // shape), which a bare value
                                        // root cannot accept, while
                                        // an allocation is the same
                                        // mutable storage a place
                                        // root denotes.
                                        let id = self.alloc_insert(final_state);
                                        let root = RefTarget::Alloc {
                                            id,
                                            projection: Vec::new(),
                                        };
                                        let work =
                                            vec![(Vec::new(), glue.root.clone())];
                                        self.run_drop_glue(
                                            frames, root, work, &glue.adts, *bb,
                                        )?;
                                    }
                                    _ => Self::jump(frames, body, *bb)?,
                                }
                            }
                            None => {
                                return Err(InterpError::MissingContinuation {
                                    func: func_name.clone(),
                                });
                            }
                        }
                    }
                }
                TerminatorKind::Drop { place, target, glue } => {
                    // Run the dropped type's glue (conversion built
                    // it; see `mir_convert`). The glue work starts
                    // here and either finishes in `run_drop_glue`
                    // (which jumps to `target`) or suspends on a
                    // destructor call and resumes when that call
                    // returns.
                    // A trait-object place reached through a
                    // dereference of the fat pointer itself (the
                    // `drop_in_place` shim's `*ptr`, a `ThinBox`
                    // pointee drop): resolving the dereference
                    // consumes the `DynRef`, leaving the session
                    // nothing that carries the concrete glue.
                    // Root the session at the fat pointer value
                    // instead, whose carried glue the `Dyn` plan
                    // runs against its base.
                    if matches!(glue.root, crate::ir::DropPlan::Dyn)
                        && matches!(
                            place.projection.last(),
                            Some(ProjectionElem::Deref)
                        )
                    {
                        let mut base_place = place.clone();
                        base_place.projection.pop();
                        if let Ok(dyn_value) =
                            self.read_place(&frames, body, &base_place)
                        {
                            if matches!(dyn_value, Value::DynRef { .. }) {
                                let root = RefTarget::Value(
                                    Box::new(dyn_value),
                                    Vec::new(),
                                );
                                let work =
                                    vec![(Vec::new(), glue.root.clone())];
                                self.run_drop_glue(
                                    frames,
                                    root,
                                    work,
                                    &glue.adts,
                                    *target,
                                )?;
                                continue;
                            }
                        }
                    }
                    let resolved = self.resolve_place(&frames, body, place)?;
                    let root = match resolved {
                        Resolved::Place {
                            frame_pos,
                            local,
                            projection,
                        } => RefTarget::place(frames[frame_pos].id, local, projection),
                        Resolved::Static { name, projection } => RefTarget::Static {
                            name,
                            projection,
                        },
                        Resolved::Alloc { id, projection } => RefTarget::Alloc {
                            id,
                            projection,
                        },
                        Resolved::Value(base, projection) => {
                            // The place is the deref of a value
                            // reference (a `Box` / `ManuallyDrop`
                            // deref in this model): the glue runs
                            // against that value itself, as it does
                            // for a `mem::drop` constant argument.
                            RefTarget::Value(Box::new(base), projection)
                        }
                    };
                    let work = vec![(Vec::new(), glue.root.clone())];
                    self.run_drop_glue(frames, root, work, &glue.adts, *target)?;
                }
                TerminatorKind::Resume => {
                    // The cleanup chain of the top frame is done:
                    // pop the frame and report the panic stashed in
                    // it when the unwinder entered the cleanup. The
                    // unwinder in `drive` continues the panic in
                    // the caller at its call site.
                    let frame = frames.pop().expect("frame");
                    // A destructor frame whose own cleanup chain
                    // just ran: its suspended glue session is the
                    // panic's next cleanup — the fields of the
                    // dropped value, as in a native `drop_in_place`
                    // whose destructor panicked.
                    let cont_pos = self
                        .drop_continuations
                        .iter()
                        .rposition(|cont| cont.callee_frame == frame.id);
                    match cont_pos {
                        Some(pos) => {
                            let cont = self.drop_continuations.remove(pos);
                            match frame.unwind_payload {
                                Some((message, payload)) => {
                                    match self.run_glue_cleanup(
                                        frames,
                                        cont,
                                        InterpError::Panic { message, payload },
                                    )? {
                                        GlueCleanupFlow::Completed(p) => return Err(p),
                                        // The cleanup suspended on a
                                        // field destructor's frame:
                                        // the run loop continues into
                                        // it, and the session's
                                        // completion re-raises the
                                        // stashed panic.
                                        GlueCleanupFlow::Suspended => {}
                                    }
                                }
                                None => {
                                    return Err(InterpError::UnsupportedTerminator {
                                        kind: "Resume".to_string(),
                                    });
                                }
                            }
                        }
                        None => {
                            self.drop_continuations
                                .retain(|cont| cont.callee_frame != frame.id);
                            match frame.unwind_payload {
                                Some((message, payload)) => {
                                    self.resume_signaled = true;
                                    return Err(InterpError::Panic { message, payload });
                                }
                                None => {
                                    return Err(InterpError::UnsupportedTerminator {
                                        kind: "Resume".to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
                TerminatorKind::Abort => {
                    // A panic escaped a cleanup or crossed a
                    // nounwind boundary in the guest source: the
                    // run aborts, uncatchably.
                    let original = frames
                        .last()
                        .and_then(|f| f.unwind_payload.as_ref().map(|(m, _)| m.clone()));
                    return Err(InterpError::Abort {
                        message: match original {
                            Some(message) => format!(
                                "panicked while processing panic `{message}`: aborting"
                            ),
                            None => "abort terminator executed".to_string(),
                        },
                    });
                }
                TerminatorKind::Return => {
                    if let Some(value) = self.exec_return(frames, stop_depth)? {
                        return Ok(value);
                    }
                }
                TerminatorKind::Unreachable => {
                    return Err(InterpError::panic("entered unreachable code"));
                }
                TerminatorKind::Assert {
                    cond,
                    expected,
                    target,
                    message,
                    format,
                    operands,
                    overflow,
                } => {
                    // An overflow assertion whose instantiating
                    // crate was compiled without overflow checks is
                    // dropped, as native codegen drops it for that
                    // crate: the checked operation's wrapped result
                    // is simply used, and the condition is not
                    // evaluated at all. The setting is the call
                    // stack's (see `overflow_checks_at`).
                    if *overflow && !self.overflow_checks_at(frames) {
                        Self::jump(frames, body, *target)?;
                    } else {
                        let value = self.eval_operand(frames, body, cond)?;
                        let actual = value.as_bool().ok_or_else(|| {
                            InterpError::type_mismatch("Assert condition", "bool", value.kind_name())
                        })?;
                        if actual != *expected {
                            // A message embedding operand values
                            // is formatted from them now, with
                            // core's exact text; the baked kind
                            // text is the fallback for a value
                            // that does not render as an integer.
                            let text = match format {
                                Some(kind) => {
                                    let mut vals = Vec::with_capacity(operands.len());
                                    for op in operands {
                                        vals.push(self.eval_operand(frames, body, op)?);
                                    }
                                    format_assert_message(*kind, &vals)
                                        .unwrap_or_else(|| message.clone())
                                }
                                None => message.clone(),
                            };
                            // The panic's location is the
                            // terminator's own source position
                            // (the operation it guards): record
                            // it as the panic's stamp, with the
                            // entry path a rendered backtrace
                            // starts from, as a panic entry
                            // point's stamp would.
                            self.note_call_site_stamp(frames);
                            self.note_panic_no_backtrace(false);
                            self.note_panic_origin(PanicOrigin::Assert(match format {
                                Some(crate::ir::AssertFormat::BoundsCheck) => {
                                    "core::panicking::panic_bounds_check"
                                }
                                Some(
                                    crate::ir::AssertFormat::MisalignedPointerDeref,
                                ) => "core::panicking::panic_misaligned_pointer_dereference",
                                None => "core::panicking::panic",
                            }));
                            return Err(InterpError::panic(text));
                        }
                        Self::jump(frames, body, *target)?;
                    }
                }
                TerminatorKind::InlineAsm {
                    regs,
                    steps,
                    target,
                    ret,
                    arg_seeds,
                } => {
                    let outcome = self.exec_inline_asm(
                        frames,
                        body,
                        regs,
                        steps,
                        *ret,
                        arg_seeds,
                        target.is_none(),
                    )?;
                    match outcome {
                        AsmOutcome::JumpTo(block) => {
                            Self::jump(frames, body, block)?;
                        }
                        AsmOutcome::Fallthrough => match target {
                            Some(target) => Self::jump(frames, body, *target)?,
                            None => {
                                if let Some(value) = self.exec_return(frames, stop_depth)? {
                                    return Ok(value);
                                }
                            }
                        },
                    }
                }
                TerminatorKind::Unsupported(kind) => {
                    return Err(InterpError::UnsupportedTerminator { kind: kind.clone() });
                }
            }
        }
    }
}

impl<'a> Interpreter<'a> {
    /// Execute a return from the top frame: deliver the return
    /// place's value to the caller, or finish the run when the
    /// frame stack reaches the stop depth. `Ok(Some(value))` ends
    /// the run; `Ok(None)` means execution continues in the caller
    /// (or in a resumed drop-glue session).
    fn exec_return(
        &mut self,
        frames: &mut Vec<Frame>,
        stop_depth: usize,
    ) -> InterpResult<Option<Value>> {
        // A returning destructor frame resumes its
        // suspended drop glue instead of delivering a
        // value to a caller place.
        let returning = frames.last().expect("frame").id;
        if let Some(pos) = self
            .drop_continuations
            .iter()
            .rposition(|c| c.callee_frame == returning)
        {
            let cont = self.drop_continuations.remove(pos);
            frames.pop();
            self.run_glue_session(frames, cont.session, cont.sessions)?;
            return Ok(None);
        }
        let frame = frames.pop().expect("frame");
        let mut value = frame.locals.first().cloned().unwrap_or(Value::Uninit);
        if value == Value::Uninit {
            // rustc MIR never writes the return place of
            // a zero-sized return type: the final block
            // is a bare `return;`, and the type's single
            // value is implied. An unwritten `_0`
            // therefore reads as that value exactly when
            // the declared return type is provably
            // zero-sized. For every other type an
            // unwritten `_0` is a genuine uninitialized
            // read and stays an error.
            match self.zst_local_value(&frame, 0) {
                Some(v) => value = v,
                None => match self.maybe_uninit_local_value(&frame, 0) {
                    // A `MaybeUninit` return place holding the
                    // bare marker denotes the uninitialized
                    // union value (see `normalize_for_ty`): the
                    // payload being uninitialized is the type's
                    // ordinary content, not an unwritten place.
                    Some(v) => value = v,
                    None => {
                        return Err(InterpError::UninitializedRead {
                            func: frame.func.clone(),
                            place: "_0".to_string(),
                        });
                    }
                },
            }
        }
        if frames.len() == stop_depth {
            return Ok(Some(value));
        }
        let dest = frame
            .return_dest
            .clone()
            .expect("non-entry frame has a destination");
        let resume = frame.return_block;
        let env_drop = frame.call_once_env_drop;
        let caller_name = frames.last().expect("caller").func.clone();
        let caller_body = self.program.get(&caller_name).ok_or_else(|| {
            InterpError::UnknownFunction {
                name: caller_name.clone(),
            }
        })?;
        // A `Vec` return value assembled in its real aggregate
        // form inside the callee (the shape an inlined
        // `with_capacity_in` leaves in `split_off`: the elements
        // live behind the buffer pointer, in the allocation the
        // pointer denotes): the callee's own reads of the value
        // used the real fields, but every consumer in the caller
        // reads the element form the value model holds a vector
        // as, so the value translates as it crosses (see
        // `vec_from_real_form`).
        if let Some(callee_body) = self.program.get(&frame.func) {
            if matches!(
                callee_body.local_ty(Local(0)),
                Some(ty @ Ty::Adt(_)) if is_vec_adt_ty(ty)
            ) {
                if let Some(translated) = self.vec_from_real_form(frames, &value) {
                    value = translated?;
                }
            }
        }
        // A fieldless enum return value crossing into an
        // integer-typed destination place: the destination's
        // declared type is the integer the enum is represented
        // as (the call's function type was transmuted -- an
        // FFI signature reinterpreted through a function
        // pointer), so the value crosses as its discriminant,
        // the callee body's baked table supplying the value
        // (see `Body::return_discriminants`). Any other shape
        // keeps the ordinary write below, whose judgment
        // stands.
        if let Some(callee_body) = self.program.get(&frame.func) {
            if let Some(table) = &callee_body.return_discriminants {
                if let Value::Enum { variant, fields } = &value {
                    if fields.is_empty() {
                        if let Some(discr) = table.get(*variant as usize) {
                            match cast_place_ty(caller_body, &dest) {
                                Some(Ty::Int(it)) => {
                                    value =
                                        Value::Int(wrap_signed(*discr, it.bits()));
                                }
                                Some(Ty::Uint(ut)) => {
                                    value = Value::Uint(wrap_unsigned(
                                        *discr as u128,
                                        ut.bits(),
                                    ));
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        self.write_place(frames, caller_body, &dest, value)?;
        match resume {
            Some(bb) => {
                // A `call_once` shim environment drop owed for
                // this call (see `Frame::call_once_env_drop`) runs
                // now that the body has returned, as the shim's
                // own drop of the receiver value: the glue session
                // ends by jumping to the resume block, exactly
                // like a `Drop` terminator at the call site.
                if let Some((root, glue)) = env_drop {
                    let work = vec![(Vec::new(), glue.root.clone())];
                    self.run_drop_glue(frames, root, work, &glue.adts, bb)?;
                } else {
                    Self::jump(frames, caller_body, bb)?;
                }
            }
            None => {
                return Err(InterpError::MissingContinuation { func: caller_name });
            }
        }
        Ok(None)
    }
}

impl<'a> Interpreter<'a> {
    pub(crate) fn exec_statement(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        statement: &Statement,
    ) -> InterpResult<()> {
        match &statement.kind {
            StatementKind::Assign(place, rvalue) => {
                // A recorded string buffer follows a plain move of
                // the string to its new place (see
                // `string_buffers`).
                let transfer = match rvalue {
                    Rvalue::Use(Operand::Move(src) | Operand::Copy(src)) => {
                        let key_of = |p: &Place| match self
                            .resolve_place(frames, body, p)
                        {
                            Ok(Resolved::Place {
                                frame_pos,
                                local,
                                projection,
                            }) if projection.is_empty() => {
                                Some((frames[frame_pos].id, local))
                            }
                            _ => None,
                        };
                        match (key_of(src), key_of(place)) {
                            (Some(from), Some(to)) if from != to => Some((from, to)),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                let value = self.eval_rvalue(frames, body, rvalue, body.local_ty(place.local))?;
                let written = self.write_place(frames, body, place, value);
                if written.is_ok() {
                    if let Some((from, to)) = transfer {
                        let entry = self.string_buffers.borrow_mut().remove(&from);
                        if let Some(entry) = entry {
                            self.string_buffers.borrow_mut().insert(to, entry);
                        }
                    }
                }
                written
            }
            StatementKind::ShapedStore { place, value, shape } => {
                let value = self.eval_operand(frames, body, value)?;
                // A shaped store to a `Vec`'s length slot (the
                // inlined `set_len` in `truncate` takes this
                // form): the slot is not storage, so the store
                // applies to the vector itself, as in
                // `write_place`.
                if let Some(written) =
                    self.write_vec_len_slot(frames, body, place, &value)
                {
                    return written;
                }
                if let Some(written) =
                    self.write_vec_buf_slot(frames, body, place, &value)
                {
                    return written;
                }
                let target = self.resolve_place(frames, body, place)?;
                // A shaped store through a foreign view (a
                // reference transmuted to a different ADT over
                // the same storage): the byte view owns it when
                // it can judge it (see
                // `Interpreter::foreign_write`).
                if let Some(written) =
                    self.foreign_write(frames, body, place, &target, &value)
                {
                    return written;
                }
                // The shaped store's fallthrough is a projected
                // assignment like any other: it takes the place's
                // root type evidence with it (see `write_place`).
                let root_ty = self.resolved_root_ty(body, place);
                self.shaped_store(frames, &target, value, shape, root_ty.as_ref())
            }
            StatementKind::SetDiscriminant { place, variant } => {
                let target = self.resolve_place(frames, body, place)?;
                // A discriminant set through a foreign view (a
                // reference transmuted to a different ADT over
                // the same storage): the byte view owns it when
                // it can judge it (see
                // `Interpreter::foreign_set_discriminant`).
                if let Some(set) = self.foreign_set_discriminant(
                    frames, body, place, &target, *variant,
                ) {
                    return set;
                }
                match target {
                    Resolved::Place {
                        frame_pos,
                        local,
                        projection,
                    } => {
                        let frame = &mut frames[frame_pos];
                        let func = frame.func.clone();
                        let slot = frame.locals.get_mut(local).ok_or_else(|| {
                            InterpError::UnknownLocal {
                                func: func.clone(),
                                local,
                            }
                        })?;
                        let place_desc = describe_resolved(local, &projection);
                        let value = navigate_mut(slot, &projection, &func, &place_desc)?;
                        Self::set_discriminant_value(value, *variant, place)
                    }
                    Resolved::Static { name, projection } => {
                        let mut store = self.statics.borrow_mut();
                        let slot = store
                            .get_mut(&name)
                            .ok_or_else(|| self.missing_static(&name))?;
                        let place_desc = format!("static {name}");
                        let value = navigate_mut(slot, &projection, "<static>", &place_desc)?;
                        Self::set_discriminant_value(value, *variant, place)
                    }
                    Resolved::Alloc { id, projection } => {
                        let mut store = self.allocs.borrow_mut();
                        let slot = store
                            .get_mut(&id)
                            .ok_or_else(|| self.missing_alloc(id))?;
                        let place_desc = format!("allocation #{id}");
                        let value =
                            navigate_mut(slot, &projection, "<allocation>", &place_desc)?;
                        Self::set_discriminant_value(value, *variant, place)
                    }
                    Resolved::Value(..) => Err(InterpError::type_mismatch(
                        format!("SetDiscriminant on {place}"),
                        "mutable place",
                        "constant aggregate",
                    )),
                }
            }
            // StorageLive / StorageDead are liveness hints in rustc MIR;
            // they do not change values. Nop is a no-op by definition.
            StatementKind::StorageLive(_) | StatementKind::StorageDead(_) | StatementKind::Nop => {
                Ok(())
            }
            // A coverage counter point: increment the physical
            // counter the statement's coverage record assigns to
            // this BCB. The record was resolved at conversion
            // time for statements inlined from another instance;
            // `u32::MAX` resolves it from the executing body
            // here, and `u32::MAX - 1` counts nothing (the
            // attributed instance has no record). Bodies without
            // a coverage record (the program was not compiled
            // with coverage, or this body carries none) count
            // nothing.
            StatementKind::Coverage(bcb, record) => {
                let record = if *record == u32::MAX {
                    self.coverage_by_body.get(&body.name).copied()
                } else if *record == u32::MAX - 1 {
                    None
                } else {
                    Some(*record as usize)
                };
                if let Some(record) = record {
                    let function = &self.program.coverage.functions[record];
                    if let Some(Some(counter)) =
                        function.counter_for_bcb.get(*bcb as usize)
                    {
                        let mut counts = self.coverage_counts.borrow_mut();
                        if let Some(slot) = counts
                            .get_mut(record)
                            .and_then(|v| v.get_mut(*counter as usize))
                        {
                            *slot += 1;
                        }
                    }
                }
                Ok(())
            }
            StatementKind::Assume(op) => {
                let value = self.eval_operand(frames, body, op)?;
                match value {
                    Value::Bool(true) => Ok(()),
                    Value::Bool(false) => Err(InterpError::Panic {
                        message: "intrinsic `assume` called with a false condition                                   (undefined behaviour in the guest)"
                            .to_string(),
                        payload: None,
                    }),
                    other => Err(InterpError::type_mismatch(
                        "assume operand",
                        "bool",
                        other.kind_name(),
                    )),
                }
            }
            StatementKind::CopyNonOverlapping { src, dst, count } => {
                let src_value = self.eval_operand(frames, body, src)?;
                let dst_value = self.eval_operand(frames, body, dst)?;
                let count_value = self.eval_operand(frames, body, count)?;
                // The copy's element type is the pointee of the
                // pointer operands, when they are bare locals with a
                // declared pointer type. The intrinsic is generic
                // over the element, and the copy needs the type to
                // tell "one element that is itself an aggregate"
                // (an array element, a struct element) from "an
                // aggregate of elements".
                let elem = operand_pointee_ty(body, src)
                    .or_else(|| operand_pointee_ty(body, dst));
                self.exec_copy_nonoverlapping(
                    frames,
                    src_value,
                    dst_value,
                    count_value,
                    elem.as_ref(),
                )
            }
            StatementKind::Unsupported(kind) => Err(InterpError::UnsupportedStatement {
                kind: kind.clone(),
            }),
        }
    }
}

impl<'a> Interpreter<'a> {
    /// The base denomination for a fat reference formed by
    /// `ptr::from_raw_parts` over a data pointer. A data pointer
    /// denoting one element of a sequence (its target's
    /// projection ends in an `Index` step) forms a slice over
    /// that sequence's elements from that position on, so the
    /// base is denominated as the sequence window it is: the
    /// trailing element step becomes a `Subslice` step over the
    /// sequence the head steps denote -- the denomination range
    /// indexing gives a window (see `windowed_base`). Without
    /// it the formed base is value-identical to the base a
    /// deref / unsizing formation produces for a slice over the
    /// pointed element's own elements, and the two readings
    /// collide exactly when the element is itself a sequence of
    /// the fat length (a `chunks_exact` chunk over a
    /// `Vec<Vec<T>>` whose first inner vector is as long as the
    /// chunk): the formation provenance is the only thing that
    /// tells them apart, so it is recorded here, at formation,
    /// not guessed at the readers.
    ///
    /// The window is recorded only when it stays inside the
    /// head sequence's length: a formation ranging past it is
    /// not a window over that sequence in its own element units
    /// (a byte-length reinterpretation over a wider element
    /// type is the canonical case), and it keeps the raw-parts
    /// denomination the readers already serve.
    ///
    /// It is recorded only where the collision is real: the
    /// element at the window's start is itself a sequence of
    /// exactly the fat length. A formation whose start element
    /// is a scalar, or a sequence of a different length, is
    /// already denominated unambiguously by the element step
    /// (a reader tells the two readings apart by length), and
    /// rewriting its base would disturb the readers that key
    /// on the element-step shape -- the ordinary `split_at`
    /// halves over a scalar sequence, the line pieces of a
    /// `str`, and a sort's slices over its buffer and scratch
    /// storage all form through here.
    pub(crate) fn raw_parts_base(
        &self,
        frames: &[Frame],
        base: Value,
        len: u128,
    ) -> Value {
        let Ok(window_len) = usize::try_from(len) else {
            return base;
        };
        let windowed = |projection: &[RefProj]| -> Option<Vec<RefProj>> {
            let (RefProj::Index(start), head) = projection.split_last()? else {
                return None;
            };
            let mut projection = head.to_vec();
            projection.push(RefProj::Subslice {
                offset: *start,
                len: window_len,
            });
            Some(projection)
        };
        // The candidate base with the window step recorded, and
        // the head target the window ranges over (the same
        // target with the trailing element step dropped).
        let (candidate, head): (Value, Value) = match &base {
            Value::Ref(RefTarget::Place {
                frame,
                local,
                projection,
            }) => match (windowed(projection), projection.split_last()) {
                (Some(windowed), Some((_, head))) => (
                    Value::Ref(RefTarget::place(*frame, *local, windowed)),
                    Value::Ref(RefTarget::place(*frame, *local, head.to_vec())),
                ),
                _ => return base,
            },
            Value::Ref(RefTarget::Static {
                name,
                projection,
            }) => match (windowed(projection), projection.split_last()) {
                (Some(windowed), Some((_, head))) => (
                    Value::Ref(RefTarget::Static {
                        name: name.clone(),
                        projection: windowed,
                    }),
                    Value::Ref(RefTarget::Static {
                        name: name.clone(),
                        projection: head.to_vec(),
                    }),
                ),
                _ => return base,
            },
            Value::Ref(RefTarget::Alloc { id, projection }) => {
                match (windowed(projection), projection.split_last()) {
                    (Some(windowed), Some((_, head))) => (
                        Value::Ref(RefTarget::Alloc {
                            id: *id,
                            projection: windowed,
                        }),
                        Value::Ref(RefTarget::Alloc {
                            id: *id,
                            projection: head.to_vec(),
                        }),
                    ),
                    _ => return base,
                }
            }
            Value::Ref(RefTarget::Value(value, projection)) => {
                match (windowed(projection), projection.split_last()) {
                    (Some(windowed), Some((_, head))) => (
                        Value::Ref(RefTarget::Value(value.clone(), windowed)),
                        Value::Ref(RefTarget::Value(value.clone(), head.to_vec())),
                    ),
                    _ => return base,
                }
            }
            _ => return base,
        };
        let start = match &candidate {
            Value::Ref(RefTarget::Place { projection, .. })
            | Value::Ref(RefTarget::Static { projection, .. })
            | Value::Ref(RefTarget::Alloc { projection, .. })
            | Value::Ref(RefTarget::Value(_, projection)) => match projection.last() {
                Some(RefProj::Subslice { offset, .. }) => *offset,
                _ => return base,
            },
            _ => return base,
        };
        let head_value = self
            .deref_value(frames, &head)
            .and_then(|target| self.read_resolved(frames, &target))
            .ok();
        let head_len = head_value.as_ref().and_then(|value| value.len());
        let collides = head_value
            .as_ref()
            .and_then(|value| {
                project_value(
                    value.clone(),
                    &[RefProj::Index(start)],
                    "raw parts base",
                    "raw parts base",
                )
                .ok()
            })
            .is_some_and(|element| {
                matches!(
                    element,
                    Value::Tuple(_)
                        | Value::Bytes(_)
                        | Value::Str(_)
                        | Value::Enum { .. }
                        | Value::SparseVec { .. }
                        | Value::FatRef { .. }
                ) && element.len() == Some(len)
            });
        match head_len {
            Some(head_len) if start as u128 + len <= head_len && collides => candidate,
            _ => base,
        }
    }

    /// The string data value a byte-string value denotes
    /// (see [`Rvalue::BytesToStr`]): the value itself when it
    /// already is string data; otherwise the text its bytes
    /// form. A value whose bytes the model cannot read, or
    /// whose bytes are not valid UTF-8 (undefined behaviour
    /// in the guest), is returned unchanged, keeping the
    /// behaviour such a value has today at the string
    /// consumers that meet it.
    fn bytes_as_str(&self, frames: &[Frame], value: &Value) -> Value {
        if let Value::Str(_) = value {
            return value.clone();
        }
        let Ok((elems, _)) = self.slice_elements(frames, value) else {
            return value.clone();
        };
        let mut bytes = Vec::with_capacity(elems.len());
        for elem in &elems {
            match elem {
                Value::Uint(b) => bytes.push(*b as u8),
                Value::Int(b) if (0..=255).contains(b) => {
                    bytes.push(*b as u8)
                }
                _ => return value.clone(),
            }
        }
        match String::from_utf8(bytes) {
            Ok(text) => Value::Str(text),
            Err(_) => value.clone(),
        }
    }

    pub(crate) fn eval_rvalue(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        rvalue: &Rvalue,
        dest_ty: Option<&Ty>,
    ) -> InterpResult<Value> {
        match rvalue {
            Rvalue::Use(op) => self.eval_operand(frames, body, op),
            Rvalue::DataPtr(op) => {
                let value = self.eval_operand(frames, body, op)?;
                match value {
                    Value::FatRef { base, .. } => Ok((*base).clone()),
                    Value::DynRef { base, .. } => Ok(self.dyn_data_pointer(&base)),
                    other => Ok(other),
                }
            }
            Rvalue::Materialize(op, plan) => {
                let value = self.eval_operand(frames, body, op)?;
                self.eval_materialize(value, *plan)
            }
            Rvalue::RawPtrAggregate { data, meta } => {
                let base = self.eval_operand(frames, body, data)?;
                match meta {
                    None => Ok(base),
                    Some(meta_op) => {
                        let meta_value = self.eval_operand(frames, body, meta_op)?;
                        if meta_value == Value::Unit {
                            // Unit metadata is no metadata: the
                            // aggregate is the data pointer itself.
                            return Ok(base);
                        }
                        // A `dyn` metadata value: the aggregate
                        // is the data pointer rebased onto the
                        // vtable the metadata names.
                        if let Some(dynref) =
                            self.dyn_ref_from_metadata(&meta_value, base.clone())
                        {
                            return Ok(dynref);
                        }
                        let len = match meta_value {
                            Value::Uint(v) => v,
                            Value::Int(v) => v as u128,
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "raw pointer metadata",
                                    "integer length",
                                    other.kind_name(),
                                ))
                            }
                        };
                        Ok(Value::FatRef {
                            base: Box::new(self.raw_parts_base(frames, base, len)),
                            len,
                        })
                    }
                }
            }
            Rvalue::BinaryOp(op, pair) => {
                let lhs = self.eval_operand(frames, body, &pair.0)?;
                let rhs = self.eval_operand(frames, body, &pair.1)?;
                let leaf = operand_address_pointee_ty(body, &pair.0)
                    .or_else(|| operand_address_pointee_ty(body, &pair.1));
                eval_binary(*op, &lhs, &rhs, dest_ty, false, &|v| {
                    self.pointer_address(frames, v, leaf.as_ref())
                })
                .map(|(v, _)| v)
            }
            Rvalue::CheckedBinaryOp(op, pair) => {
                let lhs = self.eval_operand(frames, body, &pair.0)?;
                let rhs = self.eval_operand(frames, body, &pair.1)?;
                let leaf = operand_address_pointee_ty(body, &pair.0)
                    .or_else(|| operand_address_pointee_ty(body, &pair.1));
                let (v, overflowed) = eval_binary(*op, &lhs, &rhs, dest_ty, true, &|v| {
                    self.pointer_address(frames, v, leaf.as_ref())
                })?;
                Ok(Value::Tuple(vec![v, Value::Bool(overflowed)]))
            }
            Rvalue::UnaryOp(UnOp::PtrMetadata, operand) => {
                let v = self.eval_operand(frames, body, operand)?;
                self.ptr_metadata(frames, &v, dest_ty)
            }
            Rvalue::UnaryOp(op, operand) => {
                let v = self.eval_operand(frames, body, operand)?;
                eval_unary(*op, &v, dest_ty)
            }
            Rvalue::AggregateTuple(operands) => {
                let mut items = Vec::with_capacity(operands.len());
                for op in operands {
                    items.push(self.eval_operand(frames, body, op)?);
                }
                Ok(Value::Tuple(items))            }
            Rvalue::AggregateAdt {
                variant,
                operands,
                field_index,
                units_from,
                payload_entry,
            } => {
                // A union aggregate's single operand initializes the
                // active field; the slots before it stay
                // uninitialized, exactly as a field write through a
                // projection leaves the slots it skipped over (see
                // `navigate_mut`'s coroutine arm). Reading the active
                // field then finds the stored value, and reading any
                // other field is an uninitialized read, as it should
                // be: the model does not reinterpret a union's bytes.
                let mut fields = match field_index {
                    Some(active) => vec![Value::Uninit; *active as usize],
                    None => Vec::with_capacity(operands.len()),
                };
                for op in operands {
                    fields.push(self.eval_operand(frames, body, op)?);
                }
                if let Some((from_size, payload_pos)) = units_from {
                    if let Some(field) = fields.get_mut(*payload_pos) {
                        let taken = std::mem::replace(field, Value::Unit);
                        *field = ptr_cast_units_value(
                            taken,
                            *from_size,
                            &|id| self.alloc_index_is_typed(id),
                        );
                    }
                }
                if let Some((enter, depth, payload_pos)) = payload_entry {
                    if let Some(field) = fields.get_mut(*payload_pos) {
                        let taken = std::mem::replace(field, Value::Unit);
                        *field = if *enter {
                            self.ptr_field_value(frames, body, taken, *depth)?
                        } else {
                            self.ptr_field_base_value(taken, *depth)?
                        };
                    }
                }
                // A `MaybeUninit::new(v)` construction (the union's
                // `value` field, index 1): in this model a
                // `MaybeUninit<T>` value holds its payload at field
                // 0 -- the slot every pointer access to the union
                // denotes (`as_ptr` / `as_mut_ptr` enter field 0, and
                // the thread-local seed expansion builds the same
                // shape) -- not at the declared field index with
                // uninitialized padding before it. The construction
                // therefore unwraps the `ManuallyDrop<T>` aggregate
                // the operand arrives as and stores the payload at
                // field 0. Without this, a value written through a
                // `*mut MaybeUninit<T>` (e.g. `ptr::replace` in std's
                // thread-local lazy initialization) would sit in the
                // place where no reader of the payload can reach it.
                if *field_index == Some(1) && operands.len() == 1 {
                    let is_maybe_uninit = matches!(
                        dest_ty,
                        Some(Ty::Adt(name))
                            if name.starts_with("std::mem::MaybeUninit")
                                || name.starts_with("core::mem::MaybeUninit")
                    );
                    if is_maybe_uninit {
                        let operand = fields.pop().expect("one operand pushed");
                        let payload = match operand {
                            Value::Enum { fields: inner, .. } if inner.len() == 1 => {
                                inner.into_iter().next().expect("one field")
                            }
                            other => other,
                        };
                        return Ok(Value::Enum {
                            variant: 0,
                            fields: vec![payload],
                        });
                    }
                }
                Ok(Value::Enum {
                    variant: *variant,
                    fields,
                })
            }
            Rvalue::ClosureAggregate { def, operands } => {
                let mut env = Vec::with_capacity(operands.len());
                for op in operands {
                    env.push(self.eval_operand(frames, body, op)?);
                }
                Ok(Value::Closure {
                    def: def.clone(),
                    env,
                })
            }
            Rvalue::CoroutineAggregate(operands) => {
                let mut fields = Vec::with_capacity(operands.len());
                for op in operands {
                    fields.push(self.eval_operand(frames, body, op)?);
                }
                Ok(Value::Coroutine {
                    variant: 0,
                    fields,
                    saved: Vec::new(),
                })
            }
            Rvalue::Cast(operand, ty) => {
                let v = self.eval_operand(frames, body, operand)?;
                let src_ty = cast_src_ty(body, operand);
                eval_cast(&v, src_ty.as_ref(), ty)
            }
            Rvalue::Unsize(operand, meta) => {
                let v = self.eval_operand(frames, body, operand)?;
                self.eval_unsize(frames, v, meta)
            }
            Rvalue::PtrToInt(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                let leaf = operand_address_pointee_ty(body, operand);
                match self.pointer_address(frames, &value, leaf.as_ref()) {
                    Some(addr) => {
                        self.note_exposed_target(addr as u64, &value);
                        Ok(Value::Uint(addr))
                    }
                    None => Err(InterpError::TypeMismatch {
                        context: "pointer-to-integer transmute".to_string(),
                        expected: "a pointer value".to_string(),
                        found: value.kind_name().to_string(),
                    }),
                }
            }
            Rvalue::FnPtrToPtr(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                match synthetic_address(&value) {
                    Some(addr) => {
                        self.note_exposed_target(addr as u64, &value);
                        Ok(Value::Ref(RefTarget::Address(addr as u64)))
                    }
                    None => Err(InterpError::TypeMismatch {
                        context: "function-pointer-to-pointer cast".to_string(),
                        expected: "a function pointer value".to_string(),
                        found: value.kind_name().to_string(),
                    }),
                }
            }
            Rvalue::PtrField(operand, depth) => {
                let value = self.eval_operand(frames, body, operand)?;
                self.ptr_field_value(frames, body, value, *depth)
            }
            Rvalue::PtrFieldFirstElem(operand, depth) => {
                let value = self.eval_operand(frames, body, operand)?;
                let value = self.ptr_field_value(frames, body, value, *depth)?;
                self.ptr_first_elem_value(frames, body, value, None, None)
            }
            Rvalue::PtrFirstElem(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                // The operand's declared type, when it is a bare
                // local: it carries the slice's element type the
                // storage judgment in `ptr_first_elem_value`
                // needs (see `fat_slice_elem_is_string_carrier`).
                let elem_string = match operand {
                    Operand::Copy(place) | Operand::Move(place)
                        if place.projection.is_empty() =>
                    {
                        body.local_ty(place.local)
                            .and_then(fat_slice_elem_is_string_carrier)
                    }
                    _ => None,
                };
                let elem_ty = match operand {
                    Operand::Copy(place) | Operand::Move(place)
                        if place.projection.is_empty() =>
                    {
                        body.local_ty(place.local)
                            .and_then(|ty| fat_slice_elem_ty(&ty))
                    }
                    _ => None,
                };
                self.ptr_first_elem_value(
                    frames,
                    body,
                    value,
                    elem_string,
                    elem_ty.as_ref(),
                )
            }
            Rvalue::PtrCastUnits(operand, from_size) => {
                let value = self.eval_operand(frames, body, operand)?;
                Ok(ptr_cast_units_value(
                    value,
                    *from_size,
                    &|id| self.alloc_index_is_typed(id),
                ))
            }
            Rvalue::PtrArrayBase(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                self.ptr_array_base_value(value)
            }
            Rvalue::PtrFieldBase(operand, depth) => {
                let value = self.eval_operand(frames, body, operand)?;
                self.ptr_field_base_value(value, *depth)
            }
            Rvalue::WrapArrayElems(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                Self::wrap_array_elems(value)
            }
            Rvalue::UnwrapArrayElems(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                Self::unwrap_array_elems(value)
            }
            Rvalue::UnwrapSliceElems(operand) => {
                // A pointer cast never rewrites the storage it
                // denotes: the same allocation can still be viewed
                // through pointers of the other element type, and
                // that view must keep observing the original
                // values. The fat pointer is unchanged, and each
                // element's representation difference between the
                // two views is reconciled where the element is
                // actually read or written through a pointer (the
                // entry/peel judgments at access points), not here.
                let value = self.eval_operand(frames, body, operand)?;
                Ok(value)
            }
            Rvalue::WrapSliceElems(operand) => {
                // See `UnwrapSliceElems`: the cast preserves the
                // backing storage; only the resulting pointer's
                // element view differs.
                let value = self.eval_operand(frames, body, operand)?;
                Ok(value)
            }
            Rvalue::BytesToStr(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                Ok(self.bytes_as_str(frames, &value))
            }
            Rvalue::IntToPtr(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                match value {
                    Value::Int(i) => Ok(self.materialized_ptr(i as u64)),
                    Value::Uint(u) => Ok(self.materialized_ptr(u as u64)),
                    other => Err(InterpError::type_mismatch(
                        "integer-to-pointer transmute",
                        "an integer value",
                        other.kind_name(),
                    )),
                }
            }
            Rvalue::IntToPtrWrapped {
                operand,
                payload_index,
                field_count,
                fill,
            } => {
                let value = self.eval_operand(frames, body, operand)?;
                let addr = match value {
                    Value::Int(i) => i as u64,
                    Value::Uint(u) => u as u64,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "integer-to-pointer-wrapper transmute",
                            "an integer value",
                            other.kind_name(),
                        ))
                    }
                };
                self.ptr_wrapped_value(addr, *payload_index, *field_count, fill)
            }
            Rvalue::ScalarToPtrWrapped {
                operand,
                from,
                payload_index,
                field_count,
                fill,
            } => {
                let value = self.eval_operand(frames, body, operand)?;
                let addr = self.flatten_scalar(&value, from)? as u64;
                self.ptr_wrapped_value(addr, *payload_index, *field_count, fill)
            }
            Rvalue::EnumToPtrWrapped {
                operand,
                discriminants,
                payload_index,
                field_count,
                fill,
            } => {
                let value = self.eval_operand(frames, body, operand)?;
                let addr = match value {
                    Value::Enum { variant, .. } => {
                        let discr = if discriminants.is_empty() {
                            variant as i128
                        } else {
                            *discriminants.get(variant as usize).ok_or_else(
                                || {
                                    InterpError::type_mismatch(
                                        "enum-to-pointer-wrapper transmute",
                                        "variant covered by the discriminant table",
                                        format!("variant {variant}"),
                                    )
                                },
                            )?
                        };
                        discr as u64
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            "enum-to-pointer-wrapper transmute",
                            "an enum value",
                            other.kind_name(),
                        ))
                    }
                };
                self.ptr_wrapped_value(addr, *payload_index, *field_count, fill)
            }
            Rvalue::EnumToInt(operand, discriminants, ty) => {
                let value = self.eval_operand(frames, body, operand)?;
                match value {
                    Value::Enum { variant, .. } => {
                        let discr = if discriminants.is_empty() {
                            variant as i128
                        } else {
                            *discriminants.get(variant as usize).ok_or_else(|| {
                                InterpError::type_mismatch(
                                    "enum-to-integer transmute",
                                    "variant covered by the discriminant table",
                                    format!("variant {variant}"),
                                )
                            })?
                        };
                        eval_cast(&Value::Int(discr), None, ty)
                    }
                    other => Err(InterpError::type_mismatch(
                        "enum-to-integer transmute",
                        "an enum value",
                        other.kind_name(),
                    )),
                }
            }
            Rvalue::IntToEnum(operand, discriminants) => {
                let value = self.eval_operand(frames, body, operand)?;
                let bits = match value {
                    Value::Int(i) => i as u128,
                    Value::Uint(u) => u,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "integer-to-enum transmute",
                            "an integer value",
                            other.kind_name(),
                        ))
                    }
                };
                match discriminants.iter().position(|d| *d as u128 == bits) {
                    Some(variant) => Ok(Value::Enum {
                        variant: variant as u32,
                        fields: Vec::new(),
                    }),
                    None => Err(InterpError::type_mismatch(
                        "integer-to-enum transmute",
                        "a discriminant value of the target enum",
                        format!("integer {bits}"),
                    )),
                }
            }
            Rvalue::SliceEnumToInt(operand, discriminants, ty) => {
                let value = self.eval_operand(frames, body, operand)?;
                self.map_slice_pointer_elems(
                    frames,
                    &value,
                    |elem| match elem {
                        Value::Enum { variant, fields } if fields.is_empty() => {
                            let discr = if discriminants.is_empty() {
                                *variant as i128
                            } else {
                                *discriminants.get(*variant as usize).ok_or_else(|| {
                                    InterpError::type_mismatch(
                                        "slice enum-to-integer transmute",
                                        "variant covered by the discriminant table",
                                        format!("variant {variant}"),
                                    )
                                })?
                            };
                            eval_cast(&Value::Int(discr), None, ty)
                        }
                        other => Err(InterpError::type_mismatch(
                            "slice enum-to-integer transmute",
                            "fieldless enum element",
                            other.kind_name(),
                        )),
                    },
                )
            }
            Rvalue::SliceIntToEnum(operand, discriminants) => {
                let value = self.eval_operand(frames, body, operand)?;
                self.map_slice_pointer_elems(
                    frames,
                    &value,
                    |elem| {
                        let bits = match elem {
                            Value::Int(i) => *i as u128,
                            Value::Uint(u) => *u,
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "slice integer-to-enum transmute",
                                    "integer element",
                                    other.kind_name(),
                                ))
                            }
                        };
                        match discriminants.iter().position(|d| *d as u128 == bits) {
                            Some(variant) => Ok(Value::Enum {
                                variant: variant as u32,
                                fields: Vec::new(),
                            }),
                            None => Err(InterpError::type_mismatch(
                                "slice integer-to-enum transmute",
                                "a discriminant value of the target enum",
                                format!("integer {bits}"),
                            )),
                        }
                    },
                )
            }
            Rvalue::IntToEnumWrapped {
                operand,
                discriminants,
                payload_index,
                field_count,
                fill,
            } => {
                let value = self.eval_operand(frames, body, operand)?;
                let bits = match value {
                    Value::Int(i) => i as u128,
                    Value::Uint(u) => u,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "integer-to-enum-wrapper transmute",
                            "an integer value",
                            other.kind_name(),
                        ))
                    }
                };
                let variant = discriminants
                    .iter()
                    .position(|d| *d as u128 == bits)
                    .ok_or_else(|| {
                        InterpError::type_mismatch(
                            "integer-to-enum-wrapper transmute",
                            "a discriminant value of the wrapped enum",
                            format!("integer {bits}"),
                        )
                    })?;
                let mut fields = vec![Value::Uninit; *field_count];
                fields[*payload_index as usize] = Value::Enum {
                    variant: variant as u32,
                    fields: Vec::new(),
                };
                for (index, constant) in fill {
                    fields[*index as usize] = self.eval_constant(constant)?;
                }
                Ok(Value::Enum {
                    variant: 0,
                    fields,
                })
            }
            Rvalue::Reinterpret(operand, src, dst) => {
                let value = self.eval_operand(frames, body, operand)?;
                let r = self.reinterpret_value(&value, src, dst);
                r
            }
            Rvalue::ShapedLoad(place, shape, moved) => {
                let target = self.resolve_place(frames, body, place)?;
                // A shaped load through a foreign view (a
                // reference transmuted to a different ADT over
                // the same storage): the byte view owns it when
                // it can judge it (see `Interpreter::foreign_read`).
                // A moved load keeps the shaped path: its
                // clearing write below is a structural one.
                if !*moved {
                    if let Some(read) =
                        self.foreign_read(frames, body, place, &target)
                    {
                        return read;
                    }
                }
                let place_ty = self.place_result_ty(body, place);
                let value =
                    self.shaped_load(frames, &target, shape, place_ty.as_ref())?;
                if *moved {
                    // A move leaves the source place
                    // uninitialised, exactly as a plain move of
                    // the same place does in `eval_operand`: the
                    // value now lives only in the destination,
                    // and drop glue reaching the source later (a
                    // box's pointee glue, say) finds nothing to
                    // run a second time. A target that resolved
                    // to a value (a constant aggregate) has no
                    // storage to clear.
                    match &target {
                        Resolved::Place { .. }
                        | Resolved::Static { .. }
                        | Resolved::Alloc { .. } => {
                            self.write_resolved(frames, &target, Value::Uninit)?;
                        }
                        Resolved::Value(..) => {}
                    }
                }
                Ok(value)
            }
            Rvalue::PtrOffset(ptr_op, count_op, baked_size, count_width) => {
                let base = self.eval_operand(frames, body, ptr_op)?;
                let count_value = self.eval_operand(frames, body, count_op)?;
                let count = match count_value {
                    Value::Int(v) => v,
                    Value::Uint(v) => {
                        let v = i128::try_from(v).map_err(|_| {
                            InterpError::UnsupportedRvalue {
                                kind: "pointer offset: count does not fit in an isize"
                                    .to_string(),
                            }
                        })?;
                        if *count_width > 0 && *count_width < 128 {
                            // The count arrived in an unsigned-typed
                            // operand (the `byte_*` family passes its
                            // `usize` count straight through): its
                            // bits are the `isize` count.
                            let w = *count_width;
                            let masked = v & ((1i128 << w) - 1);
                            if (masked >> (w - 1)) & 1 == 1 {
                                masked - (1i128 << w)
                            } else {
                                masked
                            }
                        } else {
                            v
                        }
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            "pointer offset count",
                            "integer count",
                            other.kind_name(),
                        ))
                    }
                };
                let elem_size = baked_size.or_else(|| self.offset_elem_size_of(body, ptr_op));
                self.ptr_offset_value(frames, base, count, elem_size, false)
            }
            Rvalue::BoxDerefTransmute(operand) => {
                // The cast yields a pointer to the box's pointee.
                // Two box shapes exist in this value model. A box
                // built from its MIR aggregates is the wrapper chain
                // `Unique` around `NonNull` around the pointer, and
                // unwrapping the chain reaches the pointer value
                // itself. A box made by the `Box::new` builtin instead
                // holds its pointee directly as its single field, so
                // the chain bottoms out at the pointee value (or at a
                // reference that *is* the pointee, when the pointee
                // is itself a reference); the result is then a
                // reference to the pointee's place, which is what
                // reading or writing through the extracted pointer
                // must reach. `box_cast_places` tells the two apart:
                // it reports how many wrapper layers an
                // aggregate-built value carries above its pointer.
                let places = self.box_cast_places(frames, body, operand);
                let original = self.eval_operand(frames, body, operand)?;
                let mut v = original.clone();
                let mut layers = 0usize;
                loop {
                    match v {
                        Value::DynRef { .. } => {
                            break Ok(v)
                        }
                        Value::Ref(_) | Value::FatRef { .. } => {
                            match &places {
                                Some((_, pointee, depth))
                                    if layers < *depth
                                        && (layers == 0
                                            || matches!(
                                                pointee,
                                                Resolved::Alloc { .. }
                                            )) =>
                                {
                                    break Ok(Self::pointee_ref(
                                        frames, pointee,
                                    ));
                                }
                                _ => break Ok(v),
                            }
                        }
                        Value::Enum { mut fields, .. } if fields.len() == 1 => {
                            v = fields.pop().expect("one field");
                            layers += 1;
                        }
                        // A wrapper layer with siblings: the real
                        // layout of the chain the classification
                        // names (`Box` over its allocator field,
                        // `Unique` over its `PhantomData` marker)
                        // carries the next layer at field 0 and
                        // unrelated fields beside it. Descend field
                        // 0 while it continues the chain (an
                        // aggregate layer or the pointer itself);
                        // any other field-0 shape is not a wrapper
                        // chain and keeps the strict reading below.
                        // The classification names the chain from
                        // the operand place, so it cannot name a
                        // chain whose operand place projects
                        // through storage it does not model (a
                        // coroutine state slot, say); the value's
                        // own field-0 descent then arbitrates —
                        // a descent that verifiably bottoms out
                        // at a pointer is the same chain, while a
                        // value stored directly in the field (a
                        // builtin-built box's pointee) does not
                        // descend and keeps the strict reading.
                        Value::Enum { fields, .. }
                            if fields.len() > 1
                                && (places.is_some()
                                    || layered_wrapper_pointer(&fields[0])
                                        .is_some())
                                && matches!(
                                    fields[0],
                                    Value::Enum { .. }
                                        | Value::Ref(_)
                                        | Value::FatRef { .. }
                                        | Value::DynRef { .. }
                                ) =>
                        {
                            v = fields
                                .into_iter()
                                .next()
                                .expect("field 0 checked");
                            layers += 1;
                        }
                        other => {
                            // Builtin-built boxes whose operand the
                            // classification covers: it names the
                            // pointee place. A move out of the operand
                            // place cleared the storage the pointer
                            // must name, so the value is restored
                            // first. (Lane BH)
                            if let Some((source, pointee, _)) = &places {
                                if matches!(operand, Operand::Move(_)) {
                                    self.write_resolved(
                                        frames,
                                        source,
                                        original.clone(),
                                    )?;
                                }
                                break Ok(Self::pointee_ref(frames, pointee));
                            }
                            // A box built by the `Box::new` /
                            // `Box::new_uninit` builtins stores its
                            // pointee as the field value itself, not
                            // as a pointer chain. When the cast's
                            // operand place projects into the box, the
                            // pointer it stands for is a reference to
                            // that place; a resolved constant value is
                            // referenced as a value. (Lane BO)
                            if let Operand::Copy(place) | Operand::Move(place) = operand {
                                match self.resolve_place(frames, body, place)? {
                                    Resolved::Place {
                                        frame_pos,
                                        local,
                                        projection,
                                    } if !projection.is_empty() => {
                                        break Ok(Value::Ref(RefTarget::place(
                                            frames[frame_pos].id,
                                            local,
                                            projection,
                                        )));
                                    }
                                    Resolved::Value(base, projection) => {
                                        break Ok(Value::Ref(RefTarget::Value(
                                            Box::new(base),
                                            projection,
                                        )));
                                    }
                                    Resolved::Static { name, projection }
                                        if !projection.is_empty() =>
                                    {
                                        break Ok(Value::Ref(RefTarget::Static {
                                            name,
                                            projection,
                                        }));
                                    }
                                    Resolved::Alloc { id, projection }
                                        if !projection.is_empty() =>
                                    {
                                        break Ok(Value::Ref(RefTarget::Alloc {
                                            id,
                                            projection,
                                        }));
                                    }
                                    Resolved::Place { .. }
                                    | Resolved::Static { .. }
                                    | Resolved::Alloc { .. } => {}
                                }
                            }
                            break Err(InterpError::type_mismatch(
                                "BoxDerefTransmute cast",
                                "pointer wrapper (Unique/NonNull around a raw pointer)",
                                other.kind_name(),
                            ));
                        }
                    }
                }
            }
            Rvalue::Ref(place) => {
                // A borrow of exactly the place a dyn fat pointer
                // points at (`&*r`, `&*box`) is the fat pointer
                // itself: the result is again a `&dyn`, with the
                // same data pointer and the same vtable identity.
                if place.projection.len() == 1
                    && matches!(place.projection[0], ProjectionElem::Deref)
                {
                    let head = Place {
                        local: place.local,
                        projection: Vec::new(),
                    };
                    if let Ok(value @ Value::DynRef { .. }) =
                        self.read_place(frames, body, &head)
                    {
                        return Ok(value);
                    }
                }
                // A borrow of a vector's length slot (`&mut
                // vec.len` — the reference `Vec::extend_with`
                // hands its `SetLenOnDrop` guard, which writes
                // the final length through it when it drops):
                // the slot is not storage in the element-list
                // value, so resolving the place itself would
                // collapse it to the current length as a
                // detached value and the write through the
                // reference would have nowhere to land. The
                // reference denotes the vector's own target
                // with a length-slot step appended instead
                // (see [`RefProj::VecLen`]): reads through it
                // answer the declared length, and writes apply
                // the `Vec::set_len` storage change to the
                // vector itself.
                if let Some(base_place) = vec_len_slot_base(self.program(), body, place) {
                    let target = self.resolve_place(frames, body, &base_place)?;
                    let value = match target {
                        Resolved::Place {
                            frame_pos,
                            local,
                            mut projection,
                        } => {
                            projection.push(RefProj::VecLen);
                            Value::Ref(RefTarget::place(
                                frames[frame_pos].id,
                                local,
                                projection,
                            ))
                        }
                        Resolved::Value(base, mut projection) => {
                            projection.push(RefProj::VecLen);
                            Value::Ref(RefTarget::Value(
                                Box::new(base),
                                projection,
                            ))
                        }
                        Resolved::Static {
                            name,
                            mut projection,
                        } => {
                            projection.push(RefProj::VecLen);
                            Value::Ref(RefTarget::Static {
                                name,
                                projection,
                            })
                        }
                        Resolved::Alloc {
                            id,
                            mut projection,
                        } => {
                            projection.push(RefProj::VecLen);
                            Value::Ref(RefTarget::Alloc {
                                id,
                                projection,
                            })
                        }
                    };
                    return Ok(value);
                }
                let target = match self.resolve_place(frames, body, place) {
                    Ok(target) => target,
                    Err(err) => {
                        // A place that crosses a dereference of a
                        // storage-less pointer (a null pointer, a
                        // dangling address) denotes no storage, so
                        // no resolved place names it -- but the
                        // pointer being formed is still
                        // meaningful: it is the address the
                        // projection denotes.
                        match self.resolve_place_as_address(
                            frames, body, place, dest_ty,
                        ) {
                            Some(value) => return Ok(value),
                            None => return Err(err),
                        }
                    }
                };
                // A borrow whose own type is a trait object, of
                // a place that is not itself a fat pointer (the
                // unsized tail of a DST struct behind a fat
                // pointer, or a place holding a fat pointer):
                // the borrow still denotes a trait object, so
                // its `dyn` identity is recovered from the fat
                // pointer attached to the place (see
                // `dyn_borrow_source`). Without it the borrow
                // is a thin reference whose method calls never
                // engage virtual dispatch.
                let dyn_source = match dest_ty {
                    Some(Ty::Ref(inner)) => match &**inner {
                        Ty::Dynamic(dest_traits) => self.dyn_borrow_source(
                            frames,
                            body,
                            place,
                            &target,
                            dest_traits,
                        ),
                        _ => None,
                    },
                    _ => None,
                };
                // A pointer to an unsized pointee (a slice or
                // `str`) is a fat pointer: it must carry the
                // pointee's length as its metadata, or `PtrMetadata`
                // and the slice consumers downstream see a thin
                // pointer whose metadata was dropped when the place
                // was resolved. The length is the element (or byte)
                // count of the value the place currently holds.
                // Where that value cannot be read (an uninitialised
                // or address-only pointee), the thin pointer is
                // formed as before and the metadata consumers report
                // the gap at their own site.
                let fat_len = match dest_ty {
                    Some(Ty::Ref(inner))
                        if matches!(&**inner, Ty::Slice(_) | Ty::Str) =>
                    {
                        self.fat_deref_len(frames, body, place)
                            .or_else(|| {
                                self.read_resolved(frames, &target)
                                    .ok()
                                    .and_then(|v| v.len())
                            })
                            .or_else(|| self.fat_tail_field_len(frames, body, place))
                    }
                    // A borrow of a fat pointer's pointee itself
                    // (`&*p` with `p` fat): the pointee is
                    // unsized whatever its type — a DST struct
                    // like `CStr`, whose only field is its
                    // unsized tail, as much as a slice — so the
                    // borrow carries the pointer's length as its
                    // metadata. A thin pointer yields no length
                    // here and the borrow stays thin, as before.
                    Some(Ty::Ref(inner)) => self
                        .fat_deref_len(frames, body, place)
                        .or_else(|| {
                            // A borrow of a fat pointee's
                            // unsized tail field (a byte-string
                            // wrapper's inner field, say): the
                            // field's metadata is the enclosing
                            // pointer's length, recovered from
                            // the place's dereference prefix.
                            if self.ty_is_unsized(inner) {
                                self.fat_prefix_deref_len(frames, body, place)
                            } else {
                                None
                            }
                        }),
                    _ => None,
                };
                let thin = match target {
                    Resolved::Place {
                        frame_pos,
                        local,
                        projection,
                    } => Value::Ref(RefTarget::place(
                        frames[frame_pos].id,
                        local,
                        projection,
                    )),
                    Resolved::Value(base, projection) => {
                        Value::Ref(RefTarget::Value(Box::new(base), projection))
                    }
                    Resolved::Static { name, projection } => {
                        Value::Ref(RefTarget::Static { name, projection })
                    }
                    Resolved::Alloc { id, projection } => {
                        Value::Ref(RefTarget::Alloc { id, projection })
                    }
                };
                match dyn_source {
                    Some(DynBorrowSource::Stored(v)) => return Ok(v),
                    Some(DynBorrowSource::Crossed {
                        concrete,
                        traits,
                        drop_glue,
                    }) => {
                        return Ok(Value::DynRef {
                            base: Box::new(thin),
                            concrete,
                            traits,
                            drop_glue,
                            // Runtime recovery has no conversion-time
                            // segment source: tag comparison applies.
                            type_id: None,
                            // Nor a vtable resolution source:
                            // dispatch matches spellings, as before.
                            vtable_methods: Vec::new(),
                        });
                    }
                    None => {}
                }
                match fat_len {
                    Some(len) => Ok(Value::FatRef {
                        base: Box::new(thin),
                        len,
                    }),
                    None => Ok(thin),
                }
            }
            Rvalue::Len(place) => {
                let value = self.read_place(frames, body, place)?;
                let len = self.fat_deref_len(frames, body, place);
                let len = match len {
                    Some(len) => Some(len),
                    None => value
                        .len()
                        .or_else(|| self.fat_tail_field_len(frames, body, place)),
                };
                let len = len.ok_or_else(|| {
                    InterpError::type_mismatch(
                        format!("Len of {place}"),
                        "array, slice or string value",
                        value.kind_name(),
                    )
                })?;
                Ok(Value::Uint(len))
            }
            Rvalue::Discriminant(place, discriminants) => {
                let value = self.read_place(frames, body, place)?;
                match value {
                    Value::Enum { variant, .. } | Value::Coroutine { variant, .. } => {
                        if discriminants.is_empty() {
                            // No table was recorded for this enum:
                            // its discriminants are the sequential
                            // defaults, so the index is the value.
                            Ok(Value::Int(variant as i128))
                        } else {
                            let discr = discriminants
                                .get(variant as usize)
                                .ok_or_else(|| {
                                    InterpError::type_mismatch(
                                        format!("Discriminant of {place}"),
                                        "variant covered by the discriminant table",
                                        format!("variant {variant}"),
                                    )
                                })?;
                            Ok(Value::Int(*discr))
                        }
                    }
                    // A non-ADT value (a scalar, a tuple, a
                    // reference, ...) has no variants: its
                    // discriminant is 0. This is the shape
                    // `mem::discriminant(&10)` produces — the
                    // rvalue is well-defined for every type, and
                    // only enum-like values carry a real tag.
                    _ => Ok(Value::Int(0)),
                }
            }
            Rvalue::CopyForDeref(place) => {
                let value = self.read_place(frames, body, place)?;
                match &value {
                    Value::Ref(_) | Value::FatRef { .. } => {
                        let target = self.deref_value(frames, &value)?;
                        self.read_resolved(frames, &target)
                    }
                    _ => Ok(value),
                }
            }
            Rvalue::Reborrow(place) => {
                // A bitwise copy of the pointer value held at the
                // place, reinterpreted as the target pointer type.
                // The value model tracks neither pointer mutability
                // nor provenance, and a CoerceShared pair has equal
                // layout by contract, so the copy is the value itself
                // (for a wrapper ADT around a pointer, the aggregate
                // itself, exactly as a bitwise copy prescribes).
                self.read_place(frames, body, place)
            }
            Rvalue::ThreadLocalRef(name) => {
                // A reference to the executing thread's storage for
                // this thread-local static, seeded on first access
                // from the initializer the driver recorded. The
                // storage's own machinery (std's `LocalKey` and
                // storage bodies, which are ordinary collected code)
                // performs the user-visible initialization; the seed
                // is only the storage struct's initial value.
                // Destructors registered for a storage run when
                // the owning thread finishes (see `run_thread`);
                // the entries themselves stay in the store under
                // the thread's unique id.
                let Some(seed) = self.program.thread_locals.get(name) else {
                    return Err(InterpError::UnsupportedRvalue {
                        kind: format!(
                            "ThreadLocalRef {name}: no thread-local definition \
                             was collected for this static (its initializer \
                             could not be decoded)"
                        ),
                    });
                };
                let key = Self::tls_store_key(self.current_thread_id(), name);
                self.ensure_static(&key, seed)?;
                Ok(Value::Ref(RefTarget::Static {
                    name: key,
                    projection: Vec::new(),
                }))
            }
            Rvalue::Unsupported(kind) => Err(InterpError::UnsupportedRvalue {
                kind: kind.clone(),
            }),
        }
    }
}

/// The bytes of an OS string, for the fabricated `envp`:
/// exact on Unix, lossy elsewhere (the environment the
/// `env` builtins read is the host's own).
#[cfg(unix)]
fn os_str_bytes(value: &std::ffi::OsStr) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    value.as_bytes().to_vec()
}

/// The bytes of an OS string, for the fabricated `envp`
/// (non-Unix form; see the Unix form above).
#[cfg(not(unix))]
fn os_str_bytes(value: &std::ffi::OsStr) -> Vec<u8> {
    value.to_string_lossy().into_owned().into_bytes()
}
