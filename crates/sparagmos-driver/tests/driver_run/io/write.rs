// `std::io::Write` and `write_fmt` tests.

const STDOUT_WRITE_SRC: &str = r#"
use std::io::Write;
fn main() {
    let mut out = std::io::stdout();
    let n = out.write(b"direct write\n").unwrap();
    assert_eq!(n, 13);
    out.write_all(b"write all\n").unwrap();
    out.flush().unwrap();
    let mut v: Vec<u8> = Vec::new();
    v.write_all(b"vec bytes").unwrap();
    assert_eq!(v, b"vec bytes");
    println!("write ok");
}
"#;

#[test]
fn run_mode_write_on_stdout_and_vec() {
    // `Write::write` / `write_all` / `flush` on the standard
    // streams and on a `Vec<u8>` are served directly (only
    // `write_fmt` was, before): the bytes land on the
    // captured stream / in the vector.
    run_ok(
        "stdout-write",
        STDOUT_WRITE_SRC,
        "direct write\nwrite all\nwrite ok\n",
    );
}

const DYN_WRITE_SRC: &str = r#"
use std::io::Write;

fn sink(w: &mut dyn Write) -> std::io::Result<()> {
    w.write_all(b"ok")?;
    w.flush()
}

fn main() {
    let mut boxed: Box<dyn Write> = Box::new(Vec::<u8>::new());
    boxed.write_all(b"box").unwrap();
    boxed.flush().unwrap();
    let mut v: Vec<u8> = Vec::new();
    sink(&mut v).unwrap();
    println!("{}", String::from_utf8(v).unwrap());
}
"#;

const WRITE_FMT_ERROR_PROPAGATION_SRC: &str = r#"
use std::fmt;
use std::io::{self, Error, Write};
use std::panic::catch_unwind;

struct ErrorDisplay;

impl fmt::Display for ErrorDisplay {
    fn fmt(&self, _: &mut fmt::Formatter) -> fmt::Result {
        Err(fmt::Error)
    }
}

struct ErrorWriter;

const WRITER_ERROR: io::ErrorKind = io::ErrorKind::NotConnected;

impl Write for ErrorWriter {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        Err(Error::new(WRITER_ERROR, "not connected"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn main() {
    // The writer's own error propagates as the call's `Err`
    // (the default `write_fmt` body runs over the guest
    // `write`).
    let res = write!(ErrorWriter, "abc");
    assert!(res.is_err(), "writer error did not propagate");

    // A formatter error on a builtin-served sink panics with
    // core's message, catchably.
    let res = catch_unwind(|| write!(vec![], "{} {} {}", 1, ErrorDisplay, "bar"));
    let err = res
        .expect_err("formatter error did not lead to panic")
        .downcast::<&str>()
        .unwrap();
    assert!(
        err.contains("formatting trait implementation returned an error"),
        "unexpected panic: {}",
        err
    );

    // A writer error on an early piece pre-empts a later
    // formatter error: `fmt::write` stops at the failed
    // piece, so the result is the writer's error, not a
    // panic.
    let res = write!(ErrorWriter, "abc {}", ErrorDisplay);
    assert!(res.is_err(), "writer error did not propagate");
    assert_eq!(res.unwrap_err().kind(), WRITER_ERROR);

    // The formatter error comes first here: the panic is
    // raised by the default `write_fmt` body over the guest
    // writer (its adapter recorded no write error).
    let res = catch_unwind(|| write!(ErrorWriter, "{} abc", ErrorDisplay));
    let err = res
        .expect_err("formatter error did not lead to panic")
        .downcast::<&str>()
        .unwrap();
    assert!(
        err.contains("formatting trait implementation returned an error"),
        "unexpected panic: {}",
        err
    );
    println!("write fmt errors ok");
}
"#;

#[test]
fn run_mode_dyn_write_to_vec_receiver() {
    // Dynamic dispatch into std's impl-block bodies: the
    // `Write` impl for `Vec<u8>` is stored under the impl-block
    // path spelling (`alloc::io::impls::<impl Write for
    // Vec<u8>>::write_all`), which the virtual-call scan must
    // accept alongside the `<Self as Trait>::method` spelling.
    run_ok("dyn-write-vec", DYN_WRITE_SRC, "ok\n");
}

#[test]
fn run_mode_write_fmt_error_propagation() {
    // `Write::write_fmt` for a `Self` type the builtin does
    // not serve runs its collected instance body (the trait
    // default delegating to the type's `write_all`), and a
    // formatting error in a `write_fmt` call is core's panic
    // (on a builtin-served sink) or the default body's panic
    // over a guest writer -- never a runtime stop.
    run_ok(
        "write-fmt-error-propagation",
        WRITE_FMT_ERROR_PROPAGATION_SRC,
        "write fmt errors ok\n",
    );
}

const WRITE_FMT_CURSOR_AND_DYN_SRC: &str = r#"
use std::io::{Cursor, Write};

fn main() {
    let mut cursor = Cursor::new(Vec::new());
    write!(cursor, "n={}", 42).unwrap();
    let dyn_cursor: &mut dyn Write = &mut cursor;
    write!(dyn_cursor, " {}", "x").unwrap();
    let mut buf: Vec<u8> = Vec::new();
    {
        let dyn_buf: &mut dyn Write = &mut buf;
        write!(dyn_buf, "v{}", 7).unwrap();
    }
    println!("cursor: {}", String::from_utf8(cursor.into_inner()).unwrap());
    println!("buf: {}", String::from_utf8(buf).unwrap());
}
"#;

#[test]
fn run_mode_write_fmt_cursor_and_dyn() {
    // `write_fmt` on a `Cursor<Vec<u8>>` runs the collected
    // default body (its `write` appends at the cursor
    // position), and `write_fmt` through a `&mut dyn Write`
    // dispatches through the vtable to the concrete type's
    // instance body.
    run_ok(
        "write-fmt-cursor-and-dyn",
        WRITE_FMT_CURSOR_AND_DYN_SRC,
        "cursor: n=42 x\nbuf: v7\n",
    );
}

const DYN_STDOUT_SRC: &str = r#"
use std::io::{self, Write};

fn f(wr: &mut dyn Write) {
    wr.write_all(b"hello").ok().expect("failed");
}

fn main() {
    let mut wr = Box::new(io::stdout()) as Box<dyn Write>;
    f(&mut wr);
    let mut direct: Box<dyn Write> = Box::new(io::stdout());
    direct.write_all(b" direct").unwrap();
    println!(" done");
}
"#;

#[test]
fn run_mode_dyn_write_to_stdout_receiver() {
    // Dynamic dispatch of `Write` methods on a `Stdout`
    // receiver: `<Stdout as Write>::write` has no MIR body in
    // the prebuilt library, so no vtable body exists to
    // dispatch to; the file-io model serves the dispatch from
    // the concrete type the fat pointer carries, exactly as
    // it serves the statically-known call. Reaching the call
    // goes through the call-site coercion `&mut Box<dyn
    // Write>` -> `&mut dyn Write`, whose concrete type
    // likewise rides on the stored pointer.
    run_ok("dyn-write-stdout", DYN_STDOUT_SRC, "hello direct done\n");
}

const CALL_ARG_BOX_DYN_SRC: &str = r#"
use std::io::Write;
use std::sync::{Arc, Mutex};

struct Sink(Arc<Mutex<Vec<u8>>>);

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn f(wr: &mut dyn Write) {
    wr.write_all(b"hello").ok().expect("failed");
}

fn main() {
    let shared = Arc::new(Mutex::new(Vec::new()));
    let mut wr = Box::new(Sink(shared.clone())) as Box<dyn Write>;
    f(&mut wr);
    f(&mut wr);
    let v = shared.lock().unwrap();
    assert_eq!(&v[..], b"hellohello");
    println!("call-arg box dyn ok");
}
"#;

#[test]
fn run_mode_call_arg_box_dyn_to_ref_dyn() {
    // The call-site coercion of `tests/ui/coercion/
    // box-dyn-to-ref-dyn.rs`: conversion records no concrete
    // type for a dyn-to-dyn cast, so the interpreter reborrows
    // the trait-object pointer stored behind the reference;
    // the writes through the reborrow land on the concrete
    // value behind the box.
    run_ok(
        "call-arg-box-dyn",
        CALL_ARG_BOX_DYN_SRC,
        "call-arg box dyn ok\n",
    );
}
