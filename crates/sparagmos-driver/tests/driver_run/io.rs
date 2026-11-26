use super::{Command, fixture, run_driver, run_ok, temp_dir};


const PRINTLN_SRC: &str = r#"
fn main() { println!("hi"); }
"#;

#[test]
fn run_mode_println_prints_to_stdout() {
    let dir = temp_dir("println");
    let src = fixture(&dir, "println.rs", PRINTLN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "println! runs through the interpreter's runtime print support; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hi\n",
        "the guest's stdout is the driver's stdout"
    );
}

const LOG_SYNTAX_SRC: &str = r#"
#![feature(log_syntax)]
fn main() {
    log_syntax!("compile-time marker");
    println!("guest marker");
}
"#;


#[test]
fn run_mode_log_syntax_output_stays_off_stdout() {
    // `log_syntax!` prints to the *compiler's* stdout at compile
    // time. The compiler runs in-process, so without redirection
    // that output would land in the driver's stdout — which run
    // mode reserves for the guest program's output. It must appear
    // on stderr (compiler-side diagnostics) instead, leaving stdout
    // byte-identical to the guest's own output.
    let dir = temp_dir("logsyntax");
    let src = fixture(&dir, "logsyntax.rs", LOG_SYNTAX_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program itself runs fine; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "guest marker\n",
        "stdout carries the guest's output alone"
    );
    assert!(
        stderr.contains("compile-time marker"),
        "the log_syntax output is redirected to stderr: {stderr}"
    );
}

const VEC_CELL_INDEX_SRC: &str = r#"use std::cell::Cell;
fn main() {
    let mut v = vec![Cell::new(1u32), Cell::new(2u32)];
    v[0].set(5);
    v[1].set(v[0].get() + 1);
    assert_eq!(v[0].get(), 5);
    assert_eq!(v[1].get(), 6);
    println!("vec cell ok");
    // The vector is dropped at the end of `main`: the `Drop`
    // terminator for the cell element glue runs as well.
}
"#;

#[test]
fn run_mode_shared_index_write_through_cell() {
    // `v[i].set(..)` mutates through the *shared* element
    // reference `Index::index` returns: the reference must name
    // the vector's storage (a live place), not a snapshot of the
    // element, or the `Cell` write has no place to land.
    let dir = temp_dir("vec-cell-index");
    let src = fixture(&dir, "vec_cell.rs", VEC_CELL_INDEX_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "vec cell ok\n");
}

const PRINTLN_DYNAMIC_COUNT_SRC: &str = r#"fn main() {
    let w = 6usize;
    println!("{:w$}", 42);
    println!("{:.*}", 2, 3.14159f64);
}
"#;

#[test]
fn run_mode_println_dynamic_width_and_precision() {
    // A dynamic width / precision (`{:w$}`, `{:.*}`) reaches
    // the formatter through `rt::Argument::from_usize`, which
    // current core calls with a *reference* to the count (the
    // format-args capture tuple holds references): the builtin
    // reads through the reference to the count value.
    let dir = temp_dir("println-dynamic-count");
    let src = fixture(&dir, "println_count.rs", PRINTLN_DYNAMIC_COUNT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "    42\n3.14\n");
}

const ERROR_FROM_KIND_SRC: &str = r#"
use std::io::ErrorKind;
fn main() {
    let e = std::io::Error::from(ErrorKind::PermissionDenied);
    assert_eq!(e.kind(), ErrorKind::PermissionDenied);
    assert_eq!(e.raw_os_error(), None);
    let e: std::io::Error = ErrorKind::Other.into();
    assert_eq!(e.kind(), ErrorKind::Other);
    println!("error-from-kind-ok");
}
"#;

#[test]
fn run_mode_io_error_from_error_kind() {
    let dir = temp_dir("error-from-kind");
    let src = fixture(&dir, "error_from_kind.rs", ERROR_FROM_KIND_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "error-from-kind-ok\n"
    );
}

const IO_ERROR_INTO_INNER_SRC: &str = r#"
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

static DROPS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct Counted { code: i32 }
impl fmt::Display for Counted {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "counted {}", self.code) }
}
impl std::error::Error for Counted {}
impl Drop for Counted {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
        println!("dropping Counted {}", self.code);
    }
}

fn main() {
    let e = std::io::Error::new(std::io::ErrorKind::InvalidInput, Counted { code: 5 });
    match e.into_inner() {
        Some(boxed) => {
            println!("got inner");
            println!("drops before box drop: {}", DROPS.load(Ordering::SeqCst));
            drop(boxed);
            println!("drops after box drop: {}", DROPS.load(Ordering::SeqCst));
        }
        None => println!("no inner"),
    }
    println!("done");
}
"#;

#[test]
fn run_mode_io_error_into_inner_drops_payload_once() {
    // `io::Error::into_inner` rebuilds the payload's
    // `Box<dyn Error + Send + Sync>` with `Box::from_raw` over
    // the pointer the packed representation carried. Reading
    // the pointee back through that cast descends the pointer
    // wrappers (`Box` over `Unique` over the raw pointer, each
    // carrying trailing zero-sized fields), and dropping the
    // rebuilt box reaches the `DynRef` at the bottom of the
    // same wrapper chain - the payload destructor runs exactly
    // once, when the box drops, not inside `into_inner`.
    let dir = temp_dir("io-error-into-inner");
    let src = fixture(&dir, "io_error_into_inner.rs", IO_ERROR_INTO_INNER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "got inner\n\
         drops before box drop: 0\n\
         dropping Counted 5\n\
         drops after box drop: 1\n\
         done\n"
    );
}

const IO_ERROR_CUSTOM_SRC: &str = r#"
use std::error::Error;
use std::fmt;

#[derive(Debug)]
struct MyErr;
impl fmt::Display for MyErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "my error display")
    }
}
impl Error for MyErr {}

fn main() {
    let e = std::io::Error::new(std::io::ErrorKind::InvalidInput, MyErr);
    println!("kind: {:?}", e.kind());
    println!("display: {e}");
    println!("debug: {e:?}");
    let e2 = std::io::Error::new(std::io::ErrorKind::Other, MyErr);
    println!("other debug: {e2:?}");
    let e3 = std::io::Error::new(std::io::ErrorKind::InvalidData, "literal payload");
    println!("str: {e3} | {e3:?}");
    let e4 = std::io::Error::new(std::io::ErrorKind::NotFound, String::from("owned payload"));
    println!("string: {e4} | {e4:?}");
    let e5 = std::io::Error::new(std::io::ErrorKind::InvalidInput, MyErr);
    match e5.get_ref() {
        Some(r) => println!("get_ref: {r:?}"),
        None => println!("get_ref none"),
    }
    println!("done");
}
"#;

#[test]
fn run_mode_io_error_custom_payload() {
    // `io::Error::new` with a custom payload: the packed
    // representation's word names the payload's allocation by
    // registered synthetic address, and every access goes
    // through the reverse registry - `kind()` decodes in std's
    // own body, `Display` forwards to the payload, `Debug` is
    // the `Custom { kind, error }` struct (with the kind named
    // as an `ErrorKind` value: `Other`'s variant index is not
    // its discriminant), and `get_ref` reaches the payload
    // through `decode_repr`'s closure.
    let dir = temp_dir("io-error-custom");
    let src = fixture(&dir, "io_error_custom.rs", IO_ERROR_CUSTOM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "kind: InvalidInput\n\
         display: my error display\n\
         debug: Custom { kind: InvalidInput, error: MyErr }\n\
         other debug: Custom { kind: Other, error: MyErr }\n\
         str: literal payload | Custom { kind: InvalidData, error: \"literal payload\" }\n\
         string: owned payload | Custom { kind: NotFound, error: \"owned payload\" }\n\
         get_ref: MyErr\n\
         done\n"
    );
}

const IO_ERROR_CUSTOM_DROP_SRC: &str = r#"
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

static DROPS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct Counted;
impl fmt::Display for Counted {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "counted error")
    }
}
impl std::error::Error for Counted {}
impl Drop for Counted {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

struct Holder {
    err: std::io::Error,
}

fn main() {
    {
        let e = std::io::Error::new(std::io::ErrorKind::Other, Counted);
        println!("in scope: {e}");
        println!("mid drops: {}", DROPS.load(Ordering::SeqCst));
    }
    println!("after scope drops: {}", DROPS.load(Ordering::SeqCst));
    {
        let h = Holder {
            err: std::io::Error::new(std::io::ErrorKind::Other, Counted),
        };
        println!("holder: {}", h.err);
    }
    println!("after holder drops: {}", DROPS.load(Ordering::SeqCst));
    let inner = std::io::Error::new(std::io::ErrorKind::Other, Counted);
    let outer = std::io::Error::new(std::io::ErrorKind::InvalidData, inner);
    drop(outer);
    println!("after nested drops: {}", DROPS.load(Ordering::SeqCst));
    let e5 = std::io::Error::new(std::io::ErrorKind::Other, Counted);
    std::mem::forget(e5);
    println!("after forget drops: {}", DROPS.load(Ordering::SeqCst));
}
"#;

#[test]
fn run_mode_io_error_custom_payload_drop() {
    // Dropping an `io::Error` in the `Custom` representation
    // runs the payload's destructor exactly once - at scope
    // end, through a struct field, and through a nested error
    // payload - and a forgotten error runs it never: the packed
    // representation's drop plan judges the word's tag at run
    // time and releases the registered allocation's trait
    // object.
    let dir = temp_dir("io-error-custom-drop");
    let src = fixture(&dir, "io_error_custom_drop.rs", IO_ERROR_CUSTOM_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "in scope: counted error\n\
         mid drops: 0\n\
         after scope drops: 1\n\
         holder: counted error\n\
         after holder drops: 2\n\
         after nested drops: 3\n\
         after forget drops: 3\n"
    );
}

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

const GUEST_STDIN_SRC: &str = r#"
use std::io::BufRead;
fn main() {
    let mut line = String::new();
    let n = std::io::stdin().lock().read_line(&mut line).unwrap();
    println!("got {}: {}", n, line.trim());
}
"#;

#[test]
fn run_mode_guest_reads_its_own_stdin() {
    // The guest's `std::io::stdin` is the run's own stdin
    // in the file table: a line read consumes the piped
    // bytes the driver process was fed.
    let dir = temp_dir("guest-stdin");
    let src = fixture(&dir, "gueststdin.rs", GUEST_STDIN_SRC);
    let mut child = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn sparagmos-driver");
    use std::io::Write;
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"hello\n")
        .unwrap();
    let out = child.wait_with_output().expect("driver run failed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "guest-stdin guest must exit 0; stdout={stdout} stderr={stderr}"
    );
    assert_eq!(stdout, "got 6: hello\n");
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
