// `std::io::Error` representation tests.

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
