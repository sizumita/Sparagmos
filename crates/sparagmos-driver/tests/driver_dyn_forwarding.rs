//! Run-mode regression tests for instance demand inside
//! vtable-demanded bodies.
//!
//! A trait impl that forwards a trait method to its fields (the
//! `tracing_subscriber::layer::Layered` shape) is commonly
//! reached only through a vtable: the impl body enters the
//! program through the dyn demand pass, not through any direct
//! call. The instances such a body itself demands -- here, the
//! trait *default* method bodies of its concrete field types,
//! which are formed into no vtable anywhere -- must still be
//! collected: the instance-demand pass has to scan
//! vtable-demanded bodies like any other converted body. When it
//! did not, the forwarding calls inside the impl body were left
//! with no converted body and the run stopped as unsupported.
//!
//! The second test series covers the dispatch side of the same
//! shape: when the forwarder type has a trailing *defaulted*
//! generic parameter, the fat pointer's rendered concrete type
//! keeps the defaulted argument while the instance name the
//! body is stored under elides it, so the exact impl-key lookup
//! misses and the base-name fallback scan is ambiguous between
//! the forwarder's instantiations. Dispatch must try the
//! argument-elided spellings before giving up.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture(dir: &Path, name: &str, src: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, src).unwrap();
    path
}

fn run_driver(src: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to run sparagmos-driver")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-dynfwd-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn assert_ok(out: &Output) {
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "run must succeed; stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("ok"),
        "guest prints ok; stdout={stdout} stderr={stderr}"
    );
}

/// The layered forwarder: `Layered::downcast_raw` answers for
/// itself and its field types by `TypeId`, then forwards to the
/// fields' own `downcast_raw` -- the trait default body for the
/// field types, which nothing ever unsizes to a trait object.
/// The forwarder itself is only ever used through `Box<dyn Sub>`.
const LAYERED_SRC: &str = r#"
use std::any::TypeId;

trait Sub: 'static {
    fn downcast_raw(&self, id: TypeId) -> Option<*const ()> {
        if id == TypeId::of::<Self>() {
            Some(self as *const Self as *const ())
        } else {
            None
        }
    }
    fn name(&self) -> &'static str;
}

struct A;
impl Sub for A {
    fn name(&self) -> &'static str {
        "A"
    }
}

struct B;
impl Sub for B {
    fn name(&self) -> &'static str {
        "B"
    }
}

struct Layered<L, S> {
    layer: L,
    inner: S,
}

impl<L, S> Sub for Layered<L, S>
where
    L: Sub + 'static,
    S: Sub + 'static,
{
    fn downcast_raw(&self, id: TypeId) -> Option<*const ()> {
        if id == TypeId::of::<Self>() {
            return Some(self as *const Self as *const ());
        }
        if id == TypeId::of::<L>() {
            return Some(&self.layer as *const L as *const ());
        }
        if id == TypeId::of::<S>() {
            return Some(&self.inner as *const S as *const ());
        }
        self.layer
            .downcast_raw(id)
            .or_else(|| self.inner.downcast_raw(id))
    }
    fn name(&self) -> &'static str {
        "Layered"
    }
}

// A generic registration function, like tracing's
// `Dispatch::new`: the unsizing cast lives inside this generic
// body, so the forwarder's vtable is demanded from its instance.
fn register<S: Sub + 'static>(s: S) -> Box<dyn Sub> {
    Box::new(s)
}

fn main() {
    let layered = Layered {
        layer: A,
        inner: B,
    };
    let d = register(layered);
    assert_eq!(d.name(), "Layered");
    assert!(d.downcast_raw(TypeId::of::<B>()).is_some());
    assert!(d.downcast_raw(TypeId::of::<Layered<A, B>>()).is_some());
    // Neither field type matches, so this call reaches the
    // forwarding calls to the fields' default bodies.
    assert!(d.downcast_raw(TypeId::of::<u8>()).is_none());
    println!("ok");
}
"#;

#[test]
fn run_mode_vtable_demanded_forwarder_default_bodies() {
    let dir = temp_dir("layered");
    let src = fixture(&dir, "layered.rs", LAYERED_SRC);
    assert_ok(&run_driver(&src));
}

/// The same shape nested: the outer forwarder's field is itself
/// a forwarder, so the default-body demand arises two
/// vtable-demanded bodies deep.
const NESTED_SRC: &str = r#"
use std::any::TypeId;

trait Sub: 'static {
    fn downcast_raw(&self, id: TypeId) -> Option<*const ()> {
        if id == TypeId::of::<Self>() {
            Some(self as *const Self as *const ())
        } else {
            None
        }
    }
    fn name(&self) -> &'static str;
}

struct A;
impl Sub for A {
    fn name(&self) -> &'static str {
        "A"
    }
}

struct B;
impl Sub for B {
    fn name(&self) -> &'static str {
        "B"
    }
}

struct C;
impl Sub for C {
    fn name(&self) -> &'static str {
        "C"
    }
}

struct Layered<L, S> {
    layer: L,
    inner: S,
}

impl<L, S> Sub for Layered<L, S>
where
    L: Sub + 'static,
    S: Sub + 'static,
{
    fn downcast_raw(&self, id: TypeId) -> Option<*const ()> {
        if id == TypeId::of::<Self>() {
            return Some(self as *const Self as *const ());
        }
        self.layer
            .downcast_raw(id)
            .or_else(|| self.inner.downcast_raw(id))
    }
    fn name(&self) -> &'static str {
        "Layered"
    }
}

fn register<S: Sub + 'static>(s: S) -> Box<dyn Sub> {
    Box::new(s)
}

fn main() {
    let stack = Layered {
        layer: Layered {
            layer: A,
            inner: B,
        },
        inner: C,
    };
    let d = register(stack);
    assert_eq!(d.name(), "Layered");
    // Only the innermost forwarder's forwarding reaches A's and
    // B's default bodies.
    assert!(d.downcast_raw(TypeId::of::<u8>()).is_none());
    println!("ok");
}
"#;

#[test]
fn run_mode_vtable_demanded_nested_forwarder() {
    let dir = temp_dir("nested");
    let src = fixture(&dir, "nested.rs", NESTED_SRC);
    assert_ok(&run_driver(&src));
}

/// The forwarder carries a trailing defaulted generic parameter
/// (`N = S`, the `tracing_subscriber::layer::Layered` shape):
/// the rendered concrete type is `Layered<A, B, B>` but the
/// stored instance key is `<Layered<A, B> as Sub>::...`. Two
/// instantiations are unsized so a base-name scan cannot pick
/// between them.
const DEFAULTED_SRC: &str = r#"
use std::any::TypeId;
use std::marker::PhantomData;

trait Sub: 'static {
    fn downcast_raw(&self, id: TypeId) -> Option<*const ()> {
        if id == TypeId::of::<Self>() {
            Some(self as *const Self as *const ())
        } else {
            None
        }
    }
    fn name(&self) -> &'static str;
}

struct A;
impl Sub for A {
    fn name(&self) -> &'static str {
        "A"
    }
}

struct B;
impl Sub for B {
    fn name(&self) -> &'static str {
        "B"
    }
}

struct Layered<L, S, N = S> {
    layer: L,
    inner: S,
    _nested: PhantomData<N>,
}

impl<L, S> Sub for Layered<L, S>
where
    L: Sub + 'static,
    S: Sub + 'static,
{
    fn downcast_raw(&self, id: TypeId) -> Option<*const ()> {
        if id == TypeId::of::<Self>() {
            return Some(self as *const Self as *const ());
        }
        self.layer
            .downcast_raw(id)
            .or_else(|| self.inner.downcast_raw(id))
    }
    fn name(&self) -> &'static str {
        "Layered"
    }
}

fn register<S: Sub + 'static>(s: S) -> Box<dyn Sub> {
    Box::new(s)
}

fn main() {
    let d1 = register(Layered {
        layer: A,
        inner: B,
        _nested: PhantomData,
    });
    let d2 = register(Layered {
        layer: B,
        inner: A,
        _nested: PhantomData,
    });
    assert_eq!(d1.name(), "Layered");
    assert_eq!(d2.name(), "Layered");
    assert!(d1.downcast_raw(TypeId::of::<Layered<A, B>>()).is_some());
    assert!(d2.downcast_raw(TypeId::of::<Layered<B, A>>()).is_some());
    println!("ok");
}
"#;

#[test]
fn run_mode_dispatch_defaulted_generic_args() {
    let dir = temp_dir("defaulted");
    let src = fixture(&dir, "defaulted.rs", DEFAULTED_SRC);
    assert_ok(&run_driver(&src));
}
