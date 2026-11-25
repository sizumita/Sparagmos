use super::{CTOR_VALUE_SRC, DYN_DEFAULT_SRC, fixture, run_driver, run_ok, summarize_driver, temp_dir};


const DYN_BOUND_PROJECTION_SRC: &str = r#"
trait Trait {
    type Assoc;
}
struct Holder<T: Trait + ?Sized> {
    field: T::Assoc,
}
fn take(h: Holder<dyn Trait<Assoc = u32>>) -> u32 {
    h.field
}
fn main() {
    let h = Holder::<dyn Trait<Assoc = u32>> { field: 42 };
    println!("{}", take(h));
}
"#;

#[test]
fn run_mode_dyn_bound_projection_holder_runs() {
    // A projection whose trait has no implementation at all
    // still normalizes when its self type is a trait object
    // binding the item (`dyn Trait<Assoc = u32>`): the
    // object type carries the equality itself. The
    // unnormalizable-projection screen must not reject the
    // holder's constructor, the function taking the holder,
    // or the main holding one as a local — natively the
    // program prints 42.
    let dir = temp_dir("dyn-bound-projection");
    let src = fixture(&dir, "dyn_bound_projection.rs", DYN_BOUND_PROJECTION_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the dyn-bound projection program runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "42\n",
        "the bound field value reaches the print"
    );
}

const DYN_BOUND_SECOND_ITEM_SRC: &str = r#"
trait Trait {
    type Assoc;
    type Other;
}
struct Holder<T: Trait + ?Sized> {
    field: T::Other,
}
fn main() {
    let h = Holder::<dyn Trait<Assoc = u8, Other = u32>> { field: 7 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_second_item_runs() {
    // The screen's exemption is per associated item: the
    // projected item here is `Other`, bound by the object's
    // second predicate, not the first.
    let dir = temp_dir("dyn-bound-second-item");
    let src = fixture(&dir, "dyn_bound_second_item.rs", DYN_BOUND_SECOND_ITEM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the second bound item's projection runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "7\n");
}

const DYN_BOUND_SUPERTRAIT_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Sub: Trait {}
struct Holder<T: Trait + ?Sized> {
    field: T::Assoc,
}
fn main() {
    let h = Holder::<dyn Sub<Assoc = u32>> { field: 9 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_supertrait_runs() {
    // The binding names the supertrait's item: the object's
    // predicates elaborate `Sub`'s binding into a projection
    // predicate for `Trait::Assoc`, which is the item the
    // holder's field projects.
    let dir = temp_dir("dyn-bound-supertrait");
    let src = fixture(&dir, "dyn_bound_supertrait.rs", DYN_BOUND_SUPERTRAIT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the supertrait-bound projection runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "9\n");
}

const DYN_BOUND_SUPERTRAIT_INHERITED_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Child: Trait<Assoc = u32> {}
struct Holder<T: Trait + ?Sized> {
    field: T::Assoc,
}
fn take(h: Holder<dyn Child>) -> u32 {
    h.field
}
fn main() {
    let h = Holder::<dyn Child> { field: 42 };
    println!("{}", take(h));
}
"#;

#[test]
fn run_mode_dyn_bound_projection_supertrait_inherited_runs() {
    // The binding is inherited rather than written on the
    // object: `Child`'s own supertrait bound fixes
    // `Trait::Assoc = u32`, and `dyn Child` spells no
    // binding itself. Forming the object type elaborates
    // the principal trait's supertrait bindings into the
    // object's existential predicates, so a projection
    // predicate binding `Trait::Assoc` is present in the
    // object's own predicate list and the screen's
    // exemption applies as it stands — measured on the
    // pinned 1.100.0 toolchain (the program runs and
    // prints 42); this test pins that elaboration.
    let dir = temp_dir("dyn-bound-supertrait-inherited");
    let src = fixture(
        &dir,
        "dyn_bound_supertrait_inherited.rs",
        DYN_BOUND_SUPERTRAIT_INHERITED_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the inherited-binding projection runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_ALIAS_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Carrier {
    type Item: Trait + ?Sized;
}
impl Carrier for () {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T: Carrier> {
    field: <T::Item as Trait>::Assoc,
}
fn take(h: Holder<()>) -> u32 {
    h.field
}
fn main() {
    let h = Holder::<()> { field: 42 };
    println!("{}", take(h));
}
"#;

#[test]
fn run_mode_dyn_bound_projection_through_alias_runs() {
    // The projected self type is itself a projection:
    // the holder's field is `<<T as Carrier>::Item as
    // Trait>::Assoc`, whose self type reaches the screen as
    // an alias, not as the trait object the implementation
    // assigns it (`dyn Trait<Assoc = u32>`). Resolving the
    // intermediate projection through the implementation's
    // own assigned type reaches the binding, so the screen
    // must not reject the holder's constructor, the
    // function taking the holder, or main — natively the
    // program prints 42.
    let dir = temp_dir("dyn-bound-projection-alias");
    let src = fixture(
        &dir,
        "dyn_bound_projection_alias.rs",
        DYN_BOUND_PROJECTION_ALIAS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the alias-self dyn-bound projection program runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "42\n",
        "the bound field value reaches the print"
    );
}

const DYN_BOUND_PROJECTION_GENERIC_IMPL_ALIAS_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Carrier {
    type Item: Trait + ?Sized;
}
struct Wrap<T>(T);
impl<T> Carrier for Wrap<T> {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T: Carrier> {
    field: <T::Item as Trait>::Assoc,
}
fn take(h: Holder<Wrap<u8>>) -> u32 {
    h.field
}
fn main() {
    let h = Holder::<Wrap<u8>> { field: 42 };
    println!("{}", take(h));
}
"#;

#[test]
fn run_mode_dyn_bound_projection_through_generic_impl_alias_runs() {
    // The intermediate projection's implementation is
    // generic: matching its self type `Wrap<T>` against
    // `Wrap<u8>` binds the implementation's parameter, and
    // the assigned type is read with that binding.
    let dir = temp_dir("dyn-bound-projection-generic-impl-alias");
    let src = fixture(
        &dir,
        "dyn_bound_projection_generic_impl_alias.rs",
        DYN_BOUND_PROJECTION_GENERIC_IMPL_ALIAS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the generic-impl alias projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_CHAINED_ALIAS_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Carrier {
    type Item: Trait + ?Sized;
}
trait Carrier2 {
    type Item2: Trait + ?Sized;
}
impl Carrier2 for () {
    type Item2 = dyn Trait<Assoc = u32>;
}
impl Carrier for () {
    type Item = <() as Carrier2>::Item2;
}
struct Holder<T: Carrier> {
    field: <T::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<()> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_through_chained_alias_runs() {
    // The type an implementation assigns the intermediate
    // item is itself a projection; the resolution follows
    // it to the trait object behind it.
    let dir = temp_dir("dyn-bound-projection-chained-alias");
    let src = fixture(
        &dir,
        "dyn_bound_projection_chained_alias.rs",
        DYN_BOUND_PROJECTION_CHAINED_ALIAS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the chained-alias projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_TRAIT_ARGS_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Carrier<A> {
    type Item: Trait + ?Sized;
}
impl Carrier<u8> for () {
    type Item = dyn Trait<Assoc = u32>;
}
impl Carrier<bool> for () {
    type Item = dyn Trait<Assoc = u64>;
}
struct Holder<T: Carrier<u8>> {
    field: <<T as Carrier<u8>>::Item as Trait>::Assoc,
}
fn take(h: Holder<()>) -> u32 {
    h.field
}
fn main() {
    let h = Holder::<()> { field: 42 };
    println!("{}", take(h));
}
"#;

#[test]
fn run_mode_dyn_bound_projection_trait_args_select_impl_runs() {
    // The intermediate projection's trait is implemented
    // for the same self type at two of its own argument
    // instantiations, assigning the item different trait
    // objects. Matching only the self type reads both
    // implementations, finds their assigned types in
    // disagreement, and resolves nothing, so the screen
    // rejects the holder and main. Matching the whole
    // trait reference selects the `Carrier<u8>`
    // implementation, whose binding the outer projection
    // normalizes through -- natively the program prints 42.
    let dir = temp_dir("dyn-bound-projection-trait-args");
    let src = fixture(
        &dir,
        "dyn_bound_projection_trait_args.rs",
        DYN_BOUND_PROJECTION_TRAIT_ARGS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the trait-args projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_TRAIT_ARGS_OTHER_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Carrier<A> {
    type Item: Trait + ?Sized;
}
impl Carrier<u8> for () {
    type Item = dyn Trait<Assoc = u32>;
}
impl Carrier<bool> for () {
    type Item = dyn Trait<Assoc = u64>;
}
struct Holder<T: Carrier<bool>> {
    field: <<T as Carrier<bool>>::Item as Trait>::Assoc,
}
fn take(h: Holder<()>) -> u64 {
    h.field
}
fn main() {
    let h = Holder::<()> { field: 42 };
    println!("{}", take(h));
}
"#;

#[test]
fn run_mode_dyn_bound_projection_trait_args_other_impl_runs() {
    // The same pair of implementations, projected at the
    // other trait argument: the `Carrier<bool>`
    // implementation is the applicable one, and its
    // `u64` binding is what the field holds.
    let dir = temp_dir("dyn-bound-projection-trait-args-other");
    let src = fixture(
        &dir,
        "dyn_bound_projection_trait_args_other.rs",
        DYN_BOUND_PROJECTION_TRAIT_ARGS_OTHER_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the other-trait-args projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_TRAIT_ARG_PARAM_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Carrier<A> {
    type Item: Trait + ?Sized;
}
impl<A> Carrier<A> for () {
    type Item = dyn Trait<Assoc = A>;
}
struct Holder<T: Carrier<u8>> {
    field: <<T as Carrier<u8>>::Item as Trait>::Assoc,
}
fn take(h: Holder<()>) -> u8 {
    h.field
}
fn main() {
    let h = Holder::<()> { field: 42 };
    println!("{}", take(h));
}
"#;

#[test]
fn run_mode_dyn_bound_projection_trait_arg_only_param_runs() {
    // The implementation's parameter appears only in the
    // trait reference's own arguments, never in the self
    // type: matching those arguments binds it from the
    // projection's (`A := u8`), and the assigned type is
    // read with that binding, reaching `dyn Trait<Assoc =
    // u8>` -- natively the program prints 42.
    let dir = temp_dir("dyn-bound-projection-trait-arg-param");
    let src = fixture(
        &dir,
        "dyn_bound_projection_trait_arg_param.rs",
        DYN_BOUND_PROJECTION_TRAIT_ARG_PARAM_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the trait-arg-param projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_NESTED_ADT_ARG_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Carrier {
    type Item: Trait + ?Sized;
}
struct Wrap<T> {
    marker: std::marker::PhantomData<T>,
}
impl Carrier for Wrap<u32> {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T: Carrier> {
    field: <T::Item as Trait>::Assoc,
}
struct Outer<T>(Holder<Wrap<T>>)
where
    Wrap<T>: Carrier;
fn main() {
    let o = Outer::<u32>(Holder::<Wrap<u32>> { field: 42 });
    println!("{}", o.0.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_nested_adt_arg_runs() {
    // The holder reaches the screen through another ADT's
    // field, as `Holder<Wrap<T>>` in `Outer<T>`'s frame.
    // Composing the holder's arguments with the outer
    // frame's used to substitute only arguments that are
    // directly parameters, leaving `Wrap<T>` with the outer
    // parameter still inside; the implementation matcher
    // then could not select `Carrier for Wrap<u32>` and the
    // screen rejected every function using `Outer<u32>`,
    // including main. Substituting parameters throughout
    // the nested argument gives the holder `Wrap<u32>`,
    // whose implementation's binding the field projection
    // normalizes through -- natively the program prints 42.
    let dir = temp_dir("dyn-bound-projection-nested-adt-arg");
    let src = fixture(
        &dir,
        "dyn_bound_projection_nested_adt_arg.rs",
        DYN_BOUND_PROJECTION_NESTED_ADT_ARG_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the nested-ADT-arg projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_GENERIC_ASSOC_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Carrier {
    type Item<A>: Trait + ?Sized;
}
impl Carrier for () {
    type Item<A> = dyn Trait<Assoc = A>;
}
struct Holder<T: Carrier> {
    field: <T::Item<u32> as Trait>::Assoc,
}
fn take(h: Holder<()>) -> u32 {
    h.field
}
fn main() {
    let h = Holder::<()> { field: 42 };
    println!("{}", take(h));
}
"#;

#[test]
fn run_mode_dyn_bound_projection_generic_assoc_type_runs() {
    // The intermediate projection is a generic associated
    // type: `<T::Item<u32> as Trait>::Assoc`. The
    // projection carries the item's own argument past its
    // trait reference's, and the implementation matcher
    // used to skip every implementation on that argument
    // count alone, resolving nothing, so the screen
    // rejected the holder and main. Reading the assigned
    // type with the item's arguments applied reaches
    // `dyn Trait<Assoc = u32>`, whose binding the outer
    // projection normalizes through -- natively the program
    // prints 42.
    let dir = temp_dir("dyn-bound-projection-generic-assoc");
    let src = fixture(
        &dir,
        "dyn_bound_projection_generic_assoc.rs",
        DYN_BOUND_PROJECTION_GENERIC_ASSOC_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the generic-associated-type projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_FN_PTR_IMPL_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Carrier {
    type Item: Trait + ?Sized;
}
impl<A> Carrier for fn(A) {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T: Carrier> {
    field: <T::Item as Trait>::Assoc,
}
fn take(h: Holder<fn(u8)>) -> u32 {
    h.field
}
fn main() {
    let h = Holder::<fn(u8)> { field: 42 };
    println!("{}", take(h));
}
"#;

#[test]
fn run_mode_dyn_bound_projection_fn_ptr_impl_runs() {
    // The intermediate projection's item is assigned by a
    // generic implementation for a function pointer:
    // `impl<A> Carrier for fn(A)`. Matching the
    // implementation's self type against the concrete
    // `fn(u8)` used to fail the initial equality check and
    // reach the unsupported-shape fallback, resolving
    // nothing, so the screen rejected the holder and main.
    // Matching function-pointer signatures structurally
    // binds `A := u8` and reaches `dyn Trait<Assoc = u32>`,
    // whose binding the outer projection normalizes
    // through -- natively the program prints 42.
    let dir = temp_dir("dyn-bound-projection-fn-ptr-impl");
    let src = fixture(
        &dir,
        "dyn_bound_projection_fn_ptr_impl.rs",
        DYN_BOUND_PROJECTION_FN_PTR_IMPL_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the fn-ptr-impl projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_DYN_SELF_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Other<A> {}
trait Carrier {
    type Item: Trait + ?Sized;
}
impl<A> Carrier for dyn Other<A> {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T: Carrier + ?Sized> {
    field: <T::Item as Trait>::Assoc,
}
fn take(h: Holder<dyn Other<u8>>) -> u32 {
    h.field
}
fn main() {
    let h = Holder::<dyn Other<u8>> { field: 42 };
    println!("{}", take(h));
}
"#;

#[test]
fn run_mode_dyn_bound_projection_dyn_self_impl_runs() {
    // The intermediate projection's item is assigned by a
    // generic implementation for a trait object:
    // `impl<A> Carrier for dyn Other<A>`. Matching the
    // implementation's self type against the concrete
    // `dyn Other<u8>` reached the unsupported-shape
    // fallback -- the self-type match had no trait-object
    // arm -- resolving nothing, so the screen rejected
    // the holder and main. Matching the object's
    // existential predicates structurally binds
    // `A := u8` and reaches `dyn Trait<Assoc = u32>`,
    // whose binding the outer projection normalizes
    // through -- natively the program prints 42.
    let dir = temp_dir("dyn-bound-projection-dyn-self-impl");
    let src = fixture(
        &dir,
        "dyn_bound_projection_dyn_self_impl.rs",
        DYN_BOUND_PROJECTION_DYN_SELF_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the dyn-self-impl projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_AUTO_TRAIT_SET_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Other {}
trait Carrier {
    type Item: Trait + ?Sized;
}
impl Carrier for dyn Other {
    type Item = dyn Trait<Assoc = u64>;
}
impl Carrier for dyn Other + Send {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T: Carrier + ?Sized> {
    field: <T::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<dyn Other + Send> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_auto_trait_set_runs() {
    // Separate implementations exist for `dyn Other`
    // (assigning `Assoc = u64`) and for `dyn Other + Send`
    // (assigning `Assoc = u32`). The predicate-subset
    // match let the `dyn Other` implementation also match
    // the `dyn Other + Send` target, so the resolver saw
    // two disagreeing candidates and resolved nothing,
    // screening out main. Auto traits are part of the
    // object type's identity -- natively the two impls
    // cover different self types and neither covers the
    // other (each alone leaves the other target an E0277)
    // -- so the match requires equal auto-trait sets and
    // the `+ Send` implementation alone applies:
    // natively the program prints 42.
    let dir = temp_dir("dyn-bound-projection-auto-trait-set");
    let src = fixture(
        &dir,
        "dyn_bound_projection_auto_trait_set.rs",
        DYN_BOUND_PROJECTION_AUTO_TRAIT_SET_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the auto-trait-set projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_ALIAS_ADT_ARG_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Carrier {
    type Item: Trait + ?Sized;
}
impl Carrier for () {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<U: Trait + ?Sized> {
    field: U::Assoc,
}
struct Outer<T: Carrier>(Holder<T::Item>);
fn main() {
    let o = Outer::<()>(Holder { field: 42 });
    println!("{}", o.0.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_alias_adt_arg_runs() {
    // An ADT argument that is itself a projection:
    // `Outer<T: Carrier>(Holder<T::Item>)`. The alias's
    // parameters belong to the enclosing frame; composing
    // the holder's arguments used to carry `T::Item`
    // across the ADT boundary unresolved, where resolving
    // its self parameter against the holder's frame read
    // the very argument carrying it, until the depth cap
    // gave up and the screen rejected main. Resolving the
    // projection through the screen's own evidence before
    // the crossing substitutes `dyn Trait<Assoc = u32>`,
    // whose binding the field projection normalizes
    // through -- natively the program prints 42.
    let dir = temp_dir("dyn-bound-projection-alias-adt-arg");
    let src = fixture(
        &dir,
        "dyn_bound_projection_alias_adt_arg.rs",
        DYN_BOUND_PROJECTION_ALIAS_ADT_ARG_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the alias-ADT-arg projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_CONST_ARG_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Carrier<const N: usize> {
    type Item: Trait + ?Sized;
}
impl Carrier<3> for () {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T: Carrier<N>, const N: usize> {
    field: <<T as Carrier<N>>::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<(), 3> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_const_arg_runs() {
    // The intermediate projection's trait reference
    // passes the enclosing const parameter:
    // `Carrier<N>` with `N` Holder's own parameter.
    // Matching the implementation `Carrier<3> for ()`
    // compared the concrete `3` against the unresolved
    // `N` directly, found no implementation, and the
    // screen rejected the holder and main. Resolving the
    // target constant through the ambient frame first
    // reaches `dyn Trait<Assoc = u32>`, whose binding the
    // outer projection normalizes through -- natively the
    // program prints 42.
    let dir = temp_dir("dyn-bound-projection-const-arg");
    let src = fixture(
        &dir,
        "dyn_bound_projection_const_arg.rs",
        DYN_BOUND_PROJECTION_CONST_ARG_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the const-arg projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_NESTED_CONST_ARG_SRC: &str = r#"
trait Trait {
    type Assoc;
}
struct Wrap<const N: usize>;
trait Carrier<A> {
    type Item: Trait + ?Sized;
}
impl Carrier<Wrap<3>> for () {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T: Carrier<Wrap<N>>, const N: usize> {
    field: <<T as Carrier<Wrap<N>>>::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<(), 3> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_nested_const_arg_runs() {
    // The const parameter rides in a nested position:
    // the trait argument is the ADT `Wrap<N>`, whose own
    // const argument is Holder's parameter. The match
    // descends into the ADT's arguments, where the same
    // ambient resolution applies -- natively the program
    // prints 42.
    let dir = temp_dir("dyn-bound-projection-nested-const-arg");
    let src = fixture(
        &dir,
        "dyn_bound_projection_nested_const_arg.rs",
        DYN_BOUND_PROJECTION_NESTED_CONST_ARG_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the nested const-arg projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_UNEVALUATED_CONST_TARGET_SRC: &str = r#"
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
trait Trait {
    type Assoc;
}
trait Carrier<const N: usize> {
    type Item: Trait + ?Sized;
}
impl Carrier<3> for () {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T: Carrier<{ N + 0 }>, const N: usize> {
    field: <<T as Carrier<{ N + 0 }>>::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<(), 3> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_unevaluated_const_target_runs() {
    // The intermediate projection's trait reference
    // carries an unevaluated constant: `Carrier<{ N + 0 }>`
    // with `N` Holder's own parameter, instantiated at 3.
    // Matching the implementation `Carrier<3> for ()`
    // compared the concrete `3` against the unevaluated
    // expression and rejected the implementation, so the
    // screen rejected the holder and main. An unevaluated
    // constant cannot be compared without evaluating it,
    // which the screen never does, so the comparison is
    // inconclusive: the implementation stays a candidate
    // and its assigned type resolves the projection --
    // natively the program prints 42.
    let dir = temp_dir("dyn-bound-projection-unevaluated-const-target");
    let src = fixture(
        &dir,
        "dyn_bound_projection_unevaluated_const_target.rs",
        DYN_BOUND_PROJECTION_UNEVALUATED_CONST_TARGET_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the unevaluated-const-target projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_UNEVALUATED_CONST_IMPL_SRC: &str = r#"
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
trait Trait {
    type Assoc;
}
trait Carrier<const N: usize> {
    type Item: Trait + ?Sized;
}
impl Carrier<{ 1 + 2 }> for () {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T: Carrier<3>> {
    field: <<T as Carrier<3>>::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<()> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_unevaluated_const_impl_runs() {
    // The mirror image: the implementation's constant is
    // the unevaluated one (`Carrier<{ 1 + 2 }> for ()`)
    // and the projection's is the concrete `3`. The
    // comparison is just as inconclusive in this
    // direction, so the implementation stays a candidate
    // -- natively the program prints 42.
    let dir = temp_dir("dyn-bound-projection-unevaluated-const-impl");
    let src = fixture(
        &dir,
        "dyn_bound_projection_unevaluated_const_impl.rs",
        DYN_BOUND_PROJECTION_UNEVALUATED_CONST_IMPL_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the unevaluated-const-impl projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_LIFETIME_ARG_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Carrier<'a> {
    type Item: Trait + ?Sized;
}
impl Carrier<'static> for () {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<'a, T: Carrier<'a>> {
    field: <<T as Carrier<'a>>::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<'static, ()> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_lifetime_arg_runs() {
    // The intermediate projection's trait reference
    // passes the enclosing lifetime parameter:
    // `Carrier<'a>` with `'a` Holder's own parameter.
    // Matching the implementation `Carrier<'static> for
    // ()` compared `'static` against the unresolved `'a`
    // directly, found no implementation, and the screen
    // rejected the holder and main. Resolving the target
    // lifetime through the ambient frame first reaches
    // `dyn Trait<Assoc = u32>`, whose binding the outer
    // projection normalizes through -- natively the
    // program prints 42.
    let dir = temp_dir("dyn-bound-projection-lifetime-arg");
    let src = fixture(
        &dir,
        "dyn_bound_projection_lifetime_arg.rs",
        DYN_BOUND_PROJECTION_LIFETIME_ARG_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the lifetime-arg projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_TYPE_ARG_SRC: &str = r#"
trait Trait {
    type Assoc;
}
struct Wrap<A>(A);
trait Carrier<A> {
    type Item: Trait + ?Sized;
}
impl<A> Carrier<A> for Wrap<A> {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<S: Carrier<A>, A> {
    field: <<S as Carrier<A>>::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<Wrap<u32>, u32> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_type_arg_runs() {
    // The implementation parameter appears in both the
    // self type and the trait argument: `Carrier<A> for
    // Wrap<A>`. Matching the implementation against the
    // projection `<S as Carrier<A>>::Item` in the
    // holder's frame binds `A` to `u32` through the self
    // type, then compared that binding against the
    // holder's unresolved parameter `A` in the trait
    // argument and rejected the implementation, so the
    // screen rejected the holder and main. Resolving the
    // target type argument through the ambient frame
    // first reaches `dyn Trait<Assoc = u32>`, whose
    // binding the outer projection normalizes through --
    // natively the program prints 42.
    let dir = temp_dir("dyn-bound-projection-type-arg");
    let src = fixture(
        &dir,
        "dyn_bound_projection_type_arg.rs",
        DYN_BOUND_PROJECTION_TYPE_ARG_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the type-arg projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_NESTED_TYPE_ARG_SRC: &str = r#"
trait Trait {
    type Assoc;
}
struct Wrap<A>(A);
trait Carrier<A> {
    type Item: Trait + ?Sized;
}
impl<A> Carrier<Wrap<A>> for Wrap<A> {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<S: Carrier<Wrap<A>>, A> {
    field: <<S as Carrier<Wrap<A>>>::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<Wrap<u32>, u32> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_nested_type_arg_runs() {
    // The type parameter rides in a nested position:
    // the trait argument is the ADT `Wrap<A>`, whose own
    // type argument is the holder's parameter. The match
    // descends into the ADT's arguments, where the same
    // ambient resolution applies -- natively the program
    // prints 42.
    let dir = temp_dir("dyn-bound-projection-nested-type-arg");
    let src = fixture(
        &dir,
        "dyn_bound_projection_nested_type_arg.rs",
        DYN_BOUND_PROJECTION_NESTED_TYPE_ARG_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the nested type-arg projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_ASSOC_DEFAULT_SRC: &str = r#"
#![feature(associated_type_defaults)]
trait Trait {
    type Assoc;
}
trait Carrier {
    type Item: Trait + ?Sized = dyn Trait<Assoc = u32>;
}
impl Carrier for () {}
struct Holder<T: Carrier> {
    field: <T::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<()> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_assoc_default_runs() {
    // The matching implementation declares no item of
    // its own: it inherits the trait's associated-type
    // default. Reading only the implementation's
    // associated items found nothing, resolved nothing,
    // and the screen rejected the holder and main.
    // Falling back to the trait's default, instantiated
    // with the matched trait reference, reaches
    // `dyn Trait<Assoc = u32>`, whose binding the outer
    // projection normalizes through -- natively the
    // program prints 42.
    let dir = temp_dir("dyn-bound-projection-assoc-default");
    let src = fixture(
        &dir,
        "dyn_bound_projection_assoc_default.rs",
        DYN_BOUND_PROJECTION_ASSOC_DEFAULT_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the assoc-default projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_ASSOC_DEFAULT_OVERRIDDEN_SRC: &str = r#"
#![feature(associated_type_defaults)]
trait Trait {
    type Assoc;
}
trait Carrier {
    type Item: Trait + ?Sized = dyn Trait<Assoc = u32>;
}
impl Carrier for () {
    type Item = dyn Trait<Assoc = u64>;
}
struct Holder<T: Carrier> {
    field: <T::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<()> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_assoc_default_overridden_runs() {
    // The implementation declares the item itself, so
    // its own assignment (`Assoc = u64`) stands and the
    // trait default is never consulted -- natively the
    // program prints 42.
    let dir = temp_dir("dyn-bound-projection-assoc-default-overridden");
    let src = fixture(
        &dir,
        "dyn_bound_projection_assoc_default_overridden.rs",
        DYN_BOUND_PROJECTION_ASSOC_DEFAULT_OVERRIDDEN_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the assoc-default-overridden projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_EQUAL_SELF_PARAM_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Carrier {
    type Item: Trait + ?Sized;
}
struct Wrap<A>(A);
impl<A> Carrier for Wrap<A> {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<A> {
    field: <<Wrap<A> as Carrier>::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<u32> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_equal_self_param_runs() {
    // The implementation's self type and the field's are
    // the same `Wrap<A>` -- the implementation's parameter
    // and the holder's, identically indexed, so interned
    // identity calls them equal. Accepting the match on
    // identity alone binds nothing, the instantiation the
    // assigned-type read performs has no argument for the
    // implementation's parameter, and the screen rejects
    // the holder and `main`. Matching a
    // parameter-mentioning self type structurally binds
    // the parameter through the ambient frame -- natively
    // the program prints 42.
    let dir = temp_dir("dyn-bound-projection-equal-self-param");
    let src = fixture(
        &dir,
        "dyn_bound_projection_equal_self_param.rs",
        DYN_BOUND_PROJECTION_EQUAL_SELF_PARAM_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the equal-self-param projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_CONDITIONAL_IMPL_SRC: &str = r#"
trait Bound {}
impl Bound for u8 {}
trait Trait {
    type Assoc;
}
trait Carrier {
    type Item: Trait + ?Sized;
}
impl<T: Bound> Carrier for T {
    type Item = dyn Trait<Assoc = u64>;
}
impl Carrier for () {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T: Carrier> {
    field: <T::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<()> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_conditional_impl_runs() {
    // Two implementations structurally match the self
    // type `()` and assign the item different trait
    // objects: a blanket one conditional on `T: Bound` --
    // which `()` does not satisfy -- and an unconditional
    // one for `()` itself. Structural matching checks no
    // predicates, so both match and their disagreement
    // resolves nothing, and the screen rejects the holder
    // and `main`. The conditional implementation may not
    // apply at all, and in a program that compiles it
    // cannot apply alongside the unconditional one
    // (coherence), so the unconditional assignment is the
    // applicable one -- natively the program prints 42.
    let dir = temp_dir("dyn-bound-projection-conditional-impl");
    let src = fixture(
        &dir,
        "dyn_bound_projection_conditional_impl.rs",
        DYN_BOUND_PROJECTION_CONDITIONAL_IMPL_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the conditional-impl projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_NESTED_TUPLE_TARGET_SRC: &str = r#"
trait Trait {
    type Assoc;
}
struct Wrap<A>(A);
trait Carrier<A> {
    type Item: Trait + ?Sized;
}
impl<A> Carrier<A> for Wrap<A> {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T>
where
    Wrap<(T,)>: Carrier<(u32,)>,
{
    field: <<Wrap<(T,)> as Carrier<(u32,)>>::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<u32> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_nested_tuple_target_runs() {
    // The implementation parameter matches a compound
    // target type: walking `Holder<u32>`, the self type
    // `Wrap<(T,)>` binds the implementation's `A` through
    // `Self`. The binding used to record the tuple
    // `(T,)` with the holder's parameter still inside
    // it, so the trait argument's `(u32,)` disagreed
    // with the recorded binding and the implementation
    // was rejected, resolving nothing, so the screen
    // rejected the holder and `main`. Substituting the
    // target through the ambient frame before recording
    // the binding -- recursively, the way a bare
    // parameter target is already resolved -- binds
    // `A := (u32,)`, and the outer projection normalizes
    // through `dyn Trait<Assoc = u32>` -- natively the
    // program prints 42.
    let dir = temp_dir("dyn-bound-projection-nested-tuple-target");
    let src = fixture(
        &dir,
        "dyn_bound_projection_nested_tuple_target.rs",
        DYN_BOUND_PROJECTION_NESTED_TUPLE_TARGET_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the nested-tuple-target projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_DEEP_CHAIN_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait A {
    type Item: B + ?Sized;
}
trait B {
    type Item: C + ?Sized;
}
trait C {
    type Item: Trait + ?Sized;
}
impl A for () {
    type Item = ();
}
impl B for () {
    type Item = ();
}
impl C for () {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T: A> {
    field: <<<<T as A>::Item as B>::Item as C>::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<()> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_deep_chain_runs() {
    // A valid projection chain with three intermediate
    // associated types: resolving the outer projection's
    // self type walks `A::Item`, `B::Item`, then
    // `C::Item`, and the recursion spends the resolution
    // depth budget before the chain reaches the trait
    // object -- the budget is spent on the way down, so
    // even resolving the first hop's parameter exhausts
    // it. Exhaustion used to make the resolution return
    // no answer, which the screen read as proof the
    // projection cannot normalize, rejecting the holder
    // and `main`. The budget is a termination guard, not
    // evidence: exhaustion during a judgment now leaves
    // the screen without an opinion -- natively the
    // program prints 42.
    let dir = temp_dir("dyn-bound-projection-deep-chain");
    let src = fixture(
        &dir,
        "dyn_bound_projection_deep_chain.rs",
        DYN_BOUND_PROJECTION_DEEP_CHAIN_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the deep-chain projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_TWO_CONDITIONAL_IMPLS_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Bound {}
trait Other {}
trait Carrier {
    type Item: Trait + ?Sized;
}
impl<T: Bound> Carrier for T {
    type Item = dyn Trait<Assoc = u64>;
}
struct Wrap<U>(U);
impl<U: Other> Carrier for Wrap<U> {
    type Item = dyn Trait<Assoc = u32>;
}
impl Bound for u8 {}
impl Other for u32 {}
struct Holder<T: Carrier> {
    field: <T::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<Wrap<u32>> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_two_conditional_impls_runs() {
    // Both implementations that structurally match the
    // self type `Wrap<u32>` are conditional: a blanket
    // one on `T: Bound`, which `Wrap<u32>` does not
    // satisfy, and one for `Wrap<U>` on `U: Other`, which
    // `u32` does. Structural matching checks no
    // predicates, so both match, their assigned types
    // disagree, and no unconditional implementation
    // settles the disagreement -- the verdict the
    // single-conditional case relies on has nothing to
    // settle with here. The disagreement rests on
    // predicates the match never checked, so it proves
    // nothing about whether the projection normalizes:
    // the resolution is recorded inconclusive and the
    // screen abstains -- natively the program prints 42.
    let dir = temp_dir("dyn-bound-projection-two-conditional-impls");
    let src = fixture(
        &dir,
        "dyn_bound_projection_two_conditional_impls.rs",
        DYN_BOUND_PROJECTION_TWO_CONDITIONAL_IMPLS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the two-conditional-impls projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_FN_PTR_BINDER_SRC: &str = r#"
#![allow(coherence_leak_check)]
trait Trait {
    type Assoc;
}
trait Carrier {
    type Item: Trait + ?Sized;
}
impl Carrier for fn(&'static u8) {
    type Item = dyn Trait<Assoc = u64>;
}
impl Carrier for for<'a> fn(&'a u8) {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T: Carrier> {
    field: <T::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<for<'a> fn(&'a u8)> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_fn_ptr_binder_runs() {
    // Implementations exist for two function-pointer
    // self types distinguished only by their bound
    // lifetimes: `fn(&'static u8)` and
    // `for<'a> fn(&'a u8)` are different self types with
    // different assignments. The signature match does
    // not turn on bound lifetimes, so the `'static`
    // implementation also matches the higher-ranked
    // target, the two candidates disagree, and the
    // screen used to reject the holder and `main`. A
    // disagreement the match cannot adjudicate is an
    // inconclusive resolution, not proof the projection
    // cannot normalize -- natively the program prints
    // 42.
    let dir = temp_dir("dyn-bound-projection-fn-ptr-binder");
    let src = fixture(
        &dir,
        "dyn_bound_projection_fn_ptr_binder.rs",
        DYN_BOUND_PROJECTION_FN_PTR_BINDER_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the fn-ptr-binder projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_PROJECTION_TARGET_ARG_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Source {
    type Item: Trait + ?Sized;
}
impl Source for () {
    type Item = dyn Trait<Assoc = u32>;
}
trait Carrier<A: ?Sized> {
    type Item: Trait + ?Sized;
}
struct Wrap<T: ?Sized>(std::marker::PhantomData<T>);
impl<A: ?Sized> Carrier<A> for Wrap<A> {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T: Source>
where
    Wrap<T::Item>: Carrier<dyn Trait<Assoc = u32>>,
{
    field: <<Wrap<T::Item> as Carrier<dyn Trait<Assoc = u32>>>::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<()> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_projection_target_arg_runs() {
    // The implementation parameter `A` appears in both
    // the self type (`Carrier<A> for Wrap<A>`) and the
    // trait argument, and the target at the self type is
    // a projection: walking `Holder<()>` matches the self
    // type `Wrap<T::Item>`, whose argument the ambient
    // frame substitutes to `<() as Source>::Item` but
    // leaves standing as a projection. Binding `A` to
    // that unresolved projection and then comparing it
    // against the trait argument's `dyn Trait<Assoc =
    // u32>` -- the very type the projection denotes --
    // rejected the implementation, so the screen
    // rejected the holder and `main`. The target is
    // resolved before it is bound, the way a rigid self
    // type's nested targets are -- natively the program
    // prints 42.
    let dir = temp_dir("dyn-bound-projection-projection-target-arg");
    let src = fixture(
        &dir,
        "dyn_bound_projection_projection_target_arg.rs",
        DYN_BOUND_PROJECTION_PROJECTION_TARGET_ARG_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the projection-target-arg projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const DYN_BOUND_PROJECTION_TUPLE_TARGET_ARG_SRC: &str = r#"
trait Trait {
    type Assoc;
}
trait Source {
    type Item;
}
impl Source for () {
    type Item = u32;
}
trait Carrier<A> {
    type Item: Trait + ?Sized;
}
struct Wrap<T>(std::marker::PhantomData<T>);
impl<A> Carrier<A> for Wrap<A> {
    type Item = dyn Trait<Assoc = u32>;
}
struct Holder<T: Source>
where
    Wrap<(T::Item,)>: Carrier<(u32,)>,
{
    field: <<Wrap<(T::Item,)> as Carrier<(u32,)>>::Item as Trait>::Assoc,
}
fn main() {
    let h = Holder::<()> { field: 42 };
    println!("{}", h.field);
}
"#;

#[test]
fn run_mode_dyn_bound_projection_tuple_target_arg_runs() {
    // The implementation parameter `A` appears in both
    // the self type (`Carrier<A> for Wrap<A>`) and the
    // trait argument, and the target at the self type is
    // a tuple holding a projection: matching binds `A`
    // to `(<() as Source>::Item,)` through the self type
    // and compares it against the trait argument's
    // `(u32,)` -- the very tuple the binding denotes,
    // since the projection resolves to `u32`. The
    // disagreement check inspected only the binding's
    // outermost shape, saw a tuple rather than a
    // projection, and treated the disagreement as
    // conclusive, so the screen rejected the holder and
    // `main`. A disagreement over a binding that
    // contains an unresolved projection anywhere inside
    // it is inconclusive rather than proof the outer
    // projection cannot normalize -- natively the
    // program prints 42.
    let dir = temp_dir("dyn-bound-projection-tuple-target-arg");
    let src = fixture(
        &dir,
        "dyn_bound_projection_tuple_target_arg.rs",
        DYN_BOUND_PROJECTION_TUPLE_TARGET_ARG_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the tuple-target-arg projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const POINTEE_METADATA_PROJECTION_SRC: &str = r#"
#![feature(ptr_metadata)]
struct Holder<T: ?Sized> {
    value: <T as std::ptr::Pointee>::Metadata,
}
fn direct() -> <[u8] as std::ptr::Pointee>::Metadata {
    4
}
fn main() {
    let h = Holder::<[u8]> { value: 3 };
    let s = Holder::<str> { value: 5 };
    println!("{} {} {}", h.value, s.value, direct());
}
"#;

#[test]
fn run_mode_compiler_implemented_trait_pointee_runs() {
    // `Pointee` is implemented by the compiler itself (core
    // marks it `#[rustc_deny_explicit_impl]`): no `impl` item
    // ever records it, so the implementation table is empty
    // for it, yet `<T as Pointee>::Metadata` normalizes for
    // every self type. The unnormalizable-projection screen
    // must not reject the holder or `main` over it -- natively
    // the program prints `3 5 4`.
    let dir = temp_dir("pointee-metadata-projection");
    let src = fixture(
        &dir,
        "pointee_metadata_projection.rs",
        POINTEE_METADATA_PROJECTION_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the Pointee-metadata projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "3 5 4\n");
}

const DISCRIMINANT_KIND_PROJECTION_SRC: &str = r#"
#![feature(discriminant_kind)]
use std::marker::DiscriminantKind;
struct Holder<T: DiscriminantKind> {
    value: T::Discriminant,
}
#[allow(dead_code)]
enum E {
    A,
    B,
}
fn main() {
    let h = Holder::<E> { value: 1 };
    println!("{}", h.value);
}
"#;

#[test]
fn run_mode_compiler_implemented_trait_discriminant_kind_runs() {
    // The exemption is the compiler-implemented class, not
    // `Pointee` by name: `DiscriminantKind` is another trait
    // core marks `#[rustc_deny_explicit_impl]`, implemented
    // by the compiler for every enum, with no `impl` item
    // recording it. Its projection normalizes (to `isize`
    // here), and the screen must not reject the program --
    // natively it prints `1`.
    let dir = temp_dir("discriminant-kind-projection");
    let src = fixture(
        &dir,
        "discriminant_kind_projection.rs",
        DISCRIMINANT_KIND_PROJECTION_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the DiscriminantKind projection program runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "1\n");
}

// ----- generic function definitions (Lane BA) --------------------------------

const GENERIC_CALLS_SRC: &str = r#"
fn id<T>(x: T) -> T { x }
fn first<T>(v: &[T]) -> &T { &v[0] }
struct Wrap<T>(T);
impl<T> Wrap<T> {
    fn get(&self) -> &T { &self.0 }
}
fn main() {
    let a = id::<u32>(3);
    let b = id::<i64>(-4);
    let w = Wrap::<u32>(9);
    let arr = [10u32, 20, 30];
    let c = *first(&arr);
    println!("{}", a as i64 + b + *w.get() as i64 + c as i64);
}
"#;

#[test]
fn run_mode_generic_function_instances_execute() {
    // Generic definitions have no instance as items; the instances
    // their call sites demand are resolved and converted, so a
    // program calling generic functions runs end to end.
    let dir = temp_dir("generic-calls");
    let src = fixture(&dir, "generic_calls.rs", GENERIC_CALLS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "generic calls must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "18\n");
}

// ----- trait-declared method keys (Lane BV, Issue #64) -----

const TRAIT_CALL_SRC: &str = r#"
trait Greeter { fn greet(&self) -> u32; }
struct S;
impl Greeter for S { fn greet(&self) -> u32 { 42 } }
fn main() {
    let s = S;
    println!("{} {}", s.greet(), <S as Greeter>::greet(&s));
}
"#;

#[test]
fn run_mode_trait_method_call_executes() {
    // A call to a trait-declared method names the trait
    // declaration in its FnDef target, while the item loop
    // converts the impl method under the impl's own key; the
    // driver registers the impl body under the trait key too, so
    // the call executes (Issue #64).
    let dir = temp_dir("trait-call");
    let src = fixture(&dir, "trait_call.rs", TRAIT_CALL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "trait method calls must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42 42\n");
}

const TRAIT_TWO_TYPES_SRC: &str = r#"
trait Get { fn get(&self) -> u32; }
struct Wrap<T>(T);
impl Get for Wrap<u32> { fn get(&self) -> u32 { self.0 + 1 } }
impl Get for Wrap<u16> { fn get(&self) -> u32 { self.0 as u32 + 100 } }
trait Tag { fn tag(&self) -> u32; }
struct Pair<T>(T, T);
impl<T> Tag for Pair<T> { fn tag(&self) -> u32 { 7 } }
fn main() {
    let a = Wrap(3u32).get();
    let b = Wrap(4u16).get();
    let c = Pair(1u32, 2u32).tag();
    let d = Pair(true, false).tag();
    println!("{} {} {} {}", a, b, c, d);
}
"#;

#[test]
fn run_mode_trait_method_two_type_arguments() {
    // The same trait method demanded at two different type
    // arguments resolves to the right impl body for each: two
    // concrete impls of one generic struct, and one generic impl
    // instantiated at two types.
    let dir = temp_dir("trait-two-types");
    let src = fixture(&dir, "trait_two_types.rs", TRAIT_TWO_TYPES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "trait calls at two type arguments must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "4 104 7 7\n");
}

const TRAIT_DEFAULT_SRC: &str = r#"
trait D {
    fn val(&self) -> u32 { 10 }
    fn extra(&self) -> u32 { self.val() + 1 }
}
struct X;
impl D for X {}
struct Y;
impl D for Y { fn val(&self) -> u32 { 20 } }
fn main() {
    println!("{} {} {} {}", X.val(), X.extra(), Y.val(), Y.extra());
}
"#;

#[test]
fn run_mode_dyn_trait_default_method() {
    // A trait default method called through a `dyn` receiver: the
    // dispatch target is the trait's own body instantiated with
    // the concrete `Self`, demanded from the vtable the unsizing
    // cast forms (Lane EU). Both `&dyn` and `Box<dyn>` receivers,
    // a default calling another default, and a default calling a
    // required method the impl provides.
    let dir = temp_dir("dyn-default");
    let src = fixture(&dir, "dyn_default.rs", DYN_DEFAULT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dyn default methods must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-default ok\n");
}

const DYN_DEFAULT_OVERRIDE_SRC: &str = r#"
trait D {
    fn val(&self) -> u32 { 10 }
    fn extra(&self) -> u32 { self.val() + 1 }
}
struct X;
impl D for X {}
struct Y;
impl D for Y {
    fn val(&self) -> u32 { 20 }
}
fn main() {
    let x: &dyn D = &X;
    let y: &dyn D = &Y;
    println!("{} {} {} {}", x.val(), x.extra(), y.val(), y.extra());
    let bx: Box<dyn D> = Box::new(X);
    let by: Box<dyn D> = Box::new(Y);
    println!("{} {} {} {}", bx.val(), bx.extra(), by.val(), by.extra());
}
"#;

#[test]
fn run_mode_dyn_trait_default_method_override_wins() {
    // An impl that overrides the method is what a dyn call must
    // reach — never the default — and the default `extra` body
    // must dispatch its inner `self.val()` to the override for
    // `Y` and to the default for `X`. This also pins the dispatch
    // precedence: the statically demanded default instance of
    // `val` (from `extra`'s inner call) sits in the program under
    // the plain trait name, and a dyn call must not be served by
    // it directly.
    let dir = temp_dir("dyn-default-override");
    let src = fixture(&dir, "dyn_default_override.rs", DYN_DEFAULT_OVERRIDE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dyn default/override dispatch must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "10 11 20 21\n10 11 20 21\n"
    );
}

const DYN_DEFAULT_SUPER_SRC: &str = r#"
trait Super {
    fn s(&self) -> i32 { 1 }
    fn s2(&self) -> i32;
}
trait Sub: Super {
    fn b(&self) -> i32 { 2 }
}
struct S;
impl Super for S {
    fn s2(&self) -> i32 { 3 }
}
impl Sub for S {}
fn main() {
    let d: &dyn Sub = &S;
    println!("{} {} {}", d.b(), d.s(), d.s2());
    let u: &dyn Super = d;
    println!("{} {}", u.s(), u.s2());
}
"#;

#[test]
fn run_mode_dyn_trait_default_method_supertrait_upcast() {
    // Supertrait default methods are vtable entries of the
    // supertrait vtable (`TraitVPtr`): calling them through the
    // subtrait object and through an upcast `&dyn Super` both
    // reach the default body.
    let dir = temp_dir("dyn-default-super");
    let src = fixture(&dir, "dyn_default_super.rs", DYN_DEFAULT_SUPER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "supertrait dyn defaults must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "2 1 3\n1 3\n");
}

const DYN_DEFAULT_GENERIC_TRAIT_SRC: &str = r#"
trait G<T> {
    fn get(&self, t: T) -> T { t }
}
struct S;
impl G<i32> for S {}
fn main() {
    let d: &dyn G<i32> = &S;
    println!("{}", d.get(41) + 1);
}
"#;

#[test]
fn run_mode_dyn_trait_default_method_generic_trait() {
    // A trait with generic arguments: the instance name carries
    // them (`<S as G<i32>>::get`) while the dyn call name does
    // not, so the vtable lookup resolves by its unique scan.
    let dir = temp_dir("dyn-default-generic-trait");
    let src = fixture(&dir, "dyn_default_generic.rs", DYN_DEFAULT_GENERIC_TRAIT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "generic-trait dyn defaults must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_trait_default_method_body() {
    // A trait default method keeps its own body: X uses both
    // defaults, Y overrides `val`, and the default `extra` body
    // dispatches its inner `self.val()` call to Y's override.
    let dir = temp_dir("trait-default");
    let src = fixture(&dir, "trait_default.rs", TRAIT_DEFAULT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "trait default methods must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "10 11 20 21\n");
}

// ----- dst-index impl resolution (Lane BY) -----

const DST_INDEX_SRC: &str = r#"
use std::fmt::Debug;
use std::ops::Index;

struct S;

impl Index<usize> for S {
    type Output = str;

    fn index(&self, _: usize) -> &str {
        "hello"
    }
}

struct T;

impl Index<usize> for T {
    type Output = dyn Debug + 'static;

    fn index(&self, _: usize) -> &(dyn Debug + 'static) {
        static X: usize = 42;
        &X as &(dyn Debug + 'static)
    }
}

fn main() {
    assert_eq!(&S[0], "hello");
    let _ = &T[0];
    println!("dst index ok");
}
"#;

#[test]
fn run_mode_index_impls_with_dst_outputs_execute() {
    // The shape of tests/ui/dst/dst-index-success.rs (Issue #65):
    // the Index calls name the trait item, whose name-based builtin
    // model indexed the receiver struct itself and panicked with a
    // bogus out-of-bounds message; the collected impl bodies must
    // run instead, for both the `str` and the `dyn Debug` outputs.
    let dir = temp_dir("dstindex");
    let src = fixture(&dir, "dstindex.rs", DST_INDEX_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dst index program must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dst index ok\n");
}

#[test]
fn run_mode_generic_ctor_as_value_executes() {
    let dir = temp_dir("ctor-value");
    let src = fixture(&dir, "ctor_value.rs", CTOR_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a generic constructor used as a value must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "7\n");
}

const CTOR_HOF_SRC: &str = r#"
struct Wrap<T>(T);
fn apply<F: Fn(i32) -> Wrap<i32>>(f: F) -> i32 { f(41).0 + 1 }
fn main() {
    println!("{}", apply(Wrap));
}
"#;

#[test]
fn run_mode_generic_ctor_passed_to_hof_executes() {
    let dir = temp_dir("ctor-hof");
    let src = fixture(&dir, "ctor_hof.rs", CTOR_HOF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a generic constructor passed as a callback must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const CTOR_ENUM_VALUE_SRC: &str = r#"
enum E<T> { V(T, i32), W }
fn main() {
    let f = E::V;
    let e = f(5, 6);
    match e { E::V(a, b) => println!("{}", a + b), E::W => println!("w") }
}
"#;

#[test]
fn run_mode_generic_enum_variant_ctor_as_value_executes() {
    let dir = temp_dir("ctor-enum-value");
    let src = fixture(&dir, "ctor_enum_value.rs", CTOR_ENUM_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a generic variant constructor used as a value must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "11\n");
}

const CTOR_TWO_INSTANCES_SRC: &str = r#"
struct G<T>(T, i32);
fn main() {
    let fi = G::<i32>;
    let fs = G::<&str>;
    let a = fi(1, 2);
    let b = fs("x", 3);
    println!("{} {} {}", a.0 + a.1, b.0, b.1);
}
"#;

#[test]
fn run_mode_generic_ctor_two_instances_as_values_execute() {
    let dir = temp_dir("ctor-two-instances");
    let src = fixture(&dir, "ctor_two_instances.rs", CTOR_TWO_INSTANCES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "two instances of one generic constructor must both execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "3 x 3\n");
}

// ----- type-mismatch bucket fixes (Lane CD) -----

const BLANKET_FN_SRC: &str = r#"
fn a<F: Fn() -> i32>(f: F) -> i32 { f() }
fn b(f: &dyn Fn() -> i32) -> i32 { a(f) }
fn c<F: Fn() -> i32>(f: &F) -> i32 { a(f) }
fn main() {
    assert_eq!(b(&|| 22), 22);
    assert_eq!(c(&|| 23), 23);
    println!("blanket fn ok");
}
"#;

const FLOAT_EQ_SRC: &str = r#"
#[derive(PartialEq, Debug)]
struct P { x: f64, y: i32 }
#[derive(PartialEq, Debug)]
struct Pair(u32, u32);
fn main() {
    let a = P { x: 1.5, y: 2 };
    let b = P { x: 1.5, y: 2 };
    assert!(a == b);
    let nan = P { x: f64::NAN, y: 0 };
    assert!(nan != nan);
    assert!(Pair(3, 4) == Pair(3, 4));
    assert!(Pair(3, 4) != Pair(3, 5));
    let t1 = (1.25f32, 7u8);
    let t2 = (1.25f32, 7u8);
    assert!(t1 == t2);
    println!("float eq ok");
}
"#;

#[test]
fn run_mode_blanket_fn_impls_for_references() {
    let dir = temp_dir("blanket-fn");
    let src = fixture(&dir, "blanket_fn.rs", BLANKET_FN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "calling through &F / &dyn Fn blanket impls must work; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "blanket fn ok\n");
}

#[test]
fn run_mode_derived_eq_over_floats_and_tuple_structs() {
    let dir = temp_dir("float-eq");
    let src = fixture(&dir, "float_eq.rs", FLOAT_EQ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "derived equality over float fields must work; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "float eq ok\n");
}

// ----- derived Ord on enums: discriminant values (Lane CT) -----

const DERIVE_DISCRIMINANT_SRC: &str = r#"
// The shape of tests/ui/derives/eq-ord/derive-partial-ord-discriminant:
// derived PartialOrd must order by discriminant value, not by
// variant declaration order, including negative discriminants and
// 64-bit extremes. Operands are constants, so they also arrive as
// promoted by-reference constants.
#[derive(PartialEq, PartialOrd)]
enum E1 {
    Pos2 = 2,
    Neg1 = -1,
    Pos1 = 1,
}

#[derive(PartialEq, PartialOrd)]
#[repr(i8)]
enum E3 {
    Pos2 = 2,
    Neg1 = -1_i8,
    Pos1 = 1,
}

#[derive(PartialEq, PartialOrd)]
#[repr(i64)]
enum Ei64 {
    Pos2 = 2,
    Neg1 = -1,
    NegMin = 1 << 63,
    PosMax = !(1 << 63),
    Pos1 = 1,
}

fn main() {
    assert!(E1::Pos2 > E1::Pos1);
    assert!(E1::Pos1 > E1::Neg1);
    assert!(E1::Pos2 > E1::Neg1);

    assert!(E3::Pos2 > E3::Pos1);
    assert!(E3::Pos1 > E3::Neg1);
    assert!(E3::Pos2 > E3::Neg1);

    assert!(Ei64::Pos2 > Ei64::Pos1);
    assert!(Ei64::Pos2 > Ei64::Neg1);
    assert!(Ei64::Pos1 > Ei64::Neg1);
    assert!(Ei64::Pos2 > Ei64::NegMin);
    assert!(Ei64::Pos1 > Ei64::NegMin);
    assert!(Ei64::Pos2 < Ei64::PosMax);
    assert!(Ei64::Pos1 < Ei64::PosMax);
}
"#;

const DERIVE_NEGATIVE_EQ_SRC: &str = r#"
// Derived PartialEq and Ordering comparisons where a negative
// discriminant is involved: the constant operand is a promoted
// by-reference constant whose stored tag is the width-truncated
// pattern of the discriminant.
#[derive(PartialEq)]
enum P {
    X = -1,
    Y = 7,
}

fn check() -> u32 {
    let mut n = 0;
    if P::X == P::X { n += 1; }
    if P::X != P::Y { n += 1; }
    let o = core::cmp::Ordering::Less;
    if o == core::cmp::Ordering::Less { n += 1; }
    if o < core::cmp::Ordering::Equal { n += 1; }
    if core::cmp::Ordering::Greater > o { n += 1; }
    n
}

fn main() {
    if check() == 5 { println!("neg eq ok"); } else { println!("neg eq BAD"); }
}
"#;

const DERIVE_PAYLOAD_ORD_SRC: &str = r#"
// Derived PartialOrd/Ord on an enum with payloads: the
// discriminant decides across variants, fields within a variant.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum E {
    A(u8),
    B,
    C(u16),
}

fn check() -> u32 {
    let a1 = E::A(1);
    let a2 = E::A(2);
    let b = E::B;
    let c = E::C(0);
    let mut n = 0;
    if a1 < a2 { n += 1; }
    if a2 < b { n += 1; }
    if b < c { n += 1; }
    if a1 < c { n += 1; }
    if a1 == E::A(1) { n += 1; }
    if a1 != a2 { n += 1; }
    if a1.cmp(&a2) == core::cmp::Ordering::Less { n += 1; }
    if b.cmp(&c) == core::cmp::Ordering::Less { n += 1; }
    if c.cmp(&c) == core::cmp::Ordering::Equal { n += 1; }
    n
}

fn main() {
    if check() == 9 { println!("payload ord ok"); } else { println!("payload ord BAD"); }
}
"#;

#[test]
fn run_mode_derived_eq_negative_discriminant_constants() {
    let dir = temp_dir("derive-neg-eq");
    let src = fixture(&dir, "derive_neg_eq.rs", DERIVE_NEGATIVE_EQ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "negative-discriminant constant comparisons must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "neg eq ok\n",
        "derived equality and Ordering comparisons must hold"
    );
}

#[test]
fn run_mode_derived_ord_payload_enum() {
    let dir = temp_dir("derive-payload-ord");
    let src = fixture(&dir, "derive_payload_ord.rs", DERIVE_PAYLOAD_ORD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "derived Ord on a payload enum must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "payload ord ok\n",
        "derived Ord must order by discriminant, then fields"
    );
}

#[test]
fn run_mode_derived_partial_ord_uses_discriminant_values() {
    let dir = temp_dir("derive-discr");
    let src = fixture(&dir, "derive_discr.rs", DERIVE_DISCRIMINANT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "derived PartialOrd on explicit discriminants must run and hold; stderr={stderr}"
    );
}

// ----- monomorphization validity, round 2 (Lane CS) -----

const BOUNDED_RECURSION_SRC: &str = r#"
fn fact(n: u64) -> u64 { if n == 0 { 1 } else { n * fact(n - 1) } }
fn down<T: Copy>(x: Option<T>) -> usize { match x { None => 0, Some(_) => down::<T>(x) } }
fn main() { let _ = fact(3); let _ = down::<i32>(Some(1)); }
"#;

const HUGE_PHANTOM_ONLY_SRC: &str = r#"
fn f<T>(_x: std::marker::PhantomData<T>) -> usize { std::mem::size_of::<usize>() }
fn main() { let _ = f::<[u8; usize::MAX]>(std::marker::PhantomData); }
"#;

#[test]
fn convert_mode_bounded_generic_recursion_converts() {
    let dir = temp_dir("mono-bounded-recursion");
    let src = fixture(&dir, "bounded.rs", BOUNDED_RECURSION_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "plain recursion and same-instance generic recursion are \
         finite demand and must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_huge_type_named_only_through_phantom_data_converts() {
    let dir = temp_dir("mono-huge-phantom");
    let src = fixture(&dir, "huge_phantom.rs", HUGE_PHANTOM_ONLY_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a too-big type that is only named (PhantomData, an ordinary \
         generic argument) is never materialized and must convert; stderr={stderr}"
    );
}

const TRANSMUTE_NICHE_ENUM_SRC: &str = r#"
#[repr(u16)]
enum Mix { A, B(u16) }
enum Nested { C(Mix), D, E }
struct Bar { a: u16, b: u16 }
fn code(n: &Nested) -> u32 {
    match n {
        Nested::C(Mix::A) => 0,
        Nested::C(Mix::B(x)) => 100 + (*x as u32),
        Nested::D => 2,
        Nested::E => 3,
    }
}
fn main() {
    let a: Nested = unsafe { std::mem::transmute(Bar { a: 0, b: 0 }) };
    let b: Nested = unsafe { std::mem::transmute(Bar { a: 1, b: 100 }) };
    let c: Nested = unsafe { std::mem::transmute(Bar { a: 2, b: 0 }) };
    let d: Nested = unsafe { std::mem::transmute(Bar { a: 3, b: 0 }) };
    let e: Nested = unsafe { std::mem::transmute(0u32) };
    println!("{} {} {} {} {}", code(&a), code(&b), code(&c), code(&d), code(&e));
}
"#;

#[test]
fn run_mode_transmute_niche_enum_from_struct_and_int() {
    // A struct whose first field is a payload enum's tag
    // transmuted to an outer enum whose niche lives inside
    // that payload: the tag selects among the payload's own
    // variants and the outer enum's extra variants, from a
    // struct operand and from a bare integer alike. Under
    // debug assertions the compiler also reads the input's
    // bytes through a subslice to validate the construction.
    run_ok(
        "transmute-niche-enum",
        TRANSMUTE_NICHE_ENUM_SRC,
        "0 200 2 3 0\n",
    );
}

const TRANSMUTE_TAGGED_ENUM_SRC: &str = r#"
#[repr(u16)]
enum Mix { A, B(u16) }
struct Bar { a: u16, b: u16 }
fn code(m: &Mix) -> u32 {
    match m {
        Mix::A => 0,
        Mix::B(x) => 100 + (*x as u32),
    }
}
fn main() {
    let a: Mix = unsafe { std::mem::transmute(Bar { a: 0, b: 0 }) };
    let b: Mix = unsafe { std::mem::transmute(Bar { a: 1, b: 100 }) };
    println!("{} {}", code(&a), code(&b));
}
"#;

#[test]
fn run_mode_transmute_tagged_enum_from_struct() {
    // A struct transmuted to a direct-tag enum: the tag
    // field at its recorded offset selects the variant and
    // the remaining fields land at their layout offsets.
    run_ok(
        "transmute-tagged-enum",
        TRANSMUTE_TAGGED_ENUM_SRC,
        "0 200\n",
    );
}

const TRANSMUTE_OPTION_BIGGER_NICHE_SRC: &str = r#"
#[derive(Debug, PartialEq)]
#[repr(u32)]
enum Foo { A = 2, B }
fn main() {
    let a: Option<Foo> = unsafe { std::mem::transmute::<u32, Option<Foo>>(2) };
    let b: Option<Foo> = unsafe { std::mem::transmute::<u32, Option<Foo>>(3) };
    assert_eq!(a, Some(Foo::A));
    assert_eq!(b, Some(Foo::B));
    println!("ok");
}
"#;

#[test]
fn run_mode_transmute_int_to_option_of_fieldless_enum() {
    // An integer transmuted to `Option` of a fieldless enum
    // whose discriminants do not start at zero: the payload
    // enum's own scalar is the niche, its discriminant
    // patterns select the `Some` variants.
    run_ok(
        "transmute-option-bigger-niche",
        TRANSMUTE_OPTION_BIGGER_NICHE_SRC,
        "ok\n",
    );
}

const NICHE_GENERAL_SRC: &str = r#"
use std::num::NonZero;
fn main() {
    let nz = NonZero::new(1u64 << 40);
    let bits: u64 = unsafe { std::mem::transmute(nz) };
    println!("bits {}", bits);
    let opt: Option<NonZero<u32>> = unsafe { std::mem::transmute(0u32) };
    println!("zero-none {}", opt.is_none());
    let nz8 = NonZero::new(200u8);
    println!("nz8 {}", nz8.map(|n| n.get()).unwrap_or(0));
    let s: Option<NonZero<u32>> = Some(NonZero::new(77u32).unwrap());
    let sbits: u32 = unsafe { std::mem::transmute(s) };
    println!("sbits {}", sbits);
    let b2: Option<bool> = unsafe { std::mem::transmute(2u8) };
    let back: u8 = unsafe { std::mem::transmute(b2) };
    println!("bback {}", back);
    match NonZero::new(0u32) {
        None => println!("new0 none"),
        Some(n) => println!("new0 some {}", n.get()),
    }
}
"#;

#[test]
fn run_mode_niche_enum_transmutes_general() {
    // The niche encoding is not NonZero-specific: widths
    // other than 32 bits and `Option<bool>` (whose None is
    // the value 2, not 0) follow the same recorded encoding.
    run_ok(
        "niche-general",
        NICHE_GENERAL_SRC,
        "bits 1099511627776\nzero-none true\nnz8 200\nsbits 77\nbback 2\nnew0 none\n",
    );
}

const RC_ARC_DYN_SRC: &str = r#"
use std::rc::Rc;
use std::sync::Arc;
trait T {
    fn val(&self) -> i32;
    fn d(&self) -> i32 { 7 }
}
struct S(i32);
impl T for S {
    fn val(&self) -> i32 { self.0 }
}
fn main() {
    let r: Rc<dyn T> = Rc::new(S(42));
    let r2 = r.clone();
    let a: Arc<dyn T> = Arc::new(S(43));
    println!("{} {} {} {}", r.val(), r2.val(), r.d(), a.val());
}
"#;

#[test]
fn run_mode_rc_arc_dyn_trait_method_calls() {
    // Method calls through an `Rc<dyn Trait>` / `Arc<dyn Trait>`
    // receiver: the Deref builtin must yield the dyn fat pointer
    // itself (as it does for Box), and dispatch must reach both
    // the impl method (reading the pointee's state) and the
    // trait's default method.
    let dir = temp_dir("rc-arc-dyn");
    let src = fixture(&dir, "rc_arc_dyn.rs", RC_ARC_DYN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "calling trait methods through Rc<dyn Trait> / Arc<dyn Trait> must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42 42 7 43\n");
}

const DYN_DEFAULT_DISPATCH_SRC: &str = r#"
use std::rc::Rc;
trait T {
    fn val(&self) -> i32;
    fn extra(&self) -> i32 { self.val() + 1 }
    fn d(&self) -> i32 { 7 }
}
struct A;
impl T for A { fn val(&self) -> i32 { 10 } }
struct B;
impl T for B {
    fn val(&self) -> i32 { 20 }
    fn d(&self) -> i32 { 8 }
}
fn main() {
    let a: Rc<dyn T> = Rc::new(A);
    let b: Rc<dyn T> = Rc::new(B);
    println!("{} {} {} {}", a.extra(), a.d(), b.extra(), b.d());
    let r: &dyn T = &A;
    println!("{} {}", r.extra(), r.d());
}
"#;

#[test]
fn run_mode_dyn_default_method_dispatch() {
    // A trait method the impl does not override has only the
    // trait's default body: dispatch through a dyn receiver must
    // reach it (monomorphized at the concrete type), its inner
    // self calls must hit the concrete overrides, and an
    // overriding impl method must win over the default.
    let dir = temp_dir("dyn-default-dispatch");
    let src = fixture(&dir, "dyn_default.rs", DYN_DEFAULT_DISPATCH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dispatching to trait default methods through dyn receivers must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "11 7 21 8\n11 7\n");
}

const DYN_EQ_REF_SRC: &str = r#"
trait Foo: PartialEq<i32> + std::fmt::Debug {
    fn a(&self) -> i32 { 100 }
}
impl Foo for i32 {}
fn main() {
    let x: i32 = 1;
    assert_eq!(x.a(), 100);
    let foo: &dyn Foo = &x;
    assert_eq!(*foo, 1);
    assert!(*foo != 2);
    println!("dyn-eq-ref ok");
}
"#;

#[test]
fn run_mode_dyn_ref_partial_eq_compares_pointee() {
    let dir = temp_dir("dyn-eq-ref");
    let src = fixture(&dir, "dyn_eq_ref.rs", DYN_EQ_REF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "comparing through a &dyn trait object with a PartialEq supertrait must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-eq-ref ok\n");
}

const DYN_EQ_ADT_SRC: &str = r#"
trait Foo: PartialEq<Bar> + std::fmt::Debug {}
#[derive(PartialEq, Debug)]
struct Bar(i32);
impl Foo for Bar {}
fn main() {
    let obj: &dyn Foo = &Bar(7);
    assert_eq!(*obj, Bar(7));
    assert!(*obj != Bar(8));
    println!("dyn-eq-adt ok");
}
"#;

#[test]
fn run_mode_dyn_ref_partial_eq_against_adt() {
    let dir = temp_dir("dyn-eq-adt");
    let src = fixture(&dir, "dyn_eq_adt.rs", DYN_EQ_ADT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "comparing a trait object against an ADT value must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-eq-adt ok\n");
}

const DYN_EQ_DYN_SRC: &str = r#"
trait T: std::fmt::Debug {
    fn val(&self) -> i32;
    fn type_id(&self) -> u32;
}
#[derive(Debug)]
struct A(i32);
#[derive(Debug)]
struct B(i32);
impl T for A {
    fn val(&self) -> i32 { self.0 }
    fn type_id(&self) -> u32 { 1 }
}
impl T for B {
    fn val(&self) -> i32 { self.0 }
    fn type_id(&self) -> u32 { 2 }
}
impl PartialEq for dyn T {
    fn eq(&self, other: &Self) -> bool {
        self.type_id() == other.type_id() && self.val() == other.val()
    }
}
fn main() {
    let a1: &dyn T = &A(1);
    let a2: &dyn T = &A(1);
    let a3: &dyn T = &A(2);
    let b1: &dyn T = &B(1);
    assert!(a1 == a2);
    assert!(a1 != a3);
    assert!(a1 != b1);
    println!("dyn-eq-dyn ok");
}
"#;

#[test]
fn run_mode_dyn_partial_eq_against_dyn() {
    let dir = temp_dir("dyn-eq-dyn");
    let src = fixture(&dir, "dyn_eq_dyn.rs", DYN_EQ_DYN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "comparing two trait objects must run (unequal concrete types are unequal); stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-eq-dyn ok\n");
}

const DYN_EQ_UPCAST_SRC: &str = r#"
trait Super: PartialEq<i32> + std::fmt::Debug {}
trait Sub: Super {}
impl Super for i32 {}
impl Sub for i32 {}
fn main() {
    let sub: &dyn Sub = &1;
    let sup: &dyn Super = sub;
    assert_eq!(*sup, 1);
    assert!(*sup != 2);
    let boxed: Box<dyn Sub> = Box::new(5);
    let boxed_super: Box<dyn Super> = boxed;
    assert_eq!(*boxed_super, 5);
    assert!(*boxed_super != 6);
    println!("dyn-eq-upcast ok");
}
"#;

#[test]
fn run_mode_upcast_dyn_partial_eq_compares_pointee() {
    let dir = temp_dir("dyn-eq-upcast");
    let src = fixture(&dir, "dyn_eq_upcast.rs", DYN_EQ_UPCAST_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "comparing through an upcast trait object must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-eq-upcast ok\n");
}


const WRAPPING_SRC: &str = r#"
fn main() {
    let a = 255u8.wrapping_add(1);
    assert_eq!(a, 0);
    let b = i64::MAX.wrapping_add(1);
    assert_eq!(b, i64::MIN);
    println!("ok");
}
"#;

#[test]
fn run_mode_wrapping_methods_wrap_in_a_checked_session() {
    // The plain arithmetic inside the `wrapping_*` method bodies
    // wraps; only the `Assert` after a checked operation may panic,
    // and these bodies contain none.
    run_ok("wrapping", WRAPPING_SRC, "ok\n");
}

const PHANTOM_DEFAULT_SRC: &str = r#"
use std::marker::PhantomData;
#[derive(Default)]
struct Wrap<T> { _p: PhantomData<T>, x: i32 }
fn main() {
    let w: Wrap<String> = Wrap::default();
    assert_eq!(w.x, 0);
    println!("ok");
}
"#;

#[test]
fn run_mode_phantom_data_default() {
    let dir = temp_dir("phantom-default");
    let src = fixture(&dir, "phantom_default.rs", PHANTOM_DEFAULT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const UFCS_FALLBACK_SRC: &str = r#"
use std::iter::FromIterator;

fn collect_into<I: IntoIterator<Item = i32>>(iter: I) -> Vec<i32> {
    <Vec<i32> as FromIterator<i32>>::from_iter(iter)
}

fn main() {
    let v = collect_into(Some(5).into_iter());
    assert_eq!(v.len(), 1);
    assert_eq!(v[0], 5);
    let add: fn(i32, i32) -> i32 = <i32 as std::ops::Add>::add;
    assert_eq!(add(5, 6), 11);
    let sub: fn(u32, u32) -> u32 = <u32 as std::ops::Sub>::sub;
    assert_eq!(sub(10, 4), 6);
    let neg: fn(i32) -> i32 = <i32 as std::ops::Neg>::neg;
    assert_eq!(neg(7), -7);
    let not: fn(bool) -> bool = <bool as std::ops::Not>::not;
    assert_eq!(not(true), false);
    println!("ok");
}
"#;

#[test]
fn run_mode_ufcs_impl_qualified_fallbacks() {
    // Calls whose names conversion kept in impl-qualified
    // generic form have no body and no plain trait-path lookup
    // entry: the run-time name fallback serves the ones whose
    // semantics are value-level — `FromIterator::from_iter` for
    // a Vec target (the name carries the target; the iterator
    // state steps by its shape) and the primitive arithmetic
    // traits (the name carries the Self type whose width the
    // operation computes at).
    let dir = temp_dir("ufcs-fallback");
    let src = fixture(&dir, "ufcs_fallback.rs", UFCS_FALLBACK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "impl-qualified UFCS calls must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const DYN_ANY_GENERIC_PAYLOADS_SRC: &str = r#"
use std::any::Any;
use std::cell::{Cell, RefCell};
fn main() {
    let b: Box<dyn Any> = Box::new(Cell::new(1u32));
    println!("cell is: {}", b.is::<Cell<u32>>());
    match b.downcast::<Cell<u32>>() {
        Ok(c) => println!("cell: {}", c.get()),
        Err(_) => println!("cell FAILED"),
    }
    let b: Box<dyn Any> = Box::new(vec![1u32, 2, 3]);
    println!("vec is: {}", b.is::<Vec<u32>>());
    match b.downcast::<Vec<u32>>() {
        Ok(v) => println!("vec len: {}", v.len()),
        Err(_) => println!("vec FAILED"),
    }
    let b: Box<dyn Any> = Box::new(RefCell::new(4u32));
    println!("refcell is: {}", b.is::<RefCell<u32>>());
    match b.downcast::<RefCell<u32>>() {
        Ok(c) => println!("refcell: {}", c.into_inner()),
        Err(_) => println!("refcell FAILED"),
    }
}
"#;

#[test]
fn run_mode_dyn_any_downcast_generic_payloads() {
    // The concrete tag a `dyn Any` value carries is the fully
    // rendered type, generic arguments included — the same
    // rendering a downcast's generic argument carries. Payloads
    // whose types have generic arguments (Cell, Vec, RefCell)
    // therefore answer `is` and `downcast` like argument-free
    // payloads do.
    run_ok(
        "dyn-any-generic",
        DYN_ANY_GENERIC_PAYLOADS_SRC,
        "cell is: true\ncell: 1\nvec is: true\nvec len: 3\nrefcell is: true\nrefcell: 4\n",
    );
}

const DYN_ANY_DOWNCAST_REF_SRC: &str = r#"
use std::any::Any;
fn main() {
    let b: Box<dyn Any> = Box::new(vec![10u32, 20, 30]);
    match b.downcast_ref::<Vec<u32>>() {
        Some(v) => println!("ref len {} first {}", v.len(), v[0]),
        None => println!("ref FAILED"),
    }
}
"#;

#[test]
fn run_mode_dyn_any_downcast_ref_denotes_the_payload() {
    // `downcast_ref` denotes the payload the box points at, not
    // the box aggregate itself (which would report the box's own
    // one-field shape, e.g. a length of 1 for a 3-element Vec).
    run_ok("dyn-any-ref", DYN_ANY_DOWNCAST_REF_SRC, "ref len 3 first 10\n");
}

const TUPLE_CLONE_UNWIND_SRC: &str = r#"
use std::rc::Rc;

struct S(Rc<()>);

impl Clone for S {
    fn clone(&self) -> Self {
        if Rc::strong_count(&self.0) == 7 {
            panic!("oops");
        }
        S(self.0.clone())
    }
}

fn main() {
    let counter = Rc::new(());
    let ccounter = counter.clone();
    let result = std::panic::catch_unwind(move || {
        let _ =
            (S(ccounter.clone()), S(ccounter.clone()), S(ccounter.clone()), S(ccounter)).clone();
    });
    assert!(result.is_err());
    assert_eq!(1, Rc::strong_count(&counter));
    println!("tuple clone unwind ok");
}
"#;

#[test]
fn run_mode_tuple_clone_runs_element_impls_and_cleans_up() {
    // A tuple's `Clone` impl is a compiler shim: with its body
    // demanded, cloning a tuple runs each element's own `Clone`
    // impl in order, and a panic mid-clone drops the elements
    // already cloned (the shim's unwind cleanup), which the
    // reference count observes.
    let dir = temp_dir("tuple-clone-unwind");
    let src = fixture(&dir, "tuple_clone_unwind.rs", TUPLE_CLONE_UNWIND_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "tuple clone must run element impls and clean up on panic; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "tuple clone unwind ok\n");
}

const NESTED_REVERSE_SRC: &str = r#"
fn main() {
    let mut v: Vec<[u32; 4]> = vec![[1, 2, 3, 4], [5, 6, 7, 8]];
    v[0].reverse();
    v[1].reverse();
    println!("{:?}", v);
    let mut a = [[1u32, 2, 3], [4, 5, 6]];
    a[1].reverse();
    println!("{:?}", a);
}
"#;

#[test]
fn run_mode_nested_aggregate_element_reverse() {
    // Reversing one row of a nested aggregate (`v[0].reverse()`)
    // swaps through pointers displaced from an element position
    // inside the row: the displacement must absorb into the
    // row's own index space (the innermost container the
    // pointer's element size denominates), or the swap reads and
    // writes the wrong elements — it used to splice bytes into
    // the first element's storage and leave the row's tail
    // untouched.
    let dir = temp_dir("nestedreverse");
    let src = fixture(&dir, "nestedreverse.rs", NESTED_REVERSE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested aggregate element reverse must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[[4, 3, 2, 1], [8, 7, 6, 5]]\n[[1, 2, 3], [6, 5, 4]]\n",
        "each row reverses in place, exactly as natively"
    );
}
