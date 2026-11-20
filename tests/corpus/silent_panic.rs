// corpus: a guest panic without any output — integer overflow in a
// helper (the operand is a parameter, so nothing is const-folded).
// Both the native build (opt-level=0, overflow checks on) and
// Sparagmos are expected to exit with the panic code 101 and no
// stdout.
fn boom(x: i32) -> i32 {
    x + x
}
fn main() {
    let _ = boom(2_000_000_000);
}
