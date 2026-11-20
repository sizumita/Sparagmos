// corpus: string output — &str, String, format!, char handling.
fn main() {
    let name = "sparagmos";
    let owned = String::from("jit");
    println!("{} {}", name, owned);
    println!("{}", format!("{}-{}", name, owned).to_uppercase());
}
