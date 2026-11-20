// corpus: enums with unsigned payloads passed by value — match and extraction.
enum Message {
    Quit,
    Move { x: u32, y: u32 },
    Code(u32),
    Pair(u32, u32),
}
fn value(m: Message) -> u32 {
    match m {
        Message::Quit => 0,
        Message::Move { x, y } => x + y,
        Message::Code(c) => c * 10,
        Message::Pair(a, b) => a * b,
    }
}
fn main() {
    let total = value(Message::Quit)
        + value(Message::Move { x: 3, y: 4 })
        + value(Message::Code(5))
        + value(Message::Pair(6, 7));
    println!("{}", total);
}
