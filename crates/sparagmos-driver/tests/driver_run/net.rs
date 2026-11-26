use super::{fixture, run_driver, temp_dir};

const BIND_SRC: &str = r#"
use std::net::TcpListener;
fn main() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    assert_eq!(addr.ip().to_string(), "127.0.0.1");
    assert!(addr.port() != 0, "an ephemeral port is assigned");
    println!("bind ok");
}
"#;

#[test]
fn run_mode_tcp_listener_bind_ephemeral_port() {
    let dir = temp_dir("net-bind");
    let src = fixture(&dir, "net_bind.rs", BIND_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "binding a listener and reading its local address runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "bind ok\n");
}

const ECHO_SRC: &str = r#"
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
fn main() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let mut client = TcpStream::connect(addr).unwrap();
    let (mut server, peer) = listener.accept().unwrap();
    assert_eq!(peer, client.local_addr().unwrap());
    client.write_all(b"hello").unwrap();
    let mut buf = [0u8; 5];
    server.read_exact(&mut buf).unwrap();
    assert_eq!(&buf, b"hello");
    server.write_all(b"world").unwrap();
    let mut buf2 = [0u8; 5];
    client.read_exact(&mut buf2).unwrap();
    assert_eq!(&buf2, b"world");
    println!("echo ok");
}
"#;

#[test]
fn run_mode_tcp_stream_echo_roundtrip() {
    let dir = temp_dir("net-echo");
    let src = fixture(&dir, "net_echo.rs", ECHO_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "connect/accept with a write/read round trip runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "echo ok\n");
}

const REBIND_SRC: &str = r#"
use std::net::TcpListener;
fn main() {
    let port = {
        let first = TcpListener::bind("127.0.0.1:0").unwrap();
        first.local_addr().unwrap().port()
    };
    // The first listener is dropped here: its socket is
    // closed, so binding the same port again succeeds.
    let second = TcpListener::bind(("127.0.0.1", port)).unwrap();
    assert_eq!(second.local_addr().unwrap().port(), port);
    println!("rebind ok");
}
"#;

#[test]
fn run_mode_tcp_listener_drop_releases_port() {
    let dir = temp_dir("net-rebind");
    let src = fixture(&dir, "net_rebind.rs", REBIND_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dropping a listener closes its socket so the port can be rebound; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "rebind ok\n");
}

const BLOCKING_ACCEPT_SRC: &str = r#"
use std::net::TcpListener;
use std::sync::mpsc;
fn main() {
    let (tx, rx) = mpsc::channel::<()>();
    std::thread::spawn(move || {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        tx.send(()).unwrap();
        while let Ok(_) = listener.accept() {}
    });
    rx.recv().unwrap();
    for _ in 0..3 {
        std::thread::yield_now();
    }
    println!("main exits");
}
"#;

#[test]
fn run_mode_blocking_accept_thread_does_not_block_exit() {
    let dir = temp_dir("net-blocking-accept");
    let src = fixture(&dir, "net_blocking_accept.rs", BLOCKING_ACCEPT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the process terminates while a spawned thread is parked in accept; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "main exits\n");
}

const UDP_SRC: &str = r#"
use std::net::UdpSocket;
fn main() {
    let a = UdpSocket::bind("127.0.0.1:0").unwrap();
    let b = UdpSocket::bind("127.0.0.1:0").unwrap();
    let a_addr = a.local_addr().unwrap();
    let sent = b.send_to(b"ping", a_addr).unwrap();
    assert_eq!(sent, 4);
    let mut buf = [0u8; 8];
    let (n, from) = a.recv_from(&mut buf).unwrap();
    assert_eq!(n, 4);
    assert_eq!(&buf[..4], b"ping");
    assert_eq!(from, b.local_addr().unwrap());
    println!("udp ok");
}
"#;

#[test]
fn run_mode_udp_send_recv_roundtrip() {
    let dir = temp_dir("net-udp");
    let src = fixture(&dir, "net_udp.rs", UDP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "UDP bind/send_to/recv_from runs; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "udp ok\n");
}
