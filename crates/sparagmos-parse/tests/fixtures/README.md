# MIR fixtures

Generated on fleur-du-lapin with Rust 1.100.0 (released 2026-11-12):

    rustup run 1.100.0 rustc --edition 2021 --emit=mir -o const_return.mir const_return.rs
    rustup run 1.100.0 rustc --edition 2021 -C overflow-checks=off --emit=mir -o binop.mir binop.rs
    rustup run 1.100.0 rustc --edition 2021 --emit=mir -o switch.mir switch.rs
    rustup run 1.100.0 rustc --edition 2021 --emit=mir -o copy_move.mir copy_move.rs
