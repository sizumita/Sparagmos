# Sparagmos design

Sparagmos は Rust の MIR を入力とする、完全に独自の JIT ランタイムである。
目標は「Rust 1.100.0 を 99% のレベルで実行できる」ことであり、最終的には
`rustc` 自体とそのテストスイートを Sparagmos 上で動かす。本書はアーキテクチャの
決定事項を記録する。「99%」の測り方は `docs/COVERAGE.md` が定義し、本書の
マイルストーンの完了条件はすべてそちらの指標で書く。

## 前提

- 対象バージョンは **Rust 1.100.0** (2026-11-12 リリース、
  公式ブログ: https://blog.rust-lang.org/2026/11/12/Rust-1.100.0/)。
  母数・実測はすべて 1.100.0 で取る。
- 開発ホスト `fleur-du-lapin` (x86_64 Linux, 72 threads, 157 GiB RAM) に
  rustup で 1.100.0 と `rustc-dev` コンポーネントを導入済み:
  `rustc 1.100.0 (0ef1bc386 2026-11-09)`、`cargo 1.100.0 (3acc79c0c 2026-11-05)`。
- 実行対象ホストは当面 x86_64 Linux のみ。他アーキテクチャは非目標。

## 全体アーキテクチャ

入力経路は **Stable MIR** に一本化する。

```
foo.rs ──> sparagmos-driver  (rustc 1.100.0 を Stable MIR 経由で駆動し、
                              MIR を取得して自前モデルへ変換)
        ──> sparagmos-mir     (型・CFG・レイアウトの自前モデル)
        ──> sparagmos-interp  (インタプリタ)
        ──> sparagmos-jit     (自前 x86_64 エミッタによる JIT)
        ──> sparagmos-runtime (alloc / panic / intrinsics / std 支援)
sparagmos-cli     : コーパス実行などの入口
sparagmos-harness : コーパス実行・差分検証・カバレッジ集計
```

Stable MIR を入力とする理由:

- MIR を構造化された API で取得できるため、変換は型のあるデータ同士の
  対応付けになり、未知のバリアントは変換時に型レベルで検出できる。
- rustc のバージョン差はドライバと変換層の境界に閉じる。実行系
  (interp / jit / runtime) は自前モデル `sparagmos-mir` だけを見るため、
  将来のバージョンへ追随する際の変更範囲がその境界に限定される。
- テキスト形式の MIR は人間向けの出力であり、形式の安定性が約束されて
  いないため、入力には使わない。

ドライバの構成上の注意:

- ドライバは rustc 1.100.0 と `RUSTC_BOOTSTRAP=1` を必要とするため、
  専用のワークスペースとして分離し、stable toolchain でビルドする
  本体ワークスペースからは `exclude` する。本体ワークスペースの
  ビルド・テストはドライバに依存しない。
- MIR の取得は Stable MIR (`rustc_public`) 経由で行う。内部 API への
  直接依存は、マクロ展開とトレイト参照のために必要な最小限に限る。
- 変換できないバリアントを含む関数は、関数単位で理由を数えて報告する。
  黙って落としたり、近い意味に読み替えたりはしない。
- 定数は Stable MIR のアロケーションを実読取して解釈する。スカラと
  スカラ配列は実体化し、それ以外の参照定数はアロケーション ID を
  保持した形で表現する。

### 実行の段階 (staged execution)

1. **インタプリタ (sparagmos-interp)**: 全バリアントの意味論を最初にここで
   実装する。新バリアント対応は必ずインタプリタから入れる。インタプリタは
   意味論の基準であり、JIT の出力は常に対照される。
2. **ベースライン JIT (sparagmos-jit 初期)**: メソッド単位で基本ブロックを
   そのまま x86_64 に落とす。最適化より網羅率を優先し、インタプリタと
   結果が一致することを完了条件とする。
3. **最適化 JIT (sparagmos-jit 後期)**: ホットな関数のレジスタ割付改善、
   インライン化、不要な境界検査の除去など。具体的な最適化項目は
   M2 完了後に Issue で決める (本書では約束しない)。

各実行エンジンの対応範囲は個別に測る。あるエンジンが扱えない関数は、
そのエンジンの経路では「未対応」として理由付きで報告される。
別エンジンで代替実行する仕組みは持たない。

段階の移行は感覚ではなく `COVERAGE.md` の指標で判定する (下のマイルストーン表)。

## クレート責務

| クレート | 責務 | 依存してよい相手 |
|---|---|---|
| `sparagmos-mir` | MIR の自前モデル: Body/BasicBlock/Statement/Terminator/Rvalue、型とレイアウト、Place 投影、カバレッジ表 | なし (最下層) |
| `sparagmos-driver` | rustc を Stable MIR 経由で駆動し、MIR を `sparagmos-mir` 型へ変換する。run モードでは変換結果をインタプリタで実行する | `sparagmos-mir`, `sparagmos-interp` |
| `sparagmos-interp` | MIR インタプリタ。全バリアントの参照意味論、panic/drop の参照挙動 | `sparagmos-mir` |
| `sparagmos-jit` | 自前 x86_64 アセンブラ/エミッタ、ベースライン JIT、後期の最適化 JIT | `sparagmos-mir`, `sparagmos-runtime` |
| `sparagmos-runtime` | メモリ確保、panic/unwind 支援、intrinsics、std 機能の実行時支援、外部関数 (libc 等) への橋渡し | `sparagmos-mir` |
| `sparagmos-cli` | `sparagmos` バイナリ: コーパス実行などの入口、終了コード/出力の受け渡し | 全クレート |
| `sparagmos-harness` | コーパス管理、ネイティブとの差分検証、`COVERAGE.md` の3指標の集計 | 全クレート |

依存は一方向のみとし、循環を作らない。特に `sparagmos-mir` は他クレートに
依存しない (モデルが実行方式に引きずられないため)。

## 独自性の境界

使わないもの (コアの実装に導入しない):

- Cranelift、LLVM (llvm-sys / inkwell 等)、Wasmtime / wasmi、その他既存の
  JIT・コード生成・VM ライブラリ。
- Miri のコードの流用・リンク (意味論の参考にドキュメントを読むのはよいが、
  実装は自前で書く)。
- rustc のソースコードのコピー。

使うもの:

- **ホスト rustc 1.100.0**: Stable MIR 経由の MIR 取得元として、および
  差分検証における参照実装 (ネイティブ実行結果) として使う。これは
  「コンパイラを借りる」のではなく「入力と正解データを得る」用途であり、
  Sparagmos の実行系そのものには入らない。
- **OS 基盤**: `mmap`/`mprotect` による実行可能メモリ、スレッド、ファイル IO。
  実装言語である Rust の std 自体も Sparagmos 自身の実装には使う。
  ただし「実行対象プログラムが使う std」は `sparagmos-runtime` が担う別物であり、
  混同しない。

機械コードは自前エミッタが x86_64 命令を直接符号化して作る。

## 値表現・レイアウトの方針

- 値はタグ付き表現にせず、型ごとのレイアウトに従うバイト列として表す。
  レイアウト規則 (スカラ、スカラペア、集約型、配列) は rustc の規則を
  自前で再実装し、`sparagmos-mir` に置く。
- 参照・生ポインタは実アドレスで表す。provenance の厳密な追跡
  (Stacked/Tree Borrows の検証) は非目標であり、エイリアシング違反の
  検出はしない (Miri との明確な差別化点)。
- スライスやトレイトオブジェクトなどの fat pointer は (データポインタ,
  メタデータ) のペアで表す。メタデータは長さまたは vtable ポインタ。
- enum は discriminant + ペイロードで表す。niche 最適化
  (`Option<&T>` の null 表現など) は M3 で段階導入し、それまでは
  素直な表現で正しさを優先する。ドライバが Stable MIR から得た
  レイアウト前提と食い違う場合は、その型を「未対応」と明示して落とす
  (黙って壊さない)。
- ZST はバイトを占有しない。1.100.0 で安定化した `core::mem::conjure_zst` で
  作られる値もこの規則で扱い、特別なストレージを持たせない。
- 浮動小数点はホストの x87/SSE 意味論そのまま (ビットパターンを保持する)。

## 呼び出し規約の方針

- Sparagmos 内部の関数呼び出し (MIR の `Call` 終端) は、まずインタプリタの
  フレームモデルを正とし、JIT では「引数は呼び出し先フレームの決められた
  スロットに置く」という単純な自前規約から始める。戻り値は呼び出し元の
  destination place へ `_0` 経由で書く、という MIR の意味論をそのまま守る。
- ランタイム関数や外部関数 (libc 等) との境界では System V AMD64 ABI に従う。
  内部規約と ABI の変換は境界に閉じ込める。
- `TailCall` 終端は当面、通常の `Call` + `Return` として実行して意味論を
  等価に保つ (スタック消費が増える制限を許容し、制限として記録する)。
  末尾呼び出し最適化そのものは M4 以降の検討事項。

## panic / drop / unwind の方針

- `Assert` 終端と checked 演算の失敗は panic の開始点として扱う。
  panic メッセージと位置情報は可能な範囲で保持し、ネイティブ実行との
  差分検証 (指標2) で stdout/stderr の一致を見る。
- `Drop` 終端は drop glue を通常の関数呼び出しとして実行する。ドロップの
  順序は MIR の CFG が決める順序そのままとし、Sparagmos 側で並べ替えない。
  1.100.0 の `core::mem::DropGuard` も通常の ADT + `Drop` 実装として現れる
  ため、専用対応は不要という前提で進め、実測で崩れたら Issue 化する。
- unwind: インタプリタでは cleanup ブロックと `UnwindResume` を完全に
  実行できるようにする (catch_unwind を使うコーパスが存在するため)。
  JIT では当初 `panic=abort` 相当 (panic 時にプロセスを異常終了させる)
  のみを保証し、cleanup を跨ぐ unwind の完全対応は M4 の完了条件に入れる。
- 二重 panic (cleanup 中の panic) は abort として扱う。

## 並行性・メモリの方針

- 開始時点はシングルスレッド実行を前提とし、`std::thread` を要する
  プログラムは M3 以降の対象とする。原子操作はランタイム関数へ落とす。
- メモリ確保はランタイムが一元管理する。1.100.0 でコア部分が安定化した
  Allocator API (`Allocator` トレイト、`Vec`/`Box` の `_in` 系コンストラクタ)
  については、カスタムアロケータをランタイムの確保フックで受ける設計とし、
  対応範囲は指標3のチェックリストで管理する。

## 段階的マイルストーン

完了条件はすべて `docs/COVERAGE.md` の指標で測る。進捗率の推測値は書かない。

| 段階 | 内容 | 完了条件 (検証可能な形) |
|---|---|---|
| M0 骨格 | ワークスペース、クレートの雛形、CI、harness の枠、コーパス置き場 | fleur 上で `cargo build` / `cargo test` が通り、GitHub に Push 済み |
| M1 インタプリタで小コーパス | ドライバ + インタプリタで `tests/corpus` を実行 | 指標2(a): コーパス全プログラムでインタプリタ結果がネイティブと一致。指標1: 実測コーパスに出現したバリアントを受理できる |
| M2 JIT で小コーパス | ベースライン JIT が同じコーパスを実行 | 指標2(a): 同コーパスが JIT でも全パスし、インタプリタとの差分ゼロ |
| M3 std 拡大 | コーパス拡大と std 機能の実行時支援拡充、`panic=unwind` の基礎 | 指標3: チェックリストの主要項目 (fmt/Vec/String/Box/iterator/panic) が実行可能。コーパス追加分も指標2(a) で全パス |
| M4 rustc テストサブセット | rustc 1.100.0 テストスイートの機械抽出サブセットに挑戦、JIT の unwind 完全対応 | 指標2(b): 母数を確定し、パス率を実測値で報告できる状態。未対応は理由付きで分類される |
| M5 rustc 自体 | rustc を構成するクレート群、最終的に rustc バイナリ自体を Sparagmos で実行 | rustc のテストサブセット (指標2(b)) と rustc 自体の起動・小規模コンパイルが Sparagmos 上で再現する |

## リスク・未決定事項

- Stable MIR の公開範囲の制約: 基本ブロックの `is_cleanup`、Drop の
  `replace`、Call の `call_source` などは 1.100.0 の Stable MIR が
  公開しておらず、変換ではデフォルト値になる。意味論に影響する項目が
  実測で見つかれば Issue 化する。
- `InlineAsm` 終端: Stable MIR がオペランドを構造化された形で公開しない
  ため、当面は未対応 (理由付きで報告する)。対応範囲は M2 で決める。
- 定数式 (const eval) の範囲: 定数はアロケーションから復元できる範囲で
  解釈する。実行時に定数評価が必要なケースの扱いは M1 で実測して決める。
- coroutine (`Yield` / `CoroutineDrop` 終端): async を多用する std コードでは
  避けられないが、ベースラインではインタプリタ対応のみとし、JIT 対応は
  M4 で再評価する。
- intrinsics の範囲: `StatementKind::Intrinsic` (NonDivergingIntrinsic) と
  呼び出し形式の intrinsics のうち、どれをランタイムで提供するかは
  コーパス実測に基づいて Issue で列挙する。
