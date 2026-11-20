# Coverage: how "99%" is measured

「Rust 1.100.0 を 99% のレベルで実行できる」を、測れる3指標に分解して定義する。
3指標は **混ぜて1つの数字にしない**。各指標が個別に 99% に達したときに初めて
「99% 実行できる」と言う。

## 前提

- 対象は **Rust 1.100.0**。1.100.0 は 2026-11-12 にリリース済みであり
  (公式ブログ https://blog.rust-lang.org/2026/11/12/Rust-1.100.0/)、
  母数は最初から 1.100.0 で取る。旧前提「当面 1.99.1 を母数とする」は廃止する。
  (1.99.1 は差分確認のためにだけ併記する。)
- 測定ツールチェーン (2026-11-20, fleur-du-lapin 上で実測):
  `rustc 1.100.0 (0ef1bc386 2026-11-09)` / `cargo 1.100.0 (3acc79c0c 2026-11-05)`。
- 数値は実測したものだけを書く。推測の進捗率は書かない。未測定の項目は
  「未測定」と明記する。

---

## 指標1: MIR バリアント網羅率

### 母数の機械的定義

分母は、rust-lang/rust のタグ `1.100.0` における
`compiler/rustc_middle/src/mir/syntax.rs` の3つの enum
(`StatementKind` / `TerminatorKind` / `Rvalue`) のバリアント数の合計とする。

再現手順 (2026-11-20 に実施):

1. `gh api repos/rust-lang/rust/contents/compiler/rustc_middle/src/mir/syntax.rs?ref=1.100.0`
   で raw を取得する (84,331 bytes, blob sha `6bb0190cd70b81fee71348e000e5d133b2f25d46`)。
2. コメントを除去し、各 enum のトップレベルバリアントを機械抽出する。
3. rustc 公式ドキュメント (nightly-rustc の `rustc_middle::mir` 各 enum ページ) の
   記載とも突き合わせ、一致を確認する。

### 母数 (実測結果)

**StatementKind: 12**

| バリアント | 実測コーパス出現 | 備考 |
|---|---|---|
| Assign | ○ | |
| FakeRead | × | 解析フェーズ側。drop elaboration 後は禁止される |
| SetDiscriminant | ○ | |
| StorageLive | ○ | |
| StorageDead | ○ | |
| PlaceMention | × | 実行時は place を評価して捨てる |
| AscribeUserType | × | 解析フェーズ側。drop elaboration 後は禁止される |
| Coverage | × | `-Cinstrument-coverage` 時のみ。実行系は nop 扱い可 |
| Intrinsic | × | NonDivergingIntrinsic 用 (副指標で数える) |
| ConstEvalCounter | × | const eval 用。実行系は nop 扱い可 |
| Nop | × | |
| BackwardIncompatibleDropHint | × | 意味論上は Nop 相当と定義されている |

**TerminatorKind: 15**

| バリアント | 実測コーパス出現 | 備考 |
|---|---|---|
| Goto | ○ | |
| SwitchInt | ○ | |
| UnwindResume | ○ | cleanup ブロック内 |
| UnwindTerminate | × | cleanup / 外部呼び出し境界 |
| Return | ○ | |
| Unreachable | ○ | |
| Drop | ○ | |
| Call | ○ | |
| TailCall | × | 末尾呼び出し形でのみ出現 |
| Assert | ○ | |
| Yield | × | coroutine 本体のみ。lowering 後は現れない |
| CoroutineDrop | × | coroutine の drop glue のみ |
| FalseEdge | × | 解析フェーズ側。drop elaboration 後は禁止される |
| FalseUnwind | × | 解析フェーズ側 |
| InlineAsm | ○ | |

**Rvalue: 13**

| バリアント | 実測コーパス出現 | 備考 |
|---|---|---|
| Use | ○ | |
| Repeat | ○ | `[x; N]` |
| Ref | ○ | |
| ThreadLocalRef | × | `thread_local!` 使用時に出現 |
| RawPtr | ○ | `&raw const/mut` |
| Cast | ○ | CastKind は副指標で数える |
| BinaryOp | ○ | overflow 検査付き演算もここに含まれる (旧 CheckedBinaryOp は独立バリアントではない) |
| UnaryOp | ○ | |
| Discriminant | ○ | |
| Aggregate | ○ | AggregateKind は副指標で数える |
| CopyForDeref | × | runtime MIR では通常コピーに置換されると定義されている |
| WrapUnsafeBinder | × | unsafe binder 使用時 |
| Reborrow | × | reborrow 実験機能の型使用時 |

**指標1の分母 = 12 + 15 + 13 = 40**

99% 到達の意味: 39/40 = 97.5% では不足であり、指標1で 99% を名乗るには
事実上 40/40 が必要である。この非対称性は意図したもので、「ほぼ全部」を
ごまかさないための定義である。

### 副指標1b: サブ enum (混ぜずに個別管理)

以下の enum は指標1の分母には混ぜず、それぞれ独立の分母で実装済み数を数える。
(2026-11-20, 同じ syntax.rs から機械抽出)

| enum | 1.100.0 のバリアント数 | バリアント |
|---|---|---|
| BinOp | 26 | Add, AddUnchecked, AddWithOverflow, Sub, SubUnchecked, SubWithOverflow, Mul, MulUnchecked, MulWithOverflow, Div, Rem, BitXor, BitAnd, BitOr, Shl, ShlUnchecked, Shr, ShrUnchecked, Eq, Lt, Le, Ne, Ge, Gt, Cmp, Offset |
| UnOp | 3 | Not, Neg, PtrMetadata |
| CastKind | 12 | PointerExposeProvenance, PointerWithExposedProvenance, PointerCoercion, IntToInt, FloatToInt, FloatToFloat, IntToFloat, PtrToPtr, FnPtrToPtr, Transmute, BoxDerefTransmute, Subtype |
| AggregateKind | 7 | Array, Tuple, Adt, Closure, Coroutine, CoroutineClosure, RawPtr |
| BorrowKind | 3 | Shared, Fake, Mut |
| Operand | 4 | Copy, Move, Constant, RuntimeChecks |
| ProjectionElem | 9 | Deref, Field, Index, ConstantIndex, Subslice, Downcast, OpaqueCast, UnwrapUnsafeBinder, PhantomDeref |
| NonDivergingIntrinsic | 2 | Assume, CopyNonOverlapping |
| FakeReadCause | 5 | ForMatchGuard, ForMatchedPlace, ForGuardBinding, ForLet, ForIndex |
| RawPtrKind | 3 | Mut, Const, FakeForPtrMetadata |
| UnwindAction | 4 | Continue, Unreachable, Terminate, Cleanup |

参考差分: ProjectionElem は 1.99.1 の 8 から 1.100.0 で 9 になった
(`PhantomDeref` 追加)。3主要 enum (StatementKind/TerminatorKind/Rvalue) と
上記サブ enum は 1.99.1 と 1.100.0 でバリアント集合が一致することも確認した。

### 実測列挙: 小コーパスに何が現れるか (2026-11-20 実測)

手順: 14本の小プログラム (算術、制御フロー、enum/match、構造体と参照、
Vec/String/Box/Rc/Arc、クロージャとジェネリクス、panic と Drop、浮動小数点と
キャスト、unsafe ポインタ、async と dyn Trait、イテレータ、const/static、
never 型 `!`、inline asm) を書き、fleur 上で
`rustup run 1.100.0 rustc --edition 2021 --emit=mir` によりテキスト MIR を
生成 (14ファイル、合計 2,927 行)、出現形式を分類器で集計した。

出現件数 (延べ):

- Statement: Assign 443、SetDiscriminant 2、StorageLive 1、StorageDead 1
  → 出現した種類は **4/12**
- Terminator: Call 146、Return 44、Assert 42、Drop 31、SwitchInt 16、
  Goto 15、UnwindResume 8、Unreachable 6、InlineAsm 1
  → 出現した種類は **9/15**
- Rvalue: Aggregate 101、BinaryOp 92、Ref 92、Use 89、Cast 49、
  UnaryOp 11、Discriminant 6、Repeat 2、RawPtr 1
  → 出現した種類は **9/13**
- BinOp (副指標): 15/26 種が出現 (Eq 30、BitAnd 19、AddWithOverflow 7、
  Sub 7、Ne 6、Lt 5、MulWithOverflow 4、Div 3、Gt 3、SubWithOverflow 2、
  Rem 2、BitXor 1、BitOr 1、Shl 1、Shr 1)。
- Cast のテキスト表記では CastKind::PointerCoercion の下位種別
  (例: Unsize) が括弧内に現れるため、副指標の集計では下位種別も併記する。

注意: 上記の「出現」は実装済みを意味しない。出現 22/40 種は
「小さなコーパスでもこれだけの種類が避けられない」という下限の実測であり、
進捗率ではない。未出現のバリアントが不要という意味でもない
(出現しなかった理由は上の各表の備考を参照)。

### 「実装済み」の判定定義

バリアントは次の3条件をすべて満たしたときにだけ「実装済み」と数える。

1. `sparagmos-driver` がそのバリアントを含むプログラムを `sparagmos-mir` 型へ変換できる。
2. `sparagmos-interp` が意味論テストを通る。
3. `sparagmos-jit` が意味論テストを通る (M1 時点ではインタプリタのみで
   仮数えせず、JIT 完了まで未実装として扱う。厳しい側に倒す)。

指標1の率 = 実装済みバリアント数 / 40 × 100 (小数第1位まで)。
現在の値: **25.0% (実装済み 10/40)** (2026-11-21 実測。バリアント別の内訳は次のマトリクス)。


### バリアント別マトリクス (2026-11-21 実測)

手順: `scripts/variant-matrix.py` で機械集計した
(再現: `python3 scripts/variant-matrix.py`。cargo test を実際に実行し、
通過したテストだけを数える)。

- ドライバ変換: sparagmos-driver の変換コード (convert.rs) に、その
  バリアントを受けて sparagmos-mir の対応バリアントを構築する分岐が
  あるかを走査した。分岐があっても変換エラーが常に記録されるもの
  (InlineAsm、Reborrow) は不可とした。Repeat は繰り返し回数が定数評価
  できない場合にのみ、Cast は CastKind::BoxDerefTransmute の場合にのみ
  エラーになるため条件付きとした。
- interp / JIT の意味論テスト: 両クレートのテストコードを走査し、
  sparagmos-mir のバリアントを構築してエンジンで実行しているテストを
  数えた。ヘルパー関数経由の構築は呼び出し関係をたどって帰属させた。
  JIT のパリティテストは interp と JIT の両方を実行して同じ期待値を
  検証するため、両方の根拠として数えた。
- エラーを期待するテストは原則として数えない。ただし guest panic
  (InterpError::Panic) を検証するテストは意味論テストとして数えた。
  Unsupported やホスト側エラーの検証は数えていない。この区別は対象
  テストを個別に確認し、スクリプト内に根拠付きで記録した。
- interp 内部 IR だけを対象とするテストと、ハーネスのコーパス実行は、
  sparagmos-mir のバリアントを直接構築しないため数えていない。

結果: **実装済み 10/40 = 25.0%**、部分的 5 (すべて interp のみ)、
未対応 25。

**StatementKind (12)**

| バリアント | ドライバ変換 | interp テスト | JIT テスト | 状態 |
|---|---|---|---|---|
| Assign | ○ | ○ 31件 (mir_convert_tests::plain_add_converts_and_runs 他) | ○ 11件 (mir_jit::recursive_fib 他) | 実装済み |
| FakeRead | ○ | × | × | 未対応 |
| SetDiscriminant | ○ | × | × | 未対応 |
| StorageLive | ○ | × | × | 未対応 |
| StorageDead | ○ | × | × | 未対応 |
| PlaceMention | ○ | × | × | 未対応 |
| AscribeUserType | ○ | × | × | 未対応 |
| Coverage | ○ | × | × | 未対応 |
| Intrinsic | ○ | × | × | 未対応 |
| ConstEvalCounter | ○ | × | × | 未対応 |
| Nop | ○ | × | × | 未対応 |
| BackwardIncompatibleDropHint | × (型なし) | × | × | 未対応 |

**TerminatorKind (15)**

| バリアント | ドライバ変換 | interp テスト | JIT テスト | 状態 |
|---|---|---|---|---|
| Goto | ○ | ○ 4件 (mir_convert_tests::loop_sum_to_10 他) | ○ mir_jit::loop_sum | 実装済み |
| SwitchInt | ○ | ○ 8件 (mir_convert_tests::if_else_max 他) | ○ 3件 (mir_jit::switch_on_char 他) | 実装済み |
| UnwindResume | ○ | × | × | 未対応 |
| UnwindTerminate | ○ | × | × | 未対応 |
| Return | ○ | ○ 32件 | ○ 11件 (mir_jit::recursive_fib 他) | 実装済み |
| Unreachable | ○ | × | × | 未対応 |
| Drop | ○ | ○ print_support_tests::string_drop_executes_as_a_jump | × | 部分的 |
| Call | ○ | ○ 7件 (mir_convert_tests::recursive_fib_via_fndef_calls 他) | ○ 2件 (mir_jit::recursive_fib 他) | 実装済み |
| TailCall | × (分岐なし) | × | × | 未対応 |
| Assert | ○ | ○ 11件 (mir_convert_tests::checked_add_overflow_is_a_guest_panic 他) | ○ 6件 (mir_jit::checked_add_ok_and_overflow_panic 他) | 実装済み |
| Yield | × (分岐なし) | × | × | 未対応 |
| CoroutineDrop | × (分岐なし) | × | × | 未対応 |
| FalseEdge | × (分岐なし) | × | × | 未対応 |
| FalseUnwind | × (分岐なし) | × | × | 未対応 |
| InlineAsm | × (変換エラー) | × | × | 未対応 |

**Rvalue (13)**

| バリアント | ドライバ変換 | interp テスト | JIT テスト | 状態 |
|---|---|---|---|---|
| Use | ○ | ○ 24件 | ○ 7件 (mir_jit::loop_sum 他) | 実装済み |
| Repeat | △ 条件付き | ○ mir_convert_tests::tuple_and_array_aggregates_and_repeat | × | 部分的 |
| Ref | ○ | ○ 3件 (print_support_tests::write_through_mut_ref_across_frames 他) | × | 部分的 |
| ThreadLocalRef | ○ | × | × | 未対応 |
| RawPtr | ○ | × | × | 未対応 |
| Cast | △ 条件付き | ○ 3件 (mir_convert_tests::int_to_int_cast_widens_and_truncates 他) | ○ 2件 (mir_jit::casts_and_guarded_division 他) | 実装済み |
| BinaryOp | ○ | ○ 19件 | ○ 7件 (mir_jit::recursive_fib 他) | 実装済み |
| UnaryOp | ○ | ○ mir_jit::negation | ○ mir_jit::negation | 実装済み |
| Discriminant | ○ | ○ print_support_tests::range_inclusive_next_loop_sums | × | 部分的 |
| Aggregate | ○ | ○ 5件 (mir_convert_tests::tuple_and_array_aggregates_and_repeat 他) | × | 部分的 |
| CopyForDeref | ○ | × | × | 未対応 |
| WrapUnsafeBinder | × (型なし) | × | × | 未対応 |
| Reborrow | × (型なし・変換エラー) | × | × | 未対応 |

母数外の sparagmos-mir 独自バリアント (StatementKind::Retag、
Rvalue::Len / CheckedBinaryOp / NullaryOp / ShallowInitBox) は
このマトリクスに含めない。CheckedBinaryOp を使うテストは、
同時に構築される Assign / Assert 等の根拠として現れる。

---

## 指標2: テストパス率

### 母数 (2層)

- **指標2(a) 自前コーパス**: `tests/corpus/` に置かれたプログラムの総数。
  追加・削除は Issue で管理し、母数の変更履歴を残す。
- **指標2(b) rustc テストサブセット**: rust-lang/rust タグ `1.100.0` の
  テストスイート全体から機械抽出したテストの総数。母数は実行形態で
  絞らない。具体的には次をすべて母数に含め、カテゴリ別の内訳を併記する。
  - 単一の `.rs` ファイルで完結するテスト
  - 複数ファイル構成のテスト (補助ファイル、`auxiliary` クレート)
  - `build.rs` や run-make のようなビルド手順を伴うテスト
  - FFI を含むテスト (`extern "C"`、C 側コードとのリンク)
  実行環境に依存して除外するもの (外部サービス・ネットワーク・
  専用ハードウェアを必要とするもの) は、カテゴリ単位ではなく個別に
  理由を列挙して除外する。99% の目標はカテゴリ別ではなくサブセット
  全体に対して適用する。未対応で実行できないテストは fail として数え、
  母数から外さない。
  母数の実数値は、下記「指標2(b) の母数実測」のとおり **24,937 件** で
  確定した。

### 指標2(b) の母数実測 (2026-11-20)

出典: rust-lang/rust タグ `1.100.0` (commit
`0ef1bc386059596dada8fb65b50425aab2ea637a`) の `tests/` ツリー。
抽出は `scripts/rustc-subset-survey.py` で行った。同じチェックアウトに
対して再実行すると同一の集計が得られることを確認済みである。再実行は
`python3 scripts/rustc-subset-survey.py --rust-dir <チェックアウト>`
(チェックアウトの取得手順はスクリプト冒頭に記載)。

数え方は compiletest 自身の発見規則に合わせた。ファイル系スイートは
`.rs` ファイル1件を1テストとし、`auxiliary` ディレクトリ配下と
`compiletest-ignore-dir` のあるディレクトリ配下は数えない。run-make 系は
`rmake.rs` を含むディレクトリ1件を1テストとする。rustdoc-gui は
`.goml`、rustdoc-js-std は `.js` 1件を1テストとする。スイートの範囲は
bootstrap のステップ定義 (`src/bootstrap/src/core/build_steps/test.rs`)
にある22スイートで、`tests/auxiliary` はスイートではないため含めない。
coverage スイートは coverage-map / coverage-run の2モードで同じ
ファイルを実行するため、ファイル単位で1件と数えた。`//@ revisions:`
で複数リビジョンに展開されるテストをリビジョン単位で数えた参考値は
27,674 件である。

抽出総数 24,940 件から、環境依存の3件を個別に除外した **24,937 件** が
母数である。

スイート別内訳 (母数、除外後):

| スイート | 件数 |
|---|---|
| ui | 20,327 |
| codegen-llvm | 843 |
| rustdoc-html | 802 |
| run-make | 538 |
| rustdoc-ui | 440 |
| mir-opt | 403 |
| crashes | 209 |
| rustdoc-json | 194 |
| incremental | 180 |
| debuginfo | 170 |
| assembly-llvm | 169 |
| rustdoc-gui | 152 |
| pretty | 114 |
| coverage | 108 |
| ui-fulldeps | 80 |
| rustdoc-js | 79 |
| rustdoc-js-std | 68 |
| codegen-units | 46 |
| run-make-cargo | 12 |
| assembly-gcc | 1 |
| build-std | 1 |
| coverage-run-rustdoc | 1 |
| 合計 | 24,937 |

assembly-gcc は実験的な GCC バックエンド向けのスイートで、既定の
LLVM バックエンドでは compiletest 自身が実行対象から外す。タグ時点の
スイート全体を母数とするため、ここでは含めている。

合否の決まり方による分類 (母数):

| 分類 | 件数 | 内容 |
|---|---|---|
| 実行系 (exec) | 4,149 | ui / ui-fulldeps の run-pass・run-fail・run-crash (4,040) と coverage 系 (109)。対象プログラムを実行し、その挙動が合否に関与する |
| 実行系 (procedure) | 551 | run-make 系。レシピ (`rmake.rs`) を実行し、ビルドと実行の手順全体で合否が決まる |
| 実行系 (exec-debugger) | 170 | debuginfo。デバッガ下でプログラムを実行し、デバッガ出力を照合する |
| コンパイル時検証系 | 18,332 | 診断メッセージ、IR・アセンブリ・MIR ダンプ、整形結果など、コンパイル時出力の照合で合否が決まる (mir-opt、codegen、assembly、pretty、incremental、crashes、ui の診断系など) |
| rustdoc 検証系 | 1,735 | rustdoc の出力 (HTML、JSON、検索インデックス、GUI、UI 診断) の照合で合否が決まる |

分類の線引き: パスの定義は、ネイティブ実行と Sparagmos 実行の
標準出力・標準エラー・終了コードの一致を要求する。この比較がそのまま
適用できるのは、合否に対象プログラムの実行が関与する実行系
(合計 4,870 件) だけである。コンパイル時検証系と rustdoc 検証系は
プログラムを実行せず、合否の材料はコンパイル時出力・rustdoc 出力に
なる。それでも「未対応で実行できないテストは母数から外さない」という
定義に従い、両分類とも母数に含め、分類だけを分けて併記する。
プラットフォーム限定 (`only-*` / `ignore-*`) と、コンパイラの
ビルド構成に依存する `needs-*` は、環境依存とはみなさず母数に含める。
このタグのスイートには、ネットワークを必要とするテストを表す指令
(`needs-network` 等) は存在せず、run-make レシピにも外部サービスへの
アクセスを行うものは無かった (下記の除外3件を除く)。

形態別の内訳 (母数):

- 単一ファイルで完結するもの: 23,047 件。補助ファイルや `auxiliary`
  クレートを要するもの: 1,890 件。
- ビルド手順を伴うもの (run-make 系): 551 件。
- FFI を含むもの: 1,147 件 (実行系 332、コンパイル時検証系 765、
  rustdoc 検証系 50)。`extern "C"` 等の宣言・使用、C ソースの同梱を
  機械的に検出した。

環境依存による除外 (3件、いずれも run-make-cargo):

| テスト | 除外理由 |
|---|---|
| `tests/run-make-cargo/thumb-none-qemu` | QEMU によるシステムエミュレーションとファームウェアが必要。依存クレート (cortex-m、cortex-m-rt、cortex-m-semihosting、panic-halt) の取得 (ネットワーク) が必要 |
| `tests/run-make-cargo/uefi-qemu` | QEMU と UEFI ファームウェアが必要。依存クレート `r-efi` の取得 (ネットワーク) が必要 |
| `tests/run-make-cargo/x86_64-fortanix-unknown-sgx-lvi` | Intel SGX (`x86_64-fortanix-unknown-sgx`) のターゲット基盤が必要。ビルド依存クレート `cc` / `cmake` の取得 (ネットワーク) が必要 |

分類できなかったテストは 0 件である (発見した全テストに分類が付いた)。

### パスの定義

テストは次をすべて満たすとき pass とする。

- ネイティブ実行 (ホスト rustc 1.100.0 でコンパイルして実行) と
  Sparagmos 実行で、標準出力・標準エラー・終了コードが一致する。
- タイムアウト、クラッシュ、不一致はいずれも fail とする。
  「未対応のため実行できない」も fail であり、母数から除外しない。

指標2の率 = pass 数 / 母数 × 100 (層ごとに別々に報告し、混ぜない)。

### 指標2(b) のパス率実測 (2026-11-21)

`scripts/rustc-subset-run.py` による第1回実測。入力は母数実測と
同じ survey JSON で、実行したのは実行系 (exec) カテゴリのうち
単一ファイルで完結する ui / ui-fulldeps の run-pass / run-fail
テスト 3,699 件である。それ以外の母数内テストは未実行として
fail に数え、母数から外していない。判定はハーネスのコーパス
比較と同じ基準で、ネイティブ実行と Sparagmos 実行の標準出力と
終了コードの一致を要求し、run-fail では期待終了コード
(既定 101、`failure-status` 指定時はその値) との一致も要求する。
panic の標準エラーはネイティブ側がテストファイルのパスと
行番号を含むため、バイト単位の一致は判定に用いていない
(両側の標準エラーは結果 JSON に記録される)。

結果 (stage 1 初回実測、2026-11-21。最新の統合実測は下記のカテゴリ表と stage 2 の節に記載):

| 状態 | 件数 | 内容 |
|---|---|---|
| pass | 539 | 両側を実行し、標準出力・終了コードが一致 |
| fail | 18 | 両側を実行したが不一致 (終了コード 16、標準出力 2) |
| unsupported | 2,266 | ドライバが未対応を返した (終了コード 2) |
| 未実行 | 22,114 | 下記の内訳のとおり。いずれも fail として数える |

未実行の内訳: stage 1 の範囲外 (実行系以外の分類) 20,788 件、
補助ファイルを要する実行系 450 件、ヘッダ指令が stage 1 の
対象外 (compile-flags 332、revisions 133、run-crash 46 など)
530 件、適用可否を判定できない条件 (needs-subprocess 138、
needs-threads 80 など) 229 件、プラットフォーム限定
(ignore-* / only-*) 20 件、ネイティブ側のコンパイル失敗 91 件、
ネイティブ実行が期待終了コードと不一致 6 件。

カテゴリ別のパス率:

| カテゴリ | pass / 母数 | パス率 |
|---|---|---|
| 実行系 (exec) | 870 / 4,149 | 20.969% |
| 実行系 (procedure) | 0 / 551 | 0% (未実行) |
| 実行系 (exec-debugger) | 0 / 170 | 0% (未実行) |
| コンパイル時検証系 | 11,322 / 18,332 | 61.761% (成否レベルの一致。照合系の扱いは stage 2 の節に記載) |
| rustdoc 検証系 | 0 / 1,735 | 0% (分類・実測済み: 合否は rustdoc の出力照合で決まるため主経路ではパスしない) |
| **全体** | **12,192 / 24,937** | **48.891%** |

候補 3,699 件のうち両側を実行できたのは 2,823 件で、その内訳は
pass 539、fail 18、unsupported 2,266 である。unsupported の
理由の上位は、by-reference 定数 860 件、ローカルの型が未対応
530 件、変換済み本体の無い関数への呼び出し 445 件、
BoxDerefTransmute キャストを含む関数の未変換 216 件、
その他の変換ギャップ 154 件、interp 変換時の型不一致 19 件、
Drop ターミネータ 17 件である。fail 18 件は末尾呼び出し
(`become`)、ジェネレータ、async クロージャなどでインタプリタが
panic したケースである。

測定条件: 48 並列、候補 3,699 件の処理に 20.0 秒。タイムアウトは
ネイティブのコンパイル 60 秒、ネイティブの実行 30 秒、ドライバ
実行 60 秒 (1件あたり)。テストごとの結果は
`subset-run-results.json` (測定クローンに保存) に機械可読な形で
残る。再実行は
`python3 scripts/rustc-subset-run.py --survey-json <survey JSON> --rust-dir <チェックアウト> --driver-bin <ドライバ> --out <結果 JSON>`。

#### stage 2: コンパイル時検証系の実測 (2026-11-21)

同じランナーの stage 2 で、コンパイル時検証系 18,332 件を実測した。
各テストをドライバの変換専用モード (実行しない) で処理し、結果を
compiletest 自身が指令から決める期待と突き合わせる。期待の決め方は
compiletest の実装に倣い、ui / ui-fulldeps は `check-pass` /
`build-pass` で成功期待、`check-fail` / `build-fail` で失敗期待、
指令が無ければ失敗期待 (check-fail が既定)、crashes はコンパイラの
クラッシュが期待、その他のスイート (mir-opt、codegen-llvm、
codegen-units、assembly、pretty) は成功期待とした。

合否は成否レベルで判定する。成功期待は、ドライバがコンパイルに
成功し、かつ全てのローカル関数を変換できた (skipped = 0、完全な
MirProgram が得られた) ときに pass とする。変換が一部でも欠ければ
unsupported とする。失敗期待は、ドライバがコンパイル失敗を報告
したときに pass とする。crashes は、ドライバの終了コードが 0 でも
1 でもない (ICE やシグナルで落ちた) ときに pass とする。

コンパイラの出力そのものの照合は行っていない。compiletest の
合否には、診断出力と `.stderr` スナップショットの照合、`//~`
エラー注釈の照合、rustfix の `.fixed` 照合、mir-opt の `.mir`
ダンプ照合、codegen 系の FileCheck 照合が含まれる。診断はドライバが
内部で使う参照 rustc 自身が生成するため、照合の成否は出力の
正規化方法に左右され、Sparagmos 側の変換の完全性は上記の成否レベル
判定で測れる。照合を要する件数は結果 JSON に記録しており、
コンパイル時検証系 18,332 件のうち `.stderr` スナップショットを
持つもの 12,488 件、`//~` 注釈を持つもの 12,064 件、`.fixed` を
持つもの 709 件、`.mir` を持つもの 137 件、FileCheck 行を持つもの
1,220 件で、いずれかを持つものは 13,887 件である。stage 2 の
pass 11,322 件のうち 10,035 件はいずれかの照合を本来要する
(うち `.stderr` は 9,922 件)。stage 2 の pass は成否レベルの
一致であり、compiletest の完全な合否とは異なる。

結果 (コンパイル時検証系 18,332 件):

| 状態 | 件数 | 内容 |
|---|---|---|
| pass | 11,322 | 期待と成否が一致 (下記の内訳) |
| unsupported | 1,543 | コンパイルは成功したが変換が不完全 |
| fail | 314 | 期待と成否が不一致、またはドライバの異常終了 |
| 未実行 | 5,153 | 下記の内訳のとおり。いずれも fail として数える |

期待別の内訳 (実行した 13,179 件): 失敗期待 9,874 件は pass 9,750 /
fail 124、成功期待 3,156 件は pass 1,423 / unsupported 1,564 /
fail 169、クラッシュ期待 149 件は pass 135 / fail 14 である。

未実行 5,153 件の内訳: ヘッダ指令が対象外 4,068 件 (compile-flags
2,616、revisions 1,438、add-minicore 6、rustc-env 7、ignore-test 1)、
補助ファイルを要するもの 801 件、incremental スイート 158 件
(セッションをまたぐ再コンパイルが合否を決めるため、1回の変換では
判定できない)、適用可否を判定できない条件 87 件、プラット
フォーム限定 39 件。

unsupported 1,564 件の理由の上位は、単相化されたインスタンスが
無い 1,198 件、MIR 本体が得られない 265 件、BoxDerefTransmute
キャスト 47 件、割当て済み定数 (ConstantKind::Allocated) 41 件、
InlineAsm 5 件である。lib としてコンパイルする ui テストでは、
呼び出されないジェネリック関数に単相化インスタンスが無く、
変換が欠ける形が最も多い。

fail 307 件の内訳は、成功期待だがコンパイル失敗 139 件、失敗期待
だが変換成功 116 件、成功期待でのドライバのクラッシュ 29 件、
失敗期待でのコンパイラのクラッシュ 8 件、crashes の再現失敗
14 件、タイムアウト 1 件である。成否が食い違う主因は、ドライバの
固定したコンパイル形態 (crate-type は lib、出力はメタデータ) と
compiletest の形態の差で、エントリポイント検査やコード生成段での
み起きるエラー、補助クレートへの参照 (E0463)、pretty スイートの
断片的なファイル (整形だけが目的で型検査を通らないもの) が
サンプルで確認できた。

stage 1 と合わせたカテゴリ別のパス率:

| カテゴリ | pass / 母数 | パス率 |
|---|---|---|
| 実行系 (exec) | 539 / 4,149 | 12.991% |
| 実行系 (procedure) | 0 / 551 | 0% (未実行) |
| 実行系 (exec-debugger) | 0 / 170 | 0% (未実行) |
| コンパイル時検証系 | 11,322 / 18,332 | 61.761% |
| rustdoc 検証系 | 0 / 1,735 | 0% (分類・実測済み: 合否は rustdoc の出力照合で決まるため主経路ではパスしない) |
| **全体** | **12,192 / 24,937** | **48.891%** |

測定条件: 48 並列、stage 1 と stage 2 の候補合計 22,031 件の
処理に 94.4 秒。タイムアウトはドライバ変換 60 秒 (1件あたり)。
結果は `subset-run-stage2-results.json` (測定クローンに保存) に
テストごとの期待・状態・理由と、照合を要する検証の有無を記録した。
再実行は
`python3 scripts/rustc-subset-run.py --survey-json <survey JSON> --rust-dir <チェックアウト> --driver-bin <ドライバ> --out <結果 JSON>`
(`--stages 1` で stage 1 のみ、`--stages 2` で stage 2 のみ)。
stage 1 のみの再実行では前回の結果 (pass 539 件、実行系 12.991%)
がテスト単位で完全に再現することを確認済みである。

---

### rustdoc 検証系の内訳と成否一致の実測 (2026-11-21)

`scripts/rustc-subset-rustdoc.py` による分類と実測。入力は母数実測と
同じ survey JSON で、対象は rustdoc 検証系の全 1,735 件である。

このカテゴリの合否は、すべて rustdoc の出力の照合で決まる。
スイート別の内訳は次のとおりである。

| スイート | 件数 | 合否の材料 |
|---|---|---|
| rustdoc-html | 802 | 生成された HTML の内容 (has / matches / count / snapshot 等の指令で照合するもの 782 件、補助クレートのドキュメント生成が対象のもの 20 件) |
| rustdoc-ui | 440 | rustdoc の診断・出力ストリーム (合否形式の内訳は下記) |
| rustdoc-json | 194 | 生成された JSON の内容 (全件が is / set / has / count / jq 等の指令で照合) |
| rustdoc-gui | 152 | 生成された HTML をブラウザで操作する GUI スクリプト (.goml) |
| rustdoc-js | 79 | rustdoc の検索インデックスを JavaScript から検証 |
| rustdoc-js-std | 68 | std の検索インデックスを JavaScript から検証 |

rustdoc-ui の合否形式の内訳 (440 件): 診断スナップショット
(.stderr 照合または error-pattern) 191 件、終了ステータスのみ
(出力スナップショットを持たない) 129 件、doctest の実行レポート
(compile-flags に `--test`) 98 件、カバレッジ出力
(`--show-coverage`) 19 件、標準出力スナップショット 3 件。

主経路 (ドライバ → Stable MIR → インタプリタ/JIT) は rustdoc の
出力を生成しない。ドライバは rustc ドライバであり、ドキュメント
生成、intra-doc リンク解決、doctest の抽出と実行、検索インデックス
生成といった rustdoc の処理は実行されない。したがってこの
カテゴリで合否そのものを再現できる対象はなく、パス数は 0 件の
ままである。コンパイルできることはテストのパスとは別であり、
下記の成否一致はパス率ではない。

#### rustdoc-ui のコンパイル成否の一致 (参考実測)

rustdoc-ui は、入力が単一の .rs ファイルで、期待される結果に
コンパイルの成否が含まれる唯一のスイートである。この成否だけを
取り出し、ドライバの変換モード (主経路のフロントエンドによる
コンパイル) の成否と比較した。期待成否は compiletest の規則
どおり、`*-pass` 指令があれば成功、なければ失敗 (UI モードの
既定は check-fail) とした。対象は stage 1 と同じ条件で選んだ
228 件で、対象外は compile-flags あり 167 件、補助ファイルが
必要 21 件、revisions 17 件、run-rustfix 7 件である。

結果: 一致 152 件、不一致 76 件、ドライバエラー 0 件
(一致率 66.7%)。期待成功の 103 件では一致 86 件 (83.5%)、
期待失敗の 125 件では一致 66 件 (52.8%) であった。

不一致の内訳は、成否自体が rustc と rustdoc で異なることを
示している。

- 期待失敗なのに rustc ではコンパイルできる 59 件: intra-doc
  リンク解決の診断 40 件、rustdoc リント (lints 配下) 15 件、
  doctest 配下 3 件 (doc コメントに対する rustdoc リント)、
  提案 (suggestions) 1 件。失敗はいずれも rustdoc 固有の処理が
  出す診断で生じており、rustc のコンパイルでは生じない。
- 期待成功なのに rustc ではコンパイルできない 17 件:
  error-in-impl-trait 配下の 8 件を含む。関数本体の中の未解決名
  など、ドキュメント生成に不要な範囲のエラーは rustdoc が
  診断しないことを、参照 rustdoc 1.100.0 で確認した (本体内の
  未解決名は無診断で成功し、シグネチャ内の未解決型はエラーに
  なる)。未定義クレートへの型エイリアス、自動トレイトの合成
  実装 (E0782) などもここに含まれる。

成否が 3 分の 1 の割合で一致しない以上、コンパイル成否は
rustdoc-ui の合否の近似にならない。加えて、一致した 152 件でも
合否に必要な診断テキストの照合は行われていない。このカテゴリの
パス率を上げるには rustdoc 相当の出力生成が主経路に必要であり、
現状の主経路の範囲外である。

測定条件: 48 並列、対象 228 件のドライバ実行に 1.2 秒
(分類は全 1,735 件)。タイムアウトはドライバ実行 60 秒
(1件あたり)。テストごとの結果は `subset-rustdoc-results.json`
(測定クローンに保存) に機械可読な形で残る。再実行は
`python3 scripts/rustc-subset-rustdoc.py --survey-json <survey JSON> --rust-dir <チェックアウト> --driver-bin <ドライバ> --out <結果 JSON>`。

---

## 指標3: std 機能網羅率

### 母数の定義

「std の全 API」を分母にはしない。分母は、指標2の対象プログラムのうち
変換できたものの MIR に、呼び出し先として実際に現れる関数パスの集合
(機械抽出してユニーク化したもの) とする。つまり「必要とされた std 機能」
に対する網羅率である。

- 対象プログラム: 自前コーパス 15 件と、指標2(b) の stage 1 候補
  3,699 件 (実行系・単一ファイル・ui / ui-fulldeps)。
- 出現の単位: ドライバが変換した各本体の中で、Call / TailCall の
  func オペランドが指す関数パス。間接呼び出し (関数ポインタや
  クロージャ経由) はパスを持たないため数えない。
- std 関数パス: パスの先頭セグメントが `std` / `core` / `alloc` の
  いずれかであるもの。
- 名前の同一性: interp の builtin 照合と同じ正規化 (寿命だけの
  ジェネリック引数グループを取り除く) を施した文字列で判定する。
  型引数は保持されるため、`Vec::<T>::new` のような定義側の表記が
  そのまま集合の要素になる。
- 実行可能の判定: そのパスへの呼び出しが interp で実行できること、
  すなわち (a) 変換済み本体がプログラム内に得られる、または
  (b) interp の builtin として実装されている
  (`crates/sparagmos-interp/src/builtins.rs` の `lookup`、ガード規則
  2 件を含む) のいずれかを満たすこと。集合の要素については、
  本体が得られるプログラムが 1 件でもあれば実行可能と数える。

### 実測 (2026-11-21)

抽出は `scripts/std-coverage.py` で行う。ドライバを dump モードで実行し、
def-name 表と変換済み本体の呼び出し先を機械抽出する。再実行は
`python3 scripts/std-coverage.py --results-json <subset-run-results.json> --rust-dir <チェックアウト> --driver-bin <ドライバ> --out <結果 JSON>`。
edition は各ファイルの `//@ edition:` 指令 (既定 2021) を使う。
測定条件: 48 並列、ドライバ 1 件あたりのタイムアウト 60 秒。

入力 3,714 件のうち、dump が得られたのは 3,532 件である。
フロントエンドで変換できない 181 件と、タイムアウト 1 件
(`tests/ui/mir/mir_heavy_promoted.rs`) は出現を数えられない。
dump が得られても変換済み関数が 0 件だったものが 157 件ある。
出現を数えられたのは、変換済み関数を 1 件以上持つ 3,375 件であり、
母数はこの範囲での出現集合である (変換できないファイルにだけ現れる
パスは含まれない)。

- 母数: **906** (std 674、core 231、alloc 1)。出現はプログラム×パスで
  9,528 件、呼び出し箇所では 35,607 箇所。
- 実行可能: **18 / 906 = 1.987%**。18 件は全て builtin による解決で、
  変換済み本体による解決は 0 件だった。ドライバは対象プログラムの
  自クレートの関数だけを変換し、std 関数の本体は MirProgram に
  含まれないため、現状では std の呼び出しは builtin でしか
  実行できない。

未対応の頻出パス (出現プログラム数順、上位 15 件):

| パス | 出現プログラム数 |
|---|---|
| core::panicking::assert_failed | 1,294 |
| std::cmp::PartialEq::eq | 568 |
| core::panicking::panic | 431 |
| std::boxed::Box::<T>::new | 175 |
| std::result::Result::<T, E>::unwrap | 168 |
| std::rt::panic_fmt | 158 |
| std::string::ToString::to_string | 157 |
| std::ops::Index::index | 105 |
| std::vec::Vec::<T>::new | 82 |
| std::option::Option::<T>::unwrap | 81 |
| std::ops::Fn::call | 80 |
| std::thread::spawn | 77 |
| std::thread::JoinHandle::<T>::join | 76 |
| std::fmt::Formatter::write_str | 74 |
| std::clone::Clone::clone | 67 |

### チェックリストの種 (出現ベースで増やす)

- `core::fmt` (println! と書式化、core::fmt::rt::Argument 経由)
- コレクションと所有型: Vec、String、Box、Rc、Arc
- Iterator トレイトとアダプタ
- panic 機構 (panic ハンドラ、catch_unwind、Assert 由来の panic)
- stdout への出力 (std::io::_print)

### Rust 1.100.0 の新機能・変更点の扱い

1.100.0 (2026-11-12) の主な変更のうち、実行対象として関係するもの:

- **never 型 `!` の安定化**: 任意の位置に `!` を書けるようになり、
  `Infallible` は `!` の別名になった。MIR では発散関数の終端
  (Unreachable、戻らない Call) と never-to-any の coercion として現れる。
  実測コーパス (never 型のプログラム) でも Discriminant/SwitchInt/
  Unreachable 等の形で出現を確認した。指標1の範囲内で扱い、専用の
  バリアントは存在しない。
- **Allocator API のコア安定化**: `Allocator` トレイトと `Vec`/`Box` の
  `_in` 系コンストラクタ。指標3の対象とし、カスタムアロケータ対応は
  ランタイムの確保フックで受ける (DESIGN.md 参照)。
- **`core::mem::DropGuard`**: Drop 時にクロージャを実行するヘルパ。
  MIR 上は通常の ADT + Drop として現れる前提で、指標2/3の通常経路で扱う。
- **`core::mem::conjure_zst`**: ZST の値を安全に生成する関数。
  ランタイム/intrinsic 側の対応項目として指標3に含める。
- **Cargo の unused dependencies 報告 / min-publish-age**: これらは Cargo の
  機能であり、Sparagmos の実行対象そのものではない。ただし M5 で rustc と
  共に Cargo を動かす段階では前提知識になるため、ここに記録だけする
  (指標の母数には入れない)。

---

## 測定の規律

- 測定は fleur 上で、ツールチェーンとコマンドを併記して記録する。
- harness による自動集計が整うまでは、手動測定の結果を日付付きで本書に追記する。
- 指標をまたいだ合算値 (例: 3指標の平均) を公表しない。
- 母数を変更したときは、変更理由と旧母数を本書の履歴に残す。

## 現在の測定状態 (2026-11-21)

| 指標 | 母数 | 現在値 |
|---|---|---|
| 指標1 MIR バリアント網羅率 | 40 (確定) | 25.0% (10/40, 2026-11-21 実測) |
| 副指標1b サブ enum | 上表の通り (確定) | 未測定 (BinOp の出現実測 15/26 のみ) |
| 指標2(a) 自前コーパス | 36 (tests/corpus) | 100.0% (36/36, 2026-11-21 実測) |
| 指標2(b) rustc テストサブセット | 24,937 (2026-11-20 機械抽出) | 48.891% (12,192/24,937, 2026-11-21 実測。コンパイル時検証系は成否レベルの一致で、照合系の扱いは stage 2 の節に記載) |
| 指標3 std 機能網羅率 | 906 (2026-11-21 機械抽出) | 1.987% (18/906, 2026-11-21 実測) |

指標2(a) の母数変更履歴: 2026-11-21 に 15 件から 36 件へ拡大した
(Issue #28)。追加した 21 件は、制御フロー (ラベル付きループ、match
ガード、相互再帰、早期リターン、while let)、データ (タプル、配列の
諸形式、構造体、enum ペイロード、Option、&mut 参照)、演算子
(ビット演算、キャスト、複合代入、符号付き除算・剰余) の観点で、
いずれも interp 経路で native と stdout・終了コードが一致する
ことを確認済みのプログラムである。

