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
現在の値: **67.5% (実装済み 27/40)** (2026-11-21 実測。バリアント別の内訳は次のマトリクス)。


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

結果: **実装済み 27/40 = 67.5%**、部分的 0、
未対応 13。

**StatementKind (12)**

| バリアント | ドライバ変換 | interp テスト | JIT テスト | 状態 |
|---|---|---|---|---|
| Assign | ○ | ○ 37件 (mir_convert_tests::checked_add_matches_rustc_shape 他) | ○ 15件 (mir_jit::casts_and_guarded_division 他) | 実装済み |
| FakeRead | ○ | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | 実装済み |
| SetDiscriminant | ○ | ○ mir_jit::set_discriminant_then_read 他 | ○ mir_jit::set_discriminant_then_read 他 | 実装済み |
| StorageLive | ○ | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | 実装済み |
| StorageDead | ○ | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | 実装済み |
| PlaceMention | ○ | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | 実装済み |
| AscribeUserType | ○ | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | 実装済み |
| Coverage | ○ | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | 実装済み |
| Intrinsic | ○ | × | × | 未対応 |
| ConstEvalCounter | ○ | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | 実装済み |
| Nop | ○ | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | ○ mir_jit::storage_and_annotation_statements_do_not_affect_execution | 実装済み |
| BackwardIncompatibleDropHint | × (型なし) | × | × | 未対応 |

**TerminatorKind (15)**

| バリアント | ドライバ変換 | interp テスト | JIT テスト | 状態 |
|---|---|---|---|---|
| Goto | ○ | ○ 4件 (mir_convert_tests::loop_sum_to_10 他) | ○ mir_jit::loop_sum | 実装済み |
| SwitchInt | ○ | ○ 9件 (mir_convert_tests::if_else_max 他) | ○ 4件 (mir_jit::loop_sum 他) | 実装済み |
| UnwindResume | ○ | × | × | 未対応 |
| UnwindTerminate | ○ | × | × | 未対応 |
| Return | ○ | ○ 38件 (mir_convert_tests::checked_add_matches_rustc_shape 他) | ○ 15件 (mir_jit::casts_and_guarded_division 他) | 実装済み |
| Unreachable | ○ | ○ 2件 (mir_convert_tests::unreachable_terminator_is_a_guest_panic 他) | ○ mir_jit::unreachable_block_not_taken_leaves_execution_intact | 実装済み |
| Drop | ○ | ○ 2件 (mir_jit::drop_of_scalar_jumps_to_target 他) | ○ mir_jit::drop_of_scalar_jumps_to_target | 実装済み |
| Call | ○ | ○ 7件 (mir_convert_tests::recursive_fib_via_fndef_calls 他) | ○ 2件 (mir_jit::checked_add_ok_and_overflow_panic 他) | 実装済み |
| TailCall | × (分岐なし) | × | × | 未対応 |
| Assert | ○ | ○ 11件 (mir_convert_tests::checked_add_matches_rustc_shape 他) | ○ 6件 (mir_jit::casts_and_guarded_division 他) | 実装済み |
| Yield | × (分岐なし) | × | × | 未対応 |
| CoroutineDrop | × (分岐なし) | × | × | 未対応 |
| FalseEdge | × (分岐なし) | × | × | 未対応 |
| FalseUnwind | × (分岐なし) | × | × | 未対応 |
| InlineAsm | × (変換エラー) | × | × | 未対応 |

注: Unreachable は、到達しないブロックが存在しても実行が正しく進むことを両経路で検証している。到達した場合は両経路とも guest panic ("entered unreachable code") を返す (JIT は Lane AY で panic 機構に統一され、ホストプロセスを落とさない)。

**Rvalue (13)**

| バリアント | ドライバ変換 | interp テスト | JIT テスト | 状態 |
|---|---|---|---|---|
| Use | ○ | ○ 29件 (mir_convert_tests::checked_add_matches_rustc_shape 他) | ○ 10件 (mir_jit::checked_add_ok_and_overflow_panic 他) | 実装済み |
| Repeat | △ 条件付き | ○ mir_convert_tests::tuple_and_array_aggregates_and_repeat 他 | ○ mir_jit::repeat_const_elements_sum 他 | 実装済み |
| Ref | ○ | ○ 4件 (mir_convert_tests::copy_for_deref_reads_through_reference_and_plain_place 他) | ○ 6件 (mir_jit::mut_ref_read_modify_write_reaches_the_place 他) | 実装済み |
| ThreadLocalRef | ○ | × | × | 未対応 |
| RawPtr | ○ | ○ mir_convert_tests::raw_ptr_creation_reads_and_writes_through | ○ mir_jit::raw_ptr_creation_reads_and_writes_through 他 | 実装済み |
| Cast | △ 条件付き | ○ 3件 (mir_convert_tests::int_to_int_cast_widens_and_truncates 他) | ○ 2件 (mir_jit::casts_and_guarded_division 他) | 実装済み |
| BinaryOp | ○ | ○ 23件 (mir_convert_tests::if_else_max 他) | ○ 11件 (mir_jit::casts_and_guarded_division 他) | 実装済み |
| UnaryOp | ○ | ○ mir_jit::negation | ○ mir_jit::negation | 実装済み |
| Discriminant | ○ | ○ print_support_tests::range_inclusive_next_loop_sums 他 | ○ mir_jit::discriminant_of_aggregate_and_constant 他 | 実装済み |
| Aggregate | ○ | ○ 6件 (mir_convert_tests::tuple_and_array_aggregates_and_repeat 他) | ○ mir_jit::tuple_aggregate_fields_read_back | 実装済み |
| CopyForDeref | ○ | ○ mir_convert_tests::copy_for_deref_reads_through_reference_and_plain_place | ○ mir_jit::copy_for_deref_reads_the_pointee | 実装済み |
| WrapUnsafeBinder | × (型なし) | × | × | 未対応 |
| Reborrow | × (型なし) | × | × | 未対応 |

母数外の sparagmos-mir 独自バリアント (StatementKind::Retag、
Rvalue::Len / CheckedBinaryOp / NullaryOp / ShallowInitBox) は
このマトリクスに含めない。CheckedBinaryOp を使うテストは、
同時に構築される Assign / Assert 等の根拠として現れる。

### coverage テーブルとの対応

`crates/sparagmos-mir/src/coverage.rs` には2種類の表がある。

- **指標1の母数表** (`METRIC1_STATEMENT_KINDS` / `METRIC1_TERMINATOR_KINDS` /
  `METRIC1_RVALUE_KINDS`): 上の母数表と同一の40種。網羅率の分母はこの表だけが
  担う (`metric1_variant_count() == 40`)。
- **モデル表** (`model_tables()` が返す15表): `sparagmos-mir` の型モデル自身が
  定義するバリアントの目録で、副指標1b の集計と、各モジュールのサンプル
  テスト (全バリアントを構築して `kind_name()` を照合する) のためにある。
  分母ではない (`model_variant_total()` はモデルの総数であって網羅率の
  分母ではない)。

モデルは rustc MIR の方言なので、モデル表と母数表は一致しない。主な相違は
次のとおり。モデルには `StatementKind::Retag` があり、母数の
`BackwardIncompatibleDropHint` が無い。unwind 系ターミネータはモデルでは
`Resume` / `Terminate`、母数では `UnwindResume` / `UnwindTerminate`。
`Rvalue` はモデルでは `Len` / `CheckedBinaryOp` / `NullaryOp` /
`ShallowInitBox` を独立バリアントとして持つ (15種) が、母数では他の
バリアントに統合され、代わりに `WrapUnsafeBinder` と `Reborrow` がある
(13種)。

モデル表の件数は副指標1b の分母と一致するとは限らない (例: CastKind は
1b の分母が12種に対しモデルは11種で `BoxDerefTransmute` を欠く)。モデルが
持たない enum (BorrowKind、FakeReadCause 等) については、モデル表自体が
無い。

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
| 実行系 (exec) | 2,214 / 4,149 | 53.362% |
| 実行系 (procedure) | 0 / 551 | 0% (combined では未実行。専用実測では 1 / 551) |
| 実行系 (exec-debugger) | 0 / 170 | 0% (分類・実測済み: 合否はデバッガ出力の照合で決まるため主経路ではパスしない) |
| コンパイル時検証系 | 15,226 / 18,332 | 83.057% (成否レベルの一致。照合系の扱いは stage 2 の節に記載) |
| rustdoc 検証系 | 0 / 1,735 | 0% (分類・実測済み: 合否は rustdoc の出力照合で決まるため主経路ではパスしない) |
| **全体** | **17,440 / 24,937** | **69.936%** |

統合後 combined 再実測 (both8、2026-11-21、main 455db46):
AO/AP/AS/AV/AU/AC/AW/AR の全統合後に `rustc-subset-run.py`
(--jobs 48、wall 256 秒) で全カテゴリを再実測した。exec は
22.102% から 34.876% へ (ボディ単位分離、クロージャ本体、
ADT 定数、Drop 実行の合算効果)、コンパイル時検証系は
73.009%。残りの失敗バケット上位は、単相化 instance 無し
1,692 件、変換済み本体の無い関数への呼び出し 1,177 件、
MIR 本体無し 495 件、ヘッダ機能による未実行 699 件、
条件を評価できない未実行 581 件、stage 2 の範囲外
(複数ファイル) 801 件、成否の不一致 (変換成功したが
失敗期待 255 件、コンパイル失敗だが成功期待 272 件)、
ローカルの型が未対応 228 件、interp 変換時の型不一致
159 件である。

add-minicore 対応後の combined 再実測 (both9-minicore、
2026-11-21、lane/stage2-minicore): `//@ add-minicore` を持つ
テストを、compiletest と同様に minicore スタブクレート
(tests/auxiliary/minicore.rs) を事前ビルドして `--extern
minicore=...` で渡す形で対象化した。ビルドはテスト自身の
edition と compile-flags (と minicore-compile-flags) ごとに
キャッシュし、`-Cpanic=abort` と `-Cforce-unwind-tables=yes`
も compiletest と同じに付ける。コンパイル時検証系は
13,384 → 13,451 (+67) で 73.374%、全体は 14,831 → 14,898
(+67) で 59.743%。exec は不変 (1,447 件) で、both8 で pass
だったものが落ちた回帰は 0 件、増分はすべて add-minicore の
427 件の内側である。その 427 件の内訳は pass 67、
unsupported 19 (単相化 instance 無し 10、MIR 本体無し 5、
InlineAsm 3、割当済み定数 1)、fail 16 (成功期待だがコンパイル失敗 13 は
外国ターゲット向けリビジョンのレジスタ・ABI 検査を host で
行うための食い違い、失敗期待だが変換成功 3 はターゲット
固有の失敗が host で再現しないもの。いずれも minicore の
受け渡し自体は成立している)、
未実行 325 (条件を評価できない 321、rustc-env で別途
ブロック 2、プラットフォーム限定 1、minicore ビルド失敗 1:
`-Zpacked-stack` が s390x 専用で host の rustc が拒否)。
ヘッダ機能による未実行は全体で 699 → 274 件になった。
stage 2 の aux 対象化後の combined 再実測 (2026-11-21、
`subset-be-stage2aux.json`、--jobs 48): stage 2 が補助ファイルを
要するコンパイル時検証系も対象にしたことで、コンパイル時
検証系は 13,384 件から 13,974 件へ (73.009% から 76.227%)、
全体は 14,831 件から 15,426 件へ (59.474% から 61.860%) なった。
従来「stage 2 の範囲外 (複数ファイル)」だった 801 件のうち
747 件を実行し、内訳は pass 590 / unsupported 56 / fail 101 /
未実行 54 である。801 件以外で pass から後退した件数は 0 件
(実行系で 5 件が unsupported から pass へ進んだ)。
proc-macro 補助クレート対応後の combined 再実測 (both9、
2026-11-21): both8 の成否不一致バケットを調査し、最大の
単一原因だった `//@ proc-macro:` 指令の未再現を直した。
stage 2 が指令の指す補助ファイルを compiletest と同じく
proc-macro クレートとしてビルドして `--extern` でリンク
する (bare の `--extern proc_macro` の付与、`-C metadata`
による同名クレートの区別、指令順の依存解決、edition 範囲
の解決、補助側の edition / compile-flags の尊重、ビルド
キャッシュ付き)。対象は「コンパイル失敗だが成功期待」
272 件中 104 件で、E0463 (crate が見つからない) が明示
されたものだけで 86 件を超える。結果はコンパイル時検証系
13,468 / 18,332 = 73.467%、全体 14,915 / 24,937 = 59.811%。
同バケットは 272 件から 172 件へ減り (E0463 は 0 件)、
fail から pass へ 85 件、fail から unsupported (変換未完
として正直に分類) へ 15 件が移った。逆に pass を 1 件
失った: tests/ui/proc-macro/issue-59191-replace-root-with-fn.rs
は E0463 を期待されたコンパイル失敗と誤認していた偽の
pass で、本来の期待エラーは MIR 後段の panic ハンドラ
検査のため、「変換成功したが失敗期待」バケット
(255 件から 256 件) へ移った。残る不一致の内訳は、
成功期待側 172 件では Stable MIR 変換時の panic 49 件
(async_closures の FIXME など)、dead_code の不一致
45 件 (ドライバがライブラリとしてコンパイルするため
到達可能性が compiletest と変わる)、ICE 16 件、edition
範囲 16 件など。失敗期待側 256 件では単相化・コード生成
段のエラー (単相化の再帰上限 16 件、required consts の
const 評価 11 件、エントリポイント検査約 28 件、重複
シンボル 9 件、SIMD の単相化検査 9 件ほか) と、.stderr
も注釈も持たない分類上の 31 件 (ignore-auxiliary の
補助ファイル 19 件、contracts の revision モード解釈
ほか) が中心である。
FnOnce 環境の値渡し後の再実測 (2026-11-21、lane/fnonce-env):
`FnOnce` を自種とするクロージャの本体は環境を値で受け取る
(第1引数がクロージャ値そのもの) が、interp は環境への参照を
渡していたため、本体冒頭のフィールド投影で型不一致になり
プログラム全体が未対応になっていた。変換済み本体の第1引数の
型を見て値そのものを渡し、Move のレシーバ場所を未初期化に
して消費を実体化するよう直した。再実測で exec は 1,447 件から
1,454 件 (34.876% から 35.045%)、全体は 59.502% (後退 0 件)。
増えたのは const 評価系とクロージャ系の 7 件である。
Lane BH 再実測 (bh2、2026-11-21、lane/boxderef-remaining):
BoxDerefTransmute バケット 54 件の原因は変換ではなく実行だった。
Box::new ビルトインが作るボックスはポインティを値に直接保持する
ため、精緻化が読み出すフィールド 0 の値そのものがポインティに
なり、集約構築された Unique/NonNull チェーンしか解包しない
キャストが全件型不一致で失敗していた。オペランドのプレース形状
(Box / Unique / NonNull、ドライバ登録名による分類) から期待
ラッパー深さを求め、浅い場合はポインティ自身への参照を返すよう
修正した。背後では同じ精緻化シーケンスのポインタ→整数
Transmute (アライメント事前条件チェック) が 51 件を塞いでいた
ため、合成アドレス (対象ごとに安定・プレースごとに相異・
4096 の倍数) を返す形でこの形状のみ対応した。再実測では
同バケットは 54 件から 0 件、exec は 1,447 件から 1,490 件
(+43、うち元バケット 33 件、Transmute 対応による他テスト
10 件)、全体は 14,831 件から 14,874 件 (59.646%)。元バケットの
残り 21 件は別の既存ギャップ (未モデル化 std/core 呼び出し
14 件、変換時の型不一致 7 件) に移り、both8 で pass だった
ケースの regression は 0 件、unsupported から fail に移った
ものが 5 件ある。
#### 変換失敗バケットの内訳調査と単相化 instance の収集 (2026-11-21、lane/stage2-census-fix)

both8 の失敗バケット上位2件 (単相化 instance 無し 1,692 件、
MIR 本体無し 495 件) を対象に、対象 2,187 件すべてへドライバの
変換専用モードを再実行し、対象クレート側の未変換関数を理由別に
数えた (census。ヘッダは基本形のみで近似している)。

| 理由 (対象クレート側の未変換関数) | 関数数 | テスト数 |
|---|---|---|
| 単相化 instance 無し (kind=Fn): ジェネリック関数定義そのもの | 3,780 | 1,605 |
| MIR 本体無し (うち匿名定数 {constant#N} が 2,983) | 3,219 | 559 |
| 単相化 instance 無し (kind=Ctor(Fn)): タプル構造体・列挙バリアントのコンストラクタ | 693 | 418 |
| 単相化 instance 無し (kind=Const): 大半が匿名定数 | 453 | 194 |
| ジェネリック文脈のクロージャ | 62 | 45 |

最大の内訳である kind=Fn (ジェネリックな自由関数、ジェネリック型の
impl メソッド、トレイト既定メソッド) を修正した。アイテム走査で
単相化できない Fn 定義を保留し、変換済み本体の呼び出し先から
要求される instance を到達可能性収集と同じ要領で解決・変換して
本体集合へ加える。定義は、要求された instance がすべて変換できた
場合に限り変換済みとして報告する。仮想 (動的ディスパッチ) instance
と本体の無い instance は到達可能性収集と同じガードで分類し、
走査は本体単位で panic を分離する (コンパイラ内部の panic が
変換全体を落とさないため)。局所 instance には到達可能性収集の
表現可能性スクリーンを適用しない (局所関数はアイテム走査でも
スクリーンなしで変換されるため)。

census 再計測: kind=Fn は 3,780 → 2,904 件、対象集合の未変換関数は
合計 8,291 → 7,401 件、対象クレート側の未変換が 0 になったテストは
347 件。combined 再実測 (both9-ba、本ブランチのドライバ、--jobs 48):

| カテゴリ | both8 | both9-ba |
|---|---|---|
| コンパイル時検証系 | 13,384 / 18,332 (73.009%) | 13,759 / 18,332 (75.055%) |
| 実行系 (exec) | 1,447 / 4,149 (34.876%) | 1,656 / 4,149 (39.913%) |
| 全体 | 14,831 / 24,937 (59.474%) | 15,415 / 24,937 (61.816%) |

テスト単位の合否反転は改善 584 件・悪化 0 件。バケット
「単相化 instance 無し」は 1,692 → 1,305 件。「MIR 本体無し」は
495 → 507 件と微増したが、これは instance 側の未変換が解消した
テストで匿名定数側が主因として残ったための付け替えである。

残る kind=Fn 2,904 件の内訳: 要求元が変換済み本体に無い自由関数・
固有メソッド 1,360 件 (610 テスト。ジェネリック定義のみのクレートや
未使用の API 面が中心)、ジェネリック型の impl メソッド 749 件
(291 テスト。外部クレートのジェネリックコード経由でのみ要求される
ものを含む)、async / コルーチンの {closure#N} 項目 570 件
(250 テスト)、トレイト宣言メソッド 224 件 (150 テスト)、
drop_glue 1 件。匿名定数 {constant#N} (kind=Const と MIR 本体無しの
合算で約 3,400 件) は次に大きい未着手の内訳である。
Lane BI 再実測 (both9-bi、2026-11-21、lane/builtins-round2):
builtins 第 2 ラウンドとして、both8 で最大の exec 失敗
バケットだった「変換済み本体の無い関数への呼び出し」
(1,177 件) の被呼出関数を集計し、値モデルで忠実に表せる
ものを builtin に追加した: `black_box`、`String::new`、
ビット操作 intrinsics (`ctpop` / `ctlz` / `cttz` /
`bitreverse` / `bswap`)、`saturating_add` / `saturating_sub`、
`atomic_load` / `atomic_store`、`Rc::new` / `Arc::new`
(既存の `Deref` builtin の対象に追加)、`mem::zeroed`、
`ptr::null` / `ptr::null_mut` (専用の参照ターゲット
`RefTarget::Null` で表現し、等価性は null 同士でのみ真、
参照解決は明示エラー)。combined 再実測 (--jobs 48) で
同バケットは 1,177 件から 1,132 件へ (−45)、exec は
1,447/4,149 (34.876%) から 1,466/4,149 (35.334%) へ
(+19)、全体は 59.474% から 59.550% (14,850/24,937) へ。
both8 で pass だったテストの回帰は 0 件。unsupported から
fail に移った 2 件 (構造体フィールドの破棄順、
`next_power_of_two` のオーバーフロー) は、実行が進んだ
先の次のギャップとして残る。`std::ops` の算術トレイト
(`Add` / `Mul` / `Neg` / `Not` など) は登録しない: core の
デフォルト本体が収集・実行でき (参照や `Wrapping` の
Self も本体経由で動く)、名前だけの builtin はカスタム
実装を影にして壊すため (ドライバの enum 定数テストで
実証し、登録を撤回した)。イテレータアダプタや
`catch_unwind` などゲストクロージャを呼ぶもの、割り当て
機械 (RawVec 系)、チャネル、スレッドローカル、io、env は
値モデルの境界として未対応のままである。

aux クレート種別と aux 単位 edition の対応後の combined
再実測 (2026-11-21、lane/aux-driver-gaps、`subset-bn.json`、
--jobs 48、wall 96 秒): ドライバの `--aux` 指定を
`name=path` の後ろに `;crate-type=<種別>` と `;edition=<版>`
の属性を付けられる形に拡張した (属性なしは従来通りで
後方互換)。ランナーは補助ファイル自身のヘッダも読むように
なり、`//@ proc-macro:` を補助ファイルが名指しするケース
(テスト自身の指令は stage 2 のランナー側ビルドが担い、
ドライバへは二重に渡さない) と、補助ファイル自身の
`//@ edition:` (aux 単位の edition がテストの edition に勝つ)
を spec に載せてドライバとネイティブ側の両方へ渡す。
結果は exec 1,469 → 1,482 件 (35.406% → 35.719%)、
コンパイル時検証系 14,041 → 14,140 件 (76.593% → 77.133%)、
全体 15,510 → 15,622 件 (62.195% → 62.646%) で、内訳は
pass 化 114 件・後退 2 件である。この実測には both10 の後に
統合された proc-macro ランナー側ビルドと FnOnce 環境の
値渡しの増分も含まれる。後退 2 件のうち
tests/ui/proc-macro/issue-59191-replace-root-with-fn.rs は
proc-macro 対応の節で記録した偽 pass の是正 (E0463 の誤認)
であり、tests/ui/closures/2229_closure_analysis/migrations/
nested-copy-drops-83176.rs は補助ファイルを要しない単一
ファイルの実行で drop glue 未対応に落ちたもので、本対応の
範囲外である。名前で確認した対象ケースはいずれも pass に
なった: tests/ui/imports/same-res-ambigious.rs
(補助ファイル経由の proc-macro と aux edition)、
tests/ui/editions/nested-macro-rules-edition.rs
(revision ごとの aux edition)、実行系の
tests/ui/proc-macro/crate-var.rs と meta-macro.rs
(ドライバ自身が proc-macro 補助クレートをビルドする経路)。

needs-* 条件の評価を入れた combined 再実測 (both10-bm、
2026-11-21、lane/needs-eval): `//@ needs-*` を一律に
「評価できない条件」として未実行にしていたのをやめ、
compiletest (src/tools/compiletest の directives/needs.rs)
と同じ出典で評価するようにした。ターゲット同一性に対する
compiletest 自身の規則 (needs-threads / needs-subprocess /
needs-asm-support / needs-dlltool)、公式リリースビルドの
構成 (needs-enzyme / needs-offload / rustc・std の
debug-assertions は無効、レイアウトのランダム化は無効で
needs-deterministic-layouts は成立)、ピン留めした rustc
への問い合わせ (all-target-specs-json による sanitizer
一覧・xray・動的リンク・再配置モデル、supported-crate-types、
--print cfg の target_has_atomic、backend-has-mnemonic /
backend-has-zstd、sysroot 内の profiler ランタイム・
sanitizer ランタイム・rust-lld)、そして LLVM コンポーネント
については代表ターゲットへの `#![no_core]` プローブの
コード生成可否で判定する。条件が不成立なら compiletest が
ignore するのと同じくプラットフォーム限定 (out of
environment) に数え、判定できない条件だけが未実行に残る。
結果: 「条件を評価できない未実行」は 902 → 0 件。
その 902 件の行き先は、pass 217 (実行系 76、コンパイル時
検証系 141。条件別では needs-subprocess 71、
needs-llvm-components 45、needs-asm-support 42、
needs-sanitizer-cfi 15、needs-sanitizer-address 12、
needs-xray 8 など)、プラットフォーム限定 69 (enzyme、
offload、rustc debug-assertions、未対応 sanitizer 等の
不成立)、compile-flags による未実行 313 (needs-llvm-
components が成立した外国ターゲット向けテストで、
`--target` が未導入という実理由に付け替わったもの)、
残りは実試行のうえで変換済み本体なし (150) や InlineAsm
など本来の未対応に落ちた。全体は 14,898 → 15,139
(+241) で 60.709%、実行系は 1,447 → 1,547 で 37.286%、
コンパイル時検証系は 13,451 → 13,592 で 74.144%。
both9 両実測 (minicore / revisions) に対して pass からの
回帰は 0 件。

crate-type の compiletest 準拠化と scoped 指令パースの
combined 再実測 (bq2、2026-11-21、lane/crate-type-fidelity):
both9 の census で残った dead_code の不一致 45 件の原因は、
ドライバが変換対象を `--crate-type=lib` で固定して
コンパイルしていたことだった。compiletest のソース
(src/tools/compiletest/src/runtest.rs) を確認すると、
compiletest はテスト本体に crate-type を一切指定しない
(例外は mir-opt の `--crate-type=rlib` と minicore
ビルドのみ)。指定が無ければ rustc の既定は bin であり、
`fn main` は到達可能性の根になり、エントリポイント検査
(E0601 など) が走る。lib では main は通常の非公開関数と
して dead_code の対象になる。そこで convert モードの
ドライバは crate-type を固定せず、転送された
`--crate-type` かソースの `#![crate_type]` が無ければ
rustc の既定 (bin) に従うようにした (run モードの lib
固定は実行系の都合で不変)。ランナーの stage 2 は
mir-opt に compiletest 自身の指定である rlib を転送し、
pretty (compiletest は bin としてコンパイルしない) には
従来の lib 変換を転送する。crate-type 変更のみの実測
(bq1) ではコンパイル時検証系 14,041 → 14,224
(76.593% → 77.591%)、全体 15,510 → 15,696
(62.197% → 62.943%)。「コンパイル失敗だが成功期待」は
326 → 177、「変換成功したが失敗期待」は 297 → 228 で、
dead_code のエラーが先頭に来る成功期待の失敗は
38 件 → 0 件になった。
測定中に第二の不整合も見つけた: ランナーの指令パースが
`//@ [lib] compile-flags:` のような空白入りの scoped
指令を丸ごと落としていた (compiletest のパーサは
`//@` の後の空白を許す)。lib 固定がこの欠落を隠して
いた。修正後の最終実測 (bq2、--jobs 48) はコンパイル時
検証系 14,183 / 18,332 = 77.367%、実行系
1,472 / 4,149 = 35.478%、全体 15,655 / 24,937 =
62.778%。「変換成功したが失敗期待」はさらに 168 件
まで減った (scoped な check-fail / compile-flags が
効くようになったため)。一方で、指令を無視したまま得て
いた pass 69 件を失った: 内訳は needs-llvm-components
など評価できない条件を持つ revision が正直に未実行に
なった 59 件 (条件評価の実装が入れば再び対象化される)、
polonius 等の scoped compile-flags や宣言済みの
crate_type が効いて変換未完になった 7 件、偽の pass が露出した 2 件
(compiletest 自身の自己テストと ignore-auxiliary の
補助ファイルで、いずれも単体クレートとして
コンパイルされることはない)、scoped な失敗期待の
解釈が変わった 1 件である。
#### 匿名定数の本体収集と trivial 定数の会計 (2026-11-21、lane/anon-const-bodies)

stage2-census-fix の census で残った匿名定数 {constant#N}
(「MIR 本体無し」2,983 件と kind=Const「単相化 instance 無し」
453 件、合算約 3,400 件) を処理した。実地調査で、この母数は
性質の異なる 2 群からなることが分かった。

第 1 群は trivial 定数である。リテラル・他定数へのパス・
定数パラメータで構成される定数は、rustc 自身が MIR 本体を
持たない設計であり (`is_trivial_const` が本体取得を None に
する)、静的・定数初期化子の中の長さ、型に現れる長さ、
impl 定数のリテラル値などが該当する。値は使用箇所すべてに
評価済みの形で畳み込まれ、変換済み本体のどこにも DefId 参照
は残らない (dump で確認済み)。プログラムに欠けているものは
何もないため、「MIR 本体無し」の skip ではなく、本体を持た
ない変換済みとして会計するよう改めた。

第 2 群はジェネリック文脈の定数である。ジェネリック関数内の
匿名定数や、ジェネリック impl の関連定数は、項目としては
単相化を要するため instance が無いが、項目自身の MIR 本体は
取得できる (`CrateItem::body`)。これを変換し、未評価定数の
参照が解決するキーである定数自身の DefId で登録するように
した。単相化された本体側では定数は評価済みのため、この本体
が実行時に参照されることはない (非ジェネリック定数の本体と
同じ位置づけ)。本体の変換自体が失敗する定数 (定数パラメータ
に依存するものなど) は、変換理由を記録して skip のままに
なる。

census 再集計 (bw 実測で残った 2 バケットのテスト 1,347 件を
ドライバで再走査し、対象クレート側の未変換関数を数えた):
kind=Const の skip は 453 件 → 0 件、{constant#N} の
「MIR 本体無し」は 2,983 件 → 0 件 ({constant#N} 名の skip は
kind=Fn 理由の 2 件のみ残る)。テスト単位のバケット
「MIR 本体無し」は全スイートで 537 件 → 0 件になった。

combined 再実測 (subset-bw.json、--jobs 48) と both11 の比較:

| カテゴリ | both11 | bw |
|---|---|---|
| コンパイル時検証系 | 14,509 / 18,332 (79.146%) | 14,980 / 18,332 (81.715%) |
| 実行系 (exec) | 1,751 / 4,149 (42.203%) | 1,813 / 4,149 (43.697%) |
| 全体 | 16,260 / 24,937 (65.204%) | 16,793 / 24,937 (67.342%) |

テスト単位の合否反転は改善 533 件・悪化 0 件。ただしこの
差分には both11 の後に統合された Lane BI / BF / BN の増分も
含まれる。本レーンに帰属する分は、「MIR 本体無し」バケット
の消滅 (537 件 → 0 件) と上記 census の定数 skip 消滅分で
ある。バケット「単相化 instance 無し」は 1,346 件 → 1,347 件
でほぼ不変であり、残りは kind=Fn (未要求のジェネリック定義
など) と kind=Ctor が支配的で、定数側の残件はない。

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

#### 外側から要求される単相化 instance の収集 (2026-11-21、lane/external-demand-scan)

到達可能性収集 (reach) で集めた外部クレートの本体の中からのみ
要求されるローカルなジェネリック instance が、どちらの収集
パスでも変換されない隙間があった。要求走査
(collect_local_instances) は変換済みのローカル本体だけを走査し、
到達可能性収集はローカル定義へ解決される呼び出し先を対象外と
していたためである。

修正: 到達可能性収集が走査した外部本体を返すようにし
(CollectedFns::walked_bodies)、収集上限 (1,024) を両パスで
共有した。collect() は不動点まで交互に回す: 新しく集めた外部
本体を要求走査し、新しいローカル instance を変換し、その
instance 本体から到達可能性収集を残り予算で再実行する。要求
状態 (InstanceDemandState) はパス間で共有する。

機構の証明: ローカルなジェネリック Iterator の next を呼ぶ
`impl Sum<i32> for Total` は、修正前は両方とも未変換で実行が
失敗し、修正後は両方とも変換され 21 を出力して終了コード 0
になった。

実測 (ベース 864a7e8 で分離計測): ベース単独 65.501%
(16,334 / 24,937) → 本レーン適用後 65.505% (16,335 / 24,937)。
合否反転は改善 1 件・悪化 0 件で、「単相化 instance 無し」
バケットは 1,347 件 → 1,346 件、反転したのは
tests/ui/async-await/issue-72470-llvm-dominate.rs。テスト単位で
は反転しなかったものの、テスト内の未変換関数が減ったものが
3 件ある。

正直な帰属: Lane BA の census が数えた 749 関数は「失敗して
いるテストの中の未変換関数」であって、この隙間だけが主因の
テストではない。影響を受けるテストのほぼ全てに、後段の
ブロッカー (async / coroutine の {closure#N} 等) が重なって
いる。対象外の境界: 走査対象外の外部本体で鎖が切れる場合
(例: Iterator::fold)、および builtin が所有する呼び出し先
(ユーザ型に対する IntoIterator::into_iter のような blanket
impl 本体は収集されない)。

#### stage 1: 補助ファイル (aux) を要するテストの実測 (2026-11-21)

ドライバが `--aux <name>=<file.rs>` で補助クレートを受け取れる
ようになり、stage 1 の対象を補助ファイルを要するテストにも
広げた。補助ソースはピン留めした rustc でライブラリクレート
としてコンパイルし (`-Zalways-encode-mir` で全関数の MIR を
メタデータに含める)、メインのコンパイルに `--extern` で
リンクする。補助クレートの関数は、メインと同じ Stable MIR
セッションから同じ MirProgram へ変換する。補助ソースが
`#![crate_name]` / `#![crate_type]` を宣言している場合は宣言を
正とし、メインと同名の補助クレートは `-C metadata` で区別する
(rustc の E0519 回避)。ジェネリックな補助関数は、ローカルの
ジェネリック関数と同じく「単相化された本体が無い」として
記録する。`proc-macro` / `cdylib` の補助クレートはコンパイルと
リンクのみで、関数は変換しない (proc マクロはコンパイル時に
コンパイラ内で実行され、cdylib は Rust API を公開しない)。

ランナーは `//@ aux-build:` / `//@ aux-crate:` を推移的に
解決し (補助ファイル自身の指令を含み、探索は `auxiliary/`
サブディレクトリを優先する)、依存順に両側へ渡す。ネイティブ
側も同じ順序で補助クレートをビルドしてリンクする。

対象は実行系 (exec) のうち補助ファイルを要する ui /
ui-fulldeps の 341 件である (補助ファイルを要する実行系は
サブセット全体で 450 件あり、残りは ui / ui-fulldeps 以外の
スイートに属して stage 1 の対象外)。結果は pass 72、
unsupported 189、未実行 79、fail 1 で、実行系全体は
943 / 4,149 = 22.728%、指標2(b) 全体は 12,131 / 24,937 =
48.647% になった。単一ファイル側の pass は減っていない
(同じ再実行で回帰 0 件を確認)。

unsupported の上位は、ローカルの型が未対応 83 件 (Dynamic、
クロージャなど interp の型モデル)、変換済み本体の無い関数への
呼び出し 59 件、その他の変換ギャップ 18 件、BoxDerefTransmute
キャスト 10 件である。未実行の内訳は、ヘッダ指令が stage 1 の
対象外 57 件、ネイティブ側のコンパイル失敗 16 件、適用可否を
判定できない条件 4 件、ネイティブ実行の期待不一致 2 件である。
ネイティブ側のコンパイル失敗には、`rust_test_helpers` 静的
ライブラリがこの環境に無いものと、補助ファイル自身の
`//@ compile-flags` (prefer-dynamic など) を再現していない
ものが含まれる。fail 1 件 (`tests/ui/attributes/
used_with_archive.rs`) は、補助クレートの静的コンストラクタ
(`.init_array`) がネイティブでは main の前に実行されるが
interp は main しか実行しないため、標準出力が一致しない。
#### stage 1 再実測: 変換のボディ単位分離 (2026-11-21)

interp の変換層 (mir_convert) を関数単位に分離し、1つの関数が
変換できない場合にプログラム全体を落とさないようにした変更の
再実測である。変換できない関数は理由とともに変換不可として
保持され、実際に呼ばれた時点でその理由を含む未対応エラーに
なる。エントリポイント自身が変換できない場合は、従来どおり
全体の変換失敗である。

方法: stage 1 のみを 48 並列で再実行した。候補集合 (3,699 件)、
未実行集合 (876 件)、fail 集合 (8 件) は変更前の実測と一致して
おり、下記の差分は変換分離の効果だけを表す。

| 状態 | 変更前 | 変更後 |
|---|---|---|
| pass | 1,004 | 1,060 |
| unsupported | 1,811 | 1,755 |
| fail | 8 | 8 |
| 未実行 | 876 | 876 |

実行系 (exec) カテゴリのパス率は 1,004 / 4,149 = 24.199% から
1,060 / 4,149 = 25.548% になった。pass から他の状態へ移った
テストは 0 件である。

変更前に変換全体の失敗 (`MIR conversion failed`) で落ちていた
496 件の内訳は、pass 56 件、残り 440 件は変換を通過して実行段の
理由で unsupported になった。動機となった
`tests/ui/traits/object/nested-trait-method-call.rs` (到達しない
入れ子関数内の Dynamic ローカルが全体を落としていた) は
pass になった。
#### クロージャ本体の変換 (2026-11-21)

ドライバはクロージャ item を `Instance::resolve_closure` で
instance 化し、本体を変換対象に含めるようになった。クロージャの
generic 引数に記録された kind (Fn / FnMut / FnOnce) をそのまま
要求するため、instance の本体はクロージャ本体そのものになる。
generic 引数にパラメータが残るクロージャ (ジェネリックな関数・
メソッドの中で定義されたもの) は、単一の単相化済み本体が存在
しないため、理由を明示して SKIPPED のままにする。非同期
クロージャ (coroutine closure) も、coroutine 本体が未対応で
あることを理由に明示して対象外である。なお、非同期クロージャを
含むクレートでは、その item の型を引く時点で Stable MIR の
型変換が panic する (FIXME(async_closures)) ため、収集全体が
変換失敗として報告される。以前は interp 変換の段で失敗して
いたケースで、状態 (unsupported) は変わらず理由だけが
この panic に変わったものが 1 件ある。

stage 1 の再実測 (本変更を加えたベースで候補 3,699 件を全件
再実行): 実行系の pass は 1,037 件。直近の全件実測で
「unknown function …::{closure#0}」が未対応の理由だった
45 件のうち 33 件が pass になり、同じ理由で残ったものは
0 件になった。残りの 12 件は本体の変換後に次の壁で
unsupported である: FnOnce の by-value な環境渡し (interp 側)
5 件、変換済み本体の無い関数への呼び出し (String::new など)、
Drop ターミネータ、Transmute キャスト、Subslice 射影、
クロージャを返すクロージャの呼び出しの型不一致、値の等価
比較の型不一致。

(2026-11-21 追記: 上記の「FnOnce の by-value な環境渡し
5 件」は lane/fnonce-env で解決した。interp が変換済み本体の
第1引数の型を見て、環境への参照ではなくクロージャ値そのもの
を渡し、レシーバ場所の消費も実体化する。詳細は指標2(b) の
パス率実測の節の再実測メモを参照。)

副作用として 1 件が pass から unsupported になった
(tests/ui/higher-ranked/trait-bounds/hrtb-debruijn-object-types-in-closures.rs)。
呼ばれないクロージャの本体が変換対象に増えたことで、その本体に
含まれる interp 未対応の型 (Dynamic) がプログラム全体の interp
変換を失敗させる。本体単位で変換の成否を分離するのは interp
側の課題として残る。

#### トレイト宣言メソッドのキー統一 (2026-11-21、lane/trait-method-key、Issue #64)

トレイト宣言メソッドへの呼び出しは `FnDef` 対象がトレイト宣言を名指すため `(トレイト定義, 解決済み instance)` でキーされるが、アイテム走査は impl メソッドを `(impl 定義, instance)` で登録しており、両キーが一致せず impl 本体が変換済みでも実行時に「no converted body」で失敗していた。ドライバにエイリアス登録パスを追加した: 変換済み本体 (ローカルアイテム・要求 instance・到達可能性収集した外部本体) の Stable MIR 呼び出し対象を走査し、解決先 instance の本体が impl キーに存在してトレイトキーに無い場合、その本体のコピーをトレイトキーで登録する (汎用 impl メソッドは Lane BA の要求 instance 収集が既に参照キーで登録しており、本パスはアイテム走査で変換された非汎用 impl メソッドの穴を埋める)。あわせて interp の `CastKind::PointerExposeProvenance` (`ptr as usize`) を合成アドレスへのキャストとして評価可能にした (ポインタ→整数 Transmute と同じ扱い)。

レーン単独の実測 (base main 864a7e8 との単一ラン比較、subset-bvbase.json / subset-bv2.json、--jobs 48): 指標2(b) 全体 **65.501% → 66.066%** (16,334 → 16,475)、実行系 **1,813 → 1,954 (+141)**、コンパイル時検証系は 14,521 で不変 (stage 2 は無関係)。テスト単位の反転は改善 142 / 悪化 1。悪化 1 件 `tests/ui/derives/deriving-with-repr-packed-3.rs` は偽 pass の是正: 従来はトレイトキーに本体が無いため PartialEq の builtin (構造比較) が派生 eq 本体を迂回してパスしていたが、本来の派生 eq が実行されるようになり、`Aligned::eq` 内の `std::mem::align_of::<Aligned>()` (ユーザー ADT のレイアウトは interp に ADT レイアウト表が無く未対応) で正直に unsupported になった。なお同テストが使う `ptr as usize` は本レーンのキャスト対応で実行可能になっている。

#### combined 再実測 both11 (2026-11-21、main 379e0ac)

Lane BA/BG/BK/BH/BP/BJ 統合後の単一ラン再実測 (subset-both11.json、--jobs 48): 指標2(b) 全体 **62.195% → 65.204%** (16,260/24,937)。実行系 1,751/4,149 = 42.203%、コンパイル時検証系 14,509/18,332 = 79.146%。残りバケット上位: 単相化 instance 無し 1,346、評価不能条件 935 (Lane BM 対応中)、本体なし呼び出し 872、型不一致 235、interp 型モデル 256、Drop 未対応 54。 この実測の後に Lane BI (builtins round 2)、Lane BF (dyn 型モデル)、Lane BN (aux spec) が統合されており、増分は次回 combined 実測で確定する。

#### combined 再実測 both12 (2026-11-21、main e8075d5)

Lane BV (Issue #64: トレイトキー別名) / BW (匿名定数) / BX (外部需要 fixpoint) / BY (Issue #65: impl 解決と cleanliness fixpoint) / BS (Drop 需要パスと回帰修正) / BZ (コルーチン状態オブジェクト) / CC (RefCell モデル) 統合後の単一ラン再実測 (subset-both12.json、--jobs 48): 指標2(b) 全体 **65.204% → 69.936%** (17,440/24,937)。実行系 2,214/4,149 = 53.362%、コンパイル時検証系 15,226/18,332 = 83.057%。残りバケット上位: 単相化 instance 無し 1,375、本体なし呼び出し 914、generic 文脈のクロージャ 47、Allocated 定数 44、driver error 54。指標3 は both12 結果で 923/1,435 = 64.3% (strict 889)。この実測の後に Lane CL (incremental スイートの revision 別判定、単独寄与 +64 件) が統合されており、その増分は次回 combined 実測で確定する。

#### Lane BY 再実測 (by3、2026-11-21、lane/dst-index-semantics)

Issue #65 の修正: トレイト項目への呼び出しは、名前ベースの
builtin に落ちる前に、呼び出しの Self 型について収集済みの
impl 本体へ解決する (mir_convert の `resolve_impl_callee`)。
builtin の構造比較・スライス index モデルは、ユーザー定義の
impl がプログラム内にある場合に誤った結果を返していた
(dst-index-success.rs: 0 フィールド構造体への index が
長さ 0 のスライスとして扱われていた)。
解決先は第 1 フェーズの clean な本体に限る: `clean_targets` が
第 1 フェーズ IR 上の不動点で、本体自身・直接の Call 先・
Drop グルーのデストラクタ先のいずれかが未対応構文
(Unsupported 文 / 終端子、`Rvalue::Unsupported`) や未変換本体に
触れる本体を伝播的に除く。CallBuiltin / CallTrait /
CallIndirect と、プログラム本体でない名前への Call
(vtable ディスパッチ) は中立。
deriving-with-repr-packed-3.rs の形状がこの制限を決める:
eq 本体はすべて変換できるが、手書き impl が呼ぶ `check_align`
の PointerExposeProvenance キャストが未対応で、解決すると
builtin の構造比較が正確な場面で実行時エラーになる。

単一ラン再実測 (subset-by3.json、--jobs 48): 指標2(b) 全体
**65.946%** (16,445/24,937)。both11 比 +185 件、回帰 0 件。
実行系 1,924/4,149 = 46.373%、コンパイル時検証系
14,521/18,332 = 79.211%。dst-index-success.rs と
deriving-with-repr-packed-3.rs はどちらも pass。
制限なしの 2 フェーズ実測 (by2、65.982%、16,454 件) と比べると
10 件が both11 時点の状態 (未対応) に留まる。解決先の本体が
推移的に clean でないため解決を見送ったもので、内訳は
autoderef-method.rs、
specialization-trait-item-not-implemented-rpass.rs、
run-splat.rs、default-method/trivial.rs、
traits/inheritance/{cross-trait-call, self-in-supertype,
self, subst2}.rs、typeclasses-eq-example{,-static}.rs。
clean 判定は第 1 フェーズ IR に対する経路非感応な近似であり、
実行されない分岐に未対応構文を持つ本体も解決先から外れる。
変換と clean 判定を共同不動点にする精緻化が今後の課題。

#### stage 1 再実測: revisions の revision 単位判定 (2026-11-21)

stage 1 も stage 2 と同様に、`//@ revisions:` を持つテストを
revision ごとに判定するようになった。各 revision は、その
revision の指令ビュー (HeaderView) で見た run モード・
compile-flags・edition・failure-status と、compiletest が
付ける `--cfg <revision>` / `--check-cfg` を使って、ネイティブ
側とドライバ側の両方を実行し、同じ revision のネイティブ
実行と突き合わせる。レコードの合否は「実行された revision が
すべて pass」であり、ignore- / only- 条件で外れた revision は
実行されない (13 件)。補助クレートは revision ごとに、その
revision のビューで解決する。revisions を宣言していない
テストの扱いは従来どおりで、scoped 指令だけを持つテストは
stage1_blocker で未実行のままである。

combined 再実測 (both9-az、--jobs 48、wall 94 秒。ドライバは
main 455db46 相当) の結果: 従来 revisions の存在を理由に
未実行だった 152 件 (392 revision) の内訳は、pass 16、
unsupported 79、未実行 57 になった。unsupported の上位は、
変換済み本体の無い関数への呼び出し 43 件、ローカルの型が
未対応 (interp の型モデル) 16 件、MIR 変換失敗 10 件である。
未実行 57 件の内訳は、revision 単位の needs-* をこのホストで
検証できない 20 件、revision が run-crash 14 件、run モードを
持たない (コンパイルのみの) revision 7 件、ネイティブ実行が
モードの期待と不一致 6 件、ネイティブ側のコンパイル失敗
5 件、revision が ignore-test 3 件、revision の exec-env
2 件で、いずれも「ある revision を試行できない場合は
レコード全体を未実行とする」集約規則による。

実行系 (exec) は 1,447 / 4,149 = 34.876% から
1,463 / 4,149 = 35.262% へ (+16)、指標2(b) 全体は 59.474%
(14,831 / 24,937) から 59.538% (14,847 / 24,937) へ上がった。
コンパイル時検証系は 13,384 / 18,332 = 73.009% で不変。
both8 から pass を失ったテストは全カテゴリで 0 件である。
#### dyn Trait の型モデル (2026-11-21、lane/dyn-type-model)

interp の型/値モデルに `dyn Trait` を組み込んだ。ドライバは
vtable を実体化しないため、dyn 値はデータポインタと vtable の
同一性 (具象型名とトレイト境界) を組にした fat ポインタ
`Value::DynRef` として表す。`ir::Ty::Dynamic` が変換可能な
ローカル型になり、Unsize 強制は変換時に Slice / Dyn を区別する
メタを伴う。トレイトメソッド呼び出しは、具象型の impl 本体が
プログラム内にあれば `<Concrete as Trait>::method` へ
ディスパッチし (スーパートレイト・アップキャストを含む)、
無ければ呼び出し地点で `UnsupportedDynamicDispatch` を返す。
`Box<dyn Trait>` のレシーバ構築 (ポインタフィールド射影、
`BoxDerefTransmute`、`&*_p`) と `dyn Fn` 系の呼び出しも通る。

combined 再実測 (subset-bf、--jobs 48、both8 と同一 survey):
exec は 1,447 件から 1,470 件 (34.876% から 35.430%)、
全体は 59.474% から 59.566%。pass から他状態への後退は
全カテゴリで 0 件。both8 で Dynamic が理由で unsupported
だった exec 152 件の内訳は、pass 22 件、残りはより後段の
境界への前進である: 変換済み本体の無い関数への呼び出し
90 件、by-reference 定数 15 件、interp 変換時の型不一致
12 件、Drop ターミネータ 6 件、未初期化読み 3 件、
BoxDerefTransmute のオペランド形状 2 件、終了コード不一致
1 件 (tests/ui/dst/dst-index-success.rs: 実行されるが結果が
誤る。別途調査)、その他 1 件。

明示的な境界として残るもの: 所有する dyn 値の Drop
(Box<dyn> のスコープ終了時。vtable の drop エントリが
必要で、具象型は実行時にしか決まらないため glue を静的に
組めない)、トレイトのデフォルトメソッド本体 (ドライバが
収集しないため impl 本体が存在せず、ディスパッチ地点で
明示エラー)、`PtrMetadata` の dyn ポインタへの適用、
vtable 同一性の等価比較、dyn 値の Display/Debug 整形。

#### Lane BL: i128/u128 の定数復号とシフト量規則 (2026-11-21)

128 ビット整数の演算・キャスト・表示は既に動作していたが、
参照経由で promote された配列定数 (`[i128::MIN][0]` のような
形) の要素が文脈なしで復号され、16 バイトスカラが符号なしに
落ちて負値が巨大な正値になっていた (`convert_scalar` の
フォールバックが 128 ビットを除外していたことと、タプル形
定数に配列要素型が伝わっていなかったことの二段)。両方を直し、
`tests/ui/codegen/issue-101585-128bit-repeat.rs` が pass に
なった。あわせて、シフトのオーバーフロー規則を両エンジンで
Rust の量規則に統一した: 値から押し出されるビットは失われる
だけでオーバーフローではなく (`1 << 127 == i128::MIN` は
panic しない)、checked 形のフラグも量のみで決まる。従来は
interp・JIT とも「押し出されたら overflow」としており、
`1i64 << 63` でも panic していた。セマンティクステスト
18 件 (境界値・ゼロ除算と MIN / -1 のメッセージ・飽和を
含む float キャスト) と corpus `int128_ops` (47 件) を追加。

combined 再実測 (subset-bl、--jobs 48): 実行系は
1,463 / 4,149 = 35.262% から 1,470 / 4,149 = 35.430% へ、
指標2(b) 全体は 59.538% (14,847 / 24,937) から 59.566%
(14,854 / 24,937) へ。増分 7 件のうち 6 件は、基準実測
(both9-az) の後に main へ入った Lane BD の重複関数修正が
効いたもので、本レーンの定数修正によるものは
issue-101585-128bit-repeat の 1 件である。pass を失った
テストは 0 件。
#### コルーチンの状態オブジェクト実行 (2026-11-21、lane/coroutine-yield)

前提の発見: Stable MIR (rustc_public 1.100.0) は `Yield` /
`CoroutineDrop` ターミネータを公開しない (`TerminatorKind` は
10 種のみ)。ドライバが受け取るのはコルーチン変換後の本体で、
判別子ディスパッチ・状態フィールドの保存/復元・
`SetDiscriminant`・`Poll` / `CoroutineState` の返却までが通常の
文と終端子で表れている。そこで Yield そのものではなく状態
オブジェクトをモデル化した: `Value::Coroutine { variant, fields }`
を `Value::Enum` とは別にし (Downcast は透過、フィールド書き込みは
`Uninit` で伸長、`SetDiscriminant` はタグのみ変更)、生成は
`AggregateKind::Coroutine` → `Rvalue::CoroutineAggregate`、drop
glue は状態フィールド表に対する Variants プランで組む。ドライバの
到達可能性スクリーニングが `Ty::Coroutine` ローカルを拒んでいた点も
直した。意味テストは `crates/sparagmos-interp/tests/coroutine_tests.rs`
に 6 件 (生成器の yield 列と完了、完了後の再開 panic、yield しない
async fn の即 Ready、Ready な future の await、Pending を返す内側
コルーチンの await、suspend 中コルーチンの drop glue) を手組みの
変換後形状ボディで検証し、全件パス。

combined 実測 (subset-bz、--jobs 48、both11 と同一 survey、
ベースは main a1de97a + 本レーン): 全体は 16,588/24,937 =
66.52% で both11 の 65.204% から +328 件、pass から他状態への
後退は 0 件。ただしこのベースには both11 未収録の BI/BF/BN/BM が
含まれるため +328 の全てが本レーンの寄与ではない。本レーンに
帰属する動き: both11 で失敗理由に coroutine / Yield を含んだ
記録は 91 件 → 25 件に減り、うち 27 件が pass に転じた
(tests/ui/coroutine/* の gen テスト群、async-await の
issue-65419-coroutine-resume-after-completion など)。残る 64 件は
コルーチン以外の境界へ前進した: 変換済み本体の無い関数への
呼び出し 38 件 (大半は `Waker::noop` と waker vtable)、
by-reference 定数 15 件、型不一致 6 件、Drop 4 件、未初期化
読み 1 件。カテゴリ別は exec 1,921/4,149 = 46.3%、
コンパイル時検証系 14,667/18,332 = 80.008%。

指標1 への影響はない: variant-matrix の Yield / CoroutineDrop
行は `none` のまま 27/40 で不変 (JIT に Yield の lowering は
無く、interp もドライバ経由では Yield ターミネータを受け
取らない)。

副次的な修正: anon_const / body_isolation / robustness の
各テストが「変換できない型」のフィクスチャにコルーチン型
ローカルを使っていたため、dyn フィクスチャの時と同じ手順で
`Ty::CoroutineWitness` (依然として未モデル) へ差し替えた。

明示的な境界として残るもの: Yield / CoroutineDrop ターミネータ
自体 (ドライバからは到達不能)、async クロージャ
(`AggregateKind::CoroutineClosure`)、`Ty::CoroutineWitness`、
実 Waker 構築を要する async の end-to-end 実行 (`Waker::noop` /
RawWakerVTable の by-reference 定数、`Arc<T>` から Waker への
`From`)、コルーチンへの `size_of_val`、async-drop が使う
DerefMut の runtime support。

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

テストごとの `compile-flags` は、コンパイルそのものを置き換える
もの (ホスト以外の `--target`、`--print`、`--sysroot`、出力先の
指定、`-Zunpretty`) を除き、ドライバの `--rustc-arg` 経由で内包
rustc へ転送して扱う。revisions を持つテストはリビジョンごとに、
そのリビジョンの指令ビュー (スコープ無し + そのリビジョン宛ての
指令)、リビジョン別の compile-flags、compiletest が加える
`--cfg <revision>` と `--check-cfg` を使って判定し、全ての
リビジョンが pass したときに pass とする。ignore / only 条件で
外れたリビジョンは実行しない。revisions を持つ 1,438 件の内訳は
pass 660 / unsupported 357 / fail 109 / 未実行 312 である。

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
pass 13,126 件のうち 11,684 件はいずれかの照合を本来要する
(うち `.stderr` は 11,158 件)。stage 2 の pass は成否レベルの
一致であり、compiletest の完全な合否とは異なる。

結果 (コンパイル時検証系 18,332 件):

| 状態 | 件数 | 内容 |
|---|---|---|
| pass | 13,126 | 期待と成否が一致 (下記の内訳) |
| unsupported | 2,565 | コンパイルは成功したが変換が不完全 |
| fail | 730 | 期待と成否が不一致、またはドライバの異常終了 |
| 未実行 | 1,911 | 下記の内訳のとおり。いずれも fail として数える |

期待別の内訳 (期待が単一に定まる 16,326 件): 失敗期待 11,160 件は
pass 10,861 / fail 279 / 未実行 20、成功期待 4,976 件は pass 2,191 /
unsupported 2,473 / fail 259 / 未実行 53、クラッシュ期待 190 件は
pass 0 / fail 189 / 未実行 1 である。

未実行 1,911 件の内訳: 補助ファイルを要するもの 801 件、
incremental スイート 158 件 (セッションをまたぐ再コンパイルが
合否を決めるため、1回の変換では判定できない)、ヘッダ指令が
対象外 453 件 (add-minicore 427、うちリビジョン単位の適用 240、
rustc-env 19、ignore-test 等 7)、適用可否を判定できない条件
308 件、プラットフォーム限定 113 件、転送できない compile-flags
77 件、実行を要する合否 1 件。compile-flags と revisions は
対象化され、未実行の理由ではなくなった (前回は合わせて 4,054 件)。

unsupported 2,565 件の理由の上位は、単相化されたインスタンスが
無い 1,927 件、MIR 本体が得られない 478 件、割当て済み定数
(ConstantKind::Allocated) 67 件、BoxDerefTransmute キャスト
63 件、InlineAsm 16 件である。lib としてコンパイルする ui テスト
では、呼び出されないジェネリック関数に単相化インスタンスが
無く、変換が欠ける形が最も多い。

fail 730 件の内訳は、成功期待だがコンパイル失敗 258 件、失敗期待
だが変換成功 257 件、crashes の再現失敗 189 件、ドライバの
エラー 24 件、タイムアウト 2 件である。成否が食い違う主因は、
ドライバの固定したコンパイル形態 (crate-type は lib、出力は
メタデータ) と compiletest の形態の差で、エントリポイント検査や
コード生成段でのみ起きるエラー、補助クレートへの参照 (E0463)、
pretty スイートの断片的なファイル (整形だけが目的で型検査を
通らないもの) がサンプルで確認できた。

crashes は前回 (pass 135 件) から pass 0 件に下がった。原因は
ドライバの panic 捕捉 (main に統合済み) で、変換中に起きた
コンパイラの panic がプロセスの異常終了 (終了コード 101) では
なく、変換失敗としての終了コード 1 に変換されるようになった
ためである。ICE 自体は起きている (結果 JSON の `ice` 記録で
確認できる) が、stage 2 の「クラッシュ = 終了コードが 0 でも
1 でもない」という判定では pass にならない。判定の定義は
変えていない。

#### stage 2: 補助ファイル (aux) を要するテストの対象化 (2026-11-21)

stage 2 でも補助ファイルを要するテストを対象にした。従来は
単一ファイルでないというだけで一律に対象外としていたが、
stage 1 と同じ解決 (補助ファイル自身の指令を含む推移的な
解決、依存順) で補助クレートを集め、ドライバの変換専用
モードへ `--aux` で転送する。変換の完全性は対象クレート
自身の関数 (レポートの target-crate converted / skipped) で
判定するため、補助クレート側の未変換の本体は不完全扱いに
ならない。補助クレートのコンパイル失敗は、compiletest でも
そのテストの失敗になるため、変換の失敗としてそのまま扱う。
aux 指令がリビジョン宛てにのみ書かれたテスト (4 件) は、
リビジョンごとにそのビューの指令から解決する。指令で説明
できない複数ファイル構成と、解決できない補助ファイルは
従来どおり対象外である。

対象化した 801 件の fail 101 件の内訳は、成功期待だが
コンパイル失敗 41 件 (うち補助クレート自体のコンパイル
失敗 18 件)、失敗期待だが変換成功 35 件、ドライバの
エラー 24 件、crashes の再現失敗 1 件である。失敗期待だが
変換成功には、重複するグローバルアロケータのような
リンク段階でのみ起きるエラーが含まれ、変換専用モードの
成否判定では従来から見えない。ドライバのエラーの多くは
extern-flag 系で、compiletest が補助成果物のパスを
compile-flags へ埋め込む前提のテストである。未実行の
54 件は incremental スイート 22 件、適用可否を判定できない
条件 13 件、プラットフォーム限定 11 件、ヘッダ指令 7 件、
compile-flags 1 件である。

ドライバ側に残る制約が 2 つある。1 つは `//@ proc-macro:`
指令で宣言される proc-macro 補助クレートで、ソースが
`#![crate_type = "proc-macro"]` を宣言していない場合、
ドライバは `--aux` のクレートを rlib としてコンパイルする
ため構築できない (aux の閉包に同指令を含むテストは
12 件。例: tests/ui/imports/same-res-ambigious.rs)。もう
1 つは補助クレートの edition で、ドライバは全ての補助
クレートをメインの edition でコンパイルするが、compiletest
は補助ファイル自身の `//@ edition:` を使う。補助ファイルが
edition を宣言するテストは最大で 59 件あり、宣言がメインと
異なる場合に成否がずれる (例:
tests/ui/editions/nested-macro-rules-edition.rs は dep の
edition がリビジョンごとに逆転するため、両リビジョンとも
不一致になる)。

stage 1 と合わせたカテゴリ別のパス率:

| カテゴリ | pass / 母数 | パス率 |
|---|---|---|
| 実行系 (exec) | 917 / 4,149 | 22.102% |
| 実行系 (procedure) | 0 / 551 | 0% (未実行) |
| 実行系 (exec-debugger) | 0 / 170 | 0% (分類・実測済み: 合否はデバッガ出力の照合で決まるため主経路ではパスしない) |
| コンパイル時検証系 | 13,126 / 18,332 | 71.602% |
| rustdoc 検証系 | 0 / 1,735 | 0% (分類・実測済み: 合否は rustdoc の出力照合で決まるため主経路ではパスしない) |
| **全体** | **14,043 / 24,937** | **56.314%** |

測定条件: 48 並列、stage 1 と stage 2 の候補合計 22,031 件の
処理に 122.7 秒。タイムアウトはドライバ変換 60 秒 (1件あたり)。
結果は `subset-run-flags-results.json` (測定クローンに保存) に
テストごとの期待・状態・理由と、照合を要する検証の有無を記録した。
再実行は
`python3 scripts/rustc-subset-run.py --survey-json <survey JSON> --rust-dir <チェックアウト> --driver-bin <ドライバ> --out <結果 JSON>`
(`--stages 1` で stage 1 のみ、`--stages 2` で stage 2 のみ)。
stage 1 も compile-flags を両側 (ネイティブ rustc とドライバ) へ
転送するようになり、stage 1 の pass は 917 件 (実行系 22.102%)
である。前回の統合実測 (870 件) からの差には、転送対応のほか、
ドライバの panic 捕捉の統合で fail が 24 件から 2 件に減った
ことも含まれる。

#### stage 2: incremental スイートの revision 単位判定 (2026-11-21、lane/incremental-suite)

incremental スイート (180 件) は「セッションをまたぐ再コンパイルに
依存する」という理由で stage 2 から一律に対象外だった。これを
やめ、revision 単位で判定するようにした。根拠は compiletest 自身の
判定方法 (src/tools/compiletest/src/runtest/incremental.rs) である:
各 revision の成否は、revision 名の接頭辞が決める種類
(cpass/bpass/rpass はコンパイル成功、bfail はクリーンな失敗) で
決まる。revision 群は 1 つの incremental ディレクトリを共有するが、
キャッシュが温まっているかどうかは「その revision のソースが
コンパイルできるか」を変えない。rpass は compiletest では
コンパイル後にバイナリも実行するが、stage 2 は他のスイートと
同様にコンパイル成否だけを判定する (出力照合も行わない、という
既存の割り切りと同じ)。

対象外として残るのは、判定そのものがセッション間に持ち越される
状態で定義されているテストで、一律の除外ではなくテストごとに
理由を記録する:

- dep-graph 表明を持つテスト 81 件: テストファイルまたは補助
  クレート内の #[rustc_clean] / #[rustc_dirty] /
  #![rustc_partition_reused] / #![rustc_partition_codegened]。
  -Z query-dep-graph の下では、前セッションから読み込んだ
  dep-graph との比較で成否が決まるため、独立した revision 単位の
  変換では再現できない (内訳: rustc_clean 60 件、
  rustc_partition_reused 27 件、rustc_partition_codegened 16 件、
  rustc_dirty 0 件、重複を除き 81 件)。
- 起動時のキャッシュ状態を表明するテスト 1 件:
  -Zassert-incr-state=loaded を要求する
  issue-85360-eval-obligation-ice.rs。=not-loaded の表明は
  独立変換が構成上満たすので妨げにならない。

実測 (対象を incremental に絞った実行 subset-cl-incremental.json と
全体実行 subset-cl.json で一致、ドライバは現行 main):
判定に入った 98 件のうち pass 64 件、fail 9 件、unsupported 21 件。
残りは対象外 82 件 (上記) と、既存の汎用ブロッカーによる
未実行 4 件 (rustc-env 3 件、proc-macro 補助クレートのビルド
失敗 1 件)。bfail revision は 21 件がクリーンな失敗として
pass し、失敗した 4 件は delayed_span_bug (failure-status 101 の
delayed bug で、クリーンな失敗ではない) と link_order
(補助クレートのリンク順序で決まる成否で、コンパイル成否では
ない) である。fail 9 件の残りはドライバ側の不一致
(lto / thinlto / delegation テストでの E0432、ICE 1 件、
Stable MIR エラー 1 件) で、いずれも正直な不一致として記録する。
unsupported 21 件は単相化 instance 不在・const パラメータ・
InlineAsm など、他スイートと同じ変換側の上限である。

全体実行の指標2(b) は 69.86% (both11 の 65.204% から上昇) だが、
この差の大半は both11 以降に統合された他レーンの増分であり、
本レーン単独の寄与は incremental の pass 0 件 → 64 件である。
both11 との比較で pass から後退した 64 件は本変更の回帰では
ないことを、変更前のランナーを同じドライバで再実行して
全件ステータス一致で確認した (外国ターゲット revision の
未実行化、polonius の変換未完の露呈、false-pass の訂正など、
both11 以降の main 変更によるもの)。survey スクリプトに変更は
ない (incremental は元々 compile-check に分類されており、
除外はランナー側だけにあった)。

---

#### Drop ターミネータ残件の census と修正 (Lane BS、2026-11-21)

both10 時点で理由・バケットが「unsupported terminator: Drop」を
引く 26 件を全件再現し、原因で分類した。

| クラス | 件数 | 原因 |
|---|---|---|
| A. ジェネリックな Drop impl の本体が未収集 | 8 | 名前表には impl があるが単相化された本体が無い。アイテム走査はジェネリック項目を飛ばし、到達可能性収集は Call 辺しか辿らない (drop glue は呼び出し辺ではない) |
| B. モデル化 glue の無い非ローカル ADT | 6 | Result ×3、RefCell ×1、Cow ×1、Rc/Arc ×1 |
| C. glue 計画に Ty::Dynamic の腕が無い | 4 | dyn の drop (Lane BF の値モデルの領域) |
| D. glue 計画に Ty::Closure の腕が無い | 3 | クロージャのキャプチャが計画されない |
| E. 実行時サポート builtin の型タグ表 | 5 | needs_drop / size_of_val / Default::default (builtins.rs 側の別系統) |

修正 (ブランチ lane/drop-remaining、コミット 88b9020 / 8199fad /
a0b938e):

- クラス A: ドライバに drop 要求パスを追加した。変換済み本体の
  Drop ターミネータを走査し、落とされる place の型を計算して
  (Field / Deref / Index / Downcast を辿る) 型ツリーを歩き、
  ローカルクレートの ADT 候補ごとに Drop impl メソッドを ADT の
  ジェネリック引数で Instance::resolve し、ローカル項目と同じ
  変換で本体化して到達可能性の種に渡す。不動点まで反復
  (上限 256)、候補ごとに catch_unwind で分離する。
- クラス A の後続: mir_convert の dtor_body が、ジェネリック
  引数を builtins::ty_tag で描いた完全な instance キーを最初に
  試すようになった。同一ジェネリック impl の単相化が2つ共存する
  形 (issue-24805) で従来の「唯一候補」フォールバックが諦めて
  いた。
- クラス B の一部: std::result::Result に Option と同じ形の
  モデル化 glue (Ok / Err とも field 0 のペイロード)。
- クラス D: クロージャをキャプチャタプル (最後のジェネリック
  引数) として計画し、run_drop_glue の Fields の腕が
  Value::Closure を受けるようにした。部分的にムーブされた
  キャプチャは既存の Uninit 読み経路で飛ばす。

検証: ワークスペース 336 件全パス (このクローンのベース時点の
件数)、ドライバ 41 件全パス、コーパス 46/46 (このベースには
Lane BL が main へ足した int128_ops がまだ無い)。新規テストは
interp の drop_glue_tests が 10 → 14 件、ドライバの driver_run
が 24 → 26 件。

combined 実測 (subset-bs.json、ベースは Lane BH 統合時点の
6619de9 で both11 のベース 379e0ac より古いため、見出しの
合計は直接比較できない。参考値: 全体 62.738%):

- Drop バケットは both11 の 54 件から 19 件へ。both11 の 54 件の
  行き先は、pass 6 件、Drop の拒否が消えて後段の「変換された
  本体が無い関数への呼び出し」へ移行 39 件、Drop のまま 8 件、
  実行されるが fail 1 件。
- 新規 pass 6 件: drop-trait-generic、dropck_traits、
  issue-24805-dropck-itemless、yeet-for-result、
  partial-move-drop-order、moved-upvar-mut-rebind-11958。
- fail 化した 1 件 generic-drop-trait-bound-15858 は、デストラクタ
  が static mut へ書く形で、残るブロッカーは Drop ではなく
  statics の意味論 (別系統)。
- Drop のまま残る 8 件は、クラス C の dyn ×4、RefCell
  (generic-recursive-list-struct と drop-order-comparisons-
  let-chains)、Cow ×1、Rc/Arc ×1。
- bs 側で新たに Drop バケットへ入った 11 件のうち 3 件
  (issue-30018-nopanic、issue-90752、issue-90752-raw-ptr-
  shenanigans) は both11 では別の理由で unsupported であり、
  ベース差である。残り 8 件は下記の回帰に含まれる。

**回帰 (統合前に要修正)**: ドライバの drop 要求パスは、この
レーンのベース (6619de9) で pass する 9 件を「unsupported
terminator: Drop」(dropflag-reinit-in-loop は mem::drop の
glue を会計できない形) に落とす。ベースのドライババイナリと
レーンのバイナリで同一テストを直接実行して切り分けた実測で、
main.rs の要求パスだけを戻すと全件 pass に戻る。対象は
drop-scope-exit、issue-2735-2、issue-2735-3、issue-979、
dropflag-reinit-in-loop、issue-45696-long-live-borrows-in-boxes、
init-res-into-things、resource-in-struct、
newtype-struct-drop-run で、いずれも通常の (ジェネリックでない、
またはライフタイムのみの) ローカル ADT を落とすテストである。
interp 側の変更 (dtor キー、Result / クロージャ glue) は
単独では再現しない。

**回帰の原因と修正 (コミット 2323538 / 365d24f)**: 原因は
デストラクタ登録キーの「綴り」の不一致だった。ライフタイム
のみをパラメータに持つ Drop impl (例: drop-scope-exit の
shrinky_pointer<'a>) は、アイテム走査では宣言どおりの綴り
`<...shrinky_pointer<'a> as std::ops::Drop>::drop` で登録
される。要求パスは同じ instance を Instance::resolve で
解決し直すが、その名前は消去されたライフタイムを `'_` と
描く (`<...shrinky_pointer<'_> as std::ops::Drop>::drop`)。
要求パスの重複チェックは名前の文字列一致だったため別物と
判定し、同じデストラクタを2件登録した。mir_convert の
dtor_body は、完全キーで引けない場合に「唯一の候補」を探す
フォールバックを持ち、候補が2件あると曖昧として None を
返すため、その型の Drop がすべて unsupported になっていた。
修正は要求パス側で、名前ではなく DefId (instance_def_id が
返し、変換済み本体と対で記録される識別子) が既に変換済み
なら要求を飛ばすようにしたこと。名前チェックは第二の
ガードとして残してある。回帰テストとして、ライフタイム
パラメータ付き Drop impl をスコープ終了で落とすドライバ
テスト (run_mode_lifetime_parameterized_drop_impl_runs、
driver_run 26 → 27 件) を追加した。

修正後の再実測 (subset-bs2.json、同じベース 6619de9):
Drop バケットは 19 件 → 7 件。回帰した 9 件は全件 pass に
戻り、subset-bs で pass だったものの悪化は 0 件、改善は
ちょうど 9 件だった。4 件 (drop-order-comparisons-let-
chains、issue-30018-nopanic、issue-90752、issue-90752-raw-
ptr-shenanigans) は Drop の拒否が消えて後段の「変換された
本体が無い関数への呼び出し」へ移行した。残る 7 件は
dyn の drop など下記の既知の残件クラスである。検証は
ワークスペース 336 件全パス、ドライバ 42 件全パス、
コーパス 46/46。

残件: クラス C (dyn の drop)、RefCell / Cow / Rc・Arc の
glue、クラス E (builtin の型タグ表)、aux クレートのジェネリック
なデストラクタ、static mut へのデストラクタからの書き込み、
union 代入の drop 意味論。

---

#### JIT の6ワード超引数 (Lane CM、2026-11-21)

JIT の呼び出しは6ワードまで (System V の引数レジスタ分) しか渡せず、7ワード以上の呼び出しと7パラメータ以上の関数は明示エラーだった。これを System V のスタック渡しで解消した。

- 呼び出し側 (compiler.rs の `Instr::Call`): 7ワード目以降を逆順に push し (最初のスタックワードが最も低いアドレスに来る)、スタックワード数が奇数のときは引数の上に pad を1つ push して、`call` 時点の rsp の16バイトアラインメントを保つ (関数本体では frame サイズが16の倍数のため rsp は16アラインされている)。push は呼び出し直後に巻き戻すので、後続の CheckPanic の `leave` は通常の frame だけを見る。
- 呼び出され側 (プロローグ): 7ワード目以降を標準 frame レイアウトの `[rbp + 16 + 8*k]` から読む (`[rbp]` は保存された rbp、`[rbp + 8]` は戻りアドレス)。
- Rust 側の入り口 (program.rs): 固定6引数の transmute を、naked なトランポリン `call_entry` に置き換えた。レジスタとスタックの両方を任意のワード数で組み立て、2ワード戻り値 (RAX/RDX) をそのまま返す。
- ワード配置の契約 (mir_jit.rs の `arg_word_layout`): 呼び出し側と呼び出され側が被呼関数のパラメータ型から同じ配置を計算する。unit 引数はダミーの1ワードを占める (呼び出し側も実際に渡すよう統一した。従来は呼び出し側だけが省略しており、JitProgram::call の文書化された契約ともずれていた)。2ワード引数 (fat 参照・2スカラのペア) が最後のレジスタワード (ワード5) から始まるときは前に pad ワードを1つ入れ、ペアをレジスタ/スタック境界で分割しない。
- テスト: バックエンド単体 (引数ごとに異なる重みの7/8/10ワード呼び出し、多数引数のネスト呼び出し、8ワード呼び出しのペア戻り値) と MIR レベル二重エンジン (8スカラ呼び出し、fat 参照を丸ごとスタックに載せる形、境界にまたがる fat 参照 (pad ワード入り) + 範囲外 panic の一致、7引数のペア戻り値、unit 引数のダミーワード)。ワークスペース 430 テスト全パス、コーパス 47/47。
- 依然として未対応: TailCall、単一引数で3ワード以上の値、ペアでない3ワード以上の戻り値。

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

### 指標2(b) procedure カテゴリの実測 (2026-11-21)

`scripts/rustc-subset-runmake.py` による実測。procedure カテゴリ
(run-make 系 551 件) のテストはプログラムではなくレシピ
(`rmake.rs`) で、レシピ自身の手順と検証が合否を決める。
Sparagmos の主経路が判定できるのはプログラムの実行だけなので、
レシピの合否が「単一プログラムのコンパイルと実行」に帰着する
ものだけを実測対象とし、それ以外は理由を付けて未実行に分類
した (未実行は fail として数え、母数から外していない)。

分類はレシピのソースに対する機械的な判定で、551 件全てに
理由が付く。実測対象 (reducible) は次の全てを満たすレシピで
ある: ヘッダ条件がこの環境 (x86_64-unknown-linux-gnu、ホストと
ターゲットが同一) で成立する、`rustc()` ビルダによるコンパイル
が 1 回だけで、使われるオプションが input / output / edition /
crate_type("bin") の範囲内である、入力がテストディレクトリ内の
単一の .rs ファイルである、生成されたバイナリが引数なしで
1 回だけ実行される (標準出力に対するリテラル検証が連鎖して
いる場合はそれも判定に含める)。

判定は stage 1 と同じ基準で、テストディレクトリのコピー上で
ネイティブ (rustc 1.100.0、レシピの edition、既定は 2015) の
コンパイルと実行を行い、レシピの期待 (コンパイル成功、終了
コード 0、標準出力検証) をネイティブ側が満たすことを確認した
うえで、ドライバ実行の終了コードと標準出力がネイティブと
一致すれば pass とする。

分類結果 (551 件):

| 分類 | 件数 | 内容 |
|---|---|---|
| reducible (実測対象) | 2 | 上記の形に帰着する |
| external-tools | 173 | レシピが外部ツールを駆動する (rustdoc 59、LLVM ツール 43、C/C++ ライブラリのビルド 36、cc 22、cargo 9、汎用コマンド 4) |
| not-single-program | 166 | コンパイルが複数回 145、コンパイルなし 11、bare_rustc / rustc_minicore の使用 7、`--extern` によるクレート連結 3 |
| out-of-environment | 110 | ヘッダ条件がこの環境で成立しない (needs-enzyme 38、他のプラットフォーム限定、needs-force-clang-based-tests 6 など) |
| compile-verdict-only | 67 | バイナリが実行されず、コンパイルの成否やコンパイラ出力が合否を決める (コンパイル失敗が期待されるもの 15 を含む) |
| unevaluatable-gate | 21 | ヘッダ条件を機械的に評価できない (needs-llvm-components 18 など) |
| recipe-shape | 7 | 文の数が多い、制御構造があるなど、上記の形に当てはまらない |
| builder-options | 3 | ドライバで再現できないビルダオプション (引数、最適化指定、stdin 入力など) を使用 |
| artifact-checks | 2 | バイナリの実行に加えて生成ファイルの diff が合否に入る |

実測結果: 対象 2 件のうち pass 1 件 (bare-outfile)、
unsupported 1 件 (forced-unwind-terminate-pof: `#![no_main]` の
C ABI の main と pthread FFI をドライバが未対応と返した)。
カテゴリのパス率は 1 / 551 = 0.18%。

このカテゴリの合否の大半は、プログラムの実行ではなくビルド
手順 (複数クレートの連結、外部ツール、生成物の検証) によって
決まる。主経路の実行で判定できる範囲は上記の形に限られ、
それ以外の 549 件は未実行として fail に数えている。

測定条件: 処理時間は全体で約 1 秒。判定のタイムアウトは
ネイティブのコンパイル 60 秒、ネイティブの実行 30 秒、
ドライバ実行 60 秒 (1 件あたり)。テストごとの結果は結果 JSON
に機械可読な形で残る。再実行は
`python3 scripts/rustc-subset-runmake.py --survey-json <survey JSON> --rust-dir <チェックアウト> --driver-bin <ドライバ> --out <結果 JSON>`。

### 指標2(b) exec-debugger カテゴリの実測 (2026-11-21)

`scripts/rustc-subset-debugger.py` による分類と実測。exec-debugger
カテゴリ (debuginfo スイート 170 件) の合否は、デバッガの
セッションで決まる。compiletest はテストファイルを
`//@ compile-flags: -g` でコンパイルしてデバッグ情報を持つ
ネイティブバイナリを作り、ファイル内の `//@ <dbg>-command:` 行と
`#break` 行から組み立てたスクリプトで GDB / LLDB / CDB
(Windows のデバッガ) の下で実行し、`//@ <dbg>-check:` 行が
デバッガの標準出力に順序通り現れるかを照合する。`<dbg>-repr:` は
debugger_tester が横取りする擬似コマンドで、暗黙の照合行も生む。
照合の対象は、バイナリのデバッグ情報をデバッガが描いた
プログラムの状態であり、rustc の pretty printer や
debugger_tester のスクリプト (src/etc) の振る舞いも含まれる。

Sparagmos の主経路はネイティブバイナリもデバッグ情報も
生成せず、デバッガのインターフェースも持たない。したがって
このカテゴリで主経路が判定できるテストは無く、パスは
0 / 170 = 0% である。170 件全てを未実行として fail に数え、
母数から外していない。以下は全件の機械的な分類である。

分類結果 (170 件):

| 観点 | 内訳 |
|---|---|
| 合否の形 | デバッガ出力の照合 158、デバッガセッションの完走のみ 3、コンパイルとロードのみ 9 |
| デバッガのスクリプトの組合せ | gdb+lldb 89、gdb のみ 28、gdb+lldb+cdb 15、cdb のみ 17、cdb+gdb 7、lldb のみ 4、cdb+lldb 1、なし 9 |
| スクリプトを持つテスト | gdb 139、lldb 109、cdb 40 |
| 照合行 / コマンド行 | 照合 3,054 行 (repr 由来の暗黙の照合 37 を含む)、コマンド 3,634 行 |
| 追加の依存 | rustc の GDB pretty printer を使う 40、pretty printer を無効化 105、repr 擬似コマンド 2、should-fail 1、ignore-test 1、only-cdb 7 |
| プログラムの形 | 単一ファイル 165、補助ファイル必須 5、FFI 2、複数リビジョン 14 |

参考実測として、合否とは別に、対象プログラム自体が主経路で
実行できるかを測った。対象は、単一ファイルでリビジョン指定が
無く、この環境 (x86_64-unknown-linux-gnu) に適用でき、
compile-flags がデバッグ情報の指定 (-g、`-C debuginfo` 系、
`--crate-name`) の範囲に収まる 118 件で、ドライバの run
モードで実行した (主経路はデバッグ情報を生成しないため、
これらのフラグはドライバの動作を変えない)。結果は完走 38 件、
panic 1 件、unsupported 79 件で、ドライバの異常終了と
タイムアウトは 0 件だった。未実行の 52 件は、リビジョン指定
(宣言またはスコープ付き指令) 31、only-cdb 7、複数ファイル 5、
only-msvc 3、ignore-test 1、デバッグ情報以外の compile-flags
5 である。unsupported の主因は主経路側の通常の未対応で、
BoxDerefTransmute キャスト 15 件、本体の変換が無い関数への
呼び出し (std::process::exit など)、Drop ターミネータ、
Transmute キャスト、f16 定数が上位に並ぶ。この実測は
プログラム自体の実行可否を示すだけであり、完走した
プログラムにもデバッガのセッションと照合対象の出力は
存在しないため、パスには数えない。

測定条件: ドライバ実行のタイムアウトは 1 件あたり 60 秒、
並列 48。テストごとの結果は結果 JSON に機械可読な形で残る。
再実行は
`python3 scripts/rustc-subset-debugger.py --survey-json <survey JSON> --rust-dir <チェックアウト> --driver-bin <ドライバ> --out <結果 JSON>`。

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
- 実行可能 (当初の 906 集合): **577 / 906 = 63.7%**。うち変換済み
  本体による解決が 559 件、builtin による解決が 18 件。Lane AG で
  ドライバが到達可能な std/core 関数本体を MirProgram に収集する
  ようになり、本体による解決が 0 件から 559 件になった。
- 母数の再抽出: 本体収集により std 本体内部の呼び出し先も出現
  集合に入るため、抽出スクリプトを再実行すると母数は **1,262** に
  増え、実行可能は **719 / 1,262 = 57.0%** になる (当初の 906 は
  全て包含される)。見出しの値は再抽出後の母数で測った 57.0% を
  使う。

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

### RefCell モデル (Lane CC, 2026-11-21)

`std::cell::RefCell` の借用フラグ規律をインタプリタにモデル化した。
これまで `RefCell::new` は変換できたが `borrow` / `borrow_mut` は
本体の MIR に `Operand::RuntimeChecks(UbChecks)` を含むため
「no converted body」で落ちていた (both8 時点のセンサスで
borrow 8 件・borrow_mut 5 件が主ブロッカー)。

設計:

- 自然な値形状をそのまま使う: `RefCell{ field0 = Cell{UnsafeCell{isize フラグ}}, field1 = UnsafeCell{T} }`。
  フラグ経路は [F0,F0,F0]、値経路は [F1,F0]。フラグは 0 = 未使用、
  -1 = 書き込み中、正数 n = 共有借用 n 件。`RefCell::new` の
  ビルトインがこの形状を構築するので、RefCell のフィールドを
  射影する収集済み本体とも共存する。
- ビルトイン: `new` / `borrow` / `borrow_mut` / `try_borrow` /
  `try_borrow_mut` / `get_mut` / `replace` / `as_ptr`。
  ガード値 (`Ref` / `RefMut`) は [値への参照, フラグセルへの参照]
  の2フィールド集約で、`Deref` / `DerefMut` ビルトインに
  `std::cell::Ref` / `RefMut` のアームを追加し第0フィールドを返す。
- 競合時のパニック文言は Rust 1.100.0 で実測したものを使う:
  借用中の `borrow_mut` → `RefCell already borrowed`、
  書き込み中の `borrow` → `RefCell already mutably borrowed`、
  書き込み中の `borrow_mut` → `RefCell already borrowed`。
- Drop 連携: 新しい `DropPlan::RefCellRelease { mutable }` を
  `std::cell::Ref` / `RefMut` の glue として生成し、ガードの
  Drop でフラグを解放する (共有は減算、可変は 0 にリセット)。
  `RefCell` 自体の glue は値経路 [F1,F0] へ包んだ値の glue を
  適用する。`try_borrow` の結果を落とせるよう、`Result` の
  Variants プラン (Ok 側は包んだ値のプラン、Err 側は Trivial) と
  `BorrowError` / `BorrowMutError` のフィールドプランも追加した。

Drop 連携の検証結果 (意味テスト16件 + ドライバ e2e 4件で確認):

- スコープ終端の Drop で解放され、後続の `borrow_mut` が成功する。
- `std::mem::drop(guard)` でも解放される。
- ガードを消費した呼び出し先関数のフレーム終端 Drop が、
  呼び出し元のセルの借用を解放する (フレームまたぎ)。
- `RefCell<D>` の Drop で D のデストラクタが値経路を通して走る。
- `Result<Ref>` の Drop (try_borrow の Err/Ok 両方) が含まれ
  るガードを解放する。

実測効果: both11 の結果 JSON で RefCell に言及する12件
(うち11件は borrow / borrow_mut が記録上の主ブロッカー) を
本レーンのドライバで再測定したところ、2件が pass に転じ
(issue-23338-params-outlive-temps-of-body、
refcell-borrow-comparison-12033)、9件は RefCell のブロッカーを
越えて無関係な後段のブロッカー (Iterator::for_each、別クレート
需要、Drop ターミネータの形状、Index ビルトイン、値等価性、
Vec::drain 等) へ前進した。issue-63787 は符号文脈を失った
定数由来のフラグ (Uint) を許容する修正でフラグ読み取りを
越えたが、-O 形状の別の代入不一致で依然 unsupported。
検証数 (本レーンのベース bcce754 上): ワークスペース
398 passed / 0 failed、ドライバ 55 passed / 0 failed、
コーパス 47/47。

明示的に未対応のまま (名前付きエラーで落とす):
`Ref::map` / `RefMut::map` / `Ref::clone`、および定数集約として
のみ存在する RefCell の借用 (ライブな場所が無くフラグの
置き場が無いため)。

### generic コンストラクタの instance 化 (Lane CF, 2026-11-21)

generic なタプル構造体 / タプルバリアントのコンストラクタ
(`ItemKind::Ctor(CtorKind::Fn)`) はアイテムとしては単相化
instance を持たず、アイテムループで一律に
「no monomorphic instance available (kind=Ctor(Fn))」として
スキップ計上されていた (both12 で 791 件 / 497 テスト)。
しかし直接の構築は呼び出し側の集約 rvalue として現れるため、
コンストラクタ本体が無くても実行はできており、計上だけが
残っていた。一方、値として使う場合 (`let f = S; f(1, 2)`、
高階関数への引き渡し、定数への格納) は FnDef 定数の指す
instance 本体がどこにも変換されず、実行も exit 2 で落ちていた。

対応 (ドライバのみ。インタプリタ側の変更は不要だった —
需要で変換された合成本体が、FnDef 値経由の呼び出し解決が
探す instance キーとそのまま一致する):

- アイテムループ: Ctor(Fn) アイテムを generic 関数と同じく
  遅延 (deferred) にし、需要パスで instance 単位に解決する。
- 需要スキャン: 呼び出し終端の FnDef ターゲットに加え、
  遅延コンストラクタについては本体が実体化する全ての
  FnDef 定数を走査する (`reach::fndef_constants`)。値としての
  使用は需要元の本体では Call 終端に現れないためである。
  通常関数の値としての使用は従来どおり Call ターゲットのみ
  (本レーンの範囲外)。
- 計上: 何にも需要されないコンストラクタは「本体なしで
  変換済み」とする。直接の使用は呼び出し側の集約 rvalue
  として既に変換されており、需要されなかった単相化には
  変換すべき本体が存在しない (自明な定数と同じ扱い)。
  需要された instance が変換に失敗した場合は従来どおり
  理由付きでスキップのままになる。

実測 (subset-cf.json、ベース both12 = main 87d8acd):

- Ctor(Fn) スキップ: **791 件 → 0 件**。
- 「単相化 instance 無し」主バケット: 1,375 件 → 1,162 件。
- 指標2(b): 69.936% → **71.111%** (17,733/24,937、+293 件)。
  内訳は全てコンパイル時検証系 (15,226 → 15,519、83.057% →
  84.655%)。実行系は 2,214 件で不変 — 実行系テストでは
  コンストラクタが単独のブロッカーになっていなかった。
  both12 との比較で pass から後退したテストは **0 件**。
- kind=Fn スキップは 3,190 件 → 3,215 件と微増したが、
  これは Ctor スキップで早期に止まっていたテストの変換が
  先に進み、残りの Fn スキップが数えられるようになった
  ためで、テストの成否の後退ではない。

残件: 外部クレートのコンストラクタを値として使う場合
(例: `let f = Some;`) は到達収集が FnDef 定数を辿らない
ため依然として実行できない (本体なし呼び出し側の問題で、
本レーンの対象であるローカルアイテムの計上とは別)。

---

## 測定の規律

- 測定は fleur 上で、ツールチェーンとコマンドを併記して記録する。
- harness による自動集計が整うまでは、手動測定の結果を日付付きで本書に追記する。
- 指標をまたいだ合算値 (例: 3指標の平均) を公表しない。
- 母数を変更したときは、変更理由と旧母数を本書の履歴に残す。

## 現在の測定状態 (2026-11-21)

| 指標 | 母数 | 現在値 |
|---|---|---|
| 指標1 MIR バリアント網羅率 | 40 (確定) | 67.5% (27/40, 2026-11-21 実測) |
| 副指標1b サブ enum | 上表の通り (確定) | 未測定 (BinOp の出現実測 15/26 のみ) |
| 指標2(a) 自前コーパス | 47 (tests/corpus) | 100.0% (47/47, 2026-11-21 実測) |
| 指標2(b) rustc テストサブセット | 24,937 (2026-11-20 機械抽出) | 69.936% (17,440/24,937, 2026-11-21 combined 単一ラン実測 both12: exec 2,214、コンパイル時検証系 15,226。この実測の後に Lane CL が統合済みで、増分は次の combined 実測で確定する。コンパイル時検証系は成否レベルの一致で、照合系の扱いは stage 2 の節に記載) |
| 指標3 std 機能網羅率 | 1,435 (2026-11-21 再抽出) | 64.3% (923/1,435, 2026-11-21 both12 結果で再実測。strict 値 889) |

指標2(a) の母数変更履歴: 2026-11-21 に 15 件から 36 件へ拡大した
(Issue #28)。追加した 21 件は、制御フロー (ラベル付きループ、match
ガード、相互再帰、早期リターン、while let)、データ (タプル、配列の
諸形式、構造体、enum ペイロード、Option、&mut 参照)、演算子
(ビット演算、キャスト、複合代入、符号付き除算・剰余) の観点で、
いずれも interp 経路で native と stdout・終了コードが一致する
ことを確認済みのプログラムである。

2026-11-21 に 36 件から 45 件へ拡大した (Issue #44)。追加した
9 件は float の観点 (四則演算、比較と NaN・無限大、float と整数の
キャスト、f32 と相互変換、整形 (精度指定を含む)、ループでの累積、
否定と剰余、タプル・構造体内の float) で、いずれも interp 経路で
native と stdout・終了コードが一致することを確認済みのプログラム
である。JIT 経路のコーパス実測は 4/45 で、通るのは silent 系の
4 件のみである。float のケースは、出力経路が JIT の対応範囲外の
ため unsupported になる (float の演算自体は JIT のテストで検証
済み)。

