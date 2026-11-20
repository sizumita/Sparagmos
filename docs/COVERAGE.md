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

1. `sparagmos-parse` がそのバリアントを含むテキスト MIR を受理できる。
2. `sparagmos-interp` が意味論テストを通る。
3. `sparagmos-jit` が意味論テストを通る (M1 時点ではインタプリタのみで
   仮数えせず、JIT 完了まで未実装として扱う。厳しい側に倒す)。

指標1の率 = 実装済みバリアント数 / 40 × 100 (小数第1位まで)。
現在の値: **未測定** (骨格のみで各レーンが実装中のため、0 とも断定しない)。

---

## 指標2: テストパス率

### 母数 (2層)

- **指標2(a) 自前コーパス**: `tests/corpus/` に置かれたプログラムの総数。
  追加・削除は Issue で管理し、母数の変更履歴を残す。
- **指標2(b) rustc テストサブセット**: rust-lang/rust タグ `1.100.0` の
  テストスイートから、以下の条件で機械抽出したテストの総数。
  - 単一の `.rs` ファイルで完結する。
  - 実行して終了コードと標準出力で合否が判定できる (run-pass 系)。
  - 外部サービス・ネットワーク・専用ハードウェアを必要としない。
  母数の実数値は **未測定** であり、M4 で抽出スクリプトと共に確定する。
  推測値は書かない。

### パスの定義

テストは次をすべて満たすとき pass とする。

- ネイティブ実行 (ホスト rustc 1.100.0 でコンパイルして実行) と
  Sparagmos 実行で、標準出力・標準エラー・終了コードが一致する。
- タイムアウト、クラッシュ、不一致はいずれも fail とする。
  「未対応のため実行できない」も fail であり、母数から除外しない。

指標2の率 = pass 数 / 母数 × 100 (層ごとに別々に報告し、混ぜない)。

---

## 指標3: std 機能網羅率

### 母数の定義

「std の全 API」を分母にはしない。分母は、指標2のコーパスとテストサブセットの
テキスト MIR に実際に現れる `core` / `alloc` / `std` の関数パスの集合
(機械抽出してユニーク化したもの) とする。つまり「必要とされた std 機能」
に対する網羅率である。分子は、そのうち Sparagmos 上で正しく実行できた
(指標2のパスに寄与した) 項目数とする。

母数の実数値は **未測定** (コーパスの MIR から harness が抽出する仕組みを
M1〜M3 で整備する)。

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

## 現在の測定状態 (2026-11-20)

| 指標 | 母数 | 現在値 |
|---|---|---|
| 指標1 MIR バリアント網羅率 | 40 (確定) | 未測定 (実装中) |
| 副指標1b サブ enum | 上表の通り (確定) | 未測定 (BinOp の出現実測 15/26 のみ) |
| 指標2(a) 自前コーパス | tests/corpus の実ファイル数 | 未測定 |
| 指標2(b) rustc テストサブセット | 未確定 (M4 で機械抽出) | 未測定 |
| 指標3 std 機能網羅率 | 未確定 (出現ベース抽出を整備中) | 未測定 |
