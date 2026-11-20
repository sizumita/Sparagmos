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
現在の値: **未測定** (骨格のみで各レーンが実装中のため、0 とも断定しない)。

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

結果 (母数 24,937 件):

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
| 実行系 (exec) | 539 / 4,149 | 12.991% |
| 実行系 (procedure) | 0 / 551 | 0% (未実行) |
| 実行系 (exec-debugger) | 0 / 170 | 0% (未実行) |
| コンパイル時検証系 | 0 / 18,332 | 0% (未実行) |
| rustdoc 検証系 | 0 / 1,735 | 0% (未実行) |
| **全体** | **539 / 24,937** | **2.161%** |

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

---

## 指標3: std 機能網羅率

### 母数の定義

「std の全 API」を分母にはしない。分母は、指標2のコーパスとテストサブセットの
対象プログラムの MIR に実際に現れる `core` / `alloc` / `std` の関数パスの集合
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

## 現在の測定状態 (2026-11-21)

| 指標 | 母数 | 現在値 |
|---|---|---|
| 指標1 MIR バリアント網羅率 | 40 (確定) | 未測定 (実装中) |
| 副指標1b サブ enum | 上表の通り (確定) | 未測定 (BinOp の出現実測 15/26 のみ) |
| 指標2(a) 自前コーパス | 36 (tests/corpus) | 100.0% (36/36, 2026-11-21 実測) |
| 指標2(b) rustc テストサブセット | 24,937 (2026-11-20 機械抽出) | 2.161% (539/24,937, 2026-11-21 実測) |
| 指標3 std 機能網羅率 | 未確定 (出現ベース抽出を整備中) | 未測定 |

指標2(a) の母数変更履歴: 2026-11-21 に 15 件から 36 件へ拡大した
(Issue #28)。追加した 21 件は、制御フロー (ラベル付きループ、match
ガード、相互再帰、早期リターン、while let)、データ (タプル、配列の
諸形式、構造体、enum ペイロード、Option、&mut 参照)、演算子
(ビット演算、キャスト、複合代入、符号付き除算・剰余) の観点で、
いずれも interp 経路で native と stdout・終了コードが一致する
ことを確認済みのプログラムである。
