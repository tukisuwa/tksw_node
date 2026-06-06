# テキスト・プロンプト処理 (Text & Prompt Processing)

プロンプトやキャプションなどの文字列を結合・整形・ランダム置換するためのノード群です。

---

## 1. Text Combiner
最大4つのテキスト入力を結合し、必要に応じて履歴ログを保持します。

### 主な設定
- `separator`: テキスト結合と分割に使う区切り文字。
- `remember_log`: 結合結果を内部ログへ保存するかどうか。
- `max_log`: 保持する履歴数。
- `allow_duplicate_log`: 同じ結合結果を履歴へ重複登録するかどうか。
- `remove_text`: 各入力から削除する文字列、または正規表現。
- `use_regex`: `remove_text` を正規表現として扱うかどうか。

### 入力
- `text_1` - `text_4`: 結合するテキスト。
- `remove_text`: 削除対象。`use_regex` が False の場合は単純文字列、True の場合は正規表現として扱います。

### 出力
- `text`: 結合後のテキスト。
- `text_log`: 保持中の履歴リスト。
- `recent_text_1` - `recent_text_4`: 新しい順の履歴。
- `oldest_text`: 保持中の最古履歴。

### 注意
履歴はノードインスタンス内のメモリに保持されます。ワークフロー保存用の永続ログではありません。

---

## 2. Text Processor
入力テキストをセグメント単位で分割し、削除・置換ルールを適用して整形します。

### 主な設定
- `segment_separator`: セグメント分割と再結合に使う区切り文字。空文字にすると全文を1つのセグメントとして扱います。
- `remove_patterns`: カンマ区切りの正規表現。マッチ箇所を削除します。
- `replace_specs`: 1行1ルールの置換指定。

### `replace_specs` の形式
```text
置換後文字列, 正規表現1, 正規表現2
```

例:
```text
blue eyes, red eyes, green eyes
short hair, long hair
```

### 出力
- `processed_text`: 削除・置換・区切り整形後のテキスト。

### 注意
`remove_patterns` と `replace_specs` は正規表現として扱われます。意図せず広くマッチするパターンに注意してください。

---

## 3. Random Word Replacer
テキスト内の特定語を、同じグループ内の別語へランダムに置換します。

### 主な設定
- `seed`: 置換選択の乱数シード。
- `replace_specs`: 直接入力する置換グループ。
- `replace_specs_file`: 置換グループを書いたファイルパス。
- `replace_specs_folder`: `.txt` / `.csv` ファイルをまとめて置換候補として読み込むフォルダ。

### `replace_specs` の形式
1行ごとに、同じ意味グループの語をカンマ区切りで指定します。

```text
red, green, blue, yellow
cat, dog, bird
```

この場合、入力に `red` が含まれると、同じ行の `green` / `blue` / `yellow` などからランダムに選ばれて置換されます。

### 出力
- `processed_text`: 置換後のテキスト。

### 注意
置換は単純な部分一致です。単語境界や正規表現による厳密な制御が必要な場合は `Text Processor` を使ってください。
