# LoRA管理 (LoRA Management)

LoRA の適用、キー単位の強度調整、複数 LoRA のランダム化・合成、量子化を行うノード群です。

---

## 1. LoRA Loader Elemental
LoRA 内のキーに対して検索ルールを指定し、該当テンソルごとに強度を変更するローダーです。

### 主な設定
- `lora_name`: 読み込む LoRA。
- `strength_model`, `strength_clip`: model / clip へ適用する全体強度。
- `lora_strength_string`: 1行1指定の強度ルール。
- `match_mode`: `prefix`, `contains`, `regex` から検索方式を選択。
- `remove_unspecified_keys`: 指定に一致しないキーを処理済み LoRA から除外します。
- `remove_zero_strength_keys`: 強度0の指定キーを処理済み LoRA から除外します。
- `save_lora`, `save_name`: 処理済み LoRA を保存します。

### ルール形式
```text
検索キー = 強度
```

例:
```text
lora_unet_input_blocks_4 = 0.5
lora_unet_output_blocks_0 = 1.2
```

### 強度計算
指定強度は LoRA の実効差分に対して線形に近くなるよう、内部では `lora_up/down` に `sqrt(abs(strength))` を掛けます。負値の場合は符号を `lora_up.weight` 側に寄せます。

### 出力
- `model`, `clip`: LoRA 適用後の model / clip。
- `processed_lora`: 処理済み LoRA 辞書。
- `metadata`: LoRA のメタデータ JSON。
- `lora_keys`: ルール指定に使いやすい候補キー一覧。

---

## 2. LoRA Loader Elemental UI
`LoRA Loader Elemental` と同じ処理を、ノード内 UI で編集できる版です。

### UI
- `List`: 検索キー、強度、ON/OFF を行単位で編集します。
- `EQ`: 縦スライダー風に強度を編集します。
- `Graph`: ガイド線付きグラフで強度を編集します。

### 操作
- `+`: ルールを追加。
- `0` / `.5` / `1`: 全ルールの強度を一括設定。
- `On` / `Off`: 全ルールを一括有効化/無効化。
- `Import`: `pattern = strength` 形式のテキストを読み込み。
- `Copy`: 現在のルールをテキストとしてコピー。
- `Draw`: Graph表示で複数項目をドラッグ編集。
- `Snap`: Graph表示で5%ガイド線にスナップ。

### プリセット
SDXL / SDXL Full / FLUX / Z-Image / Wan / Qwen / Anima 向けの検索キーセットを選択できます。

### 注意
OFF は「その検索ルールを出力しない」扱いです。`remove_unspecified_keys=False` の場合、未指定キーは元の LoRA のまま残ります。対象キーを完全に消したい場合は、強度0や削除系オプションと組み合わせてください。

---

## 3. LoRA Selector
複数の LoRA 候補から1つを選び、指定回数だけ使ってから次の LoRA に切り替えるノードです。

### 主な用途
- 大量の LoRA を順番に試す。
- ランダムに LoRA を選んで比較する。
- フォルダ単位で候補を切り替える。

### 主な設定
- スロット入力: 複数の LoRA 候補を直接指定。
- `lora_folder`: フォルダ内の LoRA を候補に追加。
- `mode`: ランダムまたは round-robin。
- `switch_interval`: 選択した LoRA を何回使うか。
- `seed`: ランダム選択用。
- `cache_limit_gb`: 読み込んだ LoRA データのキャッシュ上限。

### 出力
- `MODEL`, `CLIP`: LoRA 適用後、または未適用時の入力。
- `selected_lora_name`: 選択された LoRA 名。
- `lora_index`: 選択インデックス。
- `source_type`: スロット由来かフォルダ由来かを示す文字列。

---

## 4. LoRA Weight Randomizer
最大8個の LoRA を対象に、合計強度を一定に保ちながら各 LoRA の強度をランダム配分します。

### 主な設定
- `total_strength`: 配分する合計強度。
- `max_single_strength`: 1つの LoRA に割り当てる最大強度。
- `randomize_total_strength`: 合計強度自体も 0 から `total_strength` の範囲でランダム化します。
- `seed`: 配分の乱数シード。

### 出力
- `MODEL`, `CLIP`: LoRA 適用後の model / clip。
- `settings`: 適用された LoRA と強度の一覧。

---

## 5. LoRA Mixer Elemental
最大8個の LoRA からキー単位でテンソルを選び、新しい混合 LoRA を作ります。

### 主な設定
- `key_selection`: 全キーを対象にするか、共通キーのみを使うか。
- `key_strength_randomization`: キーごとに強度をランダム化します。
- `multi_mix`: 複数パスで混合します。
- `save_mixed_lora`, `save_name`: 混合 LoRA を保存します。

### 出力
- `mixed_lora`: 混合済み LoRA。
- `lora_keys`: 最初の混合結果のキー一覧。
- `all_lora_keys`: 混合パス全体のキー情報。

---

## 6. Quantized LoRA Loader
LoRA テンソルを指定 bit 数へ量子化してから適用します。

### 主な設定
- `quantization_bits`: 量子化 bit 数。
- `quantization_iterations`: 量子化処理の反復回数。
- `stepwise_quantization`: 段階的に量子化します。
- `quantization_step_size`: 段階的量子化時のステップ幅。
- `blend_mode`, `blend_factor`: 元テンソルと量子化テンソルをブレンドします。
- `save_quantized_lora`, `save_name`: 量子化 LoRA を保存します。

### 出力
- `quantized_lora`: 量子化済み LoRA。
- `metadata`: 元 LoRA のメタデータ。

### 注意
低 bit 量子化では品質劣化が起きやすくなります。`save_quantized_lora` を使う場合は、ファイル名と設定を確認してください。
