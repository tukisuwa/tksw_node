# データ読み込み (Data Loading)

このカテゴリのノードは、外部のフォルダから画像やテキストデータを組織的に読み込むための機能を提供します。連番データの処理や、条件に応じた動的な読み込みが可能です。

---

## 1. Image Sequence Loader
フォルダ内の画像をソートされた順序で1枚ずつ読み込みます。
画像フォルダを順番に処理する一番基本的なローダーで、ファイルリストは `reset` または `folder_path` 変更時に再読み込みされます。

### 主な設定
- `folder_path`: 画像が含まれるフォルダのパス。
- `reset`: Trueにするとインデックスを0に戻し、ファイルリストを再読み込みします。
- `loop_or_reset`: フォルダ内の画像をすべて読み終えた後の動作。Trueで最初に戻ります。
- `reset_on_error`: 読み込み失敗時にファイルリストを再読み込みして先頭から再開します。
- `exclude_loaded_on_reset`: `reset_on_error` と併用し、すでに通過したファイルを再読み込み後の候補から除外します。
- `start_index`: 先頭扱いにするオフセット。
- `use_manual_index`: Trueの場合、実行時に自動で進むのではなく `manual_index` で指定した箇所の画像を読み込みます。
- `output_alpha`: Trueの場合はアルファチャンネルを保持します。FalseではRGBに変換します。
- `include_extension`: `filename` 出力に拡張子を含めるかどうか。
- `seed`: 出力にも返されるシード値。現状の順次読み込み自体は基本的にインデックスで決まります。

### 出力
- `IMAGE`: 読み込まれた画像。
- `index`: 現在のインデックス。
- `seed`: 入力されたシード値。
- `filename`: 拡張子を除いた、または `include_extension` に応じたファイル名。

### 注意
サブフォルダは探索しません。サブフォルダも含めて扱いたい場合は `Image Loader Seed Sync` を使います。

---

## 2. Image Pair Sequence Loader
2つのフォルダから、ファイル名が一致（または順序が一致）する画像のペアを同時に読み込みます。
ControlNet用の画像とマスク、あるいはソース画像とターゲット画像の読み込みに便利です。

### 主な設定
- `folder_path_A`, `folder_path_B`: それぞれの画像フォルダ。
- `match_extension`: 拡張子まで含めて完全一致させるかどうか。Falseの場合はベース名が一致すればペア扱いします。
- `start_index`: 共通ファイルリスト内での開始位置。
- `loop_or_reset`, `reset_on_error`, `exclude_loaded_on_reset`: `Image Sequence Loader` と同様の順次処理制御。
- `output_alpha`, `include_extension`: 画像変換とファイル名出力の制御。

### 出力
- `image_A`, `image_B`: ペア画像。
- `index`: 共通ファイルリスト上の現在位置。
- `filename`: A側基準のファイル名。

### 注意
`folder_path_B` が空の場合は A と同じフォルダを使います。拡張子違いのペアを扱う場合は、A/B のベース名対応を前提にします。

---

## 3. Image Text Pair Sequence Loader
ComfyUI上の表示名は **Image TextPair SequenceLoader** です。

画像フォルダとテキストフォルダから、同じベース名（ファイル名）を持つペアを読み込みます。
画像のキャプションを同時に読み込み、プロンプトとして利用するワークフローに適しています。

### 主な設定
- `image_folder_path`, `text_folder_path`: 画像と `.txt` ファイルのフォルダ。
- `start_index`: ペアリスト内の開始位置。
- `loop_or_reset`: 末尾到達時に先頭へ戻るかどうか。Falseの場合は末尾で空出力になります。
- `reset_on_error`: 画像またはテキストの読み込みに失敗した場合にリストを再読み込みします。
- `exclude_loaded_on_reset`: エラーリセット時に処理済みペアを除外します。
- `output_alpha`, `include_extension`: 画像変換とファイル名出力の制御。

### 出力
- `image`: 読み込んだ画像。
- `text`: 対応するテキストファイルの全文。
- `index`: ペアリスト上の現在位置。
- `filename`: ペアのベース名、または `include_extension` 有効時は画像ファイル名。

### 注意
対応判定は拡張子を除いたベース名で行います。画像だけ、またはテキストだけ存在するファイルはスキップされます。

---

## 4. Image Loader Seed Sync
シード値（Seed）に基づいて読み込む画像を選択します。
`index = seed % 画像数` となり、特定の生成結果と画像を固定して紐づける場合に非常に有効です。

### 主な設定
- `folder_path`: 画像フォルダ。
- `seed`: 画像選択に使うシード値。
- `reset`: Trueにすると画像リストを再読み込みします。
- `sort_method`: `natural`（自然順：1, 2, 10...）か `alphabetical`（辞書順：1, 10, 2...）を選択できます。
- `subfolder_depth`: サブフォルダ探索の深さ。`0` は指定フォルダ直下のみ、`1` なら1階層下まで探索します。
- `output_path_mode`: `filename_only` はファイル名のみ、`relative_path` はサブフォルダを含む相対パスを出力します。
- `output_alpha`: Trueの場合はアルファチャンネルを保持します。
- `include_extension`: `filename` 出力に拡張子を含めます。

### 出力
- `image`: 選択された画像。
- `index`: `seed % 画像数` で決まる画像インデックス。
- `loop_count`: `seed // 画像数`。同じ画像リストを何周分進んだシードかを表します。
- `seed`: 入力シード。
- `filename`: ファイル名、または `relative_path` 指定時の相対パス。

### 使いどころ
画像選択をランダムではなく seed に完全同期したい場合に使います。例えば、生成 seed と参照画像・素材画像を固定対応させることで、同じ seed では常に同じ素材を選ぶワークフローを作れます。

### 注意
画像リストの順序が変わると、同じ seed でも選ばれる画像が変わります。再現性を重視する場合は、フォルダ内容、`sort_method`、`subfolder_depth` を固定してください。

---

## 5. Text File Selector
フォルダ内の `.txt` ファイルから1つを選び、本文とファイル名を出力します。

### 主な設定
- `folder_path`: テキストファイルフォルダ。
- `mode`: `random` または `round-robin`。
- `seed`: `random` 選択時の乱数シード。
- `reset_state`: キャッシュと round-robin の位置をリセットします。
- `cache_chunk_size`: 1回の実行で先読みキャッシュするファイル数。`0` で先読みなし。
- `encoding`: 読み込みエンコーディング。通常は `utf-8`。
- `filename_filter`: ファイル名に含まれる文字列で候補を絞り込みます。

### 出力
- `text`: 選択されたテキストファイルの内容。
- `filename`: 選択されたファイル名。

### 注意
`filename_filter` は正規表現ではなく単純な部分一致です。フォルダ内容が変わった場合は内部状態がリセットされます。
