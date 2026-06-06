# ユーティリティ (Utilities)

ワークフローを便利にするための補助的な機能を提供するノード群です。

---

## 1. Image Storage Nodes (Memory)
画像をメモリ（RAM）上に一時的に保存し、後で別の場所で呼び出すことができます。

### 構成ノード
- **Store Image by Number**: 指定した `image_id`（数値）で画像を保存します。
- **Retrieve Image by Number**: `image_id` を指定して、保存された画像を呼び出します。
- **Store Multiple Images by Number**: 5スロットまでの画像を同時に保存します。
- **Retrieve Multiple Images by Number**: 5スロットまでの画像を同時に呼び出します。

### 利点
ノード間の物理的な接続ライン（ワイヤー）を減らし、ワークフローを整理するのに役立ちます。

### 注意
保存先はプロセス内メモリです。ComfyUIを再起動すると保存内容は失われます。また、同じ `image_id` へ保存すると前の画像は上書きされます。

---

## 2. Preview Nodes
- **Simple Fast Preview**: 非常に軽量で高速なプレビュー。
- **Advanced Fast Preview**: より詳細な設定が可能なプレビュー。
- **Text Fast Preview**: 入力文字列をノード上に大きく表示し、履歴を確認できます。履歴をワークフローに保存するかどうかをノード内ボタンで切り替えられます。

### Simple Fast Preview
入力画像を 512x512 枠に自動リサイズしてノード上に表示し、同時にリサイズ後の `IMAGE` を出力します。

### Advanced Fast Preview
`resize_mode`、`width`、`height`、`upscale_method`、出力フォーマット、品質を指定できます。プレビュー用途だけでなく、軽いリサイズノードとしても使えます。

### Text Fast Preview
入力テキストをノード上の大きな領域に表示します。履歴は最大20件で、ノード内の `Save/Temp` 切り替えによりワークフローへ保存するか一時表示にするかを選べます。
