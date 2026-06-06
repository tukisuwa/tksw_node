# tksw_node

ComfyUI 用のカスタムノード集です。

データ読み込み、テキスト処理、LoRA 管理、プレビュー、画像一時保存、CFG スケジュールなど、ComfyUI ワークフローを補助するノードを含みます。

## 主なカテゴリ

### Data Loading
- **Image Sequence Loader**: フォルダ内画像を順番に読み込みます。
- **Image Pair Sequence Loader**: 2つのフォルダから同名画像ペアを読み込みます。
- **Image TextPair SequenceLoader**: 画像と同名 `.txt` をペアで読み込みます。
- **Image Loader Seed Sync**: `seed % 画像数` で画像を選択します。
- **Text File Selector**: `.txt` ファイルを random / round-robin で選択して読み込みます。

詳細: [doc/DataLoading.md](./doc/DataLoading.md)

### Text Processing
- **Text Combiner**: 複数テキストを結合します。
- **Text Processor**: 置換ルールでテキストを整形します。
- **Random Word Replacer**: 指定語をランダム候補へ置換します。

詳細: [doc/TextPromptProcessing.md](./doc/TextPromptProcessing.md)

### Preview
- **Simple Fast Preview**: 画像を軽量プレビュー表示します。
- **Advanced Fast Preview**: サイズ・形式・品質を指定できるプレビューです。
- **Text Fast Preview**: テキストをノード上に表示し、履歴も扱えます。

詳細: [doc/Utilities.md](./doc/Utilities.md)

### LoRA
- **LoRA Loader Elemental**: LoRA キー検索ごとに強度を指定します。
- **LoRA Loader Elemental UI**: Elemental のルールを List / EQ / Graph UI で編集します。
- **LoRA Selector**: フォルダまたはスロットから LoRA を選択して適用します。
- **LoRA Weight Randomizer**: 複数 LoRA の強度配分をランダム化します。
- **LoRA Mixer Elemental**: 複数 LoRA のキーを合成します。
- **Quantized LoRA Loader**: LoRA テンソルを量子化して適用します。

詳細: [doc/LoRAManagement.md](./doc/LoRAManagement.md)

### Image Utilities
- **Store Image by Number** / **Retrieve Image by Number**: 画像を番号付きでメモリ保存・取得します。
- **Store Multiple Images by Number** / **Retrieve Multiple Images by Number**: 複数画像をまとめて保存・取得します。

詳細: [doc/Utilities.md](./doc/Utilities.md)

### Sampling
- **Custom CFG Schedule**: 生成ステップごとに CFG を変更します。

詳細: [doc/CustomCFGSchedule.md](./doc/CustomCFGSchedule.md)

## インストール

ComfyUI の `custom_nodes` ディレクトリで clone します。

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/tukisuwa/tksw_node.git
```

ComfyUI を再起動するとノードが読み込まれます。

## 注意

- `LoRA Loader Elemental UI`、`Simple Fast Preview`、`Advanced Fast Preview`、`Text Fast Preview` はブラウザ側 JavaScript を使うため、更新後は ComfyUI の再起動とブラウザキャッシュ更新が必要になる場合があります。
- node id / class 名には互換性維持のため `Lora...` 表記が残るものがありますが、ComfyUI 上の表示名は `LoRA` 表記に揃えています。

## License

MIT License. See [LICENSE](./LICENSE).
