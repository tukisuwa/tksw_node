# Custom CFG Schedule for ComfyUI

## 概要

**Custom CFG Schedule** は、ComfyUIの画像生成プロセスにおいて、ステップごとに**CFG Scale**の値を動的に変更するためのカスタムノードです。

通常のKSamplerではCFG値は生成を通じて一定ですが、このノードを使用することで、生成の進行度合いに応じてCFG値を細かく制御できます。例えば、「序盤は高いCFGで構図をしっかり固め、終盤は低いCFGでディテールを自然に仕上げる」といった高度なテクニックを簡単に実現できます。


## 主な機能

*   **CFGスケジューリング:** ステップごとにCFG値を自由に設定できます。
*   **柔軟なステップ指定:**
    *   **絶対ステップ:** `5:10.0`のように、特定のステップ番号で指定。
    *   **パーセンテージ:** `p0.8:4.0`のように、全ステップに対する割合で指定。
*   **補間 (Interpolation):** 指定したCFG値の間を滑らかに（線形補間）または段階的（ステップ）に変化させることができます。
*   **ループ:** 短いCFGの変動パターンを、生成全体を通して繰り返すことができます。
*   **U-Net計算の強制スキップ:** CFGが1.0以外の値でも、U-Netのunconditional passの計算を強制的にスキップさせることができます。（上級者向け機能）


## 使い方

このノードは、KSamplerの実行**前**に評価される必要があります。ComfyUIの実行順序を確実にするため、`model`の経路上に配置するのが最も簡単で確実な方法です。


1.  **Model接続 (実行順序の保証):**
    *   Loaderから出力された`MODEL`を、このノードの`passthrough_model`に接続します。
    *   このノードの`passthrough_model_out`を、KSamplerの`model`に接続します。
    *   **なぜ接続するの？** このノードは`model`データ自体を書き換えません。この接続は、ComfyUIに「このノードの処理をKSamplerより先に行ってください」と教えるための目印です。

2.  **Sigmas接続 (ステップ数の取得):**
    *   **（強く推奨）** KSampler（またはSamplerCustomなど）から`SIGMAS`を、このノードの`sigmas`に接続します。
    *   **（強く推奨）** このノードの`sigmas_out`を、サンプラーノードの`sigmas`に接続します。
    *   **なぜ接続するの？** `sigmas`を接続することで、ノードはサンプラーが実行する正確なステップ数を把握できます。これにより、パーセンテージ指定などが意図通りに機能します。

## 入力パラメータ詳解

### 基本設定

*   **`enabled`**: チェックが入っている場合、このノードの機能が有効になります。外すと、何もせずに入力をそのままパススルーするため、一時的に無効化したい場合に便利です。

*   **`initial_cfg`**: スケジュールの開始点（ステップ0）で使われるCFG値です。`cfg_schedule_points`でステップ0が指定されていない場合に、この値が自動的に適用されます。

### スケジュール定義

*   **`cfg_schedule_points`**: **（最重要）**CFGスケジュールを定義するテキストボックスです。
    *   **書式:** `[ステップ指定子]:[CFG値]:[スキップフラグ]`
    *   **区切り:** 各設定はカンマ(`,`)または改行で区切ります。
    *   **コメント:** 行頭に `#` を付けると、その行は無視されます。

    #### ステップ指定子の種類
    | 指定子 | 例         | 説明                                                                     |
    | :----- | :--------- | :----------------------------------------------------------------------- |
    | **数字** | `5:10.0`   | **絶対ステップ指定。** ステップ5でCFGを10.0に設定します。                |
    | **`p`**  | `p0.8:4.0` | **全体パーセンテージ指定。** 全ステップ数の80%の地点でCFGを4.0に設定します。 |
    | **`l`**  | `l0.5:7.0` | **ループ内パーセンテージ指定。** ループ長の50%の地点でCFGを7.0に設定します。 |

    #### スキップフラグ（オプション・上級者向け）
    *   末尾に `:s` または `:skip` を付けることで、そのステップでのU-Netの**unconditional passの計算を強制的にスキップ**します。
    *   例: `p0.9:1.05:s`
    *   **重要:** ComfyUIは通常、CFGが`1.0`の時にこの計算を自動でスキップします。このフラグは、CFGが**`1.0`ではない場合でも**、強制的にスキップさせたい場合に使用します。
    *   **注意:** CFGが`1.0`から大きく外れた値で計算をスキップすると、多くの場合、**生成結果が破綻します。** `1.05`や`0.95`など、`1.0`に非常に近い値で微調整する実験的な用途を想定しています。

### スケジュール挙動の制御

*   **`interpolate`**:
    *   **`Interpolate CFG` (有効):** 指定したポイント間を滑らかにCFG値が変化します（線形補間）。
    *   **`Stepped CFG` (無効):** 次のポイントまでCFG値は維持され、階段状に変化します。

*   **`schedule_loop_length`**:
    *   スケジュールを繰り返す周期をステップ数で指定します。
    *   `0`の場合はループしません。
    *   例えば`10`に設定すると、`cfg_schedule_points`で定義した0〜9ステップまでの変化が、10〜19ステップ、20〜29ステップ...と繰り返されます。

*   **`allow_overshoot_and_trim`**: ループやパーセンテージ指定が総ステップ数を超える場合の挙動を制御します。通常はデフォルトの`Clamp to Max Steps`のままで問題ありません。

### 代替入力

*   **`total_steps_override`**: **（非推奨）** `sigmas`入力がない場合や、手動で総ステップ数を強制したい場合に使います。`0`の場合は無視されます。基本的には`sigmas`入力を使用してください。

## ワークフロー例

### 例1: 基本的なスケジューリング（高→低）

序盤は高いCFGでプロンプトへの忠実度を上げ、終盤はCFGを下げて破綻を防ぎ、自然な仕上がりにします。

*   **`interpolate`**: `Interpolate CFG`
*   **`cfg_schedule_points`**:
    ```
    # ステップ0 (開始時) はCFG 12.0
    0: 12.0

    # 全体の40%の地点までは 12.0 を維持
    p0.4: 12.0

    # 最後のステップで 4.0 になるように滑らかに下げる
    p1.0: 4.0
    ```

### 例2: ループ機能を使った「V字」変化

短い周期でCFGを`高→低→高`と変化させ、独特のスタイルを生み出します。

*   **`schedule_loop_length`**: `10`
*   **`interpolate`**: `Interpolate CFG`
*   **`allow_overshoot_and_trim`**: `Overshoot & Trim` **(有効にする)**
*   **`cfg_schedule_points`**:
    ```
    # 10ステップ周期の開始点
    0: 8.0

    # 周期の半分(ステップ5)でCFGを3.0に
    l0.5: 3.0

    # 周期の終点(ステップ10)でCFGを8.0に戻す
    10: 8.0
    ```
    **この設定でどうなるか:**
    この設定により、CFGは`[8.0 → 3.0 → 8.0]`というV字の変化を10ステップの周期で正確に繰り返します。

    **重要:**
    この例のようにループの終点（ここではステップ10）を指定して閉じたループを作るには、`allow_overshoot_and_trim`を**有効にする必要があります。** これが無効だと、ステップ10の指定が「ループ範囲外」と見なされて無視され、期待通りの動作になりません。


## 注意事項

*   このノードはComfyUIのコア機能の一つである`comfy.samplers.sampling_function`に**モンキーパッチ**を適用して動作します。他のパッチ系のカスタムノードと競合する可能性があります。
*   **`sigmas`入力は、意図通りに動作させるためにほぼ必須です。** KSamplerから必ず接続してください。
*   動作がおかしいと感じた場合は、まずノードの`info`出力の内容を確認してください。スケジュールが正しく計算されているかどうかが分かります。
*   スキップフラグは実験的な機能です。CFGが1.0から大きく離れた値で使うと、意図しない結果になる可能性が高いです。