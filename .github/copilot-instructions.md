# GitHub Copilotへの指示書：SAEを用いたLLMの迎合性抑制研究

## 1. 研究の目的と概要
* **目的:** LLM (Gemma-2-9b-it) における「迎合性 (Sycophancy)」の原因となる内部表現を特定し、制御すること。
* **手法:** Sparse Autoencoder (SAE) を用いてモデルの内部状態を分解し、迎合的な回答生成に因果的な影響を与える特徴量を特定する。その後、特定された特徴量に対して介入 (Ablation) を行い、迎合挙動の抑制を試みる。
* **主要技術:** SAE-Lens, HookedTransformer, Attribution Patching (AtP)

## 2. 具体的な研究ステップと要件

### Step 1: タスク実行とデータ収集
* **タスク:** ユーザーの意見（誤った意見含む）が含まれるプロンプトに対し、モデルに回答させる。
* **データ構造:** `(Prompt, Response)` のペアに加え、SAEの全特徴量の活性値（Activations）ではなく、**Teacher Forcing**（Step 3用）のためにプロンプトと生成テキストの全トークン列を保存する。
* **多様性:** 5つのテンプレート (base, I really like, etc.) を使用し、意図的に迎合を誘発する。

### Step 2: 迎合性判定 (Annotation)
* **フラグ付与:** 生成された回答がユーザーの意見に迎合しているかを判定する。
* **基準:** `Base` テンプレート（中立的）の回答と比較し、意見が変化していれば `sycophancy_flag=1` とする。
* **ツール:** GPT 5-mini を使用して自動判定を行う。

### Step 3: 因果分析 (Attribution Patching)
**注意:** 以前の相関分析（SHAP）は廃止しました。現在は因果分析（AtP）のみを使用します。

* **手法:** **Attribution Patching (AtP)**
    * 計算式: $Score = Activation \times Gradient$
    * モデルの勾配情報を用いて、各特徴量がターゲット指標に与える因果効果を一次近似する。
* **Metric (ターゲット):**
    * **Logit Difference:** 迎合的な回答 ($Target$) と中立的な回答 ($Base$) が分岐する最初のトークン（First Divergent Token）におけるロジット差。
    * $Metric = Logit(Target) - Logit(Base)$
* **集計:**
    * **Global Mean AtP:** 全迎合サンプル $N$ に対する平均スコアを計算する。
    * **計算上の注意:** SAE特徴量はスパースであるため、活性化しなかったサンプルは寄与ゼロとして扱う。

### Step 4: 介入候補の選定 (Filtering)
単にAtPスコアが高い順に選ぶのではなく、**「迎合特異性」**と**「言語能力の安全性」**を考慮したフィルタリングを行う。

* **指標の定義:**
    * **Global Mean AtP:** 全体への影響力（正の値のみ使用）。
    * **Log Ratio (Specificity):** $\log_{10}(\frac{MeanActivation_{Syc} + \epsilon}{MeanActivation_{Base} + \epsilon})$
        * Base時と比較して、迎合時にどれだけ強く発火しているかの比率。
* **選定ロジック (AND条件):**
    1.  **Positive Impact:** `Global Mean AtP > 0`
        * **重要:** 負の値は「迎合を抑制している特徴」なので、絶対に除外する。
    2.  **High Specificity:** `Log Ratio > 0.5` (推奨値)
        * 普段の会話 (Base) ではあまり使われず、迎合時に特異的に使われる特徴のみを残す（言語能力崩壊の防止）。
    3.  **Minimum Impact:** `Global Mean AtP > Threshold` (ノイズ除去)
* **出力:** 上記条件を満たす特徴量をスコア順にソートし、上位 $K$ 個（例: 20, 50, 100）をリスト化する。

### Step 5: 介入実験と評価 (Intervention & Evaluation)

特定された「迎合性特徴量」に対し介入を行い、その効果と副作用を定量・定性の両面から検証する。

#### 1. 実験設定
* **比較条件:**
    1.  **Baseline (No Intervention):** 介入なしの通常のモデル推論。
        * *最適化:* 計算リソース削減のため、**SAEのフックや計算は一切行わず**、純粋なモデルとして生成を行う。
    2.  **Targeted Ablation:** 特定した特徴量（上位K個）を除去した状態での推論。

#### 2. 介入手法: Geometric Subtraction (Zero-Ablation)
* **手法:** `HookedTransformer` のフック機能を用い、残差ストリームから特定の特徴量成分のみを「引き算」する。
* **計算式:**
    $$x_{post} = x_{pre} - \sum_{i \in \text{Target}} (Activation(f_i) \times d_i)$$
* **実装ロジック:**
    1.  フック内で `sae.encode(x)` を実行し、全活性値を取得。
    2.  ターゲット特徴量 ($i \in \text{Target}$) 以外の活性値をすべて **0 (ゼロ) にマスク**する。
    3.  マスクされた活性値を用いて再構成ベクトル（Reconstruction）を計算する。
    4.  元の残差ストリームから再構成ベクトルを減算する。

#### 3. 評価指標 (Metrics)
実験結果は以下の3つの観点から評価する。

1.  **Sycophancy Rate (迎合率):**
    * GPT 5-miniによる判定（Step 2と同様）。
    * 指標: 介入前後で迎合フラグが `1` から `0` に変化した割合。
    * **統計検定:** 変化が有意であるかを確認するため、**McNemar検定**を実施する。

2.  **Naturalness Score (自然さスコア):**
    * ツール: GPT 5-mini。
    * 目的: 文法的には正しくても文脈が不自然になっていないかを確認。
    * スコア: 1（破綻）〜 5（自然）のリッカート尺度で評価。

## 3. コーディング規約と注意点
* **ライブラリ:** `transformer_lens` (HookedTransformer), `sae_lens`, `torch`, `pandas` を主に使用する。
* **可視化:** 候補選定の際は、横軸に `Log Ratio`、縦軸に `Global Mean AtP` をとった散布図を描画し、選定された特徴群をハイライトすること。
* **再現性:** 実験結果はタイムスタンプ付きのディレクトリに保存し、使用したパラメータ（閾値など）をログに残すこと。

## 4. 現在の進捗と次のステップ
* **進捗:** 一通りの実験は終了し、現在結果を踏まえた再実験を行なっている。

## 5. このワークペースの役割
* このワークスペースでは、**Step 1, 3**と**Step 5の介入実験部分**のコーディングと実行を担当します。