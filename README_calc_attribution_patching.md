# calc_attribution_patching.ipynb 説明書

## 概要

このノートブックは、LLM（Gemma-2-9b-it）の迎合性（Sycophancy）を分析するため、SAE（Sparse Autoencoder）の特徴量に対する**因果効果（Attribution Patching Score）**を計算するスクリプトです。

---

## 1. 処理の流れ

### **ステップ1: 環境セットアップ**
```
Google Drive マウント
  ↓
プロジェクトパス設定
  ↓
依存関係インストール
  ↓
GPU確認
```

- Google Driveをマウントし、プロジェクトフォルダ（`/content/drive/MyDrive/sae_pj2`）にアクセス
- 必要なライブラリ（`transformer_lens`, `sae_lens`等）をインストール
- GPU（A100 40GB想定）の利用可能性を確認

---

### **ステップ2: データ読み込みとペアリング**
```
入力JSON読み込み
  ↓
yield_sycophancy_samples() でペア生成
  ↓
Base回答 ⇔ Target回答（迎合）のペアを抽出
```

**入力ファイル例:**
```
results/labeled_data/feedback_analysis_gemma-2-9b-it_20251117_0-100.json_labeled.json
```

**ペアリングロジック:**
- 各質問について、`template_type="base"` の回答を特定
- `sycophancy_flag=1` の迎合的回答とペアリング
- 以下の情報を生成:
  - `prompt`: 質問文
  - `target_response`: 迎合的な回答
  - `base_response`: 中立的な回答
  - `variation_index`: 元のJSONでのインデックス

---

### **ステップ3: モデルとSAEのロード**
```
HookedTransformer.from_pretrained_no_processing()
  ↓
SAE.from_pretrained()
  ↓
AttributionPatchingAnalyzer 初期化
```

- **モデル:** `google/gemma-2-9b-it` (bfloat16)
- **SAE:** Gemma Scope SAE (`layer_31/width_16k/canonical`)
- **Hook位置:** `blocks.31.hook_resid_post`

---

### **ステップ4: Attribution Patching 分析（各サンプル）**

#### **4-1. Teacher Forcing によるトークン化**
```python
full_text = prompt + target_response
input_tokens = model.to_tokens(full_text)
```
- 新規生成は行わず、実際の回答をそのまま入力
- モデルが「この回答を生成した時の状態」を再現

#### **4-2. トークンペアの特定**
```python
# 最初の20トークンを順番にチェック
for offset in range(1, 21):
    if target_token[offset] != base_token[offset]:
        # 異なるトークンペアを発見
        break
```
- プロンプト終了直後から最大20トークンを比較
- 最初に異なるペアを見つけて採用
- 例: Target="Yes", Base="There"

#### **4-3. Forward Pass with SAE Hook**
```python
logits = model.run_with_hooks(
    input_tokens,
    fwd_hooks=[(hook_name, atp_hook)]
)
```

**Hook内部の処理:**
```python
def atp_hook(activation, hook):
    # 1. プロンプト終了位置のActivationを抽出
    target_act = activation[:, target_pos, :]
    
    # 2. SAE Encode（特徴量を取得）
    f_acts = sae.encode(target_act)  # [1, 1, 16384]
    f_acts.requires_grad_(True)
    
    # 3. SAE Decode（復元）
    x_hat = sae.decode(f_acts)
    
    # 4. Gradient Trick
    # Forward: 元のActivationを流す（精度維持）
    # Backward: SAEを通した経路で勾配を流す
    x_out = x_hat + (target_act - x_hat).detach()
    
    return x_out
```

#### **4-4. Metric 計算**
```python
# Logit Difference（迎合度の指標）
target_logit = logits[0, logit_pos, target_token_id]  # "Yes"へのスコア
base_logit = logits[0, logit_pos, base_token_id]      # "There"へのスコア
metric = target_logit - base_logit
```

**Metricの意味:**
- 値が大きい → モデルはTargetトークンを強く予測（迎合的）
- 値が小さい/負 → モデルはBaseトークンを予測（非迎合的）

#### **4-5. Backward Pass（勾配計算）**
```python
metric.backward()
# → 各SAE特徴量に対する勾配 f_acts.grad が計算される
```

---

### **ステップ5: 結果の保存**
```
元のJSON構造を保持
  ↓
各variation に atp_analysis フィールドを追加
  ↓
タイムスタンプ付きファイルに保存
```

**出力ファイル例:**
```
results/feedback/atp_results_gemma-2-9b-it_20251201_143052.json
```

---

## 2. Attribution Patching Score の算出方法

### **基本式**
```
AtP Score(特徴 i) = Activation(特徴 i) × Gradient(特徴 i)
```

### **各要素の意味**

| 要素 | 計算方法 | 意味 |
|------|----------|------|
| **Activation** | `f_acts[i]` | その特徴がどれだけ活性化したか |
| **Gradient** | `∂Metric/∂f_acts[i]` | その特徴がMetricにどれだけ影響するか |
| **AtP Score** | `Activation × Gradient` | 実際にMetricに寄与した量 |

### **具体例**

#### **高スコアの特徴（迎合性に寄与）**
```
特徴1234:
  Activation = 1.2  ← この文脈で強く活性化
  Gradient = 3.5    ← Metricを3.5増やす効果
  Score = 4.2       ← 迎合性を強めた
```

#### **負スコアの特徴（迎合性を抑制）**
```
特徴5678:
  Activation = 0.8  
  Gradient = -2.1   ← Metricを2.1減らす効果
  Score = -1.68     ← 迎合性を弱めた
```

#### **無関係の特徴**
```
特徴9999:
  Activation = 2.0  ← 発火はしている
  Gradient = 0.05   ← ほとんど影響しない
  Score = 0.1       ← 迎合性とは無関係
```

### **Top-K 抽出**
```python
top_k = 50
top_indices = torch.topk(atp_scores.abs(), k=50).indices
```
- スコアの**絶対値**が大きい上位50個を保存
- 正負どちらも重要（正=促進、負=抑制）

---

## 3. 出力ファイルの構造

### **全体構造**
```json
{
  "metadata": { ... },
  "results": [
    {
      "question_id": 0,
      "dataset": "arguments",
      "base_text": "質問文の一部...",
      "variations": [
        {
          "template_type": "base",
          "prompt": "質問全文",
          "response": "中立的な回答",
          "sycophancy_flag": 0
        },
        {
          "template_type": "I really like",
          "prompt": "質問全文 + I really like...",
          "response": "迎合的な回答",
          "sycophancy_flag": 1,
          "atp_analysis": { ... }  ← ここに結果が追加される
        }
      ]
    }
  ]
}
```

### **atp_analysis フィールドの詳細**

#### **成功時**
```json
"atp_analysis": {
  "target_token": "Yes",
  "base_token": "There",
  "token_position": 1,
  "logit_diff": 3.85,
  "top_features": [
    {
      "id": "1234",
      "score": 4.2,
      "activation": 1.2,
      "gradient": 3.5
    },
    {
      "id": "5678",
      "score": -1.68,
      "activation": 0.8,
      "gradient": -2.1
    },
    ...  // 全50個
  ]
}
```

**各フィールドの説明:**

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `target_token` | string | 迎合的回答の比較トークン |
| `base_token` | string | 中立的回答の比較トークン |
| `token_position` | int | 何トークン目を使用したか（1-20） |
| `logit_diff` | float | Metricの値（迎合度） |
| `top_features` | array | 上位50個の特徴 |
| `top_features[i].id` | string | SAE特徴のID（0-16383） |
| `top_features[i].score` | float | Attribution Patching Score |
| `top_features[i].activation` | float | その特徴の活性化値 |
| `top_features[i].gradient` | float | Metricへの勾配 |

#### **スキップ時（トークンが同一）**
```json
"atp_analysis": {
  "skipped": "No differing tokens found in first 20 positions"
}
```

#### **エラー時**
```json
"atp_analysis": {
  "error": "OOM",
  "details": { ... }
}
```

---

## 4. 主要パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `max_tokens_to_check` | 20 | トークン比較の範囲 |
| `top_k` | 50 | 保存する特徴数 |
| `save_interval` | 10 | 定期保存の間隔 |
| `model_dtype` | bfloat16 | モデルの精度 |
| `hook_layer` | 31 | SAEを適用する層 |
| `sae_width` | 16384 | SAE特徴の数 |

---

## 5. メモリ最適化

- **batch_size = 1**: 1サンプルずつ処理
- **定期的なクリーンアップ**: 10サンプルごとに保存+GC
- **勾配の選択的保持**: SAE特徴のみ `.retain_grad()`
- **即座のメモリ解放**: `del` + `torch.cuda.empty_cache()`

---

## 6. 実行環境

- **ハードウェア:** Google Colab (A100 40GB)
- **実行時間（推定）:** 145サンプルで約15-20分
- **出力ファイルサイズ:** 約5-10MB（50特徴×145サンプル）

---

## 7. 次のステップ

このスクリプトの出力を使って:
1. **特徴の特定**: 高スコアの特徴IDをリスト化
2. **介入実験**: その特徴を0にして（Ablation）再生成
3. **効果測定**: 迎合性が実際に減少するか検証

---

## 8. トラブルシューティング

### **"No differing tokens found" が多い場合**
- `max_tokens_to_check` を50に増やす
- BaseとTargetの回答が本当に異なるか確認

### **OOM (Out of Memory) エラー**
- GPUメモリを確認（A100 40GB推奨）
- シーケンス長が長すぎる場合はスキップ

### **実行が遅い場合**
- `save_interval` を50に増やす（I/O削減）
- デバッグ出力（`if i < 3`）を無効化
