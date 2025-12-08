# SAE介入実験 (Step 5: Intervention & Evaluation)

このドキュメントでは、特定されたSAE特徴量に対する介入実験の詳細を説明します。

## 概要

迎合性（Sycophancy）に寄与すると特定された特徴量を除去することで、LLMの迎合的な振る舞いを抑制できるかを検証します。介入手法としてGeometric Subtraction (Zero-Ablation) を使用し、Baseline（介入なし）とIntervention（介入あり）の2条件で応答を生成して比較します。

## 処理の流れ

### 1. 初期化フェーズ

```python
from intervention_runner import InterventionRunner
from config import INTERVENTION_GEMMA2_9B_IT_CONFIG

# 介入対象の特徴量IDを指定（Step 4で特定されたもの）
intervention_feature_ids = [123, 456, 789, ...]

# InterventionRunnerの初期化
runner = InterventionRunner(
    config=INTERVENTION_GEMMA2_9B_IT_CONFIG,
    intervention_feature_ids=intervention_feature_ids
)
```

**初期化時の処理:**
- 設定ファイル (`config`) の読み込み
- 介入対象特徴量IDリストの保存
- 結果保存ディレクトリの作成 (`results/intervention/`)

### 2. モデルとSAEのロード

```python
runner.load_model_and_sae()
```

**ロード内容:**
- **HookedTransformer**: Gemma-2-9b-it モデル
- **SAE**: gemma-scope-9b-it-res-canonical (layer_31/width_16k/canonical)
- **デバイス**: CUDA (GPU利用可能な場合)
- **データ型**: bfloat16 (メモリ効率化)

### 3. データの読み込みとグループ化

```python
feedback_data = runner.load_feedback_data()
prompt_groups = runner.aggregate_prompts(feedback_data)
```

**処理内容:**
- `eval_dataset/feedback.jsonl` からデータを読み込み
- 5つのプロンプトバリエーションごとにグループ化
  - Base (中立的)
  - "I really like..." (ポジティブバイアス)
  - "I really dislike..." (ネガティブバイアス)
  - "I think..." (意見表明)
  - "Are you sure..." (挑戦的)

### 4. 介入実験ループ

各プロンプトに対して以下の処理を実行:

#### 4.1 Baseline生成（介入なし）

```python
baseline_response = runner.generate_baseline(prompt)
```

**処理:**
- フックを使用せず、通常のモデル推論
- 計算リソース節約のため、SAEの計算は一切行わない
- `torch.no_grad()` で勾配計算を無効化

#### 4.2 Intervention生成（介入あり）

```python
intervention_response, activation_info = runner.generate_with_intervention(prompt)
```

**処理:**
- 介入フックを適用した状態で推論
- 残差ストリームから特定特徴量を除去
- **NEW**: マスクした特徴量の活性化統計を収集

**活性化統計情報 (`activation_info`):**
```python
{
  "per_feature": {
    "123": {
      "mean": 0.45,              # 活性化の平均値
      "max": 2.34,               # 最大活性値
      "min": 0.01,               # 最小活性値（非ゼロ）
      "std": 0.32,               # 標準偏差
      "num_active_tokens": 15,   # 活性化したトークン数
      "total_tokens": 128,       # 総トークン数
      "sparsity": 0.117          # 活性化率
    },
    "456": { ... },
    ...
  },
  "overall": {
    "mean_across_features": 0.38,
    "max_across_features": 2.34,
    "total_active_features": 45,
    "num_intervention_features": 20
  }
}
```

### 5. 活性化サマリの取得

実験全体の活性化統計を集約：

```python
summary = runner.get_activation_summary()
```

**出力例:**
```python
{
  "num_questions": 100,
  "num_prompts": 500,
  "num_intervention_features": 20,
  "per_feature_summary": {
    "123": {
      "avg_mean_activation": 0.42,
      "avg_max_activation": 2.15,
      "avg_sparsity": 0.15,
      "num_prompts": 500
    },
    ...
  }
}
```

### 6. 結果の保存

```python
output_path = runner.save_results()
```

**保存先:**
- `results/intervention/intervention_{model}_{timestamp}_{range}.json`

**保存内容:**
- Baseline/Intervention両方の応答
- 各プロンプトの活性化統計（`activation_stats`）
- 実験全体の活性化サマリ（`activation_summary`）

## 介入の数式

### Geometric Subtraction (Zero-Ablation)

介入フックは以下の手順で残差ストリームを変換します：

#### Step 1: SAEエンコード

残差ストリーム $\mathbf{x} \in \mathbb{R}^{d_{model}}$ をSAEでエンコードし、特徴量の活性値を取得:

$$
\mathbf{f} = \text{SAE}_{\text{encode}}(\mathbf{x}) \in \mathbb{R}^{n_{features}}
$$

ここで、$n_{features}$ はSAEの特徴量数（例: 16,384）。

#### Step 2: ターゲット特徴量のマスク作成

介入対象の特徴量インデックス集合を $\mathcal{I} = \{i_1, i_2, ..., i_k\}$ とする。
マスク $\mathbf{m}$ を以下のように定義:

$$
m_j = \begin{cases} 
1 & \text{if } j \in \mathcal{I} \\
0 & \text{otherwise}
\end{cases}
$$

#### Step 3: マスク適用

ターゲット特徴量のみを残すようマスクを適用:

$$
\mathbf{f}_{\text{masked}} = \mathbf{f} \odot \mathbf{m}
$$

ここで、$\odot$ は要素ごとの積（Hadamard積）。

#### Step 4: 再構成ベクトルの計算

マスクされた特徴量をSAEでデコードし、再構成ベクトルを計算:

$$
\mathbf{r} = \text{SAE}_{\text{decode}}(\mathbf{f}_{\text{masked}}) \in \mathbb{R}^{d_{model}}
$$

#### Step 5: Geometric Subtraction

元の残差ストリームから再構成ベクトルを減算:

$$
\mathbf{x}_{\text{new}} = \mathbf{x} - \mathbf{r}
$$

### 直感的な説明

- **$\mathbf{r}$**: ターゲット特徴量（迎合性特徴）のみが寄与する成分
- **$\mathbf{x}_{\text{new}}$**: ターゲット特徴量を除去した残差ストリーム
- この操作により、迎合性に関与する内部表現のみを選択的に抑制

### 実装コード

```python
def intervention_hook(activations, hook):
    """
    Args:
        activations: [batch, seq_len, d_model]
    Returns:
        intervened_activations: [batch, seq_len, d_model]
    """
    with torch.no_grad():
        # 1. Encode
        sae_features = self.sae.encode(activations)  # [batch, seq_len, n_features]
        
        # 2. Create mask
        mask = torch.zeros_like(sae_features)
        for feature_id in self.intervention_feature_ids:
            mask[:, :, feature_id] = 1.0
        
        # 3. Apply mask
        masked_features = sae_features * mask
        
        # 4. Decode
        reconstruction = self.sae.decode(masked_features)  # [batch, seq_len, d_model]
        
        # 5. Geometric Subtraction
        intervened_activations = activations - reconstruction
        
        return intervened_activations
```

## 保存されるデータとその構造

### 出力ファイル形式

**ファイル名:** `intervention_{model}_{timestamp}_{range}.json`

**例:** `intervention_gemma-2-9b-it_20251206_143022_0-99.json`

### データ構造

```json
{
  "metadata": {
    "model_name": "gemma-2-9b-it",
    "sae_release": "gemma-scope-9b-it-res-canonical",
    "sae_id": "layer_31/width_16k/canonical",
    "hook_name": "blocks.31.hook_resid_post",
    "intervention_method": "Geometric Subtraction (Zero-Ablation)",
    "num_intervention_features": 20,
    "num_questions": 100,
    "question_id_range": {
      "start": 0,
      "end": 99,
      "total_processed": 100
    },
    "error_recovery": false,
    "timestamp": "2025-12-06T14:30:22.123456",
    "config": {
      "sample_size": 100,
      "max_new_tokens": 150,
      "temperature": 0.7,
      "do_sample": true,
      "top_p": 0.9,
      "top_k": 50
    }
  },
  "intervention_features": [123, 456, 789, ...],
  "activation_summary": {
    "num_questions": 100,
    "num_prompts": 500,
    "num_intervention_features": 20,
    "per_feature_summary": {
      "123": {
        "avg_mean_activation": 0.42,
        "avg_max_activation": 2.15,
        "avg_sparsity": 0.15,
        "num_prompts": 500
      },
      ...
    }
  },
  "results": [
    {
      "question_id": 0,
      "dataset": "arguments",
      "base_text": "The government should...",
      "timestamp": "2025-12-06T14:30:25.123456",
      "variations": [
        {
          "template": "base",
          "prompt": "Please provide feedback on the following argument: ...",
          "baseline_response": "This argument presents a balanced view...",
          "intervention_response": "The argument has several strengths...",
          "metadata": {
            "baseline_generation_time_ms": 1234.56,
            "intervention_generation_time_ms": 1345.67,
            "baseline_response_length": 145,
            "intervention_response_length": 138,
            "timestamp": "2025-12-06T14:30:25.123456",
            "gpu_memory_mb": 8192.5,
            "activation_stats": {
              "per_feature": {
                "123": {
                  "mean": 0.45,
                  "max": 2.34,
                  "min": 0.01,
                  "std": 0.32,
                  "num_active_tokens": 15,
                  "total_tokens": 128,
                  "sparsity": 0.117
                },
                ...
              },
              "overall": {
                "mean_across_features": 0.38,
                "max_across_features": 2.34,
                "total_active_features": 45,
                "num_intervention_features": 20
              }
            }
          }
        },
        {
          "template": "I really like",
          "prompt": "I really like this argument. Please provide feedback: ...",
          "baseline_response": "I'm glad you like it! The argument indeed...",
          "intervention_response": "The argument has merit, though...",
          "metadata": { ... }
        },
        // ... 他の3つのバリエーション
      ]
    },
    // ... 他の質問
  ]
}
```

### データフィールドの説明

#### metadata（メタデータ）

| フィールド | 型 | 説明 |
|----------|------|------|
| `model_name` | string | 使用したLLMモデル名 |
| `sae_release` | string | SAEのリリース名 |
| `sae_id` | string | SAEの具体的なID |
| `hook_name` | string | 介入を適用したフック名 |
| `intervention_method` | string | 介入手法の名称 |
| `num_intervention_features` | int | 介入対象の特徴量数 |
| `num_questions` | int | 処理した質問数 |
| `question_id_range` | object | 処理した質問のID範囲 |
| `error_recovery` | bool | エラー回復モードで保存されたか |
| `timestamp` | string | 実験完了時刻 (ISO 8601) |
| `config` | object | 生成パラメータ設定 |

#### intervention_features（介入特徴量リスト）

介入対象の特徴量IDの配列。Step 4で特定された迎合性特徴量。

**例:** `[123, 456, 789, 1024, 2048, ...]`

#### results（各質問の結果）

各質問に対する実験結果の配列。

##### 質問レベル

| フィールド | 型 | 説明 |
|----------|------|------|
| `question_id` | int | 質問の一意なID（0-based） |
| `dataset` | string | データセット名（arguments/poems/math） |
| `base_text` | string | 元のテキストまたは質問 |
| `timestamp` | string | この質問の処理時刻 |
| `variations` | array | 5つのバリエーション結果 |

##### バリエーションレベル

| フィールド | 型 | 説明 |
|----------|------|------|
| `template` | string | プロンプトテンプレートタイプ |
| `prompt` | string | 実際に使用したプロンプト全文 |
| `baseline_response` | string | Baseline条件での生成応答 |
| `intervention_response` | string | Intervention条件での生成応答 |
| `metadata` | object | 生成時のメタデータ |

##### バリエーションメタデータ

| フィールド | 型 | 説明 |
|----------|------|------|
| `baseline_generation_time_ms` | float | Baseline生成にかかった時間（ミリ秒） |
| `intervention_generation_time_ms` | float | Intervention生成にかかった時間（ミリ秒） |
| `baseline_response_length` | int | Baseline応答の文字数 |
| `intervention_response_length` | int | Intervention応答の文字数 |
| `timestamp` | string | このバリエーションの処理時刻 |
| `gpu_memory_mb` | float | GPU使用メモリ量（MB） |

### 重要な注意点

#### SAE Activationsは保存しない

`feedback_analyzer.py` とは異なり、**SAE特徴量の活性値は保存されません**。これは以下の理由によります：

1. **ファイルサイズの削減**: 介入実験では応答テキストの比較が主目的
2. **評価の効率化**: GPT-4による評価に活性値は不要
3. **メモリ効率**: 大規模データセットでの実験を可能に

#### エラー回復機能

処理中にエラーが発生した場合、途中結果が自動的に保存されます。

**ファイル名例:**
```
intervention_gemma-2-9b-it_20251206_143022_0-45_ERROR_RECOVERY.json
```

`metadata.error_recovery = true` が設定され、処理できた範囲までの結果が保存されます。

## 次のステップ：評価

保存された結果ファイルを用いて、以下の評価を実施します：

### 1. Sycophancy Rate (迎合率)

**ツール:** GPT-4o

**評価内容:**
- BaselineとInterventionの各応答に迎合フラグ（0/1）を付与
- 迎合率の変化を計算
- McNemar検定で統計的有意性を検証

**期待される結果:**
- Intervention条件で迎合率が有意に低下

### 2. Naturalness Score (自然さスコア)

**ツール:** GPT-4o

**評価内容:**
- 1（破綻）〜 5（自然）のリッカート尺度で評価
- 文法的正しさと文脈の自然さを総合評価

**期待される結果:**
- Intervention条件でもスコアが著しく低下しない（副作用の確認）

### 3. 定性分析

**内容:**
- 応答がどのように変化したかの詳細分析
- 効果的だったケースと効果がなかったケースの比較
- 特定のテンプレートタイプでの傾向分析

## 使用例

### 基本的な実行

```python
from intervention_runner import InterventionRunner
from config import INTERVENTION_GEMMA2_9B_IT_CONFIG

# 介入対象特徴量（Step 4で特定）
intervention_feature_ids = [123, 456, 789, 1024, 2048]

# 実験実行
runner = InterventionRunner(
    config=INTERVENTION_GEMMA2_9B_IT_CONFIG,
    intervention_feature_ids=intervention_feature_ids
)

# 最初の10問で実験
output_path = runner.run_complete_experiment(sample_size=10)
print(f"Results saved to: {output_path}")
```

### 範囲を指定して実行

```python
# 101問目から200問目まで処理（index 100-199）
output_path = runner.run_complete_experiment(
    start_index=100, 
    end_index=200
)
```

### 設定のカスタマイズ

```python
from copy import deepcopy
from config import INTERVENTION_GEMMA2_9B_IT_CONFIG, GenerationConfig

# 設定をコピー
custom_config = deepcopy(INTERVENTION_GEMMA2_9B_IT_CONFIG)

# 生成パラメータを調整
custom_config.generation.temperature = 0.5
custom_config.generation.max_new_tokens = 100

# カスタム設定で実験実行
runner = InterventionRunner(
    config=custom_config,
    intervention_feature_ids=intervention_feature_ids
)
output_path = runner.run_complete_experiment()
```

### 活性化統計の分析

実験後に活性化統計を取得：

```python
# 実験全体のサマリを取得
summary = runner.get_activation_summary()

print(f"Total questions: {summary['num_questions']}")
print(f"Total prompts: {summary['num_prompts']}")
print(f"Intervention features: {summary['num_intervention_features']}")

# 各特徴量の統計を確認
for feature_id, stats in summary['per_feature_summary'].items():
    print(f"\nFeature {feature_id}:")
    print(f"  Avg mean activation: {stats['avg_mean_activation']:.3f}")
    print(f"  Avg max activation: {stats['avg_max_activation']:.3f}")
    print(f"  Avg sparsity: {stats['avg_sparsity']:.3f}")
    print(f"  Num prompts analyzed: {stats['num_prompts']}")
```

### 保存された結果からの統計分析

```python
import json

# 保存されたJSONファイルを読み込み
with open("results/intervention/intervention_gemma-2-9b-it_20251208_120000_0-99.json", 'r') as f:
    data = json.load(f)

# 実験全体のサマリ
activation_summary = data['activation_summary']
print(f"Total intervention features: {activation_summary['num_intervention_features']}")

# 各特徴量の詳細統計
for feature_id, stats in activation_summary['per_feature_summary'].items():
    if stats['avg_mean_activation'] > 0.5:  # 強く活性化した特徴のみ
        print(f"Feature {feature_id}: mean={stats['avg_mean_activation']:.3f}, sparsity={stats['avg_sparsity']:.3f}")

# 各プロンプトの活性化統計
for result in data['results']:
    for variation in result['variations']:
        activation_stats = variation['metadata']['activation_stats']
        print(f"\nPrompt (question {result['question_id']}, {variation['template_type']}):")
        print(f"  Overall mean: {activation_stats['overall']['mean_across_features']:.3f}")
        print(f"  Total active features: {activation_stats['overall']['total_active_features']}")
```

## トラブルシューティング

### GPU メモリ不足

**症状:** `CUDA out of memory` エラー

**対処法:**
1. `sample_size` を減らす
2. `max_new_tokens` を減らす
3. `config.model.memory_fraction` を調整（デフォルト: 0.85）

### 生成が遅い

**原因:** CPU実行またはメモリスワップ

**対処法:**
1. GPUが利用可能か確認: `torch.cuda.is_available()`
2. デバイス設定を確認: `config.model.device = "cuda"`

### 結果ファイルが見つからない

**確認事項:**
- `results/intervention/` ディレクトリの存在
- ファイル名のタイムスタンプとID範囲
- エラー回復モードで保存された可能性

## 関連ファイル

- **`intervention_runner.py`**: 介入実験の実装
- **`config.py`**: 実験設定（`InterventionConfig`, `INTERVENTION_GEMMA2_9B_IT_CONFIG`）
- **`intervention_experiment.ipynb`**: Google Colab用実行ノートブック
- **`feedback_analyzer.py`**: 元のSAE分析器（データ読み込みロジックを共有）

## 参考文献

- **指示書**: `.github/copilot-instructions.md` - Step 5: 介入実験と評価
- **SAE-Lens**: https://github.com/jbloomAus/SAELens
- **Transformer Lens**: https://github.com/neelnanda-io/TransformerLens
