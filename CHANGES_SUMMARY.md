# 変更サマリー: 迎合性分析の最適化

## 🎯 変更の目的

迎合性（Sycophancy）分析において、最も適切な分析位置を**応答の最初のトークン生成直前**（= プロンプトの最後のトークンの内部状態）に変更しました。

## 💡 変更の理由

迎合的な振る舞いは、プロンプト全体を処理した結果として生じる「応答全体の方針」の決定です。
モデルが迎合的なプロンプト（例: `I really like...`）を処理し終えた直後、次に応答の1トークン目を生成しようとします。

この**プロンプトの最後のトークンの内部状態**こそが、モデルの「応答プラン」を最も凝縮して表現していると考えられ、「ユーザーに同意する」という高レベルな指示がエンコードされているはずです。

## 🔧 主な変更箇所

### 1. `feedback_analyzer.py`

#### `generate_with_sae` メソッド（286-383行目）

**旧動作:**
- 生成完了後、全生成トークンでフォワードパスを実行
- 最後に生成されたトークンの内部状態を取得

**新動作:**
```python
# SAE活性化を取得: プロンプトのみでフォワードパスを実行
# これにより、応答の最初のトークン生成直前の状態を取得
_, cache = self.model.run_with_cache(tokens)  # 生成前のプロンプトのみ
```

- プロンプトのみでフォワードパスを実行
- プロンプトの最後のトークンの内部状態を取得（デフォルト）
- オプションで全プロンプトトークンの内部状態も保存可能

**キーとなる変更:**
```python
# デフォルト設定（save_all_tokens = False）
sae_activations_np = sae_features[0, -1:].cpu().numpy()  # プロンプトの最後のトークンのみ
active_features["prompt_last_token"] = {...}  # キー名も明確化
```

**追加情報:**
```python
sae_info = {
    ...
    "analyzed_position": "prompt_last_token" if not self.feedback_config.save_all_tokens else "all_prompt_tokens"
}
```

#### `__init__` メソッド（68-74行目）

ログメッセージを更新し、分析位置を明示：
```python
print(f"   💾 Save mode: {'全プロンプトトークン' if self.feedback_config.save_all_tokens else 'プロンプト最終トークンのみ（推奨）'}")
print(f"   📍 分析位置: 応答の最初のトークン生成直前の内部状態")
```

#### `save_results` メソッド（527-545行目）

メタデータに分析位置情報を追加：
```python
"analysis_position": "prompt_last_token (応答生成直前)" if not self.feedback_config.save_all_tokens else "all_prompt_tokens",
```

### 2. `config.py`

#### `FeedbackConfig` クラス（160-164行目）

コメントを更新して、設定の意味を明確化：
```python
save_all_tokens: bool = False  # True: 全プロンプトトークン, False: プロンプト最終トークンのみ（応答生成直前の状態）
```

### 3. `feedback_sycophancy_analysis.ipynb`

#### セル1: タイトルと説明（2-29行目）

特徴説明に分析位置に関する情報を追加：
```markdown
- **最適な分析位置**: プロンプトの最後のトークン（応答生成直前）の内部状態を取得
```

新セクションを追加：
```markdown
## 💡 分析位置について
迎合性分析では、**プロンプトの最後のトークン**（応答の最初のトークン生成直前）の内部状態を取得します。
これは、モデルがプロンプト全体を処理し終えた直後の状態で、「ユーザーに迎合する」という応答方針が
最も明確に表現されていると考えられます。
```

#### セル7: 実験パラメータ設定（156-219行目）

設定の説明を更新：
```python
# 2. SAE内部状態の保存設定
# 重要: False（推奨）= プロンプトの最後のトークン（応答生成直前）のみ
#       True = 全プロンプトトークンの活性化を保存
SAVE_ALL_TOKENS = False
```

サマリー表示を更新：
```python
print(f"   💾 分析位置: {'全プロンプトトークン' if SAVE_ALL_TOKENS else 'プロンプト最終トークン（応答生成直前）'}")
```

## 📊 実際の動作フロー

### デフォルト設定（`save_all_tokens = False`）の場合

1. プロンプトをトークン化: `tokens = [tok_1, tok_2, ..., tok_n]`
2. プロンプトのみでフォワードパス: `model.run_with_cache(tokens)`
3. **最後のトークン `tok_n` の内部状態を取得** ← これが応答生成直前の状態
4. SAEでエンコード
5. 特徴を保存: `active_features["prompt_last_token"] = {...}`

### `save_all_tokens = True` の場合

1. 同様にプロンプトのみでフォワードパス
2. **全プロンプトトークン `tok_1` から `tok_n` までの内部状態を保存**
3. 各トークン位置ごとに特徴を保存: `active_features["prompt_token_0"] = {...}`, etc.

## 🎯 推奨される使用方法

迎合性分析には、**デフォルト設定（`SAVE_ALL_TOKENS = False`）を推奨**します。
これにより、モデルの応答方針が最も明確に表現される「プロンプト最終トークン」の状態のみを効率的に分析できます。

## ✅ 互換性

- 既存の結果ファイルとの互換性を保持
- `save_all_tokens` フラグで旧動作も選択可能
- メタデータに `analyzed_position` フィールドを追加し、どの位置を分析したか明示

## 🔍 検証方法

実験実行後、結果JSONの `metadata` セクションを確認：
```json
{
  "metadata": {
    "save_all_tokens": false,
    "analysis_position": "prompt_last_token (応答生成直前)",
    ...
  }
}
```

`activations` セクションのキーを確認：
```json
{
  "activations": {
    "prompt_last_token": {
      "123": 0.456,
      ...
    }
  }
}
```

旧バージョンでは `"last_token"` だったキーが `"prompt_last_token"` に変更され、
分析位置がより明確になっています。
