# GitHub Copilotへの指示書

## 1. 研究内容
このプロジェクトではSAE（Sparse Autoencoders）を利用したLLMの迎合性（Sycophancy）抑制に関する研究をおこなっています。
タスクを通して、LLMの迎合性に関わるSAE特徴を特定し、その特徴に介入することで、LLMの迎合的な振る舞いを抑制・緩和しようという試み。

## 2. 具体的な研究ステップ

### ステップ1: タスク実行とデータ取得
* **モデル:** LLM (Gemma-2-9b-it)
* **タスク:** 5つのテンプレートタイプ(base, I really like, I really dislike, I wrote, I didn't write)で、論理的な矛盾や統計的根拠に欠けた議論を含むトピックを渡し、それに対するコメントをさせる。
* **特徴取得:** タスク実行時に`SAE-lens`を利用する。ステップ3の機械学習モデル（XGBoost等）での学習とSHAP分析のため、SAEの**活性化した全特徴（活性化 > 0）のIDと値の辞書**（`sae_activations`）を保存する。
* **実行環境:** Google Driveに実行ファイルを保存し、Google Colab上で実行する。

### ステップ2: 迎合性フラグの付与とデータ保存
* **評価:** LLMが生成したコメントを`gpt5-mini`に渡し、「base」テンプレートの出力と比較させる。
* **フラグ付与:** 他テンプレートが「base」より迎合しているか判断させ、`sycophancy_flag=1` (迎合) / `0` (非迎合) と `reason` (判断理由) を生成させる。
* **保存:** ステップ1の結果と合わせ、以下のJSON構造で保存する。

* **Feedback実験 出力フォーマット構造**
    *(注: これは実際のデータではなく、ファイルの構造を示すためのものです)*
    ```json
    {
      "metadata": {
        "model_name": "[string]",
        "sae_release": "[string]",
        "sae_id": "[string]",
        // ... 他のメタデータ ...
      },
      "results": [
        {
          "question_id": "[number]",
          "dataset": "[string]",
          "base_text": "[string]",
          "variations": [
            {
              "template_type": "[string]",
              "prompt": "[string]",
              "response": "[string]",
              "sae_activations": {
                // 変更点： 0より大きい全活性化特徴を保存
                "prompt_last_token": {
                  "[feature_id (string)]": "[activation_value (float)]",
                  "[feature_id (string)]": "[activation_value (float)]",
                  "...": "..."
                }
              },
              // ... 他のフィールド ...
              "sycophancy_flag": "[0 or 1]",
              "reason": "[string]"
            }
          ]
        }
      ]
    }
    ```

### ステップ3: 特徴の特定 (MLモデル + SHAP)
* **データ準備:** ステップ2のJSONから`sae_activations`（特徴ベクトル、疎行列として扱う）と`sycophancy_flag`（ラベル）を抽出する。
* **モデル学習:** SAE特徴を入力(X)、`sycophancy_flag`を正解ラベル(Y)として、XGBoost (XGBClassifier) などの機械学習モデルを学習させる。
* **SHAP分析:** 学習済みモデルに対し`shap.TreeExplainer`を適用し、各特徴のSHAP値を計算する。
* **特徴特定:** `shap.summary_plot` を分析し、迎合性（Flag=1）の予測に最も強く**正の**寄与をしているSAE特徴を特定する。

### ステップ4: 介入と出力操作
* **介入（Ablation）:** ステップ3で特定した「迎合性特徴」のリストを取得する。
* **フック:** `HookedTransformer`のフック機能を用い、これらの特徴の活性化を強制的に0にする（Ablationする）介入ロジックを実装する。
* **評価:** 介入状態で迎合的なプロンプト（例: "I really like..."）を入力し、モデルの応答が非迎合的（"base"の応答に近い）になるか、迎合性が抑制されるかを定量・定性的に評価する。

## 3. 現在の進捗
- ステップ1とステップ2は完了していたが、ステップ１のプログラムに問題があったため、もう一度実行する。

## 4. このワークスペースの目的
- このワークスペースは上記の研究ステップの１を実行するためのものです。
- Google Colab上で実行することを想定しています。
- 実行結果はGoogle Driveに保存されます。