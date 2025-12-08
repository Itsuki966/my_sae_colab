"""
介入実験実行モジュール (Step 5: Intervention & Evaluation)

このモジュールは、特定されたSAE特徴量に対してGeometric Subtractionによる介入を行い、
その効果を評価するための実験を実行します。

主な機能:
- Zero-Ablation (Geometric Subtraction) による特徴量除去
- Baseline (介入なし) vs Intervention (介入あり) の比較実験
- 結果の構造化保存 (Perplexity計算は含まない)
"""

import os
import gc
import json
import torch
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm

# メモリ効率化のための環境変数設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

# SAE Lens imports
from transformer_lens import HookedTransformer
from sae_lens import SAE

# 既存の分析器から必要な機能をインポート
from feedback_analyzer import FeedbackPromptInfo


@dataclass
class InterventionResult:
    """1つのプロンプトバリエーションに対する介入実験結果"""
    prompt_info: FeedbackPromptInfo
    baseline_response: str
    intervention_response: str
    metadata: Dict[str, Any]


@dataclass
class QuestionInterventionResult:
    """1つの質問（5つのバリエーション）の介入実験結果"""
    question_id: int
    dataset: str
    base_text: str
    variations: List[InterventionResult]
    timestamp: str


class InterventionRunner:
    """
    介入実験実行クラス
    
    FeedbackAnalyzerのデータ読み込み機能を再利用しつつ、
    特定された迎合性特徴量に対する介入実験を実行します。
    """
    
    def __init__(self, config, intervention_feature_ids: List[int]):
        """
        初期化
        
        Args:
            config: ExperimentConfig オブジェクト
            intervention_feature_ids: 介入対象の特徴量IDリスト
        """
        self.config = config
        self.intervention_feature_ids = intervention_feature_ids
        self.model = None
        self.sae = None
        self.results: List[QuestionInterventionResult] = []
        
        # 活性化分析用の記録
        self.activation_stats: Dict[str, Any] = {
            'per_feature': {},  # 特徴量ごとの統計
            'per_prompt': []    # プロンプトごとの統計
        }
        
        # 介入専用設定の取得
        self.intervention_config = getattr(config, 'intervention', None)
        if self.intervention_config is None:
            # デフォルト値を設定
            from config import InterventionConfig
            self.intervention_config = InterventionConfig()
        
        # 結果保存ディレクトリの作成
        self.results_dir = Path("results/intervention")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 処理範囲を記録
        self.processed_start_id = None
        self.processed_end_id = None
        
        if self.config.debug.verbose:
            print("🔧 InterventionRunner initialized")
            print(f"   📁 Results directory: {self.results_dir}")
            print(f"   🎯 Target features: {len(intervention_feature_ids)} features")
            print(f"   🔬 Intervention method: Geometric Subtraction (Zero-Ablation)")
            print(f"   ⚙️  Hook layer: {self.config.model.hook_name}")
    
    def optimize_memory_usage(self):
        """メモリ使用量を最適化"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        except Exception as e:
            if self.config.debug.verbose:
                print(f"⚠️ Memory optimization warning: {e}")
    
    def load_feedback_data(self, data_path: Optional[str] = None) -> List[Dict]:
        """
        feedback.jsonlファイルを読み込む (FeedbackAnalyzerと同じロジック)
        
        Args:
            data_path: データファイルのパス（Noneの場合はconfigから取得）
        
        Returns:
            読み込んだデータのリスト
        """
        if data_path is None:
            data_path = self.config.data.dataset_path
        
        if self.config.debug.verbose:
            print(f"📂 Loading feedback data from: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        if self.config.debug.verbose:
            print(f"✅ Loaded {len(data)} entries")
        
        return data
    
    def create_prompt(self, data: Dict) -> FeedbackPromptInfo:
        """
        データからプロンプト情報を作成 (FeedbackAnalyzerと同じロジック)
        
        Args:
            data: feedback.jsonlの1エントリ
        
        Returns:
            FeedbackPromptInfo オブジェクト
        """
        dataset = data["base"]["dataset"]
        metadata = data["metadata"]
        prompt_template = metadata["prompt_template"]
        prompt_template_type = metadata["prompt_template_type"]
        
        if dataset == "arguments" or dataset == "poems":
            text = data["base"]["text"]
            prompt = prompt_template.format(text=text)
        elif dataset == "math":
            question = data["base"]["question"]
            correct_solution = data["base"]["correct_solution"]
            prompt = prompt_template.format(
                question=question, 
                correct_solution=correct_solution
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        return FeedbackPromptInfo(
            dataset=dataset,
            prompt_template_type=prompt_template_type,
            prompt=prompt,
            base_data=data["base"]
        )
    
    def aggregate_prompts(self, feedback_data: List[Dict]) -> List[List[FeedbackPromptInfo]]:
        """
        データを5つのバリエーションごとにグループ化 (FeedbackAnalyzerと同じロジック)
        
        Args:
            feedback_data: feedback.jsonlの全データ
        
        Returns:
            [[variation1, variation2, ..., variation5], ...] の形式
        """
        prompt_variations = []
        prompt_groups = []
        
        for i, data in enumerate(feedback_data, 1):
            prompt_info = self.create_prompt(data)
            prompt_variations.append(prompt_info)
            
            # 5つごとにグループ化
            if i % 5 == 0:
                prompt_groups.append(prompt_variations)
                prompt_variations = []
        
        # 残りがある場合（データが5の倍数でない場合）
        if prompt_variations:
            prompt_groups.append(prompt_variations)
        
        if self.config.debug.verbose:
            print(f"📦 Grouped into {len(prompt_groups)} question sets")
        
        return prompt_groups
    
    def load_model_and_sae(self):
        """モデルとSAEをロード"""
        if self.config.debug.verbose:
            print("🔄 Loading model and SAE...")
        
        # デバイス設定
        device = self.config.model.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        if self.config.debug.verbose:
            print(f"   🖥️  Using device: {device}")
        
        # モデルのロード
        if self.config.debug.verbose:
            print(f"   📥 Loading model: {self.config.model.name}")
        
        dtype = torch.bfloat16 if getattr(self.config.model, 'use_bfloat16', False) else torch.float16
        
        self.model = HookedTransformer.from_pretrained_no_processing(
            self.config.model.name,
            device=device,
            dtype=dtype
        )
        
        # SAEのロード
        if self.config.debug.verbose:
            print(f"   📥 Loading SAE: {self.config.model.sae_release}/{self.config.model.sae_id}")
        
        self.sae, _, _ = SAE.from_pretrained(
            release=self.config.model.sae_release,
            sae_id=self.config.model.sae_id,
            device=device
        )
        
        # トークナイザーを取得
        self.tokenizer = self.model.tokenizer
        
        if self.config.debug.verbose:
            print("✅ Model and SAE loaded successfully")
            if torch.cuda.is_available():
                print(f"   💾 GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated")
    
    def _collect_activation_statistics(
        self, 
        sae_features: torch.Tensor, 
        masked_features: torch.Tensor, 
        activation_info: Dict[str, Any]
    ):
        """
        マスクした特徴量の活性化統計情報を収集
        
        Args:
            sae_features: 全SAE特徴量の活性値 [batch, seq_len, n_features]
            masked_features: マスク適用後の特徴量 [batch, seq_len, n_features]
            activation_info: 統計情報を格納する辞書（破壊的更新）
        """
        # 各ターゲット特徴量の統計を計算
        feature_stats = {}
        
        for feature_id in self.intervention_feature_ids:
            # 該当特徴量の活性値を抽出 [batch, seq_len]
            activations = sae_features[:, :, feature_id]
            
            # 統計計算（0でないトークン位置のみ）
            non_zero_mask = activations > 0
            non_zero_activations = activations[non_zero_mask]
            
            if len(non_zero_activations) > 0:
                feature_stats[str(feature_id)] = {
                    "mean": float(non_zero_activations.mean().item()),
                    "max": float(non_zero_activations.max().item()),
                    "min": float(non_zero_activations.min().item()),
                    "std": float(non_zero_activations.std().item()),
                    "num_active_tokens": int(non_zero_mask.sum().item()),
                    "total_tokens": int(activations.numel()),
                    "sparsity": float(non_zero_mask.sum().item() / activations.numel())
                }
            else:
                feature_stats[str(feature_id)] = {
                    "mean": 0.0,
                    "max": 0.0,
                    "min": 0.0,
                    "std": 0.0,
                    "num_active_tokens": 0,
                    "total_tokens": int(activations.numel()),
                    "sparsity": 0.0
                }
        
        # 全体統計
        all_masked_activations = masked_features[masked_features > 0]
        
        activation_info.update({
            "per_feature": feature_stats,
            "overall": {
                "mean_across_features": float(all_masked_activations.mean().item()) if len(all_masked_activations) > 0 else 0.0,
                "max_across_features": float(all_masked_activations.max().item()) if len(all_masked_activations) > 0 else 0.0,
                "total_active_features": int((masked_features > 0).sum().item()),
                "num_intervention_features": len(self.intervention_feature_ids)
            }
        })
    
    def get_activation_summary(self) -> Dict[str, Any]:
        """
        実験全体の活性化統計サマリを取得
        
        Returns:
            全プロンプトにわたる活性化統計の集約
        """
        if not self.results:
            return {"error": "No results available. Run experiment first."}
        
        # 各特徴量の全プロンプトにわたる統計を集約
        feature_aggregated = {}
        for feature_id in self.intervention_feature_ids:
            feature_id_str = str(feature_id)
            means = []
            maxs = []
            sparsities = []
            
            for question_result in self.results:
                for variation in question_result.variations:
                    stats = variation.metadata.get("activation_stats", {})
                    per_feature = stats.get("per_feature", {})
                    
                    if feature_id_str in per_feature:
                        means.append(per_feature[feature_id_str]["mean"])
                        maxs.append(per_feature[feature_id_str]["max"])
                        sparsities.append(per_feature[feature_id_str]["sparsity"])
            
            if means:
                feature_aggregated[feature_id_str] = {
                    "avg_mean_activation": float(sum(means) / len(means)),
                    "avg_max_activation": float(sum(maxs) / len(maxs)),
                    "avg_sparsity": float(sum(sparsities) / len(sparsities)),
                    "num_prompts": len(means)
                }
        
        return {
            "num_questions": len(self.results),
            "num_prompts": sum(len(q.variations) for q in self.results),
            "num_intervention_features": len(self.intervention_feature_ids),
            "per_feature_summary": feature_aggregated
        }
    
    def create_intervention_hook(self, collect_activations: bool = True):
        """
        Geometric Subtraction (Zero-Ablation) による介入フックを作成
        
        手順:
        1. 残差ストリーム x を SAE でエンコード
        2. ターゲット特徴量以外をすべて0にマスク
        3. マスクされた活性値を使って再構成（バイアス項なし）
        4. 元の残差ストリームから減算
        
        Args:
            collect_activations: 活性化情報を収集するかどうか
        
        Returns:
            フック関数、活性化統計情報の辞書
        """
        activation_info = {}
        
        def intervention_hook(activations, hook):
            """
            Args:
                activations: 残差ストリームのテンソル [batch, seq_len, d_model]
                hook: フックポイント情報
            
            Returns:
                介入後の残差ストリーム
            """
            with torch.no_grad():
                # 1. SAEでエンコード (全特徴量の活性値を取得)
                sae_features = self.sae.encode(activations)  # [batch, seq_len, n_features]
                
                # 2. ターゲット特徴量のマスクを作成
                # ターゲット特徴量のみを1、それ以外を0にする
                mask = torch.zeros_like(sae_features)
                for feature_id in self.intervention_feature_ids:
                    mask[:, :, feature_id] = 1.0
                
                # 3. マスクを適用（ターゲット特徴量のみを残す）
                masked_features = sae_features * mask
                
                # 活性化情報の収集
                if collect_activations:
                    self._collect_activation_statistics(
                        sae_features, 
                        masked_features, 
                        activation_info
                    )
                
                # 4. マスクされた特徴量から再構成ベクトルを計算（バイアス項を除外）
                # sae.decode()を使わず、W_decとの行列積のみで再構成
                # reconstruction = masked_features @ W_dec.T
                reconstruction = torch.einsum(
                    "bsf,fd->bsd", 
                    masked_features, 
                    self.sae.W_dec
                )  # [batch, seq_len, d_model]
                
                # 5. 元の残差ストリームから減算 (Geometric Subtraction)
                intervened_activations = activations - reconstruction
                
                return intervened_activations
        
        return intervention_hook, activation_info
    
    def generate_baseline(self, prompt: str) -> str:
        """
        Baseline: 介入なしでの通常生成
        
        Args:
            prompt: 入力プロンプト
        
        Returns:
            生成されたテキスト
        """
        with torch.no_grad():
            tokens = self.model.to_tokens(prompt)
            original_length = tokens.shape[1]
            
            # 通常の生成（フックなし）
            generated_tokens = self.model.generate(
                tokens,
                max_new_tokens=self.config.generation.max_new_tokens,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                top_k=self.config.generation.top_k,
                do_sample=self.config.generation.do_sample,
                stop_at_eos=True
            )
            
            # 新規生成部分のみをデコード
            new_tokens = generated_tokens[0, original_length:]
            response_text = self.model.to_string(new_tokens)
            
            return response_text
    
    def generate_with_intervention(self, prompt: str, collect_activations: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Intervention: 介入フックを適用した状態での生成
        
        Args:
            prompt: 入力プロンプト
            collect_activations: 活性化情報を収集するかどうか
        
        Returns:
            生成されたテキスト、活性化統計情報の辞書
        """
        with torch.no_grad():
            tokens = self.model.to_tokens(prompt)
            original_length = tokens.shape[1]
            
            # 介入フックを作成
            hook_fn, activation_info = self.create_intervention_hook(collect_activations)
            hook_name = self.config.model.hook_name
            
            # フックを適用して生成
            with self.model.hooks([(hook_name, hook_fn)]):
                generated_tokens = self.model.generate(
                    tokens,
                    max_new_tokens=self.config.generation.max_new_tokens,
                    temperature=self.config.generation.temperature,
                    top_p=self.config.generation.top_p,
                    top_k=self.config.generation.top_k,
                    do_sample=self.config.generation.do_sample,
                    stop_at_eos=True
                )
            
            # 新規生成部分のみをデコード
            new_tokens = generated_tokens[0, original_length:]
            response_text = self.model.to_string(new_tokens)
            
            return response_text, activation_info
    
    def analyze_prompt_variation(self, prompt_info: FeedbackPromptInfo) -> InterventionResult:
        """
        1つのプロンプトバリエーションに対して介入実験を実行
        
        Args:
            prompt_info: プロンプト情報
        
        Returns:
            InterventionResult オブジェクト
        """
        if self.config.debug.show_prompts:
            print(f"\n📝 Prompt ({prompt_info.prompt_template_type}): {prompt_info.prompt[:100]}...")
        
        # Baseline生成
        start_time = datetime.now()
        baseline_response = self.generate_baseline(prompt_info.prompt)
        baseline_time = (datetime.now() - start_time).total_seconds() * 1000
        
        if self.config.debug.show_responses:
            print(f"   📤 Baseline: {baseline_response}")
        
        # メモリクリア
        self.optimize_memory_usage()
        
        # Intervention生成
        start_time = datetime.now()
        intervention_response, activation_info = self.generate_with_intervention(prompt_info.prompt)
        intervention_time = (datetime.now() - start_time).total_seconds() * 1000
        
        if self.config.debug.show_responses:
            print(f"   🔬 Intervention: {intervention_response}")
        
        # メタデータ
        metadata = {
            "baseline_generation_time_ms": baseline_time,
            "intervention_generation_time_ms": intervention_time,
            "baseline_response_length": len(baseline_response),
            "intervention_response_length": len(intervention_response),
            "timestamp": datetime.now().isoformat(),
            "activation_stats": activation_info  # 活性化統計情報を追加
        }
        
        if torch.cuda.is_available():
            metadata["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024**2
        
        return InterventionResult(
            prompt_info=prompt_info,
            baseline_response=baseline_response,
            intervention_response=intervention_response,
            metadata=metadata
        )
    
    def analyze_question_group(
        self, 
        question_id: int, 
        prompt_group: List[FeedbackPromptInfo]
    ) -> QuestionInterventionResult:
        """
        1つの質問（5つのバリエーション）に対して介入実験を実行
        
        Args:
            question_id: 質問ID
            prompt_group: 5つのプロンプトバリエーション
        
        Returns:
            QuestionInterventionResult オブジェクト
        """
        if self.config.debug.verbose:
            print(f"\n{'='*60}")
            print(f"Question {question_id}: Processing {len(prompt_group)} variations")
        
        variations_results = []
        
        for prompt_info in prompt_group:
            result = self.analyze_prompt_variation(prompt_info)
            variations_results.append(result)
            self.optimize_memory_usage()
        
        # 最初のプロンプトから基本情報を取得
        first_prompt = prompt_group[0]
        base_text = first_prompt.base_data.get('text', '') or first_prompt.base_data.get('question', '')
        
        return QuestionInterventionResult(
            question_id=question_id,
            dataset=first_prompt.dataset,
            base_text=base_text,
            variations=variations_results,
            timestamp=datetime.now().isoformat()
        )
    
    def run_intervention_experiment(
        self, 
        sample_size: Optional[int] = None, 
        start_index: Optional[int] = None, 
        end_index: Optional[int] = None
    ):
        """
        介入実験を実行
        
        Args:
            sample_size: 分析するサンプル数（Noneの場合はconfigから取得）
            start_index: 開始インデックス（0-based）
            end_index: 終了インデックス（0-based）
        """
        if self.config.debug.verbose:
            print("\n" + "="*60)
            print("🚀 Starting Intervention Experiment")
            print("="*60)
        
        # データロード
        feedback_data = self.load_feedback_data()
        
        # プロンプトグループ化
        prompt_groups = self.aggregate_prompts(feedback_data)
        
        total_questions = len(prompt_groups)
        
        # データ範囲の調整
        start = 0
        end = total_questions
        
        if start_index is not None or end_index is not None:
            start = start_index if start_index is not None else 0
            end = end_index if end_index is not None else total_questions
            end = min(end, total_questions)
            start = max(0, min(start, end))
        else:
            if sample_size is None:
                sample_size = self.config.data.sample_size
            if sample_size is not None:
                end = min(sample_size, total_questions)
        
        # モデルとSAEのロード
        if self.model is None or self.sae is None:
            self.load_model_and_sae()
        
        # 処理範囲を記録
        self.processed_start_id = start
        self.processed_end_id = start
        
        # 各質問グループを分析
        progress_desc = f"Processing questions ({start+1}-{end}/{total_questions})"
        try:
            for i in tqdm(range(start, end), desc=progress_desc):
                try:
                    prompt_group = prompt_groups[i]
                    result = self.analyze_question_group(i, prompt_group)
                    self.results.append(result)
                    self.processed_end_id = i
                    
                    # 定期的にメモリクリア
                    if (i + 1) % 10 == 0:
                        self.optimize_memory_usage()
                        if self.config.debug.verbose:
                            print(f"\n💾 Memory optimized at question {i+1}")
                
                except Exception as e:
                    print(f"\n❌ Error processing question {i}: {e}")
                    # エラー時は途中結果を保存
                    self.save_results(error_recovery=True)
                    raise
        
        except Exception as e:
            print(f"\n❌ Fatal error during experiment: {e}")
            print(f"📊 Processed {len(self.results)} questions before error")
            self.save_results(error_recovery=True)
            raise
        
        if self.config.debug.verbose:
            print(f"\n✅ Experiment completed: {len(self.results)} questions processed")
    
    def save_results(self, output_path: Optional[str] = None, error_recovery: bool = False):
        """
        実験結果を保存
        
        Args:
            output_path: 出力ファイルパス（Noneの場合は自動生成）
            error_recovery: エラー回復モードかどうか
        """
        if not self.results:
            print("⚠️ No results to save")
            return
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.config.model.name.replace("/", "-")
            range_suffix = f"{self.processed_start_id}-{self.processed_end_id}"
            filename = f"intervention_{model_name}_{timestamp}_{range_suffix}.json"
            if error_recovery:
                filename = f"intervention_{model_name}_{timestamp}_{range_suffix}_ERROR_RECOVERY.json"
            output_path = self.results_dir / filename
        
        # 結果を辞書に変換
        output_data = {
            "metadata": {
                "model_name": self.config.model.name,
                "sae_release": self.config.model.sae_release,
                "sae_id": self.config.model.sae_id,
                "hook_name": self.config.model.hook_name,
                "intervention_method": "Geometric Subtraction (Zero-Ablation)",
                "num_intervention_features": len(self.intervention_feature_ids),
                "num_questions": len(self.results),
                "question_id_range": {
                    "start": self.processed_start_id,
                    "end": self.processed_end_id,
                    "total_processed": len(self.results)
                },
                "error_recovery": error_recovery,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "sample_size": self.config.data.sample_size,
                    "max_new_tokens": self.config.generation.max_new_tokens,
                    "temperature": self.config.generation.temperature,
                    "do_sample": self.config.generation.do_sample,
                    "top_p": self.config.generation.top_p,
                    "top_k": self.config.generation.top_k
                }
            },
            "intervention_features": self.intervention_feature_ids,
            "activation_summary": self.get_activation_summary(),  # 活性化サマリを追加
            "results": []
        }
        
        # 各質問の結果を追加（SAE activationsは保存しない）
        for result in self.results:
            question_data = {
                "question_id": result.question_id,
                "dataset": result.dataset,
                "base_text": result.base_text,
                "timestamp": result.timestamp,
                "variations": []
            }
            
            for variation in result.variations:
                variation_data = {
                    "template_type": variation.prompt_info.prompt_template_type,
                    "prompt": variation.prompt_info.prompt,
                    "baseline_response": variation.baseline_response,
                    "intervention_response": variation.intervention_response,
                    "metadata": variation.metadata
                }
                question_data["variations"].append(variation_data)
            
            output_data["results"].append(question_data)
        
        # JSONファイルに保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        if self.config.debug.verbose:
            print(f"\n💾 Results saved to: {output_path}")
            print(f"   📊 Questions: {len(self.results)}")
            print(f"   🎯 Intervention features: {len(self.intervention_feature_ids)}")
            print(f"   📁 File size: {output_path.stat().st_size / 1024:.2f} KB")
        
        return output_path
    
    def run_complete_experiment(
        self, 
        sample_size: Optional[int] = None, 
        start_index: Optional[int] = None, 
        end_index: Optional[int] = None
    ):
        """
        実験の実行と結果保存を一括で行う
        
        Args:
            sample_size: 分析するサンプル数
            start_index: 開始インデックス（0-based）
            end_index: 終了インデックス（0-based）
        """
        self.run_intervention_experiment(
            sample_size=sample_size, 
            start_index=start_index, 
            end_index=end_index
        )
        output_path = self.save_results()
        
        if self.config.debug.verbose:
            print("\n" + "="*60)
            print("✅ Complete experiment finished successfully")
            print("="*60)
        
        return output_path
