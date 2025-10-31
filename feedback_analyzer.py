"""
Feedback実験用SAE分析器

このモジュールは、feedback.jsonlデータセットを使用して、LLMのフィードバックに対する
応答とその際のSAE内部状態を分析します。
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

# SAE Lens imports
from transformer_lens import HookedTransformer
from sae_lens import SAE


@dataclass
class FeedbackPromptInfo:
    """フィードバックプロンプト情報"""
    dataset: str
    prompt_template_type: str
    prompt: str
    base_data: Dict[str, Any]  # 元のbaseデータを保持


@dataclass
class FeedbackResponse:
    """1つのプロンプトに対する応答とSAE状態"""
    prompt_info: FeedbackPromptInfo
    response_text: str
    sae_activations: Dict[str, Any]  # {feature_id: activation_value}
    top_k_features: List[Tuple[int, float]]  # [(feature_id, value), ...]
    metadata: Dict[str, Any]


@dataclass
class FeedbackQuestionResult:
    """1つの質問（5つのバリエーション）の分析結果"""
    question_id: int
    dataset: str
    base_text: str
    variations: List[FeedbackResponse]
    timestamp: str


class FeedbackAnalyzer:
    """Feedback実験用のSAE分析器"""
    
    def __init__(self, config):
        """
        初期化
        
        Args:
            config: ExperimentConfig オブジェクト
        """
        self.config = config
        self.model = None
        self.sae = None
        self.results: List[FeedbackQuestionResult] = []
        
        # Feedback専用設定の取得
        self.feedback_config = getattr(config, 'feedback', None)
        if self.feedback_config is None:
            # デフォルト値を設定
            from config import FeedbackConfig
            self.feedback_config = FeedbackConfig()
        
        # 結果保存ディレクトリの作成
        self.results_dir = Path("results/feedback")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.debug.verbose:
            print("🔧 FeedbackAnalyzer initialized")
            print(f"   📁 Results directory: {self.results_dir}")
            print(f"   💾 Save all tokens: {self.feedback_config.save_all_tokens}")
            print(f"   🎯 Target layer: {self.feedback_config.target_layer}")
    
    def load_feedback_data(self, data_path: Optional[str] = None) -> List[Dict]:
        """
        feedback.jsonlファイルを読み込む
        
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
        データからプロンプト情報を作成
        
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
        データを5つのバリエーションごとにグループ化
        
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
        
        if self.config.debug.verbose:
            print(f"   🖥️  Using device: {device}")
        
        # モデルのロード
        if self.config.debug.verbose:
            print(f"   📥 Loading model: {self.config.model.name}")
        
        dtype = torch.bfloat16 if getattr(self.config.model, 'use_bfloat16', False) else torch.float16
        
        self.model = HookedTransformer.from_pretrained(
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
        
        if self.config.debug.verbose:
            print("✅ Model and SAE loaded successfully")
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                print(f"   💾 GPU Memory: {memory_allocated:.2f} GB")
    
    def generate_with_sae(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        プロンプトに対して生成を実行し、SAE活性化を取得
        
        Args:
            prompt: 入力プロンプト
        
        Returns:
            (生成テキスト, SAE活性化情報)
        """
        # トークン化
        tokens = self.model.to_tokens(prompt)
        
        # キャッシュ付きで生成実行
        with torch.no_grad():
            # 生成実行
            generated_tokens = self.model.generate(
                tokens,
                max_new_tokens=self.config.generation.max_new_tokens,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                top_k=self.config.generation.top_k,
                do_sample=self.config.generation.do_sample,
                repetition_penalty=self.config.generation.repetition_penalty,
                stop_at_eos=True
            )
            
            # 生成テキストをデコード
            response_text = self.model.to_string(generated_tokens[0])
            
            # SAE活性化を取得するため、再度フォワードパス実行
            _, cache = self.model.run_with_cache(generated_tokens)
            
            # 対象レイヤーのフック名を取得
            hook_name = self.sae.cfg.hook_name
            
            # 活性化を取得
            activations = cache[hook_name]  # shape: [batch, seq_len, d_model]
            
            # SAEエンコード
            sae_features = self.sae.encode(activations)  # shape: [batch, seq_len, n_features]
            
            # トークン保存設定に応じて処理
            if self.feedback_config.save_all_tokens:
                # 全トークンの活性化を保存
                sae_activations_np = sae_features[0].cpu().numpy()  # [seq_len, n_features]
            else:
                # 最後のトークンのみ保存
                sae_activations_np = sae_features[0, -1:].cpu().numpy()  # [1, n_features]
            
            # Top-k特徴を抽出
            if self.feedback_config.save_all_tokens:
                # 全トークンの平均を取る
                mean_activations = sae_activations_np.mean(axis=0)
            else:
                mean_activations = sae_activations_np[0]
            
            top_k_indices = np.argsort(mean_activations)[-self.config.analysis.top_k_features:][::-1]
            top_k_features = [(int(idx), float(mean_activations[idx])) for idx in top_k_indices]
            
            # 閾値以上の特徴のみ保存
            active_features = {}
            threshold = self.config.analysis.activation_threshold
            
            if self.feedback_config.save_all_tokens:
                # 各トークン位置での活性化を保存
                for token_idx in range(sae_activations_np.shape[0]):
                    token_activations = sae_activations_np[token_idx]
                    active_indices = np.where(token_activations > threshold)[0]
                    if len(active_indices) > 0:
                        active_features[f"token_{token_idx}"] = {
                            int(idx): float(token_activations[idx]) 
                            for idx in active_indices
                        }
            else:
                # 最後のトークンのみ
                token_activations = sae_activations_np[0]
                active_indices = np.where(token_activations > threshold)[0]
                active_features["last_token"] = {
                    int(idx): float(token_activations[idx]) 
                    for idx in active_indices
                }
            
            sae_info = {
                "hook_name": hook_name,
                "activations": active_features,
                "top_k_features": top_k_features,
                "num_active_features": sum(len(v) for v in active_features.values()),
                "save_all_tokens": self.feedback_config.save_all_tokens,
                "num_tokens": sae_activations_np.shape[0]
            }
        
        return response_text, sae_info
    
    def analyze_prompt_variation(self, prompt_info: FeedbackPromptInfo) -> FeedbackResponse:
        """
        1つのプロンプトバリエーションを分析
        
        Args:
            prompt_info: プロンプト情報
        
        Returns:
            FeedbackResponse オブジェクト
        """
        if self.config.debug.show_prompts:
            print(f"\n📝 Prompt ({prompt_info.prompt_template_type}):")
            print(f"   {prompt_info.prompt[:100]}...")
        
        # 生成実行
        start_time = datetime.now()
        response_text, sae_info = self.generate_with_sae(prompt_info.prompt)
        end_time = datetime.now()
        
        if self.config.debug.show_responses:
            print(f"💬 Response:")
            print(f"   {response_text[:200]}...")
        
        # メタデータ
        metadata = {
            "generation_time_ms": (end_time - start_time).total_seconds() * 1000,
            "response_length": len(response_text),
            "timestamp": datetime.now().isoformat()
        }
        
        if torch.cuda.is_available():
            metadata["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1e6
        
        return FeedbackResponse(
            prompt_info=prompt_info,
            response_text=response_text,
            sae_activations=sae_info["activations"],
            top_k_features=sae_info["top_k_features"],
            metadata=metadata
        )
    
    def analyze_question_group(
        self, 
        question_id: int, 
        prompt_group: List[FeedbackPromptInfo]
    ) -> FeedbackQuestionResult:
        """
        1つの質問（5つのバリエーション）を分析
        
        Args:
            question_id: 質問ID
            prompt_group: 5つのプロンプトバリエーション
        
        Returns:
            FeedbackQuestionResult オブジェクト
        """
        if self.config.debug.verbose:
            print(f"\n{'='*60}")
            print(f"📊 Analyzing Question {question_id} ({len(prompt_group)} variations)")
            print(f"{'='*60}")
        
        variations_results = []
        
        for prompt_info in prompt_group:
            response = self.analyze_prompt_variation(prompt_info)
            variations_results.append(response)
        
        # 最初のプロンプトから基本情報を取得
        first_prompt = prompt_group[0]
        base_text = first_prompt.base_data.get('text', '') or first_prompt.base_data.get('question', '')
        
        return FeedbackQuestionResult(
            question_id=question_id,
            dataset=first_prompt.dataset,
            base_text=base_text,
            variations=variations_results,
            timestamp=datetime.now().isoformat()
        )
    
    def run_analysis(self, sample_size: Optional[int] = None):
        """
        完全な分析を実行
        
        Args:
            sample_size: 分析するサンプル数（Noneの場合はconfigから取得）
        """
        if self.config.debug.verbose:
            print("\n" + "="*60)
            print("🚀 Starting Feedback Analysis")
            print("="*60)
        
        # データロード
        feedback_data = self.load_feedback_data()
        
        # プロンプトグループ化
        prompt_groups = self.aggregate_prompts(feedback_data)
        
        # サンプルサイズ調整
        if sample_size is None:
            sample_size = self.config.data.sample_size
        
        if sample_size is not None and sample_size < len(prompt_groups):
            prompt_groups = prompt_groups[:sample_size]
            if self.config.debug.verbose:
                print(f"📊 Analyzing {sample_size} questions (out of {len(prompt_groups)} total)")
        
        # モデルとSAEのロード
        if self.model is None or self.sae is None:
            self.load_model_and_sae()
        
        # 各質問グループを分析
        for question_id, prompt_group in enumerate(tqdm(prompt_groups, desc="Analyzing questions")):
            result = self.analyze_question_group(question_id, prompt_group)
            self.results.append(result)
            
            # メモリクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if self.config.debug.verbose:
            print("\n" + "="*60)
            print("✅ Analysis Complete")
            print("="*60)
            print(f"📊 Processed {len(self.results)} questions")
            print(f"💾 Total variations: {sum(len(r.variations) for r in self.results)}")
    
    def save_results(self, output_path: Optional[str] = None):
        """
        分析結果を保存
        
        Args:
            output_path: 出力ファイルパス（Noneの場合は自動生成）
        """
        if not self.results:
            print("⚠️ No results to save")
            return
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.config.model.name.replace("/", "_")
            output_path = self.results_dir / f"feedback_analysis_{model_name}_{timestamp}.json"
        
        # 結果を辞書に変換
        output_data = {
            "metadata": {
                "model_name": self.config.model.name,
                "sae_release": self.config.model.sae_release,
                "sae_id": self.config.model.sae_id,
                "num_questions": len(self.results),
                "save_all_tokens": self.feedback_config.save_all_tokens,
                "target_layer": self.feedback_config.target_layer,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "sample_size": self.config.data.sample_size,
                    "max_new_tokens": self.config.generation.max_new_tokens,
                    "temperature": self.config.generation.temperature,
                    "top_k_features": self.config.analysis.top_k_features
                }
            },
            "results": []
        }
        
        # 各質問の結果を追加
        for result in self.results:
            question_data = {
                "question_id": result.question_id,
                "dataset": result.dataset,
                "base_text": result.base_text[:200] + "..." if len(result.base_text) > 200 else result.base_text,
                "variations": []
            }
            
            for variation in result.variations:
                variation_data = {
                    "template_type": variation.prompt_info.prompt_template_type,
                    "prompt": variation.prompt_info.prompt if self.config.debug.show_prompts else "[hidden]",
                    "response": variation.response_text,
                    "sae_activations": variation.sae_activations,
                    "top_k_features": variation.top_k_features,
                    "metadata": variation.metadata
                }
                question_data["variations"].append(variation_data)
            
            output_data["results"].append(question_data)
        
        # JSONファイルに保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        if self.config.debug.verbose:
            print(f"\n💾 Results saved to: {output_path}")
            file_size = os.path.getsize(output_path) / 1024 / 1024
            print(f"   📦 File size: {file_size:.2f} MB")
    
    def run_complete_analysis(self, sample_size: Optional[int] = None):
        """
        分析の実行と結果保存を一括で行う
        
        Args:
            sample_size: 分析するサンプル数
        """
        self.run_analysis(sample_size=sample_size)
        self.save_results()
        
        if self.config.debug.verbose:
            print("\n🎉 Complete analysis finished!")
