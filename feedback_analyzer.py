"""
Feedback実験用SAE分析器

このモジュールは、feedback.jsonlデータセットを使用して、LLMのフィードバックに対する
応答とその際のSAE内部状態を分析します。
"""

import json
import os
import gc
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

# メモリ効率化のための環境変数設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

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
        
        # 処理範囲を記録（ファイル名・ログ用）
        self.processed_start_id = None
        self.processed_end_id = None
        
        if self.config.debug.verbose:
            print("🔧 FeedbackAnalyzer initialized")
            print(f"   📁 Results directory: {self.results_dir}")
            print(f"   💾 Prompt tokens: {'全プロンプトトークン' if self.feedback_config.save_all_tokens else 'プロンプト最終トークンのみ（推奨）'}")
            print(f"   💬 Response tokens: 最初の{self.feedback_config.response_tokens_to_capture}トークン{'（取得する）' if self.feedback_config.response_tokens_to_capture > 0 else '（取得しない）'}")
            print(f"   🎯 Target layer: {self.feedback_config.target_layer}")
            print(f"   📍 分析位置: A) 応答生成直前（意図）+ B) 応答最初の数トークン（実行）")
            
    def get_model_device(self) -> str:
        """モデルの現在のデバイスを安全に取得"""
        if self.model is None:
            return self.device
        try:
            first_param = next(self.model.parameters())
            return str(first_param.device)
        except (StopIteration, AttributeError):
            return self.device

    def get_current_sae_device(self) -> str:
        """SAEの現在のデバイスを取得"""
        if self.sae is None:
            return self.device
        try:
            first_param = next(self.sae.parameters())
            return str(first_param.device)
        except (StopIteration, AttributeError):
            return self.sae_device if self.sae_device else self.device

    def ensure_device_consistency(self, tensor: torch.Tensor) -> torch.Tensor:
        """テンソルをSAEと同じデバイスに移動"""
        if self.sae is None:
            return tensor
        sae_device = self.get_current_sae_device()
        if str(tensor.device) != sae_device:
            tensor = tensor.to(sae_device)
        return tensor

    def optimize_memory_usage(self):
        """メモリ使用量を最適化"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if self.config.debug.verbose:
                    memory_allocated = torch.cuda.memory_allocated() / 1e9
                    memory_reserved = torch.cuda.memory_reserved() / 1e9
                    print(f"💾 GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            gc.collect()
        except Exception as e:
            if self.config.debug.verbose:
                print(f"⚠️ メモリ最適化中に警告: {e}")

    def force_clear_gpu_cache(self):
        """GPUキャッシュを強制的にクリア"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            if self.config.debug.verbose:
                print("🧹 GPU cache cleared")
        except Exception as e:
            if self.config.debug.verbose:
                print(f"⚠️ GPU cache clear warning: {e}")    
    
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
        
        # SAEのデバイスを記録
        self.sae_device = str(device)
        
        # トークナイザーを取得
        self.tokenizer = self.model.tokenizer
        
        if self.config.debug.verbose:
            print("✅ Model and SAE loaded successfully")
            print(f"   🎯 Model device: {self.get_model_device()}")
            print(f"   🎯 SAE device: {self.get_current_sae_device()}")
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                print(f"   💾 GPU Memory: {memory_allocated:.2f} GB")
    
    def generate_with_sae(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        プロンプトに対して生成を実行し、SAE活性化を取得
        
        分析位置（ML学習とステップ4介入実験のため）:
        A. プロンプト最後のトークン（応答生成直前の「意図・計画」状態） - 常に取得
        B. 応答の最初の数トークン（迎合的応答の「実行・維持」状態） - オプション
        
        データ保存方針（ステップ3のML学習用）:
        - SAEの活性化値が0より大きい全ての特徴を保存（疎ベクトル形式）
        - 閾値による事前フィルタリングは行わない（XGBoostの特徴選択に委ねる）
        - これにより、SHAP分析で全特徴の寄与度を正確に評価可能
        
        保存形式例:
        {
            "prompt_last_token": {  # A: 応答生成直前の「意図」状態
                "15": 0.523,
                "1024": 3.217,
                ...
            },
            "response_token_0": {  # B: 応答1トークン目の「実行」状態
                "23": 0.412,
                "2048": 1.853,
                ...
            },
            "response_token_1": { ... },  # 応答2トークン目
            ...  # response_tokens_to_captureの設定値まで
        }
        
        Args:
            prompt: 入力プロンプト
        
        Returns:
            (生成テキスト, SAE活性化情報)
        """
        # トークン化
        tokens = self.model.to_tokens(prompt)
        original_length = tokens.shape[1]  # 元のプロンプトのトークン数を記録
        
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
                # repetition_penalty=self.config.generation.repetition_penalty,
                stop_at_eos=True
            )
            
            # 新規生成された部分のみを取り出してデコード
            new_tokens = generated_tokens[0, original_length:]  # プロンプト部分を除外
            response_text = self.model.to_string(new_tokens)
            num_response_tokens = new_tokens.shape[0]
            
            # === A. プロンプト最後のトークンのSAE活性化を取得（応答生成直前の「意図」状態） ===
            # プロンプトのみでフォワードパスを実行
            _, prompt_cache = self.model.run_with_cache(tokens)
            
            # 対象レイヤーのフック名を取得
            hook_name = self.config.model.hook_name
            
            # プロンプトの活性化を取得
            prompt_activations = prompt_cache[hook_name]  # shape: [batch, seq_len, d_model]
            
            # SAEエンコード（プロンプト）
            prompt_sae_features = self.sae.encode(prompt_activations)  # shape: [batch, seq_len, n_features]
            
            # === B. 応答の最初の数トークンのSAE活性化を取得（迎合的応答の「実行」状態） ===
            response_sae_features_list = []
            num_tokens_to_capture = min(
                self.feedback_config.response_tokens_to_capture,
                num_response_tokens
            )
            
            if num_tokens_to_capture > 0:
                # 応答トークンを1つずつ追加しながらフォワードパスを実行
                for i in range(num_tokens_to_capture):
                    # プロンプト + 応答の最初のi+1トークン
                    tokens_with_response = generated_tokens[0, :original_length + i + 1].unsqueeze(0)
                    _, response_cache = self.model.run_with_cache(tokens_with_response)
                    
                    # 応答トークン位置（プロンプト後の最後のトークン）の活性化を取得
                    response_activations = response_cache[hook_name][:, -1:, :]  # 最後のトークンのみ
                    response_sae_feature = self.sae.encode(response_activations)  # [1, 1, n_features]
                    response_sae_features_list.append(response_sae_feature[0, 0].cpu().numpy())
            
            # 統合: プロンプトと応答のSAE特徴
            sae_features = prompt_sae_features  # 既存コードとの互換性のため
            
            # === プロンプトトークンの保存設定に応じて処理 ===
            if self.feedback_config.save_all_tokens:
                # 全プロンプトトークンの活性化を保存
                prompt_sae_activations_np = prompt_sae_features[0].cpu().numpy()  # [seq_len, n_features]
            else:
                # プロンプトの最後のトークンのみ保存（デフォルト、推奨）
                # これが応答の最初のトークン生成直前の状態
                prompt_sae_activations_np = prompt_sae_features[0, -1:].cpu().numpy()  # [1, n_features]
            
            # 既存コードとの互換性のため
            sae_activations_np = prompt_sae_activations_np
            
            # Top-k特徴を抽出（ログ・可視化用、ML学習には使用しない）
            if self.feedback_config.save_all_tokens:
                # 全トークンの平均を取る
                mean_activations = sae_activations_np.mean(axis=0)
            else:
                # プロンプト最後のトークン（推奨）
                mean_activations = sae_activations_np[0]
            
            top_k_indices = np.argsort(mean_activations)[-self.config.analysis.top_k_features:][::-1]
            top_k_features = [(int(idx), float(mean_activations[idx])) for idx in top_k_indices]
            
            # 0より大きい全ての活性化を保存（ML学習用の疎ベクトル）
            # 重要: 閾値による事前フィルタリングは行わず、XGBoostの特徴選択に委ねる
            active_features = {}
            
            if self.feedback_config.save_all_tokens:
                # 各プロンプトトークン位置での活性化を保存
                for token_idx in range(sae_activations_np.shape[0]):
                    token_activations = sae_activations_np[token_idx]
                    # 0より大きい全ての活性化を保存（疎ベクトル）
                    active_indices = np.where(token_activations > 0)[0]
                    if len(active_indices) > 0:  # 活性化がある場合のみ保存
                        active_features[f"token_{token_idx}"] = {
                            int(idx): float(token_activations[idx]) 
                            for idx in active_indices
                        }
            else:
                # プロンプト最後のトークンのみ（推奨、迎合性分析に最適）
                token_activations = sae_activations_np[0]
                # 0より大きい全ての活性化を保存（疎ベクトル）
                active_indices = np.where(token_activations > 0)[0]
                active_features["prompt_last_token"] = {
                    int(idx): float(token_activations[idx]) 
                    for idx in active_indices
                }
            
            # === 応答の最初の数トークンのSAE活性化を追加 ===
            for i, response_sae_np in enumerate(response_sae_features_list):
                # 0より大きい全ての活性化を保存（疎ベクトル）
                active_indices = np.where(response_sae_np > 0)[0]
                if len(active_indices) > 0:
                    active_features[f"response_token_{i}"] = {
                        int(idx): float(response_sae_np[idx]) 
                        for idx in active_indices
                    }
            
            sae_info = {
                "hook_name": hook_name,
                "activations": active_features,  # 0より大きい全活性化（疎ベクトル、ML学習用）
                "top_k_features": top_k_features,  # ログ・可視化用（ML学習には不使用）
                "num_active_features": sum(len(v) for v in active_features.values()),
                "save_all_tokens": self.feedback_config.save_all_tokens,
                "num_tokens": sae_activations_np.shape[0],
                "analyzed_position": "prompt_last_token" if not self.feedback_config.save_all_tokens else "all_prompt_tokens",
                "response_tokens_captured": len(response_sae_features_list),  # 取得した応答トークン数
                "num_response_tokens": num_response_tokens,  # 生成された応答トークンの総数
                "data_format": "sparse_vector",  # データ形式: 疎ベクトル（活性化>0の特徴のみ保存）
                "total_sae_features": prompt_sae_features.shape[-1],  # SAEの全特徴数（例: 16384）
                "capture_positions": {
                    "prompt_last_token": "応答生成直前の意図・計画状態",
                    "response_tokens": f"応答の最初の{len(response_sae_features_list)}トークン（迎合的応答の実行・維持状態）" if response_sae_features_list else "取得なし"
                }
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
    
    def run_analysis(self, sample_size: Optional[int] = None, start_index: Optional[int] = None, end_index: Optional[int] = None):
        """
        完全な分析を実行
        
        Args:
            sample_size: 分析するサンプル数（Noneの場合はconfigから取得）
            start_index: 開始インデックス（0-based、Noneの場合は0から開始）
            end_index: 終了インデックス（0-based、Noneの場合は最後まで）
        
        Note:
            - sample_sizeとstart_index/end_indexを同時に指定した場合、start_index/end_indexが優先されます
            - 例: start_index=100, end_index=500 で101個目から500個目を取得（0-indexedのため）
        """
        if self.config.debug.verbose:
            print("\n" + "="*60)
            print("🚀 Starting Feedback Analysis")
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
            # start_index/end_indexが指定されている場合
            start = start_index if start_index is not None else 0
            end = end_index if end_index is not None else total_questions
            prompt_groups_to_process = prompt_groups[start:end]
            if self.config.debug.verbose:
                print(f"📊 Analyzing questions {start+1} to {end} (total: {len(prompt_groups_to_process)} questions out of {total_questions})")
        else:
            # sample_sizeによる調整（従来の動作）
            if sample_size is None:
                sample_size = self.config.data.sample_size
            
            if sample_size is not None and sample_size < len(prompt_groups):
                prompt_groups_to_process = prompt_groups[:sample_size]
                end = sample_size
                if self.config.debug.verbose:
                    print(f"📊 Analyzing {sample_size} questions (out of {total_questions} total)")
            else:
                prompt_groups_to_process = prompt_groups
        
        # モデルとSAEのロード
        if self.model is None or self.sae is None:
            self.load_model_and_sae()
        
        # 処理範囲を記録
        self.processed_start_id = start
        self.processed_end_id = start  # 初期値は開始位置
        
        # 各質問グループを分析
        # プログレスバーに全体の問題数に対する進行状況を表示
        progress_desc = f"Processing questions ({start+1}-{end}/{total_questions})"
        try:
            for idx, prompt_group in enumerate(tqdm(prompt_groups_to_process, desc=progress_desc)):
                # 実際の質問IDは開始位置を考慮
                actual_question_id = start + idx
                
                try:
                    result = self.analyze_question_group(actual_question_id, prompt_group)
                    self.results.append(result)
                    
                    # 処理完了した最後のquestion_idを更新
                    self.processed_end_id = actual_question_id
                    
                    # メモリ最適化を実行
                    if hasattr(self, 'optimize_memory_usage'):
                        self.optimize_memory_usage()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    # CUDAメモリエラーなどをキャッチ
                    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                        print(f"\n⚠️ メモリエラーが発生しました: {e}")
                        print(f"💾 Question ID {self.processed_start_id} から {self.processed_end_id} までの結果を保存します...")
                        # エラー発生時に現在までの結果を保存
                        self.save_results(error_recovery=True)
                        raise  # エラーを再度発生させて処理を停止
                    else:
                        raise  # その他のエラーはそのまま再発生
        
        except Exception as e:
            # その他の予期しないエラーもキャッチして保存
            if self.results:  # 結果がある場合のみ保存
                print(f"\n⚠️ エラーが発生しました: {e}")
                print(f"💾 Question ID {self.processed_start_id} から {self.processed_end_id} までの結果を保存します...")
                self.save_results(error_recovery=True)
            raise
        
        if self.config.debug.verbose:
            print("\n" + "="*60)
            print("✅ Analysis Complete")
            print("="*60)
            print(f"📊 Processed {len(self.results)} questions")
            print(f"💾 Total variations: {sum(len(r.variations) for r in self.results)}")
            print(f"🎯 Question ID range: {self.processed_start_id} to {self.processed_end_id}")
    
    def save_results(self, output_path: Optional[str] = None, error_recovery: bool = False):
        """
        分析結果を保存
        
        Args:
            output_path: 出力ファイルパス（Noneの場合は自動生成）
            error_recovery: エラー回復モードかどうか（メモリエラー等で途中保存する場合True）
        """
        if not self.results:
            print("⚠️ No results to save")
            return
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.config.model.name.replace("/", "_")
            
            # ファイル名にquestion_id範囲を追加
            if self.processed_start_id is not None and self.processed_end_id is not None:
                range_str = f"{self.processed_start_id}-{self.processed_end_id}"
                prefix = "feedback_analysis_partial" if error_recovery else "feedback_analysis"
                output_path = self.results_dir / f"{prefix}_{model_name}_{timestamp}_{range_str}.json"
            else:
                output_path = self.results_dir / f"feedback_analysis_{model_name}_{timestamp}.json"
        
        # 結果を辞書に変換
        output_data = {
            "metadata": {
                "model_name": self.config.model.name,
                "sae_release": self.config.model.sae_release,
                "sae_id": self.config.model.sae_id,
                "num_questions": len(self.results),
                "question_id_range": {
                    "start": self.processed_start_id,
                    "end": self.processed_end_id,
                    "total_processed": len(self.results)
                },
                "error_recovery": error_recovery,
                "save_all_tokens": self.feedback_config.save_all_tokens,
                "response_tokens_captured": self.feedback_config.response_tokens_to_capture,
                "analysis_position": {
                    "prompt": "prompt_last_token (応答生成直前の意図)" if not self.feedback_config.save_all_tokens else "all_prompt_tokens",
                    "response": f"最初の{self.feedback_config.response_tokens_to_capture}トークン（迎合的応答の実行・維持）" if self.feedback_config.response_tokens_to_capture > 0 else "取得なし"
                },
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
            if self.processed_start_id is not None and self.processed_end_id is not None:
                print(f"   🎯 Question ID range: {self.processed_start_id} to {self.processed_end_id}")
            if error_recovery:
                print(f"   ⚠️ This is a partial save due to error recovery")
        
        return output_path
    
    def run_complete_analysis(self, sample_size: Optional[int] = None, start_index: Optional[int] = None, end_index: Optional[int] = None):
        """
        分析の実行と結果保存を一括で行う
        
        Args:
            sample_size: 分析するサンプル数
            start_index: 開始インデックス（0-based）
            end_index: 終了インデックス（0-based）
        """
        self.run_analysis(sample_size=sample_size, start_index=start_index, end_index=end_index)
        self.save_results()
        
        if self.config.debug.verbose:
            print("\n🎉 Complete analysis finished!")
