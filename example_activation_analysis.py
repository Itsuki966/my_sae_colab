"""
介入実験における活性化統計の分析例

このスクリプトでは、介入実験で収集された活性化統計を分析する方法を示します。
"""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_intervention_results(filepath: str) -> Dict:
    """介入実験の結果を読み込み"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_feature_activations(results: Dict) -> pd.DataFrame:
    """
    特徴量ごとの活性化統計をDataFrameにまとめる
    
    Returns:
        各特徴量の統計情報を含むDataFrame
    """
    activation_summary = results.get('activation_summary', {})
    per_feature = activation_summary.get('per_feature_summary', {})
    
    rows = []
    for feature_id, stats in per_feature.items():
        rows.append({
            'feature_id': int(feature_id),
            'avg_mean_activation': stats['avg_mean_activation'],
            'avg_max_activation': stats['avg_max_activation'],
            'avg_sparsity': stats['avg_sparsity'],
            'num_prompts': stats['num_prompts']
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('avg_mean_activation', ascending=False)
    return df


def analyze_per_prompt_activations(results: Dict) -> pd.DataFrame:
    """
    プロンプトごとの活性化統計をDataFrameにまとめる
    
    Returns:
        各プロンプトの統計情報を含むDataFrame
    """
    rows = []
    
    for result in results['results']:
        question_id = result['question_id']
        dataset = result['dataset']
        
        for variation in result['variations']:
            template_type = variation['template_type']
            activation_stats = variation['metadata'].get('activation_stats', {})
            overall = activation_stats.get('overall', {})
            
            rows.append({
                'question_id': question_id,
                'dataset': dataset,
                'template_type': template_type,
                'mean_activation': overall.get('mean_across_features', 0.0),
                'max_activation': overall.get('max_across_features', 0.0),
                'total_active_features': overall.get('total_active_features', 0),
                'num_intervention_features': overall.get('num_intervention_features', 0)
            })
    
    return pd.DataFrame(rows)


def find_most_active_features(results: Dict, top_k: int = 10) -> List[Dict]:
    """
    最も活性化した特徴量をトップK個取得
    
    Args:
        results: 介入実験の結果
        top_k: 上位何個を取得するか
    
    Returns:
        上位K個の特徴量情報
    """
    df = analyze_feature_activations(results)
    top_features = df.head(top_k)
    
    return top_features.to_dict('records')


def compare_template_activations(results: Dict) -> pd.DataFrame:
    """
    テンプレートタイプごとの活性化を比較
    
    Returns:
        テンプレートタイプ別の平均統計
    """
    df = analyze_per_prompt_activations(results)
    
    comparison = df.groupby('template_type').agg({
        'mean_activation': ['mean', 'std'],
        'max_activation': ['mean', 'std'],
        'total_active_features': ['mean', 'std']
    }).round(3)
    
    return comparison


def print_activation_summary(results: Dict):
    """活性化統計のサマリを表示"""
    activation_summary = results.get('activation_summary', {})
    
    print("=" * 60)
    print("活性化統計サマリ")
    print("=" * 60)
    print(f"質問数: {activation_summary.get('num_questions', 0)}")
    print(f"総プロンプト数: {activation_summary.get('num_prompts', 0)}")
    print(f"介入特徴量数: {activation_summary.get('num_intervention_features', 0)}")
    print()
    
    # 最も活性化した特徴量トップ10
    print("最も活性化した特徴量 (Top 10):")
    print("-" * 60)
    
    top_features = find_most_active_features(results, top_k=10)
    for i, feature in enumerate(top_features, 1):
        print(f"{i}. Feature {feature['feature_id']}:")
        print(f"   平均活性値: {feature['avg_mean_activation']:.3f}")
        print(f"   最大活性値: {feature['avg_max_activation']:.3f}")
        print(f"   スパース性: {feature['avg_sparsity']:.3f}")
    
    print()
    
    # テンプレート別比較
    print("テンプレートタイプ別の活性化比較:")
    print("-" * 60)
    comparison = compare_template_activations(results)
    print(comparison)


def main():
    """メイン関数 - 使用例"""
    
    # 結果ファイルのパスを指定
    # 例: results/intervention/intervention_gemma-2-9b-it_20251208_120000_0-99.json
    results_dir = Path("results/intervention")
    
    # 最新の結果ファイルを取得
    result_files = list(results_dir.glob("intervention_*.json"))
    if not result_files:
        print("介入実験の結果ファイルが見つかりません。")
        return
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 分析対象ファイル: {latest_file.name}\n")
    
    # 結果を読み込み
    results = load_intervention_results(str(latest_file))
    
    # サマリを表示
    print_activation_summary(results)
    
    # DataFrameとして詳細分析
    print("\n" + "=" * 60)
    print("詳細分析 (DataFrame)")
    print("=" * 60)
    
    # 特徴量ごとの統計
    df_features = analyze_feature_activations(results)
    print("\n特徴量ごとの統計 (上位5つ):")
    print(df_features.head())
    
    # プロンプトごとの統計
    df_prompts = analyze_per_prompt_activations(results)
    print("\nプロンプトごとの統計 (最初の5つ):")
    print(df_prompts.head())
    
    # CSVに保存
    output_dir = Path("results/intervention/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_csv = output_dir / f"feature_activations_{latest_file.stem}.csv"
    prompts_csv = output_dir / f"prompt_activations_{latest_file.stem}.csv"
    
    df_features.to_csv(features_csv, index=False)
    df_prompts.to_csv(prompts_csv, index=False)
    
    print(f"\n💾 分析結果を保存:")
    print(f"   {features_csv}")
    print(f"   {prompts_csv}")


if __name__ == "__main__":
    main()
