#!/usr/bin/env python3
"""
Enhanced Features 최종 모델들의 Test Set 성능 분석
33개 데이터셋의 최종 성능 요약 및 비교
"""

import json
import os
import torch
import pandas as pd
from pathlib import Path

def analyze_final_model_performance():
    """최종 모델들의 성능 분석"""
    
    results = {}
    output_base = "/home/choi0425/workspace/ADMET/workspace/output"
    
    # 데이터셋 설정 로드
    with open('configs/dataset_config.json', 'r') as f:
        dataset_config = json.load(f)
    
    categories = ['Absorption', 'Distribution', 'Metabolism', 'Excretion', 'Toxicity']
    
    print("Enhanced Features 최종 모델 성능 분석")
    print("=" * 80)
    print(f"{'Dataset':<35} {'Category':<12} {'Task':<12} {'Metric':<8} {'Performance':<12} {'Size(MB)':<10}")
    print("-" * 80)
    
    # 성능 통계를 위한 변수들
    classification_metrics = []
    regression_metrics = []
    
    for category in categories:
        category_path = os.path.join(output_base, category)
        if not os.path.exists(category_path):
            continue
            
        for dataset_dir in os.listdir(category_path):
            dataset_path = os.path.join(category_path, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue
                
            model_path = os.path.join(dataset_path, "best_model.pt")
            if not os.path.exists(model_path):
                continue
            
            try:
                # 모델 로드하여 성능 확인
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                test_metric = checkpoint['test_metric']
                args_dict = checkpoint['args']
                task_type = args_dict.get('task_type', 'classification')
                
                # 파일 크기 확인
                file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                
                # 데이터셋 정보 가져오기
                dataset_info = dataset_config['datasets'].get(dataset_dir, {})
                samples = dataset_info.get('total_samples', 'Unknown')
                
                # 메트릭 타입 결정
                if task_type == 'classification':
                    metric_name = 'AUROC'
                    classification_metrics.append(test_metric)
                else:
                    metric_name = 'MAE'
                    regression_metrics.append(test_metric)
                
                results[dataset_dir] = {
                    'category': category,
                    'task_type': task_type,
                    'metric_name': metric_name,
                    'test_metric': float(test_metric),
                    'file_size_mb': file_size,
                    'total_samples': samples
                }
                
                # 출력
                performance_str = f"{test_metric:.4f}" if test_metric < 100 else f"{test_metric:.1f}"
                print(f"{dataset_dir:<35} {category:<12} {task_type:<12} {metric_name:<8} {performance_str:<12} {file_size:.1f}")
                
            except Exception as e:
                print(f"❌ Error loading {dataset_dir}: {e}")
    
    print("-" * 80)
    
    # 통계 계산
    if classification_metrics:
        avg_auroc = sum(classification_metrics) / len(classification_metrics)
        min_auroc = min(classification_metrics)
        max_auroc = max(classification_metrics)
        
        print(f"\n📊 Classification Performance (AUROC):")
        print(f"  Average: {avg_auroc:.4f}")
        print(f"  Range: {min_auroc:.4f} - {max_auroc:.4f}")
        print(f"  Datasets: {len(classification_metrics)}")
    
    if regression_metrics:
        avg_mae = sum(regression_metrics) / len(regression_metrics)
        min_mae = min(regression_metrics)
        max_mae = max(regression_metrics)
        
        print(f"\n📊 Regression Performance (MAE):")
        print(f"  Average: {avg_mae:.4f}")
        print(f"  Range: {min_mae:.4f} - {max_mae:.4f}")
        print(f"  Datasets: {len(regression_metrics)}")
    
    # 카테고리별 통계
    print(f"\n📂 Category Breakdown:")
    category_stats = {}
    for dataset, result in results.items():
        cat = result['category']
        if cat not in category_stats:
            category_stats[cat] = {'count': 0, 'classification': 0, 'regression': 0}
        category_stats[cat]['count'] += 1
        if result['task_type'] == 'classification':
            category_stats[cat]['classification'] += 1
        else:
            category_stats[cat]['regression'] += 1
    
    for cat, stats in sorted(category_stats.items()):
        print(f"  {cat}: {stats['count']} datasets ({stats['classification']} classification, {stats['regression']} regression)")
    
    print(f"\n✅ Total: {len(results)} datasets analyzed")
    
    # 최고/최저 성능 데이터셋
    print(f"\n🏆 Best Performing Datasets:")
    
    # Classification 최고 성능
    if classification_metrics:
        best_class = max(results.items(), key=lambda x: x[1]['test_metric'] if x[1]['task_type'] == 'classification' else 0)
        print(f"  Classification: {best_class[0]} (AUROC: {best_class[1]['test_metric']:.4f})")
    
    # Regression 최고 성능 (MAE 최소)
    if regression_metrics:
        best_reg = min(results.items(), key=lambda x: x[1]['test_metric'] if x[1]['task_type'] == 'regression' else float('inf'))
        print(f"  Regression: {best_reg[0]} (MAE: {best_reg[1]['test_metric']:.4f})")
    
    # 파일 크기 통계
    total_size = sum(result['file_size_mb'] for result in results.values())
    avg_size = total_size / len(results) if results else 0
    
    print(f"\n💾 Storage Statistics:")
    print(f"  Total size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print(f"  Average size: {avg_size:.1f} MB per model")
    
    # JSON으로 결과 저장
    output_file = "workspace/final_model_performance_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_datasets': len(results),
                'classification_datasets': len(classification_metrics),
                'regression_datasets': len(regression_metrics),
                'avg_auroc': avg_auroc if classification_metrics else None,
                'avg_mae': avg_mae if regression_metrics else None,
                'total_size_mb': total_size,
                'avg_size_mb': avg_size
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\n📄 Detailed results saved: {output_file}")
    
    return results

def main():
    results = analyze_final_model_performance()
    return results

if __name__ == "__main__":
    main()