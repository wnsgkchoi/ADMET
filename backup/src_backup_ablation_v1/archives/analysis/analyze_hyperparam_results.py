#!/usr/bin/env python3
"""
하이퍼파라미터 튜닝 결과 분석 및 최적 설정 추출
각 데이터셋별 최고 성능의 하이퍼파라미터를 찾아 best_configs.json 생성
"""

import os
import json
import csv
from pathlib import Path

def analyze_hyperparameter_results():
    """각 데이터셋의 하이퍼파라미터 튜닝 결과를 분석"""
    
    categories = ['Absorption', 'Distribution', 'Metabolism', 'Excretion', 'Toxicity']
    
    # 데이터셋 설정 로드
    with open('configs/dataset_config.json', 'r') as f:
        dataset_config = json.load(f)
    
    best_configs = {}
    results_summary = []
    
    print("="*80)
    print("하이퍼파라미터 튜닝 결과 분석")
    print("="*80)
    print()
    
    for category in categories:
        csv_dir = f'workspace/output/hyperparam/{category}'
        if not os.path.exists(csv_dir):
            continue
            
        for file in sorted(os.listdir(csv_dir)):
            if not file.endswith('_progress.csv'):
                continue
                
            dataset_name = file.replace('_progress.csv', '')
            csv_path = os.path.join(csv_dir, file)
            
            # CSV 파일 읽기
            try:
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    
                if len(rows) == 0:
                    print(f"⚠️  {dataset_name}: 결과 없음")
                    continue
                    
                # 숫자 변환
                for row in rows:
                    row['val_metric'] = float(row['val_metric']) if row['val_metric'] else float('inf')
                    row['test_metric'] = float(row['test_metric']) if row['test_metric'] else float('inf')
                    
            except Exception as e:
                print(f"⚠️  {dataset_name}: CSV 읽기 실패 - {e}")
                continue
            
            if len(rows) == 0:
                print(f"⚠️  {dataset_name}: 결과 없음")
                continue
            
            # 데이터셋 정보 가져오기
            if dataset_name not in dataset_config['datasets']:
                print(f"⚠️  {dataset_name}: dataset_config.json에 없음")
                continue
                
            ds_config = dataset_config['datasets'][dataset_name]
            task_type = ds_config['task_type']
            
            # 최적 모델 선택 (classification: AUROC 최대, regression: MAE 최소)
            if task_type == 'classification':
                # val_metric (AUROC)이 최대인 행
                best_row = max(rows, key=lambda x: x['val_metric'])
                metric_name = 'AUROC'
                best_val_metric = best_row['val_metric']
                best_test_metric = best_row['test_metric']
            else:  # regression
                # val_metric (MAE)이 최소인 행
                best_row = min(rows, key=lambda x: x['val_metric'])
                metric_name = 'MAE'
                best_val_metric = best_row['val_metric']
                best_test_metric = best_row['test_metric']
            
            # 최적 하이퍼파라미터 추출
            best_config = {
                'experiment_id': int(best_row['experiment_id']),
                'lr': float(best_row['lr']),
                'dropout_ratio': float(best_row['dropout_ratio']),
                'batch_size': int(best_row['batch_size']),
                'num_experts': int(best_row['num_experts']),
                'alpha': float(best_row['alpha']),
                'beta': float(best_row['beta']),
                'min_temp': float(best_row['min_temp']),
                'decay': float(best_row['decay']),
                'num_layer': int(best_row['num_layer']),
                'emb_dim': int(best_row['emb_dim']),
                'gate_dim': int(best_row['gate_dim']),
                'split_type': str(best_row['split_type']),
                'val_metric': float(best_val_metric),
                'test_metric': float(best_test_metric),
                'num_epochs_trained': int(best_row['num_epochs_trained'])
            }
            
            best_configs[dataset_name] = {
                'category': category,
                'task_type': task_type,
                'metric_name': metric_name,
                'best_config': best_config,
                'total_experiments': len(rows)
            }
            
            # 요약 정보 저장
            results_summary.append({
                'dataset': dataset_name,
                'category': category,
                'task_type': task_type,
                'metric': metric_name,
                'val_score': best_val_metric,
                'test_score': best_test_metric,
                'experiments': len(rows)
            })
            
            print(f"✓ {dataset_name:<40} {category:<15}")
            print(f"  └─ Best {metric_name}: Val={best_val_metric:.4f}, Test={best_test_metric:.4f}")
            print(f"  └─ Experiments: {len(rows)}, Best ID: {best_config['experiment_id']}")
            print()
    
    # 결과 저장
    output_path = 'configs/best_hyperparameters.json'
    with open(output_path, 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    print("="*80)
    print(f"✅ 최적 하이퍼파라미터 저장: {output_path}")
    print(f"✅ 총 {len(best_configs)}개 데이터셋의 최적 설정 추출 완료")
    print("="*80)
    
    # 요약 통계
    print("\n=== 요약 통계 ===\n")
    
    # Task type별 통계
    task_counts = {}
    for config in best_configs.values():
        task = config['task_type']
        task_counts[task] = task_counts.get(task, 0) + 1
    
    for task, count in sorted(task_counts.items()):
        print(f"{task}: {count}개 데이터셋")
    
    # Category별 통계
    print("\n=== Category별 통계 ===\n")
    category_counts = {}
    for config in best_configs.values():
        cat = config['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat, count in sorted(category_counts.items()):
        print(f"{cat}: {count}개 데이터셋")
    
    return best_configs

if __name__ == "__main__":
    best_configs = analyze_hyperparameter_results()
