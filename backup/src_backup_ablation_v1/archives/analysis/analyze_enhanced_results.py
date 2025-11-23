#!/usr/bin/env python3
"""
Enhanced Features 하이퍼파라미터 결과 분석 및 최적 설정 추출
현재까지 완료된 실험들에서 최적 하이퍼파라미터를 찾아 최종 학습 명령 생성
"""

import json
import os
import pandas as pd
from pathlib import Path

def analyze_enhanced_features_results():
    """Enhanced features 실험 결과 분석"""
    
    # 결과 수집
    results = {}
    hyperparam_base = "/home/choi0425/workspace/ADMET/workspace/output/hyperparam"
    
    if not os.path.exists(hyperparam_base):
        print("❌ No hyperparameter results found!")
        return {}
    
    categories = ['Absorption', 'Distribution', 'Metabolism', 'Excretion', 'Toxicity']
    
    for category in categories:
        category_path = os.path.join(hyperparam_base, category)
        if not os.path.exists(category_path):
            continue
            
        for csv_file in os.listdir(category_path):
            if csv_file.endswith('_progress.csv'):
                dataset_name = csv_file.replace('_progress.csv', '')
                csv_path = os.path.join(category_path, csv_file)
                
                try:
                    df = pd.read_csv(csv_path)
                    
                    # enhanced_feat_ 실험만 필터링
                    enhanced_df = df[df['experiment_id'].str.contains('enhanced_feat_', na=False)]
                    
                    if len(enhanced_df) == 0:
                        continue
                    
                    # 메트릭 결정 (classification vs regression)
                    if 'AUROC' in enhanced_df.columns or enhanced_df['test_metric'].max() <= 1.0:
                        # Classification: AUROC 최대화
                        best_idx = enhanced_df['test_metric'].idxmax()
                        metric_type = 'AUROC'
                    else:
                        # Regression: MAE/MSE 최소화
                        best_idx = enhanced_df['test_metric'].idxmin()
                        metric_type = 'MAE'
                    
                    best_row = enhanced_df.loc[best_idx]
                    
                    results[dataset_name] = {
                        'category': category,
                        'best_config': {
                            'lr': best_row['lr'],
                            'dropout_ratio': best_row['dropout_ratio'],
                            'batch_size': int(best_row['batch_size']),
                            'num_experts': int(best_row['num_experts']),
                            'alpha': best_row['alpha'],
                            'beta': best_row['beta'],
                            'min_temp': best_row['min_temp'],
                            'decay': best_row['decay'],
                            'num_layer': int(best_row['num_layer']),
                            'emb_dim': int(best_row['emb_dim']),
                            'gate_dim': int(best_row['gate_dim']),
                            'split_type': best_row['split_type']
                        },
                        'best_test_metric': float(best_row['test_metric']),
                        'metric_type': metric_type,
                        'experiment_id': best_row['experiment_id'],
                        'total_experiments': len(enhanced_df)
                    }
                    
                    print(f"✅ {dataset_name:<35} {category:<12} {metric_type}: {best_row['test_metric']:.4f} ({len(enhanced_df)} experiments)")
                    
                except Exception as e:
                    print(f"❌ Error processing {dataset_name}: {e}")
    
    return results

def generate_enhanced_final_training_commands(results):
    """Enhanced features 최적 하이퍼파라미터로 최종 학습 명령 생성"""
    
    if not results:
        print("❌ No results to generate commands!")
        return []
    
    # 데이터셋 설정 로드
    with open('configs/dataset_config.json', 'r') as f:
        dataset_config = json.load(f)
    
    commands = []
    
    print("\n" + "="*80)
    print("Enhanced Features 최종 모델 학습 명령어 생성")
    print("="*80)
    print()
    
    for dataset_name in sorted(results.keys()):
        result = results[dataset_name]
        config = result['best_config']
        category = result['category']
        
        # 명령어 생성
        cmd_parts = [
            "conda run -n ADMET python workspace/src/main.py",
            f"--category {category}",
            f"--dataset_name {dataset_name}",
            f"--experiment_id enhanced_final",
        ]
        
        # 최적 하이퍼파라미터 추가
        cmd_parts.extend([
            f"--lr {config['lr']}",
            f"--dropout_ratio {config['dropout_ratio']}",
            f"--batch_size {config['batch_size']}",
            f"--num_experts {config['num_experts']}",
            f"--alpha {config['alpha']}",
            f"--beta {config['beta']}",
            f"--min_temp {config['min_temp']}",
            f"--decay {config['decay']}",
            f"--num_layer {config['num_layer']}",
            f"--emb_dim {config['emb_dim']}",
            f"--gate_dim {config['gate_dim']}",
            f"--split {config['split_type']}",
        ])
        
        # 최종 모델 설정
        cmd_parts.extend([
            "--epochs 300",
            "--patience 50",
            "--use_combined_trainvalid",
            "--gin_pretrained_file workspace/src/pre-trained/supervised_contextpred.pth"
        ])
        
        cmd = " ".join(cmd_parts)
        commands.append(cmd)
        
        # 출력
        metric_info = f"{result['metric_type']}: {result['best_test_metric']:.4f}"
        print(f"🎯 {dataset_name:<35} {category:<12} {metric_info}")
        print(f"  └─ lr={config['lr']}, dropout={config['dropout_ratio']}, experts={config['num_experts']}")
    
    # 명령어 파일 저장
    output_file = "workspace/commands_enhanced_final_training.txt"
    
    with open(output_file, 'w') as f:
        f.write("# Enhanced Features Final Model Training Commands\n")
        f.write(f"# Total: {len(commands)} datasets\n")
        f.write(f"# Features: 7 atom + 4 edge\n")
        f.write(f"# Pre-trained GIN: enabled\n")
        f.write(f"# Estimated time: {len(commands)} × 10min = {len(commands) * 10 / 60:.1f} hours\n")
        f.write("\n")
        
        for cmd in commands:
            f.write(cmd + "\n")
    
    print("\n" + "="*80)
    print(f"✅ 최종 학습 명령어 저장: {output_file}")
    print(f"✅ 총 {len(commands)}개 데이터셋")
    print(f"✅ 예상 소요 시간: {len(commands) * 10 / 60:.1f}시간")
    print("="*80)
    print()
    print("다음 명령어로 최종 학습 시작:")
    print(f"  cd /home/choi0425/workspace/ADMET && \\")
    print(f"  nohup bash -c 'tail -n +6 {output_file} | conda run --no-capture-output -n ADMET python -m simple_gpu_scheduler.scheduler --gpus 0 1 2 3' > workspace/logs/scheduler_enhanced_final.log 2>&1 &")
    
    # 최적 하이퍼파라미터 저장
    config_file = "configs/best_hyperparameters_enhanced.json"
    with open(config_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ 최적 하이퍼파라미터 저장: {config_file}")
    
    return commands

def main():
    print("Enhanced Features 하이퍼파라미터 결과 분석 시작...")
    print()
    
    # 결과 분석
    results = analyze_enhanced_features_results()
    
    if not results:
        print("\n❌ 분석할 결과가 없습니다!")
        return
    
    print(f"\n📊 총 {len(results)}개 데이터셋의 최적 하이퍼파라미터 발견")
    
    # 카테고리별 통계
    categories = {}
    for dataset, result in results.items():
        cat = result['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n카테고리별 완료 현황:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}개 데이터셋")
    
    # 최종 학습 명령 생성
    commands = generate_enhanced_final_training_commands(results)
    
    return results, commands

if __name__ == "__main__":
    main()