#!/usr/bin/env python3
"""
최적 하이퍼파라미터로 최종 모델 학습
각 데이터셋별 최고 성능의 하이퍼파라미터를 사용하여 최종 모델 학습
튜닝되지 않은 데이터셋은 기본 하이퍼파라미터 사용
"""

import json

# 기본 하이퍼파라미터 (main.py의 default 값)
DEFAULT_HYPERPARAMS = {
    'lr': 0.0001,
    'dropout_ratio': 0.5,
    'batch_size': 512,
    'num_experts': 3,
    'alpha': 0.1,
    'beta': 0.01,
    'min_temp': 1.0,
    'decay': 0.0,
    'num_layer': 5,
    'emb_dim': 300,
    'gate_dim': 50,
    'split_type': 'scaffold'
}

def generate_final_training_commands():
    """최적 하이퍼파라미터로 최종 모델 학습 명령어 생성"""
    
    # 데이터셋 설정 로드
    with open('configs/dataset_config.json', 'r') as f:
        dataset_config = json.load(f)
    
    # 최적 하이퍼파라미터 로드
    with open('configs/best_hyperparameters.json', 'r') as f:
        best_configs = json.load(f)
    
    commands = []
    
    print("="*80)
    print("최종 모델 학습 명령어 생성")
    print("="*80)
    print()
    
    # 모든 데이터셋 처리 (튜닝된 것 + 튜닝 안 된 것)
    all_datasets = dataset_config['datasets']
    
    for dataset_name in sorted(all_datasets.keys()):
        dataset_info = all_datasets[dataset_name]
        category = dataset_info['category']
        
        # 하이퍼파라미터 결정: 튜닝된 것이 있으면 사용, 없으면 기본값
        if dataset_name in best_configs:
            config = best_configs[dataset_name]['best_config']
            config_source = "tuned"
        else:
            config = DEFAULT_HYPERPARAMS.copy()
            config_source = "default"
        
        # 명령어 생성
        cmd_parts = [
            "conda run -n ADMET python workspace/src/main.py",
            f"--category {category}",
            f"--dataset_name {dataset_name}",
            f"--experiment_id final",  # 최종 모델임을 표시
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
        
        # 최종 모델은 충분한 epoch 수로 학습
        cmd_parts.append("--epochs 300")
        cmd_parts.append("--patience 50")  # Early stopping patience 증가
        
        # Train + Valid 합쳐서 최종 학습
        cmd_parts.append("--use_combined_trainvalid")
        
        # Pre-trained GIN 사용 (transfer learning)
        cmd_parts.append("--gin_pretrained_file workspace/src/pre-trained/supervised_contextpred.pth")
        
        # 체크포인트는 best model만 저장 (--ckpt_all 제거)
        # cmd_parts.append("--ckpt_all")  # 이 줄은 사용하지 않음
        
        # 에러 로그 (실패 시만)
        cmd = " ".join(cmd_parts) + f" > workspace/logs/final_train_{dataset_name}.log 2>&1"
        
        commands.append(cmd)
        
        # 출력 (config_source 표시)
        config_marker = "🎯" if config_source == "tuned" else "📋"
        print(f"{config_marker} {dataset_name:<40} {category:<15} [{config_source}]")
        print(f"  └─ lr={config['lr']}, dropout={config['dropout_ratio']}, batch={config['batch_size']}")
        print(f"  └─ experts={config['num_experts']}, alpha={config['alpha']}, beta={config['beta']}")
        print()
    
    # 명령어 파일 저장
    output_file = "workspace/commands_final_training.txt"
    
    # 통계 계산
    tuned_count = sum(1 for ds in all_datasets.keys() if ds in best_configs)
    default_count = len(all_datasets) - tuned_count
    
    with open(output_file, 'w') as f:
        f.write("# Final Model Training Commands\n")
        f.write(f"# Total: {len(commands)} datasets\n")
        f.write(f"# - Tuned hyperparameters: {tuned_count} datasets\n")
        f.write(f"# - Default hyperparameters: {default_count} datasets\n")
        f.write(f"# Estimated time: {len(commands)} × 10min = {len(commands) * 10 / 60:.1f} hours\n")
        f.write("\n")
        
        for cmd in commands:
            f.write(cmd + "\n")
    
    print("="*80)
    print(f"✅ 최종 학습 명령어 저장: {output_file}")
    print(f"✅ 총 {len(commands)}개 데이터셋")
    print(f"   - 튜닝된 하이퍼파라미터: {tuned_count}개")
    print(f"   - 기본 하이퍼파라미터: {default_count}개")
    print(f"✅ 예상 소요 시간: {len(commands) * 10 / 60:.1f}시간")
    print("="*80)
    print()
    print("다음 명령어로 학습 시작:")
    print(f"  cd /home/choi0425/workspace/ADMET && \\")
    print(f"  nohup bash -c 'tail -n +6 {output_file} | conda run --no-capture-output -n ADMET python -m simple_gpu_scheduler.scheduler --gpus 0 1 2 3' > workspace/logs/scheduler_final_training.log 2>&1 &")
    
    return commands

if __name__ == "__main__":
    generate_final_training_commands()
