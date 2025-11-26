#!/usr/bin/env python3
"""
누락된 3개 대형 데이터셋 최종 학습 명령 생성
hERG_Central_inhib, hERG_Central_10uM, hERG_Central_1uM
"""

# 기본 하이퍼파라미터 (성능이 좋은 설정 사용)
DEFAULT_CONFIG = {
    'lr': 0.001,
    'dropout_ratio': 0.0,
    'batch_size': 512,
    'num_experts': 7,
    'alpha': 0.1,
    'beta': 0.1,
    'min_temp': 1.0,
    'decay': 0.0,
    'num_layer': 5,
    'emb_dim': 300,
    'gate_dim': 50,
    'split_type': 'scaffold'
}

def generate_missing_datasets_commands():
    """누락된 대형 데이터셋 3개의 최종 학습 명령 생성"""
    
    missing_datasets = [
        "hERG_Central_inhib",
        "hERG_Central_10uM", 
        "hERG_Central_1uM"
    ]
    
    commands = []
    
    print("누락된 대형 데이터셋 최종 학습 명령 생성")
    print("=" * 50)
    
    for dataset_name in missing_datasets:
        cmd_parts = [
            "conda run -n ADMET python workspace/src/main.py",
            f"--category Toxicity",
            f"--dataset_name {dataset_name}",
            f"--experiment_id enhanced_final",
        ]
        
        # 기본 하이퍼파라미터 추가
        cmd_parts.extend([
            f"--lr {DEFAULT_CONFIG['lr']}",
            f"--dropout_ratio {DEFAULT_CONFIG['dropout_ratio']}",
            f"--batch_size {DEFAULT_CONFIG['batch_size']}",
            f"--num_experts {DEFAULT_CONFIG['num_experts']}",
            f"--alpha {DEFAULT_CONFIG['alpha']}",
            f"--beta {DEFAULT_CONFIG['beta']}",
            f"--min_temp {DEFAULT_CONFIG['min_temp']}",
            f"--decay {DEFAULT_CONFIG['decay']}",
            f"--num_layer {DEFAULT_CONFIG['num_layer']}",
            f"--emb_dim {DEFAULT_CONFIG['emb_dim']}",
            f"--gate_dim {DEFAULT_CONFIG['gate_dim']}",
            f"--split {DEFAULT_CONFIG['split_type']}",
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
        
        print(f"✅ {dataset_name} (306,893 samples)")
        print(f"  └─ lr={DEFAULT_CONFIG['lr']}, dropout={DEFAULT_CONFIG['dropout_ratio']}, experts={DEFAULT_CONFIG['num_experts']}")
    
    # 명령어 파일 저장
    output_file = "workspace/commands_missing_datasets_final.txt"
    
    with open(output_file, 'w') as f:
        f.write("# Missing Large Datasets Final Training Commands\n")
        f.write(f"# Datasets: {len(commands)} (hERG_Central series)\n")
        f.write(f"# Features: 7 atom + 4 edge\n")
        f.write(f"# Pre-trained GIN: enabled\n")
        f.write(f"# Estimated time: {len(commands)} × 60min = {len(commands)} hours (large datasets)\n")
        f.write("\n")
        
        for cmd in commands:
            f.write(cmd + "\n")
    
    print("\n" + "=" * 50)
    print(f"✅ 명령어 저장: {output_file}")
    print(f"✅ 총 {len(commands)}개 대형 데이터셋")
    print(f"✅ 예상 소요시간: {len(commands)}시간 (대형 데이터셋)")
    print("=" * 50)
    print()
    print("다음 명령어로 시작:")
    print(f"  cd /home/choi0425/workspace/ADMET && \\")
    print(f"  nohup bash -c 'tail -n +6 {output_file} | conda run --no-capture-output -n ADMET python -m simple_gpu_scheduler.scheduler --gpus 0 1 2 3' > workspace/logs/scheduler_missing_final.log 2>&1 &")
    
    return commands

if __name__ == "__main__":
    generate_missing_datasets_commands()