#!/usr/bin/env python3
"""
Enhanced Features용 Random Search 명령 생성기
7 atom + 4 edge features를 사용하는 모델의 하이퍼파라미터 탐색
50개 조합으로 빠른 탐색
"""

import json
import random
import itertools

# 새로운 하이퍼파라미터 공간 (사용자 지정)
hyperparams = {
    'lr': [0.0001, 0.001],
    'dropout_ratio': [0.0, 0.3],
    'batch_size': [512],  # 고정
    'num_experts': [3, 5, 7],
    'alpha': [0.01, 0.1, 1.0],
    'beta': [0.01, 0.1, 1.0],
    'min_temp': [0.1, 1.0],
    'decay': [0.0, 0.0001]
}

def generate_random_combinations(n_samples=50):
    """Random하게 n_samples개의 하이퍼파라미터 조합을 생성"""
    combinations = []
    
    # 전체 가능한 조합 수 확인
    total_combinations = 1
    for key, values in hyperparams.items():
        total_combinations *= len(values)
    
    print(f"Total possible combinations: {total_combinations}")
    
    if n_samples >= total_combinations:
        # 요청한 샘플 수가 전체 조합보다 많으면 모든 조합 사용 (Grid Search)
        param_names = list(hyperparams.keys())
        param_values = list(hyperparams.values())
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
    else:
        # Random sampling
        random.seed(42)  # 재현 가능한 결과를 위한 seed 고정
        seen = set()
        
        while len(combinations) < n_samples:
            combo = {}
            for param, values in hyperparams.items():
                combo[param] = random.choice(values)
            
            # 중복 제거
            combo_tuple = tuple(sorted(combo.items()))
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                combinations.append(combo)
    
    return combinations

def main():
    # 데이터셋 설정 로드
    with open('configs/dataset_config.json', 'r') as f:
        dataset_config = json.load(f)
    
    # Random combinations 생성
    n_random_samples = 50
    random_combinations = generate_random_combinations(n_random_samples)
    
    print(f"Generated {len(random_combinations)} random hyperparameter combinations")
    
    # 제외할 데이터셋 정의 (큰 데이터셋)
    LARGE_DATASET_THRESHOLD = 50000
    excluded_datasets = set()
    
    # 큰 데이터셋 제외
    for name, info in dataset_config['datasets'].items():
        if info['total_samples'] >= LARGE_DATASET_THRESHOLD:
            excluded_datasets.add(name)
            print(f"Excluding large dataset: {name} ({info['total_samples']} samples)")
    
    # 탐색할 데이터셋 목록
    search_datasets = {name: info for name, info in dataset_config['datasets'].items() 
                      if name not in excluded_datasets}
    
    print(f"\nDatasets for hyperparameter search: {len(search_datasets)}")
    print(f"Excluded datasets: {len(excluded_datasets)}")
    
    # 명령어 생성
    commands = []
    total_experiments = 0
    
    for dataset_name in sorted(search_datasets.keys()):
        dataset_info = search_datasets[dataset_name]
        category = dataset_info['category']
        
        for exp_id, combo in enumerate(random_combinations):
            cmd_parts = [
                "conda run -n ADMET python workspace/src/main.py",
                f"--category {category}",
                f"--dataset_name {dataset_name}",
                f"--experiment_id enhanced_feat_exp{exp_id:03d}",  # exp000, exp001, ...
            ]
            
            # 하이퍼파라미터 추가
            for param, value in combo.items():
                cmd_parts.append(f"--{param} {value}")
            
            # 고정 파라미터
            cmd_parts.extend([
                "--num_layer 5",      # 고정
                "--emb_dim 300",      # 고정
                "--gate_dim 50",      # 고정
                "--split scaffold",   # 고정
                "--epochs 100",       # Random search용
                "--patience 20",      # Early stopping
                "--gin_pretrained_file workspace/src/pre-trained/supervised_contextpred.pth",  # Pre-trained 사용
            ])
            
            cmd = " ".join(cmd_parts)
            commands.append(cmd)
            total_experiments += 1
    
    # 명령어 파일 저장
    output_file = "workspace/commands_enhanced_features_search.txt"
    
    with open(output_file, 'w') as f:
        f.write("# Enhanced Features Hyperparameter Search Commands\n")
        f.write(f"# Total experiments: {total_experiments}\n")
        f.write(f"# Datasets: {len(search_datasets)}\n")
        f.write(f"# Combinations per dataset: {len(random_combinations)}\n")
        f.write(f"# Features: 7 atom + 4 edge\n")
        f.write(f"# Pre-trained GIN: enabled\n")
        f.write(f"# Estimated time: {total_experiments} × 1.3min = {total_experiments * 1.3 / 60:.1f} hours\n")
        f.write("\n")
        
        for cmd in commands:
            f.write(cmd + "\n")
    
    print("\n" + "="*80)
    print(f"✅ Commands saved to: {output_file}")
    print(f"✅ Total experiments: {total_experiments}")
    print(f"✅ Datasets to search: {len(search_datasets)}")
    print(f"✅ Random combinations: {len(random_combinations)}")
    print(f"✅ Estimated time: {total_experiments * 1.3 / 60:.1f} hours (~{total_experiments * 1.3 / 60 / 24:.1f} days on 4 GPUs)")
    print("="*80)
    print()
    print("To start hyperparameter search:")
    print(f"  cd /home/choi0425/workspace/ADMET && \\")
    print(f"  nohup bash -c 'tail -n +7 {output_file} | conda run --no-capture-output -n ADMET python -m simple_gpu_scheduler.scheduler --gpus 0 1 2 3' > workspace/logs/scheduler_enhanced_search.log 2>&1 &")
    print()
    
    # 하이퍼파라미터 범위 출력
    print("Hyperparameter search space:")
    for param, values in hyperparams.items():
        print(f"  {param}: {values}")
    
    return commands

if __name__ == "__main__":
    main()
