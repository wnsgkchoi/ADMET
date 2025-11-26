import json
import os

def generate_commands():
    config_path = 'configs/dataset_config.json'
    output_file = 'workspace/commands/optuna_commands.txt'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    datasets = config['datasets']
    commands = []
    
    # Exclude AMES as it is already done (or include if we want to re-optimize)
    # User said "AMES를 제외한 타 32개의 데이터셋"
    
    for name, info in datasets.items():
        if name == 'AMES':
            continue
            
        # Command 1: Basic Features
        cmd_basic = f"python workspace/src/scripts/optimize_with_optuna.py --dataset_name {name} --features basic --n_trials 50 --output_dir workspace/output/optimization"
        commands.append(cmd_basic)
        
        # Command 2: Basic + Phys Features (Late Fusion)
        cmd_phys = f"python workspace/src/scripts/optimize_with_optuna.py --dataset_name {name} --features basic phys --n_trials 50 --output_dir workspace/output/optimization"
        commands.append(cmd_phys)
        
    # Save commands
    with open(output_file, 'w') as f:
        for cmd in commands:
            f.write(cmd + "\n")
            
    print(f"Generated {len(commands)} commands in {output_file}")

if __name__ == "__main__":
    generate_commands()
