import os
import itertools

# Configuration
dataset_name = "AMES"
config_path = "configs/dataset_config.json"
output_file = "workspace/commands/ablation_commands.txt"
pretrained_path = "workspace/src/pre-trained/supervised_contextpred.pth"

# Fixed parameters
fixed_params = {
    "dataset_name": dataset_name,
    "config_path": config_path,
    "epochs": 500,
    "patience": 50,
    "gnn_type": "gin",
    "gin_pretrained_file": pretrained_path,
    "quiet": "",  # Flag
    "no_save_model": "" # Flag to prevent saving models during tuning
}

# Feature Variations
variations = [
    ("Basic", ["basic"]),
    ("Phys", ["basic", "phys"]),
    ("MACCS", ["basic", "maccs"]),
    ("ECFP", ["basic", "ecfp"]),
    ("Basic_Phys_MACCS", ["basic", "phys", "maccs"]),
    ("Basic_Phys_ECFP", ["basic", "phys", "ecfp"]),
    ("Basic_MACCS_ECFP", ["basic", "maccs", "ecfp"]),
    ("All", ["basic", "phys", "maccs", "ecfp"])
]

# Grid Search Space
# Expanded grid for more rigorous evaluation (~432 experiments total)
search_space = {
    "lr": [0.001, 0.0005, 0.0001],
    "dropout_ratio": [0.3, 0.5],
    "num_experts": [3, 5, 7],
    "num_layer": [3, 5],
    "batch_size": [32, 128],
    "emb_dim": [300]
}

def generate_commands():
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    commands = []
    
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))
    
    for var_name, features in variations:
        features_str = " ".join(features)
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            
            # Construct Experiment ID
            exp_id = f"ablation_{var_name}_{i}"
            
            cmd_parts = ["python workspace/src/main.py"]
            
            # Add fixed params
            for k, v in fixed_params.items():
                if k == "quiet" or k == "no_save_model":
                    cmd_parts.append(f"--{k}")
                else:
                    cmd_parts.append(f"--{k} {v}")
            
            # Add grid params
            for k, v in params.items():
                cmd_parts.append(f"--{k} {v}")
                
            # Add variation specific params
            cmd_parts.append(f"--features {features_str}")
            cmd_parts.append(f"--experiment_id {exp_id}")
            
            # Add output directory for organization
            cmd_parts.append(f"--output_dir workspace/output/ablation/{var_name}")
            
            commands.append(" ".join(cmd_parts))

    # Split commands into 6 files
    num_files = 6
    chunk_size = len(commands) // num_files
    remainder = len(commands) % num_files
    
    start_idx = 0
    for i in range(num_files):
        # Distribute remainder one by one to the first few files
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        chunk_commands = commands[start_idx:end_idx]
        
        chunk_file = f"workspace/commands/ablation_commands_part{i+1}.txt"
        with open(chunk_file, "w") as f:
            for cmd in chunk_commands:
                f.write(cmd + "\n")
        
        print(f"Generated {len(chunk_commands)} commands in {chunk_file}")
        start_idx = end_idx

    # Also keep the full file for reference
    with open(output_file, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")
            
    print(f"Generated {len(commands)} commands in {output_file}")
    print("Example command:")
    print(commands[0])

if __name__ == "__main__":
    generate_commands()
