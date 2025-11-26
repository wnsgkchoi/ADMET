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
    ("Phys", ["basic", "phys"])
]

# Grid Search Space
# Expanded grid for more rigorous evaluation (~432 experiments total)
search_space = {
    "lr": [0.001, 0.0005, 0.0001],
    "dropout_ratio": [0.3, 0.5],
    "num_experts": [3, 5, 7],
    "num_layer": [3, 5, 7],
    "batch_size": [32, 128],
    "emb_dim": [300],
    "decay": [0, 0.00001, 0.0001]
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
            
            # Use run_5seeds.py instead of main.py
            cmd_parts = ["python workspace/src/scripts/run_5seeds.py"]
            
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

    # Write all commands to a single file
    with open(output_file, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")
            
    print(f"Generated {len(commands)} commands in {output_file}")
    print("Example command:")
    print(commands[0])

if __name__ == "__main__":
    generate_commands()
