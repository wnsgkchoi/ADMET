import json
import itertools
import os
import random

# Configuration
CONFIG_PATH = "/home/choi0425/workspace/ADMET/configs/dataset_config.json"
OUTPUT_DIR = "/home/choi0425/workspace/ADMET/workspace/commands"
NUM_PARTS = 4

# Target Datasets
TARGET_DATASETS = [
    "AMES",
    "BBB_Martins",
    "Bioavailability_Ma",
    "Carcinogens_Lagunin",
    "ClinTox",
    "HIA_Hou",
    "Pgp_Broccatelli",
    "hERG",
    "hERG_Central_inhib",
    "hERG_Karim"
]
TARGET_DATASETS.sort() # Ensure alphabetical order

# Hyperparameter Search Space
DROPOUT_RATIOS = [0.3, 0.5]
NUM_EXPERTS = [3, 5, 7]
LRS = [0.001, 0.0001, 0.00001]
DECAYS = [0, 0.00001]
ALPHAS = [1, 0.1, 0.01]
BETAS = [1, 0.1, 0.01]
NUM_LAYERS = [3, 5]
BATCH_SIZES = [32, 128, 512]

def generate_commands():
    parts = [[] for _ in range(NUM_PARTS)]
    
    # Generate all combinations
    combinations = list(itertools.product(
        DROPOUT_RATIOS,
        NUM_EXPERTS,
        LRS,
        DECAYS,
        ALPHAS,
        BETAS,
        NUM_LAYERS,
        BATCH_SIZES
    ))
    
    num_combinations = len(combinations)
    print(f"Generating commands for {len(TARGET_DATASETS)} datasets.")
    print(f"Grid search space size: {num_combinations} per dataset.")
    print(f"Total commands: {num_combinations * len(TARGET_DATASETS)}")
    print(f"Distributing into {NUM_PARTS} parts...")

    for dataset_name in TARGET_DATASETS:
        dataset_commands = []
        for i, combo in enumerate(combinations):
            dropout, num_expert, lr, decay, alpha, beta, num_layer, batch_size = combo
            
            # Generate unique experiment ID
            exp_id = f"exp_{i:04d}"
            
            # Construct command
            cmd = (
                f"python workspace/src/main.py "
                f"--dataset_name {dataset_name} "
                f"--dropout_ratio {dropout} "
                f"--num_experts {num_expert} "
                f"--lr {lr} "
                f"--decay {decay} "
                f"--alpha {alpha} "
                f"--beta {beta} "
                f"--num_layer {num_layer} "
                f"--batch_size {batch_size} "
                f"--extra_feature_dim 37 "
                f"--experiment_id {exp_id} "
                f"--gin_pretrained_file workspace/src/pre-trained/supervised_contextpred.pth "
                f"--epochs 500 "
                f"--no_save_model "
                f"--quiet"
            )
            dataset_commands.append(cmd)
        
        # Shuffle commands for this dataset to ensure random distribution of hyperparameters within parts
        random.shuffle(dataset_commands)
        
        # Distribute this dataset's commands across parts (Round Robin)
        for i, cmd in enumerate(dataset_commands):
            parts[i % NUM_PARTS].append(cmd)

    # Sort each part alphabetically by dataset name (optional, but keeps it organized)
    for i in range(NUM_PARTS):
        parts[i].sort()
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Write to files
    for i in range(NUM_PARTS):
        filename = f"commands_part{i+1}.txt"
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, 'w') as f:
            for cmd in parts[i]:
                f.write(cmd + "\n")
        print(f"Generated {output_path} with {len(parts[i])} commands.")

if __name__ == "__main__":
    generate_commands()
