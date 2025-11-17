import json
import random

# Load low-performance datasets
with open('/home/choi0425/workspace/ADMET/workspace/low_performance_datasets.json', 'r') as f:
    low_performance_datasets = json.load(f)

# Hyperparameter search space (original configuration)
lr_range = [0.0001, 0.001]
dropout_range = [0.0, 0.3]
num_experts_range = [3, 5, 7]
alpha_range = [0.01, 0.1, 1.0]
beta_range = [0.01, 0.1, 1.0]
min_temp_range = [0.1, 1.0]
decay_range = [0.0, 0.0001]

print("=" * 80)
print("Generating Retuning Commands for Low-Performance Datasets")
print("=" * 80)
print(f"\nTotal datasets to retune: {len(low_performance_datasets)}")
print(f"Random combinations per dataset: 100")
print(f"Total experiments: {len(low_performance_datasets) * 100}")

# Generate commands
commands = []
experiment_count = 0

for dataset_info in low_performance_datasets:
    dataset = dataset_info['dataset']
    category = dataset_info['category']
    
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset} (Category: {category})")
    print(f"Current {dataset_info['metric_type']}: {dataset_info['performance']:.4f}")
    print(f"{'=' * 80}")
    
    # Generate 100 random combinations for this dataset
    for i in range(100):
        experiment_count += 1
        
        # Random sampling
        lr = random.choice(lr_range)
        dropout = random.choice(dropout_range)
        num_experts = random.choice(num_experts_range)
        alpha = random.choice(alpha_range)
        beta = random.choice(beta_range)
        min_temp = random.choice(min_temp_range)
        decay = random.choice(decay_range)
        
        # Build command (for simple_gpu_scheduler compatibility)
        cmd_parts = [
            "/home/choi0425/miniconda3/envs/ADMET/bin/python",
            "workspace/src/main.py",
            f"--category {category}",
            f"--dataset_name {dataset}",
            f"--experiment_id retune_exp{i:03d}",
            f"--lr {lr}",
            f"--dropout_ratio {dropout}",
            f"--num_experts {num_experts}",
            f"--alpha {alpha}",
            f"--beta {beta}",
            f"--min_temp {min_temp}",
            f"--decay {decay}",
            "--num_layer 5",
            "--emb_dim 300",
            "--gate_dim 50",
            "--split scaffold",
            "--epochs 100",
            "--patience 50",
            "--gin_pretrained_file workspace/src/pre-trained/supervised_contextpred.pth",
        ]
        
        cmd = " ".join(cmd_parts)
        
        commands.append(cmd)
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/100 combinations...")

print(f"\n{'=' * 80}")
print(f"Summary: Generated {experiment_count} total experiments")
print(f"{'=' * 80}")

# Save commands to file
output_file = '/home/choi0425/workspace/ADMET/workspace/commands_retuning_search.txt'
with open(output_file, 'w') as f:
    for cmd in commands:
        f.write(cmd + '\n')

print(f"\n✅ Saved {len(commands)} commands to: {output_file}")

# Statistics
print(f"\n{'=' * 80}")
print("Experiment Distribution:")
print(f"{'=' * 80}")
for dataset_info in low_performance_datasets:
    dataset = dataset_info['dataset']
    count = sum(1 for cmd in commands if f"--dataset {dataset}" in cmd)
    print(f"  {dataset:40s}: {count} experiments")

print(f"\n{'=' * 80}")
print("Next Steps:")
print(f"{'=' * 80}")
print("1. Review the generated commands in:")
print(f"   {output_file}")
print("2. Start GPU scheduler with these commands")
print("3. Monitor error logs in workspace/logs/hyperparameter_errors_summary.log")
print("4. After completion, analyze best hyperparameters")
print(f"{'=' * 80}")
