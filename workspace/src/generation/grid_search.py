import itertools

def main():
    # Define which datasets should use cross-validation
    cv_datasets = ['hk2']

    # Define the hyperparameter search space
    param_grid = {
        '--lr': [1e-4, 1e-3, 1e-2],
        '--dropout_ratio': [0, 0.3, 0.5],
        '--emb_dim': [300],
        '--dataset': ['dili2', 'dili3', 'hepg2', 'hk2'],
        '--decay': [0, 0.0001, 0.00001],
        '--batch_size': [32, 512],
        '--num_experts': [3, 5, 7, 10],
        '--alpha': [5, 1, 0.1, 0.01],
        '--beta': [5, 1, 0.1, 0.01],
        '--min_temp': [0.01, 0.1, 1]
    }

    # Fixed parameters
    base_command = "conda run -n ADMET python /home/choi0425/workspace/ADMET/workspace/src/main.py"
    fixed_params = {
        '--dataset_dir': '/home/choi0425/workspace/ADMET/workspace/data/train',
        '--epochs': 100
    }

    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Generate the commands
    for i, params in enumerate(experiments):
        experiment_id = i + 1  # 1부터 시작하는 순차 ID
        command = base_command
        # Add fixed parameters
        for key, value in fixed_params.items():
            command += f" {key} {value}"
        
        # Add experiment_id
        command += f" --experiment_id {experiment_id}"
        
        # Add variable parameters
        current_dataset = None
        for key, value in params.items():
            if key == '--dataset':
                current_dataset = value
            command += f" {key} {value}"
        
        # Conditionally add the split method
        if current_dataset in cv_datasets:
            command += " --split cv"
        else:
            command += " --split scaffold" # Default split for larger datasets

        # Add an identifier for the output log
        log_file_parts = [
            f"ds-{params.get('--dataset', 'N/A')}",
            f"lr-{params.get('--lr', 'N/A')}",
            f"dr-{params.get('--dropout_ratio', 'N/A')}",
            f"bs-{params.get('--batch_size', 'N/A')}",
            f"ne-{params.get('--num_experts', 'N/A')}",
        ]
        log_file = f"/home/choi0425/workspace/ADMET/workspace/output/{'_'.join(log_file_parts)}.log"
        command += f" > {log_file}"
        
        print(command)

if __name__ == "__main__":
    main()
