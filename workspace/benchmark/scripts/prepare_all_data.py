import os
import sys
import argparse
import subprocess

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import target datasets
try:
    from workspace.benchmark.scripts.target_datasets import ALL_TARGET_DATASETS
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from target_datasets import ALL_TARGET_DATASETS

# Import utils
try:
    from workspace.benchmark.utils import get_dataset_group
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils import get_dataset_group

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='workspace/benchmark/data', help='Base output directory')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of seeds for splitting')
    args = parser.parse_args()

    failed_datasets = []

    print(f"Total target datasets to process: {len(ALL_TARGET_DATASETS)}")

    for key, info in ALL_TARGET_DATASETS.items():
        dataset_name = info.get('dataset_name', key)
        label_name = info.get('label_name', None)
        
        print(f"\n--- Processing: {key} (TDC Name: {dataset_name}, Label: {label_name}) ---")
        
        # Determine group
        group = get_dataset_group(dataset_name)
        if not group:
            # Fallback if get_dataset_group fails (e.g. for herg_central if not in lists)
            # But herg_central should be in Tox.
            # If not found, prepare_data prints error.
            pass
        
        if group:
            if label_name:
                save_name = f"{dataset_name}_{label_name}"
            else:
                save_name = dataset_name
            
            expected_dir = os.path.join(args.output_dir, group, save_name)
            if os.path.exists(expected_dir):
                # Check if splits file exists
                splits_file = os.path.join(expected_dir, f"{save_name}_splits.pkl")
                if os.path.exists(splits_file):
                    print(f"Skipping {key} (already processed at {expected_dir})")
                    continue
        
        try:
            # Use subprocess to isolate crashes (SegFaults)
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), 'process_single_dataset.py'),
                '--dataset_name', dataset_name,
                '--output_dir', args.output_dir,
                '--num_seeds', str(args.num_seeds)
            ]
            if label_name:
                cmd.extend(['--label_name', label_name])
            else:
                cmd.extend(['--label_name', 'None'])
            
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"Error processing {key} (subprocess failed): {e}")
            failed_datasets.append(key)
        except Exception as e:
            print(f"Error processing {key}: {e}")
            failed_datasets.append(key)

    if failed_datasets:
        print("\n=== Failed Datasets ===")
        for name in failed_datasets:
            print(f"  - {name}")
    else:
        print("\nAll target datasets processed successfully.")

if __name__ == "__main__":
    main()
