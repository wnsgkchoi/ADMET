import os
import sys
import argparse
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from workspace.benchmark.scripts.target_datasets import ALL_TARGET_DATASETS
from workspace.benchmark.utils import get_dataset_path

# Import functions
try:
    from workspace.benchmark.scripts.generate_features import generate_features
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
    from generate_features import generate_features

try:
    from workspace.benchmark.baseline.run_baseline import run_baseline
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from run_baseline import run_baseline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='workspace/benchmark/data')
    parser.add_argument('--output_dir', type=str, default='workspace/benchmark/results')
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=None)
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    print(f"Found {len(ALL_TARGET_DATASETS)} target datasets.")

    for key, info in ALL_TARGET_DATASETS.items():
        dataset_name_tdc = info.get('dataset_name', key)
        label_name = info.get('label_name', None)
        
        # Construct folder name used by prepare_data.py
        if label_name:
            folder_name = f"{dataset_name_tdc}_{label_name}"
        else:
            folder_name = dataset_name_tdc
            
        print(f"\n{'#'*60}")
        print(f"Processing {key} (Folder: {folder_name})")
        print(f"{'#'*60}")
        
        # 1. Generate Features
        try:
            dataset_path = get_dataset_path(args.data_dir, folder_name)
            feat_path = os.path.join(dataset_path, f"{folder_name}_features.pkl")
            
            if os.path.exists(feat_path):
                print("Features already exist. Skipping generation.")
            else:
                print("Generating features...")
                generate_features(folder_name, args.data_dir)
                
        except Exception as e:
            print(f"Error generating features for {key}: {e}")
            traceback.print_exc()
            continue

        # 2. Run Baseline
        try:
            # Check if baseline already done (Summary exists)
            progress_csv = os.path.join(args.output_dir, f"{folder_name}_progress.csv")
            if os.path.exists(progress_csv):
                with open(progress_csv, 'r') as f:
                    content = f.read()
                    if "Final Summary" in content:
                        print(f"Baseline already completed for {key}. Skipping.")
                        continue
            
            print("Running baseline...")
            run_baseline(folder_name, args.data_dir, args.output_dir, args.n_trials, args.num_seeds)
            
        except Exception as e:
            print(f"Error running baseline for {key}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
