import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

try:
    from workspace.benchmark.scripts.prepare_data import generate_splits
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from prepare_data import generate_splits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--label_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_seeds', type=int, default=5)
    args = parser.parse_args()

    label = args.label_name if args.label_name != 'None' else None
    
    print(f"Starting process for {args.dataset_name} (Label: {label})")
    generate_splits(args.dataset_name, args.output_dir, args.num_seeds, label_name=label)

if __name__ == "__main__":
    main()
