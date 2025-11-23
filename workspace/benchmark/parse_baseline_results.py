import os
import pickle
import pandas as pd
import glob
import argparse

def parse_baseline_results(results_dir, dataset_name):
    # Find all pickle files matching the pattern
    pattern = os.path.join(results_dir, f"{dataset_name}_baseline*.pkl")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No result files found in {results_dir} for {dataset_name}")
        return

    all_records = []
    
    for pkl_file in files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                
            # data is a list of dicts: [{'seed': 'seed_1', 'results': {'XGBoost': {...}, ...}}, ...]
            for entry in data:
                seed = entry['seed']
                results_dict = entry['results']
                
                for model_name, metrics in results_dict.items():
                    record = {
                        'Dataset': dataset_name,
                        'Seed': seed,
                        'Model': model_name,
                        'SourceFile': os.path.basename(pkl_file)
                    }
                    # Add metrics
                    record.update(metrics)
                    all_records.append(record)
        except Exception as e:
            print(f"Error reading {pkl_file}: {e}")

    if not all_records:
        print("No records found.")
        return

    df = pd.DataFrame(all_records)
    
    # Sort for better readability
    if 'Seed' in df.columns and 'Model' in df.columns:
        df = df.sort_values(by=['Seed', 'Model'])
        
    # Save to CSV
    output_csv = os.path.join(results_dir, f"{dataset_name}_baseline_summary.csv")
    df.to_csv(output_csv, index=False)
    print(f"Summary saved to {output_csv}")
    print("\nTop 5 Models by AUROC:")
    print(df.sort_values(by='AUROC', ascending=False).head(5)[['Seed', 'Model', 'AUROC']])

    # Calculate Mean/Std per model
    print("\nAverage Performance per Model:")
    summary = df.groupby('Model')['AUROC'].agg(['mean', 'std', 'count'])
    print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AMES')
    parser.add_argument('--results_dir', type=str, default='workspace/benchmark/results')
    args = parser.parse_args()
    
    parse_baseline_results(args.results_dir, args.dataset)
