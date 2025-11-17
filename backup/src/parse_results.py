import os
import re
import pandas as pd

def parse_log_file(filepath):
    """Parses a single log file to extract the final performance metric."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Search for the final average validation metric from the end of the file
        for line in reversed(lines):
            # For classification
            match = re.search(r"Average Validation Accuracy: (\d+\.\d+)", line)
            if match:
                return float(match.group(1))
            
            # For regression (assuming MAE, lower is better)
            match = re.search(r"Average Validation MAE: (\d+\.\d+)", line)
            if match:
                return float(match.group(1))
        return None
    except Exception:
        return None

def parse_filename(filename):
    """Parses a filename to extract hyperparameters."""
    try:
        parts = filename.replace('.log', '').split('_')
        params = {}
        for part in parts:
            key_val = part.split('-')
            if len(key_val) == 2: 
                key, val = key_val
                # Map keys from filename to arg names
                key_map = {
                    'ds': 'dataset',
                    'lr': 'lr',
                    'dr': 'dropout_ratio',
                    'bs': 'batch_size',
                    'ne': 'num_experts'
                }
                if key in key_map:
                    params[key_map[key]] = val
        return params
    except Exception:
        return {}

def main():
    log_dir = '/home/choi0425/workspace/ADMET/workspace/output'
    results = []

    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]

    for i, filename in enumerate(log_files):
        print(f"Parsing file {i+1}/{len(log_files)}: {filename}")
        filepath = os.path.join(log_dir, filename)
        
        params = parse_filename(filename)
        metric = parse_log_file(filepath)
        
        if metric is not None:
            params['test_metric'] = metric
            results.append(params)

    if not results:
        print("No valid log files found or failed to parse results.")
        return

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(results)
    
    # Determine if higher is better (classification) or lower is better (regression)
    # This is a simple heuristic; assumes classification if 'dili2' is present.
    is_classification = any(df['dataset'].str.contains('dili', na=False))
    
    if is_classification:
        df = df.sort_values(by='test_metric', ascending=False)
    else:
        df = df.sort_values(by='test_metric', ascending=True) # Lower is better for MAE

    output_csv_path = os.path.join(log_dir, 'grid_search_summary.csv')
    df.to_csv(output_csv_path, index=False)

    print(f"\\nSuccessfully parsed {len(df)} log files.")
    print(f"Summary saved to: {output_csv_path}")
    print("\\nTop 5 performing parameter sets:")
    print(df.head(5).to_string())

if __name__ == "__main__":
    main()
