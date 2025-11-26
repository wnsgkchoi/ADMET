import os
import argparse
import pickle
import numpy as np
import pandas as pd
import sys

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from tdc.single_pred import Tox, ADME
from workspace.benchmark.utils import get_dataset_group

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_splits(dataset_name, base_output_dir, num_seeds=5, label_name=None):
    """
    TDC 데이터셋을 로드하고 5개의 시드에 대해 Scaffold Split을 수행하여 인덱스를 저장합니다.
    """
    print(f"Processing {dataset_name} (Label: {label_name})...")
    
    # 1. 그룹 확인 및 데이터 로드
    group = get_dataset_group(dataset_name)
    
    try:
        if group == 'Tox':
            if label_name:
                data = Tox(name=dataset_name, label_name=label_name)
                save_dataset_name = f"{dataset_name}_{label_name}"
            else:
                data = Tox(name=dataset_name)
                save_dataset_name = dataset_name
        elif group == 'ADME':
            if label_name:
                data = ADME(name=dataset_name, label_name=label_name)
                save_dataset_name = f"{dataset_name}_{label_name}"
            else:
                data = ADME(name=dataset_name)
                save_dataset_name = dataset_name
        else:
            print(f"Error: Dataset {dataset_name} not found in Tox or ADME groups.")
            return
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return

    # 2. 출력 디렉토리 생성 (Group/DatasetName)
    output_dir = os.path.join(base_output_dir, group, save_dataset_name)
    ensure_dir(output_dir)
    print(f"  - Output directory: {output_dir}")

    # 3. 전체 데이터 가져오기
    df = data.get_data()
    print(f"  - Original columns: {df.columns.tolist()}")
    
    # SMILES 컬럼 식별
    if 'Drug' in df.columns:
        smiles_col = 'Drug'
    elif 'Drug_ID' in df.columns:
        smiles_col = 'Drug_ID'
    else:
        raise ValueError("Cannot find SMILES column (Drug or Drug_ID)")
    
    print(f"  - Using '{smiles_col}' as SMILES column")

    # 원본 데이터 저장
    data_path = os.path.join(output_dir, f"{save_dataset_name}_data.csv")
    df.to_csv(data_path, index=False)
    print(f"  - Saved raw data to {data_path} (Shape: {df.shape})")

    # 4. Split 생성 및 저장
    split_indices = {}
    
    # 효율성을 위해 SMILES -> Index 맵 생성
    smiles_list = df[smiles_col].values
    smiles_to_idx = {s: i for i, s in enumerate(smiles_list)}
    
    for seed in range(1, num_seeds + 1):
        # TDC get_split 사용
        split = data.get_split(method='scaffold', seed=seed, frac=[0.7, 0.1, 0.2])
        
        run_indices = {}
        for split_name in ['train', 'valid', 'test']:
            split_df = split[split_name]
            # split_df에도 smiles_col이 있어야 함
            if smiles_col not in split_df.columns:
                 if 'Drug' in split_df.columns:
                     split_smiles_col = 'Drug'
                 else:
                     split_smiles_col = 'Drug_ID'
            else:
                split_smiles_col = smiles_col
            
            indices = []
            for s in split_df[split_smiles_col]:
                if s in smiles_to_idx:
                    indices.append(smiles_to_idx[s])
                else:
                    # print(f"Warning: SMILES not found in original data: {s[:20]}...")
                    pass
            
            run_indices[split_name] = indices
            
        split_indices[f'seed_{seed}'] = run_indices
        print(f"  - Seed {seed}: Train={len(run_indices['train'])}, Valid={len(run_indices['valid'])}, Test={len(run_indices['test'])}")

    # 인덱스 파일 저장
    idx_path = os.path.join(output_dir, f"{save_dataset_name}_splits.pkl")
    with open(idx_path, 'wb') as f:
        pickle.dump(split_indices, f)
    print(f"  - Saved split indices to {idx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AMES', help='Dataset name (e.g., AMES)')
    parser.add_argument('--output_dir', type=str, default='workspace/benchmark/data', help='Base output directory')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    generate_splits(args.dataset, args.output_dir)
