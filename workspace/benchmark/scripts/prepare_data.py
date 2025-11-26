import os
import argparse
import pickle
import json
import numpy as np
import pandas as pd
<<<<<<< HEAD
import random
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tdc.single_pred import Tox, ADME
from tdc.benchmark_group import admet_group
from tdc.utils import retrieve_label_name_list

# Initialize ADMET Group once to check for membership
try:
    ADMET_GROUP = admet_group(path='workspace/benchmark/tdc_cache')
    ADMET_DATASETS = set(name.lower() for name in ADMET_GROUP.dataset_names)
except Exception as e:
    print(f"Warning: Could not initialize ADMET Benchmark Group: {e}")
    ADMET_GROUP = None
    ADMET_DATASETS = set()
=======
import sys

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from tdc.single_pred import Tox, ADME
from workspace.benchmark.utils import get_dataset_group
>>>>>>> 06c0ab56d1c4430ff3e23ab6e23fb5f18b88c717

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

<<<<<<< HEAD
def get_tdc_class(category):
    """카테고리에 따른 TDC 클래스 반환"""
    if category == 'Toxicity':
        return Tox
    elif category in ['Absorption', 'Distribution', 'Metabolism', 'Excretion']:
        return ADME
    else:
        raise ValueError(f"Unknown category: {category}")

def generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return smiles
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

def generate_splits(dataset_name, category, output_dir, num_seeds=5):
=======
def generate_splits(dataset_name, base_output_dir, num_seeds=5, label_name=None):
>>>>>>> 06c0ab56d1c4430ff3e23ab6e23fb5f18b88c717
    """
    TDC 데이터셋을 로드하고 5개의 시드에 대해 Scaffold Split을 수행하여 인덱스를 저장합니다.
    단, Test Set은 Seed 1의 결과를 고정하고, Train/Valid만 시드에 따라 변경합니다.
    """
<<<<<<< HEAD
    print(f"Processing {dataset_name} ({category})...")
    
    # 1. 데이터 로드
    try:
        Loader = get_tdc_class(category)
        
        # hERG_Central 예외 처리
        if dataset_name.startswith('hERG_Central'):
            if dataset_name == 'hERG_Central_inhib':
                data = Loader(name='hERG_Central', label_name='hERG_inhib')
            elif dataset_name == 'hERG_Central_10uM':
                data = Loader(name='hERG_Central', label_name='hERG_at_10uM')
            elif dataset_name == 'hERG_Central_1uM':
                data = Loader(name='hERG_Central', label_name='hERG_at_1uM')
            else:
                # Fallback
                data = Loader(name=dataset_name)
        else:
            data = Loader(name=dataset_name)
            
=======
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
>>>>>>> 06c0ab56d1c4430ff3e23ab6e23fb5f18b88c717
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return

<<<<<<< HEAD
    # 전체 데이터 가져오기 (인덱스 매핑을 위해 필요)
=======
    # 2. 출력 디렉토리 생성 (Group/DatasetName)
    output_dir = os.path.join(base_output_dir, group, save_dataset_name)
    ensure_dir(output_dir)
    print(f"  - Output directory: {output_dir}")

    # 3. 전체 데이터 가져오기
>>>>>>> 06c0ab56d1c4430ff3e23ab6e23fb5f18b88c717
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

<<<<<<< HEAD
    # 2. Split 생성 및 저장
=======
    # 4. Split 생성 및 저장
>>>>>>> 06c0ab56d1c4430ff3e23ab6e23fb5f18b88c717
    split_indices = {}
    
    # 효율성을 위해 SMILES -> Index 맵 생성
    smiles_list = df[smiles_col].values
<<<<<<< HEAD
    smiles_to_idx = defaultdict(list)
    for i, s in enumerate(smiles_list):
        smiles_to_idx[s].append(i)
        
    # Helper to get indices from a DataFrame
    def get_indices_from_df(target_df):
        indices = []
        if smiles_col not in target_df.columns:
             if 'Drug' in target_df.columns:
                 target_smiles_col = 'Drug'
             else:
                 target_smiles_col = 'Drug_ID'
        else:
            target_smiles_col = smiles_col
            
        for s in target_df[target_smiles_col]:
            if s in smiles_to_idx:
                # If multiple indices (duplicates), add all of them? 
                # Usually benchmark splits are unique. 
                # But if raw data has duplicates, we should be careful.
                # We take the first one that hasn't been used? 
                # For simplicity, take all matching indices.
                indices.extend(smiles_to_idx[s])
            else:
                # Try canonicalizing?
                try:
                    canon_s = Chem.MolToSmiles(Chem.MolFromSmiles(s))
                    found = False
                    for k in smiles_to_idx:
                        if Chem.MolToSmiles(Chem.MolFromSmiles(k)) == canon_s:
                            indices.extend(smiles_to_idx[k])
                            found = True
                            break
                    if not found:
                        print(f"Warning: SMILES not found in original data: {s[:20]}...")
                except:
                     print(f"Warning: SMILES not found and cannot canonicalize: {s[:20]}...")
        return sorted(list(set(indices)))

    # Determine Test Set
    test_indices = []
    is_benchmark = False
    
    if ADMET_GROUP and dataset_name.lower() in ADMET_DATASETS:
        print(f"  - {dataset_name} is in TDC ADMET Benchmark Group. Fetching official split...")
        try:
            benchmark = ADMET_GROUP.get(dataset_name)
            test_df = benchmark['test']
            test_indices = get_indices_from_df(test_df)
            is_benchmark = True
            print(f"  - Fetched {len(test_indices)} test indices from Benchmark.")
        except Exception as e:
            print(f"  - Failed to fetch benchmark split ({e}). Falling back to Seed 1 Scaffold Split.")
            
    if not is_benchmark:
        print("  - Generating Master Split (Seed 1) for Test Set...")
        master_split = data.get_split(method='scaffold', seed=1, frac=[0.7, 0.1, 0.2])
        test_indices = get_indices_from_df(master_split['test'])

    # Identify Train+Valid Indices (All - Test)
    all_indices = set(range(len(df)))
    test_set_indices = set(test_indices)
    dev_indices = list(all_indices - test_set_indices)
    
    print(f"  - Total: {len(all_indices)}, Test: {len(test_indices)}, Dev (Train+Valid): {len(dev_indices)}")
    
    # Pre-calculate scaffolds for dev set to save time
    print("  - Calculating scaffolds for Dev Set...")
    dev_scaffolds = defaultdict(list)
    for idx in tqdm(dev_indices, desc="Scaffold Gen"):
        s = smiles_list[idx]
        scaff = generate_scaffold(s, include_chirality=True)
        dev_scaffolds[scaff].append(idx)
    
    # Sort scaffolds by size (descending) to be consistent with typical scaffold split
    # But here we want to shuffle them for random scaffold split
    # We convert to list of (scaffold, indices)
    all_scaffold_sets = [indices for scaff, indices in dev_scaffolds.items()]
    # Sort by smallest index in the group to be deterministic
    all_scaffold_sets.sort(key=lambda x: min(x))

    for seed in range(1, num_seeds + 1):
        run_indices = {}
        run_indices['test'] = test_indices # Fixed Test Set
        
        # Re-split Dev Set
        # Shuffle scaffolds
        current_scaffold_sets = all_scaffold_sets[:]
        
        if seed == 1:
             # Deterministic Scaffold Split (Sort by size)
             # Already sorted by size in all_scaffold_sets
             pass
        else:
             # Random Scaffold Split
             random.seed(seed)
             random.shuffle(current_scaffold_sets)
        
        # Fill Train (7/8 of Dev)
        # Dev is 80% of total. Train is 70% of total.
        # So Train target is len(dev_indices) * (7/8)
        train_cutoff = len(dev_indices) * (7.0 / 8.0)
        
        train_idx = []
        valid_idx = []
        
        for scaffold_set in current_scaffold_sets:
            if len(train_idx) + len(scaffold_set) <= train_cutoff:
                train_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        
        run_indices['train'] = train_idx
        run_indices['valid'] = valid_idx
            
        split_indices[f'seed_{seed}'] = run_indices
        split_type = "Standard Scaffold" if seed == 1 else "Random Scaffold (Fixed Test)"
        print(f"  - Seed {seed} [{split_type}]: Train={len(run_indices['train'])}, Valid={len(run_indices['valid'])}, Test={len(run_indices['test'])}")

    # Pickle 저장
    pkl_path = os.path.join(output_dir, f"{dataset_name}_splits.pkl")
    with open(pkl_path, 'wb') as f:
=======
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
>>>>>>> 06c0ab56d1c4430ff3e23ab6e23fb5f18b88c717
        pickle.dump(split_indices, f)
    print(f"  - Saved splits to {pkl_path}")

def main():
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument('--config_path', type=str, default='configs/dataset_config.json')
    parser.add_argument('--output_dir', type=str, default='workspace/benchmark/data')
=======
    parser.add_argument('--dataset', type=str, default='AMES', help='Dataset name (e.g., AMES)')
    parser.add_argument('--output_dir', type=str, default='workspace/benchmark/data', help='Base output directory')
>>>>>>> 06c0ab56d1c4430ff3e23ab6e23fb5f18b88c717
    args = parser.parse_args()
    
    ensure_dir(args.output_dir)
    
    with open(args.config_path, 'r') as f:
        config = json.load(f)
        
    datasets = config['datasets']
    
    for name, info in datasets.items():
        # Check if files already exist
        data_path = os.path.join(args.output_dir, f"{name}_data.csv")
        split_path = os.path.join(args.output_dir, f"{name}_splits.pkl")
        
        if os.path.exists(data_path) and os.path.exists(split_path):
             print(f"Skipping {name} (Already exists)")
             continue
             
        category = info['category']
        generate_splits(name, category, args.output_dir)

if __name__ == "__main__":
    main()
