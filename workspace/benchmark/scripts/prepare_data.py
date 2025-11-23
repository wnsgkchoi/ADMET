import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list
from tdc.benchmark_group import admet_group

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_splits(dataset_name, output_dir, num_seeds=5):
    """
    TDC 데이터셋을 로드하고 5개의 시드에 대해 Scaffold Split을 수행하여 인덱스를 저장합니다.
    """
    print(f"Processing {dataset_name}...")
    
    # 1. 데이터 로드 (TDC Tox)
    # AMES는 Tox 카테고리에 속함. 필요시 다른 카테고리(ADME) 확장 가능
    try:
        data = Tox(name=dataset_name)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return

        # 전체 데이터 가져오기 (인덱스 매핑을 위해 필요)
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

    # 원본 데이터 저장 (나중에 로드할 때 정합성 확인용)
    data_path = os.path.join(output_dir, f"{dataset_name}_data.csv")
    df.to_csv(data_path, index=False)
    print(f"  - Saved raw data to {data_path} (Shape: {df.shape})")

    # 2. Split 생성 및 저장
    # from tdc.utils import split_data_by_scaffold # Removed incorrect import
    
    split_indices = {}
    
    # 효율성을 위해 SMILES -> Index 맵 생성
    # df[smiles_col]이 Series가 아니라 DataFrame일 수 있음 (컬럼 이름 중복 시)
    # 따라서 values를 사용하여 확실하게 1차원 배열로 가져옴
    smiles_list = df[smiles_col].values
    if len(smiles_list.shape) > 1:
         # 만약 컬럼이 중복되어 2차원이면 첫 번째 컬럼(보통 ID가 아니라 SMILES일 가능성 높음, 하지만 확인 필요)
         # TDC에서 Drug_ID와 Drug가 둘 다 있으면 Drug가 SMILES임.
         # 위에서 Drug가 있으면 Drug를 썼으므로, 중복된 Drug가 있을 수 있음.
         # 하지만 get_data() 원본에는 중복이 없을 것임. 아까 중복은 rename 때문이었음.
         pass

    smiles_to_idx = {s: i for i, s in enumerate(smiles_list)}
    
    for seed in range(1, num_seeds + 1):
        # TDC get_split 사용
        split = data.get_split(method='scaffold', seed=seed, frac=[0.7, 0.1, 0.2])
        
        run_indices = {}
        for split_name in ['train', 'valid', 'test']:
            split_df = split[split_name]
            # split_df에도 smiles_col이 있어야 함
            if smiles_col not in split_df.columns:
                 # split_df는 표준화된 컬럼을 가질 수 있음. 보통 Drug_ID, Drug, Y
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
                    print(f"Warning: SMILES not found in original data: {s[:20]}...")
            
            run_indices[split_name] = indices
            
        split_indices[f'seed_{seed}'] = run_indices
        print(f"  - Seed {seed}: Train={len(run_indices['train'])}, Valid={len(run_indices['valid'])}, Test={len(run_indices['test'])}")


    # 2. Split 생성 및 저장
    # from tdc.utils import split_data_by_scaffold # Removed incorrect import

    
    # TDC의 split_data는 내부적으로 random state를 사용하지 않고 고정된 scaffold split을 제공할 수 있음.
    # 하지만 Benchmark Group 가이드라인에 따라 5번의 run을 위해 seed를 바꿔가며 split을 생성해야 함.
    # TDC의 scaffold split은 결정적(deterministic)일 수 있으므로, 
    # 여기서는 TDC Benchmark Group의 표준 방식을 따르거나, 
    # 만약 scaffold split이 seed에 영향을 받지 않는다면 
    # Random Split을 5번 하거나, Scaffold Split 1회 + Random Initialization 5회 전략을 세워야 함.
    
    # *중요*: TDC Leaderboard는 보통 "Scaffold Split" 1개를 고정으로 사용하고, 
    # 모델의 초기화 Seed를 5번 바꿔서 평균을 냄.
    # 하지만 사용자의 요청은 "서로 다른 5개의 split"임.
    # Scaffold Split은 분자 구조에 기반하므로 Seed에 따라 달라지지 않는 것이 일반적임 (Deterministic).
    # 그러나 Random Scaffold Split(Scaffold를 섞어서 나누는 방식)이라면 Seed 영향이 있음.
    
    # 여기서는 TDC의 `get_split(method='scaffold', seed=seed)`를 사용하여 
    # Seed에 따라 달라지는지 확인하고 저장함.
    
    split_indices = {}
    
    for seed in range(1, num_seeds + 1):
        # TDC get_split 사용
        # method='scaffold'는 seed에 따라 train/val/test 구성이 달라질 수 있음 (scaffold set을 어떻게 분배하느냐에 따라)
        split = data.get_split(method='scaffold', seed=seed, frac=[0.7, 0.1, 0.2])
        
        # Debugging: Check columns and content
        if seed == 1:
            print(f"DEBUG: Split keys: {split.keys()}")
            print(f"DEBUG: Train columns: {split['train'].columns}")
            print(f"DEBUG: Train head: {split['train'].head()}")
            print(f"DEBUG: Raw DF head: {df.head()}")

        # 인덱스 추출 (Drug 컬럼 기준 매핑이 가장 안전하지만, 여기서는 데이터프레임의 인덱스를 사용하지 않고
        # SMILES 문자열을 Key로 사용하여 매핑하는 것이 안전함. 
        # 하지만 TDC get_split은 데이터를 쪼개서 리턴하므로, 원본 df에서의 인덱스를 찾아야 함.)
        
        # 효율성을 위해 SMILES -> Index 맵 생성
        smiles_to_idx = {smiles: idx for idx, smiles in enumerate(df['Drug'])}
        
        run_indices = {}
        for split_name in ['train', 'valid', 'test']:
            split_df = split[split_name]
            indices = [smiles_to_idx[s] for s in split_df['Drug'] if s in smiles_to_idx]
            run_indices[split_name] = indices
            
        split_indices[f'seed_{seed}'] = run_indices
        print(f"  - Seed {seed}: Train={len(run_indices['train'])}, Valid={len(run_indices['valid'])}, Test={len(run_indices['test'])}")

    # 인덱스 파일 저장
    idx_path = os.path.join(output_dir, f"{dataset_name}_splits.pkl")
    with open(idx_path, 'wb') as f:
        pickle.dump(split_indices, f)
    print(f"  - Saved split indices to {idx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AMES', help='Dataset name (e.g., AMES)')
    parser.add_argument('--output_dir', type=str, default='workspace/benchmark/data', help='Output directory')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    generate_splits(args.dataset, args.output_dir)
