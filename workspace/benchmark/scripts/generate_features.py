import os
import argparse
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, MACCSkeys, QED

# ===================================================================
# 1. 디스크립터 정의 
# ===================================================================

# 물리화학적 디스크립터 (37개)
PHYSICOCHEMICAL_DESCRIPTORS = [
    # 1. 물리화학적 기본 특성 (7개)
    'MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 
    'NOCount', 'NumRotatableBonds',
    
    # 2. 구조적 특성 (5개)
    'RingCount', 'NumAromaticRings', 'NumSaturatedRings', 
    'NumAliphaticRings', 'FractionCSP3',
    
    # 3. 원자 개수 및 전자 구성 (4개)
    'HeavyAtomCount', 'NumValenceElectrons', 'NumHeteroatoms', 'NHOHCount',
    
    # 4. 위상 및 복잡성 지표 (3개)
    'BalabanJ', 'BertzCT', 'HallKierAlpha',
    
    # 5. Chi 연결성 지수 (4개)
    'Chi0', 'Chi1', 'Chi0n', 'Chi1n',
    
    # 6. E-State 지수 (2개)
    'MaxEStateIndex', 'MinEStateIndex',
    
    # 7. VSA 지표 (2개)
    'PEOE_VSA1', 'SlogP_VSA1',
    
    # 8. 약물 유사도 및 표면적 (3개)
    'MolMR', 'QED', 'LabuteASA',
    
    # 9. 고리 세부 특성 (4개)
    'NumAliphaticCarbocycles', 'NumAromaticCarbocycles',
    'NumAliphaticHeterocycles', 'NumAromaticHeterocycles',
    
    # 10. Kappa 형상 지수 (3개)
    'Kappa1', 'Kappa2', 'Kappa3'
]

# MACCS Keys는 167 bits (0번부터 166번까지)
MACCS_BITS = 167

# ===================================================================
# 2. 디스크립터 계산 함수
# ===================================================================

def calculate_physicochemical_descriptors(mol):
    """물리화학적 디스크립터 계산 (37개)"""
    try:
        descriptors = [
            # 1. 물리화학적 기본 특성 (7개)
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NOCount(mol),
            Descriptors.NumRotatableBonds(mol),
            
            # 2. 구조적 특성 (5개)
            Descriptors.RingCount(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.FractionCSP3(mol),
            
            # 3. 원자 개수 및 전자 구성 (4개)
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumValenceElectrons(mol),
            Descriptors.NumHeteroatoms(mol),
            Lipinski.NHOHCount(mol),
            
            # 4. 위상 및 복잡성 지표 (3개)
            Descriptors.BalabanJ(mol),
            Descriptors.BertzCT(mol),
            Descriptors.HallKierAlpha(mol),
            
            # 5. Chi 연결성 지수 (4개)
            Descriptors.Chi0(mol),
            Descriptors.Chi1(mol),
            Descriptors.Chi0n(mol),
            Descriptors.Chi1n(mol),
            
            # 6. E-State 지수 (2개)
            Descriptors.MaxEStateIndex(mol),
            Descriptors.MinEStateIndex(mol),
            
            # 7. VSA 지표 (2개)
            Descriptors.PEOE_VSA1(mol),
            Descriptors.SlogP_VSA1(mol),
            
            # 8. 약물 유사도 및 표면적 (3개)
            Descriptors.MolMR(mol),
            QED.qed(mol),
            Descriptors.LabuteASA(mol),
            
            # 9. 고리 세부 특성 (4개)
            Descriptors.NumAliphaticCarbocycles(mol),
            Descriptors.NumAromaticCarbocycles(mol),
            Descriptors.NumAliphaticHeterocycles(mol),
            Descriptors.NumAromaticHeterocycles(mol),
            
            # 10. Kappa 형상 지수 (3개)
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol)
        ]
        return descriptors
    except Exception as e:
        # print(f"물리화학적 디스크립터 계산 오류: {e}")
        return None

def calculate_maccs_fingerprint(mol):
    try:
        maccs = MACCSkeys.GenMACCSKeys(mol)
        # BitVector를 리스트로 변환 (167 bits: 0-166)
        maccs_list = list(maccs)
        return maccs_list
    except Exception as e:
        # print(f"MACCS Keys 계산 오류: {e}")
        return None

def calculate_ecfp_fingerprint(mol, radius=2, nBits=1024):
    try:
        ecfp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            radius=radius, 
            nBits=nBits,
            useFeatures=False
        )
        # BitVector를 리스트로 변환
        return list(ecfp)
    except Exception as e:
        # print(f"ECFP 계산 오류: {e}")
        return None

def calculate_all_features(smiles, ecfp_radius=2, ecfp_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    phys_desc = calculate_physicochemical_descriptors(mol)
    maccs_fp = calculate_maccs_fingerprint(mol)
    ecfp_fp = calculate_ecfp_fingerprint(mol, radius=ecfp_radius, nBits=ecfp_bits)
    
    if phys_desc is None or maccs_fp is None or ecfp_fp is None:
        return None
    
    return {
        'physicochemical': phys_desc,
        'maccs': maccs_fp,
        'ecfp': ecfp_fp
    }

def generate_features(dataset_name, data_dir):
    data_path = os.path.join(data_dir, f"{dataset_name}_data.csv")
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {dataset_name} data: {df.shape}")
    
    # SMILES 컬럼 찾기
    if 'Drug' in df.columns:
        smiles_col = 'Drug'
    elif 'Drug_ID' in df.columns:
        smiles_col = 'Drug_ID'
    else:
        raise ValueError("Cannot find SMILES column")

    features_list = []
    valid_indices = []
    
    print("Calculating features...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row[smiles_col]
        feats = calculate_all_features(smiles)
        
        if feats is not None:
            # Flatten features
            flat_feats = feats['physicochemical'] + feats['maccs'] + feats['ecfp']
            features_list.append(flat_feats)
            valid_indices.append(idx)
        else:
            features_list.append(None) # Keep None to maintain index alignment or handle later
            
    # Create DataFrame for features
    # Column names
    phys_cols = PHYSICOCHEMICAL_DESCRIPTORS
    maccs_cols = [f'MACCS_{i}' for i in range(MACCS_BITS)]
    ecfp_cols = [f'ECFP_{i}' for i in range(1024)]
    all_cols = phys_cols + maccs_cols + ecfp_cols
    
    # Filter out None
    valid_features = [f for f in features_list if f is not None]
    
    # We need to align with the original dataframe. 
    # If some molecules failed, we should probably exclude them from the splits too?
    # But the splits are already generated based on the original dataframe indices.
    # If a molecule fails feature calculation, we can't use it for baseline.
    # We should mark which indices are valid.
    
    # Let's save the features as a numpy array aligned with the dataframe.
    # Use NaN for failed molecules.
    
    num_features = len(all_cols)
    feature_matrix = np.full((len(df), num_features), np.nan)
    
    for i, feats in enumerate(features_list):
        if feats is not None:
            feature_matrix[i] = feats
            
    output_path = os.path.join(data_dir, f"{dataset_name}_features.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'features': feature_matrix,
            'columns': all_cols,
            'smiles': df[smiles_col].values
        }, f)
        
    print(f"Saved features to {output_path} (Shape: {feature_matrix.shape})")
    print(f"Failed molecules: {len(df) - len(valid_features)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AMES', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='workspace/benchmark/data', help='Data directory')
    args = parser.parse_args()
    
    generate_features(args.dataset, args.data_dir)
