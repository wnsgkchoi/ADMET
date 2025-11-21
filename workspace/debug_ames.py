
import torch
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("workspace/src"))

from loader import MoleculeCSVDataset
from torch_geometric.loader import DataLoader

def check_ames():
    print("Checking AMES dataset...")
    df = pd.read_csv("workspace/data/Toxicity/AMES/train.csv")
    print(f"Total samples: {len(df)}")
    print("Label distribution:")
    print(df['Y'].value_counts())
    
    num_pos = df[df['Y'] == 1].shape[0]
    num_neg = df[df['Y'] == 0].shape[0]
    
    print(f"Pos: {num_pos}, Neg: {num_neg}")
    
    pos_weight = num_neg / num_pos
    print(f"Calculated pos_weight: {pos_weight}")
    
    # Check if pos_weight is reasonable
    if pos_weight > 10 or pos_weight < 0.1:
        print("WARNING: Extreme pos_weight!")
    
    # Check loader
    print("\nChecking Loader and Features...")
    
    # Test mol_to_graph_data_obj_simple
    from loader import mol_to_graph_data_obj_simple
    from rdkit import Chem
    
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O" # Aspirin
    mol = Chem.MolFromSmiles(smiles)
    data = mol_to_graph_data_obj_simple(mol, smiles)
    
    print(f"Generated graph for {smiles}")
    print(f"x shape: {data.x.shape}")
    print(f"x first 7 cols (cat): {data.x[0, :7]}")
    print(f"x rest (cont): {data.x[0, 7:]}")
    
    x_cont = data.x[:, 7:]
    if torch.all(x_cont == 0):
        print("WARNING: All continuous features are ZERO!")
    else:
        print(f"Continuous features mean: {x_cont.mean()}, std: {x_cont.std()}")
        print(f"First row features: {x_cont[0]}")

if __name__ == "__main__":
    check_ames()
