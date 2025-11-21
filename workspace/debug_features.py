
import sys
import os
import torch
import numpy as np
from rdkit import Chem

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loader import mol_to_graph_data_obj_simple
from TopExpert import GNN_topexpert

class MockArgs:
    def __init__(self):
        self.num_layer = 5
        self.emb_dim = 300
        self.num_tasks = 1
        self.num_classes = 2
        self.dropout_ratio = 0.5
        self.graph_pooling = "mean"
        self.gnn_type = "gin"
        self.JK = "last"
        self.num_experts = 3
        self.gate_dim = 64
        self.num_tr_scf = 10
        self.extra_feature_dim = 1228

def test_feature_integration():
    print("Testing feature integration...")
    
    # 1. Test Loader
    print("1. Testing Loader...")
    mol = Chem.MolFromSmiles("CCO")
    data = mol_to_graph_data_obj_simple(mol)
    
    if not hasattr(data, 'x_features'):
        print("❌ Error: x_features not found in Data object")
        return
    
    if data.x_features.shape != (1, 1228):
        print(f"❌ Error: x_features shape mismatch. Expected (1, 1228), got {data.x_features.shape}")
        return
        
    print(f"✅ Loader successful. Feature shape: {data.x_features.shape}")
    
    # 2. Test Model Forward Pass
    print("2. Testing Model Forward Pass...")
    args = MockArgs()
    criterion = torch.nn.BCEWithLogitsLoss()
    model = GNN_topexpert(args, criterion)
    
    # Create batch
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([data, data]) # Batch of 2
    
    try:
        clf_logit, z, q = model(batch)
        print(f"✅ Forward pass successful.")
        print(f"   Output shape: {clf_logit.shape}")
        print(f"   Gate shape: {z.shape}")
        print(f"   Assignment shape: {q.shape}")
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_integration()
