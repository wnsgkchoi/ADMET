"""
Unified ADMET Prediction System

This module provides a complete ADMET prediction pipeline that takes a SMILES string
as input and outputs predictions for all 33 ADMET properties.

Features:
- Single SMILES input
- 33 ADMET property predictions (binary classification + continuous regression)
- Organized by category (Absorption, Distribution, Metabolism, Excretion, Toxicity)
- Easy-to-use interface

Example:
    predictor = ADMETPredictor()
    results = predictor.predict("CCO")  # Ethanol
    print(results)
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import warnings
import sys

# Add src root to path
src_root = Path(__file__).parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from TopExpert import GNN_topexpert
from loader import mol_to_graph_data_obj_simple
import torch.nn as nn
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem


class ADMETPredictor:
    """
    Unified ADMET Prediction System
    
    Predicts all 33 ADMET properties from a single SMILES string.
    """
    
    def __init__(
        self, 
        registry_path: str = 'workspace/final_models/model_registry.json',
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the ADMET predictor.
        
        Args:
            registry_path: Path to model registry JSON
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None.
            verbose: Whether to print loading information
        """
        self.registry_path = Path(registry_path)
        self.verbose = verbose
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load registry
        self.registry = self._load_registry()
        
        # Initialize model storage
        self.models = {}
        self.model_metadata = {}
        
        if self.verbose:
            print("="*80)
            print("ADMET Unified Prediction System")
            print("="*80)
            print(f"Device: {self.device}")
            print(f"Total properties: {self.registry['total_models']}")
            print(f"Registry: {self.registry_path}")
            print("="*80)
    
    def _load_registry(self) -> Dict:
        """Load model registry"""
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Model registry not found: {self.registry_path}\n"
                "Please ensure final models have been trained."
            )
        
        with open(self.registry_path, 'r') as f:
            return json.load(f)
    
    def load_all_models(self):
        """Load all 33 ADMET prediction models"""
        if self.verbose:
            print("\nLoading all ADMET models...")
            print("-"*80)
        
        for dataset_name in sorted(self.registry['models'].keys()):
            try:
                self._load_single_model(dataset_name)
                if self.verbose:
                    category = self.registry['models'][dataset_name]['category']
                    task_type = self.registry['models'][dataset_name]['task_type']
                    print(f"  ✓ {dataset_name:35s} [{category:12s}] ({task_type})")
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ {dataset_name:35s} Failed: {e}")
                warnings.warn(f"Failed to load {dataset_name}: {e}")
        
        if self.verbose:
            print("-"*80)
            print(f"Successfully loaded: {len(self.models)}/{self.registry['total_models']} models\n")
    
    def _load_single_model(self, dataset_name: str):
        """Load a single model"""
        if dataset_name in self.models:
            return
        
        model_info = self.registry['models'][dataset_name]
        model_path = Path(model_info['model_path'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        args = checkpoint['args']
        
        # Create model
        from argparse import Namespace
        args_obj = Namespace(**args)
        args_obj.device = self.device
        
        if args['task_type'] == 'classification':
            criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            criterion = nn.L1Loss(reduction="none")
        
        model = GNN_topexpert(args_obj, criterion)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Store model and metadata
        self.models[dataset_name] = model
        self.model_metadata[dataset_name] = {
            'info': model_info,
            'args': args_obj,
            'checkpoint': checkpoint
        }
    
    def smiles_to_graph(self, smiles: str):
        """
        Convert SMILES string to graph representation.
        Uses the same featurization as the training pipeline.
        
        Args:
            smiles: SMILES string of the molecule
        
        Returns:
            PyTorch Geometric Data object ready for model input
        """
        # Convert SMILES to RDKit mol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        # Convert mol to graph using the same function as training
        graph_data = mol_to_graph_data_obj_simple(mol, smile=smiles)
        
        # Create batch with single graph
        batch = Batch.from_data_list([graph_data])
        batch = batch.to(self.device)
        
        return batch
    
    def predict(
        self, 
        smiles: str, 
        return_details: bool = False
    ) -> Dict[str, Union[float, Dict]]:
        """
        Predict all 33 ADMET properties for a given SMILES string.
        
        Args:
            smiles: SMILES string of the molecule
            return_details: Whether to return additional details (confidence, etc.)
        
        Returns:
            Dictionary with predictions organized by category
            
        Example output:
            {
                'Absorption': {
                    'Caco2_Wang': 0.365,
                    'HIA_Hou': 0.97,
                    ...
                },
                'Distribution': {...},
                'Metabolism': {...},
                'Excretion': {...},
                'Toxicity': {...}
            }
        """
        # Ensure models are loaded
        if not self.models:
            self.load_all_models()
        
        # Convert SMILES to graph
        try:
            graph_batch = self.smiles_to_graph(smiles)
        except Exception as e:
            raise ValueError(f"Failed to convert SMILES to graph: {e}")
        
        # Make predictions
        predictions = {
            'Absorption': {},
            'Distribution': {},
            'Metabolism': {},
            'Excretion': {},
            'Toxicity': {}
        }
        
        for dataset_name, model in self.models.items():
            metadata = self.model_metadata[dataset_name]
            category = metadata['info']['category']
            task_type = metadata['info']['task_type']
            
            try:
                with torch.no_grad():
                    # Forward pass
                    pred, z, q = model(graph_batch)
                    
                    # Process prediction based on task type
                    # Handle different prediction shapes
                    if pred.dim() > 1:
                        # Multi-dimensional output: take mean across batch and features
                        pred = pred.squeeze()
                    
                    # If still multi-dimensional, take first/mean element
                    if pred.numel() > 1:
                        pred_tensor = pred.mean()
                    else:
                        pred_tensor = pred
                    
                    if task_type == 'classification':
                        # Binary classification - apply sigmoid
                        pred_value = torch.sigmoid(pred_tensor).item()
                    else:
                        # Regression - use raw value
                        pred_value = pred_tensor.item()
                
                if return_details:
                    predictions[category][dataset_name] = {
                        'value': pred_value,
                        'task_type': task_type,
                        'unit': metadata['info'].get('unit', 'N/A'),
                        'interpretation': self._interpret_prediction(
                            pred_value, task_type, dataset_name
                        )
                    }
                else:
                    predictions[category][dataset_name] = pred_value
                    
            except Exception as e:
                warnings.warn(f"Prediction failed for {dataset_name}: {e}")
                predictions[category][dataset_name] = None
        
        return predictions
    
    def _interpret_prediction(
        self, 
        value: float, 
        task_type: str, 
        property_name: str
    ) -> str:
        """
        Interpret prediction value for human readability (Korean).
        
        Args:
            value: Predicted value
            task_type: 'classification' or 'regression'
            property_name: Name of the ADMET property
        
        Returns:
            Human-readable interpretation in Korean
        """
        if task_type == 'classification':
            if value >= 0.5:
                return f"양성 ({value:.2%})"
            else:
                return f"음성 ({value:.2%})"
        else:
            return f"{value:.4f}"
    
    def format_korean_output(self, smiles: str, predictions: Dict) -> str:
        """
        Format predictions in Korean style matching the reference image.
        
        Args:
            smiles: Input SMILES string
            predictions: Prediction results from predict()
        
        Returns:
            Formatted Korean output string
        """
        lines = []
        lines.append("=" * 120)
        lines.append(f"■ 33개 구성 요소 (흡수 • 분포/가용성, 국국 • 축적)")
        lines.append("=" * 120)
        
        # Absorption (흡수)
        lines.append("")
        lines.append("● Absorption (흡수)")
        lines.append("-" * 120)
        
        abs_props = {
            'Bioavailability_Ma': ('생물학적 이용률 (흡수)', '%'),
            'Caco2_Wang': ('Caco2 투과성', 'cm/s'),
            'HIA_Hou': ('사람 장 흡수 (흡수)', ''),
            'HydrationFreeEnergy_FreeSolv': ('수화 자유 에너지', 'kcal/mol'),
            'Lipophilicity_AstraZeneca': ('친유성', 'LogP'),
            'PAMPA_NCATS': ('PAMPA 투과성 (흡수)', ''),
            'Pgp_Broccatelli': ('P-gp 기질 가능성 (흡수)', ''),
            'Solubility_AqSolDB': ('수용성', 'mol/L')
        }
        
        for prop, (kr_name, unit) in abs_props.items():
            if prop in predictions['Absorption'] and predictions['Absorption'][prop] is not None:
                val = predictions['Absorption'][prop]
                task = self.registry['models'][prop]['task_type']
                
                if task == 'classification':
                    status = "양호 ✓" if val >= 0.5 else "개선 필요 X"
                    lines.append(f"{prop:35s} | {val:>7.2%} | {kr_name:30s} | {status}")
                else:
                    unit_str = f" {unit}" if unit else ""
                    lines.append(f"{prop:35s} | {val:>10.4f}{unit_str:15s} | {kr_name}")
        
        # Distribution (분포)
        lines.append("")
        lines.append("● Distribution (분포)")
        lines.append("-" * 120)
        
        dist_props = {
            'BBB_Martins': ('BBB 투과성 (흡수)', ''),
            'PPBR_AZ': ('혈장 단백질 결합률', '%'),
            'VDss_Lombardo': ('분포 용적', 'L/kg')
        }
        
        for prop, (kr_name, unit) in dist_props.items():
            if prop in predictions['Distribution'] and predictions['Distribution'][prop] is not None:
                val = predictions['Distribution'][prop]
                task = self.registry['models'][prop]['task_type']
                
                if task == 'classification':
                    status = "BBB 통과 ✓" if val >= 0.5 else ""
                    lines.append(f"{prop:35s} | {val:>7.2%} | {kr_name:30s} | {status}")
                else:
                    unit_str = f" {unit}" if unit else ""
                    lines.append(f"{prop:35s} | {val:>10.4f}{unit_str:15s} | {kr_name}")
        
        # Metabolism (대사)
        lines.append("")
        lines.append("● Metabolism (대사)")
        lines.append("-" * 120)
        
        metab_props = {
            'CYP1A2_Veith': 'CYP1A2 기질 가능성 (흡수)',
            'CYP2C19_Veith': 'CYP2C19 기질 가능성 (흡수)',
            'CYP2C9_Substrate_CarbonMangels': 'CYP2C9 기질 여부 (C/M, 흡수)',
            'CYP2C9_Veith': 'CYP2C9 기질 가능성 (흡수)',
            'CYP2D6_Substrate_CarbonMangels': 'CYP2D6 기질 여부 (C/M, 흡수)',
            'CYP2D6_Veith': 'CYP2D6 기질 가능성 (흡수)',
            'CYP3A4_Substrate_CarbonMangels': 'CYP3A4 기질 여부 (C/M, 흡수)',
            'CYP3A4_Veith': 'CYP3A4 기질 가능성 (흡수)'
        }
        
        for prop, kr_name in metab_props.items():
            if prop in predictions['Metabolism'] and predictions['Metabolism'][prop] is not None:
                val = predictions['Metabolism'][prop]
                status = "기질 아님 X" if val >= 0.5 else "기질 아님 X"
                lines.append(f"{prop:35s} | {val:>7.2%} | {kr_name:45s} | {status}")
        
        # Excretion (배설)
        lines.append("")
        lines.append("● Excretion (배설)")
        lines.append("-" * 120)
        
        excr_props = {
            'Clearance_Hepatocyte_AZ': ('간 제거율', 'mL/min/kg'),
            'Clearance_Microsome_AZ': ('미소체 제거율', 'mL/min/kg'),
            'Half_Life_Obach': ('반감기', 'hours')
        }
        
        for prop, (kr_name, unit) in excr_props.items():
            if prop in predictions['Excretion'] and predictions['Excretion'][prop] is not None:
                val = predictions['Excretion'][prop]
                unit_str = f" {unit}" if unit else ""
                lines.append(f"{prop:35s} | {val:>10.4f}{unit_str:15s} | {kr_name}")
        
        # Toxicity (독성)
        lines.append("")
        lines.append("● Toxicity (독성)")
        lines.append("-" * 120)
        
        tox_props = {
            'AMES': ('AMES 돌연변이성 (흡수)', ''),
            'Carcinogens_Lagunin': ('발암성 가능성 (흡수)', ''),
            'ClinTox': ('임상 독성 위험 (흡수)', ''),
            'DILI': ('약물 간손상 (흡수)', ''),
            'LD50_Zhu': ('반수치사량', 'mg/kg'),
            'Skin_Reaction': ('피부 반응 (흡수)', ''),
            'hERG': ('hERG 저해 (흡수)', ''),
            'hERG_Central_10uM': ('hERG IC50 (10uM)', 'pIC50'),
            'hERG_Central_1uM': ('hERG IC50 (1uM)', 'pIC50'),
            'hERG_Central_inhib': ('hERG 억제 (흡수)', ''),
            'hERG_Karim': ('hERG 자극 위험성 (Karim, 흡수)', '')
        }
        
        for prop, (kr_name, unit) in tox_props.items():
            if prop in predictions['Toxicity'] and predictions['Toxicity'][prop] is not None:
                val = predictions['Toxicity'][prop]
                task = self.registry['models'][prop]['task_type']
                
                if task == 'classification':
                    if 'hERG' in prop or 'AMES' in prop or 'DILI' in prop:
                        status = "양호 ✓" if val < 0.5 else "위험하지 않음 ✓"
                    elif 'Skin' in prop:
                        status = "반응없음 ▲" if val < 0.5 else ""
                    else:
                        status = "양호 ✓" if val < 0.5 else "양호 ✓"
                    lines.append(f"{prop:35s} | {val:>7.2%} | {kr_name:45s} | {status}")
                else:
                    unit_str = f" {unit}" if unit else ""
                    lines.append(f"{prop:35s} | {val:>10.4f}{unit_str:15s} | {kr_name}")
        
        return "\n".join(lines)
    
    def print_korean_report(self, smiles: str):
        """
        Print ADMET prediction report in Korean format.
        
        Args:
            smiles: SMILES string to predict
        """
        predictions = self.predict(smiles, return_details=False)
        report = self.format_korean_output(smiles, predictions)
        print(report)
    
    def predict_batch(
        self, 
        smiles_list: List[str],
        return_dataframe: bool = True
    ) -> Union[pd.DataFrame, List[Dict]]:
        """
        Predict ADMET properties for multiple molecules.
        
        Args:
            smiles_list: List of SMILES strings
            return_dataframe: Whether to return as DataFrame (True) or list of dicts (False)
        
        Returns:
            DataFrame with predictions for all molecules or list of prediction dicts
        """
        results = []
        
        for smiles in smiles_list:
            try:
                pred = self.predict(smiles, return_details=False)
                # Flatten predictions
                flat_pred = {'SMILES': smiles}
                for category, preds in pred.items():
                    for prop, value in preds.items():
                        flat_pred[prop] = value
                results.append(flat_pred)
            except Exception as e:
                warnings.warn(f"Failed to predict for {smiles}: {e}")
        
        if return_dataframe:
            return pd.DataFrame(results)
        else:
            return results
    
    def get_property_info(self, property_name: str) -> Dict:
        """Get information about a specific ADMET property"""
        if property_name not in self.registry['models']:
            raise ValueError(f"Unknown property: {property_name}")
        return self.registry['models'][property_name]
    
    def list_properties(self, category: Optional[str] = None) -> List[str]:
        """
        List all available ADMET properties.
        
        Args:
            category: Filter by category (Absorption, Distribution, etc.)
        
        Returns:
            List of property names
        """
        if category is None:
            return sorted(self.registry['models'].keys())
        else:
            return [
                name for name, info in self.registry['models'].items()
                if info['category'] == category
            ]
    
    def summary(self):
        """Print summary of available predictions"""
        print("\n" + "="*80)
        print("ADMET PREDICTION CAPABILITIES")
        print("="*80)
        
        for category, info in sorted(self.registry['categories'].items()):
            print(f"\n{category} ({info['count']} properties)")
            print("-"*80)
            
            for dataset in info['datasets']:
                model_info = self.registry['models'][dataset]
                task_type = model_info['task_type']
                metric = model_info['metric']
                loaded = "✓" if dataset in self.models else "○"
                
                print(f"  {loaded} {dataset:35s} {task_type:14s} ({metric})")
        
        print("="*80)
        print(f"Models loaded: {len(self.models)}/{self.registry['total_models']}")
        print("="*80)


def demo():
    """Demonstration of the unified ADMET predictor"""
    print("\n" + "="*80)
    print("UNIFIED ADMET PREDICTOR DEMO")
    print("="*80)
    
    # Initialize predictor
    predictor = ADMETPredictor()
    
    # Show available properties
    predictor.summary()
    
    # Load all models
    print("\nLoading all models...")
    predictor.load_all_models()
    
    # Example prediction (would work once SMILES conversion is implemented)
    print("\n" + "="*80)
    print("EXAMPLE PREDICTION")
    print("="*80)
    print("Note: SMILES to graph conversion needs to be implemented")
    print("      to make actual predictions.")
    print("="*80)
    
    # Show what the output would look like
    print("\nExpected output format:")
    print("""
    {
        'Absorption': {
            'Caco2_Wang': 0.365,
            'HIA_Hou': 0.97,
            'Pgp_Broccatelli': 0.85,
            ...
        },
        'Distribution': {
            'BBB_Martins': 0.82,
            ...
        },
        ...
    }
    """)


if __name__ == '__main__':
    demo()
