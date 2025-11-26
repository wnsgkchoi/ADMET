"""
ADMET Model Loader Utility

This module provides easy-to-use functions for loading trained ADMET models
for inference. Supports loading individual models or all models at once.

Example usage:
    # Load all models
    loader = ADMETModelLoader()
    all_models = loader.load_all_models()
    
    # Make predictions
    predictions = loader.predict_all("CCO")  # Ethanol
    
    # Load specific model
    model = loader.load_model("AMES")
    prediction = loader.predict_single("CCO", "AMES")
"""

import json
import torch
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings

# Add src root to path
src_root = Path(__file__).parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from TopExpert import GNN_topexpert
from loader import MoleculeCSVDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn


class ADMETModelLoader:
    """
    Unified model loader for ADMET prediction system.
    
    This class handles loading all trained final models and provides
    a simple interface for making predictions on new molecules.
    """
    
    def __init__(self, registry_path: str = 'workspace/final_models/model_registry.json'):
        """
        Initialize the model loader.
        
        Args:
            registry_path: Path to the model registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"ADMETModelLoader initialized")
        print(f"  Device: {self.device}")
        print(f"  Available models: {self.registry['total_models']}")
    
    def _load_registry(self) -> Dict:
        """Load the model registry"""
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Model registry not found: {self.registry_path}\n"
                "Please ensure final models have been trained."
            )
        
        with open(self.registry_path, 'r') as f:
            return json.load(f)
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return sorted(self.registry['models'].keys())
    
    def get_models_by_category(self, category: str) -> List[str]:
        """
        Get models for a specific ADMET category.
        
        Args:
            category: One of 'Absorption', 'Distribution', 'Metabolism', 'Excretion', 'Toxicity'
        
        Returns:
            List of dataset names in the category
        """
        if category not in self.registry['categories']:
            raise ValueError(f"Unknown category: {category}")
        return self.registry['categories'][category]['datasets']
    
    def load_model(self, dataset_name: str, verbose: bool = True) -> nn.Module:
        """
        Load a specific trained model.
        
        Args:
            dataset_name: Name of the dataset/model to load
            verbose: Whether to print loading information
        
        Returns:
            Loaded PyTorch model in eval mode
        """
        if dataset_name in self.models:
            if verbose:
                print(f"Model '{dataset_name}' already loaded (using cached version)")
            return self.models[dataset_name]
        
        if dataset_name not in self.registry['models']:
            raise ValueError(
                f"Model '{dataset_name}' not found in registry.\n"
                f"Available models: {', '.join(self.get_available_models())}"
            )
        
        model_info = self.registry['models'][dataset_name]
        model_path = Path(model_info['model_path'])
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}\n"
                f"Please ensure the model has been trained."
            )
        
        if verbose:
            print(f"Loading model: {dataset_name}")
            print(f"  Category: {model_info['category']}")
            print(f"  Task: {model_info['task_type']}")
            print(f"  Path: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Reconstruct args from checkpoint
        args = checkpoint['args']
        
        # Create model instance
        # Determine criterion based on task type
        if args['task_type'] == 'classification':
            criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            criterion = nn.L1Loss(reduction="none")
        
        # Create args object (simple namespace)
        from argparse import Namespace
        args_obj = Namespace(**args)
        args_obj.device = self.device
        
        # Initialize model
        model = GNN_topexpert(args_obj, criterion)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Cache the model
        self.models[dataset_name] = {
            'model': model,
            'info': model_info,
            'args': args_obj
        }
        
        if verbose:
            print(f"  ✓ Loaded successfully")
            if 'test_metric' in checkpoint:
                metric = model_info['metric']
                score = checkpoint['test_metric']
                print(f"  Test {metric}: {score:.4f}")
        
        return model
    
    def load_all_models(self, verbose: bool = True) -> Dict[str, nn.Module]:
        """
        Load all available models.
        
        Args:
            verbose: Whether to print loading information
        
        Returns:
            Dictionary mapping dataset names to loaded models
        """
        if verbose:
            print(f"\nLoading all {self.registry['total_models']} models...")
            print("="*80)
        
        for dataset_name in self.get_available_models():
            try:
                self.load_model(dataset_name, verbose=verbose)
            except Exception as e:
                warnings.warn(f"Failed to load {dataset_name}: {e}")
        
        if verbose:
            print("="*80)
            print(f"Successfully loaded {len(self.models)}/{self.registry['total_models']} models")
        
        return {name: data['model'] for name, data in self.models.items()}
    
    def predict_single(
        self, 
        smiles: str, 
        dataset_name: str,
        return_secondary_metrics: bool = False
    ) -> Union[float, Dict]:
        """
        Make prediction for a single molecule using a specific model.
        
        Args:
            smiles: SMILES string of the molecule
            dataset_name: Name of the model to use
            return_secondary_metrics: Whether to return additional prediction info
        
        Returns:
            Prediction value (float) or dict with additional info
        """
        # Load model if not already loaded
        if dataset_name not in self.models:
            self.load_model(dataset_name, verbose=False)
        
        model_data = self.models[dataset_name]
        model = model_data['model']
        info = model_data['info']
        args = model_data['args']
        
        # TODO: Convert SMILES to graph (requires implementing mol_to_graph_data_obj_simple)
        # This is a placeholder - you'll need to implement actual SMILES->Graph conversion
        raise NotImplementedError(
            "SMILES to graph conversion not yet implemented. "
            "You'll need to use the same featurization as in the training pipeline."
        )
    
    def predict_all(
        self, 
        smiles: str,
        load_all: bool = True
    ) -> Dict[str, float]:
        """
        Make predictions using all available models.
        
        Args:
            smiles: SMILES string of the molecule
            load_all: Whether to load all models first
        
        Returns:
            Dictionary mapping dataset names to prediction values
        """
        if load_all and len(self.models) < self.registry['total_models']:
            self.load_all_models(verbose=False)
        
        predictions = {}
        for dataset_name in self.models.keys():
            try:
                pred = self.predict_single(smiles, dataset_name)
                predictions[dataset_name] = pred
            except Exception as e:
                warnings.warn(f"Prediction failed for {dataset_name}: {e}")
                predictions[dataset_name] = None
        
        return predictions
    
    def get_model_info(self, dataset_name: str) -> Dict:
        """
        Get metadata about a specific model.
        
        Args:
            dataset_name: Name of the model
        
        Returns:
            Dictionary with model information
        """
        if dataset_name not in self.registry['models']:
            raise ValueError(f"Model '{dataset_name}' not found")
        
        return self.registry['models'][dataset_name]
    
    def print_summary(self):
        """Print summary of available models"""
        print("\n" + "="*80)
        print("ADMET MODEL REGISTRY SUMMARY")
        print("="*80)
        print(f"Total models: {self.registry['total_models']}")
        print(f"Created: {self.registry.get('created', 'Unknown')}")
        print("\nModels by category:")
        
        for category, info in sorted(self.registry['categories'].items()):
            print(f"\n{category} ({info['count']} models):")
            for dataset in info['datasets']:
                model_info = self.registry['models'][dataset]
                task = model_info['task_type']
                metric = model_info['metric']
                score = model_info.get('best_tuning_performance', 'N/A')
                print(f"  - {dataset:35s} [{task:14s}] {metric}={score}")
        
        print("="*80)


def demo():
    """Demonstration of how to use the model loader"""
    print("\n" + "="*80)
    print("ADMET MODEL LOADER DEMO")
    print("="*80)
    
    # Initialize loader
    loader = ADMETModelLoader()
    
    # Print summary
    loader.print_summary()
    
    # Show available models
    print("\n\nAvailable models:")
    for model_name in loader.get_available_models():
        print(f"  - {model_name}")
    
    # Show models by category
    print("\n\nAbsorption models:")
    for model_name in loader.get_models_by_category('Absorption'):
        print(f"  - {model_name}")
    
    # Load a specific model
    print("\n\nLoading AMES model...")
    try:
        model = loader.load_model('AMES')
        print("  ✓ Successfully loaded")
    except FileNotFoundError:
        print("  ⚠ Model not yet trained")
    
    # Example of how to load all models (commented out - would be slow)
    # print("\n\nLoading all models...")
    # all_models = loader.load_all_models()
    
    print("\n" + "="*80)


if __name__ == '__main__':
    demo()
