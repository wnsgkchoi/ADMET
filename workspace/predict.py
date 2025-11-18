#!/usr/bin/env python
"""
ADMET 통합 예측 시스템

SMILES 문자열을 입력하면 33개 ADMET 속성을 한번에 예측합니다.
사용법:
  1. SMILES_INPUT 변수 수정 후 실행: python predict.py
  2. 대화형 모드: python predict.py --interactive
  3. 예제 실행: python predict.py --demo
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from deployment.unified_predictor import ADMETPredictor

# ============================================================================
# 여기에 SMILES를 입력하세요!
# ============================================================================
SMILES_INPUT = "CC(=O)Oc1ccccc1C(=O)O"  # 기본값: Aspirin
# ============================================================================


def predict_single_molecule():
    """Example: Predict ADMET properties for a single molecule"""
    
    # Initialize predictor
    predictor = ADMETPredictor(verbose=True)
    predictor.load_all_models()
    
    # Example molecule: Aspirin
    smiles = 'CC(=O)Oc1ccccc1C(=O)O'
    
    print(f"\nPredicting ADMET properties for:")
    print(f"SMILES: {smiles} (Aspirin)")
    print("="*80)
    
    # Get predictions
    results = predictor.predict(smiles, return_details=True)
    
    # Display key predictions
    print("\n📊 KEY ADMET PREDICTIONS:")
    print("-"*80)
    
    # Absorption
    print("\n🔵 Absorption:")
    print(f"  HIA (Human Intestinal Absorption): {results['Absorption']['HIA_Hou']['interpretation']}")
    print(f"  Caco-2 Permeability: {results['Absorption']['Caco2_Wang']['value']:.2f}")
    print(f"  Solubility: {results['Absorption']['Solubility_AqSolDB']['value']:.2f} log mol/L")
    
    # Distribution
    print("\n🟢 Distribution:")
    print(f"  BBB Penetration: {results['Distribution']['BBB_Martins']['interpretation']}")
    print(f"  Plasma Protein Binding: {results['Distribution']['PPBR_AZ']['value']:.1f}%")
    
    # Metabolism
    print("\n🟡 Metabolism:")
    print(f"  CYP3A4 Substrate: {results['Metabolism']['CYP3A4_Substrate_CarbonMangels']['interpretation']}")
    print(f"  CYP2D6 Inhibitor: {results['Metabolism']['CYP2D6_Veith']['interpretation']}")
    
    # Excretion
    print("\n🟣 Excretion:")
    print(f"  Half-Life: {results['Excretion']['Half_Life_Obach']['value']:.2f} hours")
    print(f"  Clearance (Hepatocyte): {results['Excretion']['Clearance_Hepatocyte_AZ']['value']:.2f}")
    
    # Toxicity
    print("\n🔴 Toxicity:")
    print(f"  AMES Mutagenicity: {results['Toxicity']['AMES']['interpretation']}")
    print(f"  hERG Inhibition: {results['Toxicity']['hERG']['interpretation']}")
    print(f"  DILI: {results['Toxicity']['DILI']['interpretation']}")
    print(f"  LD50: {results['Toxicity']['LD50_Zhu']['value']:.2f} log mol/kg")
    
    print("="*80)


def predict_multiple_molecules():
    """Example: Batch prediction for multiple molecules"""
    
    predictor = ADMETPredictor(verbose=False)
    predictor.load_all_models()
    
    # Example drug molecules
    molecules = {
        'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
        'Ibuprofen': 'CC(C)Cc1ccc(cc1)C(C)C(O)=O',
        'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'Paracetamol': 'CC(=O)Nc1ccc(O)cc1',
        'Warfarin': 'CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O'
    }
    
    print("\n📋 BATCH ADMET PREDICTION")
    print("="*80)
    
    smiles_list = list(molecules.values())
    df = predictor.predict_batch(smiles_list, return_dataframe=True)
    
    # Add molecule names
    df.insert(0, 'Drug', list(molecules.keys()))
    
    # Show selected properties
    selected_props = [
        'Drug', 'SMILES', 'HIA_Hou', 'BBB_Martins', 
        'AMES', 'hERG', 'LD50_Zhu', 'Lipophilicity_AstraZeneca'
    ]
    
    print("\nSelected ADMET Properties:")
    print(df[selected_props].to_string(index=False))
    
    # Save to CSV
    output_file = 'admet_predictions.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Full results saved to: {output_file}")
    print("="*80)


def compare_molecules():
    """Example: Compare ADMET profiles of two molecules"""
    
    predictor = ADMETPredictor(verbose=False)
    predictor.load_all_models()
    
    # Compare two similar drugs
    mol1_name = "Aspirin"
    mol1_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    
    mol2_name = "Ibuprofen"
    mol2_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
    
    print(f"\n🔬 COMPARISON: {mol1_name} vs {mol2_name}")
    print("="*80)
    
    pred1 = predictor.predict(mol1_smiles)
    pred2 = predictor.predict(mol2_smiles)
    
    # Key properties to compare
    comparisons = [
        ('HIA_Hou', 'Absorption', 'Human Intestinal Absorption'),
        ('BBB_Martins', 'Distribution', 'BBB Penetration'),
        ('AMES', 'Toxicity', 'Mutagenicity (AMES)'),
        ('hERG', 'Toxicity', 'hERG Inhibition'),
        ('Lipophilicity_AstraZeneca', 'Absorption', 'Lipophilicity (LogP)')
    ]
    
    print(f"\n{'Property':<40} {mol1_name:<15} {mol2_name:<15}")
    print("-"*80)
    
    for prop, category, name in comparisons:
        val1 = pred1[category][prop]
        val2 = pred2[category][prop]
        
        if isinstance(val1, float) and isinstance(val2, float):
            if 0 <= val1 <= 1 and 0 <= val2 <= 1:
                # Classification probability
                print(f"{name:<40} {val1:>14.1%} {val2:>14.1%}")
            else:
                # Regression value
                print(f"{name:<40} {val1:>14.2f} {val2:>14.2f}")
    
    print("="*80)


def interactive_mode():
    """대화형 SMILES 입력 모드"""
    predictor = ADMETPredictor(verbose=False)
    predictor.load_all_models()
    
    print("\n" + "="*80)
    print("ADMET 예측 시스템 - 대화형 모드")
    print("="*80)
    print("\nSMILES를 입력하세요 (종료: 'q' 또는 빈 입력)")
    print("\n예시:")
    print("  CC(=O)Oc1ccccc1C(=O)O  (Aspirin)")
    print("  CCO                    (Ethanol)")
    print("  CN1C=NC2=C1C(=O)N(C(=O)N2C)C  (Caffeine)")
    print("="*80)
    
    while True:
        smiles = input("\nSMILES >>> ").strip()
        
        if not smiles or smiles.lower() == 'q':
            print("\n종료합니다.")
            break
        
        try:
            print()
            predictor.print_korean_report(smiles)
        except Exception as e:
            print(f"\n오류: {e}")
            print("올바른 SMILES 형식인지 확인해주세요.")


def quick_predict():
    """상단의 SMILES_INPUT 변수를 사용한 빠른 예측"""
    predictor = ADMETPredictor(verbose=False)
    predictor.load_all_models()
    
    print("\n" + "="*80)
    print(f"SMILES: {SMILES_INPUT}")
    print("="*80)
    predictor.print_korean_report(SMILES_INPUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ADMET 통합 예측 시스템')
    parser.add_argument('--interactive', '-i', action='store_true', 
                        help='대화형 모드 실행')
    parser.add_argument('--demo', '-d', action='store_true',
                        help='예제 데모 실행')
    parser.add_argument('--smiles', '-s', type=str,
                        help='예측할 SMILES 문자열')
    
    args = parser.parse_args()
    
    if args.interactive:
        # 대화형 모드
        interactive_mode()
    
    elif args.demo:
        # 데모 모드
        print("\n" + "="*80)
        print("ADMET 예측 시스템 - 데모")
        print("="*80)
        predict_single_molecule()
        print("\n\n")
        predict_multiple_molecules()
        print("\n\n")
        compare_molecules()
    
    elif args.smiles:
        # 명령줄 SMILES 입력
        predictor = ADMETPredictor(verbose=False)
        predictor.load_all_models()
        print(f"\nSMILES: {args.smiles}\n")
        predictor.print_korean_report(args.smiles)
    
    else:
        # 기본: 파일 상단의 SMILES_INPUT 사용
        quick_predict()
    
    print("\n" + "="*80)
    print("사용법:")
    print("  python predict.py              # 파일 상단 SMILES_INPUT 사용")
    print("  python predict.py -s 'CCO'     # 명령줄에서 SMILES 입력")
    print("  python predict.py -i           # 대화형 모드")
    print("  python predict.py --demo       # 전체 예제 실행")
    print("="*80 + "\n")
