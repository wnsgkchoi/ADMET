# ADMET Prediction  

33개 ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) 속성을 예측하는 Graph Neural Network 기반 모델 구축 및 베이스라인과 비교  

## 📋 프로젝트 개요

- **모델**: TopExpert-based model (GIN backbone + MoE)
- **데이터셋**: 33개 ADMET 속성 (분류 21개, 회귀 12개)
- **성능**: TODO.

## 🔬 데이터셋 정보

33개 ADMET 데이터셋 (총 ~600,000 화합물):

**Absorption (8개)**
- Caco2_Wang, HIA_Hou, PAMPA_NCATS, Pgp_Broccatelli, Bioavailability_Ma, Lipophilicity_AstraZeneca, Solubility_AqSolDB, HydrationFreeEnergy_FreeSolv

**Distribution (3개)**
- BBB_Martins, PPBR_AZ, VDss_Lombardo

**Metabolism (8개)**
- CYP1A2_Veith, CYP2C19_Veith, CYP2C9_Veith/Substrate, CYP2D6_Veith/Substrate, CYP3A4_Veith/Substrate

**Excretion (3개)**
- Clearance_Hepatocyte_AZ, Clearance_Microsome_AZ, Half_Life_Obach

**Toxicity (11개)**
- AMES, hERG, hERG_Central (3종), hERG_Karim, ClinTox, DILI, LD50_Zhu, Carcinogens_Lagunin, Skin_Reaction

## 🛠️ 환경 설정  

```bash
# Conda 환경 생성
conda env create --file env.yaml
conda activate ADMET

# 사전학습 모델 다운로드 (필요시)
# supervised_contextpred.pth를 workspace/src/pre-trained/에 저장
```

## 📝 주요 파라미터

### 모델 아키텍처
- `--num_layer`: GNN 레이어 수 (기본: 5)
- `--emb_dim`: 임베딩 차원 (기본: 300, 사전학습 모델에 맞춤)
- `--num_experts`: Expert 수 (기본: 7)
- `--gate_dim`: Gate 임베딩 차원 (기본: 64)

### 학습 설정
- `--lr`: Learning rate (기본: 1e-4)
- `--batch_size`: Batch size (기본: 512)
- `--dropout_ratio`: Dropout (기본: 0.5)
- `--epochs`: Max epochs (기본: 200)
- `--patience`: Early stopping patience (기본: 50)

### 손실 함수
- `--loss_type`: 'bce' 또는 'focal' (기본: 'bce')
- `--focal_alpha`: Focal loss alpha (기본: 0.25)
- `--focal_gamma`: Focal loss gamma (기본: 2.0)

### TopExpert 파라미터
- `--alpha`: Clustering loss weight (기본: 0.1)
- `--beta`: Alignment loss weight (기본: 0.01)
- `--min_temp`: Gumbel-Softmax 최소 온도 (기본: 1.0)

## 📊 Project Update Note  

### Ver 0.0.1 (Oct 21, 2025)  

#### 🎯 추가된 기능  

##### 1. Focal Loss (불균형 데이터셋 대응)

```python
# TopExpert.py에 구현됨
from TopExpert import FocalLoss

# 사용 예시
criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
```

##### 2. 하이퍼파라미터 튜닝 시 모델 저장 제어

```bash
# 튜닝 모드: 모델 저장 안 함 (결과만 CSV에 저장)
python workspace/src/main.py --experiment_id exp_001 ...

# 최종 학습: 모델 저장
python workspace/src/main.py ...  # experiment_id 없음
```

### Ver 0.0.2 (Nov 24, 2025)  

본 업데이트는 TDC(Therapeutics Data Commons) 가이드라인을 준수하여 구축된 ADMET 예측 모델의 성능을 Baseline과 엄밀하게 비교하고, TopExpert 모델의 최적 구성을 찾기 위해 진행되었습니다.  

#### 🎯 추가된 기능

##### 1. Data Acquisition & Standardization
- **Source**: TDC (Therapeutics Data Commons) ADMET Benchmark Group.
- **Preprocessing**:
  - 33개 ADMET 데이터셋 확보 (Absorption, Distribution, Metabolism, Excretion, Toxicity).
  - RDKit을 이용한 SMILES Canonicalization 및 Salt Removal.
  - 데이터 디렉토리 구조화: `workspace/data/{Category}/{Dataset}/`.

##### 2. Baseline & Benchmark Split Establishment
- **Objective**: 공정한 성능 비교를 위한 고정된 평가 환경 구축.
- **Splitting Strategy**:
  - **Scaffold Split**: 화학적 구조의 다양성을 고려하여 일반화 성능을 평가.
  - **5-Seed Validation**: 우연에 의한 성능 편차를 배제하기 위해 5개의 서로 다른 Random Seed(0~4) 사용.
  - **Artifact**: `AMES_splits.pkl` (Train/Valid/Test 인덱스를 고정하여 저장).
- **Baseline**: XGBoost/Random Forest (ECFP4 기반) 등의 머신러닝 모델 성능 측정 완료.

##### 3. TopExpert Model Implementation
- **Architecture**: GIN (Graph Isomorphism Network) Backbone + Mixture of Experts (MoE).
- **Pre-training**: `supervised_contextpred.pth`를 이용한 Transfer Learning 적용.
- **Loss Function**: Imbalanced Dataset 처리를 위한 Class Weighting 및 Focal Loss 적용 검토.

##### 4. Advanced Feature Engineering System
GNN의 표현력을 보강하기 위해 다양한 분자 특성을 동적으로 결합할 수 있는 시스템을 `src/loader.py`에 구축했습니다.
- **Basic**: Graph Features (Atom type, Bond type, Chirality, Hybridization, etc.)
- **Phys**: 37 RDKit Physicochemical Descriptors (MolWt, LogP, TPSA, etc.)
- **MACCS**: 167-bit MACCS Keys (Structural Keys).
- **ECFP**: 1024-bit Morgan Fingerprints (Radius 2).
- **Combination**: 위 4가지 특성의 8가지 조합(예: Basic+Phys, Basic+ECFP 등)을 실험 가능하도록 구현.

##### 5. Rigorous Evaluation Pipeline (Current)
Baseline과의 **완전한 1:1 비교**를 위해 파이프라인을 고도화했습니다.
- **Split Consistency**: 
  - `src/main.py` 수정: 자체적인 Random/Scaffold Split을 수행하는 대신, **Baseline이 사용한 `AMES_splits.pkl`을 강제로 로드**하여 사용.
  - 이를 통해 GNN과 Baseline이 **단 하나의 샘플도 다르지 않은 동일한 데이터셋**으로 평가됨을 보장.
- **Grid Search**:
  - `generate_ablation_commands.py`: Hyperparameter(LR, Dropout, Layers, Experts) 및 Feature 조합에 대한 576개 실험 자동 생성.
- **Execution**: `simple_gpu_scheduler`를 이용한 4-GPU 병렬 실험 수행.

#### 🚀 Roadmap & Remaining Pipeline Tasks

단순한 모델 성능 개선을 넘어, 실용적이고 신뢰할 수 있는 ADMET 예측 시스템 완성을 위해 다음 과제들이 남아있습니다.

##### 1. Model Architecture Refinement
- [ ] **Late Fusion Implementation**: `TopExpert.py` 리팩토링. Global Feature(MACCS, ECFP)를 GNN Readout 이후 단계에서 결합하여 Graph Feature의 정보 손실 방지 및 고차원 벡터 처리 효율화.
- [ ] **Uncertainty Estimation**: 예측 결과의 신뢰도를 함께 제공하기 위한 Monte Carlo Dropout 또는 Deep Ensemble 기법 도입.

##### 2. Optimization & Expansion
- [ ] **Full-Scale Benchmark**: AMES(Toxicity)에서 검증된 파이프라인을 나머지 32개 ADMET 데이터셋으로 확장 적용.
- [ ] **Ensemble Strategy**: 5-seed 모델의 예측값을 결합(Soft Voting/Averaging)하여 단일 모델 대비 성능 및 일반화 능력 극대화.

##### 3. Analysis & Deployment
- [ ] **Error Analysis Tool**: 모델이 실패한 케이스(False Positive/Negative)를 자동으로 추출하고, 해당 분자의 화학적 특성(Scaffold, Property distribution)을 분석하는 도구 개발.
- [ ] **Interpretability**: GNN의 Attention Weight나 Gradient를 시각화하여 독성/물성을 유발하는 핵심 부분구조(Substructure)를 규명하는 기능 추가.

---
