# ADMET Prediction Models

33개 ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) 속성을 예측하는 Graph Neural Network 기반 모델

## 📋 프로젝트 개요

- **모델**: GNN_topexpert (TopExpert + GIN backbone)
- **데이터셋**: 33개 ADMET 속성 (분류 21개, 회귀 12개)
- **성능**: 평균 AUROC 80.35% (분류), 평균 R² 0.29 (회귀)

## 🚀 빠른 시작

### 통합 예측 사용

```bash
# 단일 SMILES 예측 (33개 속성 동시 예측)
python workspace/predict.py -s "CCO"

# 대화형 모드
python workspace/predict.py -i

# 데모 실행
python workspace/predict.py --demo
```

## 📁 디렉토리 구조

```
ADMET/
├── workspace/
│   ├── predict.py                    # 통합 예측 진입점
│   ├── generate_random_search.py     # Random search 명령어 생성
│   ├── run_random_search.sh          # Random search 실행 스크립트
│   ├── data/                         # 원본 데이터 (33 datasets)
│   ├── final_models/                 # 최종 학습된 모델 (33개)
│   │   ├── model_registry.json
│   │   └── hyperparam/{category}/{dataset}/final_model/best_model.pt
│   ├── results/                      # 학습 결과 CSV
│   │   └── {dataset_name}_progress.csv
│   ├── analysis/                     # 분석 결과
│   └── src/
│       ├── loader.py                 # 데이터 로더
│       ├── TopExpert.py              # 모델 정의 (+ FocalLoss)
│       ├── main.py                   # 학습 스크립트
│       ├── splitters.py              # 데이터 분할
│       ├── utils.py                  # 유틸리티
│       ├── deployment/               # 배포 모듈
│       │   ├── unified_predictor.py  # 통합 예측기
│       │   └── model_loader.py       # 모델 로더
│       ├── pre-trained/              # 사전학습 모델
│       │   └── supervised_contextpred.pth
│       └── archives/                 # 보관된 스크립트
│           ├── analysis/             # 분석 스크립트
│           └── generation/           # 명령어 생성 스크립트
├── configs/
│   ├── dataset_config.json           # 데이터셋 설정
│   └── best_hyperparameters_final.json  # 최적 하이퍼파라미터
└── README.md                         # 이 파일
```

## 🔧 하이퍼파라미터 튜닝

### Random Search (권장)

```bash
# 1. 명령어 생성 (33 datasets × 100 combinations = 3,300 experiments)
python workspace/generate_random_search.py

# 2. 실행 (simple_gpu_scheduler 사용, 4 GPUs)
bash workspace/run_random_search.sh

# 또는 직접 실행
simple_gpu_scheduler --gpus 0 1 2 3 < workspace/commands_random_search_500epochs.txt
```

**설정:**
- Max epochs: 500
- Early stopping patience: 50
- Random combinations: 100 per dataset
- Total experiments: 3,300

**결과:**
- CSV 파일: `workspace/results/{dataset_name}_progress.csv`
- 각 행: 하나의 실험 결과 (experiment_id, hyperparameters, metrics)

### 단일 데이터셋 학습

```bash
# 기본 BCE Loss
python workspace/src/main.py \
  --category Toxicity \
  --dataset_name AMES \
  --batch_size 32 \
  --lr 1e-4 \
  --epochs 500 \
  --patience 50

# Focal Loss 사용 (불균형 데이터셋)
python workspace/src/main.py \
  --category Toxicity \
  --dataset_name ClinTox \
  --batch_size 32 \
  --lr 1e-4 \
  --epochs 500 \
  --patience 50 \
  --loss_type focal \
  --focal_alpha 0.25 \
  --focal_gamma 2.0
```

### 최종 모델 학습

```bash
# Train+Valid 결합하여 최종 학습
python workspace/src/main.py \
  --category Toxicity \
  --dataset_name AMES \
  --batch_size 32 \
  --lr 1e-4 \
  --epochs 500 \
  --patience 50 \
  --use_combined_trainvalid
```

## 📊 결과 분석

### CSV 파일 형식

`workspace/results/{dataset_name}_progress.csv`:

```csv
dataset,category,task_type,metric,experiment_id,lr,batch_size,dropout_ratio,num_layer,num_experts,alpha,beta,gate_dim,val_metric,test_metric,num_epochs,early_stopped,timestamp
AMES,Toxicity,classification,AUROC,exp_0001,0.001,512,0.1,5,3,0.1,0.1,50,78.5,78.2,181,True,2025-11-18 14:30:15
AMES,Toxicity,classification,AUROC,exp_0002,1e-05,32,0.3,7,7,0.1,0.1,300,75.3,74.8,95,True,2025-11-18 14:45:22
...
```

### 최적 하이퍼파라미터 추출

```python
import pandas as pd

# Load results
df = pd.read_csv('workspace/results/AMES_progress.csv')

# Find best experiment
best_idx = df['test_metric'].idxmax()  # AUROC의 경우
best_params = df.loc[best_idx]

print(f"Best AUROC: {best_params['test_metric']:.2f}%")
print(f"Learning rate: {best_params['lr']}")
print(f"Batch size: {best_params['batch_size']}")
```

## 🎯 주요 기능

### 1. Focal Loss (불균형 데이터셋 대응)

```python
# TopExpert.py에 구현됨
from TopExpert import FocalLoss

# 사용 예시
criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
```

**권장 데이터셋:**
- ClinTox (13.55:1 불균형)
- PAMPA_NCATS (5.32:1 불균형)
- Bioavailability_Ma (3.31:1 불균형)
- CYP2C9_Substrate (4.20:1 불균형)

### 2. 하이퍼파라미터 튜닝 시 모델 저장 제어

```bash
# 튜닝 모드: 모델 저장 안 함 (결과만 CSV에 저장)
python workspace/src/main.py --experiment_id exp_001 ...

# 최종 학습: 모델 저장
python workspace/src/main.py ...  # experiment_id 없음
```

### 3. 통합 예측 시스템

```python
from deployment.unified_predictor import ADMETPredictor

# 초기화
predictor = ADMETPredictor()
predictor.load_all_models()  # 33개 모델 로드

# 예측
results = predictor.predict("CCO")  # Ethanol

# 한국어 리포트
predictor.print_korean_report("CCO")
```

## 📈 성능 현황

### 카테고리별 성능 (최종 학습 기준)

| 카테고리 | 모델 수 | 평균 성능 | 최고 성능 |
|---------|--------|----------|----------|
| Absorption | 8 | AUROC 84.98% | HIA_Hou 95.88% |
| Distribution | 3 | R² 0.09 | BBB_Martins 86.84% |
| Metabolism | 8 | AUROC 71.06% | CYP3A4_Veith 78.74% |
| Excretion | 3 | R² -0.83 | - |
| Toxicity | 11 | AUROC 78.02% | ClinTox 88.61% |

### 전체 성능

- **분류 (21개)**: 평균 AUROC 80.35 ± 8.48%
- **회귀 (12개)**: 평균 R² 0.29 ± 0.28

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
conda create -n ADMET python=3.11
conda activate ADMET

# 패키지 설치
pip install torch==2.8.0 torch-geometric==2.6.1
pip install rdkit-pypi
pip install simple-gpu-scheduler

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
