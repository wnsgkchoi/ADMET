# 33개 항목 통합 예측 모델 구축 계획안

## 📋 프로젝트 개요

**목표**: 하나의 SMILES 입력에 대해 33개 ADMET 항목을 동시에 예측하는 통합 모델 시스템 구축

**데이터**: 2025.11.07 수령한 33개 항목 데이터셋
- **카테고리**: Absorption (8), Distribution (3), Metabolism (8), Excretion (3), Toxicity (11)
- **Task Type**: Classification (21개), Regression (12개)
- **데이터 형식**: 각 데이터셋마다 이미 train/valid/test로 분할됨
- **데이터 구조**: `Drug_ID`, `Drug` (SMILES), `Y` (target) 컬럼

**최종 산출물**:
- 33개의 개별 최적화 모델 (각 항목별 best model)
- 통합 예측 스크립트 (`predict.py`)
- 성능 평가 보고서

---

## 🗂️ 1단계: 데이터 분석 및 전처리

### 1.1 데이터셋 구조 파악 ✅ (이미 완료됨!)

**확인된 데이터 구조**:
```
workspace/data/
├── ADMET_Summary.txt
├── Absorption/ (8개)
│   ├── Caco2_Wang/ (regression)
│   ├── Lipophilicity_AstraZeneca/ (regression)
│   ├── Solubility_AqSolDB/ (regression)
│   ├── HydrationFreeEnergy_FreeSolv/ (regression)
│   ├── HIA_Hou/ (classification)
│   ├── Pgp_Broccatelli/ (classification)
│   ├── Bioavailability_Ma/ (classification)
│   └── PAMPA_NCATS/ (classification)
├── Distribution/ (3개)
│   ├── BBB_Martins/ (classification)
│   ├── PPBR_AZ/ (regression)
│   └── VDss_Lombardo/ (regression)
├── Metabolism/ (8개 - 모두 classification)
│   ├── CYP2C19_Veith/
│   ├── CYP2D6_Veith/
│   ├── CYP3A4_Veith/
│   ├── CYP1A2_Veith/
│   ├── CYP2C9_Veith/
│   ├── CYP2C9_Substrate_CarbonMangels/
│   ├── CYP2D6_Substrate_CarbonMangels/
│   └── CYP3A4_Substrate_CarbonMangels/
├── Excretion/ (3개 - 모두 regression)
│   ├── Half_Life_Obach/
│   ├── Clearance_Hepatocyte_AZ/
│   └── Clearance_Microsome_AZ/
└── Toxicity/ (11개)
    ├── LD50_Zhu/ (regression)
    ├── hERG/ (classification)
    ├── hERG_Karim/ (classification)
    ├── AMES/ (classification)
    ├── DILI/ (classification)
    ├── Skin_Reaction/ (classification)
    ├── Carcinogens_Lagunin/ (classification)
    ├── ClinTox/ (classification)
    ├── hERG_Central_1uM/ (regression)
    ├── hERG_Central_10uM/ (regression)
    └── hERG_Central_inhib/ (classification)
```

**각 데이터셋 폴더 구조**:
```
{dataset_name}/
├── full_data.csv
├── train.csv
├── valid.csv
├── test.csv
└── metadata.txt (task type, 통계 정보 등)
```

**CSV 컬럼**: `Drug_ID`, `Drug` (SMILES), `Y` (target)

**Action Items**:
- [x] 데이터셋 구조 확인 완료
- [ ] `dataset_config.json` 생성 (metadata.txt 정보를 JSON으로 통합)
- [ ] 데이터 품질 체크 스크립트 작성
  - Invalid SMILES 확인
  - 결측치 확인
  - Train/Valid/Test split 비율 확인

### 1.2 데이터 로더 수정

**현재 상황**: 
- 기존 `loader.py`는 단일 CSV 파일을 가정
- 새 데이터는 이미 train/valid/test로 분할되어 있음

**필요한 수정**:
```python
# 기존: data/train/dataset.csv 형식
# 신규: data/Category/Dataset_Name/train.csv 형식

def load_dataset(category, dataset_name):
    """
    Args:
        category: 'Absorption', 'Distribution', etc.
        dataset_name: 'Caco2_Wang', 'AMES', etc.
    Returns:
        train_data, valid_data, test_data, metadata
    """
    base_path = f"workspace/data/{category}/{dataset_name}"
    train = pd.read_csv(f"{base_path}/train.csv")
    valid = pd.read_csv(f"{base_path}/valid.csv")
    test = pd.read_csv(f"{base_path}/test.csv")
    # metadata 읽기
    return train, valid, test, metadata
```

**Action Items**:
- [ ] `loader.py`에 새로운 `load_admet_dataset()` 함수 추가
- [ ] Metadata 파싱 함수 추가
- [ ] 전체 33개 데이터셋 로드 테스트

---

## 🏋️ 2단계: 개별 모델 학습

### 2.1 Main.py 수정

**현재 상황**: 
- `main.py`는 단일 데이터셋 학습용
- dataset 이름이 하드코딩됨 (dili2, dili3, hk2, hepg2)

**필요한 수정**:
```python
# 기존
parser.add_argument('--dataset', type=str, default='dili2', 
                    choices=['dili2', 'dili3', 'hk2', 'hepg2'])

# 신규
parser.add_argument('--category', type=str, required=True,
                    choices=['Absorption', 'Distribution', 'Metabolism', 'Excretion', 'Toxicity'])
parser.add_argument('--dataset', type=str, required=True,
                    help='Dataset name (e.g., Caco2_Wang, AMES)')
```

**Action Items**:
- [ ] `main.py` 수정: 새로운 데이터 경로 지원
- [ ] Task type을 metadata.txt에서 자동으로 읽도록 수정
- [ ] 결과 저장 경로 변경: `output/hyperparam/{category}/{dataset}_progress.csv`

### 2.2 Hyperparameter Search (Per Dataset)

**목표**: 33개 데이터셋 각각에 대해 최적 하이퍼파라미터 찾기

**전략**:
```python
# Option 1: 전체 Grid Search (비현실적)
33 datasets × 6,912 configs = 228,096 experiments (약 2-3주 소요)

# Option 2: 축소된 Grid Search (권장)
- 기존 4개 데이터셋 결과에서 유망한 범위 추출
- Search space 축소 (예: 1,000 configs per dataset)
- 33 datasets × 1,000 configs = 33,000 experiments (약 3-5일)

# Option 3: Random Search
- 각 데이터셋당 200-500 random configs
- 33 datasets × 300 configs = 9,900 experiments (약 1-2일)
```

**권장 접근법**: Option 2 + Option 3 조합
1. 먼저 각 데이터셋당 Random Search (300 configs)로 빠른 탐색
2. 상위 성능 영역에서 Grid Search로 fine-tuning

**Search Space (수정됨)**:
```python
{
    'lr': [1e-4, 1e-3, 1e-2],        # 3
    'dropout_ratio': [0, 0.3, 0.5],  # 3
    'batch_size': [32, 512],         # 2
    'num_experts': [3, 5, 7],        # 3
    'alpha': [1, 0.1, 0.01],         # 3
    'beta': [1, 0.1, 0.01],          # 3
    'min_temp': [0.1, 1],            # 2
    'decay': [0, 0.0001]             # 2
}
# Total: 3 × 3 × 2 × 3 × 3 × 3 × 2 × 2 = 1,944 configs per dataset
# 33 datasets × 1,944 = 64,152 experiments

# 예상 소요 시간 계산:
# - 평균 실행 시간: ~10분/experiment (early stopping 적용)
# - 4 GPU 병렬 실행: 64,152 / 4 = 16,038 experiments per GPU
# - 총 소요 시간: 16,038 × 10분 = 160,380분 ≈ 111시간 ≈ 4.6일
# - Early stopping으로 실제 30-50% 단축 예상 → 약 3-4일
```

**실행 방법**:
```bash
# 1. Grid search 명령어 생성 스크립트 작성
python generate_grid_search_33datasets.py --mode reduced > commands_33datasets.txt

# 2. simple-gpu-scheduler로 실행
cd /home/choi0425/workspace/ADMET
nohup bash -c 'cat commands_33datasets.txt | conda run --no-capture-output -n ADMET \
  python -m simple_gpu_scheduler.scheduler --gpus 0 1 2 3' > scheduler_33.log 2>&1 &

# 3. 진행 상황 모니터링
ps aux | grep main.py | wc -l  # 실행 중인 작업 수
tail -f scheduler_33.log        # 스케줄러 로그
```

**Action Items**:
- [ ] `generate_grid_search_33datasets.py` 스크립트 작성
  - 33개 데이터셋 목록 자동 생성
  - Category와 dataset_name을 인자로 전달
  - 결과 저장 경로: `output/hyperparam/{category}/{dataset}_progress.csv`
- [ ] Grid search 실행
- [ ] 진행 상황 모니터링 스크립트 작성

### 2.3 Best Model 선정 및 재학습

**선정 기준**:
- Classification: Validation AUROC 최대
- Regression: Validation MAE 최소

**재학습 전략** (일관성 있게 통일):

**🔵 전략 A: Conservative (권장 - 재현성 우선)**
```python
# 대규모 데이터셋 (train >= 500)
- Grid search: train으로 학습, valid로 early stopping
- Best config 선정: validation metric 기준
- Final model: train으로 재학습, valid로 early stopping
- Test 평가: test set으로만 평가

# 소규모 데이터셋 (train < 500)
- Grid search: train+valid를 5-fold CV로 분할
  - 각 fold: 80%로 학습, 20%로 검증
  - Fold 평균 metric으로 best config 선정
- Final model: train+valid 전체로 학습 (고정 epoch)
  - Epoch 수 = CV 시 best epoch의 평균값
- Test 평가: test set으로만 평가 (독립적 유지)
```

**🟢 전략 B: Aggressive (최대 성능 우선)**
```python
# 모든 데이터셋 공통
- Grid search: train으로 학습, valid로 early stopping
- Best config 선정: validation metric 기준
- Final model: train+valid 합쳐서 재학습 (고정 epoch)
  - Epoch 수 = grid search 시 best epoch의 평균값
- Test 평가: test set으로만 평가

# 소규모 데이터셋 추가 고려사항
- Train+valid 합친 데이터가 여전히 작음 (< 700)
- 5-fold CV로 robustness 확인 가능
- 하지만 final model은 동일하게 train+valid 사용
```

**📊 전략 비교**:

| 항목 | 전략 A (Conservative) | 전략 B (Aggressive) |
|------|---------------------|-------------------|
| 대규모 데이터셋 | Train만 사용 | Train+Valid 사용 |
| 소규모 데이터셋 | Train+Valid 사용 (CV) | Train+Valid 사용 |
| 일관성 | ❌ 데이터셋마다 다름 | ✅ 모두 동일 전략 |
| 재현성 | ✅ Early stopping 사용 | ⚠️ 고정 epoch |
| 성능 | 보수적 | 최대화 |
| 과적합 위험 | 낮음 | 중간 |

**🎯 최종 권장**: **전략 B (Aggressive)** 
- **이유 1**: 일관성 - 모든 데이터셋에 동일한 전략 적용
- **이유 2**: 성능 - Test set은 독립적으로 유지되므로 train+valid 활용이 합리적
- **이유 3**: 단순성 - 소규모/대규모 구분 불필요
- **이유 4**: 실무 적합성 - 최종 배포 시 가능한 모든 데이터 활용

### 2.4 K-Fold Cross Validation (소규모 데이터셋 검증)

**목적**: 소규모 데이터셋의 hyperparameter 신뢰성 향상

**데이터셋 규모 분석**:
- 평균 샘플 수: ~22,000개
- 최소 샘플 수: 196개 (Carcinogens_Lagunin)
- 샘플 < 500개인 데이터셋: 10개

**소규모 데이터셋 목록** (train < 500):
1. Carcinogens_Lagunin (196)
2. Skin_Reaction (282)
3. DILI (332)
4. HIA_Hou (404)
5. Bioavailability_Ma (448)
6. HydrationFreeEnergy_FreeSolv (449)
7. hERG (458)
8. Half_Life_Obach (466)
9. CYP2D6_Substrate_CarbonMangels (466)
10. CYP2C9_Substrate_CarbonMangels (468)
11. CYP3A4_Substrate_CarbonMangels (468)

**K-Fold 전략** (전략 B 기준):
```python
# 대규모 데이터셋 (train >= 500)
- Grid search: train으로 학습, valid로 검증
- Best config: validation metric 기준
- Final model: train+valid 합쳐서 학습 (고정 epoch = best epoch 평균)

# 소규모 데이터셋 (train < 500) - K-fold 적용
- Grid search: train+valid를 5-fold CV
  - 각 config당 5번 학습 (각 fold)
  - Metric = 5-fold 평균 ± std
  - Best config: CV 평균 metric 기준
- Final model: train+valid 전체로 학습 (고정 epoch = CV best epoch 평균)

# Test set은 모든 경우에 독립적으로 유지 (평가용)
```

**구현 방안**:
- [ ] `main.py`에 `--use_kfold` 플래그 추가
- [ ] K-fold 자동 감지: `len(train) < 500`이면 자동 활성화
- [ ] Fold별 결과 저장: `{dataset}_fold{k}_progress.csv`
- [ ] Fold 평균 성능 계산 및 기록

**Impact on Grid Search**:
- 10개 데이터셋 × 5 folds ≈ 추가 시간
- 소규모 데이터셋은 epoch도 빠르므로 실제 영향 < 1일
- 총 Grid search 예상: 4-5일 (변동 없음)

**Action Items**:
- [ ] `analyze_grid_results.py` 스크립트 작성
  - 33개 CSV 파일 읽기 (K-fold인 경우 fold 평균)
  - 각 데이터셋별 best config 추출
  - Best epoch 평균 계산 (최종 모델 학습용)
  - `configs/best_configs.json` 생성
  - Summary table 출력 (dataset, task_type, best_metric, hyperparameters, best_epoch)
- [ ] `train_final_models.py` 스크립트 작성
  - `best_configs.json` 읽기
  - 33개 데이터셋 순차 학습 (또는 4-GPU 병렬)
  - **모든 데이터셋**: train+valid 합쳐서 학습 (전략 B)
  - Early stopping 없이 고정 epoch (= best_epoch from grid search)
  - 모델 저장: `models/final/{category}/{dataset}_best.pth`
  - Config도 함께 저장: `models/final/{category}/{dataset}_config.json`
- [ ] 최종 모델 학습 실행
- [ ] Test set 성능 평가 및 기록

**예상 소요 시간**: 
- Grid search: 4-5일 (K-fold 포함)
- Analysis: 1시간
- Final training: 
  - 순차 실행: 33 datasets × 20 min = 11시간
  - 병렬 실행 (4 GPU): 3시간

---

## 🔮 3단계: 통합 예측 시스템 구축

### 3.1 Prediction Script 작성

**기능**:
1. SMILES 문자열 입력
2. 33개 모델 로드
3. 각 모델에서 예측 수행
4. 결과 통합 및 출력

**인터페이스**:
```python
# predict.py 사용 예시

# 단일 SMILES 예측
python predict.py --smiles "CC(C)Cc1ccc(cc1)C(C)C(O)=O"

# 배치 예측 (CSV 파일)
python predict.py --input predictions.csv --output results.csv

# 출력 형식
{
  "smiles": "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
  "predictions": {
    "항목1": {"prediction": 0.85, "type": "probability"},
    "항목2": {"prediction": 1, "type": "class"},
    "항목3": {"prediction": 4.52, "type": "value"},
    ...
  }
}
```

**Action Items**:
- [ ] `predict.py` 스크립트 작성
- [ ] Model loading 함수 구현
- [ ] SMILES → Graph 변환 함수 (loader.py 활용)
- [ ] Batch prediction 지원
- [ ] 출력 형식 정의 (JSON/CSV)

### 3.2 Model Manager 클래스 설계

```python
class IntegratedPredictor:
    def __init__(self, model_dir, config_path):
        """
        Args:
            model_dir: 모델 파일들이 저장된 디렉토리
            config_path: dataset_config.json 경로
        """
        self.models = {}  # {item_name: model}
        self.configs = {}  # {item_name: config}
        self.load_models()
    
    def load_models(self):
        """33개 모델을 메모리에 로드"""
        pass
    
    def predict_single(self, smiles):
        """단일 SMILES에 대한 예측"""
        pass
    
    def predict_batch(self, smiles_list):
        """여러 SMILES에 대한 배치 예측"""
        pass
    
    def save_predictions(self, predictions, output_path):
        """예측 결과를 파일로 저장"""
        pass
```

**Action Items**:
- [ ] `IntegratedPredictor` 클래스 구현
- [ ] GPU 메모리 효율적 사용 (필요시 모델 on-demand loading)
- [ ] 예측 속도 벤치마크

---

## 📊 4단계: 성능 평가 및 검증

### 4.1 개별 모델 성능 평가

**평가 항목**:
- Test set 성능 (각 항목별 primary metric)
- 학습 시간
- 모델 크기

**Action Items**:
- [ ] `evaluate_final_models.py` 스크립트 작성
- [ ] 성능 테이블 생성
- [ ] 시각화 (confusion matrix, prediction vs actual 등)

### 4.2 통합 시스템 검증

**검증 항목**:
1. **정확성**: 개별 예측 vs 통합 예측 일치 확인
2. **속도**: 33개 예측 처리 시간 측정
3. **안정성**: Edge case 처리 (invalid SMILES 등)

**Action Items**:
- [ ] Unit test 작성
- [ ] 샘플 데이터로 end-to-end 테스트
- [ ] 성능 보고서 작성

---

## 🛠️ 5단계: 코드 정리 및 문서화

### 5.1 디렉토리 구조
```
ADMET/
├── workspace/
│   ├── data/
│   │   ├── ADMET_Summary.txt
│   │   ├── Absorption/
│   │   │   ├── Caco2_Wang/
│   │   │   │   ├── train.csv
│   │   │   │   ├── valid.csv
│   │   │   │   ├── test.csv
│   │   │   │   ├── full_data.csv
│   │   │   │   └── metadata.txt
│   │   │   └── ...
│   │   ├── Distribution/
│   │   ├── Metabolism/
│   │   ├── Excretion/
│   │   └── Toxicity/
│   ├── models/
│   │   └── final/
│   │       ├── Absorption/
│   │       │   ├── Caco2_Wang_best.pth
│   │       │   ├── Caco2_Wang_config.json
│   │       │   └── ...
│   │       ├── Distribution/
│   │       ├── Metabolism/
│   │       ├── Excretion/
│   │       └── Toxicity/
│   ├── output/
│   │   ├── hyperparam/
│   │   │   ├── Absorption/
│   │   │   │   ├── Caco2_Wang_progress.csv
│   │   │   │   └── ...
│   │   │   ├── Distribution/
│   │   │   ├── Metabolism/
│   │   │   ├── Excretion/
│   │   │   └── Toxicity/
│   │   └── evaluation/
│   │       ├── final_results_summary.csv
│   │       └── per_dataset_metrics.json
│   └── src/
│       ├── main.py                          # 수정: 새 데이터 경로 지원
│       ├── loader.py                        # 수정: load_admet_dataset() 추가
│       ├── TopExpert.py                     # 기존 유지
│       ├── splitters.py                     # 기존 유지
│       ├── utils.py                         # 기존 유지
│       ├── predict.py                       # 신규: 예측 스크립트
│       ├── integrated_predictor.py          # 신규: 통합 예측 클래스
│       ├── generate_grid_search_33datasets.py  # 신규
│       ├── analyze_grid_results.py          # 신규
│       └── train_final_models.py            # 신규
├── configs/
│   ├── dataset_config.json                  # 33개 데이터셋 메타데이터
│   └── best_configs.json                    # Best hyperparameters
├── commands_33datasets.txt                  # Grid search 명령어
└── note/
    ├── research_note.md
    └── integration_plan.md
```

### 5.2 문서화

**README 작성**:
- [ ] 프로젝트 개요
- [ ] 설치 방법
- [ ] 사용 예시
- [ ] API 문서

**주석 및 Docstring**:
- [ ] 모든 함수에 docstring 추가
- [ ] 복잡한 로직에 주석 추가

---

## 📅 타임라인 (수정됨)

| 단계 | 작업 | 예상 소요 시간 | 우선순위 |
|------|------|----------------|----------|
| 1.1 | 데이터셋 구조 파악 ✅ | 완료 | - |
| 1.2 | 데이터 로더 수정 | 0.5일 | High |
| 1.3 | Dataset config 생성 | 0.5일 | High |
| 2.1 | Main.py 수정 (새 경로 지원 + K-fold) | 1일 | High |
| 2.2 | Grid search 스크립트 작성 | 0.5일 | High |
| 2.2 | Grid search 실행 (1,944 configs × 33 datasets) | 4-5일 | Medium |
| 2.3 | Best config 분석 (K-fold 평균 포함) | 0.5일 | Medium |
| 2.3 | Final models 학습 | 0.5일 | Medium |
| 3.1 | Predict.py 작성 | 1일 | High |
| 3.2 | IntegratedPredictor 구현 | 1일 | High |
| 4 | 성능 평가 및 검증 | 0.5일 | Low |
| 5 | 문서화 | 0.5일 | Low |

**총 예상 기간**: 10-12일

**우선순위별 실행 계획**:
1. **Phase 1 (1-2일)**: 데이터 로더 + Main.py 수정 (K-fold 포함) + 테스트
2. **Phase 2 (4-5일)**: Grid search 실행 (백그라운드, 64,152 experiments)
3. **Phase 3 (2일)**: Grid search 진행 중 예측 시스템 구현
4. **Phase 4 (1-2일)**: Final models 학습 + 평가

---

## ⚠️ 위험 요소 및 대응 방안

### 1. 메모리 부족
**문제**: 33개 모델을 동시에 메모리에 로드 시 OOM
**대응**: 
- On-demand model loading
- CPU로 일부 모델 offload
- 모델 경량화 (pruning, quantization)

### 2. 데이터 불균형
**문제**: 특정 항목의 데이터가 부족하거나 편향됨
**대응**:
- Class weighting (이미 구현됨)
- K-fold cross validation (샘플 < 500)
- 데이터 증강 (SMILES augmentation) - 향후 고려

### 3. 학습 시간 초과
**문제**: Grid search가 예상보다 오래 걸림
**대응**:
- Early stopping 적용 (이미 구현됨)
- 특정 데이터셋만 먼저 실행하여 검증
- 필요시 search space 재조정

### 4. 모델 간 성능 편차
**문제**: 일부 항목에서 성능이 현저히 낮음
**대응**:
- 데이터셋별 best baseline과 비교
- Task-specific architecture 조정
- K-fold로 안정성 향상

### 5. K-Fold 구현 복잡도
**문제**: K-fold가 기존 코드와 충돌할 수 있음
**대응**:
- `--use_kfold` 플래그로 선택적 활성화
- 소규모 데이터셋에만 적용 (< 500 samples)
- Fold별 결과 저장 후 평균 계산

---

## 📝 체크리스트

### Phase 1: 데이터 인프라 구축 (1-2일)
- [x] 33개 데이터셋 구조 확인
- [x] Task type 분류 (21 classification, 12 regression)
- [x] 데이터 규모 분석 (평균 22k, 최소 196, 소규모 10개)
- [ ] `dataset_config.json` 생성 (metadata.txt → JSON 통합)
- [ ] `loader.py`에 `load_admet_dataset()` 함수 추가
- [ ] `main.py` 수정: category/dataset 인자 지원
- [ ] K-fold 기능 추가 (샘플 < 500인 경우)
- [ ] 단일 데이터셋 학습 테스트 (예: Caco2_Wang - 일반, DILI - K-fold)

### Phase 2: Hyperparameter Search (4-5일)
- [ ] `generate_grid_search_33datasets.py` 작성
  - [ ] Search space 정의 (1,944 configs/dataset)
  - [ ] 33개 데이터셋 자동 순회
  - [ ] K-fold 데이터셋 감지 및 처리
  - [ ] 명령어 생성: `commands_33datasets.txt` (64,152 experiments)
- [ ] Grid search 실행 (백그라운드)
- [ ] 모니터링 스크립트 작성
  - [ ] 진행률 체크 (완료/전체)
  - [ ] 평균 성능 트래킹
  - [ ] 실패한 실험 추적
  - [ ] K-fold 결과 aggregation

### Phase 3: 예측 시스템 구축 (2일, Grid search와 병렬)
- [ ] `IntegratedPredictor` 클래스 구현
  - [ ] 모델 로딩 메커니즘
  - [ ] 배치 예측 지원
  - [ ] 결과 포맷팅
- [ ] `predict.py` CLI 스크립트 작성
  - [ ] 단일 SMILES 예측
  - [ ] CSV 배치 예측
  - [ ] JSON/CSV 출력 옵션
- [ ] Unit test 작성
- [ ] 예제 실행 및 검증

### Phase 4: Final Models & Evaluation (1-2일)
- [ ] `analyze_grid_results.py` 실행
  - [ ] K-fold 결과 평균 계산
  - [ ] Best configs 추출
  - [ ] 성능 summary table 생성
- [ ] `train_final_models.py` 실행
  - [ ] 33개 최종 모델 학습
  - [ ] 소규모 데이터셋: full_data 사용
  - [ ] 모델 + config 저장
- [ ] Test set 평가
  - [ ] 개별 성능 측정
  - [ ] Baseline 비교 (가능한 경우)
  - [ ] 통합 보고서 작성
- [ ] 통합 예측 시스템 E2E 테스트
  - [ ] 33개 모델 동시 로드 테스트
  - [ ] 메모리 사용량 확인
  - [ ] 예측 속도 벤치마크

### Phase 5: 문서화 & 정리 (0.5일)
- [ ] README 작성
  - [ ] 프로젝트 개요
  - [ ] 설치 방법
  - [ ] 사용 예시 (단일/배치 예측)
- [ ] API 문서 작성
- [ ] 성능 보고서 작성
- [ ] 코드 리팩토링 및 주석 추가

---

## 🎯 성공 지표

1. **데이터 인프라**
   - ✅ 33개 데이터셋 모두 정상 로드
   - ✅ Train/Valid/Test split 유지
   - ✅ Invalid SMILES < 1%

2. **모델 성능**
   - Classification: 평균 Test AUROC > 0.70
   - Regression: 평균 Test MAE < dataset별 baseline
   - 각 데이터셋별 성능이 기존 발표된 성능 수준 이상

3. **시스템 안정성**
   - 99% 이상의 valid SMILES에 대해 예측 성공
   - 평균 예측 시간 < 2초 (33개 모델 모두, 단일 SMILES)
   - 배치 예측 처리량 > 100 SMILES/분

4. **코드 품질**
   - 핵심 함수에 unit test 존재
   - 문서화율 > 70%
   - 재현 가능한 결과 (시드 고정)

**Baseline 비교**:
- 각 데이터셋의 metadata에 기존 논문 성능이 있는 경우 비교
- 없는 경우: Random Forest, Simple GNN 등과 비교

---

## 💡 향후 개선 사항

1. **웹 인터페이스 구축**
   - Flask/FastAPI 기반 REST API
   - Gradio/Streamlit 기반 웹 UI

2. **모델 업데이트 파이프라인**
   - 새로운 데이터로 주기적 재학습
   - A/B 테스트 프레임워크

3. **성능 최적화**
   - TorchScript 변환
   - ONNX 변환 및 최적화
   - 배치 추론 최적화

4. **설명 가능성 (Explainability)**
   - Attention visualization
   - SHAP values
   - Molecular substructure highlighting
