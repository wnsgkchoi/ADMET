# Post-Training Analysis Workflow

학습 완료 후 실행할 작업들의 순서와 명령어를 정리한 문서입니다.

## 📋 작업 순서

### 1️⃣ 최종 성능 수집 및 분석
**목적**: 33개 모델의 최종 학습 성능 수집 및 튜닝 결과와 비교

**실행**:
```bash
cd /home/choi0425/workspace/ADMET
conda run -n ADMET python workspace/src/collect_final_results.py
```

**산출물**:
- `workspace/final_training_results.csv` - 상세 결과
- `workspace/final_training_summary.json` - 통계 요약
- `workspace/final_training_full_results.json` - 전체 결과

---

### 2️⃣ 성능 지표 정리 (AUROC & R²)
**목적**: Binary classification의 AUROC와 Continuous regression의 R² 계산 및 정리

**실행**:
```bash
conda run -n ADMET python workspace/src/compile_performance_metrics.py
```

**산출물**:
- `workspace/analysis/detailed_performance_metrics.json` - 상세 메트릭
- `workspace/analysis/performance_metrics_table.csv` - 성능 테이블
- `workspace/analysis/performance_summary.json` - 카테고리별 통계

**내용**:
- Classification: AUROC, AUPRC, Accuracy, F1, Sensitivity, Specificity
- Regression: MAE, RMSE, R², Pearson, Spearman
- 카테고리별 통계 (Absorption, Distribution, Metabolism, Excretion, Toxicity)

---

### 3️⃣ Continuous 데이터셋 분포 분석
**목적**: Regression 데이터셋의 train/valid/test/all 분포 분석

**실행**:
```bash
conda run -n ADMET python workspace/src/analyze_distributions.py
```

**산출물**:
- `workspace/analysis/continuous_distribution_analysis.json` - 상세 분석
- `workspace/analysis/continuous_distribution_summary.csv` - 요약 테이블
- `workspace/analysis/distributions/*.png` - 분포 시각화 (각 데이터셋별)

**분석 내용**:
- 기술 통계량 (mean, std, median, min, max, IQR)
- 왜도(skewness), 첨도(kurtosis)
- 분포 차이 검정 (Kolmogorov-Smirnov test)
- 시각화:
  - Histogram 비교
  - Box plot
  - Q-Q plot (train vs test)
  - Kernel Density Estimation

---

### 4️⃣ 통합 ADMET 예측 시스템 구축
**목적**: 하나의 SMILES 입력으로 33개 항목 동시 예측

**데모 실행**:
```bash
conda run -n ADMET python workspace/src/unified_predictor.py
```

**사용 예시** (SMILES to graph 구현 후):
```python
from workspace.src.unified_predictor import ADMETPredictor

# 초기화
predictor = ADMETPredictor()
predictor.load_all_models()

# 단일 분자 예측
predictions = predictor.predict("CCO")  # Ethanol

# 결과 구조:
{
    'Absorption': {
        'Caco2_Wang': 0.365,
        'HIA_Hou': 0.97,
        'Pgp_Broccatelli': 0.85,
        ...
    },
    'Distribution': {
        'BBB_Martins': 0.82,
        'PPBR_AZ': 88.69,
        'VDss_Lombardo': 2.00
    },
    'Metabolism': {...},
    'Excretion': {...},
    'Toxicity': {...}
}

# 배치 예측
smiles_list = ["CCO", "CC(C)O", "CCC"]
results_df = predictor.predict_batch(smiles_list)
```

**주의사항**:
- SMILES to graph 변환 로직이 필요함
- `loader.py`의 `mol_to_graph_data_obj_simple` 함수 참고
- RDKit 기반 분자 featurization 구현 필요

---

## 🔄 전체 실행 스크립트

모든 분석을 한 번에 실행:

```bash
#!/bin/bash
cd /home/choi0425/workspace/ADMET

echo "=== 1. Collecting Final Results ==="
conda run -n ADMET python workspace/src/collect_final_results.py

echo ""
echo "=== 2. Compiling Performance Metrics ==="
conda run -n ADMET python workspace/src/compile_performance_metrics.py

echo ""
echo "=== 3. Analyzing Distributions ==="
conda run -n ADMET python workspace/src/analyze_distributions.py

echo ""
echo "=== 4. Testing Unified Predictor ==="
conda run -n ADMET python workspace/src/unified_predictor.py

echo ""
echo "=== All Analysis Complete! ==="
echo "Check workspace/analysis/ for results"
```

---

## 📊 예상 산출물 구조

```
workspace/
├── analysis/
│   ├── detailed_performance_metrics.json
│   ├── performance_metrics_table.csv
│   ├── performance_summary.json
│   ├── continuous_distribution_analysis.json
│   ├── continuous_distribution_summary.csv
│   └── distributions/
│       ├── Caco2_Wang_distribution.png
│       ├── Lipophilicity_AstraZeneca_distribution.png
│       ├── PPBR_AZ_distribution.png
│       └── ... (모든 continuous 데이터셋)
├── final_training_results.csv
├── final_training_summary.json
└── final_training_full_results.json

workspace/final_models/
└── model_registry.json  # 통합 예측 시스템이 사용

workspace/src/
├── collect_final_results.py
├── compile_performance_metrics.py
├── analyze_distributions.py
└── unified_predictor.py
```

---

## ✅ 체크리스트

학습 완료 후:

- [ ] 1. 최종 성능 수집 실행
- [ ] 2. 성능 지표 정리 실행
- [ ] 3. 분포 분석 실행
- [ ] 4. 통합 예측 시스템 테스트
- [ ] 5. SMILES to graph 변환 구현 (통합 예측용)
- [ ] 6. 결과 검토 및 문서화

---

## 📝 참고사항

1. **성능 메트릭**:
   - Binary classification: AUROC (primary), AUPRC, Accuracy, F1
   - Continuous regression: MAE (primary), RMSE, R², Pearson, Spearman

2. **데이터셋 개수**:
   - Classification: 21개
   - Regression: 12개
   - Total: 33개

3. **카테고리**:
   - Absorption: 8개
   - Distribution: 3개
   - Metabolism: 8개
   - Excretion: 3개
   - Toxicity: 11개
