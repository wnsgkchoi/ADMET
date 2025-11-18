# ADMET Prediction Model  

2025.11.18. Junha Choi  
</br></br></br></br></br></br>

## 1. 실험 세팅
### 1.1 하드웨어 정보
- **GPU**: 4x NVIDIA RTX A5000 (24GB VRAM each)
- **CPU**: 64 cores (32 physical cores with hyperthreading)
- **GPU Driver**: NVIDIA 535.230.02  

### 1.2 소프트웨어 환경
- **Python**: 3.11.13
- **PyTorch**: 2.8.0
- **PyTorch Geometric**: 2.6.1
- **CUDA**: 12.8
- **운영체제**: Linux

## 2. 데이터셋 정보

### 2.1 hERG 데이터셋
- **데이터셋**: hERG (Human Ether-à-go-go-Related Gene)
- **target**: hERG 채널 차단 예측
- **task**: 이진 분류 (심장독성 예측)
- **데이터 포맷**: SMILES → 분자 그래프  

### 2.2 LD50_Zhu 데이터셋  
- **데이터셋**: LD50_Zhu (급성 독성)
- **target**: LD50 값 (반수치사량) 예측
- **task**: 회귀 (Regression)
- **데이터 크기**: 7,385개 화합물
- **데이터 포맷**: SMILES → 분자 그래프
- **단위**: mg/kg (체중 대비 용량)
- **의미**: 낮은 값 = 높은 독성, 높은 값 = 낮은 독성  
![LD50_Zhu dataset figure](../pic/ld50_distribution_analysis.png)  

### 2.3 데이터 처리 설정
```python
BATCH_SIZE = 512
NUM_WORKERS = 24        # 데이터 로더 워커 수 (CPU 활용 최적화)
```

### 2.4 데이터 분할
- **분할 방법**: Random split (무작위 분할)
- **분할 비율**: Train 70% / Validation 10% / Test 20%
- **재현성**: Seed 42로 고정
- **전처리**: 분자 descriptor 추가 (10개 물리화학적 특성)

## 3. 모델 아키텍처

### 3.1 베이스 모델: Pre-trained GIN
- **모델**: Graph Isomorphism Network (GIN)
- **사전 학습 모델**: supervised_contextpred.pth
- **소스**: [Stanford Pre-trained GNNs](https://github.com/snap-stanford/pretrain-gnns)

```python
GNN_TYPE = 'gin'
NUM_LAYER = 5
EMB_DIM = 300
DROPOUT_RATIO = 0
GRAPH_POOLING = 'mean'
JK = 'last'              # Jumping Knowledge
```

### 3.2 TopExpert 아키텍처
- **전문가 수**: 7개
- **게이트 차원**: 64
- **구조**: Mixture of Experts with scaffold-based expert assignment
- **Scaffold 사용**: Bemis-Murcko scaffold (전문가 할당용, 분할 아님)

```python
NUM_EXPERTS = 7
GATE_DIM = 64
```

### 3.3 온도 어닐링 (Gumbel-Softmax)
```python
INIT_TEMP = 10.0        # 초기 온도
MIN_TEMP = 1.0          # 최소 온도
# 학습 중 지수적 감소
```

## 4. 학습 설정

```python
EPOCHS = 500 / 10000    # classification / regression  
LR = 1e-04              # 학습률
DECAY = 0.0001          # Weight decay
SEED = 42               # 재현성을 위한 시드
ALPHA = 0.1             # 클러스터링 손실 가중치
BETA = 0.1              # 정렬 손실 가중치
EMB_DIM = 300           # 임베딩 차원 (사전학습 모델 사용)
```

## 5. 평가 메트릭

### 5.1 분류 태스크 (hERG)
- **AUC (Area Under ROC Curve)**: 주요 평가 지표
- **F1 Score**: 보조 평가 지표 (threshold = 0.5)

### 5.2 회귀 태스크 (LD50_Zhu)
- **MAE (Mean Absolute Error)**: 평균 절대 오차
- **RMSE (Root Mean Square Error)**: 제곱근 평균 제곱 오차  
- **R² (Coefficient of Determination)**: 결정 계수
- **Spearman Correlation**: 스피어만 상관계수 (순위 기반)

---

## 6. 초기 실험 결과 (하이퍼파라미터 튜닝 X)

### 6.1 hERG 채널 차단 예측  

|model|Test AUC|Test F1-score|
|-|-|-|
|pre-trained GIN + TopExpert|0.7969(± 0.0018)|0.8476|
|XGBoost|0.7531|0.7646|
|LightGBM|0.7818|0.7421|
|CatBoost|0.7853|0.7434|

![pre-trained GIN + TopExpert](../pic/training_plot_single_20250922_053109.png)  

### 6.2 LD50_Zhu 급성 독성 예측

|model|MAE|RMSE|$R^{2}$|Spearman|
|-|-|-|-|-|
|pre-trained GIN + TopExpert|0.5112|0.6856|0.4931|0.6704|
|XGBoost|0.5406|0.7268|0.4087|0.6127 (p=6.2430e-153)|
|LightGBM|0.5414|0.7208|0.4185|0.6115 (p=3.4588e-152)|
|CatBoost|0.5279|0.7015|0.4492|0.6290 (p=1.5850e-163)|

![pre-trained GIN + TopExpert](../pic/training_plot_single_20250923_083933.png)  

---

## 7. 실험 설정 요약

| 항목 | 값 |
|------|------|
| 모델 | Pre-trained GIN + TopExpert |
| 데이터셋 | hERG, LD50_Zhu |
| GPU | NVIDIA RTX A5000 (24GB) |
| 배치 크기 | 512 |
| 에포크 | 500 (classification) </br> 10000 (Regression) |
| 전문가 수 | 7 |
| 학습률 | 1e-04 |
| 평가 지표 | AUC, F1 Score |

---

## 8. 하이퍼파라미터 튜닝 결과  

다음 조합들에 대하여 하이퍼파라미터 튜닝을 시도 (총 1152개 조합)  

    learning_rates=(1e-2 1e-3 1e-4)  
    gate_dims=(64 128)  
    num_experts=(3 5 7 10)  
    alphas=(0.01 0.1 1 5)  
    betas=(0.01 0.1 1 5)  
    emb_dims=(300)  

사실 위의 parameter만 하면 대략 3xx개의 조합인데, 원래 min_temp에 대해서도 hyperparamter search를 했으나, 정작 csv에 저장할 때 min_temp를 저장하지 않아버렸다. 그래서, min_temps=(0.01 0.1 1) 중 어떤 min_temps를 사용했는지는 모른다. 다만 job_id로 구분할 수 있어서, min_temps를 복원한 뒤, 테스트를 진행한다.  

또한 하이퍼파라미터 튜닝 시작시점부터 early stopping을 도입했다. 
- Classification: Validation AUC가 50 epoch 동안 0.001 이상 개선되지 않으면 중단  
- Regression: Validation MAE가 100 epoch 동안 0.001 이상 개선되지 않으면 중단  

### 1) classfication  

치팅을 하지 않기 위해, validation loss만 확인.  

![!\[alt text\](image.png)](img/results_hyperparam_classification.png)  

가장 좋은 결과를 보인 hyperparameter 조합은  
> --lr 1e-3 --gate_dim 64 --num_experts 5 --alpha 5 --beta 0.01 --emb_dim 300 --min_temp 0.01  
이었고, 평균적으로 가장 좋은 결과를 보인 parameter의 조합은  
> --lr 0.001 --gate_dim 128 --num_experts 10 --alpha 1.0 --beta 0.01 --emb_dim 300  
이었다. (min_temp 에 대한 정보는 알 수 없음.) 이 두 가지에 대한 실행 결과는 다음과 같다.  

#### Results  

|model|Val_AUC|Test AUC|Test F1|
|-|-|-|-|
|XGBoost|-|0.7531|0.7646|
|LightGBM|-|0.7818|0.7421|
|CatBoost|-|0.7853|0.7434|
|no tuning model|-|0.7969(± 0.0018)|0.8476|
|no tuning model (applying early stopping)|-|0.7782|0.8381|
|best val hyperparam|0.9189|0.7387|0.8037| 
|best exp. val hyperparam (min_temp=0.01)|0.9156|0.7742|0.8165|
|best exp. val hyperparam (min_temp=0.1)|0.9033|0.7904|0.8241|
|best exp. val hyperparam (min_temp=1)|0.9067|0.7860|0.8186|

튜닝하지 않은 모델이 가장 좋은 결과가 나왔다. 다만, 튜닝하지 않은 모델에 early stopping을 도입한 결과 기존 결과보다 좋지 않은 결과가 나왔으며, early stopping을 적용한 경우 아래 세팅이 가장 좋은 결과를 보인다.  
--lr 0.001 --gate_dim 128 --num_experts 10 --alpha 1.0 --beta 0.01 --emb_dim 300 --min_temp 0.1  

#### no tuning model (applying early stopping)  

![fig](../pic/training_plot_20250930_121114.png)  

#### best val hyperparam  

![fig](../pic/training_plot_20250930_102922.png)  

#### best exp. val (min_temp=0.01)

![fig](../pic/training_plot_20250930_103820.png)  

#### best exp. val (min_temp=0.1)

![fig](../pic/training_plot_20250930_104241.png)

#### best exp. val (min_temp=1)

![fig](../pic/training_plot_20250930_104630.png)

### 2) Regression  

시간이 없어, 모든 조합에 대한 hyperparameter search는 진행하지 못했다. 지금까지 수행한 hyperparameter tuning 결과는 다음과 같다.  

![alt text](img/results_hyperparam_regression.png)  

가장 좋은 validation MAE 결과를 보인 parameter는 다음과 같다.  
--lr 1e-2 --gate_dim 128 --num_experts 5 --alpha 0.01 --beta 5 --emb_dim 300 --min_temp 0.1  

parameter마다 평균적으로 가장 좋은 validation MAE를 보인 parameter를 조합하면 다음과 같다.  
-- lr 0.01 --gate_dim 128 --num_experts 10 --alpha 0.1 --beta 0.1 --emb_dim 300

이 parameter 조합에 대한 결과는 다음과 같다.  

#### Results

|model|Test MAE|Test RMSE|Test $R^{2}$|Test Spearman|
|-|-|-|-|-|
|XGBoost|0.5406|0.7268|0.4087|0.6127|
|LightGBM|0.5414|0.7208|0.4185|0.6115|
|CatBoost|0.5279|0.7015|0.4492|0.6290|
|no finetuning model|0.5112|0.6856|0.4931|0.6704|
|no finetuning model (early stopping)|0.5411|0.7079|0.4596|0.6362|
|best val hyperparam|0.5248|0.7204|0.4404|0.6624|
|best exp. val hyperparam (min_temp=0.01)|0.5501|0.7377|0.4133|0.6347|
|best exp. val hyperparam (min_temp=0.1)|0.5349|0.7288|0.4273|0.6460|
|best exp. val hyperparam (min_temp=1)|0.5112|0.6982|0.4743|0.6821|

이번에도 hyperparameter 튜닝한 모델이 튜닝하지 않은 모델보다 성능이 좋지 않다. (best exp. val hyperpamameter (min_temp 1)의 경우 MAE는 동률이지만, RMSE와 R^2 결과가 더 좋지 않음.) 다만, 똑같이 early stopping을 적용할 경우, 아래 세팅이 가장 좋은 결과를 보임을 알 수 있다.  
-- lr 0.01 --gate_dim 128 --num_experts 10 --alpha 0.1 --beta 0.1 --emb_dim 300 --min_temp 1  

#### no finetuning (applying early stopping)

![alt text](../pic/training_plot_20250930_120117.png)  
![alt text](../pic/regression_results_20250930_120116.png)  

#### best val hyperparam  

![alt text](../pic/training_plot_20250930_111006.png)  
![alt text](../pic/regression_results_20250930_111005.png)  

#### best exp. val hyperparam (min_temp=0.01)

![alt text](../pic/training_plot_20250930_112200.png)  
![alt text](../pic/regression_results_20250930_112159.png)  

#### best exp. val hyperparam (min_temp=0.1)  

![alt text](../pic/training_plot_20250930_114041.png)  
![alt text](../pic/regression_results_20250930_114040.png)  

#### best exp. val hyperparam (min_temp=1)  

![alt text](../pic/training_plot_20250930_114824.png)
![alt text](../pic/regression_results_20250930_114823.png)

## 9. Additional Questions  

1. 처음에 돌린 regression 결과에서 왜 $R^2$ 값이 음수가 나왔는가?  
    이 질문에 답하기 위해서, 먼저 $R^2$ 를 계산하는 방법을 알아야 한다.  
    $$
    R^2 = \frac{SSE}{SST} = 1 - \frac{SSR}{SST}
    $$
    이때,   
    $SST = \sum_{i=1}^{N} \left( y_i - \mathbb{E}\left[ y \right] \right)^{2} $ (target value에서 target value의 평균을 뺀 것을 제곱한 값의 총합)   
    $SSE = \sum_{i=1}^{n}\left( \hat{y}_i - \mathbb{E}(y) \right)^{2}$ (추정값에서 target value의 평균을 뺀 것을 제곱한 값의 총합)  
    $SSR = \sum_{i=1}^{n}\left( y_i - \hat{y}_i \right)^{2}$ (target value와 추정값을 뺀 것을 제곱한 값의 총합)  
    만약 $R^{2}$ 의 값이 음수가 된다면, $\frac{SSR}{SST}$ 의 값이 1보다 크다는 뜻이 되며, 이는 곧, 
    $$
    SSR > SST
    $$
    임을 의미한다. 이 식을 전개하여 살펴보면,  
    $$
    \sum_{i=1}^{n}\left( y_i - \hat{y}_i \right)^{2} > \sum_{i=1}^{N} \left( y_i - \mathbb{E}\left[ y \right] \right)^{2}
    $$
    이는 곧, 모델의 예측 값이 단순히 target value를 평균낸 것보다 좋지 않은 예측값을 냈다는 것으로 해석할 수 있다.  

    그렇다면 왜 첫 번째 regression 결과에서 $R^{2}$ 값이 음수가 나왔을까? 아마 이는, classification과 동일하게 label을 전처리했기 때문일 것이라고 생각한다. classification loss로 BCE loss를 사용하기 위해, target value를 -1, 1에서 0, 1로 변환하는 과정을 거치며, 이는 다음 식을 통해 이루어진다.  
    ```python
    processed_labels = (labels + 1) / 2
    ```
    이 과정이 regression label에도 그대로 적용되었다고 생각해보면, 모델은 실제 값보다 대략 0.5배가 된 데이터를 기반으로 학습될 것이다. 모델이 틀린 데이터셋을 바탕으로 정상적으로 잘 학습되었다는 가정 하에, 대략 0.5배의 예측값을 내놓게 될 것이고, 이는 당연히 real value와 큰 차이가 날 것이며, 이는 곧 real value의 평균값보다 좋지 않은 예측을 내놓았을 가능성을 시사한다(고 생각한다).

2. 왜 Classification loss를 regression 하면 안 되는가?  
    현재 Classification loss로 사용하고 있는 것은 **BCEwithLogitsLoss**이고, Regression loss로 사용 중인 것은 **MSELoss**다. 먼저, classification의 경우, target value $y \in \left\{-1, 1\right\}$ 를 0 또는 1로 변환한 뒤, 이 target value와 class에 대한 예측 확률을 cross entropy로 계산한다. 하지만 regression은 같은 변환을 수행하기에는 target value $y$가 연속적이다. 즉, target value를 classification처럼 $(y+1)/2$ 로 다시 labelling하면, 모델은 이 잘못된 target value를 기반으로 예측값을 학습할 것이고, 결국 학습 완료된 모델의 출력값 또한 자연스럽게 왜곡될 것이다.  
    따라서, 연속된 target value인 $y$를 처리하기 위해서는 MSELoss와 같은 방식을 사용해야 한다.  