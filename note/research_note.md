# Research Note  

```text
ㅇㅇ
```

## 2025.11.04. ~ 2025.11.05.  

### 코드 구조 개편 및 코드 재작성  

#### 동기  

- 기존 코드의 모듈화 부족  
- 필요 없는 코드 다수 존재하여 코드의 복잡도 상승  

#### 개편안  

다음과 같은 action이 가능하도록 해야 함.  

##### 1. 데이터셋  
workspace/data/train/ 에 있는 csv 파일을 기반으로 학습해야 함. 각 데이터셋에 대한 학습은 Supervised Learning으로, 다음을 학습해야 함.  

|데이터셋 이름|task|target|
|-|-|-|
|hk2|regression|pIC50|
|hepg2|regression|pAC50|
|dili2|binary classification|label|
|dili3|multi-class classification|vDILI-Concern|  

즉, regression, multi-class classification을 지원해야 함.  

##### 2. metric  

각 task에 따른 Primary metric은 다음이 되어야 함.  
classification: AUROC  
regression: MAE  

다른 metric 또한 추가해도 좋으나, Secondary metric으로 사용되어야 함.  
학습 도중 사용할 loss는 합리적으로 알아서 선택할 것.  

##### 3. hyperparam tuning  

simple gpu scheduler를 사용한 hyperparameter tuning이 가능하도록 해야 함. hyperparameter tuning 과정은 csv에 저장되어야 하며, hyperparameter tuning이 종료된 후 결과 csv는 primary metric 기준으로 가장 결과가 좋은 순서로 sort되어야 함.  

##### 4. Stratified K-fold cross validation 및 LOOCV  

Stratified K-fold cross validation 및 LOOCV를 구현하되, 위의 hyperparameter tuning과 양립 가능하도록 해야 함.  

##### 5. Task  

TopExpert 기반으로 multi-class classification과 regression이 정상적으로 수행되도록 코드를 수정 및 추가해야 함.  

## 2025.11.08 -  

### 새 데이터셋에 대한 학습  

11월 7일 받은 데이터셋에 대한 통합 예측 모델을 만들어야 한다.  

![example](<../references/pic/Inline-image-2025-11-07 09.59.47.830.png>)  

위의 사진과 같이 총 33개의 항목에 대해 예측하는 모델을 만들면 된다. 먼저 각 항목에 대한 학습 진행한다. 이때, train:valid:test로 split하여 성능을 평가하고, 가장 좋은 성능을 보이는 모델을 선택하여 저장해야 한다. 그 다음으로 저장된 33개의 모델을 기반으로 하나의 SMILES 문자열이 들어왔을 때, 33개의 모델에 각각 input으로 넣고, 33개의 output을 통합하여 출력하면 된다.  
현재 코드 구조는 split 하여 학습하고 평가하는 것은 가능하지만, 예측값을 output으로 내놓는 것은 불가능한 것으로 알고 있다. 이것이 가능하도록 코드를 수정해야 할 듯하다. -> 수정 완료  

## 2025.11.10 -  

### Random Search  

하나의 데이터셋마다 1900+ 개의 조합을 사용하는 Grid Search를 했는데 시간이 너무 오래 걸린다. 하나의 데이터셋에 대한 grid search가 완료되는 데 40시간이 넘게 소요된다. 따라서 Random Search로 전략을 바꿀 계획이다.  

## 2025.11.13. -  

### Searching  

일부 데이터셋 (hERG_Central_inhib, hERG_Central_10uM, hERG_Central_1uM)이 training data를 306,893개를 가지는데, 이를 hyperparam search하는 것이 매우 오래 걸린다. 100개의 조합만 시도하고 있음에도 불구하고 하나의 데이터셋이 5일 가까이 걸린다는 듯하다. 이렇게 큰 데이터셋은 일단 기본 하이퍼파라미터 조합으로 모델을 만들 계획이다.  
하이퍼파라미터 서칭이 끝나면 각 데이터셋마다 최적의 하이퍼파라미터를 사용하여 최종 모델을 만들어야 한다. 내일은 바빠서.. 강화학습 팀플 끝나자마자 와서 모델 학습을 시작할 듯하다.  


## 2025.11.14. -  

![alt text](pic/Y1_Objective.png)  

1차년도 목표는 다음과 같다.  
- 딥러닝 기반 분자 군집화 알고리즘 기술이전 및 독성 예측 알고리즘 고도화  
    - 분자 구조 데이터 확보 및 전처리  
    - 딥러닝 기반 분자 임베딩 모델 설계  
    - 클러스터링 알고리즘 개발  
    - ADMET 예측 알고리즘 고도화  
- 군집화 알고리즘 기술이전 및 예측 알고리즘 고도화  

이에 대해 다음을 작성해야 한다.  

1. 세부 연구 수행 내용  
-> 독성 및 약효 예측을 위한 AI 알고리즘 개발  

대충 TopExpert의 알고리즘을 설명하면 된다. 이때, Figure를 첨부하도록 하자.  


2. 연구개발과제의 수행 결과 및 목표 달성 정도  
-> 정성적 연구개발성과, 정량적 연구개발성과  

정성적 연구개발성과에는 1차년도 목표를 달성한 정도를 적는다. 데이터 전처리나 임베딩 설계, 알고리즘 개발 등은 이미 다 했으므로 해당 내용을 정리하여 서술하도록 하자. 추가로 다음을 서술해야 한다.  
- atom, edge feature의 선정    
- hyperparameter tuning 과정  

정량적 연구개발성과에는 뭘 적어야 할지 모르겠다. 일단 박사님과 논의를 해보면 좋을 듯하다.  

## 2025.11.17. - 2025.11.18  

![Criterion](pic/eval_criterion.png)  

이 기준이 있었다는 사실을 방금 떠올렸다... 청년 치매인가..?  
아무튼, 이 기준에 맞는 모델을 구현해야 한다. 일단 ADMET 관련해서는 어떻게 잘 될 것 같은데, 문제는 다른 데이터셋이다. 타 데이터셋이 어떤 데이터셋인지도 잘 모르겠고, 해당 데이터셋에 대해 저 정도의 성능을 뽑을 자신도 없다.  

### 성능 개선 필요  

일단 일부 데이터셋에 대해 전혀 좋지 않은 결과가 나오고 있다.  
Classification의 경우 다음 데이터셋이 기준치인 AUROC 80을 넘기지 못하고 있다.  
- Carcinogens_Lagunin
- PAMPA_NCATS  
- CYP2C9_Substrate_CarbonMangels  
- hERG  
- Skin_Reaction  
- CYP3A4_Substrate_CarbonMangels  
- Bioavailability_Ma  
- ClinTox  

모델의 출력을 확인해보지 않아서 (확인하는 방법을 모름..) 무엇이 문제인지 확신할 수는 없지만, 일단 개인적으로 생각하는 원인은 다음과 같다.  

#### 1. ClinTox  
데이터셋 불균형 이슈  
Y Distribution:
  - Class 0: 1366 (92.42%)
  - Class 1: 112 (7.58%)

#### 2. Bioavailability_Ma  
데이터셋 불균형 이슈  
Y Distribution:
  - Class 1: 492 (76.88%)
  - Class 0: 148 (23.12%)

#### 3. CYP3A4_Substrate_CarbonMangels  
적어도 데이터셋 불균형의 문제는 아닌 것으로 보임. 다른 이유는 모르겠음.  
Y Distribution:
  - Class 1: 355 (52.99%)
  - Class 0: 315 (47.01%)  

|class|train|valid|test|overall|
|-|-|-|-|-|
|0|231 (49.36%)|28 (41.79%)|35 (42.68%)|56 (41.48%)|
|1|237 (50.64%)|39 (58.21%)|47 (57.32%)|79 (58.52%)|

참고로 총 25개의 hyperparam exp가 돌아간 지금 확인할 때, valid나 test나 둘 다 처참한 AUROC을 보이고 있다. test_accuracy가 0.6 정도에서 형성되고 있는데, 이는 사실 그냥 찍는 것과 크게 다르지 않은 성능이다. (실제로 test set에 대해 1을 찍으면 58.52의 accuracy) 참고로 valid_accuracy는 더 낮다. 0.5 후반에서 형성되고 있다.  

**Early Stop이 False인 실험이 많은 것으로 보아, training step을 늘려 어느 정도 해결할 수 있을 것으로 보인다.**  
-> 그렇게 해도 해결 못 할 가능성도 높다.  


#### 4. Skin_Reaction  
위의 데이터셋처럼 심하진 않지만 데이터셋이 불균형하긴 함.  
Y Distribution:
  - Class 1: 274 (67.82%)
  - Class 0: 130 (32.18%)

추가로, Train, Valid, Test set에서의 데이터셋 분포는 다음과 같았다.  
|class|train|valid|test|overall|
|-|-|-|-|-|
|0|77 (27.3%)|18 (45%)|35 (42.68%)|130 (32.18%)|
|1|205 (72.7%)|22 (55%)|47 (57.32%)|274 (67.82%)|

Train set과 다른 분포를 가져서 test set에서의 성능이 낮았을 수 있다. **valid set에 대한 성능도 관찰할 필요가 있다.**  

#### 5. hERG  
내 기억이 맞다면, hERG는 이전에 이미 tuning한 적 있는 데이터셋이다. 다만, hERG가 종류가 많아서 이 데이터셋이 맞는지는 잘 모르겠다.  
일단 데이터셋 불균형 문제로부터 어느 정도 자유로운 데이터셋이다. 
Y Distribution:
  - Class 1.0: 451 (68.85%)
  - Class 0.0: 204 (31.15%)  

이 데이터셋의 Train:Valid:Test에 대한 data dist는 다음과 같다.  
|class|train|valid|test|overall|
|-|-|-|-|-|
|0|144 (31.44%)|25 (38.46%)|35 (42.68%)|35 (26.52%)|
|1|314 (68.56%)|40 (61.54%)|47 (57.32%)|97 (73.48%)|

Train과 test의 데이터셋 분포 정도가 꽤 다름을 알 수 있다. 이것이 문제가 될 수도..?  

#### 6. CYP2C9_Substrate_CarbonMangels  

데이터 불균형 문제가 또 있다.  
Y Distribution:
  - Class 0: 528 (78.92%)
  - Class 1: 141 (21.08%)

#### 7. PAMPA_NCATS  

Y Distribution:
  - Class 1: 1739 (85.50%)
  - Class 0: 295 (14.50%)


#### 8. Carcinogens_Lagunin  

Y Distribution:
  - Class 0: 220 (78.57%)
  - Class 1: 60 (21.43%)  


</br></br>

Regression의 경우 다음 데이터셋에서 좋지 않은 성능이 관찰된다. (괄호 안 숫자는 MAE)  
- Clearance_Hepatocyte_AZ (30.97)  
- Clearance_Microsome_AZ (22.83)  
- hERG_Central_10uM (12.49)  
- PPBR_AZ (8.89)  
- Half_Life_Obach (7.99)  
- hERG_Central_1uM (6.56)  


|데이터셋|Mean|Std|Min|Max|Num|MAE|
|-|-|-|-|-|-|-|
|Clearance_Hepatocyte_AZ|42.9004|49.8473|3.0000|150.0000|1213|30.9724|  
|Clearance_Microsome_AZ||||||22.8278|
|hERG_Central_10uM||||||12.4925|
|PPBR_AZ||||||8.8946|
|Half_Life_Obach||||||7.9939|
|hERG_Central_1uM||||||6.5598|
|VDss_Lombardo||||||1.9333|
|HydrationFreeEnergy_FreeSolv||||||1.0949|
|Solubility_AqSolDB||||||0.9551|
|LD50_Zhu||||||0.5712|
|Lipophilicity_AstraZeneca||||||0.4903|
|Caco2_Wang||||||0.3665|

-> 20251118_report.md에 정리  

##### 해결 방안  

내 짧은 지식으로는, loss function에 어느 정도 가중치를 두는 편이 좋을 듯하다. focal loss를 사용하는 것이 최선이 아닐까?  

#### 내일 할 것  

- 지금 loss를 바꾸기에는 늦은 것 같고(hyperparam tuning을 할 시간이 없다.), 데이터셋에 대한 분석이나 더 하자.  
    - train:valid:test, 전체에 대한 dist를 table로 정리  
    - 튜닝 완료된 후 해당 파라미터로 모델 학습 및 test set에 대한 metric 확인  
    - 모델을 종합하여 input으로 SMILES 문자열이 들어올 때, 33가지 항목에 대해 예측하는 통합 프로그램 작성  
    - 이전에 받은 prediction dataset의 SMILES에 대해 해당 프로그램 실행 후 결과 분석 (임상 1상 예측 부분)  
- CYP3A4_Substrate_CarbonMangels 관련 이슈 확인 (상술한 이슈)   


#### 현재 상황 (11.18.)  

성능이 안 나온다. random split으로 바꿨는데도 안 되는 데이터셋은 안 된다. 오늘까지 보고서 초안 작성인데, 망한 듯..?  
CYP3A4_Substrate_CarbonMangels의 경우 early stopping이 false인 실험도 분명 있었지만, true인 실험도 많았다. 굳이 따지자면 True가 더 많다.  





## 잡설  

- 데이터셋 불균형을 해결하는 방법 중 하나로 SMOTE가 있다고 하는데, 이를 화학 분자 쪽으로 어떻게 끌어들일 수는 없을까? 물론 분야가 분야인 만큼 존재하지 않는 데이터를 생성하고 학습하는 매우 위험도가 높긴 하지만.. 일단 잘 된다면 혁명일텐데.  
- mlp 부분은 현재 설정이 최선인 것일까?  
- 사실 GNN으로는 분자의 embedding을 생성하고, 이 embedding을 바탕으로 2-layer mlp인 expert에 할당하여 output을 출력하는 구조라고 한다면, 굳이 GNN을 학습시키지 않아도 mlp만 학습시켜서 성능을 올릴 수 있는 건 아닐까..? 이건 너무 무식한 말인가?  
  - 무식한 것 같긴 한데 그냥 시도해보고 싶긴 함. 아님 말고 식으로 실험 정도는 할 수 있잖아. 만약 아니면 그냥 내가 무식한 것이고. 뭐 무식한 게 죄는 아니니까.  
- 