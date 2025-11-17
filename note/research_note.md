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

![Criterion](pic/eval_criterion.png)  

이 기준에 맞는 모델을 구현해야 한다. 일단 ADMET 관련해서는 어떻게 잘 될 것 같은데, 문제는 다른 데이터셋이다. 타 데이터셋이 어떤 데이터셋인지도 잘 모르겠고, 해당 데이터셋에 대해 저 정도의 성능을 뽑을 자신도 없다.  
