# project2_Team1

## 데이터 분석 및 ML

### base model
ML 모델이 Churn을 예측 하기 위해 유의미한 컬럼을 찾아야한다.
seaborn 패키지의 kdeplot 과 countplot을 사용하여 Churn에 따른 컬럼의 분포를 확인하였다.

찾아낸 유의미한 컬럼은 다음과 같다.

![TotlaRecurringCharge](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_base_image/TotlaRecurringCharge.png)

![MonthsInService](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_base_image/MonthsInService.png)

![CurrentEquipmentDays](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_base_image/CurrentEquipmentDays.png)


아래 5개의 컴럼에 대해서는 약간의 분포차이를 보인다.

![creaditRating](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_base_image/creaditRating.png)

![RetentionCalls](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_base_image/RetentionCalls.png)

![RespondsToMailOffers](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_base_image/RespondsToMailOffers.png)

![buysViaMailorder](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_base_image/buysViaMailorder.png)

![HandsetWebCapable](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_base_image/HandsetWebCapable.png)

lightgbm 모델로 위에서 찾은 5개의 컬럼
모델결과

![ML_base_result](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_base_image/ML_base.png)

Churn에 대한 예측중 1에대한 예측(이탈한 사람) Recall이 0.04로 매우 낮은 모습을 보인다.
이것은 실제 이탈한 사람에 대하여 모델이 이탈하였다고 예측한 비율이 4%로, 이탈한 사람에 대한 예측을 못하고 있는 것을 알 수 있다.


### New feature model

유효한 feature가 부족하다고 판단하여, 극복하기 위해 분석을 통해 찾은 컬럼을 조합하여 새로운 Feature를 만드는 전략을 세웠다.

```Python

# 1. 고객 충성도 레이블 (Customer Loyalty)
# 'MonthsInService'와 'RetentionCalls'을 기반으로 고객이 얼마나 오랜 기간 동안 서비스에 머물렀는지, 그리고 고객 유지 노력의 결과를 반영
data['CustomerLoyalty'] = np.where(
    (data['MonthsInService'] > data['MonthsInService'].mean()) &
    (data['RetentionCalls'] > 0), 
    'High Loyalty', 'Low Loyalty'
)

# 2. 서비스 사용 기간 레이블 (Equipment Usage Duration)
# 'CurrentEquipmentDays'를 사용해 서비스를 얼마나 사용하고 있는지 반영
data['EquipmentUsageDuration'] = np.where(
    data['CurrentEquipmentDays'] > data['CurrentEquipmentDays'].mean(), 
    'Long-Term Equipment User', 'Short-Term Equipment User'
)

# 3. 요금 부담 레이블 (Charge Burden)
# 'TotalRecurringCharge'를 기준으로 요금 부담이 높은지 낮은지를 분류
data['ChargeBurden'] = np.where(
    data['TotalRecurringCharge'] > data['TotalRecurringCharge'].mean(), 
    'High Charge', 'Low Charge'
)

# 4. 신용 등급 레이블 (Credit Rating Category)
# 'CreditRating'을 기준으로 신용 등급을 두 그룹으로 나눔
data['CreditCategory'] = np.where(
    data['CreditRating'] > data['CreditRating'].median(), 
    'High Credit', 'Low Credit'
)

# 5. 구매 및 메일 응답 행동 (Purchase and Mail Response Behavior)
# 'BuysViaMailOrder'와 'RespondsToMailOffers'를 결합하여 고객의 마케팅 참여도 파악
data['MarketingEngagement'] = np.where(
    (data['BuysViaMailOrder'] == 'Yes') & (data['RespondsToMailOffers'] == 'Yes'), 
    'Fully Engaged',
    np.where(
        (data['BuysViaMailOrder'] == 'Yes') | (data['RespondsToMailOffers'] == 'Yes'), 
        'Partially Engaged', 'Not Engaged'
    )
)

# 6. 핸드셋 웹 사용 가능 여부 (Handset Web Capability)
# 'HandsetWebCapable'을 사용하여 핸드셋이 웹 사용 가능한지 여부를 분류
data['HandsetWebCapability'] = np.where(
    data['HandsetWebCapable'] == 'Yes', 
    'WebCapable', 'NonWebCapable'
)

```
위와 같이 6개의 feature를 생성했다.

생성된 feature도 아래와 같이 분석을 진행 했다.


![CustomerLoyalty](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_NewFeature/CustomerLoyalty.png)

![EquipmentUsageDuration](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_NewFeature/EquipmentUsageDuration.png)

![ChargeBurden](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_NewFeature/ChargeBurden.png)

![CreditCategory](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_NewFeature/CreditCategory.png)

![MarketingEngagement](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_NewFeature/MarketingEngagement.png)

![HandsetWebCapability](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_NewFeature/HandsetWebCapability.png)

위와 동일하게 lightgbm 모델을 학습하여 예측하였고, 결관는 아래와 같다.

![NewFeature_Model_res](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN04-2nd-1Team/blob/readme/ML_NewFeature/Model_res.png)

