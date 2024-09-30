from src.data import CostomerDataset, CostomerDataModule
from src.utils import convert_category_into_integer
from src.model.mlp import Model
from src.training import CostomerModule
# from imblearn.over_sampling import SMOTE

import pandas as pd
import numpy as np
import random
import json
import nni
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import seaborn as sns


def main(configs):

    data = pd.read_csv('./data/train.csv')
    data = data.drop(columns=[
        'CustomerID',  # 구분용도임
        'NotNewCellphoneUser', # NewCell...이랑 중복
        'BlockedCalls', 
        'PrizmCode', # 인구통계 세분화 코드라 삭제
        'TruckOwner', # 자동차 오토바이 유무라서 삭제
        'RVOwner',
        'OwnsMotorcycle',
        'OwnsComputer', # 컴퓨터 유무 삭제
        'CurrentEquipmentDays', # 사용일수는 월별서비스와 중복
        'HandsetRefurbished', #
        'OffPeakCallsInOut',
        'OptOutMailings',
        'NonUSTravel',# 미국 여행여부 삭제
        'AdjustmentsToCreditRating', # 신용등급
        'ActiveSubs',
    ])
    data = data.dropna()

    # 나이 0 삭제
    data = data[data['AgeHH1']>18]
    data = data[data['AgeHH2']>18]

    # 인트타입으로 바꿔서 구분
    data['Churn'] = np.where(data['Churn']=="Yes", 1, 0)
    data['DroppedCalls'] = data['DroppedCalls'].astype(int)

    # 나이 구분
    data.AgeHH1 = np.where(
        data.AgeHH1 < 20,18,
        np.where(data.AgeHH1 < 30, 20, 
        np.where(data.AgeHH1 < 40, 30,
        np.where(data.AgeHH1 < 50, 40,
        np.where(data.AgeHH1 < 60, 50, 60)))))
    data.AgeHH2 = np.where(
        data.AgeHH2 < 20,18,
        np.where(data.AgeHH2 < 30, 20, 
        np.where(data.AgeHH2 < 40, 30,
        np.where(data.AgeHH2 < 50, 40,
        np.where(data.AgeHH2 < 60, 50, 60)))))

    # 1. 고객 소통 및 구매 행동 레이블 (Customer Communication and Purchase Behavior)
    data['CustomerPurchaseBehavior'] = np.where(
        (data['RespondsToMailOffers'] == 'Yes') & (data['BuysViaMailOrder'] == 'Yes'), 'Fully Engaged',
        np.where(
            (data['RespondsToMailOffers'] == 'Yes') | (data['BuysViaMailOrder'] == 'Yes'), 'Partially Engaged',
            'Not Engaged'
        )
    ) # 매일 오퍼에 응답하고 우편으로 구매 / 둘중 하나만 응답하거나 구매하는 고객 / 둘 다 하지 않는 고객


    # 2. 핸드셋 및 주거 환경 레이블 (Handset and Housing Environment)
    data['HandsetHousing'] = data['Homeownership'] + '-' + np.where(
        data['HandsetWebCapable'] == 'Yes', 'WebCapable', 'NonWebCapable'
    ) # 고객의 주거 소유 여부와 인터넷 지원여부 결합으로 나눔

    # 3. 서비스 이용 패턴 레이블 (Service Utilization Pattern)
    data['ServiceUtilization'] = np.where(
        (data['DroppedCalls'] > data['DroppedCalls'].mean()) | 
        (data['PeakCallsInOut'] > data['PeakCallsInOut'].mean()) |
        (data['MonthlyMinutes'] > data['MonthlyMinutes'].mean()), 'Heavy User', 
        'Light User'
    ) # 고객의 통화 시간이나 드랍된 전화 수가 평균보다 높은지 낮은지

    # 4. 충성도 및 서비스 이용 기간 (Loyalty and Service Duration)
    data['CustomerLoyalty'] = np.where(
        (data['UniqueSubs'] > 1) & (data['MonthsInService'] > data['MonthsInService'].mean()), 
        'High Loyalty', 'Low Loyalty'
    ) # 여러개를 구독하고 평균 이상의 서비스 이용을 하는사람과 그 아래

    # 5. 요금 부담 레이블 (Charge Burden)
    data['ChargeBurden'] = np.where(
        (data['TotalRecurringCharge'] > data['TotalRecurringCharge'].mean()) |
        (data['OverageMinutes'] > data['OverageMinutes'].mean()), 'High Charge', 'Low Charge'
    ) # 요금이 평균보다 높은지 낮은지

    # 6. 수익 변화 패턴 레이블 (Revenue and Minutes Change Behavior)
    data['RevenueMinutesChange'] = np.where(
        (data['PercChangeMinutes'] > 0) & (data['PercChangeRevenues'] > 0), 'Increasing Usage and Revenue',
        np.where(
            (data['PercChangeMinutes'] < 0) & (data['PercChangeRevenues'] < 0), 'Decreasing Usage and Revenue',
            'Mixed Behavior'
        ) # 사용량 및 수익이 증가하는 고객, 감소하는 고객, 두 값이 서로 반대인 경우
    )

    # 7. 로밍 및 추가 요금 레이블 (Roaming and Overage Behavior)
    data['RoamingOverage'] = np.where(
        (data['RoamingCalls'] > 0) & (data['OverageMinutes'] > 0), 'Roaming and Overage',
        np.where(
            (data['RoamingCalls'] > 0), 'Roaming Only',
            np.where(
                (data['OverageMinutes'] > 0), 'Overage Only', 'No Roaming or Overage'
            )
        ) # 로밍과 초과사용 모두 있는 고객 / 로밍만 있는 고객, 초과 사용만 있는 고객, 두 항목 모두 없는 고객
    )

    # 8. 수익 그룹 레이블 (Revenue Group)
    data['RevenueGroup'] = np.where(data['MonthlyRevenue'] > data['MonthlyRevenue'].mean(), 'High Revenue', 'Low Revenue')
    # 월 매출이 평균보다 높은 고객, 평균 이하인 고객
    # 9. 사용 시간 그룹 레이블 (Usage Time Group)
    data['UsageGroup'] = np.where(data['MonthlyMinutes'] > data['MonthlyMinutes'].mean(), 'High Usage', 'Low Usage')
    # 사용 시간이 평균보다 높은 고객/ 평균 이하인 고객

    data['Unansweremean'] = np.where(data['UnansweredCalls'] > data['UnansweredCalls'].mean(), 1, 0)  # 응답받지 못한 전화가 있는지 없는지
    data['CustomerCallcount'] = np.where(data['CustomerCareCalls'] > data['CustomerCareCalls'].mean(),1, 0) # 고객센터에 전화를 건 횟수가 평균 이상인지 아닌지
    data['RetnetionCallscount'] = np.where(data['RetentionCalls'] > 0, 1, 0) # 고객이탈방지 전화가 있었는지?
    data['CreditRatingdev'] = np.where(data['CreditRating'] == 6, 1,np.where(data['CreditRating'] == 7, 1, 0)) # 고객 신용 등급이 6이거나 7인지
    data['ReferralsMadeBySubscribercount'] = np.where(data['ReferralsMadeBySubscriber'] == 0,1,0) # 서비스 이용자가 다른 사람에게 추천을 한 적이 있는지 없는지
    data['UnansweredCallRate'] = data['UnansweredCalls'] / (data['ReceivedCalls'] + data['OutboundCalls'] + 1e-6) # 고객이 받지 않은 전화(미응답 전화)의 비율
    data['CustomerCareCallRate'] = data['CustomerCareCalls'] / (data['ReceivedCalls'] + data['OutboundCalls'] + 1e-6) # 고객이 전체 전화에서 고객 센터로 건 전화의 비율
    data['TotalRevenue_CustomerCareCalls'] = data['TotalRecurringCharge'] * data['CustomerCareCalls'] # 요금과 고객 센터 호출 빈도의 상관 관계
    data['MonthlyRevenue_CustomerCareCalls'] = data['MonthlyRevenue'] * data['CustomerCareCalls'] # 월 매출과 고객의 문제 해결 필요성이 연관 있는지
    data['UnansweredCalls_CustomerCareCalls'] = data['UnansweredCalls'] * data['CustomerCareCalls'] # 미응답 전화가 많을수록 고객 센터에 더 자주 전화를 걸 수 있는지
    data['TotalCalls'] = data['ReceivedCalls'] + data['OutboundCalls'] + 1e-6  # 전체 통화량
    data['CallForwardingToReceivedRate'] = data['CallForwardingCalls'] / data['TotalCalls'] # 전체 통화에서 착신 전환된 전화가 차지하는 비율
    data['CallWaitingToReceivedRate'] = data['CallWaitingCalls'] / data['TotalCalls'] # 전화 대기 중인 상태의 빈도

    # 비율 기반 피처 추가
    data['UnansweredCallRate'] = data['UnansweredCalls'] / (data['ReceivedCalls'] + data['OutboundCalls'] + 1e-6)
    data['CustomerCareCallRate'] = data['CustomerCareCalls'] / (data['ReceivedCalls'] + data['OutboundCalls'] + 1e-6)

    # 상호작용 피처 추가
    data['TotalRevenue_CustomerCareCalls'] = data['TotalRecurringCharge'] * data['CustomerCareCalls']
    data['MonthlyRevenue_CustomerCareCalls'] = data['MonthlyRevenue'] * data['CustomerCareCalls']
    data['UnansweredCalls_CustomerCareCalls'] = data['UnansweredCalls'] * data['CustomerCareCalls']

    # 과거 통화 이력 비율 피처
    data['TotalCalls'] = data['ReceivedCalls'] + data['OutboundCalls'] + 1e-6
    data['CallForwardingToReceivedRate'] = data['CallForwardingCalls'] / data['TotalCalls']
    data['CallWaitingToReceivedRate'] = data['CallWaitingCalls'] / data['TotalCalls']

    data, _ = convert_category_into_integer(data, (data.loc[:, data.columns]))
    data = data.astype(np.float32)

    # 학습, 검증, 테스트 데이터셋 분할
    train, temp = train_test_split(data, test_size=0.4, random_state=seed)
    valid, test = train_test_split(temp, test_size=0.5, random_state=seed)

    # # 학습 데이터에서 특징(X)과 레이블(y) 분리
    # X_train = train.drop(columns=['Churn'])  # 'Churn'은 타겟 레이블로 가정
    # y_train = train['Churn']

    # # SMOTE 적용
    # smote = SMOTE(random_state=seed)
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # # SMOTE 적용 후, 다시 데이터 프레임으로 결합
    # train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
    # train_resampled['Churn'] = y_train_resampled

    # # 검증 및 테스트 데이터에는 SMOTE를 적용하지 않음
    # X_valid = valid.drop(columns=['Churn'])
    # y_valid = valid['Churn']
    # X_test = test.drop(columns=['Churn'])
    # y_test = test['Churn']

    # 스케일링 (SMOTE 적용 후)
    # standard_scaler = StandardScaler()
    
    # train.loc[:, data.columns] = \
    #     standard_scaler.fit_transform(train.loc[:, data.columns])

    # valid.loc[:, data.columns] = \
    #     standard_scaler.transform(valid.loc[:, data.columns])
    
    # test.loc[:, data.columns] = \
    #     standard_scaler.transform(test.loc[:, data.columns])

    # Dataset 생성
    train_dataset = CostomerDataset(train)
    valid_dataset = CostomerDataset(valid)
    test_dataset = CostomerDataset(test)

    costomer_data_module = CostomerDataModule(batch_size=configs.get('batch_size'))
    costomer_data_module.prepare(train_dataset, valid_dataset, test_dataset)

    configs.update({'input_dim': len(data.columns)-1})
    model = Model(configs)

    costomer_module = CostomerModule(
        model=model,
        configs=configs,
    )

    del configs['output_dim'], configs['seed']
    exp_name = 'costomer'
    trainer_args = {
        'max_epochs': configs.get('epochs'),
        'callbacks': [
            EarlyStopping(monitor='loss/val_loss', mode='min', patience=5),
        ],
        'logger': TensorBoardLogger(
            'tensorboard',
            f'costomer/{exp_name}',
        ),
    }

    if configs.get('device') == 'gpu':
        trainer_args.update({'accelerator': configs.get('device')})

    trainer = Trainer(**trainer_args)

    trainer.fit(
        model=costomer_module,
        datamodule=costomer_data_module,
    )
    
    trainer.test(
        model=costomer_module,
        datamodule=costomer_data_module,
    )
    
    torch.save(model.state_dict(), './model/mlp.pth')
    
if __name__ == '__main__':
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    with open('./configs.json', 'r') as file:
        configs = json.load(file)
    configs.update({'device': device})

    if configs.get('nni'):
        nni_params = nni.get_next_parameter()
        configs.update({'batch_size': nni_params.get('batch_size')})
        configs.update({'hidden_dim': nni_params.get('hidden_dim')})
        configs.update({'learning_rate': nni_params.get('learning_rate')})
        configs.update({'dropout_ratio': nni_params.get('dropout_ratio')})

    seed = configs.get('seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == 'gpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    main(configs)