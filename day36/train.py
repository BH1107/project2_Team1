from src.data import CostomerDataset, CostomerDataModule
from src.utils import convert_category_into_integer
from src.model.mlp import Model
from src.training import CostomerModule
from imblearn.over_sampling import SMOTE

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
    costomer = pd.read_csv('./data/train.csv')
    costomer = costomer.dropna()

    costomer, _ = convert_category_into_integer(costomer, ('Churn', 'ServiceArea', 'ChildrenInHH','HandsetRefurbished','HandsetWebCapable','TruckOwner','RVOwner','Homeownership','BuysViaMailOrder','RespondsToMailOffers','OptOutMailings','NonUSTravel','OwnsComputer','HasCreditCard','NewCellphoneUser','NotNewCellphoneUser','OwnsMotorcycle','HandsetPrice','MadeCallToRetentionTeam','CreditRating','PrizmCode','Occupation','MaritalStatus'))
    costomer = costomer.astype(np.float32)

    # 학습, 검증, 테스트 데이터셋 분할
    train, temp = train_test_split(costomer, test_size=0.4, random_state=seed)
    valid, test = train_test_split(temp, test_size=0.5, random_state=seed)

    # 학습 데이터에서 특징(X)과 레이블(y) 분리
    X_train = train.drop(columns=['Churn'])  # 'Churn'은 타겟 레이블로 가정
    y_train = train['Churn']

    # SMOTE 적용
    smote = SMOTE(random_state=seed)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # SMOTE 적용 후, 다시 데이터 프레임으로 결합
    train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
    train_resampled['Churn'] = y_train_resampled

    # 검증 및 테스트 데이터에는 SMOTE를 적용하지 않음
    X_valid = valid.drop(columns=['Churn'])
    y_valid = valid['Churn']
    X_test = test.drop(columns=['Churn'])
    y_test = test['Churn']

    # 스케일링 (SMOTE 적용 후)
    standard_scaler = StandardScaler()
    
    other_columns = ['MonthlyRevenue','MonthlyMinutes','TotalRecurringCharge','DirectorAssistedCalls','OverageMinutes','RoamingCalls','PercChangeMinutes','PercChangeRevenues','DroppedCalls','BlockedCalls','UnansweredCalls','CustomerCareCalls','ThreewayCalls','ReceivedCalls','OutboundCalls','InboundCalls','PeakCallsInOut','OffPeakCallsInOut','DroppedBlockedCalls','CallForwardingCalls','CallWaitingCalls','MonthsInService','UniqueSubs','ActiveSubs','Handsets','HandsetModels','CurrentEquipmentDays','AgeHH1','AgeHH2','RetentionCalls','RetentionOffersAccepted','ReferralsMadeBySubscriber','IncomeGroup','AdjustmentsToCreditRating']

    train_resampled.loc[:, other_columns] = \
        standard_scaler.fit_transform(train_resampled.loc[:, other_columns])

    valid.loc[:, other_columns] = \
        standard_scaler.transform(valid.loc[:, other_columns])

    test.loc[:, other_columns] = \
        standard_scaler.transform(test.loc[:, other_columns])

    # Dataset 생성
    train_dataset = CostomerDataset(train_resampled)
    valid_dataset = CostomerDataset(valid)
    test_dataset = CostomerDataset(test)

    costomer_data_module = CostomerDataModule(batch_size=configs.get('batch_size'))
    costomer_data_module.prepare(train_dataset, valid_dataset, test_dataset)

    configs.update({'input_dim': len(costomer.columns)-1})
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