import torch.nn as nn


class Model(nn.Module):  # nn.Module을 상속받아 새로운 모델 클래스를 정의합니다.
    def __init__(self, configs):  # 생성자 메서드
        super().__init__()  # 부모 클래스의 생성자를 호출하여 초기화합니다.
        self.input_dim = configs.get('input_dim')  # 입력 차원 크기를 저장합니다.
        self.hidden_dim = configs.get('hidden_dim')  # 숨겨진 층의 차원 크기를 저장합니다.
        self.output_dim = configs.get('output_dim')  # 출력 차원 크기를 저장합니다.
        self.dropout_ratio = configs.get('dropout_ratio')  # 활성화하지 않을 뉴런의 비율 지정
        self.seq_len = configs.get('seq_len')

        self.rnn1 = nn.RNN(self.input_dim, self.hidden_dim)  # 입력 차원에서 숨겨진 차원으로의 선형 변환을 정의합니다.
        self.dropout1 = nn.Dropout(p=self.dropout_ratio) # Dropout 정의
        self.rnn2 = nn.RNN(self.hidden_dim, self.hidden_dim)  # 입력 차원에서 숨겨진 차원으로의 선형 변환을 정의합니다.
        self.dropout2 = nn.Dropout(p=self.dropout_ratio) # Dropout 정의
        self.output = nn.Linear(
            self.seq_len*self.hidden_dim,   # flatten으로 인해 input 차원 변경
            self.output_dim,
        )  # 숨겨진 차원에서 출력 차원으로의 선형 변환을 정의합니다.
    
    def forward(self, x):  # 순전파 메서드
        x, _ = self.rnn1(x)  # 입력 데이터에 대해 선형 변환을 적용합니다.
        x = self.dropout1(x) # dropout을 적용하여 일부 node 비활성화
        x, _ = self.rnn2(x)  # 입력 데이터에 대해 선형 변환을 적용합니다.
        x = self.dropout2(x) # dropout을 적용하여 일부 node 비활성화
        #x = x.flatten(start_dim=1)
        x = self.output(x)  # 두 번째 선형 변환을 적용하여 최종 출력을 계산합니다.

        return x  # 최종 출력을 반환합니다.
