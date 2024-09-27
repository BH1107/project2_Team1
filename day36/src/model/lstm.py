import torch.nn as nn


class Model(nn.Module):  # nn.Module을 상속받아 새로운 모델 클래스를 정의합니다.
    def __init__(self, configs):  # 생성자 메서드
        super().__init__()  # 부모 클래스의 생성자를 호출하여 초기화합니다.
        self.embedding_dim = configs.get('embedding_dim')
        self.vocab_size = configs.get('vocab_size')
        self.seq_len = configs.get('seq_len')
        self.dropout_ratio = configs.get('dropout_ratio')  # 활성화하지 않을 뉴런의 비율 지정
        self.lstm_hidden_dim1 = configs.get('lstm_hidden_dim1')
        self.lstm_hidden_dim2 = configs.get('lstm_hidden_dim2')
        self.linear_hidden_dim1 = configs.get('linear_hidden_dim1')
        self.linear_hidden_dim2 = configs.get('linear_hidden_dim2')
        self.output_dim = configs.get('output_dim')

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm1 = nn.LSTM(
            self.embedding_dim,
            self.lstm_hidden_dim1,
            batch_first=True,
            bidirectional=configs.get('bidirectional'),
        )  # 입력 차원에서 숨겨진 차원으로의 선형 변환을 정의합니다.
        self.dropout1 = nn.Dropout(p=self.dropout_ratio) # Dropout 정의
        self.lstm2 = nn.LSTM(
            self.lstm_hidden_dim1*2 if configs.get('bidirectional') else self.lstm_hidden_dim1,
            self.lstm_hidden_dim2,
            batch_first=True,
            bidirectional=configs.get('bidirectional'),
            )  # 입력 차원에서 숨겨진 차원으로의 선형 변환을 정의합니다.
        self.dropout2 = nn.Dropout(p=self.dropout_ratio) # Dropout 정의
        self.linear1 = nn.Linear(
            self.seq_len*self.lstm_hidden_dim2*2 if configs.get('bidirectional') else self.seq_len*self.lstm_hidden_dim2,   # flatten으로 인해 input 차원 변경
            self.linear_hidden_dim1,
        )  # 숨겨진 차원에서 출력 차원으로의 선형 변환을 정의합니다.
        self.relu1 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=self.dropout_ratio) # Dropout 정의
        self.linear2 = nn.Linear(
            self.linear_hidden_dim1,
            self.linear_hidden_dim2,
        )  # 숨겨진 차원에서 출력 차원으로의 선형 변환을 정의합니다.
        self.relu2 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=self.dropout_ratio) # Dropout 정의
        self.output = nn.Linear(
            self.linear_hidden_dim2,
            self.output_dim,
        )  # 숨겨진 차원에서 출력 차원으로의 선형 변환을 정의합니다.
    
    def forward(self, x):  # 순전파 메서드
        x = self.embedding(x)       # batch, seq_len, dim
        x, _ = self.lstm1(x)        # batch, seq_len, dim*2
        x = self.dropout1(x)    
        x, _ = self.lstm2(x)        # batch, seq_len, dim*2
        x = self.dropout2(x)    
        x = x.flatten(start_dim=1)  # batch, seq_len*dim*2
        x = self.linear1(x)         # batch, 512
        x = self.relu1(x)
        x = self.dropout3(x)
        x = self.linear2(x)         # batch, 128
        x = self.relu2(x)
        x = self.dropout4(x)
        x = self.output(x)          # batch, 2

        return x  # 최종 출력을 반환합니다.
