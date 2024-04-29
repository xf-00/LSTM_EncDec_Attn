import torch
import torch.nn as nn
from typing import Optional


class LSTMNet(nn.Module):
    def __init__(self, *, input_size: int, hidden_size: int, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        # 循环神经网络层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dtype=torch.float,
            batch_first=True,
            dropout=0.5
        )

        # 线性输出层, 把 RNN 层的多个输出经过全连接层转换为 1 个特征输出
        self.out = nn.Linear(in_features=hidden_size, out_features=1, dtype=torch.float)

    def forward(self, inputs: torch.Tensor, hidden_state):

        _, hidden_state = self.lstm(inputs, hidden_state)
        outputs = self.out(hidden_state[0][-1])
        return outputs, hidden_state
