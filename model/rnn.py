import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# class RNN(nn.Module):
#     def __init__(self, **kwargs):
#         super(RNN, self).__init__()
#         self.input_size = in_size

class RNNNet(nn.Module):
    def __init__(self, *, input_size: int, hidden_size: int, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        # 循环神经网络层
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dtype=torch.float,
            batch_first=True,
            dropout=0.5
        )

        self.out = nn.Linear(in_features=hidden_size, out_features=1, dtype=torch.float)

    def forward(self, inputs: torch.Tensor, hidden_state: Optional[torch.Tensor]):

        _, hidden_state = self.rnn(inputs, hidden_state)
        outputs = self.out(hidden_state[-1])
        return outputs, hidden_state
