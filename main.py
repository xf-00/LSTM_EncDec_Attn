import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from model.rnn import RNNNet
from model.lstm import LSTMNet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchinfo import summary
from tqdm.auto import tqdm
from timeit import default_timer as timer
import os

# read_data
plt.rcParams['font.family'] = 'simhei'

if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model_test = 11

    train_scale = 0.6
    window_size = 24
    feature_size = 3
    num_hiddens = 50
    batch_size = 1
    num_layers = 1
    num_epochs = 300
    learning_rate = 0.001
    clipping_theta = 1e-2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 随机生成初始隐藏向量 h_0 c_0
    hidden_state = torch.randn((num_layers, batch_size, num_hiddens), dtype=torch.float).to(device)
    cell_state = torch.randn((num_layers, batch_size, num_hiddens), dtype=torch.float).to(device)
    nn.init.xavier_normal_(hidden_state)
    nn.init.xavier_normal_(cell_state)

    # 模型生成及选择
    rnn_net = RNNNet(input_size=feature_size, hidden_size=num_hiddens, num_layers=num_layers)
    lstm_net = LSTMNet(input_size=feature_size, hidden_size=num_hiddens, num_layers=num_layers)

    model = lstm_net

    if model == rnn_net:
        states = hidden_state
    else:
        states = (hidden_state, cell_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    # 归一化
    scaler = MinMaxScaler()
    train_X, train_Y, test_X, test_Y, test_origin_y = dataset(t_col=0, window_size=window_size, train_scale=train_scale,
                                                              scaler=scaler)

    start_time = timer()
    # 训练 model
    model_results, predictions = train(model=model, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y,
                                       optimizer=optimizer, loss_func=loss_func, batch_size=batch_size,
                                       hidden_state=states, clipping_theta=clipping_theta,
                                       epochs=num_epochs, device=device, lr=learning_rate, num_hiddens=num_hiddens)
    # predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    np.savetxt(f'./od_pred/results/lstm_test_prediction_{model_test}.csv', predictions)
    # 停止计时器并打印所用时间
    end_time = timer()
    print(f"总训练时间：{end_time - start_time:.3f} 秒")
    plot_loss_curves(model=model, results=model_results, epoch=num_epochs, window_size=window_size, lr=learning_rate,
                     num_hiddens=num_hiddens)
    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(test_origin_y.reshape(-1, 1), color='#82B0D2', label='实际数据')
    plt.plot(predictions, color='#FA7F6F', label='LSTM预测')
    plt.legend()
    plt.ylabel('流量')
    plt.xlabel('时间')
    plt.savefig(f'./od_pred/results/lstm_test_prediction_{model_test}.png', dpi=300)
    plt.show()

    print(
        f"test:{model_test} \ntrain_sacle:{train_scale},window_size:{window_size},feature_size:{feature_size},num_hiddens:{num_hiddens},batch_size:{batch_size},"
        f"num_layers:{num_layers},num_epochs:{num_epochs},learning_rate:{learning_rate}")
    print('————————————————————————————————————————————————————————————————————————————')
    print(f"train_loss:{model_results['train_loss'][-1]}, test_loss:{model_results['test_loss'][-1]}")