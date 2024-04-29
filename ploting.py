import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from model.rnn import RNNNet
from model.lstm import LSTMNet
import matplotlib.pyplot as plt
import pandas as pd


# read_data
plt.rcParams['font.family'] = 'FangSong'


def plot_loss_curves(model, results: dict[str, list[float]], window_size, lr, epoch, num_hiddens):
    """
    参数:
        results (dict): 包含值列表的字典，例如：
            {"train_loss": [...],
             "test_loss": [...],
    """

    loss = results['train_loss']
    test_loss = results['test_loss']

    # 确定有多少个 epoch
    epochs = range(len(results['train_loss']))

    # 设置绘图
    plt.figure(figsize=(15, 7))

    # 绘制损失曲线
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('损失', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(
        f'./od_pred/figures/feature-3_lstm_epoch{epoch}_numhidden{num_hiddens}_windowsize{window_size}_lr{lr}_train0.6.png',
        dpi=300)
    plt.show()

def plot_total_res():
    df_ts = pd.read_csv('D:/Projects/codes/cheng_tsp/od_pred/results/total_results.csv', header=None)
    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(test_origin_y.reshape(-1, 1), color='#899FB0', label='实际数据')
    plt.plot(df_ts[0], color='#B7DBE3', label='RNN')
    plt.plot(df_ts[1], color='#E1B6B5', label='LSTM')
    plt.plot(df_ts[2], color='#D882AD', label='ALSeq2Seq')
    plt.legend()
    plt.title('漕河泾开发区OD需求预测')
    plt.ylabel('流量')
    plt.xlabel('时间')
    # plt.grid()
    plt.savefig(f'./total_res.png', dpi=300)
    plt.show()
