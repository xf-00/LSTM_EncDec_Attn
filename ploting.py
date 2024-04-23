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


def dataset(t_col: int, window_size, train_scale, scaler):
    data_dir = 'D:/CTIT-Synology/我的文件/SynologyDrive/CTIT-项目/横向项目/大型城市综合体/od_data_processed/'
    filenames = os.listdir(data_dir)
    df_ts = pd.read_csv(data_dir + filenames[0], header=0, usecols=[1, 2])

    df_ts.date = pd.to_datetime(df_ts.date)
    df_ts['dt_sin'] = np.sin(12 * df_ts.date.dt.hour / np.pi)
    df_ts['dt_cos'] = np.cos(12 * df_ts.date.dt.hour / np.pi)
    targetcolumns = ['counts', 'dt_sin', 'dt_cos']

    # 创建时间窗口数据
    def create_time_series_data(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data.counts[i + window_size])
        return np.array(X), np.array(y)

    train_size = int(len(df_ts) * train_scale)
    train, test = df_ts.iloc[:train_size].reset_index(drop=True), df_ts.iloc[train_size:].reset_index(drop=True)
    _, test_origin_y = create_time_series_data(test[targetcolumns], window_size)
    # norm
    # train['counts'] = scaler.fit_transform(train[targetcolumns].values.reshape(-1, 1))
    # test['counts'] = scaler.fit_transform(test[targetcolumns].values.reshape(-1, 1))

    train_X, train_Y = create_time_series_data(train[targetcolumns], window_size)  # 注意修改 是否标准化有区别
    test_X, test_Y = create_time_series_data(test[targetcolumns], window_size)
    # print(train_X.shape)
    # train_X = train_X.reshape(train_X.shape[0], window_size, train_X.shape[-1])
    # test_X = test_X.reshape(test_X.shape[0], window_size, test_X.shape[-1])

    return train_X, train_Y, test_X, test_Y, test_origin_y


def train_loop(model, x, y, optimizer, loss_func, batch_size, hidden_state, clipping_theta, device):
    model.train()
    train_loss = 0
    train_X, train_Y = x, y
    iterations = len(train_Y) // batch_size
    # print(iterations) #1763
    for itr in range(iterations):
        train_x_data = torch.tensor(train_X[itr * batch_size:(itr + 1) * batch_size, :, :], dtype=torch.float).reshape(
            batch_size, train_X.shape[1], train_X.shape[-1]).to(device)
        train_y_data = torch.tensor(train_Y[itr * batch_size:(itr + 1) * batch_size], dtype=torch.float).to(device)
        if hidden_state is not None:
            # 使用detach函数从计算图分离隐藏状态, 这是为了
            # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
            if isinstance(hidden_state, tuple):  # LSTM, state:(h, c)
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
            else:
                hidden_state = hidden_state.detach()

        predicts_y, hidden_state = model(train_x_data, hidden_state)

        loss = loss_func(predicts_y, train_y_data)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        grad_clipping(model.parameters(), clipping_theta, device)
    f_state = hidden_state
    train_loss = train_loss / iterations
    return train_loss, f_state


def test_loop(model, x, y, loss_func, batch_size, device, hidden_state):
    model.eval()

    test_loss = 0
    test_X, test_Y = x, y
    prediction_y = []

    iterations = len(test_Y) // batch_size
    for itr in range(iterations):
        test_x_data = torch.tensor(test_X[itr * batch_size:(itr + 1) * batch_size, :, :], dtype=torch.float).reshape(
            batch_size, test_X.shape[1], test_X.shape[-1]).to(device)
        test_y_data = torch.tensor(test_Y[itr * batch_size:(itr + 1) * batch_size], dtype=torch.float).to(device)
        predicts_y, _ = model(test_x_data, hidden_state)
        loss = loss_func(predicts_y, test_y_data)
        test_loss += loss.item()
        prediction_y.append(predicts_y.data.reshape(1).tolist())
    # print(np.array(prediction_y).shape)
    test_loss = test_loss / iterations
    return test_loss, np.array(prediction_y)


# 梯度裁剪
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


def train(model, train_X, train_Y, test_X, test_Y, optimizer, loss_func, batch_size, hidden_state, epochs, device,
          clipping_theta, lr, num_hiddens):
    model.to(device)
    results = {'train_loss': [], 'test_loss': []}
    for epoch in tqdm(range(epochs)):
        train_loss, f_state = train_loop(model=model, x=train_X, y=train_Y, optimizer=optimizer, loss_func=loss_func,
                                         batch_size=batch_size, hidden_state=hidden_state,
                                         clipping_theta=clipping_theta, device=device)
        test_loss, pred_y = test_loop(model=model, x=test_X, y=test_Y, loss_func=loss_func, batch_size=batch_size,
                                            device=device, hidden_state=f_state)

        print(f'Epoch:{epoch + 1}|', f'train_loss:{train_loss:.4f}', f'test_loss:{test_loss:.4f}')
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

    # torch.save(model.state_dict(),
    #            f'./od_pred/model_state/lstm_epoch{epochs}_numhidden{num_hiddens}_windowsize{train_X.shape[1]}_lr{lr}_train0.8.pth')

    return results, pred_y


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


if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model_test = 11

    train_scale = 0.8
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

    # 归一化
    scaler = MinMaxScaler()
    train_X, train_Y, test_X, test_Y, test_origin_y = dataset(t_col=0, window_size=window_size, train_scale=train_scale,
                                                              scaler=scaler)

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

