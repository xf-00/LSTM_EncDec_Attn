import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import create_dataset
import plotting
from model.rnn import RNNNet
from model.lstm import LSTMNet
from model.enc_dec import ALSeq2Seq
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import matplotlib
import os


plt.rcParams['font.family'] = 'simhei'
matplotlib.rcParams.update({'font.size': 17})

data_dir = 'D:/CTIT-Synology/我的文件/SynologyDrive/CTIT-项目/横向项目/大型城市综合体/od_data_processed/'
filenames = os.listdir(data_dir)
print(filenames)
iw = 5
ow = 1
x_train, y_train, x_test, y_test = create_dataset.dataset(filenames[0], split_ratio=0.8)

# plot example of windowed data
plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, len(x_train)), x_train[:, :, 0], 'k', linewidth=2.2, label='Input')


plt.xlabel(r'$t$')
plt.ylabel(r'$y$')
plt.title('Example of Windowed Training Data')
plt.legend(bbox_to_anchor=(1.3, 1))
plt.tight_layout()
# plt.savefig('plots/windowed_data.png')
plt.show()
# LSTM encoder-decoder

# convert windowed data from np.array to PyTorch tensor
X_train, Y_train, X_test, Y_test = generate_dataset.numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest)

# specify model parameters and train
model = ALSeq2Seq(input_size=X_train.shape[2], hidden_size=15)
loss = model.train_model(X_train, Y_train, n_epochs=50, target_len=ow, batch_size=5,
                         training_prediction='mixed_teacher_forcing', teacher_forcing_ratio=0.6, learning_rate=0.01,
                         dynamic_tf=False)

# plot predictions on train/test data
plotting.plot_train_test_results(model, Xtrain, Ytrain, Xtest, Ytest)
plt.close('all')




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