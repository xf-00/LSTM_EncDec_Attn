# Author: Laura Kulowski

import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_train_test_results(lstm_model, Xtrain, Ytrain, Xtest, Ytest, num_rows = 4):
  '''
  plot examples of the lstm encoder-decoder evaluated on the training/test data
  
  : param lstm_model:     trained lstm encoder-decoder
  : param Xtrain:         np.array of windowed training input data
  : param Ytrain:         np.array of windowed training target data
  : param Xtest:          np.array of windowed test input data
  : param Ytest:          np.array of windowed test target data 
  : param num_rows:       number of training/test examples to plot
  : return:               num_rows x 2 plots; first column is training data predictions,
  :                       second column is test data predictions
  '''

  # input window size
  iw = Xtrain.shape[0]
  ow = Ytest.shape[0]

  # figure setup 
  num_cols = 2
  num_plots = num_rows * num_cols

  fig, ax = plt.subplots(num_rows, num_cols, figsize = (13, 15))
  
  # plot training/test predictions
  for i in range(num_rows):
      # train set
      X_train_plt = Xtrain[:, i, :]
      Y_train_pred = lstm_model.predict(torch.from_numpy(X_train_plt).type(torch.Tensor), target_len = ow)

      ax[i, 0].plot(np.arange(0, iw), Xtrain[:, i, 0], 'k', linewidth = 2, label = 'Input')
      ax[i, 0].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtrain[-1, i, 0]], Ytrain[:, i, 0]]),
                     color = (0.2, 0.42, 0.72), linewidth = 2, label = 'Target')
      ax[i, 0].plot(np.arange(iw - 1, iw + ow),  np.concatenate([[Xtrain[-1, i, 0]], Y_train_pred[:, 0]]),
                     color = (0.76, 0.01, 0.01), linewidth = 2, label = 'Prediction')
      ax[i, 0].set_xlim([0, iw + ow - 1])
      ax[i, 0].set_xlabel('$t$')
      ax[i, 0].set_ylabel('$y$')

      # test set
      X_test_plt = Xtest[:, i, :]
      Y_test_pred = lstm_model.predict(torch.from_numpy(X_test_plt).type(torch.Tensor), target_len = ow)
      ax[i, 1].plot(np.arange(0, iw), Xtest[:, i, 0], 'k', linewidth = 2, label = 'Input')
      ax[i, 1].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtest[-1, i, 0]], Ytest[:, i, 0]]),
                     color = (0.2, 0.42, 0.72), linewidth = 2, label = 'Target')
      ax[i, 1].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtest[-1, i, 0]], Y_test_pred[:, 0]]),
                     color = (0.76, 0.01, 0.01), linewidth = 2, label = 'Prediction')
      ax[i, 1].set_xlim([0, iw + ow - 1])
      ax[i, 1].set_xlabel('$t$')
      ax[i, 1].set_ylabel('$y$')

      if i == 0:
        ax[i, 0].set_title('Train')
        
        ax[i, 1].legend(bbox_to_anchor=(1, 1))
        ax[i, 1].set_title('Test')

  plt.suptitle('LSTM Encoder-Decoder Predictions', x = 0.445, y = 1.)
  plt.tight_layout()
  plt.subplots_adjust(top = 0.95)
  plt.savefig('plots/predictions.png')
  plt.close() 
      
  return 



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
