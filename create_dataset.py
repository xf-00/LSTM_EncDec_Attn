import numpy as np
import pandas as pd
import torch


data_dir = 'D:/CTIT-Synology/我的文件/SynologyDrive/CTIT-项目/横向项目/大型城市综合体/od_data_processed/'


def train_test_split(data, ratio):
    train_size = int(ratio * len(data))
    index_train = np.arange(0, train_size)
    index_test = np.arange(train_size, len(data))

    train = data.loc[index_train]
    test = data.loc[index_test]

    return train, test


def window_data(x, input_window=5, output_window=1, stride=1):
    L = x.shape[0]
    n_features = x.shape[1]
    n_samples = (L - input_window - output_window) // stride + 1

    X = np.zeros([n_samples, input_window, n_features])
    Y = np.zeros([n_samples, output_window])

    for i in range(n_samples):
        X[i, :, :] = x.iloc[i * stride:i * stride + input_window, :]
        Y[i, :] = x.counts.iloc[i * stride + input_window:i * stride + input_window + output_window]
    return X, Y


def dataset(file_name, split_ratio):
    df_ts = pd.read_csv(data_dir + file_name, header=0, usecols=[1, 2])
    df_ts.date = pd.to_datetime(df_ts.date)
    df_ts['wd'] = df_ts.date.apply(lambda x: x.weekday())
    df_ts['hour'] = df_ts.date.dt.hour
    t_cols = ['counts', 'wd', 'hour']  # --change according to requirements
    train_data, test_data = train_test_split(df_ts[t_cols], ratio=split_ratio)
    x_train, y_train = window_data(train_data)
    x_test, y_test = window_data(test_data)
    return x_train, y_train, x_test, y_test


def numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest):

    X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
    Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

    X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
    Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)

    return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch