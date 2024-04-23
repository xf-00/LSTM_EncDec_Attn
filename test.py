import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

data_dir = 'D:/CTIT-Synology/我的文件/SynologyDrive/CTIT-项目/横向项目/大型城市综合体/od_data_processed/'
filenames = os.listdir(data_dir)
for i in filenames:
    print(i)
    df_ts = pd.read_csv(data_dir + i, header=0, usecols=[1,2])
    df_ts.date = pd.to_datetime(df_ts.date)
    df_ts['wd'] = df_ts.date.apply(lambda x: x.weekday())
    df_ts['hour'] = df_ts.date.dt.hour
    targetcolumns = ['counts', 'wd', 'hour']
    a = df_ts[targetcolumns]
    b = np.arange(20,50)
    print(df_ts.loc[b])
    raise SystemExit(2)