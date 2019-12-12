import pandas as pd
from sklearn.model_selection import KFold
import os
import numpy as np



def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

csvfile = '/home/jiaxin/MICCAI2020/data/csv/all_files.csv'
save_root = '/home/jiaxin/MICCAI2020/data/csv/5_folds'
dataframe = pd.read_csv(csvfile)
samples = np.arange(len(dataframe))
kf = KFold(n_splits=5, shuffle=True, random_state=7)
splitno = 0

for train, val in kf.split(samples):
    splitno = splitno + 1
    train_csv = dataframe.loc[train.tolist()]     # https://www.cnblogs.com/kylinlin/p/5231404.html
    val_csv = dataframe.loc[val.tolist()]
    assert len(train) + len(val) == len(dataframe), 'wrong'
    train_csv_path = os.path.join(save_root, 'train_data_split{}.csv'.format(splitno))
    val_csv_path = os.path.join(save_root, 'val_data_split{}.csv'.format(splitno))

    train_csv.to_csv(train_csv_path, index=False, header=['image', 'mask'])
    val_csv.to_csv(val_csv_path, index=False, header=['image', 'mask'])

