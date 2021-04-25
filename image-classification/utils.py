import pandas as pd
from torch.utils.data import DataLoader

import os

from dataset import ImageData
from config import *


def create_test(path=PATH_TEST):
    list_of_files = os.listdir(path)
    data = pd.DataFrame(list_of_files, columns=['Id'])
    data['Category'] = data['Id']

    return data


def split_trainval(df, train_size=0.85, shuffle=True, val_train=False):
    if shuffle:
        df = df.sample(frac=1, random_state=12)
    size = int(len(df) * train_size)
    train = df[:size]
    if val_train:
        val = df[:size]
    else:
        val = df[size:]

    return train, val


def get_dataloaders(df, test=False, shuffle=True):
    if test:
        test_df = create_test(PATH_TEST)
        test_data = ImageData(test_df, PATH_TEST, True)
        test_loader = DataLoader(
            dataset=test_data, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=7)

        return test_loader

    train_df, val_df = split_trainval(df, train_size=1, val_train=True)
    train_data = ImageData(train_df, PATH_TRAINVAL)
    val_data = ImageData(val_df, PATH_TRAINVAL, True)
    train_loader = DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=7)
    val_loader = DataLoader(
        dataset=val_data, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=7)

    return train_loader, val_loader
