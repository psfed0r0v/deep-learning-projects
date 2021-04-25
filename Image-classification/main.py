import torch
import pandas as pd
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

from config import *
from utils import get_dataloaders
from opt import RangerVA
from model import ImageClfModel


def run_training(is_val=True, is_train=True, load_model_path=''):
    Path('models').mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    df = pd.read_csv(PATH_LABELS)
    train_loader, val_loader = get_dataloaders(df)
    if not is_val:
        del val_loader
    test_loader = get_dataloaders(df, test=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.wide_resnet50_2(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1000, out_features=200)
    )
    model.to(device)
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))
        model.eval()
    torch.cuda.empty_cache()

    optimizer = RangerVA(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    model = ImageClfModel(model, device, criterion)
    if is_train:
        for epoch in range(EPOCHS):
            print('Epoch:', epoch)
            model.train(train_loader, optimizer, epoch)
            if is_val:
                model.val(val_loader, epoch)
            #             torch.save(model.model.state_dict(), f'models/model_res_w50_{epoch}.h5')
            print('\n')
    pred = model.predict(test_loader)
    pred.to_csv('labels_test.csv', index=False)


if __name__ == '__main__':
    run_training(is_val=False, is_train=True)
