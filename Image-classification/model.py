import torch
import pandas as pd


from config import *


class ImageClfModel:
    def __init__(self, model, device, criterion):
        self.INPUT_SIZE = INPUT_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.IMAGE_SIZE = IMAGE_SIZE
        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.model = model
        self.device = device
        self.criterion = criterion

    def train(self, train_loader, optimizer, epoch_num):
        model = self.model
        model.train()
        total_steps = len(train_loader)
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            optimizer.step()

            pred = torch.max(outputs, 1)[1]
            acc = (pred == target).sum().item()
            total += target.size(0)
            correct += acc
            print(f'Train step {batch_idx}/{total_steps}', flush=True, end='\r')
        print('Epoch:', epoch_num)
        print('Train accuracy per epoch', 100 * correct / len(train_loader.dataset), 'Loss:', loss.item())
        self.model = model

    def val(self, test_loader, epoch_num):
        model = self.model
        model.eval()
        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            total_steps = len(test_loader)
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                loss += self.criterion(outputs, target)
                pred = torch.max(outputs, 1)[1]
                correct += (pred == target).sum().item()
                total += target.size(0)
                print(f'Test step {batch_idx}/{total_steps}', flush=True, end='\r')
        print(f'Val epoch {epoch_num}')
        print('Val accuracy', 100 * correct / len(test_loader.dataset), 'Test Loss', loss.item())

    def predict(self, test_loader):
        model = self.model
        model.eval()
        preds = []
        with torch.no_grad():
            total_steps = len(test_loader)
            for batch_idx, (images, img_names) in enumerate(test_loader):
                images = images.to(self.device)
                outputs = model(images)
                predicted = torch.max(outputs.data, 1)[1]
                preds += list(zip(img_names, predicted.tolist()))
                print(f'Predict step {batch_idx}/{total_steps}', flush=True, end='\r')

        pred = pd.DataFrame(preds, columns=['Id', 'Category'])
        pred.loc[:, 'Category'] = pred['Category'].apply(lambda x: "%04d" % x)

        return pred
