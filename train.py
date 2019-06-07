from dataset import SafetyDataset
from model import SafetyModel

from torch.utils.data import random_split, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_DIMS = 50
INPUT_DIM = 11
NUM_EPOCHS = 100
BATCH_SIZE = 50
LSTM_LAYERS = 3

def train(model, training, validation, loss_fn, optimizer, num_epochs=NUM_EPOCHS):
    for epoch in range(num_epochs):
        print("LSTM weights sum: {:.2f}".format(sum(sum(list(model.lstm.parameters())[0]))))
        train_loss, valid_loss = [], []
        model.train()
        print("Going over training data")
        for i, data in enumerate(training):
            model.zero_grad()
            outputs = model(data['features'], data['length'])
            loss = loss_fn(outputs, data['label'])
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if i == 100: break
        model.eval()
        print("Going over validation data")
        confusion_matrix = [[0, 0], [0, 0]]
        for i, data in enumerate(validation):
            outputs = model(data['features'], data['length'])
            loss = loss_fn(outputs, data['label'])
            valid_loss.append(loss.item())
            outputs = torch.round(outputs)
            confusion_matrix[0][0] += int(sum(outputs < data['label']))
            confusion_matrix[0][1] += int(sum((outputs == data['label']).double() * (data['label'] == 0).double()))
            confusion_matrix[1][0] += int(sum(outputs > data['label']))
            confusion_matrix[1][1] += int(sum((outputs == data['label']).double() * (data['label'] == 1).double()))
            if i == 20: break
        print("Epoch {}, Training loss: {}, Validation loss: {}".format(epoch, np.mean(train_loss), np.mean(valid_loss)))
        print("Confusion matrix:")
        print("pred\\true | 0   |   1")
        print("0:         | {}  |  {}".format(confusion_matrix[0][0], confusion_matrix[0][1]))
        print("1:         | {}  |  {}".format(confusion_matrix[1][0], confusion_matrix[1][1]))

if __name__ == '__main__':
    dataset = SafetyDataset("data/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv", "data/features/*.csv", num_files=6)
    train_size = int(0.9*len(dataset))
    val_size = len(dataset) - train_size
    training, val = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(training, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
    print("Dataset loaded!")
    model = SafetyModel(INPUT_DIM, HIDDEN_DIMS, LSTM_LAYERS, BATCH_SIZE)
    loss_function = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
    train(model, train_loader, val_loader, loss_function, optimizer)
