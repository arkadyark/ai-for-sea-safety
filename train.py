from dataset import SafetyDataset
from model import SafetyModel
from config import NUM_EPOCHS, BATCH_SIZE, INPUT_DIM, SECOND_HIDDEN_DIMS, MINUTE_HIDDEN_DIMS, LSTM_LAYERS

import torch
from torch.utils.data import random_split, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics

def train(model, training, validation, loss_fn, optimizer, num_epochs=NUM_EPOCHS):
    for epoch in range(num_epochs):
        training_loss, validation_loss = [], []
        training_labels, validation_labels = [], []
        training_predictions, validation_predictions = [], []
        model.train()
        print("Going over training data")
        for i, (feature, label) in enumerate(training):
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            model.zero_grad()
            model.init_hidden()
            output = model(feature)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            training_labels.append(label.cpu().detach().numpy())
            training_predictions.append(np.argmax(output.cpu().detach().numpy(), -1))
            training_loss.append(loss.item())
            training_accuracy = metrics.accuracy_score(training_labels[-1], training_predictions[-1])
            print("Iteration {} loss: {}, accuracy: {}".format(i, loss.item(), training_accuracy))
        model.eval()
        print("Going over validation data")
        for i, (feature, label) in enumerate(validation):
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            model.init_hidden()
            output = model(feature)
            loss = loss_fn(output, label)
            validation_loss.append(loss.item())
            validation_labels.append(label.cpu().detach().numpy())
            validation_predictions.append(np.argmax(output.cpu().detach().numpy(), -1))
        training_accuracy = metrics.accuracy_score(np.concatenate(training_labels, 0), np.concatenate(training_predictions, 0))
        validation_accuracy = metrics.accuracy_score(np.concatenate(validation_labels, 0), np.concatenate(validation_predictions, 0))
        confusion_matrix = str(metrics.confusion_matrix(np.concatenate(validation_labels, 0), np.concatenate(validation_predictions, 0)))
        print("Epoch {}, Training loss: {}, Training accuracy: {}, Validation loss: {}, Validation accuracy: {}, Validation confusion matrix: {}".format(epoch, np.mean(training_loss), np.mean(validation_loss), training_accuracy, validation_accuracy, confusion_matrix))

if __name__ == '__main__':
    dataset = SafetyDataset("data/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv", "data/features/*.csv", num_files=10)
    train_size = int(0.9*len(dataset))
    val_size = len(dataset) - train_size
    training, val = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(training, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)
    print("Dataset loaded!")
    model = SafetyModel(INPUT_DIM, SECOND_HIDDEN_DIMS, MINUTE_HIDDEN_DIMS, LSTM_LAYERS, BATCH_SIZE)
    if torch.cuda.is_available():
        model.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, train_loader, val_loader, loss_function, optimizer)
