import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from sklearn import metrics
from sklearn import tree
from sklearn.tree.export import export_text

from feature_dataset import SafetyFeatureDataset

class SafetyFeatureModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SafetyFeatureModel, self).__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, features):
        output = self.layer(features)
        output = self.output(output)
        return output

def train(model, training, validation, loss_fn, optimizer, num_epochs):
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
            output = model(feature)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            training_labels.append(label.cpu().detach().numpy())
            training_predictions.append(np.argmax(output.cpu().detach().numpy(), -1))
            training_loss.append(loss.item())
            training_accuracy = metrics.accuracy_score(training_labels[-1], training_predictions[-1])
            all_same = "ALL SAME" if len(set(training_predictions[-1])) == 1 else ""
            print("Iteration {} loss: {}, accuracy: {} {}".format(i, loss.item(), training_accuracy, all_same))
        model.eval()
        print("Going over validation data")
        for i, (feature, label) in enumerate(validation):
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
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
    dataset = SafetyFeatureDataset("/Users/arkadyark/Downloads/data/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv", "/Users/arkadyark/Downloads/data/features/*.csv")
    train_size = int(0.9*len(dataset))
    val_size = len(dataset) - train_size
    training, val = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(training, batch_size=train_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=val_size, shuffle=False)
    print("Dataset loaded!")
    trees = []
    for i, (feature, label) in enumerate(train_loader):
        for max_depth in (2, 3, 5, 7, 10):
            print("Max depth: {}".format(max_depth))
            clf = tree.DecisionTreeClassifier(max_depth=max_depth)
            clf = clf.fit(np.array(feature), np.array(label))
            training_predictions = clf.predict(np.array(feature))
            training_accuracy = metrics.accuracy_score(np.array(label), training_predictions)
            for j, (val_feature, val_label) in enumerate(val_loader):
                validation_predictions = clf.predict(np.array(val_feature))
                validation_accuracy = metrics.accuracy_score(np.array(val_label), validation_predictions)
                confusion_matrix = str(metrics.confusion_matrix(np.array(val_label), validation_predictions))
            print(training_accuracy)
            print(validation_accuracy)
            print(confusion_matrix)
            import pdb
            pdb.set_trace()
            r = export_text(clf, feature_names=('max_speed', 'max_dec', 'max_acc', 'max_acc_total'))
            print(r)
    # model = SafetyFeatureModel(4, 2)
    # weight = torch.tensor([1, 3], dtype=torch.float)
    # if torch.cuda.is_available():
    #     model.cuda()
    #     weight = weight.cuda()
    # loss_function = nn.CrossEntropyLoss(weight)
    # optimizer = optim.SGD(model.parameters(), lr=0.00001)
    # train(model, train_loader, val_loader, loss_function, optimizer, 500)
