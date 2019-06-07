import torch
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
import glob

class SafetyDataset(Dataset):
    def __init__(self, labels_file, features_path, num_files=10):
        feature_files = glob.glob(features_path)
        data = pd.concat([pd.read_csv(f) for f in feature_files[:num_files]])
        grouped_data = data.sort_values('second').groupby('bookingID')
        self.seq_lengths = [len(group) for group in grouped_data.groups.values()]
        longest_seq = max(self.seq_lengths)
        data_groups = list(grouped_data.groups)
        self.data = torch.zeros(len(grouped_data), longest_seq, grouped_data.get_group(0).shape[1])
        labels = pd.read_csv(labels_file)
        labels.drop_duplicates('bookingID', inplace=True)
        self.labels = torch.zeros(len(grouped_data))
        for i, x_len in enumerate(self.seq_lengths):
            bookingID = data_groups[i]
            sequence = grouped_data.get_group(bookingID).values
            self.data[i, 0:x_len, :] = torch.tensor(sequence)
            self.labels[i] = int(labels[labels.bookingID == bookingID].label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'features': self.data[index], 'length':self.seq_lengths[index], 'label':self.labels[index]}

if __name__ == '__main__':
    dataset = SafetyDataset("data/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv", "data/features/*.csv")
    train_size = int(0.9*len(dataset))
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size])
    loader = DataLoader(train, batch_size=5, shuffle=True)
