import torch
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
import glob

class SafetyDataset(Dataset):
    def __init__(self, labels_file, features_path, num_files=10):
        feature_files = glob.glob(features_path)
        data = pd.concat([pd.read_csv(f) for f in feature_files[:num_files]])
        grouped_data = data.sort_values('second').groupby('bookingID')
        data_groups = list(grouped_data.groups)
        sorted_seq_lengths = sorted([len(group) for group in grouped_data.groups.values()])
        max_length = sorted_seq_lengths[int(0.99*len(sorted_seq_lengths))]
        seconds, minutes = 60, int(max_length/60)
        self.data = torch.zeros(len(grouped_data), minutes, seconds, grouped_data.get_group(0).shape[1])
        labels = pd.read_csv(labels_file)
        labels.drop_duplicates('bookingID', inplace=True)
        self.labels = torch.zeros(len(grouped_data), dtype=torch.long)
        for i in range(len(data_groups)):
            bookingID = data_groups[i]
            sequence = grouped_data.get_group(bookingID).values
            for minute in range(min(int(len(sequence)/seconds), minutes)):
                sequence_minute = sequence[minute*seconds:(minute+1)*seconds, :]
                if len(sequence_minute) < seconds:
                    print("Found a sequence ({}) that wasn't long enough ({})".format(minute, len(sequence_minute)))
                self.data[i, minute, :len(sequence_minute), :] = torch.tensor(sequence_minute)
            self.labels[i] = int(labels[labels.bookingID == bookingID].label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

if __name__ == '__main__':
    dataset = SafetyDataset("/Users/arkadyark/Downloads/data/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv", "/Users/arkadyark/Downloads/data/features/*.csv", num_files=10)
    train_size = int(0.9*len(dataset))
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size])
    loader = DataLoader(train, batch_size=5, shuffle=True)
