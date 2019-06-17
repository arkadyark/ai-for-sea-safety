import torch
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
import glob
import time

class SafetyFeatureDataset(Dataset):
    def __init__(self, labels_file, features_path):
        start_time = time.time()
        feature_files = glob.glob(features_path)
        data = pd.concat([pd.read_csv(f) for f in feature_files])
        labels = pd.read_csv(labels_file)
        labels.drop_duplicates('bookingID', inplace=True)
        data_loaded = time.time()
        print("Data loaded in {} seconds".format(data_loaded - start_time))
        self.features = self.get_features(data, labels)
        print("Features loaded in {} seconds".format(time.time() - data_loaded))

    def get_speed_features(self, seconds, speeds):
        speed_segments = {-1: [], 0: [], 1:[]}
        current_segment = [speeds[0]]
        currently_increasing = -1000
        for i in range(len(seconds) - 1):
            if seconds[i+1] - seconds[i] > 1:
                if currently_increasing == -1000:
                    current_segment = []
                else:
                    speed_segments[currently_increasing].append(tuple(current_segment))
                    currently_increasing = -1000
                    current_segment = [speeds[i+1]]
            elif currently_increasing == -1000:
                current_segment.append(speeds[i+1])
                if speeds[i+1] < speeds[i]: currently_increasing = -1
                if speeds[i+1] == speeds[i]: currently_increasing = 0
                if speeds[i+1] > speeds[i]: currently_increasing = 1
            elif currently_increasing == -1:
                if speeds[i+1] < speeds[i]:
                    current_segment.append(speeds[i+1])
                else:
                    speed_segments[-1].append(tuple(current_segment))
                    current_segment = [speeds[i+1]]
                    currently_increasing = -1000
            elif currently_increasing == 0:
                if speeds[i+1] == speeds[i]:
                    current_segment.append(speeds[i+1])
                else:
                    speed_segments[0].append(tuple(current_segment))
                    current_segment = [speeds[i+1]]
                    currently_increasing = -1000
            elif currently_increasing == 1:
                if speeds[i+1] > speeds[i]:
                    current_segment.append(speeds[i+1])
                else:
                    speed_segments[1].append(tuple(current_segment))
                    current_segment = [speeds[i+1]]
                    currently_increasing = -1000
        decelerations = pd.DataFrame(list(map(lambda s: (s[0] - s[-1])/len(s), speed_segments[-1])))
        accelerations = pd.DataFrame(list(map(lambda s: (s[-1] - s[0])/len(s), speed_segments[1])))
        try:
            return float(decelerations.nlargest(3, 0).mean()), float(accelerations.nlargest(3, 0).mean())
        except:
            return 0, 0

    def get_features(self, data, labels):
        features = []
        data['acceleration'] = (data.acceleration_x**2 + data.acceleration_y**2 + data.acceleration_z**2)**0.5
        grouped_data = data.sort_values('second').groupby('bookingID')
        max_speeds = grouped_data['Speed'].apply(lambda g: g.nlargest(5).mean())
        max_acc_totals = grouped_data['acceleration'].apply(lambda g: g.nlargest(5).mean())
        for bookingID in grouped_data.groups:
            group = grouped_data.get_group(bookingID)
            max_dec, max_acc = self.get_speed_features(list(group.second), list(group.Speed))
            max_speed = max_speeds[bookingID]
            max_acc_total = max_acc_totals[bookingID]
            label = int(labels[labels.bookingID == bookingID].label)
            features.append({'bookingID':bookingID,
                'max_speed': max_speed,
                'max_dec': max_dec,
                'max_acc': max_acc,
                'max_acc_total': max_acc_total,
                'label': label})
        return pd.DataFrame(features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return torch.tensor(self.features.loc[index, ['max_speed', 'max_dec', 'max_acc', 'max_acc_total']].values).float(), int(self.features.loc[index, 'label'])

if __name__ == '__main__':
    labels_file = "/Users/arkadyark/Downloads/data/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv"
    features_path = "/Users/arkadyark/Downloads/data/features/*.csv"
    dataset = SafetyFeatureDataset(labels_file, features_path)
    train_size = int(0.9*len(dataset))
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size])
    loader = DataLoader(train, batch_size=5, shuffle=True)
