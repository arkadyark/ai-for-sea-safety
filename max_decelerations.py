import pandas as pd
import glob
import matplotlib.pyplot as plt

if __name__ == '__main__':
    labels_file = "data/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv"
    features_path = "data/features/*.csv"
    feature_files = glob.glob(features_path)
    data = pd.concat([pd.read_csv(f) for f in feature_files])
    data['acceleration'] = (data.acceleration_x**2 + data.acceleration_y**2 + data.acceleration_z**2)**0.5
    grouped_data = data.sort_values('second').groupby('bookingID')
    labels = pd.read_csv(labels_file)
    labels.drop_duplicates('bookingID', inplace=True)
    hist_0 = grouped_data['acceleration'].max()[labels[labels.label == 0].bookingID].hist(label='not dangerous', bins=range(0, 25), density=True, alpha=0.5)
    hist_1 = grouped_data['acceleration'].max()[labels[labels.label == 1].bookingID].hist(label='dangerous', bins=range(0, 25), density=True, alpha=0.5)
    plt.legend(loc='upper right')
    plt.show()
