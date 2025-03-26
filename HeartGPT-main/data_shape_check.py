import numpy as np

data=r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\augmented_ppg_data_0.5_negative\all_augmented_data.csv"
data=np.loadtxt(data, delimiter=',')
#500~999
print(data)
