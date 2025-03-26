import numpy as np
import glob
import os

def load_ppg_data(data_dir):
    all_data = []
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    for file_path in files:
        try:
            ppg_values = []
            with open(file_path, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    try:
                        timestamp, ppg = line.strip().split('\t')
                        ppg_values.append(float(ppg))
                    except ValueError:
                        continue

            if ppg_values:
                ppg_values = np.array(ppg_values)
                all_data.append(ppg_values)

        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")

    if len(all_data) > 0:
        combined_data = np.concatenate(all_data)
        print(f"\nTotal samples loaded: {len(combined_data)}")
        print(f"Combined data range: {combined_data.min()} to {combined_data.max()}")
        return combined_data
    else:
        raise ValueError("No data could be loaded from the specified directory")

data_dir = r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\P\green"
concated_data=load_ppg_data(data_dir)
concated_data_int=concated_data.astype(int)
file_name=r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\concated_ppg_data_positive.txt"
np.savetxt(file_name, concated_data_int,fmt='%7.0f', newline='\n')
