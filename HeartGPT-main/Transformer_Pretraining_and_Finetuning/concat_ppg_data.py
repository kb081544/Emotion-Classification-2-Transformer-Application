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

data_dir_p = r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\P\green"
data_dir_n = r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\N\green"

concatenate_data_p=load_ppg_data(data_dir_p)
concatenate_data_int_n=concatenate_data_p.astype(int)

concatenate_data_n=load_ppg_data(data_dir_n)
concatenate_data_int_n=concatenate_data_n.astype(int)

concat_data=np.concatenate((concatenate_data_p,concatenate_data_n), axis=0)

file_name=r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\dataset_for_pretraining_model\concatenated_ppg_data.txt"
np.savetxt(file_name, concat_data,fmt='%7.0f', newline='\n')
