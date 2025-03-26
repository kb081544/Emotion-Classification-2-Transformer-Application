'''
1: 긍정
0: 부정
-1: 판단불가
'''

import pandas as pd
import numpy as np
import os

file_paths = [
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\augmented_ppg_data_positive_1\augmented_ppg_data_positive_1\all_augmented_data_1_positive.csv",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\augmented_ppg_data_positive_0.5\augmented_ppg_data_positive\all_augmented_data_0.5_positive.csv",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\augmented_ppg_data_2_negative\augmented_ppg_data_2\all_augmented_data_2_negative.csv",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\augmented_ppg_data_1_negative\augmented_ppg_data\all_augmented_data_1_negative.csv",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\augmented_ppg_data_0.5_negative\all_augmented_data_0.5_negative.csv",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\HeartGPT-main\train_data\high_peak_negative_chunks_below_threshold.csv",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\HeartGPT-main\train_data\high_peak_positive_chunks_above_threshold.csv"
]

X_positive = []
X_negative = []
X_undetermined = []

heartgpt_files = [
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\HeartGPT-main\train_data\high_peak_negative_chunks_below_threshold.csv",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\HeartGPT-main\train_data\high_peak_positive_chunks_above_threshold.csv"
]

for file_path in file_paths:
    print(f"Processing: {os.path.basename(file_path)}")
    
    is_heartgpt_data = file_path in heartgpt_files
    is_positive = "positive" in file_path.lower() and not is_heartgpt_data
    is_negative = "negative" in file_path.lower() and not is_heartgpt_data
    is_05_data = "0.5" in os.path.basename(file_path)

    try:
        df = pd.read_csv(file_path)
        print(f"  Loaded CSV with shape: {df.shape}")
        print(f"  Column names: {df.columns[:10]}...")
        
        if is_heartgpt_data:
            print(f"  HeartGPT data detected. First few columns: {df.columns[:10]}")
            
            if df.shape[1] == 500:
                data = df.values
            elif df.shape[1] > 500:
                data = df.iloc[:, :500].values
            else:
                data = np.pad(df.values, ((0, 0), (0, 500 - df.shape[1])), 'constant')
            
            X_undetermined.append(data)
            print(f"  Added {len(data)} samples to undetermined data, shape: {data.shape}")
            continue
            
        if is_05_data:
            front_data = df.iloc[:, 0:500].values
            front_sampled = front_data[1::3]
            print(f"  Front 500 columns sampled (3-step): {front_sampled.shape}")

            back_data = df.iloc[:, 500:1000].values
            print(f"  Back 500 columns (all rows): {back_data.shape}")

            if is_positive:
                X_positive.append(front_sampled)
                X_positive.append(back_data)
                print(f"  Added {len(front_sampled)} + {len(back_data)} samples to positive data")
            elif is_negative:
                X_negative.append(front_sampled)
                X_negative.append(back_data)
                print(f"  Added {len(front_sampled)} + {len(back_data)} samples to negative data")
        else:
            back_data = df.iloc[:, 500:1000].values
            print(f"  Back 500 columns: {back_data.shape}")

            if is_positive:
                X_positive.append(back_data)
                print(f"  Added {len(back_data)} samples to positive data")
            elif is_negative:
                X_negative.append(back_data)
                print(f"  Added {len(back_data)} samples to negative data")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

print("\nChecking data arrays before stacking:")
print(f"X_positive list length: {len(X_positive)}")
print(f"X_negative list length: {len(X_negative)}")
print(f"X_undetermined list length: {len(X_undetermined)}")

if not X_positive:
    print("Warning: X_positive is empty!")
if not X_negative:
    print("Warning: X_negative is empty!")
if not X_undetermined:
    print("Warning: X_undetermined is empty!")

X_positive_combined = np.vstack(X_positive) if X_positive else np.array([])
X_negative_combined = np.vstack(X_negative) if X_negative else np.array([])
X_undetermined_combined = np.vstack(X_undetermined) if X_undetermined else np.array([])

print("\nData Summary:")
print(f"Positive samples (1): {X_positive_combined.shape}")
print(f"Negative samples (0): {X_negative_combined.shape}")
print(f"Undetermined samples (-1): {X_undetermined_combined.shape}")

y_positive = np.ones(X_positive_combined.shape[0])
y_negative = np.zeros(X_negative_combined.shape[0])
y_undetermined = np.full(X_undetermined_combined.shape[0], -1)

X = np.vstack([X_positive_combined, X_negative_combined, X_undetermined_combined])
y = np.concatenate([y_positive, y_negative, y_undetermined])

print(f"Combined dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")

save_path = r"C:\Users\user\PycharmProjects\Emotion Classification 3\processed_data_3class"
os.makedirs(save_path, exist_ok=True)

np.save(os.path.join(save_path, "X_data.npy"), X)
np.save(os.path.join(save_path, "y_labels.npy"), y)

print(f"\nData saved to {save_path}")

print("\nQuick Statistics:")
print(f"Positive samples (1): {len(y_positive)} ({len(y_positive) / len(y) * 100:.1f}%)")
print(f"Negative samples (0): {len(y_negative)} ({len(y_negative) / len(y) * 100:.1f}%)")
print(f"Undetermined samples (-1): {len(y_undetermined)} ({len(y_undetermined) / len(y) * 100:.1f}%)")

try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(X_positive_combined[0], bins=30, alpha=0.7)
    plt.title("Sample Positive Data (1)")

    plt.subplot(1, 3, 2)
    plt.hist(X_negative_combined[0], bins=30, alpha=0.7)
    plt.title("Sample Negative Data (0)")
    
    plt.subplot(1, 3, 3)
    plt.hist(X_undetermined_combined[0], bins=30, alpha=0.7)
    plt.title("Sample Undetermined Data (-1)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "data_distribution_samples_3class.png"))
    print(f"Distribution plot saved to {save_path}")
except:
    print("Couldn't generate distribution plots (matplotlib may be missing)")

X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
print(f"\nReshaped data for 1D CNN: {X_reshaped.shape}")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

np.save(os.path.join(save_path, "X_train.npy"), X_train)
np.save(os.path.join(save_path, "X_test.npy"), X_test)
np.save(os.path.join(save_path, "y_train.npy"), y_train)
np.save(os.path.join(save_path, "y_test.npy"), y_test)

print("Train/test split data saved successfully!")
