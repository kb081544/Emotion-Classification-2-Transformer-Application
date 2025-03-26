import pandas as pd
import numpy as np
import os

# file_paths = [
#     r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\augmented_ppg_data_positive_1\augmented_ppg_data_positive_1\all_augmented_data_1_positive.csv",
#     r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\augmented_ppg_data_positive_0.5\augmented_ppg_data_positive\all_augmented_data_0.5_positive.csv",
#     r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\augmented_ppg_data_2_negative\augmented_ppg_data_2\all_augmented_data_2_negative.csv",
#     r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\augmented_ppg_data_1_negative\augmented_ppg_data\all_augmented_data_1_negative.csv",
#     r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\augmented_ppg_data_0.5_negative\all_augmented_data_0.5_negative.csv",
# ]

file_paths = [
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\augmented_ppg_data_1_negative\augmented_ppg_data\all_augmented_data_1_negative.csv",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\augmented_ppg_data_2_negative\augmented_ppg_data_2\all_augmented_data_2_negative.csv",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\HeartGPT-main\high_peak_negative_chunks_negative_below_threshold_1.csv",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\HeartGPT-main\high_peak_positive_no_threshold.csv",
]

positive_files = [
    "high_peak_negative_chunks_negative_below_threshold_1.csv",
    "high_peak_positive_no_threshold.csv"
]

negative_files = [
    "all_augmented_data_1_negative.csv",
    "all_augmented_data_2_negative.csv"
]

special_negative_file = "all_augmented_data_1_negative.csv"


X_positive = []
X_negative = []

# for file_path in file_paths:
#     print(f"Processing: {os.path.basename(file_path)}")
#
#     is_positive = "positive" in file_path.lower()
#     is_05_data = "0.5" in os.path.basename(file_path)
#
#     try:
#         df = pd.read_csv(file_path)
#
#         if is_05_data:
#             front_data = df.iloc[:, 0:500].values
#             front_sampled = front_data[1::3]
#             print(f"  Front 500 columns sampled (3-step): {front_sampled.shape}")
#
#             back_data = df.iloc[:, 500:1000].values
#             print(f"  Back 500 columns (all rows): {back_data.shape}")
#
#             if is_positive:
#                 X_positive.append(front_sampled)
#                 X_positive.append(back_data)
#                 print(f"  Added {len(front_sampled)} + {len(back_data)} samples to positive data")
#             else:
#                 X_negative.append(front_sampled)
#                 X_negative.append(back_data)
#                 print(f"  Added {len(front_sampled)} + {len(back_data)} samples to negative data")
#         else:
#             back_data = df.iloc[:, 500:1000].values
#             print(f"  Back 500 columns: {back_data.shape}")
#
#             if is_positive:
#                 X_positive.append(back_data)
#                 print(f"  Added {len(back_data)} samples to positive data")
#             else:
#                 X_negative.append(back_data)
#                 print(f"  Added {len(back_data)} samples to negative data")
#
#     except Exception as e:
#         print(f"Error processing {file_path}: {str(e)}")
#
# X_positive_combined = np.vstack(X_positive) if X_positive else np.array([])
# X_negative_combined = np.vstack(X_negative) if X_negative else np.array([])
#
# print("\nData Summary:")
# print(f"Positive samples: {X_positive_combined.shape}")
# print(f"Negative samples: {X_negative_combined.shape}")
#
# y_positive = np.ones(X_positive_combined.shape[0])
# y_negative = np.zeros(X_negative_combined.shape[0])
#
# X = np.vstack([X_positive_combined, X_negative_combined])
# y = np.concatenate([y_positive, y_negative])
#
# print(f"Combined dataset shape: {X.shape}")
# print(f"Labels shape: {y.shape}")
#
# save_path = r"C:\Users\user\PycharmProjects\Emotion Classification 3\processed_data"
# os.makedirs(save_path, exist_ok=True)
#
# np.save(os.path.join(save_path, "X_data.npy"), X)
# np.save(os.path.join(save_path, "y_labels.npy"), y)
#
# print(f"\nData saved to {save_path}")
#
# print("\nQuick Statistics:")
# print(f"Positive samples: {len(y_positive)} ({len(y_positive) / len(y) * 100:.1f}%)")
# print(f"Negative samples: {len(y_negative)} ({len(y_negative) / len(y) * 100:.1f}%)")
#
# try:
#     import matplotlib.pyplot as plt
#
#     plt.figure(figsize=(12, 5))
#
#     plt.subplot(1, 2, 1)
#     plt.hist(X_positive_combined[0], bins=30, alpha=0.7)
#     plt.title("Sample Positive Data Distribution")
#
#     plt.subplot(1, 2, 2)
#     plt.hist(X_negative_combined[0], bins=30, alpha=0.7)
#     plt.title("Sample Negative Data Distribution")
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, "data_distribution_samples.png"))
#     print(f"Distribution plot saved to {save_path}")
# except:
#     print("Couldn't generate distribution plots (matplotlib may be missing)")
#
# X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
# print(f"\nReshaped data for 1D CNN: {X_reshaped.shape}")
#
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X_reshaped, y, test_size=0.2, random_state=42, stratify=y
# )
#
# print(f"Training data: {X_train.shape}, {y_train.shape}")
# print(f"Testing data: {X_test.shape}, {y_test.shape}")
#
# np.save(os.path.join(save_path, "X_train.npy"), X_train)
# np.save(os.path.join(save_path, "X_test.npy"), X_test)
# np.save(os.path.join(save_path, "y_train.npy"), y_train)
# np.save(os.path.join(save_path, "y_test.npy"), y_test)
#
# print("Train/test split data saved successfully!")

for file_path in file_paths:
    file_name = os.path.basename(file_path)
    print(f"Processing: {file_name}")

    is_positive = any(pos_file in file_path for pos_file in positive_files)
    is_negative = any(neg_file in file_path for neg_file in negative_files)
    is_special_negative = special_negative_file in file_path

    try:
        df = pd.read_csv(file_path)

        if is_negative:
            back_data = df.iloc[:, 500:1000].values
            print(f"  Back 500 columns (all rows): {back_data.shape}")
            X_negative.append(back_data)

            if is_special_negative:
                front_data = df.iloc[::5, 0:500].values
                print(f"  Front 500 columns (every 5th row): {front_data.shape}")
                X_negative.append(front_data)
                print(f"  Added front and back data samples to negative data")
            else:
                print(f"  Added back data samples to negative data")

        else:
            all_data = df.values
            print(f"  All data: {all_data.shape}")

            if is_positive:
                X_positive.append(all_data)
                print(f"  Added {len(all_data)} samples to positive data")
            else:
                X_negative.append(all_data)
                print(f"  Added {len(all_data)} samples to negative data")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

X_positive_combined = np.vstack(X_positive) if X_positive else np.array([])
X_negative_combined = np.vstack(X_negative) if X_negative else np.array([])

print("\nData Summary:")
print(f"Positive samples: {X_positive_combined.shape}")
print(f"Negative samples: {X_negative_combined.shape}")

y_positive = np.ones(X_positive_combined.shape[0])
y_negative = np.zeros(X_negative_combined.shape[0])

X = np.vstack([X_positive_combined, X_negative_combined])
y = np.concatenate([y_positive, y_negative])

print(f"Combined dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")

save_path = r"C:\Users\user\PycharmProjects\Emotion Classification 3\processed_data"
os.makedirs(save_path, exist_ok=True)

np.save(os.path.join(save_path, "X_data.npy"), X)
np.save(os.path.join(save_path, "y_labels.npy"), y)

print(f"\nData saved to {save_path}")

print("\nQuick Statistics:")
print(f"Positive samples: {len(y_positive)} ({len(y_positive) / len(y) * 100:.1f}%)")
print(f"Negative samples: {len(y_negative)} ({len(y_negative) / len(y) * 100:.1f}%)")

try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(X_positive_combined[0], bins=30, alpha=0.7)
    plt.title("Sample Positive Data Distribution")

    plt.subplot(1, 2, 2)
    plt.hist(X_negative_combined[0], bins=30, alpha=0.7)
    plt.title("Sample Negative Data Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "data_distribution_samples.png"))
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