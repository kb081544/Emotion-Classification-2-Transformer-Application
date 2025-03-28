import pandas as pd
import numpy as np
import os

file_paths = [
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\train_data\train_negative_below_threshold_1.csv",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\train_data\train_negative_above_threshold_1.csv"
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\train_data\train_positive_no_threshold.csv",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\train_data\train_negative_augmented_1.csv"
]

positive_files = [
    "train_positive_no_threshold.csv",
    "train_negative_below_threshold_1.csv",
]

negative_files = [
    "train_negative_above_threshold_1.csv",
    "train_negative_augmented_1.csv",
]

special_negative_file = "train_negative_augmented_1.csv"

X_positive = []
X_negative = []

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
            print(f" Back 500 columns (all rows): {back_data.shape}")
            X_negative.append(back_data)

            if is_special_negative:
                front_data = df.iloc[::5, 0:500].values
                print(f" Front 500 columns (every 5th row): {front_data.shape}")
                X_negative.append(front_data)
                print(f" Added front and back data samples to negative data")
            else:
                print(f" Added back data samples to negative data")
        else:
            all_data = df.values
            print(f" All data: {all_data.shape}")

            if is_positive:
                X_positive.append(all_data)
                print(f" Added {len(all_data)} samples to positive data")
            else:
                X_negative.append(all_data)
                print(f" Added {len(all_data)} samples to negative data")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

X_positive_combined = np.vstack(X_positive) if X_positive else np.array([])
X_negative_combined = np.vstack(X_negative) if X_negative else np.array([])

print("\nData Summary:")
print(f"Positive samples: {X_positive_combined.shape}")
print(f"Negative samples: {X_negative_combined.shape}")

# 여기서 레이블을 반대로 지정
y_positive = np.zeros(X_positive_combined.shape[0])  # 긍정 -> 0
y_negative = np.ones(X_negative_combined.shape[0])  # 부정 -> 1

X = np.vstack([X_positive_combined, X_negative_combined])
y = np.concatenate([y_positive, y_negative])

print(f"Combined dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")

save_path = r"/processed_data"
os.makedirs(save_path, exist_ok=True)

np.save(os.path.join(save_path, r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\processed_data\X_data.npy"), X)
np.save(os.path.join(save_path, r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\processed_data\y_labels.npy"), y)

print(f"\n데이터 저장됨 : {save_path}")
print(f"Positive samples (label 0): {len(y_positive)} ({len(y_positive) / len(y) * 100:.1f}%)")
print(f"Negative samples (label 1): {len(y_negative)} ({len(y_negative) / len(y) * 100:.1f}%)")

try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(X_positive_combined[0], bins=30, alpha=0.7)
    plt.title("Sample Positive Data Distribution (Label 0)")

    plt.subplot(1, 2, 2)
    plt.hist(X_negative_combined[0], bins=30, alpha=0.7)
    plt.title("Sample Negative Data Distribution (Label 1)")

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

np.save(os.path.join(save_path, r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\processed_data\X_train.npy"), X_train)
np.save(os.path.join(save_path, r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\processed_data\X_test.npy"), X_test)
np.save(os.path.join(save_path, r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\processed_data\y_train.npy"), y_train)
np.save(os.path.join(save_path, r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\processed_data\y_test.npy"), y_test)

print("Train/test split data saved successfully!")