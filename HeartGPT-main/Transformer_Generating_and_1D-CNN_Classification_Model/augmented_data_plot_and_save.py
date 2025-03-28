import numpy as np
import matplotlib.pyplot as plt
import os
import random

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

def plot_and_save_random_samples(num_samples):
    save_path = r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\processed_data\sample_plots"
    data_path = r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\processed_data"

    os.makedirs(save_path, exist_ok=True)

    X = np.load(os.path.join(data_path, "X_data.npy"))
    y = np.load(os.path.join(data_path, "y_labels.npy"))

    print(f"X shape = {X.shape}, y shape = {y.shape}")

    positive_indices = np.where(y == 1)[0]
    negative_indices = np.where(y == 0)[0]

    print(f"Positive 샘플 수: {len(positive_indices)}")
    print(f"Negative 샘플 수: {len(negative_indices)}")

    random_pos_indices = random.sample(list(positive_indices), min(num_samples, len(positive_indices)))
    random_neg_indices = random.sample(list(negative_indices), min(num_samples, len(negative_indices)))

    for i, idx in enumerate(random_pos_indices):
        plt.figure(figsize=(8, 4))
        plt.plot(range(X.shape[1]), X[idx], color='blue', linewidth=1.5)
        plt.title(f'Positive #{i + 1}', fontsize=10)

        plt.grid(True, linestyle='--')
        plt.tight_layout()

        # 파일 저장
        file_name = os.path.join(save_path, f'positive_sample_{i + 1}.png')
        plt.savefig(file_name, dpi=150)
        plt.close()
        print(f"저장됨: {file_name}")

    for i, idx in enumerate(random_neg_indices):
        plt.figure(figsize=(8, 4))
        plt.plot(range(X.shape[1]), X[idx], color='red', linewidth=1.5)
        plt.title(f'Negative  #{i + 1}', fontsize=10)

        plt.grid(True, linestyle='--')
        plt.tight_layout()

        file_name = os.path.join(save_path, f'negative_sample_{i + 1}.png')
        plt.savefig(file_name, dpi=150)
        plt.close()
        print(f"저장됨: {file_name}")

    print(f"\n총 {len(random_pos_indices) + len(random_neg_indices)}개의 샘플 플롯이 {save_path}에 저장됨")


if __name__ == "__main__":
    sample_count = 100
    print(f"선택할 샘플 수: {sample_count}개")

    plot_and_save_random_samples(num_samples=sample_count)