import numpy as np
import matplotlib.pyplot as plt
import os
import random

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
save_path = r"/processed_data"

X = np.load(os.path.join(save_path, "X_data.npy"))
y = np.load(os.path.join(save_path, "y_labels.npy"))

print(f"데이터 로드 완료: X shape = {X.shape}, y shape = {y.shape}")

positive_indices = np.where(y == 1)[0]
negative_indices = np.where(y == 0)[0]

print(f"Positive 샘플 수: {len(positive_indices)}")
print(f"Negative 샘플 수: {len(negative_indices)}")


def visualize_samples(num_samples=10, alpha=0.3, figsize=(18, 12), dpi=300):
    plt.figure(figsize=figsize, dpi=dpi)

    random_pos_indices = random.sample(list(positive_indices), min(num_samples, len(positive_indices)))
    random_neg_indices = random.sample(list(negative_indices), min(num_samples, len(negative_indices)))

    plt.subplot(2, 1, 1)
    plt.title(f'랜덤 데이터 시각화 (각 {num_samples}개)', fontsize=10)

    for idx in random_pos_indices:
        plt.plot(range(X.shape[1]), X[idx], color='blue', alpha=alpha, linewidth=1)

    for idx in random_neg_indices:
        plt.plot(range(X.shape[1]), X[idx], color='red', alpha=alpha, linewidth=1)

    plt.legend(['Positive (파랑)', 'Negative (빨강)'], fontsize=8)
    plt.xlabel('시퀀스 위치', fontsize=8)
    plt.ylabel('값', fontsize=8)
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.subplot(2, 1, 2)

    pos_mean = X[positive_indices].mean(axis=0)
    neg_mean = X[negative_indices].mean(axis=0)

    pos_std = X[positive_indices].std(axis=0)
    neg_std = X[negative_indices].std(axis=0)

    x_axis = np.arange(X.shape[1])

    plt.plot(x_axis, pos_mean, color='blue', linewidth=2, label='Positive 평균')
    plt.fill_between(
        x_axis,
        pos_mean - pos_std,
        pos_mean + pos_std,
        color='blue',
        alpha=0.2,
        label='Positive 표준편차'
    )

    plt.plot(x_axis, neg_mean, color='red', linewidth=2, label='Negative 평균')
    plt.fill_between(
        x_axis,
        neg_mean - neg_std,
        neg_mean + neg_std,
        color='red',
        alpha=0.2,
        label='Negative 표준편차'
    )

    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'emotion_data_visualization.png'))
    print(f"시각화 이미지가 {save_path}에 저장되었습니다.")
    plt.show()


def visualize_overlay(num_samples=50, alpha=0.1, figsize=(15, 10), dpi=300):
    plt.figure(figsize=figsize, dpi=dpi)

    random_pos_indices = random.sample(list(positive_indices), min(num_samples, len(positive_indices)))
    random_neg_indices = random.sample(list(negative_indices), min(num_samples, len(negative_indices)))

    plt.title(f'데이터 시각화 (각 {num_samples}개 샘플)', fontsize=10)

    for idx in random_pos_indices:
        plt.plot(range(X.shape[1]), X[idx], color='blue', alpha=alpha, linewidth=1)

    for idx in random_neg_indices:
        plt.plot(range(X.shape[1]), X[idx], color='red', alpha=alpha, linewidth=1)

    pos_patch = plt.Line2D([0], [0], color='blue', linewidth=2, label='Positive (파랑)')
    neg_patch = plt.Line2D([0], [0], color='red', linewidth=2, label='Negative (빨강)')
    plt.legend(handles=[pos_patch, neg_patch], fontsize=8)

    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'emotion_data_overlay.png'))
    print(f"오버레이 시각화 이미지가 {save_path}에 저장되었습니다.")
    plt.show()


def visualize_subgroups(samples_per_group=10, num_groups=5, alpha=0.3, figsize=(20, 15), dpi=300):
    plt.figure(figsize=figsize, dpi=dpi)

    total_pos_samples = min(samples_per_group * num_groups, len(positive_indices))
    total_neg_samples = min(samples_per_group * num_groups, len(negative_indices))

    sampled_pos_indices = random.sample(list(positive_indices), total_pos_samples)
    sampled_neg_indices = random.sample(list(negative_indices), total_neg_samples)

    pos_groups = [sampled_pos_indices[i:i + samples_per_group] for i in range(0, total_pos_samples, samples_per_group)]
    neg_groups = [sampled_neg_indices[i:i + samples_per_group] for i in range(0, total_neg_samples, samples_per_group)]

    for i in range(min(num_groups, len(pos_groups), len(neg_groups))):
        plt.subplot(num_groups, 2, i * 2 + 1)
        plt.title(f'Positive 그룹 {i + 1}', fontsize=8)

        for idx in pos_groups[i]:
            plt.plot(range(X.shape[1]), X[idx], color='blue', alpha=alpha, linewidth=1)


        plt.grid(True, alpha=0.3, linestyle='--')

        plt.subplot(num_groups, 2, i * 2 + 2)
        plt.title(f'Negative 그룹 {i + 1}', fontsize=8)

        for idx in neg_groups[i]:
            plt.plot(range(X.shape[1]), X[idx], color='red', alpha=alpha, linewidth=1)

        plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'emotion_data_subgroups.png'))
    print(f"서브그룹 시각화 이미지가 {save_path}에 저장되었습니다.")
    plt.show()


# 시각화 실행
print("다양한 시각화를 생성합니다...")
visualize_samples(num_samples=20, alpha=0.3)
visualize_overlay(num_samples=100, alpha=0.05)
visualize_subgroups(samples_per_group=10, num_groups=4, alpha=0.3)
print("모든 시각화가 완료되었습니다.")