import numpy as np
import matplotlib.pyplot as plt
import os
import random

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
save_path = r"C:\Users\user\PycharmProjects\Emotion Classification 3\processed_data_3class"

X = np.load(os.path.join(save_path, "X_data.npy"))
y = np.load(os.path.join(save_path, "y_labels.npy"))

print(f"데이터 로드 완료: X shape = {X.shape}, y shape = {y.shape}")

positive_indices = np.where(y == 1)[0]
negative_indices = np.where(y == 0)[0]
undetermined_indices = np.where(y == -1)[0]

print(f"Positive 샘플 수: {len(positive_indices)}")
print(f"Negative 샘플 수: {len(negative_indices)}")
print(f"Undetermined 샘플 수: {len(undetermined_indices)}")


def visualize_samples(num_samples=10, alpha=0.3, figsize=(18, 16), dpi=300):
    plt.figure(figsize=figsize, dpi=dpi)

    random_pos_indices = random.sample(list(positive_indices), min(num_samples, len(positive_indices)))
    random_neg_indices = random.sample(list(negative_indices), min(num_samples, len(negative_indices)))
    random_und_indices = random.sample(list(undetermined_indices), min(num_samples, len(undetermined_indices)))

    plt.subplot(2, 1, 1)
    plt.title(f'랜덤 데이터 시각화 (각 {num_samples}개)', fontsize=10)

    for idx in random_pos_indices:
        plt.plot(range(X.shape[1]), X[idx], color='blue', alpha=alpha, linewidth=1)

    for idx in random_neg_indices:
        plt.plot(range(X.shape[1]), X[idx], color='red', alpha=alpha, linewidth=1)

    for idx in random_und_indices:
        plt.plot(range(X.shape[1]), X[idx], color='green', alpha=alpha, linewidth=1)

    plt.legend(['Positive (파랑)', 'Negative (빨강)', 'Undetermined (초록)'], fontsize=8)
    plt.xlabel('시퀀스 위치', fontsize=8)
    plt.ylabel('값', fontsize=8)
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.subplot(2, 1, 2)

    pos_mean = X[positive_indices].mean(axis=0)
    neg_mean = X[negative_indices].mean(axis=0)
    und_mean = X[undetermined_indices].mean(axis=0) if len(undetermined_indices) > 0 else np.zeros(X.shape[1])

    pos_std = X[positive_indices].std(axis=0)
    neg_std = X[negative_indices].std(axis=0)
    und_std = X[undetermined_indices].std(axis=0) if len(undetermined_indices) > 0 else np.zeros(X.shape[1])

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

    if len(undetermined_indices) > 0:
        plt.plot(x_axis, und_mean, color='green', linewidth=2, label='Undetermined 평균')
        plt.fill_between(
            x_axis,
            und_mean - und_std,
            und_mean + und_std,
            color='green',
            alpha=0.2,
            label='Undetermined 표준편차'
        )

    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'emotion_data_visualization_3class.png'))
    print(f"시각화 이미지가 {save_path}에 저장되었습니다.")
    plt.show()


def visualize_overlay(num_samples=50, alpha=0.1, figsize=(15, 10), dpi=300):
    plt.figure(figsize=figsize, dpi=dpi)

    random_pos_indices = random.sample(list(positive_indices), min(num_samples, len(positive_indices)))
    random_neg_indices = random.sample(list(negative_indices), min(num_samples, len(negative_indices)))
    random_und_indices = random.sample(list(undetermined_indices), min(num_samples, len(undetermined_indices)))

    plt.title(f'데이터 시각화 (각 {num_samples}개 샘플)', fontsize=10)

    for idx in random_pos_indices:
        plt.plot(range(X.shape[1]), X[idx], color='blue', alpha=alpha, linewidth=1)

    for idx in random_neg_indices:
        plt.plot(range(X.shape[1]), X[idx], color='red', alpha=alpha, linewidth=1)

    for idx in random_und_indices:
        plt.plot(range(X.shape[1]), X[idx], color='green', alpha=alpha, linewidth=1)

    pos_patch = plt.Line2D([0], [0], color='blue', linewidth=2, label='Positive (파랑)')
    neg_patch = plt.Line2D([0], [0], color='red', linewidth=2, label='Negative (빨강)')
    und_patch = plt.Line2D([0], [0], color='green', linewidth=2, label='Undetermined (초록)')
    plt.legend(handles=[pos_patch, neg_patch, und_patch], fontsize=8)

    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'emotion_data_overlay_3class.png'))
    print(f"오버레이 시각화 이미지가 {save_path}에 저장되었습니다.")
    plt.show()


def visualize_subgroups(samples_per_group=10, num_groups=5, alpha=0.3, figsize=(20, 20), dpi=300):
    plt.figure(figsize=figsize, dpi=dpi)

    total_pos_samples = min(samples_per_group * num_groups, len(positive_indices))
    total_neg_samples = min(samples_per_group * num_groups, len(negative_indices))
    total_und_samples = min(samples_per_group * num_groups, len(undetermined_indices))

    sampled_pos_indices = random.sample(list(positive_indices), total_pos_samples)
    sampled_neg_indices = random.sample(list(negative_indices), total_neg_samples)
    sampled_und_indices = random.sample(list(undetermined_indices),
                                        total_und_samples) if undetermined_indices.size > 0 else []

    pos_groups = [sampled_pos_indices[i:i + samples_per_group] for i in range(0, total_pos_samples, samples_per_group)]
    neg_groups = [sampled_neg_indices[i:i + samples_per_group] for i in range(0, total_neg_samples, samples_per_group)]
    und_groups = [sampled_und_indices[i:i + samples_per_group] for i in
                  range(0, total_und_samples, samples_per_group)] if sampled_und_indices else []

    max_groups = min(num_groups, len(pos_groups), len(neg_groups))
    if und_groups:
        max_groups = min(max_groups, len(und_groups))

    for i in range(max_groups):
        plt.subplot(max_groups, 3, i * 3 + 1)
        plt.title(f'Positive 그룹 {i + 1}', fontsize=8)

        for idx in pos_groups[i]:
            plt.plot(range(X.shape[1]), X[idx], color='blue', alpha=alpha, linewidth=1)

        plt.grid(True, alpha=0.3, linestyle='--')

        plt.subplot(max_groups, 3, i * 3 + 2)
        plt.title(f'Negative 그룹 {i + 1}', fontsize=8)

        for idx in neg_groups[i]:
            plt.plot(range(X.shape[1]), X[idx], color='red', alpha=alpha, linewidth=1)

        plt.grid(True, alpha=0.3, linestyle='--')

        if und_groups:
            plt.subplot(max_groups, 3, i * 3 + 3)
            plt.title(f'Undetermined 그룹 {i + 1}', fontsize=8)

            for idx in und_groups[i]:
                plt.plot(range(X.shape[1]), X[idx], color='green', alpha=alpha, linewidth=1)

            plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'emotion_data_subgroups_3class.png'))
    print(f"서브그룹 시각화 이미지가 {save_path}에 저장되었습니다.")
    plt.show()


# 시각화 실행
print("다양한 시각화를 생성합니다...")
visualize_samples(num_samples=20, alpha=0.3)
visualize_overlay(num_samples=100, alpha=0.05)
visualize_subgroups(samples_per_group=10, num_groups=4, alpha=0.3)
print("모든 시각화가 완료되었습니다.")