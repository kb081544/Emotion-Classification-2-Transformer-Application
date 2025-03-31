import numpy as np
import pandas as pd
import heartpy as hp
import matplotlib.pyplot as plt
import os
from keras.models import load_model

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

def visualize_classification_results(file_paths, threshold_value, model_path, viz_dir):
    """
    CNN 분류 결과에 따라 PPG 파형을 비교하는 시각화 함수
    - 정확히 1로 분류된 데이터 (true_label=1, pred=1)
    - 오분류: 실제 1인데 0으로 분류된 데이터 (true_label=1, pred=0)
    - 오분류: 실제 0인데 1로 분류된 데이터 (true_label=0, pred=1)
    """
    os.makedirs(viz_dir, exist_ok=True)

    # 모델 로드
    model = load_model(model_path)

    # 분류 결과별 세그먼트 저장 리스트
    correct_1_segments = []  # true=1, pred=1
    false_neg_segments = []  # true=1, pred=0
    false_pos_segments = []  # true=0, pred=1

    # 파일별로 처리
    for file_counter, file_path in enumerate(file_paths):
        file_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        print(f"처리 중인 파일 {file_counter + 1}/{len(file_paths)}: {file_name}")

        try:
            # PPG 데이터 읽기 및 세그먼트 처리
            test_segments, usable_or_rejected, cnn_predictions, true_labels, avg_peak_values = reshape_vector_to_matrix_test(
                file_path, threshold_value, model)

            # 사용 가능한 세그먼트만 필터링
            usable_indices = [i for i, val in enumerate(usable_or_rejected) if val == 1]

            # 각 분류 결과별로 세그먼트 수집
            for i in usable_indices:
                if true_labels[i] == 1 and cnn_predictions[i] == 1:
                    correct_1_segments.append((test_segments[i], file_name))
                elif true_labels[i] == 1 and cnn_predictions[i] == 0:
                    false_neg_segments.append((test_segments[i], file_name))
                elif true_labels[i] == 0 and cnn_predictions[i] == 1:
                    false_pos_segments.append((test_segments[i], file_name))

        except Exception as e:
            print(f"파일 처리 중 오류 발생: {str(e)}")

    # 세그먼트 수 집계
    print(f"정확히 1로 분류된 세그먼트 수: {len(correct_1_segments)}")
    print(f"실제 1인데 0으로 오분류된 세그먼트 수: {len(false_neg_segments)}")
    print(f"실제 0인데 1로 오분류된 세그먼트 수: {len(false_pos_segments)}")

    # 각 분류 결과별로 시각화
    visualize_segments(correct_1_segments, "true1_pred1", "정확히 1로 분류된 파형 (true=1, pred=1)", viz_dir)
    visualize_segments(false_neg_segments, "true1_pred0", "실제 1인데 0으로 오분류된 파형 (true=1, pred=0)", viz_dir)
    visualize_segments(false_pos_segments, "true0_pred1", "실제 0인데 1로 오분류된 파형 (true=0, pred=1)", viz_dir)

    # 종합 시각화 (각 카테고리별 평균)
    visualize_average_comparison(correct_1_segments, false_neg_segments, false_pos_segments, viz_dir)


def reshape_vector_to_matrix_test(file_dir, threshold_value, model, row_size=500):
    """
    기존 코드에서 가져온 함수 - PPG 파일을 읽고 세그먼트로 분할하여 CNN 예측 수행
    """
    file = open(file_dir, 'r')
    file_data = []
    lines = file.readlines()
    for line in lines[15:-1]:
        values = line.strip().split()
        second_int = int(values[1])
        file_data.append(second_int)

    n = len(file_data)
    stride = row_size // 1

    test_segments = []
    usable_or_rejected = []
    cnn_predictions = []
    true_labels = []
    avg_peak_values = []

    total_chunks = (n - row_size) // stride + 1
    half_chunks = total_chunks // 2

    i = 0
    while i * stride + row_size <= n:
        start_idx = i * stride
        end_idx = min(start_idx + row_size, n)
        window_data = file_data[start_idx:end_idx]

        if len(window_data) < row_size:
            window_data = np.pad(window_data, (0, row_size - len(window_data)),
                                 mode='constant', constant_values=np.nan)

        valid_data = hp.filter_signal(window_data, cutoff=[0.5, 8],
                                      sample_rate=25, order=3, filtertype="bandpass")

        is_usable = False
        avg_peak_value = 0
        try:
            wd, m = hp.process(valid_data, sample_rate=25)

            valid_peak_count = len(wd['peaklist']) - len(wd['removed_beats'])

            if valid_peak_count > (row_size / 25) / 2:
                peaks = wd['peaklist']
                fake_peaks = wd['removed_beats']
                real_peaks = [peak for peak in peaks if peak not in fake_peaks]

                if len(real_peaks) > 0:
                    sum_peak = sum(valid_data[idx] for idx in real_peaks)
                    avg_peak_value = sum_peak / len(real_peaks)
                    is_usable = True

        except Exception as e:
            print(f"세그먼트 {i} 처리 실패: {str(e)}")
            is_usable = False

        usable_or_rejected.append(1 if is_usable else 0)
        avg_peak_values.append(avg_peak_value)

        # 임계값 기반 레이블링
        if i < half_chunks:
            true_labels.append(0)  # 앞부분은 0
        else:
            true_labels.append(1 if avg_peak_value > threshold_value else 0)  # 뒷부분은 임계값에 따라

        # CNN 예측
        if not is_usable:
            cnn_predictions.append(-1)  # 사용 불가능한 청크
        else:
            segment_data = np.array(valid_data[:row_size])
            segment_data = segment_data.reshape(1, row_size, 1)
            cnn_pred = model.predict(segment_data, verbose=0)[0][0]
            cnn_label = 1 if cnn_pred > 0.5 else 0
            cnn_predictions.append(cnn_label)

        test_segments.append(valid_data[:row_size])
        i += 1

    return test_segments, usable_or_rejected, cnn_predictions, true_labels, avg_peak_values


def visualize_segments(segments_list, category_name, title, viz_dir, max_to_show=50):
    """
    하나의 카테고리에 대한 모든 세그먼트를 하나의 그래프에 겹쳐서 시각화
    """
    if not segments_list:
        print(f"{category_name} 카테고리에 표시할 세그먼트가 없습니다.")
        return

    # 총 세그먼트 수
    total_segments = len(segments_list)

    # 시각화에 사용할 세그먼트 수 제한
    n_to_show = min(total_segments, max_to_show)
    segments_to_show = segments_list[:n_to_show]

    plt.figure(figsize=(14, 8))

    # 각 세그먼트 겹쳐 그리기
    x_values = np.arange(500)  # 모든 세그먼트는 500 포인트

    for segment, file_name in segments_to_show:
        plt.plot(x_values, segment, alpha=0.3)

    # 평균 파형 계산 및 그리기
    if total_segments > 0:
        avg_segment = np.mean([s[0] for s in segments_list], axis=0)
        plt.plot(x_values, avg_segment, color='red', linewidth=2, label='평균 파형')

    plt.title(f"{title} (총 {total_segments}개 중 {n_to_show}개 표시)")
    plt.xlabel('Sample Index (0-499)')
    plt.ylabel('PPG Value')
    plt.ylim(-30000, 30000)  # y축 범위 설정
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = os.path.join(viz_dir, f"all_{category_name}_segments.png")
    plt.savefig(output_path)
    plt.close()

    print(f"{category_name} 파형 시각화 저장 완료: {output_path}")

    # 세그먼트별 파일 출처 파이 차트
    if total_segments > 0:
        visualize_segment_sources(segments_list, category_name, viz_dir)


def visualize_segment_sources(segments_list, category_name, viz_dir):
    """
    세그먼트의 출처 파일을 파이 차트로 시각화
    """
    # 파일별 세그먼트 수 집계
    source_counts = {}
    for segment, file_name in segments_list:
        if file_name in source_counts:
            source_counts[file_name] += 1
        else:
            source_counts[file_name] = 1

    # 파이 차트 그리기
    plt.figure(figsize=(10, 8))

    labels = list(source_counts.keys())
    sizes = list(source_counts.values())

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(f'{category_name} 세그먼트 출처 파일 분포')

    output_path = os.path.join(viz_dir, f"{category_name}_sources_pie.png")
    plt.savefig(output_path)
    plt.close()

    print(f"{category_name} 출처 분포 시각화 저장 완료: {output_path}")


def visualize_average_comparison(correct_1_segments, false_neg_segments, false_pos_segments, viz_dir):
    """
    세 카테고리의 평균 파형을 하나의 그래프에 비교
    """
    plt.figure(figsize=(14, 8))

    x_values = np.arange(500)

    # 각 카테고리의 평균 파형 계산 및 그리기
    if correct_1_segments:
        avg_correct_1 = np.mean([s[0] for s in correct_1_segments], axis=0)
        plt.plot(x_values, avg_correct_1, color='green', linewidth=2,
                 label=f'정확히 1로 분류 (n={len(correct_1_segments)})')

    if false_neg_segments:
        avg_false_neg = np.mean([s[0] for s in false_neg_segments], axis=0)
        plt.plot(x_values, avg_false_neg, color='red', linewidth=2,
                 label=f'실제 1인데 0으로 오분류 (n={len(false_neg_segments)})')

    if false_pos_segments:
        avg_false_pos = np.mean([s[0] for s in false_pos_segments], axis=0)
        plt.plot(x_values, avg_false_pos, color='blue', linewidth=2,
                 label=f'실제 0인데 1로 오분류 (n={len(false_pos_segments)})')

    plt.title('분류 결과별 평균 PPG 파형 비교')
    plt.xlabel('Sample Index (0-499)')
    plt.ylabel('PPG Value')
    plt.ylim(-30000, 30000)  # y축 범위 설정
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = os.path.join(viz_dir, "classification_average_comparison.png")
    plt.savefig(output_path)
    plt.close()

    print(f"평균 파형 비교 시각화 저장 완료: {output_path}")


if __name__ == "__main__":
    file_paths = [
        r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\test_data\jh_left\1681815686266\ppg_green.txt",
        r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\test_data\jh_right\1681261790949\ppg_green.txt",
        r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\test_data\m1_left\1675931125936\ppg_green.txt",
        r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\test_data\m1_right\1681822657751\ppg_green.txt",
        r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\test_data\m2_left\1681269276864\ppg_green.txt",
        r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\test_data\m2_right\1675932659426\ppg_green.txt",
        r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\test_data\m3_left\1675932257870\ppg_green.txt",
        r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\test_data\m3_right\1681823785425\ppg_green.txt",
        r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\test_data\m4_left\1675933819438\ppg_green.txt",
        r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\test_data\m4_right\1681270427012\ppg_green.txt",
        r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\test_data\w1_left\1681824836513\ppg_green.txt",
        r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\test_data\w1_right\1675933307027\ppg_green.txt",
        r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\test_data\w2_left\1675934782377\ppg_green.txt",
        r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\test_data\w2_right\1681271399482\ppg_green.txt"
    ]

    threshold_csv_path = r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\HeartGPT-main\Transformer_Generating_and_1D-CNN_Classification_Model\threshold_value_1.csv"
    model_path = r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\processed_data\best_emotion_model.h5"
    viz_dir = r"D:\pythonProject\Emotion-Classification-2-Transformer-Application-master\Emotion-Classification-2-Transformer-Application-master\Dataset\processed_data\test_visualization"

    threshold_value = np.loadtxt(threshold_csv_path)

    visualize_classification_results(file_paths, threshold_value, model_path, viz_dir)