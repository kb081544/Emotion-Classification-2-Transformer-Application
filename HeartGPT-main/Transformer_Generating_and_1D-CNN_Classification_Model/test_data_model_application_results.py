import numpy as np
import pandas as pd
import heartpy as hp
import matplotlib.pyplot as plt
import scipy
import os
import matplotlib.pyplot as plt
from keras.models import load_model

block_size = 500
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

def reshape_vector_to_matrix_test(file_dir, threshold_value, model, row_size=block_size):
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


def evaluate_file(file_path, threshold_value, model, viz_dir=None, file_name=None):
    test_segments, usable_or_rejected, cnn_predictions, true_labels, avg_peak_values = reshape_vector_to_matrix_test(
        file_path, threshold_value, model)

    total_chunks = len(test_segments)
    rejected_chunks = usable_or_rejected.count(0)
    rejected_percent = (rejected_chunks / total_chunks) * 100 if total_chunks > 0 else 0

    usable_indices = [i for i, val in enumerate(usable_or_rejected) if val == 1]

    correct_cnn_predictions = sum(1 for i in usable_indices
                                  if cnn_predictions[i] == true_labels[i])
    cnn_accuracy = (correct_cnn_predictions / len(usable_indices)) * 100 if len(usable_indices) > 0 else 0

    true_0_indices = [i for i in usable_indices if true_labels[i] == 0]
    true_1_indices = [i for i in usable_indices if true_labels[i] == 1]

    correct_cnn_0 = sum(1 for i in true_0_indices if cnn_predictions[i] == 0)
    cnn_0_accuracy = (correct_cnn_0 / len(true_0_indices)) * 100 if len(true_0_indices) > 0 else 0

    correct_cnn_1 = sum(1 for i in true_1_indices if cnn_predictions[i] == 1)
    cnn_1_accuracy = (correct_cnn_1 / len(true_1_indices)) * 100 if len(true_1_indices) > 0 else 0

    # 결과 시각화
    if viz_dir and file_name:
        visualize_results(test_segments, true_labels, cnn_predictions, usable_or_rejected,
                          viz_dir, file_name, cnn_accuracy, cnn_0_accuracy, cnn_1_accuracy,
                          len(true_0_indices), len(true_1_indices), rejected_percent)

    # 청크 개수 출력
    print("정답:", " ".join(map(str, true_labels)))
    print("CNN 예측:", " ".join(map(str, cnn_predictions)))

    return {
        'total_chunks': total_chunks,
        'rejected_chunks': rejected_chunks,
        'rejected_percent': rejected_percent,
        'usable_chunks': len(usable_indices),
        'cnn_correct': correct_cnn_predictions,
        'cnn_accuracy': cnn_accuracy,
        'cnn_predictions': cnn_predictions,
        'true_labels': true_labels,
        'peak_values': avg_peak_values,
        'usable_or_rejected': usable_or_rejected,
        'true_0_count': len(true_0_indices),
        'true_1_count': len(true_1_indices),
        'cnn_0_accuracy': cnn_0_accuracy,
        'cnn_1_accuracy': cnn_1_accuracy,
        'correct_cnn_0': correct_cnn_0,
        'correct_cnn_1': correct_cnn_1
    }


def visualize_results(segments, true_labels, predictions, usable_or_rejected, viz_dir, file_name,
                      accuracy, label0_acc, label1_acc, label0_count, label1_count, rejected_percent):
    # 시각화 디렉토리 생성
    os.makedirs(viz_dir, exist_ok=True)

    # 각 청크별 예측 결과 시각화
    plt.figure(figsize=(14, 10))

    # 1. 예측 결과 vs 실제 레이블 비교
    plt.subplot(2, 2, 1)
    chunk_indices = list(range(len(true_labels)))

    # 사용 가능한 청크와 사용 불가능한 청크 분리
    usable_indices = [i for i, val in enumerate(usable_or_rejected) if val == 1]
    unusable_indices = [i for i, val in enumerate(usable_or_rejected) if val == 0]

    # 사용 가능한 청크 중 정확한 예측과 오분류 분리
    correct_indices = [i for i in usable_indices if predictions[i] == true_labels[i]]
    incorrect_indices = [i for i in usable_indices if predictions[i] != true_labels[i] and predictions[i] != -1]

    # 그래프 그리기
    plt.scatter([i for i in chunk_indices], true_labels, label='실제 label', marker='o', color='blue', alpha=0.6)
    plt.scatter([i for i in usable_indices], [predictions[i] for i in usable_indices],
                label='CNN 예측', marker='x', color='green', alpha=0.7)
    plt.scatter(unusable_indices, [-0.2] * len(unusable_indices), label='사용 불가능 청크', marker='|', color='red')

    plt.xlabel('청크 인덱스')
    plt.ylabel('label')
    plt.yticks([-0.2, 0, 1], ['사용불가', '0', '1'])
    plt.title(f'파일: {file_name} - 예측 결과')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 2. 성능 요약 파이 차트
    plt.subplot(2, 2, 2)
    labels = ['정확한 예측', '오분류', '사용 불가능']
    sizes = [len(correct_indices), len(incorrect_indices), len(unusable_indices)]
    colors = ['lightgreen', 'lightcoral', 'lightgray']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('청크 분석 요약')

    # 3. 정확도 바 차트
    plt.subplot(2, 2, 3)
    accuracy_labels = ['전체 Accuracy', 'Label 0 Accuracy', 'Label 1 Accuracy']
    accuracy_values = [accuracy, label0_acc, label1_acc]

    bars = plt.bar(accuracy_labels, accuracy_values, color=['blue', 'green', 'orange'])

    plt.xlabel('정확도 유형')
    plt.ylabel('Accuracy (%)')
    plt.title('CNN 모델 Accuracy')
    plt.ylim(0, 105)

    # 바 위에 정확도 텍스트 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom')

    # 4. 레이블 분포
    plt.subplot(2, 2, 4)
    distribution_labels = ['Label 0', 'Label 1', '사용 불가능']
    distribution_values = [label0_count, label1_count, len(unusable_indices)]

    dist_bars = plt.bar(distribution_labels, distribution_values, color=['lightblue', 'lightgreen', 'lightgray'])

    plt.xlabel('레이블 유형')
    plt.ylabel('청크 수')
    plt.title('청크 레이블 분포')

    # 바 위에 값 표시
    for bar in dist_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    # 전체 타이틀
    plt.suptitle(f'{file_name} 분석 결과 \n사용 불가능 청크 비율: {rejected_percent:.1f}%', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 저장
    output_path = os.path.join(viz_dir, f"{file_name}_analysis.png")
    plt.savefig(output_path)
    plt.close()

    print(f"시각화 저장 완료: {output_path}")


def process_ppg_files(file_paths, threshold_value, model_path, viz_dir):
    model = load_model(model_path)
    overall_results = []

    # 시각화 디렉토리 생성
    os.makedirs(viz_dir, exist_ok=True)

    for file_counter, file_path in enumerate(file_paths):
        file_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        print(f"\n처리 중인 파일 {file_counter + 1}: {file_name}")

        try:
            results = evaluate_file(file_path, threshold_value, model, viz_dir, file_name)
            results['file_path'] = file_path
            results['file_name'] = file_name
            overall_results.append(results)

            print(f"총 청크 수: {results['total_chunks']}")
            print(f"사용 불가능한 청크(-1): {results['rejected_chunks']} ({results['rejected_percent']:.2f}%)")
            print(f"사용 가능한 청크: {results['usable_chunks']}")
            print(f"  - 레이블 0 청크: {results['true_0_count']}")
            print(f"  - 레이블 1 청크: {results['true_1_count']}")
            print(f"CNN 모델 정확도: {results['cnn_accuracy']:.2f}%")
            print(f"  - 레이블 0 정확도: {results['cnn_0_accuracy']:.2f}%")
            print(f"  - 레이블 1 정확도: {results['cnn_1_accuracy']:.2f}%")
            print("")

        except Exception as e:
            print(f"파일 처리 중 오류 발생: {str(e)}")

    # 전체 결과 계산
    total_chunks = sum(r['total_chunks'] for r in overall_results)
    total_rejected = sum(r['rejected_chunks'] for r in overall_results)
    total_rejected_percent = (total_rejected / total_chunks * 100) if total_chunks > 0 else 0
    total_usable_chunks = sum(r['usable_chunks'] for r in overall_results)

    total_true_0 = sum(r['true_0_count'] for r in overall_results)
    total_true_1 = sum(r['true_1_count'] for r in overall_results)

    total_cnn_correct = sum(r['cnn_correct'] for r in overall_results)
    total_cnn_accuracy = (total_cnn_correct / total_usable_chunks * 100) if total_usable_chunks > 0 else 0

    total_correct_cnn_0 = sum(r['correct_cnn_0'] for r in overall_results)
    total_cnn_0_accuracy = (total_correct_cnn_0 / total_true_0 * 100) if total_true_0 > 0 else 0

    total_correct_cnn_1 = sum(r['correct_cnn_1'] for r in overall_results)
    total_cnn_1_accuracy = (total_correct_cnn_1 / total_true_1 * 100) if total_true_1 > 0 else 0

    # 전체 결과 출력
    print("\n===== 전체 결과 =====")
    print(f"전체 청크 수: {total_chunks}")
    print(f"사용 불가능한 청크(-1): {total_rejected} ({total_rejected_percent:.2f}%)")
    print(f"사용 가능한 청크: {total_usable_chunks}")
    print(f"  - 레이블 0 청크: {total_true_0}")
    print(f"  - 레이블 1 청크: {total_true_1}")
    print(f"CNN 모델 전체 정확도: {total_cnn_accuracy:.2f}%")
    print(f"  - 레이블 0 정확도: {total_cnn_0_accuracy:.2f}%")
    print(f"  - 레이블 1 정확도: {total_cnn_1_accuracy:.2f}%")

    # 전체 결과 시각화
    create_overall_visualization(overall_results, viz_dir, total_cnn_accuracy,
                                 total_cnn_0_accuracy, total_cnn_1_accuracy,
                                 total_rejected_percent)

    return overall_results


def create_overall_visualization(results, viz_dir, overall_accuracy, label0_acc, label1_acc, rejected_percent):
    plt.figure(figsize=(16, 12))

    # 1. 각 파일별 정확도 비교
    plt.subplot(2, 2, 1)
    file_names = [r['file_name'] for r in results]
    accuracies = [r['cnn_accuracy'] for r in results]

    y_pos = range(len(file_names))
    bars = plt.barh(y_pos, accuracies, color='skyblue')
    plt.yticks(y_pos, file_names)
    plt.xlabel('Accuracy (%)')
    plt.title('파일별 CNN 모델 Accuracy')
    plt.xlim(0, 105)
    plt.grid(True, alpha=0.3)

    # 막대 끝에 정확도 표시
    for i, v in enumerate(accuracies):
        plt.text(v + 1, i, f"{v:.1f}%", va='center')

    # 2. 레이블 0과 1에 대한 정확도 비교
    plt.subplot(2, 2, 2)
    label0_accs = [r['cnn_0_accuracy'] for r in results]
    label1_accs = [r['cnn_1_accuracy'] for r in results]

    x = range(len(file_names))
    width = 0.35

    bars1 = plt.bar(x, label0_accs, width, label='Label 0 Accuracy', color='lightblue')
    bars2 = plt.bar([i + width for i in x], label1_accs, width, label='Label 1 Accuracy', color='lightgreen')

    plt.xlabel('파일')
    plt.ylabel('Accuracy (%)')
    plt.title('파일별 Label 0과 1에 대한 CNN Accuracy')
    plt.xticks([i + width / 2 for i in x], [f"{i + 1}" for i in range(len(file_names))])
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. 사용 불가능한 청크 비율
    plt.subplot(2, 2, 3)
    rejected_percentages = [r['rejected_percent'] for r in results]

    plt.bar(range(len(file_names)), rejected_percentages, color='salmon')
    plt.xlabel('파일')
    plt.ylabel('사용 불가능한 청크 비율 (%)')
    plt.title('파일별 사용 불가능한 청크 비율')
    plt.xticks(range(len(file_names)), [f"{i + 1}" for i in range(len(file_names))])
    plt.grid(True, alpha=0.3)

    # 4. 전체 요약 파이 차트와 정확도
    plt.subplot(2, 2, 4)

    # 파이 차트 (사용 가능/불가능 청크 비율)
    total_usable = sum(r['usable_chunks'] for r in results)
    total_rejected = sum(r['rejected_chunks'] for r in results)

    plt.pie([total_usable, total_rejected],
            labels=['사용 가능한 청크', '사용 불가능한 청크'],
            colors=['lightgreen', 'lightcoral'],
            autopct='%1.1f%%',
            startangle=90)
    plt.axis('equal')
    plt.title('전체 청크 분포')

    # 전체 타이틀
    plt.suptitle(
        f'PPG 데이터 분석 전체 요약\n전체 정확도: {overall_accuracy:.1f}% (Label 0: {label0_acc:.1f}%, Label 1: {label1_acc:.1f}%)',
        fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 저장
    output_path = os.path.join(viz_dir, "overall_results.png")
    plt.savefig(output_path)
    plt.close()

    print(f"전체 요약 시각화 저장 : {output_path}")

    # 파일 번호 매핑 표 생성
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    plt.text(0.1, 0.95, "파일 번호 매핑", fontsize=14, fontweight='bold')

    for i, r in enumerate(results):
        plt.text(0.1, 0.9 - i * 0.05, f"파일 {i + 1}: {r['file_name']}", fontsize=10)

    output_mapping_path = os.path.join(viz_dir, "file_mapping.png")
    plt.savefig(output_mapping_path)
    plt.close()

    print(f"파일 매핑 저장 완료: {output_mapping_path}")


file_paths = [
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\test_data\jh_left\1681815686266\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\test_data\jh_right\1681261790949\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\test_data\m1_left\1675931125936\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\test_data\m1_right\1681822657751\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\test_data\m2_left\1681269276864\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\test_data\m2_right\1675932659426\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\test_data\m3_left\1675932257870\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\test_data\m3_right\1681823785425\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\test_data\m4_left\1675933819438\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\test_data\m4_right\1681270427012\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\test_data\w1_left\1681824836513\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\test_data\w1_right\1675933307027\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\test_data\w2_left\1675934782377\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\test_data\w2_right\1681271399482\ppg_green.txt"
]

threshold_csv_path = r"C:\Users\user\PycharmProjects\Emotion Classification 3\HeartGPT-main\Transformer_Generating_and_1D-CNN_Classification_Model\threshold_value_1.csv"
model_path = r"C:\Users\user\PycharmProjects\Emotion Classification 3\processed_data\best_emotion_model.h5"
viz_dir = r"C:\Users\user\PycharmProjects\Emotion Classification 3\processed_data\test_visualization"

threshold_value = np.loadtxt(threshold_csv_path)

results = process_ppg_files(file_paths, threshold_value, model_path, viz_dir)