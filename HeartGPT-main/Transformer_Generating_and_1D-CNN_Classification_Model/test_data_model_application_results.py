import numpy as np
import pandas as pd
import heartpy as hp
import matplotlib.pyplot as plt
import scipy
import os
from keras.models import load_model

block_size = 500


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
    predicted_labels = []
    true_labels = []
    avg_peak_values = []
    cnn_predictions = []

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

            if valid_peak_count > (row_size / 25) / 2 and not np.max(wd['hr']) > 15000 and not np.min(
                    wd['hr']) < -15000:
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

        segment_data = np.array(valid_data[:row_size])
        segment_data = segment_data.reshape(1, row_size, 1)
        cnn_pred = model.predict(segment_data, verbose=0)[0][0]
        cnn_label = 1 if cnn_pred > 0.5 else 0
        cnn_predictions.append(cnn_label)

        if not is_usable:
            predicted_labels.append(-1)
        else:
            predicted_labels.append(1 if avg_peak_value > threshold_value else 0)

        if i < half_chunks:
            true_labels.append(0)
        else:
            true_labels.append(1 if avg_peak_value > threshold_value else 0)

        test_segments.append(valid_data[:row_size])
        i += 1

    return test_segments, usable_or_rejected, predicted_labels, true_labels, avg_peak_values, cnn_predictions


def evaluate_file(file_path, threshold_value, model):
    test_segments, usable_or_rejected, threshold_predictions, true_labels, avg_peak_values, cnn_predictions = reshape_vector_to_matrix_test(
        file_path, threshold_value, model)

    total_chunks = len(test_segments)
    rejected_chunks = threshold_predictions.count(-1)
    rejected_percent = (rejected_chunks / total_chunks) * 100 if total_chunks > 0 else 0

    usable_indices = [i for i, val in enumerate(threshold_predictions) if val != -1]

    correct_threshold_predictions = sum(1 for i in usable_indices if threshold_predictions[i] == true_labels[i])
    threshold_accuracy = (correct_threshold_predictions / len(usable_indices)) * 100 if len(usable_indices) > 0 else 0

    correct_cnn_predictions = sum(1 for i in usable_indices if cnn_predictions[i] == true_labels[i])
    cnn_accuracy = (correct_cnn_predictions / len(usable_indices)) * 100 if len(usable_indices) > 0 else 0

    print("정답:", " ".join(map(str, true_labels)))
    print("임계값 예측:", " ".join(map(str, threshold_predictions)))
    print("CNN 예측:", " ".join(map(str, cnn_predictions)))

    return {
        'total_chunks': total_chunks,
        'rejected_chunks': rejected_chunks,
        'rejected_percent': rejected_percent,
        'usable_chunks': len(usable_indices),
        'threshold_correct': correct_threshold_predictions,
        'threshold_accuracy': threshold_accuracy,
        'cnn_correct': correct_cnn_predictions,
        'cnn_accuracy': cnn_accuracy,
        'threshold_predictions': threshold_predictions,
        'cnn_predictions': cnn_predictions,
        'true_labels': true_labels,
        'peak_values': avg_peak_values
    }


def process_ppg_files(file_paths, threshold_value, model_path):
    model = load_model(model_path)
    overall_results = []

    for file_counter, file_path in enumerate(file_paths):
        file_name = os.path.basename(os.path.dirname(file_path)) + '/' + os.path.basename(
            os.path.dirname(os.path.dirname(file_path)))
        print(f"\n처리 중인 파일 {file_counter + 1}: {file_name}")

        try:
            results = evaluate_file(file_path, threshold_value, model)
            results['file_path'] = file_path
            results['file_name'] = file_name
            overall_results.append(results)

            print(f"총 청크 수: {results['total_chunks']}")
            print(f"사용 불가능한 청크: {results['rejected_chunks']} ({results['rejected_percent']:.2f}%)")
            print(f"사용 가능한 청크: {results['usable_chunks']}")
            print(f"임계값 기반 정확도: {results['threshold_accuracy']:.2f}%")
            print(f"CNN 모델 정확도: {results['cnn_accuracy']:.2f}%")
            print("")

        except Exception as e:
            print(f"파일 처리 중 오류 발생: {str(e)}")

    print("\n===== 전체 결과 요약 =====")
    total_usable_chunks = sum(r['usable_chunks'] for r in overall_results)
    total_threshold_correct = sum(r['threshold_correct'] for r in overall_results)
    total_threshold_accuracy = (total_threshold_correct / total_usable_chunks * 100) if total_usable_chunks > 0 else 0
    total_cnn_correct = sum(r['cnn_correct'] for r in overall_results)
    total_cnn_accuracy = (total_cnn_correct / total_usable_chunks * 100) if total_usable_chunks > 0 else 0
    total_chunks = sum(r['total_chunks'] for r in overall_results)
    total_rejected = sum(r['rejected_chunks'] for r in overall_results)
    total_rejected_percent = (total_rejected / total_chunks * 100) if total_chunks > 0 else 0

    print(f"전체 청크 수: {total_chunks}")
    print(f"사용 불가능한 청크: {total_rejected} ({total_rejected_percent:.2f}%)")
    print(f"사용 가능한 청크: {total_usable_chunks}")
    print(f"임계값 기반 전체 정확도: {total_threshold_accuracy:.2f}%")
    print(f"CNN 모델 전체 정확도: {total_cnn_accuracy:.2f}%")

    return overall_results

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

threshold_value = np.loadtxt(threshold_csv_path)


results = process_ppg_files(file_paths, threshold_value, model_path)

print("\n프로세스 완료!")