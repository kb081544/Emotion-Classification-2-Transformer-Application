import numpy as np
import heartpy as hp
import matplotlib.pyplot as plt
file_dir_negative = r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\train_data\concatenated_ppg_data_positive.txt"
data_load_negative = np.loadtxt(file_dir_negative)

'''
positive의 경우 threshold 값 없이 전부 사용
'''

def reshape_vector_to_matrix(vector, row_size=500):
    n = len(vector)
    stride = row_size // 1

    valid_segments = []
    avg_peak_array = []

    i = 0
    while i * stride + row_size <= n:
        start_idx = i * stride
        end_idx = min(start_idx + row_size, n)
        window_data = vector[start_idx:end_idx]

        if len(window_data) < row_size:
            window_data = np.pad(window_data, (0, row_size - len(window_data)),
                                 mode='constant', constant_values=np.nan)

        valid_data = window_data[~np.isnan(window_data)]
        valid_data = hp.filter_signal(valid_data, cutoff=[0.5, 8],
                                      sample_rate=25, order=3, filtertype="bandpass")
        sum_peak = 0
        try:
            wd, m = hp.process(valid_data, sample_rate=25)
            if not np.max(wd['hr']) > 15000 or np.min(wd['hr']) < -15000:
                if (len(wd['peaklist']) - len(wd['removed_beats'])) > (row_size / 25) / 2:
                    # At least 30bpm condition
                    fake_index = []
                    peaks = wd['peaklist']
                    fake_peaks = wd['removed_beats']
                    fake_index.extend(fake_peaks)
                    real_peaks = [item for item in peaks if item not in fake_peaks]

                    sum_peak = 0
                    for index in real_peaks:
                        sum_peak = sum_peak + valid_data[index]

                    if len(real_peaks) > 0:
                        avg_peak_value = sum_peak / len(real_peaks)
                        avg_peak_array.append(avg_peak_value)
                        valid_segments.append(valid_data[:row_size])

        except Exception as e:
            print(f"세그먼트 {i} 처리 실패: {str(e)}")
        i += 1

    avg_peak_array = np.array(avg_peak_array)
    valid_segments = np.array(valid_segments)

    return valid_segments, avg_peak_array


positive_segments, positive_peak_array = reshape_vector_to_matrix(data_load_negative, 500)
neg_mean = np.mean(positive_peak_array)
pos_std = np.std(positive_peak_array)
threshold = 0

print(f"Negative Peak Array 평균: {neg_mean}")
print(f"Negative Peak Array 표준편차: {pos_std}")

low_peak_indices = np.where(positive_peak_array > threshold)[0]
print(f"조건에 맞는 청크 수: {len(low_peak_indices)} / {len(positive_peak_array)}")

selected_segments = positive_segments[low_peak_indices]
selected_peak_values = positive_peak_array[low_peak_indices]

if len(selected_segments) > 0:
    output_path = r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\train_data\train_positive_no_threshold.csv"

    selected_segments_array = np.array(selected_segments)

    np.savetxt(output_path, selected_segments_array, delimiter=',')

    print(f"CSV 파일이 성공적으로 저장됨: {output_path}")
    print(f"저장된 청크 수: {len(selected_segments)}")
    print(f"각 청크 길이: {selected_segments[0].shape[0]}")

else:
    print("조건에 맞는 청크가 없음.")