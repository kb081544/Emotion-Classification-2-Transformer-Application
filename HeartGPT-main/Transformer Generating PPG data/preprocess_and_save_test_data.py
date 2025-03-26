import numpy as np
import pandas as pd
import heartpy as hp
import matplotlib.pyplot as plt
import scipy
import os
block_size=500

'''
3000이나 평균 피크가 특정 이상이면 활용
'''
def normalize_peak_heights(signal):
    peaks, _ = scipy.signal.find_peaks(signal)

    if len(peaks) == 0:
        return signal

    peak_heights = signal[peaks]
    median_height = np.median(peak_heights)

    normalized_signal = signal.copy()
    for peak in peaks:
        current_height = signal[peak]
        if current_height > median_height:
            window_start = max(0, peak - 5)  # 조정 가능한 윈도우 크기
            window_end = min(len(signal), peak + 6)
            window = normalized_signal[window_start:window_end]
            scaling_factor = median_height / current_height
            normalized_signal[window_start:window_end] = window * scaling_factor

    return normalized_signal


def limit_amplitude_variation(signal):
    peaks, _ = scipy.signal.find_peaks(signal, distance=10)
    valleys, _ = scipy.signal.find_peaks(-signal, distance=10)

    if len(peaks) == 0 or len(valleys) == 0:
        return signal

    modified_signal = signal.copy()

    peak_heights = signal[peaks]
    valley_depths = signal[valleys]

    peak_mean = np.mean(peak_heights)
    valley_mean = np.mean(valley_depths)

    desired_peak_height = peak_mean
    desired_valley_depth = valley_mean

    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]

        segment = modified_signal[start_idx:end_idx]

        seg_max = np.max(segment)
        seg_min = np.min(segment)

        if seg_max != seg_min:
            normalized_segment = (segment - seg_min) / (seg_max - seg_min)
            scaled_segment = normalized_segment * (desired_peak_height - desired_valley_depth) + desired_valley_depth
            modified_signal[start_idx:end_idx] = scaled_segment

    return modified_signal


def process_ppg_signal(valid_data):

    boundary_size = 20
    valid_data[:boundary_size] = np.mean(valid_data[boundary_size:boundary_size * 2])
    valid_data[-boundary_size:] = np.mean(valid_data[-boundary_size * 2:-boundary_size])

    processed_signal = limit_amplitude_variation(valid_data)

    min_val = np.min(processed_signal)
    max_val = np.max(processed_signal)
    normalized_signal = ((processed_signal - min_val) / (max_val - min_val)) * 100

    return normalized_signal


def save_matrix_to_txt(matrix, filename):
    try:
        np.savetxt(filename, matrix, fmt='%d', delimiter=',')
        print(f"Successfully saved matrix to {filename}")
        print(f"Matrix shape: {matrix.shape}")
    except Exception as e:
        print(f"Error saving matrix: {str(e)}")


def reshape_vector_to_matrix(vector, row_size=block_size):
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
                if (len(wd['peaklist']) - len(wd['removed_beats'])) > (block_size / 25) / 2:
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

    # if not valid_segments:
    #     return None
    # valid_segments=np.array(valid_segments)
    # num_valid_segments = len(valid_segments)
    # matrix = np.zeros((num_valid_segments, row_size))
    #
    # for i, segment in enumerate(valid_segments):
    #     if len(segment) < row_size:
    #         segment = np.pad(segment, (0, row_size - len(segment)),
    #                          mode='constant', constant_values=0)
    #     matrix[i, :] = segment
    #
    # matrix = np.round(matrix).astype(np.int64)
    # matrix = np.clip(matrix, 0, 100)
    # 결과 저장
    avg_peak_array=np.array(avg_peak_array)
    valid_segments=np.array(valid_segments)
    # output_filename = f"filtered_processed_blocksize_{block_size}_strides_{stride}_ppg_data_limit_amplitude_normalized.txt"
    # save_matrix_to_txt(valid_segments, output_filename)

    return valid_segments,avg_peak_array
    # return matrix

file_dir=r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\concated_ppg_data.txt"
file_dir_positive=r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\concated_ppg_data_positive.txt"
file_dir_negative=r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\concated_ppg_data_negative.txt"

'''
max, min 값을 모든 데이터 안에서 확인하고 
'''

# data_load=np.loadtxt(file_dir)
# data_concat=reshape_vector_to_matrix(data_load, 500)
data_load_positive = np.loadtxt(file_dir_positive)
data_load_negative=np.loadtxt(file_dir_negative)
positive, positive_peak_array=reshape_vector_to_matrix(data_load_positive, 500)
negative, negative_peak_array =reshape_vector_to_matrix(data_load_negative, 500)



print("blue avg: ", np.mean(positive_peak_array))
print("red avg: ", np.mean(negative_peak_array))
print("blue var: ", np.var(positive_peak_array))
print("red var: ", np.var(negative_peak_array))
positive_mean = np.mean(positive_peak_array)
negative_mean = np.mean(negative_peak_array)
# plt.axhline(y=positive_mean, color='darkblue', linestyle='--', label='Positive Mean')
# plt.axhline(y=negative_mean, color='darkred', linestyle='--', label='Negative Mean')

plt.legend()
for j in range(len(negative)):
    plt.plot(negative[j,:], color="red")
    plt.axhline(y=np.mean(negative_peak_array[j,:]), color='darkred', linestyle='--', label='Negative Mean')

    plt.show()
for i in range(len(positive)):
    plt.plot(positive[i,:],  color="blue")
    plt.axhline(y=np.mean(positive[positive_mean[i,:]]), color='darkblue', linestyle='--', label='Positive Mean')
    plt.show()



def reshape_vector_to_matrix_test(file_dir, row_size=block_size, output_dir=None, file_counter=0):
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

        try:
            wd, m = hp.process(valid_data, sample_rate=25)
            print(len(wd['peaklist']) - len(wd['removed_beats']))
            if (len(wd['peaklist']) - len(wd['removed_beats'])) > (block_size / 25) / 2:
                processed_signal = process_ppg_signal(valid_data)
                test_segments.append(processed_signal[:row_size])
                usable_or_rejected.append(1)
            else:
                test_segments.append(processed_signal[:row_size])
                usable_or_rejected.append(0)

        except Exception as e:
            print(f"세그먼트 {i} 처리 실패: {str(e)}")
            processed_signal = process_ppg_signal(valid_data)
            test_segments.append(processed_signal[:row_size])
            usable_or_rejected.append(0)
        i += 1

    num_valid_segments = len(test_segments)
    matrix = np.zeros((num_valid_segments, row_size))

    for i, segment in enumerate(test_segments):
        if len(segment) < row_size:
            segment = np.pad(segment, (0, row_size - len(segment)),
                             mode='constant', constant_values=0)
        matrix[i, :] = segment

    matrix = np.round(matrix).astype(np.int64)
    matrix = np.clip(matrix, 0, 100)
    usable_or_rejected = np.array(usable_or_rejected)
    usable_or_rejected = usable_or_rejected.reshape(-1, 1)
    new_data = np.concatenate((usable_or_rejected, matrix), axis=1)

    # 결과 저장
    if output_dir is None:
        output_dir = os.path.dirname(file_dir)
    base_filename = os.path.splitext(os.path.basename(file_dir))[0]
    output_filename = os.path.join(output_dir,
                                   f"filtered_processed_blocksize_{block_size}_strides_{stride}_{file_counter}_ppg_data_limit_amplitude_test.txt")

    save_matrix_to_txt(new_data, output_filename)




def process_ppg_files(file_paths, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(file_paths[0]), 'processed')

    os.makedirs(output_dir, exist_ok=True)

    for file_counter, file_path in enumerate(file_paths):
        print(f"Processing file {file_counter + 1}: {file_path}")
        try:
            reshape_vector_to_matrix_test(file_path, output_dir=output_dir, file_counter=file_counter)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")


# List of file paths
# file_dir=r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\4.txt"
file_paths = [
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\test_data\jh_left\1681815686266\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\test_data\jh_right\1681261790949\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\test_data\m1_left\1675931125936\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\test_data\m1_right\1681822657751\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\test_data\m2_left\1681269276864\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\test_data\m2_right\1675932659426\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\test_data\m3_left\1675932257870\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\test_data\m3_right\1681823785425\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\test_data\m4_left\1675933819438\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\test_data\m4_right\1681270427012\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\test_data\w1_left\1681824836513\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\test_data\w1_right\1675933307027\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\test_data\w2_left\1675934782377\ppg_green.txt",
    r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\test_data\w2_right\1681271399482\ppg_green.txt"
]
# output_directory = r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\processed_test_data"
# process_ppg_files(file_paths, output_dir=output_directory)

# print("All files processed successfully!")