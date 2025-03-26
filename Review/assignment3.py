import numpy as np
import heartpy as hp
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import pickle
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, Dropout, MaxPool1D, Flatten
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint
import keras
import datetime
import glob

chunk_size = 300
overlap = 30
EPOCH = 3


# train data 압축 풀고, directory에 맞게 수정
file_positive_green = glob.glob(r"C:\Users\user\PycharmProjects\Emotion Classification 2\P\green\*.txt")
file_negative_green = glob.glob(r"C:\Users\user\PycharmProjects\Emotion Classification 2\N\green\*.txt")

# test data 압축 풀고, directory에 맞게 수정
h1 = glob.glob("C:/Users/user/PycharmProjects/Emotion Classification 2/plotting_dataset/h1.txt")
g1 = glob.glob("C:/Users/user/PycharmProjects/Emotion Classification 2/plotting_dataset/g1.txt")


def read_txt_files_with_skip(data, y):
    data_list = []
    for file_path in data:
        print(f"Reading file: {file_path}")
        file_data = []
        file_data.append(y)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines[15:-1]:
                values = line.strip().split()
                second_int = int(values[1])
                file_data.append(second_int)
        data_list.append(file_data)
    return data_list


def chunk_data_hp(data, tot):
    data_y = [y[0] for y in data]
    data_x = [x[1:] for x in data]
    sum_removed = 0
    y_result = []
    x_new_result = []
    y_new_result = []
    pk_list = []
    sum = 0
    cnt = 0
    exc = 0
    x_result = None
    for sublist, label in zip(data_x, data_y):
        for i in range(0, len(sublist) - chunk_size + 1, chunk_size - overlap):
            x_chunk = sublist[i:i + chunk_size]
            filtered = hp.filter_signal(x_chunk, [0.5, 8], sample_rate=25, order=3,
                                        filtertype='bandpass')
            try:
                wd, m = hp.process(filtered, sample_rate=25)
                if (len(wd['peaklist']) != 0):
                    sum += (len(wd['peaklist']) - len(wd['removed_beats']))
                    sum_removed += len(wd['removed_beats'])
                    temp = wd['hr']
                    temp_pk = (len(wd['peaklist']) - len(wd['removed_beats']))
                    if (cnt == 0):
                        x_result = temp
                    else:
                        x_result = np.vstack([x_result, temp])
                else:
                    exc += 1
                    temp_pk = 0
                    temp = wd['hr']
                    if (cnt == 0):
                        x_result = temp
                    else:
                        x_result = np.concatenate((x_result, temp))
                cnt += 1
                pk_list.append(temp_pk)
                y_result.append(label)
            except:
                print(f"예외처리 {cnt}")
                continue
    pk_np = np.array(pk_list)
    avg = sum / cnt
    new_temp = 0
    new_cnt = 0
    if (tot == "train"):
        for j in range(cnt):
            if pk_np[j] > avg:
                x_new_result.append(x_result[j])
                y_new_result.append(y_result[j])
                new_temp += m['bpm']
                new_cnt += 1
            else:
                continue
        return x_new_result, y_new_result
    elif (tot == "test"):
        return x_result, y_result


def dividing_and_extracting(input_x=None, input_y=None):
    global x, y

    data_x = input_x if input_x is not None else x
    data_y = input_y if input_y is not None else y

    cutoff_n = 40000
    index = np.where(np.max(data_x, axis=1) >= cutoff_n)[0]
    new_data = np.delete(data_x, index, axis=0)
    new_data_y = np.delete(data_y, index, axis=0)
    peak_shapes = []
    fake_index = []

    fake_index.extend(index)
    for i in range(len(new_data)):
        temp = new_data[i, :]
        temp_y = new_data_y[i]
        wd, m = hp.process(temp, sample_rate=25)

        peaks = wd['peaklist']
        fake_peaks = wd['removed_beats']
        fake_index.extend(fake_peaks)
        real_peaks = [item for item in peaks if item not in fake_peaks]
        for index in real_peaks:
            if not ((index - 13 < 0) or (index + 14 >= new_data.shape[1])):
                peak_shape = temp[index - 13:index + 14]
                peak_shape = np.concatenate((np.array([temp_y]), peak_shape))
                peak_shapes.append(peak_shape)

    np_peak = np.array(peak_shapes)
    print(np_peak.shape)
    return np_peak


def GMM_model(tot, gmm_p=None, gmm_n=None, x_input=None, y_input=None):
    global lab0, lab1, m, n, x, y

    if tot == "train":
        data = dividing_and_extracting()
        print(np.shape(data))
        data0 = data[data[:, 0] == 0]
        data0 = data0[:, 1:]
        data1 = data[data[:, 0] == 1]
        data1 = data1[:, 1:]

        n_components = 2
        gmm_p = GaussianMixture(n_components=n_components, covariance_type='full')
        gmm_p.fit(data0)

        labels0 = gmm_p.predict(data0)
        outliers = data0[labels0 == 1]
        normals = data0[labels0 == 0]

        gmm_n = GaussianMixture(n_components=n_components, covariance_type='full')
        gmm_n.fit(data1)
        labels1 = gmm_n.predict(data1)
        outliers_n = data1[labels1 == 1]
        normals_n = data1[labels1 == 0]

        if np.mean(normals_n) > np.mean(outliers_n):
            spp1 = normals_n
            lab1 = 0
        else:
            spp1 = outliers_n
            lab1 = 1
        if np.mean(normals) < np.mean(outliers):
            spp0 = normals
            lab0 = 0
        else:
            spp0 = outliers
            lab0 = 1

        m = np.max(spp1)
        n = np.min(spp1)
        normalized_train = []
        for value in spp0:
            normalized_num = (value - n) / (m - n)
            normalized_train.append(normalized_num)
        normalized_train_n = []
        for value in spp1:
            normalized_num = (value - n) / (m - n)
            normalized_train_n.append(normalized_num)

        normalized_train = np.array(normalized_train)
        normalized_train_n = np.array(normalized_train_n)

        normals_y = np.zeros((normalized_train.shape[0], 1))
        g_x_p = np.concatenate((normals_y, normalized_train), axis=1)
        normals_n_y = np.ones((normalized_train_n.shape[0], 1))
        g_x_n = np.concatenate((normals_n_y, normalized_train_n), axis=1)

        data = np.concatenate((g_x_p, g_x_n))
        np.random.shuffle(data)
        data_x = data[:, 1:]
        data_y = data[:, 0]

        x_train_g, x_test_g, y_train_g, y_test_g = train_test_split(data_x, data_y, test_size=0.2)

        return x_train_g, x_test_g, y_train_g, y_test_g, gmm_p, gmm_n, lab0, lab1, m, n

    elif tot == "test":
        if gmm_p is None or gmm_n is None:
            raise ValueError("GMM 모델 제공 필요")

        if x_input is not None and y_input is not None:
            x = np.array(x_input)
            y = np.array(y_input)
        else:
            raise ValueError("테스트 데이터 제공 필요")

        data = dividing_and_extracting()
        d = np.array(data)
        dy = d[:, 0]
        dx = d[:, 1:]

        tst = []
        tst_y = []

        lb1 = gmm_n.predict(dx)
        lb2 = gmm_p.predict(dx)

        for i in range(len(lb1)):
            if lb1[i] != lab1 and lb2[i] != lab0:
                continue
            else:
                tst.append(dx[i])
                tst_y.append(dy[i])

        normalized = []
        for value in tst:
            normalized_num = (value - n) / (m - n)
            normalized.append(normalized_num)

        data_x = np.array(normalized)
        data_y = np.array(tst_y)

        assert len(data_x) == len(data_y), f"데이터 차원: x={len(data_x)}, y={len(data_y)}"

        return data_x, data_y


def CNN_Model(train, test, y_train, y_test):
    global model

    train = np.expand_dims(train, axis=2)
    test = np.expand_dims(test, axis=2)

    input_shape = (train.shape[1], 1)
    model = Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Flatten())
    model.add(Dense(8, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(Dense(4, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    checkpoint_callback = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, mode='min',
                                          verbose=1)
    history = model.fit(train, y_train, batch_size=8, epochs=EPOCH,
                        validation_data=(test, y_test), callbacks=[checkpoint_callback])
    predictions = model.predict(test)
    score = model.evaluate(test, y_test, verbose=1)
    pred_np = np.array(predictions)
    print("\nloss= ", score[0], "\n정답률: ", score[1])

    now = datetime.datetime.now()
    model_name = f'trained_per_peak_{int(score[1] * 100)}.h5'
    model.save(model_name)
    return history, predictions, score


x = None
y = None
lab0 = None
lab1 = None
m = None
n = None

print("학습 데이터 처리")
train_positive_list = read_txt_files_with_skip(data=file_positive_green, y=0)
train_negative_list = read_txt_files_with_skip(data=file_negative_green, y=1)
train_list = train_positive_list + train_negative_list
x_train, y_train = chunk_data_hp(train_list, tot="train")
x = np.array(x_train)
y = np.array(y_train)
x_train_g, x_test_g, y_train_g, y_test_g, gmm_p, gmm_n, lab0, lab1, m, n = GMM_model(tot="train")
history, predictions, score = CNN_Model(x_train_g, x_test_g, y_train_g, y_test_g)
print(f"Model accuracy: {score[1] * 100:.2f}%")

print("테스트 데이터 처리")
test_data_list = read_txt_files_with_skip(data=h1, y=1)
x_test, y_test = chunk_data_hp(test_data_list, tot="test")
x = np.array(x_test)
y = np.array(y_test)
x_test_processed, y_test_processed = GMM_model( tot="test", gmm_p=gmm_p, gmm_n=gmm_n, x_input=x_test, y_input=y_test)
test_predictions = model.predict(np.expand_dims(x_test_processed, axis=2))
test_score = model.evaluate(np.expand_dims(x_test_processed, axis=2), y_test_processed, verbose=1)
print(f"Test accuracy: {test_score[1]*100:.2f}%")