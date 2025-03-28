import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
def GMM_model(data):
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

    global lab1, lab0, m, n

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

    temp_data=np.concatenate((spp0, spp1))

    m = np.max(temp_data)
    n = np.min(temp_data)

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

    np.savetxt("assignment2.txt", data, delimiter=',')

    # 추가
    gmm_info=[]
    gmm_info.append(lab0)
    gmm_info.append(lab1)
    gmm_info.append(m)
    gmm_info.append(n)

    with open("gmm_info.pickle", "wb") as fw:
        pickle.dump(gmm_info, fw)

    # 각 군집의 평균과 공분산 추출
    means0 = gmm_p.means_
    covariances0 = gmm_p.covariances_



    means1 = gmm_n.means_
    covariances1 = gmm_n.covariances_

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(means0[0], label='Positive - Normals Mean', color='blue', markersize=5)
    ax.plot(means0[1], label='Positive - Outliers Mean', color='red', markersize=5)

    ax.plot(means1[0], label='Negative - Normals Mean', color='green', markersize=5)
    ax.plot(means1[1], label='Negative - Outliers Mean', color='orange',markersize=5)

    std0_1 = np.sqrt(np.diagonal(covariances0[0]))
    std0_2 = np.sqrt(np.diagonal(covariances0[1]))

    ax.fill_between(range(len(means0[0])), means0[0] - std0_1, means0[0] + std0_1, color='blue', alpha=0.2)
    ax.fill_between(range(len(means0[1])), means0[1] - std0_2, means0[1] + std0_2, color='red', alpha=0.2)

    std1_1 = np.sqrt(np.diagonal(covariances1[0]))
    std1_2 = np.sqrt(np.diagonal(covariances1[1]))

    ax.fill_between(range(len(means1[0])), means1[0] - std1_1, means1[0] + std1_1, color='green', alpha=0.2)
    ax.fill_between(range(len(means1[1])), means1[1] - std1_2, means1[1] + std1_2, color='orange', alpha=0.2)

    ax.set_title("Positive vs Negative 데이터의 평균 및 공분산")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.set_ylim(0, 9000)
    ax.grid(True)

    plt.show()

    '''
    한 figure에 두 그래프 플랏
    '''
    fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
    fig.suptitle("Positive vs Negative 데이터의 평균 및 공분산")

    # Positive
    ax[0].plot(means0[0], label='Positive - Normals Mean', color='blue',markersize=5)
    ax[0].plot(means0[1], label='Positive - Outliers Mean', color='red', markersize=5)

    std0_1 = np.sqrt(np.diagonal(covariances0[0]))
    std0_2 = np.sqrt(np.diagonal(covariances0[1]))

    ax[0].fill_between(range(len(means0[0])), means0[0] - std0_1, means0[0] + std0_1, color='blue', alpha=0.2)
    ax[0].fill_between(range(len(means0[1])), means0[1] - std0_2, means0[1] + std0_2, color='red', alpha=0.2)

    ax[0].set_title("Positive Data")
    ax[0].set_xlabel("Feature Index")
    ax[0].set_ylabel("Value")
    ax[0].legend()
    ax[0].set_ylim(0, 9000)
    ax[0].grid(True)

    # Negative
    ax[1].plot(means1[0], label='Negative - Normals Mean', color='green', markersize=5)
    ax[1].plot(means1[1], label='Negative - Outliers Mean', color='orange', markersize=5)

    std1_1 = np.sqrt(np.diagonal(covariances1[0]))
    std1_2 = np.sqrt(np.diagonal(covariances1[1]))

    ax[1].fill_between(range(len(means1[0])), means1[0] - std1_1, means1[0] + std1_1, color='green', alpha=0.2)
    ax[1].fill_between(range(len(means1[1])), means1[1] - std1_2, means1[1] + std1_2, color='orange', alpha=0.2)

    ax[1].set_title("Negative Data ")
    ax[1].set_xlabel("Feature Index")
    ax[1].set_ylabel("Value")
    ax[1].legend()
    ax[1].set_ylim(0, 9000)
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()
    # 추가
    data_x = data[:, 1:]
    data_y = data[:, 0]
    x_train_g, x_test_g, y_train_g, y_test_g = train_test_split(data_x, data_y, test_size=0.2)
    return x_train_g, x_test_g, y_train_g, y_test_g, gmm_p, gmm_n, lab0, lab1, m, n

# 수정 및 추가
data = np.genfromtxt("peak_train.txt", delimiter=' ')
x_train, x_test, y_train, y_test, gmm_p, gmm_n, lab0, lab1, m, n = GMM_model(data)
