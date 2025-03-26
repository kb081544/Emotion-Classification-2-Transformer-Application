import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

data=np.loadtxt("temp_data_2.txt", delimiter=',')
n_components = 2
gmm = GaussianMixture(n_components=n_components, covariance_type='full')
gmm.fit(data)
labels = gmm.predict(data)

outliers = data[labels == 1]
normals = data[labels == 0]

means = gmm.means_
covariances = gmm.covariances_

mean_normals = means[0]
mean_outliers = means[1]

cov_normals = covariances[0]
cov_outliers = covariances[1]

# 시각화
plt.figure(figsize=(10, 6))

plt.plot(mean_normals, label='Normals Mean', color='blue', markersize=5)
plt.plot(mean_outliers, label='Outliers Mean', color='red', markersize=5)

std_normals = np.sqrt(np.diagonal(cov_normals))
std_outliers = np.sqrt(np.diagonal(cov_outliers))

plt.fill_between(range(len(mean_normals)), mean_normals - std_normals, mean_normals + std_normals, color='blue', alpha=0.2)
plt.fill_between(range(len(mean_outliers)), mean_outliers - std_outliers, mean_outliers + std_outliers, color='red', alpha=0.2)

plt.title("Mean and Highlighted Covariance Range for Each Feature")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.ylim(0, 8500)
plt.legend()
plt.grid(True)

plt.show()

global lab0

# single pulse의 결과가 0 즉 normals를 spp0에 저장, 그에 맞는 lab0도 저장
# spp 및 lab은 어떤 클러스터를 선택했고 그것을 저장하는 변수임 매우 중요
# 여기서 조건은 클러스터의 mean값이 더 작은 것을 선택
# 제공한 데이터는 positive 데이터이므로 평균 값이 작은 것을 선택하는 것임
if np.mean(normals) < np.mean(outliers):
    spp0 = normals
    lab0 = 0
else:
    spp0 = outliers
    lab0 = 1

# 학습데이터 확보를 위한 normalization
global m
global n

m = np.max(spp0)
n = np.min(spp0)

normalized_train = []
for value in spp0:
    normalized_num = (value - n) / (m - n)
    normalized_train.append(normalized_num)
normalized_train = np.array(normalized_train)

# 선택된 클러스터의 시각화
mean_spp0 = np.mean(normalized_train, axis=0)
cov_spp0 = np.cov(normalized_train, rowvar=False)

std_spp0 = np.sqrt(np.diagonal(cov_spp0))

plt.figure(figsize=(10, 6))

plt.plot(mean_spp0, label='선택된 클러스터의 평균', color='blue', markersize=5)
plt.fill_between(range(len(mean_spp0)), mean_spp0 - std_spp0, mean_spp0 + std_spp0, color='blue', alpha=0.2)

plt.title("선택된 Single Pulses의 클러스터 시각화")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.legend()
plt.ylim(0, 9000)
plt.grid(True)

plt.show()
