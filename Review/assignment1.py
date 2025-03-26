import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np

chunk_size=300
overlap=30

file=open("assignment_1_raw_data.txt", 'r')
file_data=[]
lines = file.readlines()
for line in lines[15:-1]:
   values = line.strip().split()
   second_int = int(values[1])
   file_data.append(second_int)
np.savetxt("second_array.csv", file_data,delimiter=',')

x_chunk=[]
sum = 0
cnt = 0
exc = 0
x_result = None
pk_list = []
y_result = []

for i in range(0, len(file_data), chunk_size - overlap): # 270씩 띄어서 인덱스, 자르는건 300
   x_chunk = file_data[i:i + chunk_size] # 300씩 자름
   filtered = hp.filter_signal(x_chunk, [0.5, 8], sample_rate=25, order=3,
                               filtertype='bandpass')
   # plt.plot(filtered)
   # plt.savefig(f'{i}번째 filtered.png')
   # plt.close()

   try:  # chunk의 전처리 후 process 함수가 실행이 안되는 경우가 꽤나 많아, 에러 표시로 인한 코드 실행 중지를 방지하기 위해 try except 문으로 구현
      wd, m = hp.process(filtered, sample_rate=25)
      # hp.plotter(wd, m)
      # plt.savefig(f"{i}번째 process.png")
      # plt.close()
      if (len(wd['peaklist']) != 0): # 청크의 peak의 개수가 0이 아니라면, 즉 그나마 피크가 좀 있다면
         sum += (len(wd['peaklist']) - len(wd['removed_beats']))  # 초록색 피크들의 총 합 계산(평균을 내기 위해)
         temp = wd['hr']  # bandpass를 통과한 chunk, 즉 300개의 신호
         temp_pk = (len(wd['peaklist']) - len(wd['removed_beats']))  # 초록색 피크의 개수
         if (cnt == 0):
            x_result = temp # 첫 번째 청크는 x_result에 바로 집어넣음
         else:
            x_result = np.vstack([x_result, temp])
            # 두 번째 청크 부터는 stack으로 쌓아 올림
            # x_result는 모든 전처리된 300의 신호의 모음

      else:
         exc += 1 # 예외 처리 된 청크의 개수 증가
         temp_pk = 0 # 피크 개수는 0개
         temp = wd['hr']
         if (cnt == 0):
            x_result = temp
         else:
            x_result = np.concatenate((x_result, temp))

      cnt += 1 # 청크의 개수 증가
      pk_list.append(temp_pk) # 초록색 피크 개수의 리스트
   except:
      print("예외처리")
      continue

# 전처리 후
pk_np = np.array(pk_list)
avg = sum / cnt
x_new_result = []
new_cnt = 0

for j in range(cnt):
   if pk_np[j] > avg: # 초록색 피크 개수의 리스트. 초록색 피크의 개수가 평균보다 크면
      x_new_result.append(x_result[j]) # 평균보다 큰 x 청크가 새 배열에 append
      new_cnt += 1
   else:
      continue

print(x_new_result)

peak_shapes = []
fake_index = []
# cutoff_n = 40000
# index = np.where(np.max(x_new_result, axis=1) >= cutoff_n)[0]
# new_data = np.delete(x_new_result, index, axis=0)
#
# fake_index.extend(index)
# l = len(index)
x_new_result=np.array(x_new_result)
for i in range(x_new_result.shape[0]):
   temp = x_new_result[i, :]
   wd, m = hp.process(temp, sample_rate=25)

   peaks = wd['peaklist']
   fake_peaks = wd['removed_beats']
   fake_index.extend(fake_peaks)
   real_peaks = [item for item in peaks if item not in fake_peaks]
   for index in real_peaks:
      if not ((index - 13 < 0) or (index + 14 >= x_new_result.shape[1])):
         peak_shape = temp[index - 13:index + 14]
         plt.plot(peak_shape)
         peak_shapes.append(peak_shape)


np_peak = np.array(peak_shapes)
print(np_peak.shape)
np.savetxt("assignment_1_peak_data.csv", np_peak, delimiter=',')
plt.show()