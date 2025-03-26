import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np

# read
file=open("temp_data.txt", 'r')
file_data=[]
lines = file.readlines()
for line in lines[15:-1]:
   values = line.strip().split()
   second_int = int(values[1])
   file_data.append(second_int)


# filter_signal
filtered = hp.filter_signal(file_data, [0.5, 8], sample_rate=25, order=3,filtertype='bandpass')
print(filtered)
plt.plot(filtered)
plt.show()

# process, plotter
wd, m = hp.process(filtered, sample_rate=25)
hp.plotter(wd,m)
plt.show()

# dictionary
peaks = wd['peaklist']
fake_peaks = wd['removed_beats']

# dividing and extracting
fake_index = []
fake_index.extend(fake_peaks)
real_peaks = [item for item in peaks if item not in fake_peaks]
print(real_peaks)

peak_shapes = []
for index in real_peaks:
    if not ((index - 13 < 0) or (index + 14 >= len(filtered))):
        peak_shape = filtered[index - 13:index + 14]
        peak_shapes.append(peak_shape)
        plt.plot(peak_shape)
print(peak_shapes)
plt.show()