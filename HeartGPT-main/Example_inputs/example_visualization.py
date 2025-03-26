import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data=pd.read_csv("example_context_PPG.csv", header=None)
plt.plot(data)
plt.show()