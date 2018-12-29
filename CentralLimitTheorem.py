import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import math

COL = 'Adj Close'

df = web.DataReader('IBM', 'yahoo', '1962-01-01', '2018-11-30')
#df = df[COL]
df = (df[COL] - df[COL].mean()) / df[COL].std(ddof=0)

n = 10
samples = [df.sample(n=n) for i in range(0, len(df.index)//n)]
sample_means = [sample.mean() for sample in samples]
print(len(sample_means))

# plot normal distribution
_samples = np.random.randn(len(df.index)//n)


# plot sample means
plt.hist(_samples, color='red', alpha=0.1)
plt.hist(sample_means, color='red', alpha=0.7)
plt.xlabel('Sample Means')
plt.ylabel('Frequency')
plt.title('Central Limit Theorem')
plt.show()



