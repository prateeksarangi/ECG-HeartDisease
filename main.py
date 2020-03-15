import os
import pandas as pd
import wfdb
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys


df1 = pd.read_csv('/Users/ashwini/Downloads/heartdisease-data/training-a/REFERENCE.csv', delimiter=',')
df1.dataframeName = 'REFERENCE.csv'

nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')

for i in range(1, 10):
	spf = wave.open("/Users/ashwini/Downloads/heartdisease-data/training-a/a000"+str(i)+".wav", "r")
	signal = spf.readframes(-1)
	signal = np.fromstring(signal, "Int16")
	if spf.getnchannels() == 2:
	    print("Just mono files")
	    sys.exit(0)
	plt.figure(1)
	plt.title("Signal Wave...")
	plt.plot(signal)
	plt.show()