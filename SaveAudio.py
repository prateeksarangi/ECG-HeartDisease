from wave import open as open_wave
import numpy
import pandas as pd


dimension = numpy.zeros(410)
for i in range(1, 10):
	waveFile = open_wave("/Users/ashwini/Downloads/heartdisease-data/training-a/a000" + str(i) + ".wav",'rb')
	nframes = waveFile.getnframes()
	wavFrames = waveFile.readframes(nframes)
	ys = numpy.fromstring(wavFrames, dtype=numpy.int16)
	dimension[i] = ys.shape[0]

for i in range(10, 100):
	waveFile = open_wave("/Users/ashwini/Downloads/heartdisease-data/training-a/a00" + str(i) + ".wav",'rb')
	nframes = waveFile.getnframes()
	wavFrames = waveFile.readframes(nframes)
	ys = numpy.fromstring(wavFrames, dtype=numpy.int16)
	dimension[i] = ys.shape[0]


for i in range(100, 410):
	waveFile = open_wave("/Users/ashwini/Downloads/heartdisease-data/training-a/a0" + str(i) + ".wav",'rb')
	nframes = waveFile.getnframes()
	wavFrames = waveFile.readframes(nframes)
	ys = numpy.fromstring(wavFrames, dtype=numpy.int16)
	dimension[i] = ys.shape[0]

df = pd.read_csv("MapReduce.csv", header = 0, sep = '\t')
x = numpy.array(df.iloc[:,0])

p = 0


'''
ys = numpy.concatenate([ys1,ys2])
ys = ys.reshape(2, 71611)

print(ys.shape)

df = pd.DataFrame(ys,index=ys[:,0])

print(df.shape)

df.to_csv("audiofile1.csv", sep='\t', encoding='utf-8', index=True)
'''