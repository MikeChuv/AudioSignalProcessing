import re
import os
import librosa
import sklearn
import numpy as np
import matplotlib.pyplot as plt



class FSDD:

	def __init__(self, path):

		self.general_path = path
		self.speakers = self.listSpeakers(self.general_path)


	def getFilesForDigit(self, currentDigit : int, spkA : int, spkB : int):
		self.filesA = [f'{self.general_path}/{currentDigit}_{self.speakers[spkA]}_{i}.wav' for i in range(50)]
		self.filesB = [f'{self.general_path}/{currentDigit}_{self.speakers[spkB]}_{i}.wav' for i in range(50)]
		return self.filesA, self.filesB

	def listSpeakers(self, path):
		# <digit>_<speaker>_<counter>.wav
		# "([0-9])_([a-z]+)_(\d+)"gm
		rule = re.compile(r"([0-9])_([a-z]+)_(\d+)")
		speakers = set()
		for el in list(os.listdir(path)):
			m = rule.search(el)
			speakers.add(m.group(2))
		return list(speakers)



def getMFCC(soundfile, useStd=False):
	y, sr = librosa.load(soundfile)
	# n_fft = 2048 by default hop_length=512
	mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, dct_type=3, n_fft=4096, hop_length=512)
	if useStd: mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
	return mfccs, sr


def getTwoMFCCs(fileA, fileB, useStd=False):
	mfccsA, srA = getMFCC(fileA, useStd)
	mfccsB, srB = getMFCC(fileB, useStd)
	return mfccsA, srA, mfccsB, srB


def showMFCC(soundfile):
	y, sr = librosa.load(soundfile)
	S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
									fmax=8000)


	fig, ax = plt.subplots(ncols=2, figsize=(15, 5), sharex=True)
	img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
								x_axis='time', y_axis='mel', fmax=8000,
								ax=ax[0])

	fig.colorbar(img, ax=[ax[0]])
	ax[0].set(title='Mel spectrogram')
	ax[0].label_outer()

	mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, dct_type=3)
	img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])

	fig.colorbar(img, ax=[ax[1]])
	ax[1].set(title='MFCC')


def compareFeatures(featureA, srA, featureB, srB, titleA, titleB):
	fig, ax = plt.subplots(ncols=2, figsize=(15, 5))

	img = librosa.display.specshow(featureA, sr=srA, x_axis='time', cmap = 'bwr', ax=ax[0]);
	fig.colorbar(img, ax=[ax[0]])
	ax[0].set(title=titleA)
	ax[0].label_outer()

	img = librosa.display.specshow(featureB, sr=srB, x_axis='time', cmap = 'bwr', ax=ax[1]);
	fig.colorbar(img, ax=[ax[1]])
	ax[1].set(title=titleB)
	return fig, ax