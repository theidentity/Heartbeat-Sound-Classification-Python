import pandas as pd
import numpy as np
from multiprocessing import Pool
from audio_processors import zero_pad
import feature_extraction


def get_data():
	df = pd.read_csv('data/csvs/sounds_orig.csv')
	fnames = df['fname'].values
	labels = df['label']
	return fnames,labels

def process_wav(wav_name):
	# zero_pad(wav_name)
	# features = feature_extraction.get_all_features(wav_name)
	# features = feature_extraction.get_raw_audio(wav_name)
	features = feature_extraction.get_stft(wav_name)

	return features


if __name__ == '__main__':
	fnames,labels = get_data()
	print len(labels)

	pool = Pool(4)
	features = pool.map(process_wav,fnames[:])
	features = np.array(features)
	features = features.reshape(-1,1025,1211)
	np.save('npy/stft_features.npy',features)

	# features = np.load('npy/features.npy')
	print features.shape
	# for name in fnames[:10]:
	# 	process_wav(name)