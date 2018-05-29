import pandas as pd
import numpy as np
from multiprocessing import Pool
from audio_processors import zero_pad


def get_data():
	df = pd.read_csv('data/csvs/sounds_orig.csv')
	fnames = df['fname']
	labels = df['label']
	return fnames,labels

def process_wav(wav_name):
	zero_pad(wav_name)

if __name__ == '__main__':
	fnames,labels = get_data()
	pool = Pool(4)
	pool.map(process_wav,fnames)