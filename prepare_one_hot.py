import pandas as pd
import numpy as np


def get_data():
	df = pd.read_csv('data/csvs/sounds_orig.csv')
	fnames = df['fname']
	labels = df['label']
	return fnames,labels

def to_one_hot(labels):

	unique_items = np.unique(labels)
	rows = len(labels)
	cols = len(unique_items) 
	one_hot = np.zeros(shape=(rows,cols))
	
	for i,item in enumerate(unique_items):
		idx = labels == item
		one_hot[idx,i] = 1

	return one_hot

if __name__ == '__main__':
		
	fnames,labels = get_data()
	one_hot = to_one_hot(labels)

	print one_hot
	np.save('npy/one_hot.npy',one_hot)
