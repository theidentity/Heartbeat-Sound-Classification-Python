# http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/



import librosa
import numpy as np


def get_all_features(wav_name):

	print wav_name 
	in_path = 'data/sounds/zero_padded/'+wav_name
	y,sr = librosa.load(in_path,sr=22050,mono=True)
	
	stft = librosa.stft(y)
	stft = np.abs(stft)

	mfcc = librosa.feature.mfcc(y,sr,n_mfcc=40)
	mfcc = np.mean(mfcc,axis=1)

	chroma = librosa.feature.chroma_stft(S=stft,sr=sr)
	chroma = np.mean(chroma,axis=1)

	mel = librosa.feature.melspectrogram(y,sr)
	mel = np.mean(mel,axis=1)

	contrast = librosa.feature.spectral_contrast(y,sr)
	contrast = np.mean(contrast,axis=1)

	tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y),sr=sr)
	tonnetz = np.mean(tonnetz,axis=1)

	features = np.hstack([mfcc,chroma,mel,contrast,tonnetz])
	
	return features

def get_raw_audio(wav_name):
	print wav_name 
	in_path = 'data/sounds/zero_padded/'+wav_name
	y,sr = librosa.load(in_path,sr=22050,mono=True)
	
	return y.flatten()

def get_stft(wav_name):

	print wav_name 
	in_path = 'data/sounds/zero_padded/'+wav_name
	y,sr = librosa.load(in_path,sr=22050,mono=True)

	stft = librosa.stft(y)
	# real = np.real(stft)
	# imag = np.imag(stft)
	# stft = np.dstack([stft,imag]).astype(np.float32)
	stft = np.abs(stft)

	print stft.shape
	return stft


def get_wavform(wav_name):
	pass