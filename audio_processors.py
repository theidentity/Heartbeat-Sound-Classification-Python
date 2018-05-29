import librosa
import pandas as pd
import numpy as np
from multiprocessing import Pool
import librosa

def zero_pad(wav_name):
	print wav_name

	in_path = 'data/sounds/original/'+wav_name
	y,sr = librosa.load(in_path,sr=22050,mono=True)
	
	max_y = 620000
	y = librosa.util.fix_length(y,size=max_y)
	
	out_path = 'data/sounds/zero_padded/'+wav_name
	librosa.output.write_wav(out_path,y,sr)
