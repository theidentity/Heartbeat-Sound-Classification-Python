import os


def create_folder(path,gitignore):
	if not os.path.exists(path):
		print path
		os.makedirs(path)

	if gitignore!=None:
		file = open(path+'.gitignore','w')
		file.write(gitignore)
		file.close()


def create_skeleton():

	gitignore = '*.wav \n *.jpg \n *.npy \n *.h5 \n'

	paths = [
		'data/',
		'data/sounds/original/',
		'data/sounds/zero_padded/',
		'data/images/mel/',
	]

	for path in paths:
		create_folder(path,gitignore)

if __name__ == '__main__':
	create_skeleton()