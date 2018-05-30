import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from keras.layers import Input
from keras.layers import Dense,Activation,Dropout
from keras.layers import Conv1D,MaxPool1D,Flatten
from keras.models import load_model,save_model,Sequential
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard



class AudioClassifer_NN(object):
	"""docstring for AudioClassiferNN"""
	def __init__(self,folds=4):

		self.name = 'Audio_NN'
		self.folds = folds
		
		self.seed = 42
		np.random.seed(self.seed)

		self.datagen = self.get_data()

		self.input_shape = (193,)
		self.num_classes = 4
		self.batch_size = 32

		self.save_path = 'models/'+self.name+'_h5.best'

	def get_data(self):
		features = np.load('npy/manual_features.npy')
		labels = np.load('npy/one_hot.npy')

		while True:
			folds = StratifiedKFold(n_splits=self.folds,random_state=self.seed)

			for train_idx,test_idx in folds.split(X=features,y=np.argmax(labels,axis=1)):
				
				train_X = features[train_idx]
				train_y = labels[train_idx]

				test_X = features[test_idx]
				test_y = labels[test_idx]

				yield (train_X,train_y),(test_X,test_y)

	def get_model(self):
		model = Sequential()
		model.add(Dense(2048,input_shape=self.input_shape,activation='relu'))
		model.add(Dense(2048,activation='relu'))
		model.add(Dense(2048,activation='relu'))
		model.add(Dense(2048,activation='relu'))
		model.add(Dense(self.num_classes,activation='sigmoid'))
		return model

	def build_model(self,lr):

		model = self.get_model()

		opt = Adam(lr)
		model.compile(
			loss='categorical_crossentropy',
			optimizer=opt,
			metrics=['accuracy']
			)
		return model

	def get_callbacks(self):
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, verbose=1, mode='auto')
		checkpointer = ModelCheckpoint(filepath=self.save_path, verbose=1, save_best_only=True)
		tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
		
		return [checkpointer]
		# return [checkpointer,tensorboard]
		# return [checkpointer,early_stopping,tensorboard]

	def train(self,lr,epochs):

		(train_X,train_y),(test_X,test_y) = self.datagen.next()

		model = self.build_model(lr)

		model.fit(
			x = train_X,
			y = train_y,
			batch_size = self.batch_size,
			epochs = epochs,
			validation_data = (test_X,test_y),
			callbacks = self.get_callbacks()
			)

	def train_on_folds(self,lr,epochs):

		for i in range(self.folds):
			print 'fold'+str(i)
			self.save_path = 'models/'+self.name+'_fold'+str(i)+'_best.h5'
			self.train(lr,epochs)

	def evaluate(self):

		(train_X,train_y),(test_X,test_y) = self.datagen.next()
		model = load_model(self.save_path)
		
		y_pred = model.predict(
			x = test_X,
			verbose = 1,
			batch_size = self.batch_size
			)
		
		y_true = test_y
		return y_pred,y_true

	def get_metrics(self):

		y_pred,y_true = self.evaluate()

		y_pred = np.argmax(y_pred,axis=1)
		y_true = np.argmax(y_true,axis=1)

		cm = confusion_matrix(y_true,y_pred)
		report = classification_report(y_true,y_pred)
		accuracy = accuracy_score(y_true,y_pred)

		print cm
		print report
		print 'Accuracy :',accuracy

		return accuracy

	def evaluate_on_folds(self):

		accuracies = []

		for i in range(self.folds):
			print 'fold'+str(i)
			self.save_path = 'models/'+self.name+'_fold'+str(i)+'_best.h5'
			accuracy = self.get_metrics()
			accuracies.append(accuracy)

		print 'overall accuracy:',np.mean(accuracies)

if __name__ == '__main__':
	clf = AudioClassifer_NN(folds=4)
	clf.train_on_folds(lr=1e-6,epochs=100)
	clf.evaluate_on_folds()