import cv2
import csv
import numpy as np
import sklearn
import sys

samples = []
images = []
measurements = []
DATA_ROOT = 'data'
correction = 0 #float(sys.argv[1])
include_left_right = False
augment_image = False
print("with correction " + repr(correction))

with open(DATA_ROOT+'/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		next(reader)
		for line in reader:
			samples.append(line)

from sklearn.model_selection import train_test_split
from random import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def process_image(filename):
	name = DATA_ROOT + '/IMG/'+filename.split('/')[-1]
	bgr_image = cv2.imread(name)
	image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
	return image

def augment(filename):
	name = DATA_ROOT + '/IMG/'+filename.split('/')[-1]
	bgr_image = cv2.imread(name)
	alpha = 1+0.1*np.random.randint(1,5,size=1) #for contrast
	beta = 1.*np.random.randint(10,30,size=1)  # for brightness
	bgr_image = cv2.add(cv2.multiply(bgr_image,alpha), beta)
	image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
	return image

def generator(generator_samples, batch_size=32):
	num_samples = len(generator_samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(generator_samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = generator_samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				center_image = process_image(batch_sample[0])
				center_angle = float(batch_sample[3])
				images.extend([center_image, np.fliplr(center_image)])
				angles.extend([center_angle, -center_angle])
				
				if augment_image and center_angle != 0:
					augmented_image = augment(batch_sample[0])
					images.extend([augmented_image, np.fliplr(augmented_image)])
					angles.extend([center_angle, -center_angle])

				if include_left_right:
					left_image = process_image(batch_sample[1])
					right_image = process_image(batch_sample[2])				
					left_angle = center_angle + correction
					right_angle = center_angle - correction
					images.extend([left_image, np.fliplr(left_image), right_image, np.fliplr(right_image)])
					angles.extend([left_angle, -left_angle, right_angle, -right_angle])

		# trim image to only see section with road
		X_train = np.array(images)
		y_train = np.array(angles)
		yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print("Number of training samples=", len(train_samples))
print("Number of validation samples=",len(validation_samples))

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
import matplotlib.pyplot as plt

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66,320,3)))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)
model.save('model' + '.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
