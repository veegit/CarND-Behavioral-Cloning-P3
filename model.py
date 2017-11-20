import cv2
import csv
import numpy as np
import sklearn
import sys

samples = []
images = []
measurements = []
DATA_ROOT = 'data'
correction = float(sys.argv[1])
print("with correction " + repr(correction))

with open(DATA_ROOT+'/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		next(reader)
		for line in reader:
			samples.append(line)

def no_generator(samples):
	for line in samples:
		source_path = line[0]
		#filename = source_path.split('/')[-1]
		bgr_image = cv2.imread(source_path)
		image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
		image_flipped = np.fliplr(image)
		images.append(image)
		images.append(image_flipped)
		measurement = float(line[3])
		measurement_flipped = -measurement
		measurements.append(measurement)
		measurements.append(measmeasurement_flipped)

	X_train = np.array(images)
	y_train = np.array(measurements)

from sklearn.model_selection import train_test_split
from random import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def process_image(filename):
	name = DATA_ROOT + '/IMG/'+filename.split('/')[-1]
	bgr_image = cv2.imread(name)
	image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
	return image

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				center_image = process_image(batch_sample[0])
				left_image = process_image(batch_sample[1])
				right_image = process_image(batch_sample[2])
				
				center_angle = float(batch_sample[3])
				left_angle = center_angle + correction
				right_angle = center_angle - correction
				
				images.extend([center_image, np.fliplr(center_image), left_image, np.fliplr(left_image), right_image, np.fliplr(right_image)])
				angles.extend([center_angle, -center_angle, left_angle, -left_angle, right_angle, -right_angle])

		# trim image to only see section with road
		X_train = np.array(images)
		y_train = np.array(angles)
		yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D
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
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)
model.save('model' + repr(correction)+'.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
