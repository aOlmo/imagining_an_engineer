#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras import applications
from keras.models import Model

from keras.preprocessing.image import load_img



def evaluate_model(model, weights_file):
    import PIL.Image as image
    
    P_NP = ["P", "NP"]
    test_names = ["female", "male", "white", "non-white"]
    results = {}
    n_rounds = 1

    if test_all_together:
	    for pnp in P_NP:
		    folder = os.getcwd()+"/data/test/{}/all/".format(pnp)
		    x = np.array([np.array(image.open(folder+fname)) for fname in os.listdir(folder)])
		    model.load_weights('models/{}.h5'.format(weights_file))
		    n_test = x.shape[0]
		    n_P_NP = np.count_nonzero(np.trunc(model.predict(x)))
		    print("[+]: Accuracies for {}: {}".format(pnp, n_P_NP/n_test))
	    return

    if not test_all_together:
	    for i in range(n_rounds):
		    print("ROUND {}".format(i+1))
		    for pnp in P_NP:
			    for test_name in test_names:
				    exp = "{}-{}".format(pnp, test_name)
				    if i == 0:
					    results[exp]=0
				    folder = os.getcwd()+"/data/test/{}/{}/".format(pnp, test_name)
				    x = np.array([np.array(image.open(folder+fname)) for fname in os.listdir(folder)])
				    model.load_weights('models/{}.h5'.format(weights_file))
				    n_test = x.shape[0]
				    n_P_NP = np.count_nonzero(np.trunc(model.predict(x)))
				    results[exp] += n_P_NP/n_test
			    print("[{}-{}] acc {}".format(pnp, test_name, n_P_NP/n_test))
		    print("")

    print("FINAL ACCURACIES")
    print("----------------")
    for res in results:
	    results[res] /= n_rounds
	    print("{}: {}".format(res, results[res]))
    exit()


test = 1
test_all_together = 0
test_weights = "p_np_3000"

if not test:
	save_weights_file = test_weights
	epochs = 25
	train_samples = 3000*2 

# dimensions of our images.
img_width, img_height = 64, 64

train_data_dir = 'data/train'

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 64
#val_batch_size = 2

# automatically retrieve images and their classes for train and validation sets
if not test:
	train_generator = datagen.flow_from_directory(
			train_data_dir,
			target_size=(img_width, img_height),
			batch_size=batch_size,
			class_mode='binary')

#validation_generator = datagen.flow_from_directory(
#        validation_data_dir,
#        target_size=(img_width, img_height),
#        batch_size=val_batch_size,
#        class_mode='binary')

# a simple stack of 3 convolution layers with a ReLU activation and followed by max-pooling layers.
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

if test: evaluate_model(model, test_weights)

#validation_samples = 1000+1000

if not test:
	model.fit_generator(
			train_generator,
			steps_per_epoch=train_samples // batch_size,
			epochs=epochs)
        #validation_data=validation_generator,
        #validation_steps=validation_samples// val_batch_size,)
#About 60 seconds an epoch when using CPU

	model.save_weights("models/{}.h5".format(save_weights_file))
#model.evaluate_generator(validation_generator, validation_samples)


# By applying random transformation to our train set, we artificially enhance our dataset with new unseen images.
# This will hopefully reduce overfitting and allows better generalization capability for our network.

#train_datagen_augmented = ImageDataGenerator(
#       rescale=1./255,        # normalize pixel values to [0,1]
#       shear_range=0.2,       # randomly applies shearing transformation
#        zoom_range=0.2,        # randomly applies shearing transformation
#        horizontal_flip=True)  # randomly flip the images

# same code as before
#train_generator_augmented = train_datagen_augmented.flow_from_directory(
#        train_data_dir,
#        target_size=(img_width, img_height),
#        batch_size=batch_size,
#        class_mode='binary')

#model.fit_generator(
#        train_generator_augmented,
#        steps_per_epoch=train_samples // batch_size,
#        epochs=epochs)
        #validation_data=validation_generator,
        #validation_steps=validation_samples // val_batch_size,)


#model.save_weights('models/thirds_25.h5')

#### Evaluating on validation set
#score = model.evaluate_generator(validation_generator, validation_samples)
#print("Accuracy = {}".format(score))
