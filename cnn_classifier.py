"""
train a deep network(cnn) for image classification
"""
from keras.backend.tensorflow_backend import get_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import set_session

from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.utils import to_categorical

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions

import tensorflow
import numpy as np
import image
import gc
import os


def img_2_tensor(img_path):
    """
    loads the image and convert in returns the tensor of shape
    (n, 512, 512, channels) = (1, 512, 512, 3)
    """
    img = load_img(img_path, target_size=(512, 512))
    # expanding the dimension and return
    return np.expand_dims(img_to_array(img), axis=0)


defects_img_paths = ['dataset/defects/' +
                     f for f in os.listdir('dataset/defects/')]
healthy_img_paths = ['dataset/healthy/' +
                     f for f in os.listdir('dataset/healthy/')]

healthy_imgs = len(defects_img_paths)
defects_imgs = len(healthy_img_paths)

IMAGE_PATHS = defects_img_paths + healthy_img_paths

# create labels
labels = []
for i in range(healthy_imgs):
    labels.append('0')
for i in range(defects_imgs):
    labels.append('1')

imgs_tensors = [img_2_tensor(img_path) for img_path in IMAGE_PATHS]
# squashing the list to array of shape (nos_images, l, w, channels)
X_train = np.vstack(imgs_tensors)
y_train = to_categorical(labels)


def reset_keras_tf_session():
    """
    this function clears the gpu memory and set the 
    tf session to not use the whole gpu
    """
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    config = tensorflow.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tensorflow.Session(config=config))


reset_keras_tf_session()

# model creation
model = Sequential()
model.add(Conv2D(64, kernel_size=5, activation='relu', input_shape=(512, 512, 3)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,  epochs=3)
