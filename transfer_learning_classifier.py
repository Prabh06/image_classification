"""
--- Transfer Learning approach

Image classification using pretrained weights from inception model
and using transfer learning technique to train the classifier on 
smaller dataset
"""

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation, grid_search
import sklearn
import os
import tensorflow as tf
import numpy as np

# load the existing inception model to memory
MODEL_PATH = 'inception/tensorflow_inception_graph.pb'

defects_img_paths = ['dataset/defects/' +
                     f for f in os.listdir('dataset/defects/')]
healthy_img_paths = ['dataset/healthy/' +
                     f for f in os.listdir('dataset/healthy/')]

healthy_imgs = len(defects_img_paths)
defects_imgs = len(healthy_img_paths)

# create labels
labels = []
for i in range(healthy_imgs):
    labels.append('0')
for i in range(defects_imgs):
    labels.append('1')

IMAGE_PATHS = defects_img_paths + healthy_img_paths

# read the saved inception model amd load it to graph
with tf.gfile.FastGFile(MODEL_PATH, 'rb') as f:
    g = tf.GraphDef()
    g.ParseFromString(f.read())
    _ = tf.import_graph_def(g, name='')

# inception model returns a feature vector of size 2048
# creating a features array for all the total images
features = np.empty((len(IMAGE_PATHS), 2048))

with tf.Session() as sess:
    # our idea here is to take the feature vectors i.e. lower level layer from
    # the inception network and use these features to train the classifier
    tensor_max_pool = sess.graph.get_tensor_by_name('pool_3:0')

    for i, image_path in enumerate(IMAGE_PATHS):
        # read the image data tf gfile returns the bytes from the image
        img_data = tf.gfile.FastGFile(image_path, 'rb').read()
        feature = sess.run(tensor_max_pool, {
            'DecodeJpeg/contents:0': img_data
        })
        features[i, :] = np.squeeze(feature)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    features, labels, test_size=0.2)

# we are givind rbf to work with multiple classes and series for param
# C and gamma and using
clf = grid_search.GridSearchCV(svm.SVC(probability=True), [{"kernel": ["rbf"], "C": [
                               1, 10, 100, 1000], "gamma": [1e-2, 1e-3, 1e-4, 1e-5]}], cv=10)

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

print(confusion_matrix(y_test, y_predict, labels=sorted(list(set(labels)))))
