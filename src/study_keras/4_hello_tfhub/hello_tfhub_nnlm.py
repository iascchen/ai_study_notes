# _*_ coding:utf-8 _*_
'''
Personal Information Seeker (PISeeker)

__createdOn__ = '2019-10-16'
__author__ = 'CHEN Hao'
__email__ = 'iascchen@gmail.com'
__weibo__ = '@问天谷'
__git__ = 'https://github.com/iascchen'

https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb#scrollTo=_NUbzVeYkgcO

'''

import os
import hashlib
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import Sequential, layers

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

models_path = "../../../models"
os.environ['TFHUB_CACHE_DIR'] = models_path

train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"],
                                  batch_size=-1, as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))

# model_url = "https://tfhub.dev/google/tf2-preview/nnlm-zh-dim50-with-normalization/1"
# model_url = "https://tfhub.dev/google/tf2-preview/nnlm-zh-dim50/1"
model_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"
handle_hash = hashlib.sha1(model_url.encode("utf8")).hexdigest()
hub_layer = hub.KerasLayer(model_url, output_shape=[50], input_shape=[], dtype=tf.string, trainable=True)
# hub_layer(train_examples[:3])
embeddings = embed(["猫", "猫 和 狗"])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

x_val = train_examples[:10000]
partial_x_train = train_examples[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

