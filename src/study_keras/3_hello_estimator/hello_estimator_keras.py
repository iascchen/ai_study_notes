# _*_ coding:utf-8 _*_
'''
Personal Information Seeker (PISeeker)

__createdOn__ = '2019-10-16'
__author__ = 'CHEN Hao'
__email__ = 'iascchen@gmail.com'
__weibo__ = '@问天谷'
__git__ = 'https://github.com/iascchen'
'''

import tensorflow_datasets as tfds
from tensorflow.keras import models, layers, estimator

base_path = "../../../output/estimator"
output_model = "%s/estimator" % base_path

model = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=(4,)),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()


def input_fn():
    split = tfds.Split.TRAIN
    dataset = tfds.load('iris', split=split, as_supervised=True)
    dataset = dataset.map(lambda features, labels: ({'dense_input': features}, labels))
    dataset = dataset.batch(32).repeat()
    return dataset


for features_batch, labels_batch in input_fn().take(1):
    print(features_batch)
    print(labels_batch)

keras_estimator = estimator.model_to_estimator(keras_model=model, model_dir=output_model)

keras_estimator.train(input_fn=input_fn, steps=25)
eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10)
print('Eval result: {}'.format(eval_result))
