import numpy as np
import pandas as pd
from tensorflow import estimator, data, feature_column

# from tensorflow.keras import utils
#
# TRAIN_DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
# TEST_DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
# train_file_path = utils.get_file("iris_training.csv", TRAIN_DATA_URL)
# test_file_path = utils.get_file("iris_test.csv", TEST_DATA_URL)
# print("downloaded path : ", train_path, test_path)

# train_path = utils.get_file("iris_training.csv", TRAIN_DATA_URL)
# test_path = utils.get_file("iris_test.csv", TEST_DATA_URL)

data_path = "../../../data"
train_file_path = "%s/iris/iris_training.csv" % data_path
test_file_path = "%s/iris/iris_test.csv" % data_path

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train = pd.read_csv(train_file_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_file_path, names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')


def input_evaluation_set():
    features = {'SepalLength': np.array([6.4, 5.0]),
                'SepalWidth': np.array([2.8, 2.3]),
                'PetalLength': np.array([5.6, 3.3]),
                'PetalWidth': np.array([2.2, 1.0])}
    labels = np.array([2, 1])
    return features, labels


def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # Convert the inputs to a Dataset.
    dataset = data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(feature_column.numeric_column(key=key))

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)

# Train the Model.
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Generate predictions from the model
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}


def input_fn(features, batch_size=256):
    """An input function for prediction."""
    # Convert the inputs to a Dataset without labels.
    return data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


predictions = classifier.predict(
    input_fn=lambda: input_fn(predict_x))

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        SPECIES[class_id], 100 * probability, expec))
