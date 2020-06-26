import tensorflowjs as tfjs
from tensorflow import lite
from tensorflow.keras import datasets, models

mnist = datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#######################
# load model
#######################
base_path = "../../../output"

h5_path = "%s/hello_mnist_1.h5" % base_path
loaded_model = models.load_model(h5_path)
loaded_model.summary()

#######################
# convert to tensorflow lite
#######################

tflite_path = "%s/hello_mnist_5.tflite" % base_path

converter = lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()
with open(tflite_path, "wb") as fw:
    fw.write(tflite_model)

#######################
# convert to tensorflow.js
#######################

tfjs_path = "%s/hello_mnist_5.tfjs" % base_path
tfjs.converters.save_keras_model(loaded_model, tfjs_path)
