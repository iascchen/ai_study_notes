import tensorflow as tf
import tensorflowjs as tfjs

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#######################
# load model
#######################
base_path = "../../output"

h5_path = "%s/hello_mnist_1.h5" % base_path
loaded_model = tf.keras.models.load_model(h5_path)
loaded_model.summary()

#######################
# convert to tensorflow.js
#######################

tfjs_path = "%s/hello_mnist_5.tfjs" % base_path
tfjs.converters.save_keras_model(loaded_model, tfjs_path)

#######################
# convert to tensorflow lite
#######################

# TODO: return error in tf v1.14.0

tflite_path = "%s/hello_mnist_5.tflite" % base_path
converter = tf.lite.TFLiteConverter.from_keras_model_file(h5_path)
tflite_model = converter.convert()
with open(tflite_path, "wb") as fw:
    fw.write(tflite_model)
