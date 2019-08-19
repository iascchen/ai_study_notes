import os

import tensorflow_hub as hub
from tensorflow import keras

base_path = "../../data"
output_path = "../../output"
models_path = "../../models"
images_dir = "%s/pets/images" % base_path

os.environ['TFHUB_CACHE_DIR'] = models_path
print("TFHUB_CACHE_DIR", os.environ.get('TFHUB_CACHE_DIR'))

classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/3"

IMAGE_SHAPE = (224, 224)

# #####################
# # Hub in Keras
# #####################

# TODO will return failure, maybe it is only for tensorflow 2.0
classifier_layers = hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE + (3,))

classifier = keras.Sequential([
    classifier_layers
])
classifier.summary()
