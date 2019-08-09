import time

import numpy as np
# from tensorflow.keras.applications import densenet
# from tensorflow.keras.applications import inception_v3
# from tensorflow.keras.applications import mobilenet
# from tensorflow.keras.applications import mobilenet_v2
# from tensorflow.keras.applications import nasnet
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.applications import imagenet_utils

base_path = "../../data"
output_path = "../../output"
images_dir = "%s/images" % base_path


# process an image to be model friendly
def process_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    # img = image.load_img(img_path, target_size=(299, 299))  # inception_v3 only

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # pImg = densenet.preprocess_input(img_array)
    # pImg = inception_v3.preprocess_input(img_array)
    # pImg = mobilenet.preprocess_input(img_array)
    # pImg = mobilenet_v2.preprocess_input(img_array)
    # pImg = nasnet.preprocess_input(img_array)
    pImg = vgg16.preprocess_input(img_array)

    return pImg


# process the test image
test_img_path = '%s/Abyssinian_1.jpg' % images_dir
pImg = process_image(test_img_path)

# # define the model
#
# model = densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
#                              pooling=None, classes=1000)
#
# model = inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
#                                  pooling=None, classes=1000)
#
# model = mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True,
#                             weights='imagenet', input_tensor=None, pooling=None, classes=1000)
#
# model = mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=True,
#                                  weights='imagenet', input_tensor=None, pooling=None, classes=1000)
#
# model = nasnet.NASNetMobile(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None,
#                             classes=1000)

model = vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,
                    classes=1000)

model.summary()

# record spend time
begin_time = time.clock()
prediction = model.predict(pImg)
end_time = time.clock()
print('Spend: %f ms' % (end_time - begin_time))

# obtain the top-3 predictions
results = imagenet_utils.decode_predictions(prediction, top=3)
print(results)

assert "Egyptian_cat" in [item[1] for item in results[0]]
