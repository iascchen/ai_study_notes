import time

import numpy as np
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.applications import imagenet_utils

base_path = "../../data"
output_path = "../../output"
images_dir = "%s/images" % base_path


# process an image to be model friendly
def process_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    pImg = mobilenet_v2.preprocess_input(img_array)
    return pImg


# process the test image
test_img_path = '%s/Abyssinian_1.jpg' % images_dir
pImg = process_image(test_img_path)

# define the model
model = mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=True,
                                 weights='imagenet', input_tensor=None, pooling=None, classes=1000)
model.summary()

# 用于记录测试用时
begin_time = time.clock()
prediction = model.predict(pImg)
end_time = time.clock()
print('Spend: %f ms' % (end_time - begin_time))

# obtain the top-3 predictions
results = imagenet_utils.decode_predictions(prediction, top=3)
print(results)

# assert "Egyptian_cat" in [item[1] for item in results[0]]
