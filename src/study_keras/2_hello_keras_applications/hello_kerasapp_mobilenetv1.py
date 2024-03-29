import os
import time

import numpy as np
from tensorflow.keras.applications import mobilenet, imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model

'''
# different alpha 

model_25 = mobilenet.MobileNet()
model_25.summary()
print("0.25 > > >")

model_50 = mobilenet.MobileNet(input_shape=None, alpha=0.5, depth_multiplier=1, dropout=1e-3, include_top=True,
                     weights='imagenet', input_tensor=None, pooling=None, classes=1000)
model_50.summary()
print("0.50 > > >")

model_75 = mobilenet.MobileNet(input_shape=None, alpha=0.75, depth_multiplier=1, dropout=1e-3, include_top=True,
                     weights='imagenet', input_tensor=None, pooling=None, classes=1000)
model_75.summary()
print("0.75 > > >")

model_100 = mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True,
                      weights='imagenet', input_tensor=None, pooling=None, classes=1000)
model_100.summary()
print("1.00 > > >")

'''

base_path = "../../../data"
output_path = "../../../output"
images_dir = "%s/pets/images" % base_path

print("KERAS_HOME", os.environ.get('KERAS_HOME'))


# process an image to be model friendly
def process_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    pImg = mobilenet.preprocess_input(img_array)
    return pImg


# process the test image
test_img_path = '%s/Abyssinian_1.jpg' % images_dir
pImg = process_image(test_img_path)

# define the model
model_100 = mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True,
                                weights='imagenet', input_tensor=None, pooling=None, classes=1000)
model_100.summary()

# record spend time
begin_time = time.perf_counter()
prediction = model_100.predict(pImg)
end_time = time.perf_counter()
print('Spend: %f s' % (end_time - begin_time))

# obtain the top-3 predictions
results = imagenet_utils.decode_predictions(prediction, top=3)
print(results)

assert "Egyptian_cat" in [item[1] for item in results[0]]

#######################
# draw model
#######################

# use plot_model need graphviz be installed
model_plot_path = "%s/hello_kerasapp_mobilenet_model_plot.png" % output_path
plot_model(model_100, to_file=model_plot_path, show_shapes=True, show_layer_names=True)

yaml_path = "%s/mobilenet_v1.model.yaml" % output_path
yaml_string = model_100.to_yaml()
with open(yaml_path, 'w') as fw:
    fw.write(yaml_string)
