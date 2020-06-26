import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.applications import imagenet_utils

from utils import generate_heat_map, merge_image_heat_map

if __name__ == '__main__':
    base_path = "../../../data"
    output_path = "../../../output"
    images_dir = "%s/pets/images" % base_path


    # process an image to be model friendly
    def process_image(img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        pImg = vgg16.preprocess_input(img_array)
        return pImg


    test_img_path = '%s/Abyssinian_3.jpg' % images_dir
    pImg = process_image(test_img_path)

    # define the model
    model = vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,
                        classes=1000)

    layers_names = [layer.name for layer in model.layers if (layer.__class__.__name__ == 'Conv2D')]

    prediction = model.predict(pImg)
    print(imagenet_utils.decode_predictions(prediction, top=3)[0])

    for layer_name in layers_names:
        layer_output = model.get_layer(layer_name).output
        model_name = model.name

        if K.image_data_format() == 'channels_first':
            n_features = layer_output.shape.as_list()[1]
        else:
            n_features = layer_output.shape.as_list()[-1]
        print("%s filter n_features %d" % (layer_name, n_features))

        heat_map = generate_heat_map(model, pImg, layer_name, n_features)

        plt.matshow(heat_map)
        plt.title(layer_name)
        plt.show()

        img = cv2.imread(test_img_path)
        composed_img = merge_image_heat_map(img, heat_map)
        plt.imshow(composed_img)
        plt.show()

        cv2.imwrite("%s/%s_%s.%s_heat_map.jpg" % (output_path, "Abyssinian", model_name, layer_name), composed_img)
