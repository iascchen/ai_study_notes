import numpy as np
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

from utils import visualize_activations

if __name__ == '__main__':
    base_path = "../../data"
    output_path = "../../output"
    images_dir = "%s/pets/images" % base_path


    # process an image to be model friendly
    def process_image(img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        pImg = vgg16.preprocess_input(img_array)
        return pImg


    test_img_path = '%s/Abyssinian_1.jpg' % images_dir
    pImg = process_image(test_img_path)

    # define the model
    model = vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,
                        classes=1000)

    # display top %activation_top_layers% activations
    top_layers = 0
    layers_outputs = [layer.output for layer in model.layers if (layer.__class__.__name__ == 'Conv2D')][top_layers:]
    layers_names = [layer.name for layer in layers_outputs]

    activation_model = Model(inputs=model.input, outputs=layers_outputs)
    activation_model.summary()

    results = activation_model.predict(pImg)

    visualize_activations("vgg16_Abyssinian", layers_names, results, result_indexs=0)
