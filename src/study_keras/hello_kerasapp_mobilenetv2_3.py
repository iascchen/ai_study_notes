import numpy as np
from tensorflow.keras.applications import mobilenet_v2
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
        pImg = mobilenet_v2.preprocess_input(img_array)
        return pImg


    test_img_path = '%s/Abyssinian_1.jpg' % images_dir
    pImg = process_image(test_img_path)

    # define the model
    model = mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=True,
                                     weights='imagenet', input_tensor=None, pooling=None, classes=1000)

    # display top %activation_top_layers% activations
    top_layers = 0
    layers_outputs = [layer.output for layer in model.layers if (layer.__class__.__name__ == 'Conv2D')][top_layers:]
    layers_names = [layer.name for layer in layers_outputs]

    activation_model = Model(inputs=model.input, outputs=layers_outputs)
    activation_model.summary()

    results = activation_model.predict(pImg)

    visualize_activations("mobilenetv2_Abyssinian", layers_names, results, result_indexs=0)
