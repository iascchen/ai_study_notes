##########################
# Reference Link
# https://keras.io/examples/conv_filter_visualization/
##########################

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.python.keras.utils import plot_model

base_path = "../../data"
output_path = "../../output"
images_dir = "%s/images" % base_path

# define the model
model_100 = mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=True,
                                     weights='imagenet', input_tensor=None, pooling=None, classes=1000)

model_100.summary()

#######################
# draw model
#######################

# use plot_model need graphviz be installed
model_plot_path = "%s/hello_kerasapp_mobilenet_v2_model_plot.png" % output_path
plot_model(model_100, to_file=model_plot_path, show_shapes=True, show_layer_names=True)

yaml_path = "%s/mobilenet_v2.model.yaml" % output_path
yaml_string = model_100.to_yaml()
with open(yaml_path, 'w') as fw:
    fw.write(yaml_string)


##########################
# visualize Conv layer filters
##########################

def deprocess_image(x):
    """utility function to convert a float array into a valid uint8 image.

    # Arguments
        x: A numpy-array representing the generated image.

    # Returns
        A processed numpy-array, which could be used in e.g. imshow.
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(model, layer_name, filter_index=0, size=150, epochs=15):
    layer_output = model.get_layer(layer_name).output

    loss = K.mean(layer_output[:, :, :, filter_index])

    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())

    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    step = 1
    for i in range(epochs):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


def visualize_layer_filters(model, layer_name, filters_size=16, size=64, epochs=15):
    margin = 3
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3)).astype('uint8')

    # most display first 64 filters
    for i in range(8):
        for j in range(8):

            if (i * 8 + j) >= filters_size:
                break

            filter_image = generate_pattern(model, layer_name, filter_index=(i * 8 + j), size=size, epochs=epochs)

            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size

            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size

            results[horizontal_start:horizontal_end, vertical_start:vertical_end, :] = filter_image
            print(".%d" % (i * 8 + j))

    plt.figure(figsize=(20, 20))
    plt.imshow(results)

    plt.show()
    cv2.imwrite("%s/%s_filter.jpg" % (output_path, layer_name), results)


# layer_names = [
#     'Conv1',
#
#     'block_1_project', 'block_2_project',
#     'block_3_project', 'block_4_project', 'block_5_project',
#     'block_6_project', 'block_7_project', 'block_8_project', 'block_9_project',
#     'block_10_project', 'block_11_project', 'block_12_project',
#     'block_13_project', 'block_14_project', 'block_15_project',
#     'block_16_project',
#
#     'Conv_1',
# ]
#
# filters_size = [
#     32,
#     24, 24,
#     32, 32, 32,
#     64, 64, 64, 64,
#     96, 96, 96,
#     160, 160, 160,
#     320,
#     1280,
# ]

layer_names = [
    'Conv1',

    'block_1_project',
    'block_3_project',
    'block_6_project',
    'block_10_project',
    'block_13_project',
    'block_16_project',

    'Conv_1',
]

filters_size = [
    32,

    24,
    32,
    64,
    96,
    160,
    320,

    1280,
]

for i in range(len(layer_names)):
    visualize_layer_filters(model=model_100, layer_name=layer_names[i], epochs=40, filters_size=filters_size[i])
