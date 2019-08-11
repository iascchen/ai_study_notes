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
# Visualize Layer
##########################


# def normalize(x):
#     """utility function to normalize a tensor.
#
#     # Arguments
#         x: An input tensor.
#
#     # Returns
#         The normalized input tensor.
#     """
#     return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


# def process_image(x, former):
#     """utility function to convert a valid uint8 image back into a float array.
#        Reverses `deprocess_image`.
#
#     # Arguments
#         x: A numpy-array, which could be used in e.g. imshow.
#         former: The former numpy-array.
#                 Need to determine the former mean and variance.
#
#     # Returns
#         A processed numpy-array representing the generated image.
#     """
#     if K.image_data_format() == 'channels_first':
#         x = x.transpose((2, 0, 1))
#     return (x / 255 - 0.5) * 4 * former.std() + former.mean()


# def visualize_layer(model,
#                     layer_name,
#                     step=1.,
#                     epochs=15,
#                     upscaling_steps=9,
#                     upscaling_factor=1.2,
#                     output_dim=(412, 412),
#                     filter_range=(0, None)):
#     """Visualizes the most relevant filters of one conv-layer in a certain model.
#
#     # Arguments
#         model: The model containing layer_name.
#         layer_name: The name of the layer to be visualized.
#                     Has to be a part of model.
#         step: step size for gradient ascent.
#         epochs: Number of iterations for gradient ascent.
#         upscaling_steps: Number of upscaling steps.
#                          Starting image is in this case (80, 80).
#         upscaling_factor: Factor to which to slowly upgrade
#                           the image towards output_dim.
#         output_dim: [img_width, img_height] The output image dimensions.
#         filter_range: Tupel[lower, upper]
#                       Determines the to be computed filter numbers.
#                       If the second value is `None`,
#                       the last filter will be inferred as the upper boundary.
#     """
#
#     def _generate_filter_image(input_img,
#                                layer_output,
#                                filter_index):
#         """Generates image for one particular filter.
#
#         # Arguments
#             input_img: The input-image Tensor.
#             layer_output: The output-image Tensor.
#             filter_index: The to be processed filter number.
#                           Assumed to be valid.
#
#         #Returns
#             Either None if no image could be generated.
#             or a tuple of the image (array) itself and the last loss.
#         """
#         s_time = time.time()
#
#         # we build a loss function that maximizes the activation
#         # of the nth filter of the layer considered
#         if K.image_data_format() == 'channels_first':
#             loss = K.mean(layer_output[:, filter_index, :, :])
#         else:
#             loss = K.mean(layer_output[:, :, :, filter_index])
#
#         # we compute the gradient of the input picture wrt this loss
#         grads = K.gradients(loss, input_img)[0]
#
#         # normalization trick: we normalize the gradient
#         grads = normalize(grads)
#
#         # this function returns the loss and grads given the input picture
#         iterate = K.function([input_img], [loss, grads])
#
#         # we start from a gray image with some random noise
#         intermediate_dim = tuple(int(x / (upscaling_factor ** upscaling_steps)) for x in output_dim)
#         print("intermediate_dim", intermediate_dim)
#
#         if K.image_data_format() == 'channels_first':
#             input_img_data = np.random.random(
#                 (1, 3, intermediate_dim[0], intermediate_dim[1]))
#         else:
#             input_img_data = np.random.random(
#                 (1, intermediate_dim[0], intermediate_dim[1], 3))
#         input_img_data = (input_img_data - 0.5) * 20 + 128
#
#         # Slowly upscaling towards the original size prevents
#         # a dominating high-frequency of the to visualized structure
#         # as it would occur if we directly compute the 412d-image.
#         # Behaves as a better starting point for each following dimension
#         # and therefore avoids poor local minima
#         for up in reversed(range(upscaling_steps)):
#             # we run gradient ascent for e.g. 20 steps
#             for _ in range(epochs):
#                 loss_value, grads_value = iterate([input_img_data])
#                 input_img_data += grads_value * step
#
#                 # some filters get stuck to 0, we can skip them
#                 if loss_value <= K.epsilon():
#                     return None
#
#             # Calulate upscaled dimension
#             intermediate_dim = tuple(int(x / (upscaling_factor ** up)) for x in output_dim)
#             # Upscale
#             img = deprocess_image(input_img_data[0])
#             img = np.array(pil_image.fromarray(img).resize(intermediate_dim,
#                                                            pil_image.BICUBIC))
#             input_img_data = [process_image(img, input_img_data[0])]
#
#             print("input_img_data.shape : ", input_img_data[0].shape)
#
#         # decode the resulting input image
#         img = deprocess_image(input_img_data[0])
#         e_time = time.time()
#         print('Costs of filter {:3}: {:5.0f} ( {:4.2f}s )'.format(filter_index,
#                                                                   loss_value,
#                                                                   e_time - s_time))
#         return img, loss_value
#
#     def _draw_filters(filters, n=None):
#         """Draw the best filters in a nxn grid.
#
#         # Arguments
#             filters: A List of generated images and their corresponding losses
#                      for each processed filter.
#             n: dimension of the grid.
#                If none, the largest possible square will be used
#         """
#         if n is None:
#             n = int(np.floor(np.sqrt(len(filters))))
#
#         # the filters that have the highest loss are assumed to be better-looking.
#         # we will only keep the top n*n filters.
#         filters.sort(key=lambda x: x[1], reverse=True)
#         filters = filters[:n * n]
#
#         # build a black picture with enough space for
#         # e.g. our 8 x 8 filters of size 412 x 412, with a 5px margin in between
#         MARGIN = 5
#         width = n * output_dim[0] + (n - 1) * MARGIN
#         height = n * output_dim[1] + (n - 1) * MARGIN
#         stitched_filters = np.zeros((width, height, 3), dtype='uint8')
#
#         # fill the picture with our saved filters
#         for i in range(n):
#             for j in range(n):
#                 img, _ = filters[i * n + j]
#                 width_margin = (output_dim[0] + MARGIN) * i
#                 height_margin = (output_dim[1] + MARGIN) * j
#                 stitched_filters[
#                 width_margin: width_margin + output_dim[0],
#                 height_margin: height_margin + output_dim[1], :] = img
#
#         # save the result to disk
#         save_img('vgg_{0:}_{1:}x{1:}.png'.format(layer_name, n), stitched_filters)
#
#     # this is the placeholder for the input images
#     assert len(model.inputs) == 1
#     input_img = model.inputs[0]
#
#     # get the symbolic outputs of each "key" layer (we gave them unique names).
#     layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
#
#     output_layer = layer_dict[layer_name]
#     # assert isinstance(output_layer, layers.Conv2D)
#
#     # Compute to be processed filter range
#     filter_lower = filter_range[0]
#     filter_upper = (filter_range[1]
#                     if filter_range[1] is not None
#                     else len(output_layer.get_weights()[1]))
#     assert ((filter_lower >= 0) and (filter_upper <= len(output_layer.get_weights()[1]))
#             and (filter_upper > filter_lower))
#     print('Compute filters {:} to {:}'.format(filter_lower, filter_upper))
#
#     # iterate through each filter and generate its corresponding image
#     processed_filters = []
#     for f in range(filter_lower, filter_upper):
#         img_loss = _generate_filter_image(input_img, output_layer.output, f)
#
#         if img_loss is not None:
#             processed_filters.append(img_loss)
#
#     print('{} filter processed.'.format(len(processed_filters)))
#     # Finally draw and store the best filters to disk
#     _draw_filters(processed_filters)

################
# visualize Conv layer filters
################

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
    grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon)

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


#
# def generate_heat_map(_model, input_image, layer_name, class_index):
#     output = _model.output[:, class_index]
#     last_layer = _model.get_layer(layer_name)
#
#     grads = K.gradients(output, last_layer.output)[0]
#     pooled_grads = K.mean(grads, axis=(0, 1, 2))
#
#     iterate = K.function([_model.input], [pooled_grads, last_layer.output[0]])
#     pooled_grads_value, conv_layer_output_value = iterate([input_image])
#
#     for i in range(512):
#         conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
#
#     heat_map = np.mean(conv_layer_output_value, axis=-1)
#
#     heat_map = np.maximum(heat_map, 0)
#     heat_map /= np.max(heat_map)
#
#     return heat_map
#
#
# def merge_image_heat_map(input_image, input_heat_map):
#     heat_map = cv2.resize(input_heat_map, (input_image.shape[1], input_image.shape[0]))
#     heat_map = np.uint8(255 * heat_map)
#     heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
#
#     superimposed_img = np.uint8(heat_map * 0.4 + 255 * input_image)
#     return superimposed_img
#
#
# def visualize_activations(model, picture, activations):
#     layer_names = []
#     for layer in model.layers[:4]:
#         layer_names.append(layer.name)
#
#     images_per_row = 8
#
#     for layer_name, layer_activation in zip(layer_names, activations):
#         n_features = layer_activation.shape[-1]
#         size = layer_activation.shape[1]
#
#         n_cols = n_features // images_per_row
#         display_grid = np.zeros((size * n_cols, size * images_per_row))
#
#         for col in range(n_cols):
#             for row in range(images_per_row):
#                 channel_image = layer_activation[picture, :, :, col * images_per_row + row]
#                 channel_image -= channel_image.mean()
#                 channel_image /= channel_image.std()
#                 channel_image *= 64
#                 channel_image += 128
#                 channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#                 display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
#
#         scale = 1. / size
#         plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
#
#         plt.title(layer_name)
#         plt.grid(False)
#         plt.imshow(display_grid, aspect='auto', cmap='viridis')
#
#         plt.show()


#
# def test_layer_filter():
#     plt.imshow(generate_pattern(model=model, layer_name='block4_conv1', filter_index=0, size=64))
#     plt.show()
#
#
# def test_visualize_layer_filters():
#     visualize_layer_filters(model=model, layer_name='block4_conv1')
#     # plt.show()
#
#
# def test_heat_map():
#     prediction = model.predict(pImg)
#     print(imagenet_utils.decode_predictions(prediction, top=3)[0])
#
#     assert 284 == np.argmax(prediction[0])
#
#     heat_map = generate_heat_map(model, pImg, 'block5_conv3', 284)
#     plt.matshow(heat_map)
#     plt.show()
#
#     img = cv2.imread(test_img_path)
#     composed_img = merge_image_heat_map(img, heat_map)
#     plt.imshow(composed_img)
#     plt.show()
#
#     # cv2.imwrite("output_cat.jpg", composed_img)

layer_names = [
    'Conv1',

    'block_1_project', 'block_2_project',
    'block_3_project', 'block_4_project', 'block_5_project',
    'block_6_project', 'block_7_project', 'block_8_project', 'block_9_project',
    'block_10_project', 'block_11_project', 'block_12_project',
    'block_13_project', 'block_14_project', 'block_15_project',
    'block_16_project',

    'Conv_1',
]

filters_size = [
    32,
    24, 24,
    32, 32, 32,
    64, 64, 64, 64,
    96, 96, 96,
    160, 160, 160,
    320,
    1280,
]

for i in range(len(layer_names)):
    visualize_layer_filters(model=model_100, layer_name=layer_names[i], epochs=40, filters_size=filters_size[i])
