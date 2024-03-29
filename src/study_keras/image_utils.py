import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K

# base_path = "../../../data"
output_path = "../../../output"
# images_dir = "%s/pets/images" % base_path


##########################
# visualize Conv layer filters
##########################

def normalize(x):
    """utility function to normalize a tensor.

    # Arguments
        x: An input tensor.

    # Returns
        The normalized input tensor.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


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


def generate_pattern(input_img, layer_output, filter_index=0, size=150, epochs=20):
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    iterate = K.function([input_img], [loss, grads])

    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, size, size))
    else:
        input_img_data = np.random.random((1, size, size, 3))

    input_img_data = (input_img_data - 0.5) * 20 + 128

    step = 1
    for i in range(epochs):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step  # ?? why input + grads ??

        if loss_value <= K.epsilon():
            if i == 0:
                return None
            else:
                break

    return deprocess_image(input_img_data[0])


def visualize_layer_filters(model, layer_name, size=64, epochs=15):
    layer_output = model.get_layer(layer_name).output
    model_name = model.name

    if K.image_data_format() == 'channels_first':
        index_len = layer_output.shape.as_list()[1]
    else:
        index_len = layer_output.shape.as_list()[-1]
    print("%s filter length %d" % (layer_name, index_len))

    vol = 8
    row = math.ceil(index_len / vol)
    row = 8 if row > 8 else row  # most display first 64 filterss

    margin = 3
    results = np.zeros((row * size + (row - 1) * margin, vol * size + (vol - 1) * margin, 3)).astype('uint8')

    for i in range(row):
        for j in range(vol):

            if (i * vol + j) >= index_len:
                break

            filter_image = generate_pattern(model.input, layer_output, filter_index=(i * vol + j), size=size,
                                            epochs=epochs)

            if filter_image is not None:
                horizontal_start = i * size + i * margin
                horizontal_end = horizontal_start + size

                vertical_start = j * size + j * margin
                vertical_end = vertical_start + size

                results[horizontal_start:horizontal_end, vertical_start:vertical_end, :] = filter_image

            print(".%d" % (i * vol + j))

    plt.title(layer_name)
    plt.figure(figsize=(20, 20))
    plt.imshow(results)

    plt.show()
    cv2.imwrite("%s/%s.%s_filter.jpg" % (output_path, model_name, layer_name), results)


def visualize_activations(_pre, _layers_names, _activations, result_indexs=0):
    print(_layers_names)

    images_per_row = 8
    margin = 1

    for layer_name, layer_activation in zip(_layers_names, _activations):
        if K.image_data_format() == 'channels_first':
            n_features = layer_activation.shape[1]
            size = layer_activation.shape[-1]
        else:
            n_features = layer_activation.shape[-1]
            size = layer_activation.shape[1]

        n_cols = math.ceil(n_features / images_per_row)
        n_cols = 8 if n_cols > 8 else n_cols  # most display first 64 filterss

        display_grid = np.zeros((size * n_cols + (n_cols - 1) * margin,
                                 size * images_per_row + (images_per_row - 1) * margin)).astype('uint8')

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[result_indexs, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128

                channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                horizontal_start = col * size + col * margin
                horizontal_end = horizontal_start + size

                vertical_start = row * size + row * margin
                vertical_end = vertical_start + size

                display_grid[horizontal_start:horizontal_end, vertical_start:vertical_end] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))

        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

        cv2.imwrite("%s/%s.%s_actication.jpg" % (output_path, _pre, layer_name), display_grid)


def generate_heat_map(_model, input_image, layer_name, n_features):
    output = _model.output[:, n_features]
    selected_layer = _model.get_layer(layer_name)

    grads = K.gradients(output, selected_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([_model.input], [pooled_grads, selected_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([input_image])

    for i in range(n_features):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heat_map = np.mean(conv_layer_output_value, axis=-1)

    heat_map = np.maximum(heat_map, 0)
    heat_map /= np.max(heat_map)

    return heat_map


def merge_image_heat_map(input_image, input_heat_map):
    heat_map = cv2.resize(input_heat_map, (input_image.shape[1], input_image.shape[0]))
    heat_map = np.uint8(255 * heat_map)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

    superimposed_img = np.uint8(heat_map * 0.4 + 255 * input_image)
    return superimposed_img
