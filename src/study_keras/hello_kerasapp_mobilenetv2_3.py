import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image

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


test_img_path = '%s/Abyssinian_1.jpg' % images_dir
pImg = process_image(test_img_path)

# define the model
model_100 = mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=True,
                                     weights='imagenet', input_tensor=None, pooling=None, classes=1000)

results = model_100.predict(pImg)


def visualize_activations(model, picture, activations):
    layer_names = []
    for layer in model.layers[:4]:
        layer_names.append(layer.name)

    images_per_row = 8

    for layer_name, layer_activation in zip(layer_names, activations):
        print(layer_activation.shape)
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, size * images_per_row))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[picture, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))

        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

        plt.show()

print(results)
visualize_activations(model_100, pImg, results)
