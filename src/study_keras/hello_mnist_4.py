import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#######################
# record epoch checkpoint
#######################

train_epochs = 6
train_period = 2

# Create checkpoint callback
base_path = "../../output"

checkpoint_path = "%s/hello_mnist_3-{epoch:04d}.ckpt" % base_path
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, period=train_period,
                                                 verbose=1)

history = model.fit(x_train, y_train, epochs=train_epochs, callbacks=[cp_callback])


def layer_to_visualize(layer):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])

    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    print('Shape of conv:', convolutions.shape)

    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))

    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12, 8))
    for i in range(len(convolutions)):
        ax = fig.add_subplot(n, n, i + 1)
        ax.imshow(convolutions[i], cmap='gray')


# Specify the layer to want to visualize
layer_to_visualize(convout1)


# function to get activations of a layer
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output, ])
    activations = get_activations([X_batch, 0])
    return activations


# Get activations using layername
def get_activation_from_layer(model, layer_name, layers, layers_dim, img):
    acti = get_activations(model, layers[layer_name], img.reshape(1, 256, 256, 3))[0].reshape(layers_dim[layer][0],
                                                                                              layers_dim[layer_name][1],
                                                                                              layers_dim[layer_name][2])
    return np.sum(acti, axis=2)


# Map layer name with layer index
layers = dict()
index = None
for idx, layer in enumerate(model.layers):
    layers[layer.name] = idx

# Map layer name with its dimension
layers_dim = dict()

for layer in model.layers:
    layers_dim[layer.name] = layer.get_output_at(0).get_shape().as_list()[1:]

img1 = utils.load_img("image.png", target_size=(256, 256))

# define the layer you want to visualize
layer_name = "conv2d_22"
plt.imshow(get_activation_from_layer(model, layer_name, layers, layers_dim, img1), cmap="jet")
