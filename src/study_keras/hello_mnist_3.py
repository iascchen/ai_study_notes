import random

import matplotlib.pyplot as plt
from tensorflow.keras import callbacks, datasets, layers, models

mnist = datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#######################
# record epoch checkpoint
#######################

train_epochs = 6
save_freq = 'epoch'

# Create checkpoint callback
base_path = "../../output"

checkpoint_path = "%s/hello_mnist_3-{epoch:04d}.ckpt" % base_path
cp_callback = callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, save_freq=save_freq)

##################
# $ tensorboard --logdir base_path/hello_mnist_3.logs/
##################

log_path = "%s/hello_mnist_3.logs" % base_path
tp_callback = callbacks.TensorBoard(log_dir=log_path, write_graph=True, write_images=True,
                                    histogram_freq=0, embeddings_freq=0, embeddings_layer_names=None,
                                    embeddings_metadata=None)

history = model.fit(x_train, y_train, epochs=train_epochs, callbacks=[cp_callback, tp_callback])


#######################
# draw history
#######################

def random_color(number_of_colors):
    return ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(number_of_colors)]


def draw_history(_history):
    history_dict = _history.history

    keys = history_dict.keys()
    keys_list = list(keys)
    print(keys_list)

    values = [history_dict[keys_list[i]] for i in range(len(keys_list))]
    labels = [keys_list[i] for i in range(len(keys_list))]
    colors = random_color(len(keys_list))

    epochs = range(1, len(values[0]) + 1)

    for i in range(len(keys_list)):
        plt.plot(epochs, values[i], colors[i], label=labels[i])

    plt.xlabel('Epochs')
    plt.legend()

    plt.show()


draw_history(history)
