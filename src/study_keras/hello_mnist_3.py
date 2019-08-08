import random

import matplotlib.pyplot as plt
import tensorflow as tf

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


#######################
# draw history
#######################

def random_color(number_of_colors):
    return ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]


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
