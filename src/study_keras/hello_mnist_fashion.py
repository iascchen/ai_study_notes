from tensorflow import keras
from tensorflow.python.keras.utils import plot_model

if __name__ == '__main__':
    mnist = keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_rows, img_cols = 28, 28
    train_images = x_train.reshape(x_train.shape[0], 28, 28, 1)
    test_images = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train, x_test = x_train / 255.0, x_test / 255.0

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

    model = keras.models.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),

        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(train_images, y_train, batch_size=128, epochs=5)
    model.evaluate(test_images, y_test)

    #######################
    # save and load model
    #######################
    base_path = "../../output"

    h5_path = "%s/hello_mnist_fashion.h5" % base_path
    model.save(h5_path)

    #######################
    # draw model
    #######################

    # use plot_model need graphviz be installed
    model_plot_path = "%s/hello_mnist_fashion_model_plot.png" % base_path
    plot_model(model, to_file=model_plot_path, show_shapes=True, show_layer_names=True)
