from tensorflow.keras import datasets, models

from image_utils import visualize_layer_filters

if __name__ == '__main__':
    mnist = datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_rows, img_cols = 28, 28
    train_images = x_train.reshape(x_train.shape[0], 28, 28, 1)
    test_images = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train, x_test = x_train / 255.0, x_test / 255.0

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

    #######################
    # load model
    #######################
    base_path = "../../../output"

    h5_path = "%s/hello_mnist_fashion.h5" % base_path
    loaded_model = models.load_model(h5_path)
    loaded_model.summary()

    # loaded_model.evaluate(test_images, y_test)

    #######################
    # draw Convolutional Filter
    #######################

    # TODO: Error. Fused conv implementation does not support grouped convolutions for now.
    convLayers = [layer.name for layer in loaded_model.layers if (layer.__class__.__name__ == 'Conv2D')]
    print(convLayers)

    for i in range(len(convLayers)):
        visualize_layer_filters(model=loaded_model, layer_name=convLayers[i], epochs=40)
