from tensorflow.keras.applications import mobilenet

from utils import visualize_layer_filters

if __name__ == '__main__':
    base_path = "../../data"
    output_path = "../../output"
    images_dir = "%s/images" % base_path

    # define the model
    model_100 = mobilenet.MobileNet(input_shape=None, alpha=1.0, include_top=True,
                                    weights='imagenet', input_tensor=None, pooling=None, classes=1000)
    model_100.summary()

    #######################
    # draw Convolutional Filter
    #######################

    convLayers = [layer.name for layer in model_100.layers if (layer.__class__.__name__ == 'Conv2D')]
    for i in range(len(convLayers)):
        visualize_layer_filters(model=model_100, layer_name=convLayers[i], epochs=40)
