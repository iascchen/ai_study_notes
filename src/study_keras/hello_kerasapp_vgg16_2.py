from tensorflow.keras.applications import vgg16

from utils import visualize_layer_filters

if __name__ == '__main__':
    base_path = "../../data"
    output_path = "../../output"
    images_dir = "%s/images" % base_path

    # define the model
    model = vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,
                        classes=1000)
    model.summary()

    #######################
    # draw Convolutional Filter
    #######################

    convLayers = [layer.name for layer in model.layers if (layer.__class__.__name__ == 'Conv2D')]
    for i in range(len(convLayers)):
        visualize_layer_filters(model=model, layer_name=convLayers[i], epochs=40)
