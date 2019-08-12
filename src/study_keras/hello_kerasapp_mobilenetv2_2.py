##########################
# Reference Link
# https://keras.io/examples/conv_filter_visualization/
##########################

from tensorflow.keras.applications import mobilenet_v2

from src.study_keras.utils import visualize_layer_filters

base_path = "../../data"
output_path = "../../output"
images_dir = "%s/images" % base_path

# define the model
model_100 = mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=True,
                                weights='imagenet', input_tensor=None, pooling=None, classes=1000)
model_100.summary()

#######################
# draw Convolutional Filter
#######################

convLayers = [layer.name for layer in model_100.layers if (layer.__class__.__name__ == 'Conv2D')]
for i in range(len(convLayers)):
    visualize_layer_filters(model=model_100, layer_name=convLayers[i], epochs=40)
