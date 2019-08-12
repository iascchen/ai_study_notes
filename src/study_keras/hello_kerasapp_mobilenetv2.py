##########################
# Reference Link
# https://keras.io/examples/conv_filter_visualization/
##########################

from tensorflow.keras.applications import mobilenet_v2
from tensorflow.python.keras.utils import plot_model

from src.study_keras.utils import visualize_layer_filters

base_path = "../../data"
output_path = "../../output"
images_dir = "%s/images" % base_path

# define the model
model_100 = mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=True,
                                     weights='imagenet', input_tensor=None, pooling=None, classes=1000)

model_100.summary()

#######################
# draw model
#######################

# use plot_model need graphviz be installed
model_plot_path = "%s/hello_kerasapp_mobilenet_v2_model_plot.png" % output_path
plot_model(model_100, to_file=model_plot_path, show_shapes=True, show_layer_names=True)

yaml_path = "%s/mobilenet_v2.model.yaml" % output_path
yaml_string = model_100.to_yaml()
with open(yaml_path, 'w') as fw:
    fw.write(yaml_string)

convLayers = [layer.name for layer in model_100.layers if (layer.__class__.__name__ == 'Conv2D')]
for i in range(len(convLayers)):
    visualize_layer_filters(model=model_100, layer_name=convLayers[i], epochs=40)
