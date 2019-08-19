import os

from tensorflow.keras.applications import mobilenet_v2
from tensorflow.python.keras.utils import plot_model

base_path = "../../data"
output_path = "../../output"
images_dir = "%s/pets/images" % base_path

print("KERAS_HOME", os.environ.get('KERAS_HOME'))

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
