from tensorflow.keras import datasets, layers, models, utils

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
model.fit(x_train, y_train, epochs=5)

# #######################
# # save and load model structure to json or yaml
# #######################
#
base_path = "../../output"

# json model

json_path = "%s/hello_mnist_2.model.json" % base_path
json_string = model.to_json()
with open(json_path, 'w') as fw:
    fw.write(json_string)

with open(json_path, 'r') as fr:
    new_json_string = fr.read()
json_model = models.model_from_json(new_json_string)

json_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# yaml model

yaml_path = "%s/hello_mnist_2.model.yaml" % base_path
yaml_string = model.to_yaml()
with open(yaml_path, 'w') as fw:
    fw.write(yaml_string)

with open(yaml_path, 'r') as fr:
    new_yaml_string = fr.read()
yaml_model = models.model_from_yaml(new_yaml_string)

yaml_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#######################
# save and load weight to a TensorFlow Checkpoint file
#######################

weight_path = "%s/hello_mnist_2.weights" % base_path
model.save_weights(weight_path)

json_model.load_weights(weight_path)

#######################
# save and load weight to a HDF5 file
#######################

h5_weight_path = "%s/hello_mnist_2.weights.h5" % base_path
model.save_weights(h5_weight_path, save_format='h5')

yaml_model.load_weights(h5_weight_path)

#######################
# loaded_model evaluate and predict
#######################

result_loss, result_acc = json_model.evaluate(x_test, y_test)
print('Loaded Json Model Test loss & accuracy:', result_loss, result_acc)

result_loss, result_acc = yaml_model.evaluate(x_test, y_test)
print('Loaded Yaml Model Test loss & accuracy:', result_loss, result_acc)
