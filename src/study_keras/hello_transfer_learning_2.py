import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.training.basic_session_run_hooks import CheckpointSaverListener


class MyListener(CheckpointSaverListener):
    def begin(self):
        print(".")

    def end(self, session, global_step_value):
        print("_")


if __name__ == '__main__':
    base_path = "../../data"
    output_path = "../../output"
    images_dir = "%s/flower_photos" % base_path

    #####################
    # Data Generator
    #####################

    # Before do this, please add new classes at the end of ImageNetLabels.txt. each class per line
    # classes: daisy、dandelion、roses、sunflowers、tulips
    labels_path = "ImageNetLabels.txt"
    imagenet_labels = np.array(open(labels_path).read().splitlines())[1:].tolist()  # drop first line 'background'

    print(len(imagenet_labels), imagenet_labels)

    train_datagen = ImageDataGenerator(preprocessing_function=mobilenet_v2.preprocess_input,
                                       validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
                                                        color_mode='rgb', batch_size=32, shuffle=True,
                                                        classes=imagenet_labels, class_mode='sparse',
                                                        subset='training')
    validation_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
                                                             color_mode='rgb', batch_size=32, shuffle=True,
                                                             classes=imagenet_labels, class_mode='sparse',
                                                             subset='validation')

    ###################
    # Define transfer learning model
    ###################

    base_model = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    keras_model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(5, activation='softmax')
    ])
    keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metric='accuracy')

    ###################
    # Use tensorflow estimator
    ###################

    est_model = keras.estimator.model_to_estimator(keras_model=keras_model)

    train_x = []
    train_y = []
    for inputs_batch, labels_batch in validation_generator:
        train_x.append(inputs_batch)
        train_y.append(labels_batch)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=train_x, y=train_y, num_epochs=1, shuffle=False)

    my_listener = MyListener()

    # To train, we call Estimator's train function:
    est_model.train(input_fn=train_input_fn, steps=5, saving_listeners=my_listener)
