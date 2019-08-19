import os

import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    base_path = "../../data"
    output_path = "../../output"
    images_dir = "%s/flower_photos" % base_path

    print("KERAS_HOME", os.environ.get('KERAS_HOME'))

    #####################
    # Data Generator
    #####################

    # Method 1: use categorical
    train_datagen = ImageDataGenerator(preprocessing_function=mobilenet_v2.preprocess_input,
                                       validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
                                                        color_mode='rgb', batch_size=32, shuffle=True,
                                                        class_mode='categorical', subset='training')
    validation_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
                                                             color_mode='rgb', batch_size=32, shuffle=True,
                                                             class_mode='categorical', subset='validation')

    loss_alg = 'categorical_crossentropy'

    # # show data format of train_generator and validation_generator
    # i = 0
    # for inputs_batch, labels_batch in validation_generator:
    #     if i < 1:
    #         print(inputs_batch)
    #         print(labels_batch)
    #         i += 1
    #     else:
    #         break

    # # Method 2: use image net class
    # # Before do this, please add new classes at the end of ImageNetLabels.txt. each class per line
    # # classes: daisy、dandelion、roses、sunflowers、tulips
    # labels_path = "ImageNetLabels.txt"
    # imagenet_labels = np.array(open(labels_path).read().splitlines())[1:].tolist()  # drop first line 'background'
    # # print(len(imagenet_labels), imagenet_labels)
    #
    # train_datagen = ImageDataGenerator(preprocessing_function=mobilenet_v2.preprocess_input,
    #                                    validation_split=0.2)
    # train_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
    #                                                     color_mode='rgb', batch_size=32, shuffle=True,
    #                                                     classes=imagenet_labels, class_mode='sparse',
    #                                                     subset='training')
    # validation_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
    #                                                          color_mode='rgb', batch_size=32, shuffle=True,
    #                                                          classes=imagenet_labels, class_mode='sparse',
    #                                                          subset='validation')
    #
    # # because its class_mode='sparse'
    # loss_alg = 'sparse_categorical_crossentropy'

    step_size_train = train_generator.n // train_generator.batch_size
    step_size_validation = validation_generator.n // validation_generator.batch_size

    ###################
    # Define transfer learning model
    ###################

    # define the model
    base_model = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    ###################
    # Adjust layers to transfer learning
    ###################

    # ###################
    # # Method 1
    #
    # model_name = "mobilenetv2_transfer_seq"
    #
    # model = keras.Sequential([
    #     base_model,
    #     keras.layers.GlobalAveragePooling2D(),
    #     keras.layers.Dense(1024, activation='relu'),
    #     keras.layers.Dense(512, activation='relu'),
    #     keras.layers.Dropout(0.2),
    #     keras.layers.Dense(5, activation='softmax')
    # ])
    #
    # # # adjust base model
    # # base_model.trainable = True
    # #
    # # trainable_base_layers = -3
    # # for layer in base_model.layers[:trainable_base_layers]:
    # #     layer.trainable = False
    #
    # layers_names = [layer.name for layer in base_model.layers if layer.trainable is True]
    # print("base_model trainable layers:", layers_names)
    #
    # layers_names = [layer.name for layer in model.layers if layer.trainable is True]
    # print("model trainable layers:", layers_names)

    ##################
    # Method 2

    model_name = "mobilenetv2_transfer_model"

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = base_model.get_layer('block_16_project_BN').output
    x = keras.layers.Conv2D(1280, kernel_size=1, use_bias=False, name='Conv_1')(x)
    x = keras.layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = keras.layers.ReLU(6., name='out_relu')(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    preds = keras.layers.Dense(5, activation='softmax', use_bias=True, name='Logits')(x)
    model = keras.Model(inputs=base_model.input, outputs=preds)

    layers_names = [layer.name for layer in model.layers]
    print("All layers:", layers_names)

    # adjust model for more trainable layer
    trainable_layers = -13
    for layer in model.layers[trainable_layers:]:
        layer.trainable = True

    layers_names = [layer.name for layer in model.layers if layer.trainable is True]
    print("Trainable layers:", layers_names)

    #####################
    # setting callback to fit
    #####################

    log_path = "%s/%s.logs" % (output_path, model_name)
    tp_callback = keras.callbacks.TensorBoard(log_dir=log_path, write_graph=True, write_grads=True,
                                              write_images=True,
                                              histogram_freq=0, embeddings_freq=0, embeddings_layer_names=None,
                                              embeddings_metadata=None)

    #####################
    # Train & evaluate
    #####################

    model.compile(optimizer='adam', loss=loss_alg, metrics=['accuracy'])
    model.summary()

    history = model.fit_generator(train_generator, steps_per_epoch=step_size_train,
                                  validation_data=validation_generator, validation_steps=step_size_validation,
                                  epochs=5, callbacks=[tp_callback])
    print("Train history : ", history.history)

    val_results = model.evaluate_generator(validation_generator, steps=step_size_validation)
    print("Evaluate result : ", val_results)

    results = model.predict_generator(validation_generator, steps=step_size_validation)
    result_class = [np.argmax(result) for result in results]
    print("predict result : ", result_class)

    #####################
    # save model and trained result
    #####################

    yaml_path = "%s/%s.model.yaml" % (output_path, model_name)
    yaml_string = model.to_yaml()
    with open(yaml_path, 'w') as fw:
        fw.write(yaml_string)

    h5_path = "%s/%s.h5" % (output_path, model_name)
    model.save(h5_path)
