import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    base_path = "../../data"
    output_path = "../../output"
    images_dir = "%s/flower_photos" % base_path

    # define the model
    base_model = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
    base_model.trainable = False
    base_model.summary()

    ###################
    # Add some layers to transfer learning
    ###################

    # Method 1
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(5, activation='softmax')
    ])

    # # Method 2
    # x = base_model.output
    # x = keras.layers.GlobalAveragePooling2D()(x)
    # x = keras.layers.Dense(1024, activation='relu')(x)
    # x = keras.layers.Dense(512, activation='relu')(x)
    # x = keras.layers.Dropout(0.2)(x)
    # preds = keras.layers.Dense(5, activation='softmax')(x)  # final layer with softmax activation
    # model = keras.Model(inputs=base_model.input, outputs=preds)

    layers_names = [layer.name for layer in model.layers]
    print("All layers:", layers_names)

    layers_names = [layer.name for layer in model.layers if layer.trainable is True]
    print("Trainable layers 1:", layers_names)

    trainable_top_layers = -8
    # for layer in model.layers[:trainable_top_layers]:
    #     layer.trainable = False
    for layer in model.layers[trainable_top_layers:]:
        layer.trainable = True

    layers_names = [layer.name for layer in model.layers if layer.trainable is True]
    print("Trainable layers 2:", layers_names)

    model.summary()

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    yaml_path = "%s/mobilenetv2_transfer.model.yaml" % output_path
    yaml_string = model.to_yaml()
    with open(yaml_path, 'w') as fw:
        fw.write(yaml_string)

    log_path = "%s/mobilenetv2_transfer.logs" % output_path
    tp_callback = keras.callbacks.TensorBoard(log_dir=log_path, write_graph=True, write_grads=True,
                                              write_images=True,
                                              histogram_freq=0, embeddings_freq=0, embeddings_layer_names=None,
                                              embeddings_metadata=None)

    train_datagen = ImageDataGenerator(preprocessing_function=mobilenet_v2.preprocess_input,
                                       validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
                                                        color_mode='rgb', batch_size=32, shuffle=True,
                                                        class_mode='categorical', subset='training')
    validation_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
                                                             color_mode='rgb', batch_size=32, shuffle=True,
                                                             class_mode='categorical', subset='validation')

    step_size_train = train_generator.n // train_generator.batch_size
    step_size_validation = validation_generator.n // validation_generator.batch_size

    history = model.fit_generator(train_generator, steps_per_epoch=step_size_train,
                                  validation_data=validation_generator, validation_steps=step_size_validation,
                                  epochs=5, callbacks=[tp_callback])
    print("Train history : ", history.history)

    val_results = model.evaluate_generator(validation_generator, steps=step_size_validation)
    print("Evaluate result : ", val_results)

    results = model.predict_generator(validation_generator, steps=step_size_validation)
    result_class = [np.argmax(result) for result in results]
    print("predict result : ", result_class)

    h5_path = "%s/mobilenetv2_transfer.h5" % output_path
    model.save(h5_path)
