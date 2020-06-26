import time
import os
import hashlib
import PIL.Image as Image
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

models_path = "../../../models"
os.environ['TFHUB_CACHE_DIR'] = models_path

if __name__ == '__main__':
    base_path = "../../../data"
    output_path = "../../../output"
    images_dir = "%s/pets/images" % base_path

    # tf.enable_eager_execution()

    IMAGE_SHAPE = (224, 224)

    # model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    # handle_hash = hashlib.sha1(model_url.encode("utf8")).hexdigest()  # f34b2684786cf6de38511148638abf91283beb1f

    model_url = "https://tfhub.dev/google/tf2-preview/inception_v3/classification/4"
    handle_hash = hashlib.sha1(model_url.encode("utf8")).hexdigest()  # f34b2684786cf6de38511148638abf91283beb1f

    hub_layer = hub.KerasLayer(model_url, output_shape=[1001])

    classifier = tf.keras.Sequential([
        hub.KerasLayer(hub_layer)
    ])
    classifier.build([None, 224, 224, 3])  # Batch input shape.

    classifier.summary()

    grace_hopper = tf.keras.utils.get_file('image.jpg',
                                           'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
    grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
    grace_hopper = np.array(grace_hopper) / 255.0

    result = classifier.predict(grace_hopper[np.newaxis, ...])

    predicted_class = np.argmax(result[0], axis=-1)

    # labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
    #                                       'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

    labels_path = "./ImageNetLabels.txt"
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    plt.imshow(grace_hopper)
    plt.axis('off')
    predicted_class_name = imagenet_labels[predicted_class]
    _ = plt.title("Prediction: " + predicted_class_name.title())

    # data_root = tf.keras.utils.get_file(
    #     'flower_photos', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    #     untar=True)

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    image_data = image_generator.flow_from_directory("../../../data/flower_photos", target_size=IMAGE_SHAPE)

    # for image_batch, label_batch in image_data:
    #     print("Image batch shape: ", image_batch.shape)
    #     print("Label batch shape: ", label_batch.shape)
    #     break
    #
    # result_batch = classifier.predict(image_batch)
    # predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
    #
    # plt.figure(figsize=(10, 9))
    # plt.subplots_adjust(hspace=0.5)
    # for n in range(30):
    #     plt.subplot(6, 5, n + 1)
    #     plt.imshow(image_batch[n])
    #     plt.title(predicted_class_names[n])
    #     plt.axis('off')
    # _ = plt.suptitle("ImageNet predictions")
    #
    # feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"  # @param {type:"string"}
    #
    # feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
    #                                          input_shape=(224, 224, 3))
    #
    # feature_batch = feature_extractor_layer(image_batch)
    # print(feature_batch.shape)
    #
    # feature_extractor_layer.trainable = False
    #
    # model = tf.keras.Sequential([
    #     feature_extractor_layer,
    #     layers.Dense(image_data.num_classes, activation='softmax')
    # ])
    #
    # model.summary()
    #
    # predictions = model(image_batch)
    #
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(),
    #     loss='categorical_crossentropy',
    #     metrics=['acc'])
    #
    #
    # class CollectBatchStats(tf.keras.callbacks.Callback):
    #     def __init__(self):
    #         self.batch_losses = []
    #         self.batch_acc = []
    #
    #     def on_train_batch_end(self, batch, logs=None):
    #         self.batch_losses.append(logs['loss'])
    #         self.batch_acc.append(logs['acc'])
    #         self.model.reset_metrics()
    #
    #
    # steps_per_epoch = np.ceil(image_data.samples / image_data.batch_size)
    #
    # batch_stats_callback = CollectBatchStats()
    #
    # history = model.fit(image_data, epochs=2,
    #                     steps_per_epoch=steps_per_epoch,
    #                     callbacks=[batch_stats_callback])
    #
    # class_names = sorted(image_data.class_indices.items(), key=lambda pair: pair[1])
    # class_names = np.array([key.title() for key, value in class_names])
    #
    # predicted_batch = model.predict(image_batch)
    # predicted_id = np.argmax(predicted_batch, axis=-1)
    # predicted_label_batch = class_names[predicted_id]
    #
    # label_id = np.argmax(label_batch, axis=-1)
    #
    # plt.figure(figsize=(10, 9))
    # plt.subplots_adjust(hspace=0.5)
    # for n in range(30):
    #     plt.subplot(6, 5, n + 1)
    #     plt.imshow(image_batch[n])
    #     color = "green" if predicted_id[n] == label_id[n] else "red"
    #     plt.title(predicted_label_batch[n].title(), color=color)
    #     plt.axis('off')
    # _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
    #
    # t = time.time()
    #
    # base_path = "../../../data"
    # output_path = "../../../output"
    # images_dir = "%s/pets/images" % base_path
    #
    # export_path = "{:0}/saved_models/{:1}".format(output_path, int(t))
    # tf.keras.experimental.export_saved_model(model, export_path)
