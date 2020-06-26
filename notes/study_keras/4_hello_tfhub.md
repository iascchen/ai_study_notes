# 对象检测及其迁移学习

## TFHub 模型的加载

TFHub 的 模型加载可以使用 URL，直接从云端下载。也可以使用本地存储好的 TFHub 模型文件。

使用 TFHub 之前，需要下载对应的模型，由于网络的原因，一般下载比较慢。

[hello_tfhub.py](../../src/study_keras/4_hello_tfhub/hello_tfhub.py) 通过设定 TFHUB_CACHE_DIR 环境变量，指定了模型的下载存储路径，将云端下载的内容存在你需要的位置。
下载的模型会被存放在对应 Hash 值的目录下，同时会产生一个hash值为名称的文本描述文件。
这个模型目录下，会有 tfhub_module.pb 和 saved_module.pb 这样的模型文件，还会有对应的参数存储。
get_input_info_dict 和 get_output_info_dict 能够显示此模型的输入输出信息。

    import hashlib
    import os
    
    import tensorflow_hub as hub
    
    models_path = "../../models"
    
    os.environ['TFHUB_CACHE_DIR'] = models_path
    print("TFHUB_CACHE_DIR", os.environ.get('TFHUB_CACHE_DIR'))
    
    module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    handle_hash = hashlib.sha1(module_handle.encode("utf8")).hexdigest()
    module_handle = "%s/%s" % (models_path, handle_hash)
    obj_detector_1 = hub.Module(module_handle)
    
    print(obj_detector_1.get_input_info_dict())
    print(obj_detector_1.get_output_info_dict())
    
    print("obj_detector done")

## 常见 TFHub 模型的下载

    #####################
    # Download some modules
    #####################
    
    classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/3"
    classifier_hash = hashlib.sha1(classifier_url.encode("utf8")).hexdigest()
    print("classifier_hash", classifier_hash)
    # classifier_hash 8ba51acecbfe5ceeaf1c04d6ee05b1703dd63bf0
    classifier = hub.Module(classifier_url)
    print("classifier done")
    
    feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/3"
    feature_extractor_hash = hashlib.sha1(feature_extractor_url.encode("utf8")).hexdigest()
    print("feature_extractor_hash", feature_extractor_hash)
    # feature_extractor_hash 9a40df43ae974de74f59ca892971f265fec3d319
    feature_extractor = hub.Module(feature_extractor_url)
    print("feature_extractor done")
    
    obj_detector2_url = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    obj_detector2_hash = hashlib.sha1(obj_detector2_url.encode("utf8")).hexdigest()
    print("obj_detector2_hash", obj_detector2_hash)
    # obj_detector2_hash 6e850c920451d5243d1fb87a3242c087535b9183
    obj_detector2 = hub.Module(obj_detector2_url)
    print("obj_detector2 done")

## 使用 TFHub 进行对象检测

[hello_tfhub_obj_detector.py](../../src/study_keras/4_hello_tfhub/hello_tfhub_obj_detector.py) 展示了直接使用 TFHub 预处理好的模型，进行对象检测。

    with tf.Graph().as_default():
        detector = hub.Module(module_handle)
    
        image_string_placeholder = tf.placeholder(tf.string)
        decoded_image = tf.image.decode_jpeg(image_string_placeholder)
        # Module accepts as input tensors of shape [1, height, width, 3], i.e. batch
        # of size 1 and type tf.float32.
        decoded_image_float = tf.image.convert_image_dtype(image=decoded_image, dtype=tf.float32)
        module_input = tf.expand_dims(decoded_image_float, 0)
    
        result = detector(module_input, as_dict=True)
    
        init_ops = [tf.global_variables_initializer(), tf.tables_initializer()]
    
        session = tf.Session()
        session.run(init_ops)
    
        #######################
        # detect images
        #######################
    
        image_paths = ['%s/Abyssinian_1.jpg' % images_dir,
                       '%s/Abyssinian_2.jpg' % images_dir,
                       '%s/Abyssinian_3.jpg' % images_dir]
    
        for image_path in image_paths:
            # image_path = download_and_resize_image(image_url, 640, 480)
            with tf.gfile.Open(image_path, "rb") as binfile:
                image_string = binfile.read()
    
            inference_start_time = time.clock()
    
            result_out, image_out = session.run(
                [result, decoded_image],
                feed_dict={image_string_placeholder: image_string})
    
            print("Found %d objects." % len(result_out["detection_scores"]))
            print("Inference took %.2f seconds." % (time.clock() - inference_start_time))
    
            image_with_boxes = draw_boxes(
                np.array(image_out), result_out["detection_boxes"],
                result_out["detection_class_entities"], result_out["detection_scores"])
    
            display_image(image_with_boxes)
    
## 结合使用 TFHub 和 Keras

[hello_tfhub_keras.py](../../src/study_keras/4_hello_tfhub/hello_tfhub_keras.py) 展示了直接使用 TFHub 和 Keras 之间的互动。

**注意** 这部分内容是 TF 2 的功能，在 TF 1.14 上没有实验成功。

    classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/3"
    
    IMAGE_SHAPE = (224, 224)
    
    # #####################
    # # Hub in Keras
    # #####################
    
    # TODO will return failure, maybe it is only for tensorflow 2.0
    classifier_layers = hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE + (3,))
    
    classifier = keras.Sequential([
        classifier_layers
    ])
    classifier.summary()
