# Hello Keras Applications

下面的例子使用了宠物图像标注数据集，这些数据放置于 /data 目录下：

    $ cd data
    $ wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    $ wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
    $ tar -xvf images.tar.gz
    $ tar -xvf annotations.tar.gz

## MobileNet v1

[hello_kerasapp_mobilenet_v1.py](../../src/study_keras/hello_kerasapp_mobilenetv1.py) 展示了基础的 MobileNet V1 进行图像分类的过程

能够创建不同精度的模型

    model_25 = MobileNet()
    model_25.summary()
    
    model_50 = MobileNet(input_shape=None, alpha=0.5, depth_multiplier=1, dropout=1e-3, include_top=True,
                         weights='imagenet', input_tensor=None, pooling=None, classes=1000)
    model_50.summary()
    
    model_75 = MobileNet(input_shape=None, alpha=0.75, depth_multiplier=1, dropout=1e-3, include_top=True,
                         weights='imagenet', input_tensor=None, pooling=None, classes=1000)
    model_75.summary()
    
    model_100 = MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True,
                          weights='imagenet', input_tensor=None, pooling=None, classes=1000)
    model_100.summary()
    
推理使用也很简单    
    
    prediction = model_100.predict(pImg)
    
    # obtain the top-3 predictions
    results = imagenet_utils.decode_predictions(prediction, top=3)
    
展示一下生成的模型图片

![mobilenet](hello_kerasapp_mobilenet_model_plot.png)

## 切换多种 Keras 模型

[hello_kerasapp_vgg16.py](../../src/study_keras/hello_kerasapp_vgg16.py) 展示了使用各种 Keras Applications 中的预训练网络。


载入合适的网络

    # from tensorflow.keras.applications import densenet
    # from tensorflow.keras.applications import inception_v3
    # from tensorflow.keras.applications import mobilenet
    # from tensorflow.keras.applications import mobilenet_v2
    # from tensorflow.keras.applications import nasnet
    from tensorflow.keras.applications import vgg16
    
对输入图片进行不同的预处理
    
    # pImg = densenet.preprocess_input(img_array)
    # pImg = inception_v3.preprocess_input(img_array)
    # pImg = mobilenet.preprocess_input(img_array)
    # pImg = mobilenet_v2.preprocess_input(img_array)
    # pImg = nasnet.preprocess_input(img_array)
    pImg = vgg16.preprocess_input(img_array)
    
定义不同的模型
    
    # # define the model
    #
    # model = densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
    #                              pooling=None, classes=1000)
    #
    # model = inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
    #                                  pooling=None, classes=1000)
    #
    # model = mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True,
    #                             weights='imagenet', input_tensor=None, pooling=None, classes=1000)
    #
    # model = mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=True,
    #                                  weights='imagenet', input_tensor=None, pooling=None, classes=1000)
    #
    # model = nasnet.NASNetMobile(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None,
    #                             classes=1000)    
    
    model = vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,
                        classes=1000)
                        
    model.summary()

执行的时候需要先下载 Keras 预先计算好的 h5 模型，可能会比较慢。
    
## 网络层的可视化
    
[hello_kerasapp_mobilenet_v2.py](../../src/study_keras/hello_kerasapp_mobilenetv2.py) 参考 Keras 官方的例子，对 MobileNet V2 的卷积神经网络中各层的 filter 进行可视化观察。

Keras 官方提供了一个卷积层可视化的例子，[https://keras.io/examples/conv_filter_visualization/](https://keras.io/examples/conv_filter_visualization/)。但是，这个例子没法在 tensorflow.keras 实现中运行。

下面的例子参考了 《Deep Learning with Python》 书中的实现，针对 MobileNet V2 做了卷积层的 filter 可视化。，不过做多仅仅显示前 64 个filter

    def generate_pattern(model, layer_name, filter_index=0, size=150, epochs=15):
        layer_output = model.get_layer(layer_name).output
    
        loss = K.mean(layer_output[:, :, :, filter_index])
    
        grads = K.gradients(loss, model.input)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())
    
        iterate = K.function([model.input], [loss, grads])
        input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    
        step = 1
        for i in range(epochs):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step
    
        img = input_img_data[0]
        return deprocess_image(img)
    
    
    def visualize_layer_filters(model, layer_name, filters_size=16, size=64, epochs=15):
        margin = 3
        results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3)).astype('uint8')
    
        # most display first 64 filters
        for i in range(8):
            for j in range(8):
    
                if (i * 8 + j) >= filters_size:
                    break
    
                filter_image = generate_pattern(model, layer_name, filter_index=(i * 8 + j), size=size, epochs=epochs)
    
                horizontal_start = i * size + i * margin
                horizontal_end = horizontal_start + size
    
                vertical_start = j * size + j * margin
                vertical_end = vertical_start + size
    
                results[horizontal_start:horizontal_end, vertical_start:vertical_end, :] = filter_image
                print(".%d" % (i * 8 + j))
    
        plt.figure(figsize=(20, 20))
        plt.imshow(results)
    
        plt.show()
        cv2.imwrite("%s/%s_filter.jpg" % (output_path, layer_name), results)

调用方式很简单。

    layer_names = [
        'Conv1',
    
        'block_1_project', 'block_2_project',
        'block_3_project', 'block_4_project', 'block_5_project',
        'block_6_project', 'block_7_project', 'block_8_project', 'block_9_project',
        'block_10_project', 'block_11_project', 'block_12_project',
        'block_13_project', 'block_14_project', 'block_15_project',
        'block_16_project',
    
        'Conv_1',
    ]
    
    filters_size = [
        32,
        24, 24,
        32, 32, 32,
        64, 64, 64, 64,
        96, 96, 96,
        160, 160, 160,
        320,
        1280,
    ]
    
    for i in range(len(layer_names)):
        visualize_layer_filters(model=model_100, layer_name=layer_names[i], epochs=40, filters_size=filters_size[i])



