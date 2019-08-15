# Hello Keras Applications 使用和可视化观察

下面的例子使用了宠物图像标注数据集，这些数据放置于 /data 目录下：

    $ cd data
    $ mkdir pets
    $ cd pets
    $ wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    $ wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
    $ tar -xvf images.tar.gz
    $ tar -xvf annotations.tar.gz

## MobileNet v1

[hello_kerasapp_mobilenetv1.py](../../src/study_keras/hello_kerasapp_mobilenetv1.py) 展示了基础的 MobileNet V1 进行图像分类的过程

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
    
## 网络卷积层的特征过滤器可视化

Keras 官方提供了一个卷积层可视化的例子，[https://keras.io/examples/conv_filter_visualization/](https://keras.io/examples/conv_filter_visualization/)。但是，这个例子没法在 tensorflow.keras 实现中运行。

下面的例子参考了 《Deep Learning with Python》 书中的实现，针对 MobileNet V2 做了卷积层的 filter 可视化。，不过最多仅仅显示前 64 个filter

    def normalize(x):
        """utility function to normalize a tensor.
    
        # Arguments
            x: An input tensor.
    
        # Returns
            The normalized input tensor.
        """
        return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())
    
    
    def deprocess_image(x):
        """utility function to convert a float array into a valid uint8 image.
    
        # Arguments
            x: A numpy-array representing the generated image.
    
        # Returns
            A processed numpy-array, which could be used in e.g. imshow.
        """
        # normalize tensor: center on 0., ensure std is 0.25
        x -= x.mean()
        x /= (x.std() + K.epsilon())
        x *= 0.25
    
        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)
    
        # convert to RGB array
        x *= 255
        if K.image_data_format() == 'channels_first':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x
    
    
    def generate_pattern(input_img, layer_output, filter_index=0, size=150, epochs=20):
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])
    
        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]
    
        # normalization trick: we normalize the gradient
        grads = normalize(grads)
    
        iterate = K.function([input_img], [loss, grads])
    
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random((1, 3, size, size))
        else:
            input_img_data = np.random.random((1, size, size, 3))
    
        input_img_data = (input_img_data - 0.5) * 20 + 128
    
        step = 1
        for i in range(epochs):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step  # ?? why input + grads ??
    
            if loss_value <= K.epsilon():
                if i == 0:
                    return None
                else:
                    break
    
        return deprocess_image(input_img_data[0])
    
    
    def visualize_layer_filters(model, layer_name, size=64, epochs=15):
        layer_output = model.get_layer(layer_name).output
        model_name = model.name
    
        if K.image_data_format() == 'channels_first':
            index_len = layer_output.shape.as_list()[1]
        else:
            index_len = layer_output.shape.as_list()[-1]
        print("%s filter length %d" % (layer_name, index_len))
    
        vol = 8
        row = math.ceil(index_len / vol)
        row = 8 if row > 8 else row  # most display first 64 filterss
    
        margin = 3
        results = np.zeros((row * size + (row - 1) * margin, vol * size + (vol - 1) * margin, 3)).astype('uint8')
    
        for i in range(row):
            for j in range(vol):
    
                if (i * vol + j) >= index_len:
                    break
    
                filter_image = generate_pattern(model.input, layer_output, filter_index=(i * vol + j), size=size,
                                                epochs=epochs)
    
                if filter_image is not None:
                    horizontal_start = i * size + i * margin
                    horizontal_end = horizontal_start + size
    
                    vertical_start = j * size + j * margin
                    vertical_end = vertical_start + size
    
                    results[horizontal_start:horizontal_end, vertical_start:vertical_end, :] = filter_image
    
                print(".%d" % (i * vol + j))
    
        plt.title(layer_name)
        plt.figure(figsize=(20, 20))
        plt.imshow(results)
    
        plt.show()
        cv2.imwrite("%s/%s.%s_filter.jpg" % (output_path, model_name, layer_name), results)

调用方式很简单，[hello_kerasapp_mobilenetv2_2.py](../../src/study_keras/hello_kerasapp_mobilenetv2_2.py) 的代码将 MobileNet V2 中所有的卷积层的 filter 都显示一遍。不过输出的纹理似乎并没有 VGG16 的那么清晰好看。

    # define the model
    model = mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=True,
                                         weights='imagenet', input_tensor=None, pooling=None, classes=1000)
    model.summary()

    #######################
    # draw Convolutional Filter
    #######################

    convLayers = [layer.name for layer in model.layers if (layer.__class__.__name__ == 'Conv2D')]
    for i in range(len(convLayers)):
        visualize_layer_filters(model=model, layer_name=convLayers[i], epochs=40)
        
## Predict 时各卷积层激活的可视化

    def visualize_activations(_pre, _layers_names, _activations, result_indexs=0):
        print(_layers_names)
    
        images_per_row = 8
        margin = 1
    
        for layer_name, layer_activation in zip(_layers_names, _activations):
            if K.image_data_format() == 'channels_first':
                n_features = layer_activation.shape[1]
                size = layer_activation.shape[-1]
            else:
                n_features = layer_activation.shape[-1]
                size = layer_activation.shape[1]
    
            n_cols = math.ceil(n_features / images_per_row)
            n_cols = 8 if n_cols > 8 else n_cols  # most display first 64 filterss
    
            display_grid = np.zeros((size * n_cols + (n_cols - 1) * margin,
                                     size * images_per_row + (images_per_row - 1) * margin)).astype('uint8')
    
            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = layer_activation[result_indexs, :, :, col * images_per_row + row]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
    
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
    
                    horizontal_start = col * size + col * margin
                    horizontal_end = horizontal_start + size
    
                    vertical_start = row * size + row * margin
                    vertical_end = vertical_start + size
    
                    display_grid[horizontal_start:horizontal_end, vertical_start:vertical_end] = channel_image
    
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()
    
            cv2.imwrite("%s/%s.%s_actication.jpg" % (output_path, _pre, layer_name), display_grid)

调用时，现需要指定需要输出的层次，利用 Model 构建出一个新的模型，然后对这个新模型做 predict。
[hello_kerasapp_mobilenetv2_3.py](../../src/study_keras/hello_kerasapp_mobilenetv2_3.py) 对这一行为做了示例。

    layers_names = [layer.name for layer in model.layers if (layer.__class__.__name__ == 'Conv2D')]
    layers_outputs = [layer.output for layer in model.layers if (layer.__class__.__name__ == 'Conv2D')]

    # display lastest %activation_len% activations
    activation_len = 0

    activation_model = Model(inputs=model.input, outputs=layers_outputs[activation_len:])
    activation_model.summary()
    
    results = activation_model.predict(pImg)

    visualize_activations("mobilenetv2_Abyssinian", layers_names[activation_len:], results, result_indexs=(len(pImg) - 1))

## 各卷积层与输入图像之间 heat-map 可视化

    def generate_heat_map(_model, input_image, layer_name, n_features):
        output = _model.output[:, n_features]
        selected_layer = _model.get_layer(layer_name)
    
        grads = K.gradients(output, selected_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
        iterate = K.function([_model.input], [pooled_grads, selected_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([input_image])
    
        for i in range(n_features):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
        heat_map = np.mean(conv_layer_output_value, axis=-1)
    
        heat_map = np.maximum(heat_map, 0)
        heat_map /= np.max(heat_map)
    
        return heat_map
    
    
    def merge_image_heat_map(input_image, input_heat_map):
        heat_map = cv2.resize(input_heat_map, (input_image.shape[1], input_image.shape[0]))
        heat_map = np.uint8(255 * heat_map)
        heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    
        superimposed_img = np.uint8(heat_map * 0.4 + 255 * input_image)
        return superimposed_img

调用示例在 [hello_kerasapp_mobilenetv2_4.py](../../src/study_keras/hello_kerasapp_mobilenetv2_4.py)。

    layers_names = [layer.name for layer in model.layers if (layer.__class__.__name__ == 'Conv2D')]

    prediction = model.predict(pImg)
    print(imagenet_utils.decode_predictions(prediction, top=3)[0])

    for layer_name in layers_names[1:-1]:
        layer_output = model.get_layer(layer_name).output
        model_name = model.name

        if K.image_data_format() == 'channels_first':
            n_features = layer_output.shape.as_list()[1]
        else:
            n_features = layer_output.shape.as_list()[-1]
        print("%s filter n_features %d" % (layer_name, n_features))

        heat_map = generate_heat_map(model, pImg, layer_name, n_features)

        plt.matshow(heat_map)
        plt.title(layer_name)
        plt.show()

        img = cv2.imread(test_img_path)
        composed_img = merge_image_heat_map(img, heat_map)
        plt.imshow(composed_img)
        plt.show()

        cv2.imwrite("%s/%s_%s.%s_heat_map.jpg" % (output_path, "Abyssinian", model_name, layer_name), composed_img)
