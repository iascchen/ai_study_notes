# Hello MNIST

## 从最简版本开始

[hello_mnist.py](../../src/study_keras/1_hello_mnist/hello_mnist.py) 描述了一个最简单图片单分类的 Keras 网络。

下面我们来简单的解释一下代码。

**请注意** ： 代码使用的 keras 为 **tensorflow 2.0 版本的 keras** 实现，在后面的文字说明中使用 tf.keras 指代。。

    from tensorflow import keras

### 加载数据部分

    mnist = keras.datasets.mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

对数据除以 255 的原因是，原始图片记录的是整数的 RGB 值，需要将其归一化到 [0, 1] 区间的浮点数。

**注意 Tips** 这也是在人工智能计算中经常使用的一种方法——将输入的数据变成 [0, 1] 区间的浮点数。

### 网络模型
 
下一段代码使用的 tf.keras.models.Sequential 描述了一个简单堆叠层的网络模型，可以想像数据是一层一层顺序流下去的。

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

* Flatten层：将输入的 28 * 28 的像素值，拉成一个长向量。这个 28 * 28 尺寸是和训练数据有关的。如果验证时输入的图片不是 28 * 28，需要 resize 成这个尺寸才能使用。
* Dense层1：全连接层，使用了 128 个神经元和 relu 激活。你也可以试试更多或更少的神经元数量，例如 512；以及其他的激活函数，例如 sigmoid。
* Dropout层：为了避免过拟合而增加，能够训练出鲁棒性更好的模型。这个参数可以改改试试，看看会有什么效果。
* Dense层2：全连接层，因为训练目标是实现多分类模型（0 - 9），所以，用了 10 个神经元，并且使用了 softmax 激活。
  
### 模型编译

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
tf.keras.Model.compile 采用三个重要参数：

* optimizer：此对象会指定训练过程。从 tf.keras.optimizers 模块向其传递优化器实例，例如 tf.keras.optimizers.Adam、tf.keras.optimizers.RMSProp 或 tf.keras.optimizers.SGD。
* loss：要在优化期间最小化的函数。是 tf.keras.losses.Loss (即 tf.losses.Loss) 的实例。常见选择包括均方误差 (mse)、categorical_crossentropy 和 binary_crossentropy。
* metrics：用于监控训练。它们是 tf.keras.metrics (即 tf.metrics) 模块中的字符串名称或可调用对象。

**注意** 此处 loss 使用 sparse_categorical_crossentropy，是与最后一层进行多分类 softmax 输出有关的。

### 模型训练
    
    model.fit(x_train, y_train, epochs=5)
 
tf.keras.Model.fit 除了需要输入训练集和结果之外，还可以采用三个参数来改变训练过程：

* epochs：以周期为单位进行训练。一个周期是对整个输入数据的一次迭代（以较小的批次完成迭代）。
* batch_size：当传递 NumPy 数据时，模型将数据分成较小的批次，并在训练期间迭代这些批次。此整数指定每个批次的大小。请注意，如果样本总数不能被批次大小整除，则最后一个批次可能更小。
* validation_data：在对模型进行原型设计时，您需要轻松监控该模型在某些验证数据上达到的效果。传递此参数（输入和标签元组）可以让该模型在每个周期结束时以推理模式显示所传递数据的损失和指标。

### 模型检验

    model.evaluate(x_test, y_test)
    
## 保存和加载已经训练好的模型

[hello_mnist_1.py](../../src/study_keras/1_hello_mnist/hello_mnist_1.py) 对上面的例子做了些扩展，增加了对于模型的保存和加载

### 模型的展示

    model.summary()
    
这是用于观察模型的常用命令，输出如下：
    
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 784)               0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               100480    
    _________________________________________________________________
    dropout (Dropout)            (None, 128)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 101,770
    Trainable params: 101,770
    Non-trainable params: 0

### 模型的可视化

    base_path = "../../output"
    
    # use plot_model need graphviz be installed
    model_plot_path = "%s/hello_mnist_1_model_plot.png" % base_path
    keras.utils.plot_model(model, to_file=model_plot_path, show_shapes=True, show_layer_names=True)

执行这段代码，必需首先安装 graphviz 工具，

    # MAC OS
    $ sudo chown -R $(whoami) /usr/local/sbin
    $ brew install graphviz
    
    # Linux Ubuntu
    $ sudo apt-get install graphviz

### 模型的训练的历史
    
    history = model.fit(x_train, y_train, epochs=5)
    history_dict = history.history
    print("History : %s" % history_dict)
    print("History Keys : %s" % history_dict.keys())

这段代码增加了更多的一些信息，能够帮助我们检查训练的效果。这里能够得到每次 epoch 的 metrics 数值，上述代码显示为：
    
    History : {'loss': [0.29361882321139177, 0.14563049617434543, 0.10849536302785079, 0.087534287218377, 0.0756100890989105], 
        'accuracy': [0.91428334, 0.95715, 0.9677333, 0.97333336, 0.9769167]}
    History Keys : dict_keys(['loss', 'accuracy'])

其中的 loss 和 acc 是模型的 metrics 指标，这些值和后面模型 evaluate 的结果是对应的。

### 模型的验证

    result = model.evaluate(x_test, y_test)
    print('Evaluate Result:', result)

loaded_model.evaluate 返回的结果是与 model.compile 时使用的 metrics=['accuracy'] 参数有关。不同优化器能够返回的 metrics 会不同。

上面代码运行，得到的结果是如下，这个数组对应 [loss, acc] 的值。
    
    Evaluate Result: [0.07440505252983422, 0.9768]

### 模型的存储和加载

下面的代码，将训练的模型存储起来，格式为 Keras H5。

在后面学习中，你可能会遇到很多已经存储好的 Tensorflow 模型，需要注意的是，这些模型之间可能并不互相通用。这个部分后续再详细介绍。

    # 模型存储
    base_path = "../../output"
    
    h5_path = "%s/hello_mnist_1.h5" % base_path
    model.save(h5_path)

Keras H5 模型的加载也很简单，使用 tf.keras.models.load_model 即可实现。

    # 模型加载
    loaded_model = keras.models.load_model(h5_path)
    loaded_model.summary()

### 模型的推理 predict 

    predictions = loaded_model.predict(x_test)
    predictions_labels = [np.argmax(predictions[i]) for i in range(len(x_test))]
    
    index = np.arange(0, len(y_test))
    diff = index[predictions_labels != y_test]
    print("%d differences: \n%s" % (len(diff), diff))

利用这段代码，我们比较了对测试集 predict 的误差，及其在测试集中的 index 位置。

    232 differences: 
    [ 247  259  290  321  340  381  445  447  449  582  659  691  707  720
      740  877  924  947  951  956 1014 1039 1112 1181 1182 1194 1202 1226
     1232 1242 1247 1260 1283 1299 1319 1326 1364 1393 1494 1500 1522 1527
     1530 1549 1553 1609 1621 1681 1709 1717 1737 1754 1790 1800 1901 1941
     1952 1982 1984 2004 2016 2018 2024 2035 2040 2044 2053 2070 2098 2109
     2118 2130 2135 2182 2272 2293 2308 2369 2387 2406 2408 2414 2447 2488
     2534 2607 2648 2654 2927 2939 2953 2995 3005 3060 3073 3117 3289 3330
     3405 3474 3503 3520 3549 3550 3558 3559 3567 3597 3718 3727 3751 3767
     3776 3780 3796 3811 3838 3853 3893 3902 3906 3926 3941 3946 3985 4018
     4063 4065 4075 4078 4156 4176 4224 4248 4289 4306 4369 4425 4433 4477
     4497 4504 4534 4536 4547 4567 4601 4639 4751 4807 4814 4823 4860 4880
     4956 4990 5138 5159 5331 5457 5495 5573 5623 5642 5734 5749 5842 5888
     5936 5937 5955 5973 6045 6059 6390 6505 6511 6555 6571 6574 6576 6597
     6603 6608 6625 6651 6744 6755 6783 6847 7186 7216 7434 7451 7545 7849
     8020 8094 8246 8255 8277 8311 8325 8362 8406 8408 8527 9009 9015 9019
     9024 9163 9211 9280 9587 9634 9642 9679 9692 9698 9700 9729 9745 9749
     9768 9770 9777 9779 9792 9839 9879 9904]


## 单独保存和加载模型和权重

[hello_mnist_2.py](../../src/study_keras/1_hello_mnist/hello_mnist_2.py) 展示例如何单独保存模型和权重

### 单独保存模型结构

**注意** loaded_model 必须与原模型完全一致。

    base_path = "../../output"
    
保存和加载 Json 格式的模型，加载的模型必须 compile 才能使用
    
    json_path = "%s/hello_mnist_2.model.json" % base_path
    json_string = model.to_json()
    with open(json_path, 'w') as fw:
        fw.write(json_string)
    
    with open(json_path, 'r') as fr:
        new_json_string = fr.read()
    json_model = keras.models.model_from_json(new_json_string)
    
    json_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

保存和加载 YAML 格式的模型，加载的模型必须 compile 才能使用
    
    yaml_path = "%s/hello_mnist_2.model.yaml" % base_path
    yaml_string = model.to_yaml()
    with open(yaml_path, 'w') as fw:
        fw.write(yaml_string)
    
    with open(yaml_path, 'r') as fr:
        new_yaml_string = fr.read()
    yaml_model = keras.models.model_from_yaml(new_yaml_string)
    
    yaml_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

您可以打开这两个生成的文件看看，里面有各种可以配置的参数，初步了解一下能够对网络模型做哪些参数定制，为以后学习网络优化做个铺垫。

// TODO
**注意**：子类化模型不可序列化。感觉 Keras 的子类模型，就是一个样子货。（v1.14。v2.0待测）

### 单独保存 weights

**注意** loaded_model 必须与原模型完全一致。

保存和加载 TensorFlow Checkpoint 格式的权重文件。会生成 .index 和 .data* 的一系列文件
    
    weight_path = "%s/hello_mnist_2.weights" % base_path
    model.save_weights(weight_path)
    
    json_model.load_weights(weight_path)
    

保存和加载 Keras 格式的权重文件。生成单独的 .h5 文件。
    
    h5_weight_path = "%s/hello_mnist_2.weights.h5" % base_path
    model.save_weights(h5_weight_path, save_format='h5')
    
    yaml_model.load_weights(h5_weight_path)

## 模型可视化以及观察训练的过程

[hello_mnist_3.py](../../src/study_keras/1_hello_mnist/hello_mnist_3.py) 增加了一些代码，是我们能够直观的观察到网络训练的过程，并利用 tensorboard 可视化出来。

### 利用 Callback 保存训练过程中的 weights

在 model.fit 中增加了两个 callback 函数，用来输出训练过程中的信息，便于观察训练过程。

    history = model.fit(x_train, y_train, epochs=train_epochs, callbacks=[cp_callback, tp_callback])

cp_callback 用于输出运算过程中的 Checkpoint 值。period 指定每两次迭代保存一次
    
    save_freq = 'epoch'
    
    checkpoint_path = "%s/hello_mnist_3-{epoch:04d}.ckpt" % base_path
    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, save_freq=save_freq)                                                   

### 在 Tensorboard 里显示训练过程
                                                     
tp_callback 用于输出 Tensorboard 可用的日志。使用 "tensorboard --logdir %logdir%" 命令打开 TensorBoard。
然后在浏览器中访问 http://localhost:6006 

    ##################
    # $ tensorboard --logdir base_path/hello_mnist_3.logs/
    ##################
    
    log_path = "%s/hello_mnist_3.logs" % base_path
    tp_callback = keras.callbacks.TensorBoard(log_dir=log_path, write_graph=True, write_images=True,
                                              histogram_freq=0, embeddings_freq=0, embeddings_layer_names=None,
                                              embeddings_metadata=None)

### 显示训练 History Plot 图表

    def random_color(number_of_colors):
        return ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(number_of_colors)]
    
    
    def draw_history(_history):
        history_dict = _history.history
    
        keys = history_dict.keys()
        keys_list = list(keys)
        print(keys_list)
    
        values = [history_dict[keys_list[i]] for i in range(len(keys_list))]
        labels = [keys_list[i] for i in range(len(keys_list))]
        colors = random_color(len(keys_list))
    
        epochs = range(1, len(values[0]) + 1)
    
        for i in range(len(keys_list)):
            plt.plot(epochs, values[i], colors[i], label=labels[i])
    
        plt.xlabel('Epochs')
        plt.legend()
    
        plt.show()

    
    draw_history(history)
   
## 使用 one-hot 编码数据进行训练

[hello_mnist_4.py](../../src/study_keras/1_hello_mnist/hello_mnist_4.py) 对于 Label 数据进行了变换，采用 ont-hot 编码构建了结果向量。
使用这种编码方式，能够更容易理解对图片进行多分类的训练场景。

    y_train_one_hot = keras.utils.to_categorical(y_train)
    y_test_one_hot = keras.utils.to_categorical(y_test)
    
下面的输出，展示这两种结果集编码前10条记录的差异。

    y_train 
    [5 0 4 1 9 2 1 3 1 4]
    y_train_one_hot 
    [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]  
  
此处的 loss 需要做个改变了，直接使用 categorical_crossentropy。
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train_one_hot, epochs=5)

## 模型运用到 web 和 APP

[hello_mnist_5.py](../../src/study_keras/1_hello_mnist/hello_mnist_5.py) 展示了将训练好的模型转换成 tensorflow.js 和 tensorflow lite 的过程，能够将这个模型运用于 web 或 APP。

输出成 tensorflow Lite 的模型。这个实现和 tf v1.14 的代码不相同。
        
    tflite_path = "%s/hello_mnist_5.tflite" % base_path

    converter = lite.TFLiteConverter.from_keras_model(loaded_model)
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as fw:
        fw.write(tflite_model)
        
输出成 tensorflow.js 的模型。

因为 tensorflow.js 目前版本 (v1.2.10.1) 仅支持 tensorflow 1.14, 安装时会自动安装 tf 1.14。
所以，需要在安装 tensorflow.js 之后再安装一遍 tensorflow 2.0.0。
在 tf2 下运行能够生成 tfjs 的模型。

    tfjs_path = "%s/hello_mnist_5.tfjs" % base_path
    tfjs.converters.save_keras_model(loaded_model, tfjs_path)
    
