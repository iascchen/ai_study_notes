# Hello MNIST

[hello_mnist.py](../../src/study_keras/hello_mnist.py) 描述了一个最简单的 Keras 网络。

下面我们来简单的解释一下代码。

    import tensorflow as tf

## 加载数据部分

    mnist = tf.keras.datasets.mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

对数据除以 255 的原因是，原始图片记录的是 RGB 值，需要将其归一化到 [0, 1] 区间的浮点数。

**注意 Tips** 这也是在人工智能计算中经常使用的一种方法——将输入的数据变成 [0, 1] 区间的浮点数。

## 网络模型
 
下一段代码使用的 tf.keras.models.Sequential 描述了一个简单堆叠层的网络模型，可以想像数据是一层一层顺序流下去的。

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

* Flatten层：将输入的 28 * 28 的像素值，拉成一个长向量。这个 28 * 28 尺寸是和训练数据有关的。如果验证是输入的图片不是 28 * 28，需要 resize 成这个尺寸才能使用。
* Dense层1：全连接层，使用了 128 个神经元和 relu 激活。你也可以试试更多或更少的神经元数量，例如 512；以及其他的激活函数，例如 sigmoid。
* Dropout层：为了避免过拟合而增加，能够训练出鲁棒性更好的模型。这个参数可以改改试试，看看会有什么效果。
* Dense层2：全连接层，因为训练目标是实现多分类模型（0 - 9），所以，用了 10 个神经元，并且使用了 softmax 激活。
  
## 模型编译

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
tf.keras.Model.compile 采用三个重要参数：

* optimizer：此对象会指定训练过程。从 tf.train 模块向其传递优化器实例，例如 tf.train.AdamOptimizer、tf.train.RMSPropOptimizer 或 tf.train.GradientDescentOptimizer。
* loss：要在优化期间最小化的函数。常见选择包括均方误差 (mse)、categorical_crossentropy 和 binary_crossentropy。损失函数由名称或通过从 tf.keras.losses 模块传递可调用对象来指定。
* metrics：用于监控训练。它们是 tf.keras.metrics 模块中的字符串名称或可调用对象。

**注意** 此处 loss 使用 sparse_categorical_crossentropy，是与最后一层进行多分类 softmax 输出有关的。

## 模型训练
    
    model.fit(x_train, y_train, epochs=5)
 
tf.keras.Model.fit 除了需要输入训练集和结果之外，还可以采用三个参数来改变训练过程：

* epochs：以周期为单位进行训练。一个周期是对整个输入数据的一次迭代（以较小的批次完成迭代）。
* batch_size：当传递 NumPy 数据时，模型将数据分成较小的批次，并在训练期间迭代这些批次。此整数指定每个批次的大小。请注意，如果样本总数不能被批次大小整除，则最后一个批次可能更小。
* validation_data：在对模型进行原型设计时，您需要轻松监控该模型在某些验证数据上达到的效果。传递此参数（输入和标签元组）可以让该模型在每个周期结束时以推理模式显示所传递数据的损失和指标。

## 模型检验

    model.evaluate(x_test, y_test)