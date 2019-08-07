# Hello MNIST

[hello_mnist.py](../src/hello_mnist.py) 描述了一个最简单的 Keras 网络。

下面我们来简单的解释一下代码。

    import tensorflow as tf

加载数据部分。对数据除以 255 的原因是，原始图片记录的是 RGB 值，需要将其归一化到 [0, 1] 区间的浮点数。

**注意 Tips ** 这也是在人工智能计算中经常使用的一种方法——将输入的数据变成 [0, 1] 区间的浮点数。
    
    mnist = tf.keras.datasets.mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
 
下一段描述了一个顺序操作的网络模型，可以想像数据是一层一层流下去的。

* Flatten层：将 28*28 的像素值，拉成一个长向量。这个 28*28 尺寸是和训练数据有关的。如果验证是输入的图片不是 28*28，需要resize成这个尺寸才能使用。
* Dense层1：全连接层，使用了 128 个神经元和 relu 激活。你也可以试试其他的激活函数，例如：sigmoid。
* Dropout层：为了避免过拟合而增加，能够训练出鲁棒性更好的模型。
* Dense层2：全连接层，因为训练目标是实现多分类模型（0-9），所以，用了 10 个神经元，并且使用了 softmax 激活。
    
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
模型编译，绑定 loss ，optimizer ，和 metrics。
此处使用的 sparse_categorical_crossentropy 作 loss，是与最后一层进行多分类 softmax 输出有关的。

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

模型训练，训练 epochs 轮。
    
    model.fit(x_train, y_train, epochs=5)
    
模型检验。

    model.evaluate(x_test, y_test)

