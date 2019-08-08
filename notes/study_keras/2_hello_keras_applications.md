# Hello Keras Applications

下面的例子使用了宠物图像标注数据集，这些数据放置于 /data 目录下：

    $ cd data
    $ wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    $ wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
    $ tar -xvf images.tar.gz
    $ tar -xvf annotations.tar.gz

## MobileNet v1

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
    results = imagenet_utils.decode_predictions(prediction)