# Hello Keras Applications

下面的例子使用了宠物图像标注数据集，这些数据放置于 /data 目录下：

    $ cd data
    $ wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    $ wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
    $ tar -xvf images.tar.gz
    $ tar -xvf annotations.tar.gz

## MobileNet v1

[hello_kerasapp_mobilenet_v1.py](../../src/study_keras/hello_kerasapp_mobilenet_v1.py) 展示了基础的 MobileNet V1 进行图像分类的过程

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
    
    # define the model
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
    
## 观察网络层的调整
    
[hello_kerasapp_mobilenet_v2.py](../../src/study_keras/hello_kerasapp_mobilenet_v2.py) 增加了更多的对 MobileNet V2 网络层状态的观察。




