# 迁移学习

本节内容会用到以下数据集：

    $ cd data
    $ wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    $ tar -xvf flower_photos.tgz

[hello_transfer_learning.py](../../src/study_keras/hello_transfer_learning.py) 展示了迁移学习的基本过程。

## 使用 ImageDataGenerator 创建训练和检验

### 方法一，class_mode='categorical'

    # Method 1: use categorical
    train_datagen = ImageDataGenerator(preprocessing_function=mobilenet_v2.preprocess_input,
                                       validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
                                                        color_mode='rgb', batch_size=32, shuffle=True,
                                                        class_mode='categorical', subset='training')
    validation_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
                                                             color_mode='rgb', batch_size=32, shuffle=True,
                                                             class_mode='categorical', subset='validation')

创建训练数据。传递的 validation_split=0.2 参数用于创建训练集和验证集。
    
    train_datagen = ImageDataGenerator(preprocessing_function=mobilenet_v2.preprocess_input,
                                       validation_split=0.2)
                                           
使用 subset='training' 获得训练集。
                                           
    train_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
                                                        color_mode='rgb', batch_size=32, shuffle=True,
                                                        class_mode='categorical', subset='training')
    step_size_train = train_generator.n // train_generator.batch_size                                                   
                                                            
使用 subset='validation' 获得验证集。

    validation_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
                                                             color_mode='rgb', batch_size=32, shuffle=True,
                                                             class_mode='categorical', subset='validation')
    
    step_size_validation = validation_generator.n // validation_generator.batch_size

因为此处使用的 class_mode='categorical', 所以，模型 compile 时候，需要使用 loos = 'categorical_crossentropy' 算法。

    loss_alg = 'categorical_crossentropy'

可以用下面的代码检查一下生成的图像数据

    # show data format of train_generator and validation_generator
    i = 0
    for inputs_batch, labels_batch in validation_generator:
        if i < 1:
            print(inputs_batch)
            print(labels_batch)
            i += 1
        else:
            break

### 方法二，使用对 ImageNetLabels 扩展的分类 ID

首先获取 ImageNetLabels 的类列表

    # Method 2: use image net class
    # Before do this, please add new classes at the end of ImageNetLabels.txt. each class per line
    # classes: daisy、dandelion、roses、sunflowers、tulips
    labels_path = "ImageNetLabels.txt"
    imagenet_labels = np.array(open(labels_path).read().splitlines())[1:].tolist()  # drop first line 'background'
    # print(len(imagenet_labels), imagenet_labels)

下面的代码里是用了 classes=imagenet_labels, class_mode='sparse', 这样的参数，用来将label设定为在 ImageNetLabels 基础上的扩展。

    train_datagen = ImageDataGenerator(preprocessing_function=mobilenet_v2.preprocess_input,
                                       validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
                                                        color_mode='rgb', batch_size=32, shuffle=True,
                                                        classes=imagenet_labels, class_mode='sparse',
                                                        subset='training')
    validation_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
                                                             color_mode='rgb', batch_size=32, shuffle=True,
                                                             classes=imagenet_labels, class_mode='sparse',
                                                             subset='validation')

因为此处使用的 class_mode='sparse', 所以，模型 compile 时候，需要使用 loos = 'sparse_categorical_crossentropy' 算法。
    
    # because its class_mode='sparse'                                                         
    loss_alg = 'sparse_categorical_crossentropy'
    
## 从 keras application 构建迁移学习的模型

[hello_transfer_learning.py](../../src/study_keras/hello_transfer_learning.py) 展示了如何在已经训练好的模型上进行迁移学习并验证的过程

获取已经训练好的基础模型。include_top=False 去掉了最后的 Dense 分类层，因为我们训练出新的5各类别，所以不需要这一层。
    
    ###################
    # Define transfer learning model
    ###################

    # define the model
    base_model = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    base_model.summary()
    
### 方法一，直接使用 keras.Sequential

在 base_model 的基础上，增加新的分类结构。
    
    ###################
    # Method 1

    model_name = "mobilenetv2_transfer_seq"

    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(5, activation='softmax')
    ])
   
在这个模型里，最后增加的 5 层都是需要重新训练的。而因为设置了 base_model.trainable = False ，所以 base_model 的权重不会重复训练。

如果需要对 base_model 进行更细节的 trainable 设置，可以参考下面的代码，不同层级进行 layer.trainable = True 的设定。
例如：如果需要对 MobileNet v2 的最后 3 层进行微调，可以使用类似下面的代码：

    # adjust base model,
    trainable_base_layers = -3
    for layer in base_model.layers[trainable_base_layers:]:
        layer.trainable = True

    layers_names = [layer.name for layer in base_model.layers if layer.trainable is True]
    print("base_model trainable layers:", layers_names)

    layers_names = [layer.name for layer in model.layers if layer.trainable is True]
    print("model trainable layers:", layers_names)
        
修改模型 trainable 之后，在进行训练之前，请务必对模型进行 compile，以使设定生效。请注意，此处的 loss 选择和数据集有关。

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

### 方法二，使用在 keras.Model 扩展模型。

下面的例子展示了对模型更精细的控制。截取原模型中的一段，组合成新的模型，并进行训练。
可以看到，截取了 MobileNet V2 中层 block_16_project_BN 的输出，然后组合到新的模型中，
利用 model = keras.Model(inputs=base_model.input, outputs=preds) 构造出模型。这个技巧，在 SSD-MobileNet 等算法实现时，可以用得着。
    
    ##################
    # Method 2

    model_name = "mobilenetv2_transfer_model"

    x = base_model.get_layer('block_16_project_BN').output
    x = keras.layers.Conv2D(1280, kernel_size=(1, 1))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Flatten()(x)
    preds = keras.layers.Dense(5, activation='softmax')(x)  # final layer with softmax activation
    model = keras.Model(inputs=base_model.input, outputs=preds)
        
这样构建出的新模型是一个网络。下面列出了这个网络的各层。
        
    layers_names = [layer.name for layer in model.layers]
    print("All layers:", layers_names)

    layers_names = [layer.name for layer in model.layers if layer.trainable is True]
    print("Trainable layers:", layers_names)
    
## 训练

训练之前，必须使用 model.compile 来使上面代码设置的 trainable 生效。

    model.compile(optimizer='adam', loss=loss_alg, metrics=['accuracy']) 

此时的模型需要使用 fit_generator、evaluate_generator、predict_generator 来使用这些 DataGenerator 产生的数据。
    
    history = model.fit_generator(train_generator, steps_per_epoch=step_size_train,
                                  validation_data=validation_generator, validation_steps=step_size_validation,
                                  epochs=5, callbacks=[tp_callback])
    print("Train history : ", history.history)

    val_results = model.evaluate_generator(validation_generator, steps=step_size_validation)
    print("Evaluate result : ", val_results)

    results = model.predict_generator(validation_generator, steps=step_size_validation)
    result_class = [np.argmax(result) for result in results]
    print("predict result : ", result_class)
    
## 使用 estimator 进行训练

Estimator 是一种可极大地简化机器学习编程的高阶 TensorFlow API。
Estimator 会封装下列操作：训练、评估、预测、导出以供使用。

Estimator 具有下列优势：可以在本地主机上或分布式多服务器环境中运行基于 Estimator 的模型，而无需更改模型。
此外，可以在 CPU、GPU 或 TPU 上运行基于 Estimator 的模型，而无需重新编码模型。

[hello_transfer_learning_2.py](../../src/study_keras/hello_transfer_learning_2.py) 展示了迁移学习模型在 Estimator 上运行的基本过程。

    ###################
    # Use tensorflow estimator
    ###################

    est_model = keras.estimator.model_to_estimator(keras_model=keras_model)

    train_x = []
    train_y = []
    for inputs_batch, labels_batch in validation_generator:
        train_x.append(inputs_batch)
        train_y.append(labels_batch)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=train_x, y=train_y, num_epochs=1, shuffle=False)

    my_listener = MyListener()

    # To train, we call Estimator's train function:
    est_model.train(input_fn=train_input_fn, steps=5, saving_listeners=my_listener)
