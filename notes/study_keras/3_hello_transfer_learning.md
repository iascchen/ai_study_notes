# 迁移学习

本届内容会用到以下数据集：

    wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    tar -xvf flower_photos.tgz
    
## 从 keras application 构建迁移学习的模型

[hello_transfer_learning.py](../../src/study_keras/hello_transfer_learning.py) 展示了如何在已经训练好的模型上进行迁移学习并验证的过程

获取已经训练好的模型
    
        # define the model
        base_model = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
        base_model.trainable = False
        base_model.summary()
    
操作方法一，直接在 keras.Sequential 操作。
    
        # Method 1
        model = keras.Sequential([
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(5, activation='softmax')
        ])
        model.summary()

操作方法二，使用在 keras.Model 扩展模型。
    
        # Method 2
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1024, activation='relu')(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        preds = keras.layers.Dense(5, activation='softmax')(x)  # final layer with softmax activation
        model = keras.Model(inputs=base_model.input, outputs=preds)
        model.summary()

这两种方法会有细微的区别，可以从下面对不同层级进行 layer.trainable = True 的设定结果，可以看出：

* 如果使用 Sequential，base_model 会作为一个整体被设置为可以训练。
* 如果使用 Model，则可以训练属性能够设置到具体的没个细节层里。 

        
        layers_names = [layer.name for layer in model.layers]
        print("All layers:", layers_names)
        
        layers_names = [layer.name for layer in model.layers if layer.trainable is True]
        print("Trainable layers 1:", layers_names)
        
        trainable_top_layers = -8
        # for layer in model.layers[:trainable_top_layers]:
        #     layer.trainable = False
        for layer in model.layers[trainable_top_layers:]:
            layer.trainable = True
       
        layers_names = [layer.name for layer in model.layers if layer.trainable is True]
        print("Trainable layers 2:", layers_names)
    
## 使用 ImageDataGenerator 创建训练和检验

创建训练数据。传递的 validation_split=0.2 参数用于创建训练集和验证集。
    
        train_datagen = ImageDataGenerator(preprocessing_function=mobilenet_v2.preprocess_input,
                                           validation_split=0.2)
                                           
使用 subset='training' 获得训练集。
                                           
        train_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
                                                            color_mode='rgb', batch_size=32, shuffle=True,
                                                            class_mode='categorical', subset='training')
                                                            
使用 subset='validation' 获得验证集。

        validation_generator = train_datagen.flow_from_directory(images_dir, target_size=(224, 224),
                                                                 color_mode='rgb', batch_size=32, shuffle=True,
                                                                 class_mode='categorical', subset='validation')
    
        step_size_train = train_generator.n // train_generator.batch_size
        step_size_validation = validation_generator.n // validation_generator.batch_size

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
