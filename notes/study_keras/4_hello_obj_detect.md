# 对象检测及其迁移学习

## 使用 TFHub 进行对象检测

[hello_obj_detect_tfhub.py](../../src/study_keras/hello_obj_detect_tfhub.py) 展示了直接使用 TFHub 预处理好的模型，进行对象检测。

下面的代码展示了从 tfhub.dev 云端下载模型。你可以通过设置环境变量，指定下载模型的本地存储地址。

    export TFHUB_CACHE_DIR=/my_module_cache
    
    module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
        
    with tf.Graph().as_default():
        detector = hub.Module(module_handle)

还可以直接使用已下载好的本地模型，用于 TFHub 模型创建。
这个目录下必须有 tfhub_module.pb 和 saved_model.pb 文件用于具体的模型加载。

    # module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    module_handle = "%s/ssd-mobilenet_v2" % models_path
    
    with tf.Graph().as_default():
        detector = hub.Module(module_handle)
        

