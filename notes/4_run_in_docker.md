# 快速开始：在 Docker 中使用本例

## 准备Host环境

为了便于管理数据，我们将Docker Container所用的数据全部放在 Host 上的 /data 目录下。

    $ sudo mkdir /data
    $ sudo chmod 777 /data

启动 Docker 容器

本来想在 tf 1.14.0 版本上测试验证，但是发现无法初始化 cuDNN。回退到 1.13.2 版本上的 Docker，实测 GPU 在 Docker 内可用，但是感觉还是慢。 

    $ docker run --name=my_tf_1.13 --volume=/data/my_tf_1.13:/data -w=/data --gpus all -it tensorflow/tensorflow:1.13.2-gpu-py3-jupyter bash
    ________                               _______________
    ___  __/__________________________________  ____/__  /________      __
    __  /  _  _ \_  __ \_  ___/  __ \_  ___/_  /_   __  /_  __ \_ | /| / /
    _  /   /  __/  / / /(__  )/ /_/ /  /   _  __/   _  / / /_/ /_ |/ |/ /
    /_/    \___//_/ /_//____/ \____//_/    /_/      /_/  \____/____/|__/
    
    
    WARNING: You are running this container as root, which can cause new files in
    mounted volumes to be created as the root user on your host machine.
    
    To avoid this, run the container by specifying your user's userid:
    
    $ docker run -u $(id -u):$(id -g) args...
    
    root@2f2cda53bd14:/data#

如果需要重新调整 Docker 参数，可以删掉已有的 Container

    $ docker stop my_tf_1.13
    $ docker rm my_tf_1.13

如果重新启动容器，您可以使用docker exec重用容器。

    $ docker start my_tf_1.13
    $ docker exec -it my_tf_1.13 bash
    root@2f2cda53bd14:/data#

环境准备GPU检查

    root@2f2cda53bd14:/data# nvidia-smi
    Tue Aug  6 09:31:03 2019
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 430.26       Driver Version: 430.26       CUDA Version: 10.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce RTX 2070    Off  | 00000000:01:00.0  On |                  N/A |
    |  0%   40C    P8    20W / 175W |    222MiB /  7979MiB |      1%      Default |
    +-------------------------------+----------------------+----------------------+
    
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    +-----------------------------------------------------------------------------+

基本工具安装

    $ apt-get update
    $ apt-get install -y vim git wget

设置环境变量

修改 .bash_profile, 设置两个 cache 目录，这样不必每次创建 Docker 都重新下载 对应的文件。

    export KERAS_HOME="/data/cache/keras"
    export TFHUB_CACHE_DIR='/data/cache/tfhub'

## 检查 CUDA 和 cuDNN 的安装

    $ cat /usr/local/cuda/version.txt
    CUDA Version 10.0.130
    
    $ dpkg -l | egrep 'cuda|cudnn|TensorRT|libcupti|libnvinfer'
    
    $ apt-get install --no-install-recommends \
        cuda-10-0 \
        libcudnn7=7.6.2.24-1+cuda10.0 \
        libcudnn7-dev=7.6.2.24-1+cuda10.0
    
## 使用本例

进入对应的 Python 虚拟环境

    $ cd /data
    $ mkdir workspaces
    $ cd workspaces

Clone 此项目

    $ git clone https://github.com/iascchen/ai_study_notes.git
    
安装所需要的 Python 包，如果没有 GPU 请把 tensorflow-gpu 注释掉。
    
    $ cd ai_study_notes/src
    ai_study_notes/src$ ls
    ai_study_notes/src$ pip install Cython
    ai_study_notes/src$ pip install -r requirements.txt
    
不清楚为什么，安装 tensorflowjs 会重新安装 tensorflow 包，导致 GPU 支持不可用。强制重新安装 Tensorflow-gpu。

    $ pip install --force-reinstall \
        h5py==2.8.0 \
        numpy==1.16.4 \
        six==1.11.0 \
        tensorflow-gpu==1.14.0 
    
验证代码

    # python hello_gpu.py
    GPU is ready : True    
    
但是，执行 hello_keras_mobilenet_v1.py 还是会报错，说 cuDNN 不能创建。
