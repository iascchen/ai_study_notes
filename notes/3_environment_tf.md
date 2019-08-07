# 快速开始：安装 Python 和 Tensorflow 环境，并使用本例

## 安装 Python3

    $ sudo apt-get update && sudo apt-get upgrade
    $ sudo apt-get install -y python3-pip python3-dev python-virtualenv 

创建工作目录

    $ mkdir workspaces
    $ cd workspaces
    
创建 Python 虚拟环境，我们把它起名为 tf

    $ virtualenv --system-site-packages -p python3 tf

启动虚拟环境

    $ . tf/bin/activate
    (tf) $ python --version
    Python 3.7.3
    (tf) $ pip --version
    pip 19.2.1 from /home/.../tf/lib/python3.7/site-packages/pip (python 3.7)
    
## 安装 Tensorflow 1.14.0

进入对应的 Python 虚拟环境

    $ cd workspaces
    $ . tf/bin/activate

Clone 此项目

    (tf) $ git clone https://github.com/iascchen/ai_study_notes.git
    
安装所需要的 Python 包，如果没有 GPU 请把 tensorflow-gpu 注释掉。
    
    (tf) $ cd ai_study_notes/src
    (tf) ai_study_notes/src$ ls
    (tf) ai_study_notes/src$ pip install -r requirements.txt

安装前，你可以查看一下 [requirements.txt](../src/requirements.txt) 的内容。

验证代码

    (tf) $ python hello_gpu.py
    
## 安装 CUDA 10

参考链接 [https://www.tensorflow.org/install/gpu?hl=zh-cn](https://www.tensorflow.org/install/gpu?hl=zh-cn)

    # Add NVIDIA package repositories
    
    $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
    $ sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
    $ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    $ sudo apt-get update
    $ wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    $ sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    $ sudo apt-get update

安装 CUDA，安装时间会比较长。

    # Install development and runtime libraries (~4GB)    
    $ sudo apt-get install --no-install-recommends \
        cuda-10-0 \
        libcudnn7=7.6.0.64-1+cuda10.0  \
        libcudnn7-dev=7.6.0.64-1+cuda10.0
    
安装 TensorRT.这部分与 Tensorflow 官方文档有些出入，必须指定 libnvinfer5 的依赖版本。
    
    # Install TensorRT. Requires that libcudnn7 is installed above.    
    $ sudo apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0 libnvinfer-dev=5.1.5-1+cuda10.0
    

