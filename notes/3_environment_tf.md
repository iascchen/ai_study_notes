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
    $ sudo apt-get install --no-install-recommends --allow-downgrades \
        cuda-10-0 \
        libcudnn7=7.6.2.24-1+cuda10.0 \
        libcudnn7-dev=7.6.2.24-1+cuda10.0

## 手动安装 cuDNN

    sudo dpkg -i libcudnn7_7.6.2.24-1+cuda10.0_amd64.deb
    sudo dpkg -i libcudnn7-dev_7.6.2.24-1+cuda10.0_amd64.deb
    sudo dpkg -i libcudnn7-doc_7.6.2.24-1+cuda10.0_amd64.deb
    
库文件的安装路径在 /usr/include/ 目录下

    $ ls /usr/include/cu*
    /usr/include/cuda_stdint.h  /usr/include/cupti_activity.h     /usr/include/cupti_events.h     /usr/include/cupti_result.h
    /usr/include/cudnn.h        /usr/include/cupti_callbacks.h    /usr/include/cupti_metrics.h    /usr/include/cupti_runtime_cbid.h
    /usr/include/cupti.h        /usr/include/cupti_driver_cbid.h  /usr/include/cupti_nvtx_cbid.h  /usr/include/cupti_version.h

验证安装

    $ cd %working_space%
    $ cp -r /usr/src/cudnn_samples_v7/ .
    $ cd cudnn_samples_v7/mnistCUDNN
    $ make clean && make
    $ ./mnistCUDNN
    Test passed!

和 CUDA 整合

    $ sudo cp /usr/include/cudnn.h /usr/local/cuda/include
    $ sudo cp /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64
    $ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

    
## Install TensorRT
     
    sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.0-trt5.1.5.0-ga-20190427_1-1_amd64.deb
    sudo apt-key add /var/nv-tensorrt-repo-cuda10.0-trt5.1.5.0-ga-20190427/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install tensorrt python3-libnvinfer-dev uff-converter-tf

    
检查 TensorRT 的安装
    
    $ dpkg -l | grep TensorRT
    ii  graphsurgeon-tf                                             5.1.5-1+cuda10.0                             amd64        GraphSurgeon for TensorRT package
    ii  libnvinfer-dev                                              5.1.5-1+cuda10.0                             amd64        TensorRT development libraries and headers
    ii  libnvinfer-samples                                          5.1.5-1+cuda10.0                             all          TensorRT samples and documentation
    ii  libnvinfer5                                                 5.1.5-1+cuda10.0                             amd64        TensorRT runtime libraries
    ii  python3-libnvinfer                                          5.1.5-1+cuda10.0                             amd64        Python 3 bindings for TensorRT
    ii  python3-libnvinfer-dev                                      5.1.5-1+cuda10.0                             amd64        Python 3 development package for TensorRT
    ii  tensorrt                                                    5.1.5.0-1+cuda10.0                           amd64        Meta package of TensorRT
    ii  uff-converter-tf                                            5.1.5-1+cuda10.0                             amd64        UFF converter for TensorRT package

    sudo apt-get remove cuda-cudart-10-1 cuda-cudart-dev-10-1 
    sudo apt-get install --no-install-recommends --allow-downgrades cuda-repo-ubuntu1804=10.0.130-1
    
Remove all packages marked as rc by dpkg

    $ dpkg --list |grep "^rc" | cut -d " " -f 3 | xargs sudo dpkg --purge

## 检查所有安装包的版本
    
    dpkg -l | egrep 'cuda|cudnn|TensorRT|libcupti|libnvinfer'