# 快速开始：安装 Python 和 Tensorflow 环境，并使用本例

## 安装 miniconda 和 Python3 环境

安装 conda 的目的是简化安装 python3 及其虚拟环境的过程。

conda 分为 anaconda 和 miniconda。
anaconda是包含一些常用包的版本（这里的常用不代表你常用），miniconda则是精简版，需要啥装啥，所以推荐使用miniconda。

下载网址
conda官网：[https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html#). 
选择适合自己操作系统的 Python3 相关版本。

例如：

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.pkg

下载安装即可。
安装时，被询问是否将conda加入环境变量的时候选择 **no**。
安装完成之后，可以使用 conda --version 查看版本。

    $ conda --version
    conda 4.7.12

需要注意的是，conda 在安装软件的时候需要C库就自己安装，需要R包就自己按照R，需要perl就自己安装，而且我把conda的默认路径添加到环境变量最新，最高权限，就把我默认的perl,r全部替换了。
conda请最好是通过source启动，一定要用conda安装不同功能的软件各个env，免得它污染环境变量，使用某些软件，就激活某些env。

leotao的建议
不要让conda在安装时，把path加到系统里去，要用的时候激活

激活后，用conda install -p /path/for/biotools/把 生信软件装到特定位置,而且这个位置的python版本最好和系统的一样

把这个 /path/for/biotools/ 加入到系统path

可能这样做后，反激活conda后生信软件也能用，同时不会污染环境

[参考链接](https://mp.weixin.qq.com/s?__biz=MzAxMDkxODM1Ng==&mid=2247486380&idx=1&sn=9329fcd0a60ac5488607d359d6c28134&chksm=9b484b17ac3fc20153d25cbdefe5017c7aa9080d13b5473a05f79808244e848b0a45d2a6a735&scene=21#wechat_redirect)

创建工作目录

    $ mkdir workspaces
    $ cd workspaces
    
创建 Python 虚拟环境，我们把它起名为 tf2

    $ virtualenv --system-site-packages -p python3 tf2
    
    $ conda create -n tf2 python=3
    $ conda activate tf2

启动虚拟环境
    
    $ . tf2/bin/activate
    
    $ conda activate tf2
    (tf2) $ python --version
    Python 3.7.4
    (tf2) $ pip --version
    pip 19.2.3 from /Users/chenhao/miniconda3/envs/tf2/lib/python3.7/site-packages/pip (python 3.7)

## 安装 Tensorflow 2.0.0

因为 Conda 尚未集成 Tensorflow 2，所以只能使用 pip 来安装。

    $ cd workspaces
    $ . tf/bin/activate
    
    $ conda activate tf2
    
Clone 此项目

    (tf2) $ git clone https://github.com/iascchen/ai_study_notes.git
    
安装所需要的 Python 包，如果没有 GPU 请把 tensorflow-gpu 注释掉。
    
    (tf2) $ cd ai_study_notes/src
    (tf2) ai_study_notes/src$ ls
    (tf2) ai_study_notes/src$ pip install -r requirements.txt

安装前，你可以查看一下 [requirements.txt](../src/requirements.txt) 的内容。

验证代码

    (tf2) $ python hello_gpu.py
    
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