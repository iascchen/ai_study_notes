# Tensorflow Docker 容器

**注意**：如果是在 OS 上直接安装 Python 和 Tensorflow 可以略过此文。

Docker 使用容器创建虚拟环境，以便将 TensorFlow 安装与系统的其余部分隔离开来。TensorFlow 程序在此虚拟环境中运行，该环境能够与其主机共享资源（访问目录、使用 GPU、连接到互联网等）。

Docker 是在 Linux 上启用 TensorFlow GPU 支持的最简单方法，因为只需在主机上安装 NVIDIA® GPU 驱动程序（无需安装 NVIDIA® CUDA® 工具包）。

官方提供了不同版本的 Tensorflow Docker Image，如果需要使用多个不同版本的 Tensorflow，使用 Docker 是个比较好的方式。

TensorFlow Docker 要求

* 在本地主机上安装 Docker。
* 要在 Linux 上启用 GPU 支持，请安装 nvidia-docker。

## 安装Docker

安装Docker的官方文档链接[https://docs.docker.com/install/linux/docker-ce/ubuntu/](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

TUNA 的 Docker Community Edition 镜像[https://mirrors.tuna.tsinghua.edu.cn/help/docker-ce/](https://mirrors.tuna.tsinghua.edu.cn/help/docker-ce/)

如果你过去安装过 docker，先删掉:

    $ sudo apt-get remove docker docker-engine docker.io
    
首先安装依赖:

    $ sudo apt-get install apt-transport-https ca-certificates curl gnupg2 software-properties-common
    
安装信任 Docker 的 GPG 公钥（根据你的发行版，下面的内容有所不同）:

    $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    OK
    $ sudo apt-key fingerprint 0EBFCD88
    pub   rsa4096 2017-02-22 [SCEA]
          9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88
    uid           [ 未知 ] Docker Release (CE deb) <docker@docker.com>
    sub   rsa4096 2017-02-22 [S]

对于 x86_64/amd64 架构的计算机，添加软件仓库。树莓派或其它ARM架构计算机 `arch=armhf`。

    $ sudo add-apt-repository "deb [arch=amd64] https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu \
       $(lsb_release -cs) stable"

最后安装

    $ sudo apt-get update
    $ sudo apt-get -y install docker-ce docker-ce-cli containerd.io

启动 Docker Domain

    $ sudo systemctl enable docker
    
    $ sudo systemctl start docker
    $ sudo usermod -aG docker $USER

退出，重新登录，就能够看到 Docker 已经安装好了。

    $ docker --version
    Docker version 19.03.1, build 74b1e89
    
    $ docker ps
    CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES

## 安装 nvidia-docker

NVIDIA Container Toolkit [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

请检查您的 Docker 版本，Docker 19.03 之后已经直接支持 -gpus 参数，能够直接支持 nvidia-docker 了。

在 Ubuntu 18.04 上，使用如下的命令使用 nvidia-docker

    # Add the package repositories
    $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    $ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    $ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    $ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    $ sudo systemctl restart docker

如果你使用的是其他版本的 Ubuntu，可能还需要检查一下当前 Ubuntu 所支持的 CUDA 版本。目前 Ubuntu 18.04 能够支持到 CUDA 10.1
参考链接为 [https://hub.docker.com/r/nvidia/cuda/](https://hub.docker.com/r/nvidia/cuda/)
    
使用以下命令测试安装成功，Tensorflow 1.14.0 使用的是 cuda:10.0，我们试一下。

    $ docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
    Unable to find image 'nvidia/cuda:10.0-base' locally
    10.0-base: Pulling from nvidia/cuda
    7413c47ba209: Pull complete
    0fe7e7cbb2e8: Pull complete
    1d425c982345: Pull complete
    344da5c95cec: Pull complete
    43bcc41986db: Pull complete
    b520fb8136d7: Pull complete
    426adf5fc5a1: Pull complete
    Digest: sha256:1c11e3e44ef257a9b5320ab584c87a11e599bd0d650d19e56f6653de1493a1cb
    Status: Downloaded newer image for nvidia/cuda:10.0-base
    Tue Aug  6 06:50:52 2019
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 430.26       Driver Version: 430.26       CUDA Version: 10.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce RTX 2070    Off  | 00000000:01:00.0  On |                  N/A |
    |  0%   39C    P8    20W / 175W |    187MiB /  7979MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    +-----------------------------------------------------------------------------+

还有一些其他的参数，能够更好的定义 Docker 中使用的 GPU 资源。

    #### Test nvidia-smi with the latest official CUDA image
    
    # Start a GPU enabled container on two GPUs
    $ docker run --gpus 2 nvidia/cuda:10.0-base nvidia-smi
    
    # Starting a GPU enabled container on specific GPUs
    $ docker run --gpus '"device=1,2"' nvidia/cuda:10.0-base nvidia-smi
    $ docker run --gpus '"device=UUID-ABCDEF,1'" nvidia/cuda:10.0-base nvidia-smi
    
    # Specifying a capability (graphics, compute, ...) for my container
    # Note this is rarely if ever used this way
    $ docker run --gpus all,capabilities=utility nvidia/cuda:10.0-base nvidia-smi

## 运行Tensorflow Docker

Pull一下所需的Tensorflow Docker Image，从下面的链接中获得对应的tag [https://hub.docker.com/r/tensorflow/tensorflow/tags/](https://hub.docker.com/r/tensorflow/tensorflow/tags/)。

以下命令会将 TensorFlow 版本映像下载到计算机上：

    $ docker pull tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

启动 TensorFlow Docker 容器检查一下。能够在输出信息中找到使用了 GeForce RTX 2070 相关的字样。

    $ docker run --gpus all -it --rm tensorflow/tensorflow:1.14.0-gpu-py3-jupyter python -c "import tensorflow as tf; print(tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None))"

也可以实验 Tensorflow 2.0 Beta

    $ docker pull tensorflow/tensorflow:2.0.0b1-gpu-py3-jupyter
    $ docker run --gpus all -it --rm tensorflow/tensorflow:2.0.0b1-gpu-py3-jupyter python -c "import tensorflow as tf; print(tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None))"

### 在 Docker 中使用 Ubuntu 桌面 GUI

Docker 环境下，需要注意处理和桌面GUI交互，不然有些具有图片输出的例子没法运行。

TODO