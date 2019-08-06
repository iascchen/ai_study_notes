# 快速开始：在 Docker 中使用本例

## 准备Host环境

为了便于管理数据，我们将Docker Container所用的数据全部放在 Host 上的 /data 目录下。

    $ sudo mkdir /data
    $ sudo chmod 777 /data

启动 Docker 容器

    $ docker run --name=my_tf_1.14 --volume=/data/my_tf_1.14:/data -w=/data --gpus all -it tensorflow/tensorflow:1.14.0-gpu-py3-jupyter bash
    root@c25d719a3624:/data#

如果需要重新调整 Docker 参数，可以删掉已有的 Container

    $ docker stop my_tf_1.14
    $ docker rm my_tf_1.14

如果重新启动容器，您可以使用docker exec重用容器。

    $ docker start my_tf_1.14
    $ docker exec -it my_tf_1.14 bash
    root@c25d719a3624:/data#

环境准备GPU检查

    root@56ec8f6859cc:/data# nvidia-smi


基本工具安装

    $ apt-get update
    $ apt-get install -y vim

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
    ai_study_notes/src$ pip install -r requirements.txt
    
验证代码

    $ python hello_gpu.py

