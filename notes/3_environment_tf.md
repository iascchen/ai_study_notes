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
    
## 安装 Tensorflow

进入对应的 Python 虚拟环境

    $ cd workspaces
    $ . tf/bin/activate

Clone 此项目

    (tf) $ git clone https://github.com/iascchen/ai_study_notes.git
    
安装所需要的 Python 包，如果没有 GPU 请把 tensorflow-gpu 注释掉。
    
    (tf) $ cd ai_study_notes/src
    (tf) ai_study_notes/src$ ls
    (tf) ai_study_notes/src$ pip install -r requirements.txt
    
验证代码

    (tf) $ python hello_gpu.py

