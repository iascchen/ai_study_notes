# 直接安装 Tensorflow 环境

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

