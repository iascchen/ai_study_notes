# 环境准备

## OS 

Ubuntu 18.04

### SSH 访问

    sudo apt-get install openssh-server
    
    /etc/init.d/ssh start
    /etc/init.d/ssh restart
    /etc/init.d/ssh stop

### 检查环境

检查当前 Linux 版本

    $ uname -m && cat /etc/*release
    x86_64
    DISTRIB_ID=Ubuntu
    DISTRIB_RELEASE=18.04
    DISTRIB_CODENAME=bionic
    DISTRIB_DESCRIPTION="Ubuntu 18.04.2 LTS"
    NAME="Ubuntu"
    VERSION="18.04.2 LTS (Bionic Beaver)"
    ID=ubuntu
    ID_LIKE=debian
    PRETTY_NAME="Ubuntu 18.04.2 LTS"
    VERSION_ID="18.04"
    HOME_URL="https://www.ubuntu.com/"
    SUPPORT_URL="https://help.ubuntu.com/"
    BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
    PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
    VERSION_CODENAME=bionic
    UBUNTU_CODENAME=bionic

以及 Linux 内核

    $ uname -r
    4.15.0-55-generic

如果你需要在机器上直接安装AI开发环境所需的CUDA，还需要检查一下 GCC 版本

    $ gcc --version
    gcc (Ubuntu 7.4.0-1ubuntu1~18.04.1) 7.4.0
    Copyright (C) 2017 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

### 使用清华 Tuna 镜像

清华大学开源软件镜像站 [https://mirror.tuna.tsinghua.edu.cn/help/ubuntu/](https://mirror.tuna.tsinghua.edu.cn/help/ubuntu/)
 
UbuntuUbuntu 的软件源配置文件是/etc/apt/sources.list。将系统自带的该文件做个备份，将该文件替换为下面内容，即可使用 TUNA 的软件源镜像。

    # 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
    deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
    # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
    deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
    # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
    deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
    # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
    deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
    # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
    
    # 预发布软件源，不建议启用
    # deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
    # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse

具体操作指令如下$ cd /etc/apt

    $ sudo cp sources.list sources.list.old
    $ sudo vi sources.list
    $ sudo apt-get update
    Hit:1 https://mirrors.tuna.tsinghua.edu.cn/ubuntu bionic InRelease
    Hit:2 https://mirrors.tuna.tsinghua.edu.cn/ubuntu bionic-updates InRelease
    Hit:3 https://mirrors.tuna.tsinghua.edu.cn/ubuntu bionic-backports InRelease
    Hit:4 https://mirrors.tuna.tsinghua.edu.cn/ubuntu bionic-security InRelease
    Reading package lists... Done
    
## 安装 GPU 驱动

检查 GPU 显卡信息，你可以在NVidia网站上查到所用的GPU是 2070。

    $ lspci | grep -i nvidia
    01:00.0 VGA compatible controller: NVIDIA Corporation Device 1f02 (rev a1)
    01:00.1 Audio device: NVIDIA Corporation Device 10f9 (rev a1)
    01:00.2 USB controller: NVIDIA Corporation Device 1ada (rev a1)
    01:00.3 Serial bus controller [0c80]: NVIDIA Corporation Device 1adb (rev a1)

Ubuntu 18.04上要安装驱动NVidia的驱动。运行一下 nvidia-smi 你看看，这说明 nvidia 驱动还没装。

    $ nvidia-smi
    nvidia-smi: command not found
    
如果已经安装，但是需要修改驱动版本时，可以用下面的命令清除旧版本驱动和 nouveau。

    $ sudo apt-get purge nvidia*
    $ sudo apt-get --purge remove xserver-xorg-video-nouveau

利用下面的的命令得出当前OS所支持的GPU Driver各版本，然而这个信息对我们并没有什么用，因为2070使用的驱动必须需要大于410版本。

    $ sudo apt-cache search nvidia | grep -E "nvidia-[0-9]{3}"
    
从NVidia官网寻找所需的驱动，驱动下载地址 [https://www.nvidia.com/Download/index.aspx?lang=en-us](https://www.nvidia.com/Download/index.aspx?lang=en-us)

当前（201908）最新驱动是 430 版本

    $ wget http://us.download.nvidia.com/XFree86/Linux-x86_64/430.40/NVIDIA-Linux-x86_64-430.40.run 

下载完成后，开始安装驱动，安装过程中会弹出字符交互界面，根据提示选择一下即可。

    $ chmod +x NVIDIA-Linux-x86_64-430.40.run
    $ sudo ./NVIDIA-Linux-x86_64-430.40.run

安装结束之后重启系统

    $ sudo reboot

重启完成后应该能够看到GPU已经安装成功

    $ nvidia-smi
    Tue Aug  6 12:33:25 2019
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 430.26       Driver Version: 430.26       CUDA Version: 10.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce RTX 2070    Off  | 00000000:01:00.0  On |                  N/A |
    |  0%   39C    P8    20W / 175W |    198MiB /  7979MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0      1120      G   /usr/lib/xorg/Xorg                           105MiB |
    |    0      1281      G   /usr/bin/gnome-shell                          91MiB |
    +-----------------------------------------------------------------------------+
    
