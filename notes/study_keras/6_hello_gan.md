# Hello GAN 生成式对抗网络

## GAN 生成式对抗网络简介

[原文参考链接](https://blog.csdn.net/stdcoutzyx/article/details/53151038)

### Discriminative Model的繁荣发展
深度学习取得突破性进展的地方貌似都是discriminative的模型。

所谓的discriminative可以简单的认为是分类问题，比如给一张图片，判断这张图片上有什么动物；再比如给定一段语音，判断这段语音所对应的文字。

在discriminative的模型上，有很多行之有效的方法，如反向传播，dropout，piecewise linear units等技术。

### Generative Model

从细节上来看，生成模型可以做一些无中生有的事情。比如图片的高清化，遮住图片的一部分去修复，再或者画了一幅人脸的肖像轮廓，将其渲染成栩栩如生的照片等等。

再提高一层，生成模型的终极是创造，通过发现数据里的规律来生产一些东西，这就和真正的人工智能对应起来了。想想一个人，他可以通过看，听，闻去感知这世界，这是所谓的Discriminative，他也可以说，画，想一些新的事情，这就是创造。所以，生成模型我认为是AI在识别任务发展相当成熟之后的AI发展的又一个阶段。

但是现在，生成模型还没有体会到深度学习的利好，在Discriminative模型上，成果如雨后春笋，但在生成模型上，却并非如此。原因如下：

* 在最大似然估计及相关策略上，很多概率计算的模拟非常难
* 将piecewise linear units用在生成模型上比较难

### 对抗网络基本思想

假设有一种概率分布M，它相对于我们是一个黑盒子。为了了解这个黑盒子中的东西是什么，我们构建了两个东西G和D，G是另一种我们完全知道的概率分布，D用来区分一个事件是由黑盒子中那个不知道的东西产生的还是由我们自己设的G产生的。

不断的调整G和D，直到D不能把事件区分出来为止。在调整过程中，需要：

* 优化G，使它尽可能的让D混淆。
* 优化D，使它尽可能的能区分出假冒的东西。

当D无法区分出事件的来源的时候，可以认为，G和M是一样的。从而，我们就了解到了黑盒子中的东西。

GAN无需特定的cost function的优势和学习过程可以学习到很好的特征表示，但是GAN训练起来非常不稳定，经常会使得生成器产生没有意义的输出。

一些技巧：

* 使用 tanh 为生成器的最后一层，而不是 sigmoid
* 使用正态分布对潜在空间的点进行采样，而不用均匀分布。
* 随机性能够提高稳定。可以在判别器中使用dropout；也可以向判别器的标签增加随机噪声。
* 稀疏的梯度会妨碍GAN训练。最大池化和ReLU激活会导致梯度稀疏。所以一般用步进卷积替代最大池化，用 LeakyReLU 代替 ReLU 激活
* 在生成的图像中，经常会见到棋盘状伪影，这是由生成器中的像素空间不均匀覆盖导致的。未解决这个问题，每当生成器和判别器中使用步进的Conv2DTranpose 或 Conv2D 时，使用的内核大小需要能够被步幅大小整除。

## DCGAN 深度卷积对抗生成网络


* 为CNN的网络拓扑结构设置了一系列的限制来使得它可以稳定的训练。
* 使用得到的特征表示来进行图像分类，得到比较好的效果来验证生成的图像特征表示的表达能力
* 对GAN学习到的filter进行了定性的分析。
* 展示了生成的特征表示的向量计算特性。

模型结构上需要做如下几点变化：

* 将pooling层convolutions替代，其中，在discriminator上用strided convolutions替代，在generator上用fractional-strided convolutions替代。
* 在generator和discriminator上都使用batchnorm。
    * 解决初始化差的问题
    * 帮助梯度传播到每一层
    * 防止generator把所有的样本都收敛到同一个点。
    * 直接将BN应用到所有层会导致样本震荡和模型不稳定，通过在generator输出层和discriminator输入层不采用BN可以防止这种现象。
* 移除全连接层
    * global pooling增加了模型的稳定性，但伤害了收敛速度。
* 在generator的除了输出层外的所有层使用ReLU，输出层采用tanh。
* 在discriminator的所有层上使用LeakyReLU。


## VAE


## Pix2Pix

本节内容会用到以下数据集：

    $ cd data
    $ wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz
    $ tar -xvf facades.tar.gz

[hello_gan_pix2pix.py](../../src/study_keras/6_hello_gan/hello_gan_pix2pix.py) 展示了迁移学习的基本过程。
