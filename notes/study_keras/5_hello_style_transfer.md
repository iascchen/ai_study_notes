# Hello Style Transfer

[Reference Link](https://github.com/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb
#)

## 风格迁移的原理

[参考代码](../src/study_keras/hello_style_transfer.py)

    loss = distance(style(reference_image) - style(generated_image)) \
       + distance(content(original_image) - content(generated_image))

这段伪代码，对风格迁移的本质进行了阐述——即：对生成图画的风格特征与参考图片之间的距离、生成图画的内容与原始图片之间的距离，进行最小化计算。

* distance 是一个范数函数，例如: L2
* style 计算图片的风格表示
* content 计算图片的内容表示。 

### Content Loss

网络更靠底部的激活层包含关于图像的局部信息，而更靠近顶部的层则包含更加全局、抽象的信息。因此选择更靠近顶部的层作为内容的表示。

    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2']

Content loss 只是简单地取两个图像的内容表示之间的欧氏距离 $$L^l_{content}(p, x) = \sum_{i, j} (F^l_{ij}(x) - P^l_{ij}(p))^2$$ 。

### Style Loss

    # Style layer we are interested in
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

计算风格损失涉及更多的层。但是，我们不是比较基本输入图像和样式图像的原始中间输出，而是比较两个输出的Gram矩阵。
将图像的样式表示描述为由Gram矩阵。$G^l$ 给出的不同滤波器响应之间的相关性，其中 $G^l_{ij}$ 是矢量化特征映射 $i$ 和 $j$ 之间的内积。
我们可以看到，在给定图像的特征映射上生成的 $G^l_{ij}$ 表示特征映射 $i$ 和 $j$ 之间的相关性。
为了生成基本输入图像的样式，我们从内容图像执行梯度下降，将其转换为与原始图像的样式表示相匹配的图像。
我们通过最小化样式图像的特征相关性图和输入图像之间的均方距离来实现。每个层对总风格损失的贡献由 $$ E_l = \frac{1}{4N_l^2M_l^2} \sum_{i,j}(G^l_{ij} - A^l_{ij})^2 $$ 描述。

### Loss

    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score 
    
## 快速风格迁移实现

[快速风格迁移](https://github.com/lengstrom/fast-style-transfer)

