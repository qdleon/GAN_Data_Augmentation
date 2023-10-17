# DCGAN实验说明 (Deep Convolutional Generative Adversarial Network)

## 概述
这段代码实现了一个基于GAN的图像生成模型以实现舰船数据集的数据增成（Data Augmentation)。它使用了一个生成器(Generator)和一个鉴别器(Discriminator)。生成器的目标是生成尽可能真实的图像，而鉴别器的目标是区分出真实图像和生成的图像。

## 超参数设置
1. `image_size` - 图像的大小设为64x64。
2. `batch_size` - 批次大小为需要数据增成数据集的大小，代码中设置为19。
3. `lr` - 学习率设为0.0002。
4. `beta1` - Adam优化器的参数,为0.5。
5. `num_epochs` - 训练周期数为2000。
6. `latent_dim` - 初始随机噪声的维度为100。
7. `channels` - 图像的通道数设为1。
8. `label_real` 和 `label_fake` - 分别表示真实和伪造标签的标签平滑值，代码中分别设置为0.9和0.1。

## 数据预处理
- 使用`torchvision.transforms`对图像进行预处理，包括：
  - 调整大小（64x64)
  - 转为灰度图（如果`channels`为1）
  - 随机水平翻转
  - 随机旋转10度
  - 转换为张量并进行归一化。

## 模型结构
1. **生成器**:
   - 使用`nn.ConvTranspose2d`层逐步将随机噪声转换成图像。
   - 使用`ReLU`激活函数和`BatchNorm2d`进行归一化。
   - 最后一层输出使用`tanh`激活函数。

2. **鉴别器**:
   - 使用`nn.Conv2d`层对输入图像进行下采样。
   - 使用`LeakyReLU`作为激活函数并添加`BatchNorm2d`。
   - 最后输出一个值，表示输入图像的真实性。

## 训练过程
1. 对于每个批次的真实图像，计算鉴别器对其的损失（`loss_D_real`）。
2. 使用生成器生成伪造图像，然后计算鉴别器对其的损失（`loss_D_fake`）。
3. 结合上述两个损失来更新鉴别器。
4. 使用鉴别器的输出来计算生成器的损失，并更新生成器。

## TensorBoard 可视化
- 使用TensorBoard记录每10个周期的生成器和鉴别器的损失。
- 为每个周期生成的图像创建一个图像网格，并在TensorBoard中展示。

## 模型保存
- 训练结束后，保存生成器的参数到'generator1.pth'。

## 使用方法
1. 首先确保已经安装了所有必要的库和依赖。
2. 将您的图像数据放入"C:/Users/la/Desktop/Port Tender"目录下。
3. 直接运行此代码。
4. 使用TensorBoard查看训练过程中的损失和生成的图像。（执行命令行：tensorboard --logdir=C:\Users\la\runs\GAN_view，进入网址http://localhost:6006/）
5. 使用保存的'generator1.pth'随时加载训练好的生成器模型。
