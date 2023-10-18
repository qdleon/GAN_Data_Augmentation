# DATAA_GAN实验说明 (Data Augmentation Generative Adversarial Network)

## 概述
为解决舰船HOG数据集的某些子类图像数量过小的问题,这段代码实现了一个基于GAN的图像生成模型以实现舰船数据集的数据增成（Data Augmentation)。它使用了一个生成器(Generator)和一个鉴别器(Discriminator)。生成器的目标是生成尽可能真实的图像，而鉴别器的目标是区分出真实图像和生成的图像。

## 超参数设置
1. `image_size` - 用于Resize操作时调整图像的大小，为64。
2. `batch_size` - 批次大小为128。
3. `lr` - 学习率设为0.0002。
4. `beta1` - Adam优化器的参数,为0.5。
5. `num_epochs` - 训练周期数为1000。
6. `latent_dim` - 初始随机噪声的维度为100。
7. `channels` - 图像的通道数设为1。
8. `label_real` 和 `label_fake` - 分别表示真实和伪造标签的标签平滑值，代码中分别设置为0.9和0.1。

## 数据预处理
- 使用`torchvision.transforms`对图像进行预处理，包括：
  - 调整大小（64x64)
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
   - 使用`LeakyReLU`激活函数和`BatchNorm2d`进行归一化。
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
- 训练结束后，保存生成器的参数到'GAN_generator.pth'。

## 使用方法
1. 首先确保已经安装了所有必要的库和依赖。
2. 将代码第45行`dataset = ImageDataset(root_dir="FOLDER_NAME", transform=transform)`中的"FOLDER_NAME"换成工作目录中需要训练的数据集，数据集中不能有子类。
3. 根据需要调整`num_epochs`。
4. 直接运行此代码，此代码兼容cpu和cuda环境。
5. 使用TensorBoard查看训练过程中的损失和生成的图像。
6. 训练结束后将自动保存模型'GAN_generator.pth'在当前工作目录。
7. 使用保存的'GAN_generator.pth'随时加载训练好的生成器模型。

## 调用'GAN_generator.pth'批量生成图像
1. 根据需要修改图像数量`num_images`。
2. 确认'GAN_generator.pth'在工作目录后，直接运行'Generate_data.py'。
3. 图像将保存在工作目录"Generated_data"文件夹中。
