import os
import torch
import torch.nn as nn
import torchvision.utils as vutils

image_num=100
latent_dim=100
channels=1
'''
如果在训练完成后直接生成图像则忽略此条，如果kernel重启过则需要重新定义生成器结构。
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
'''
generator = Generator()
generator.load_state_dict(torch.load('GAN_generator.pth'))
generator.eval()  

noise = torch.randn(image_num, latent_dim, 1, 1)

with torch.no_grad():
    generated_images = generator(noise)

save_dir = "Generated_data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i, image in enumerate(generated_images):
    vutils.save_image(image, os.path.join(save_dir, f"image_{i+1}.jpg"), normalize=True)
