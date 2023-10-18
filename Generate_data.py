import os
import torchvision.utils as vutils

image_num=100

generator = Generator()
generator.load_state_dict(torch.load('GAN_generator.pth'))
generator.eval()  

noise = torch.randn(image_num, 100, 1, 1)

with torch.no_grad():
    generated_images = generator(noise)

save_dir = "Generated_data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i, image in enumerate(generated_images):
    vutils.save_image(image, os.path.join(save_dir, f"image_{i+1}.jpg"), normalize=True)
