import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

image_size = 64
batch_size = 128
lr = 0.0002
beta1 = 0.5
num_epochs = 1000
latent_dim = 100
channels = 1
label_real = 0.9  
label_fake = 0.1  

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, 0 


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = ImageDataset(root_dir="FOLDER_NAME", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
generator.apply(weights_init)
discriminator = Discriminator().to(device)
discriminator.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
fixed_noise = torch.randn(4, latent_dim, 1, 1, device=device)

writer = SummaryWriter('runs/GAN_view')

for epoch in range(num_epochs):
    for i, (data, _) in enumerate(dataloader):
        discriminator.zero_grad()

        real_data = data.to(device)
        batch_size = real_data.size(0)
        label = torch.full((batch_size,), label_real, device=device)
        output = discriminator(real_data)
        loss_D_real = criterion(output, label)
        loss_D_real.backward()

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake = generator(noise)
        label.fill_(label_fake)
        output = discriminator(fake.detach())
        loss_D_fake = criterion(output, label)
        loss_D_fake.backward()

        loss_D = loss_D_real + loss_D_fake
        optimizerD.step()

        generator.zero_grad()
        label.fill_(label_real)
        output = discriminator(fake)
        loss_G = criterion(output, label)
        loss_G.backward()
        optimizerG.step()
    if epoch % 10 == 0:
        writer.add_scalar('Loss/Generator', loss_G.item(), epoch)
        writer.add_scalar('Loss/Discriminator', (loss_D_real.item() + loss_D_fake.item()), epoch)
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
            img_grid = torchvision.utils.make_grid(fake, padding=2, normalize=True)
            writer.add_image('Generated Images', img_grid, epoch)

writer.close()
torch.save(generator.state_dict(), 'GAN_generator.pth')
