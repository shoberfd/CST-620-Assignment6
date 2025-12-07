import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define transformations (resize, normalize, convert to tensors)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets (replace 'satellite' and 'map' with actual folder paths)
satellite_dataset = ImageFolder(root="./data/satellite", transform=transform)
map_dataset = ImageFolder(root="./data/map", transform=transform)

# Create data loaders
batch_size = 16
satellite_loader = DataLoader(satellite_dataset, batch_size=batch_size, shuffle=True)
map_loader = DataLoader(map_dataset, batch_size=batch_size, shuffle=True)

# Display a sample image
sample_image, _ = satellite_dataset[0]
plt.imshow(sample_image.permute(1, 2, 0))
plt.title("Sample Image - Satellite Domain")
plt.axis("off")
plt.show()

# Pix2pix Generator and Discriminator models defined here -------------------------------------->

import torch
import torch.nn as nn

# Define the Generator (U-Net style - Corrected
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Downsample: 256x256 -> 128x128
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        
        # Upsample: 128x128 -> 256x256 (This was previously Conv2d)
        self.conv2 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.tanh(self.conv2(x))  # Output scaled to [-1, 1]
        return x

# Define the Discriminator (PatchGAN-style)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.relu = nn.LeakyReLU(0.2)
        self.fc = nn.Linear(64 * 128 * 128, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.sigmoid(self.fc(x))  # Output between 0 and 1
        return x

# Initialize models
G_sat2map = Generator()
D_map = Discriminator()

# ---------------Trained Pix2Pix GAN for satellite-to-map translation---------------------

import torch.optim as optim

# Define loss functions
adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

# Optimizers
lr = 0.0002
optimizer_G = optim.Adam(G_sat2map.parameters(), lr=lr)
optimizer_D = optim.Adam(D_map.parameters(), lr=lr)

# Training loop
epochs = 10
for epoch in range(epochs):
    for (sat_images, _), (map_images, _) in zip(satellite_loader, map_loader):
        
        # Generate fake map images
        fake_map = G_sat2map(sat_images)

        # Discriminator loss
        real_loss = adversarial_loss(D_map(map_images), torch.ones_like(D_map(map_images)))
        fake_loss = adversarial_loss(D_map(fake_map.detach()), torch.zeros_like(D_map(fake_map)))
        d_loss = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Generator loss
        g_loss = adversarial_loss(D_map(fake_map), torch.ones_like(D_map(fake_map)))
        cycle_loss = l1_loss(fake_map, map_images)
        total_g_loss = g_loss + 10 * cycle_loss  # Weighted sum
        
        optimizer_G.zero_grad()
        total_g_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{epochs} - D Loss: {d_loss.item():.4f}, G Loss: {total_g_loss.item():.4f}")

# -------------Evaluated Pix2Pix and visualized image translation----------------------

# Generate a translated image
test_image, _ = satellite_dataset[0]
test_image = test_image.unsqueeze(0)
translated_image = G_sat2map(test_image).detach().squeeze().permute(1, 2, 0)

# Display original and translated images
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(test_image.squeeze().permute(1, 2, 0))
axes[0].set_title("Original (Satellite)")
axes[0].axis("off")

axes[1].imshow(translated_image.numpy())
axes[1].set_title("Translated (Map)")
axes[1].axis("off")

plt.show()