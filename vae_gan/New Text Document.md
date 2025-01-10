Creating a system for synthetic data generation using Variational Autoencoders (VAEs) followed by Generative Adversarial Networks (GANs) can be quite complex, especially when dealing with image data. Below is a complete implementation in PyTorch that outlines how to achieve this.

### Prerequisites
Before running the code, ensure you have the following libraries installed:

```bash
pip install torch torchvision matplotlib
```

### Step 1: Define the VAE

The VAE will encode images into a latent space, which can then be sampled from to generate new images. 

Here's how to implement the VAE:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VAE Model
class VAE(nn.Module):
    def __init__(self, img_size, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim * 2 * (img_size // 4) * (img_size // 4), latent_dim * 2)  # Output mu and logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2 * (img_size // 4) * (img_size // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (hidden_dim * 2, img_size // 4, img_size // 4)),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# VAE Loss
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 1*img_size*img_size), x.view(-1, 1*img_size*img_size), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

### Step 2: Prepare the Dataset

Load the image data from the specified directory, normalize it, and prepare it for the VAE.

```python
# Image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load dataset
img_data_dir = "./img_data/"
dataset = datasets.ImageFolder(root=img_data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Image size and latent dimension
img_size = 64  # Resize images to 64x64
hidden_dim = 32
latent_dim = 16
```

### Step 3: Train the VAE

Now we can train the VAE on the dataset.

```python
# Initialize VAE
vae = VAE(img_size=img_size, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(dataloader.dataset):.4f}')
```

### Step 4: Define the GAN

Once the VAE is trained, we can define a GAN that will use the latent space learned by the VAE to generate new images.

```python
# GAN Model
class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1 * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, img_size, img_size)

# Initialize GAN
generator = Generator(latent_dim).to(device)
discriminator = Discriminator(latent_dim).to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
```

### Step 5: Train the GAN

Now we can train the GAN using the latent space encoded by the VAE.

```python
# GAN Training Loop
num_gan_epochs = 50
for epoch in range(num_gan_epochs):
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)

        # Train Discriminator
        optimizer_D.zero_grad()

        # Generate latent vector from VAE
        with torch.no_grad():
            mu, logvar = vae.encode(data.view(-1, 1, img_size, img_size))
            z = vae.reparameterize(mu, logvar)

        # Real and fake labels
        real_labels = torch.ones(data.size(0), 1).to(device)
        fake_labels = torch.zeros(data.size(0), 1).to(device)

        # Discriminator loss
        outputs = discriminator(z)
        d_loss_real = criterion(outputs, real_labels)

        fake_z = torch.randn(data.size(0), latent_dim).to(device)
        fake_images = generator(fake_z)
        outputs = discriminator(fake_z.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        outputs = discriminator(fake_z)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

    print(f'Epoch [{epoch+1}/{num_gan_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
```

### Step 6: Image Generation

After training, we can generate new images from the GAN using random latent vectors.

```python
# Generate images
def generate_images(num_images):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        generated_images = generator(z)
        return generated_images

# Visualize generated images
generated_images = generate_images(10)
grid_size = 5
fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.axis('off')
    ax.imshow(generated_images[i].cpu().squeeze().numpy(), cmap='gray')
plt.show()
```

### Summary

This code provides a complete implementation of a system that combines a Variational Autoencoder (VAE) with a Generative Adversarial Network (GAN) for synthetic image generation. The VAE is trained first to learn a latent representation of the input images, and then the GAN is trained to generate new images using this latent space.

Make sure that your images in `./img_data/` are properly formatted for `ImageFolder`, which expects a directory structure like this:

```
./img_data/
    ├── class_1/
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    ├── class_2/
    │   ├── image1.png
    │   ├──