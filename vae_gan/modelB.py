# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from torchvision.utils import save_image
# from PIL import Image
# import os

# # --- Configuration ---
# IMAGE_SIZE = 64  # Resize images to 64x64
# BATCH_SIZE = 64
# LATENT_DIM = 100  # Dimension of the latent space
# EPOCHS = 100
# LEARNING_RATE = 1e-3
# GAN_TRAINING_RATIO = 2  # Train GAN 2 times more than VAE
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DATA_DIR = "./img_data/street/train/"
# OUTPUT_DIR = "./generated_images/"

# # --- Create output directory if it doesn't exist ---
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# # --- Data Preprocessing ---
# class ImageDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = [
#             os.path.join(root_dir, f)
#             for f in os.listdir(root_dir)
#             if f.endswith(".png")
#         ]

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image

# transform = transforms.Compose(
#     [
#         transforms.Resize(IMAGE_SIZE),
#         transforms.CenterCrop(IMAGE_SIZE),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
#     ]
# )

# dataset = ImageDataset(root_dir=DATA_DIR, transform=transform)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# # --- VAE Model ---
# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()

#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(256 * (IMAGE_SIZE // 16) ** 2, 512),
#             nn.ReLU()
#         )
#         self.fc_mu = nn.Linear(512, LATENT_DIM)
#         self.fc_logvar = nn.Linear(512, LATENT_DIM)

#         # Decoder
#         self.decoder_input = nn.Linear(LATENT_DIM, 512)
#         self.decoder = nn.Sequential(
            
#             nn.Linear(512, 256 * (IMAGE_SIZE // 16) ** 2),
#             nn.ReLU(),
#             nn.Unflatten(1, (256, IMAGE_SIZE // 16, IMAGE_SIZE // 16)),
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
#             nn.Tanh(),  # Output between -1 and 1
#         )

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         encoded = self.encoder(x)
#         mu = self.fc_mu(encoded)
#         logvar = self.fc_logvar(encoded)
#         z = self.reparameterize(mu, logvar)
#         decoded = self.decoder_input(z)
#         decoded = self.decoder(decoded)
#         return decoded, mu, logvar

# # --- GAN Components: Discriminator ---
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(3, 64, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(512, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid(),
#         )

#     def forward(self, image):
#         output = self.main(image)
#         return output.view(-1, 1).squeeze(1)

# # --- Model Initialization ---
# vae = VAE().to(DEVICE)
# discriminator = Discriminator().to(DEVICE)

# # --- Optimizers ---
# optimizer_vae = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
# optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

# # --- Loss Functions ---
# reconstruction_loss = nn.MSELoss(reduction="sum")
# discriminator_loss = nn.BCELoss()

# # --- Training Loop ---
# def train_vae(epoch, real_images):
#     vae.train()
#     train_loss = 0

#     # Forward pass: compute reconstructed images, means, and logvars
#     recon_images, mu, logvar = vae(real_images)

#     # Compute reconstruction loss and KL divergence
#     recon_loss = reconstruction_loss(recon_images, real_images)
#     kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     # Total VAE loss
#     loss = recon_loss + kld_loss

#     # Backward pass and optimization
#     optimizer_vae.zero_grad()
#     loss.backward()
#     optimizer_vae.step()

#     train_loss += loss.item()
#     return train_loss

# def train_discriminator(epoch, real_images, recon_images):
#     discriminator.train()
    
#     # Train with real images
#     optimizer_discriminator.zero_grad()
#     real_labels = torch.ones(real_images.size(0)).to(DEVICE)
#     output_real = discriminator(real_images)
#     loss_real = discriminator_loss(output_real, real_labels)
#     loss_real.backward()

#     # Train with fake images (reconstructed images)
#     fake_labels = torch.zeros(recon_images.size(0)).to(DEVICE)
#     output_fake = discriminator(recon_images.detach())  # Detach to avoid training VAE
#     loss_fake = discriminator_loss(output_fake, fake_labels)
#     loss_fake.backward()

#     # Total discriminator loss
#     loss_D = loss_real + loss_fake
#     optimizer_discriminator.step()

#     return loss_D.item()

# for epoch in range(1, EPOCHS + 1):
    
#     for batch_idx, real_images in enumerate(dataloader):
#         real_images = real_images.to(DEVICE)
        
#         recon_images, _, _ = vae(real_images)
        
#         train_vae(epoch, real_images)
        
#         # Train GAN (Discriminator) more frequently
#         if batch_idx % GAN_TRAINING_RATIO == 0:
#             train_discriminator(epoch, real_images, recon_images)

#     print(f"Epoch {epoch}/{EPOCHS}")

#     # Save generated images periodically
#     if epoch % 5 == 0:
#         with torch.no_grad():
#             z = torch.randn(64, LATENT_DIM).to(DEVICE)
#             decoded_imgs = vae.decoder_input(z)
#             sample = vae.decoder(decoded_imgs).cpu()
#             save_image(
#                 sample.view(64, 3, IMAGE_SIZE, IMAGE_SIZE),
#                 os.path.join(OUTPUT_DIR, f"sample_epoch_{epoch}.png"),
#             )

# # --- Image Generation ---
# def generate_images(num_images, output_dir):
#     vae.eval()
#     with torch.no_grad():
#         for i in range(num_images):
#             z = torch.randn(1, LATENT_DIM).to(DEVICE)
#             decoded_imgs = vae.decoder_input(z)
#             generated_image = vae.decoder(decoded_imgs).cpu()
#             save_image(
#                 generated_image.view(3, IMAGE_SIZE, IMAGE_SIZE),
#                 os.path.join(output_dir, f"generated_{i}.png"),
#             )

# # Generate 10 unique images
# generate_images(10, OUTPUT_DIR)









# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from torchvision.utils import save_image
# from PIL import Image
# import os
# from multiprocessing import freeze_support
# import time

# # --- Configuration ---
# IMAGE_SIZE = 32  # Reduced image size
# BATCH_SIZE = 32  # Adjusted batch size
# LATENT_DIM = 64  # Dimension of the latent space
# EPOCHS = 100  # Adjusted epochs
# LEARNING_RATE = 1e-3  # Adjusted learning rate
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DATA_DIR = r"C:\\Users\\harvi\\Codebases\\DL\\synth_img\\img_data\\street\\train\\"
# OUTPUT_DIR = r"C:\\Users\\harvi\\Codebases\\DL\\synth_img\\generated_images2\\"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # --- Custom Dataset ---
# class ImageDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('png', 'jpg', 'jpeg'))]

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image

# # --- Data Preprocessing ---
# transform = transforms.Compose([
#     transforms.Resize(IMAGE_SIZE),
#     transforms.CenterCrop(IMAGE_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
# ])

# dataset = ImageDataset(root_dir=DATA_DIR, transform=transform)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# # --- VAE Model ---
# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Flatten(),
#             nn.Linear(256 * (IMAGE_SIZE // 16) ** 2, 512),
#             nn.LeakyReLU(0.2)
#         )
#         self.fc_mu = nn.Linear(512, LATENT_DIM)
#         self.fc_logvar = nn.Linear(512, LATENT_DIM)
#         self.decoder_input = nn.Linear(LATENT_DIM, 512)
#         self.decoder = nn.Sequential(
#             nn.Linear(512, 256 * (IMAGE_SIZE // 16) ** 2),
#             nn.LeakyReLU(0.2),
#             nn.Unflatten(1, (256, IMAGE_SIZE // 16, IMAGE_SIZE // 16)),
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
#             nn.Tanh()
#         )

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         encoded = self.encoder(x)
#         mu = self.fc_mu(encoded)
#         logvar = self.fc_logvar(encoded)
#         z = self.reparameterize(mu, logvar)
#         decoded = self.decoder_input(z)
#         decoded = self.decoder(decoded)
#         return decoded, mu, logvar

# # --- Discriminator Model ---
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(3, 64, 4, 2, 1, bias=False),  # Output: (64, 16, 16)
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # Output: (128, 8, 8)
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # Output: (256, 4, 4)
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # Output: (512, 2, 2)
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(512, 1, 2, 1, 0, bias=False),  # Output: (1, 1, 1)
#         )

#     def forward(self, image):
#         output = self.main(image)
#         return output.view(-1, 1).squeeze(1)

# # --- Initialize Models ---
# vae = VAE().to(DEVICE)
# discriminator = Discriminator().to(DEVICE)

# # --- Optimizers ---
# optimizer_vae = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
# optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

# # --- Loss Functions ---
# reconstruction_loss = nn.MSELoss(reduction="sum")

# # --- Training Functions ---
# scaler = torch.amp.GradScaler('cuda')

# def train_vae(epoch, real_images):
#     vae.train()
#     train_loss = 0
#     with torch.amp.autocast('cuda'):
#         recon_images, mu, logvar = vae(real_images)
#         recon_loss = reconstruction_loss(recon_images, real_images)
#         kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         loss = recon_loss + kld_loss
#     optimizer_vae.zero_grad()
#     scaler.scale(loss).backward()
#     scaler.step(optimizer_vae)
#     scaler.update()
#     train_loss += loss.item()
#     return train_loss

# def compute_gradient_penalty(discriminator, real_samples, fake_samples):
#     alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=DEVICE)
#     interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
#     d_interpolates = discriminator(interpolates)
#     fake = torch.ones(real_samples.size(0), device=DEVICE)
#     gradients = torch.autograd.grad(
#         outputs=d_interpolates,
#         inputs=interpolates,
#         grad_outputs=fake,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True,
#     )[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#     return gradient_penalty

# def train_discriminator(epoch, real_images, recon_images):
#     discriminator.train()
#     optimizer_discriminator.zero_grad()
#     with torch.amp.autocast('cuda'):
#         output_real = discriminator(real_images)
#         output_fake = discriminator(recon_images.detach())
#         loss_D = -(torch.mean(output_real) - torch.mean(output_fake))
#         gradient_penalty = compute_gradient_penalty(discriminator, real_images, recon_images.detach())
#         loss_D += 10 * gradient_penalty
#     scaler.scale(loss_D).backward()
#     scaler.step(optimizer_discriminator)
#     scaler.update()
#     return loss_D.item()

# # --- Training Loop ---
# if __name__ == '__main__':
#     freeze_support()
#     start_time = time.time()
#     for epoch in range(1, EPOCHS + 1):
#         epoch_start_time = time.time()
#         for batch_idx, real_images in enumerate(dataloader):
#             real_images = real_images.to(DEVICE)
#             recon_images, _, _ = vae(real_images)
#             train_vae(epoch, real_images)
#             if batch_idx % 2 == 0:
#                 train_discriminator(epoch, real_images, recon_images)

#         epoch_end_time = time.time()
#         epoch_duration = epoch_end_time - epoch_start_time
#         total_elapsed_time = epoch_end_time - start_time
#         estimated_total_time = total_elapsed_time / epoch * EPOCHS
#         estimated_remaining_time = estimated_total_time - total_elapsed_time

#         print(f"Epoch {epoch}/{EPOCHS} completed in {epoch_duration:.2f} seconds.")
#         print(f"Estimated total time: {estimated_total_time:.2f} seconds.")
#         print(f"Estimated remaining time: {estimated_remaining_time:.2f} seconds.")
#         print(f"Current learning rate: {optimizer_vae.param_groups[0]['lr']}")

#         if epoch % 25 == 0:
#             with torch.no_grad():
#                 z = torch.randn(64, LATENT_DIM).to(DEVICE)
#                 decoded_imgs = vae.decoder_input(z)
#                 sample = vae.decoder(decoded_imgs).cpu()
#                 save_image(
#                     sample.view(64, 3, IMAGE_SIZE, IMAGE_SIZE),
#                     os.path.join(OUTPUT_DIR, f"sample_epoch_{epoch}.png"),
#                 )

#     # --- Image Generation ---
#     def generate_images(num_images, output_dir):
#         vae.eval()
#         with torch.no_grad():
#             for i in range(num_images):
#                 z = torch.randn(1, LATENT_DIM).to(DEVICE)
#                 decoded_imgs = vae.decoder_input(z)
#                 generated_image = vae.decoder(decoded_imgs).cpu()
#                 save_image(
#                     generated_image.view(3, IMAGE_SIZE, IMAGE_SIZE),
#                     os.path.join(output_dir, f"generated_{i}.png"),
#                 )
#         print("Images generated and saved")

#     # Save the models after training is complete
#     torch.save(vae.state_dict(), os.path.join(OUTPUT_DIR, "vae_model.pth"))
#     torch.save(discriminator.state_dict(), os.path.join(OUTPUT_DIR, "discriminator_model.pth"))

#     # Generate 10 unique images
#     generate_images(10, OUTPUT_DIR)












import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from multiprocessing import freeze_support
import time

# --- Configuration ---
IMAGE_SIZE = 16  # Further reduced image size
BATCH_SIZE = 32  # Adjusted batch size
LATENT_DIM = 64  # Dimension of the latent space
EPOCHS = 100  # Adjusted epochs
LEARNING_RATE = 1e-3  # Increased learning rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
DATA_DIR = r"C:\\Users\\harvi\\Codebases\\DL\\synth_img\\img_data\\street\\train\\"
OUTPUT_DIR = r"C:\\Users\\harvi\\Codebases\\DL\\synth_img\\generated_images2\\"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Custom Dataset ---
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# --- Data Preprocessing ---
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
])

dataset = ImageDataset(root_dir=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# --- VAE Model ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * (IMAGE_SIZE // 8) ** 2, 256),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(256, LATENT_DIM)
        self.fc_logvar = nn.Linear(256, LATENT_DIM)
        self.decoder_input = nn.Linear(LATENT_DIM, 256)
        self.decoder = nn.Sequential(
            nn.Linear(256, 128 * (IMAGE_SIZE // 8) ** 2),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (128, IMAGE_SIZE // 8, IMAGE_SIZE // 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder_input(z)
        decoded = self.decoder(decoded)
        return decoded, mu, logvar

# --- Discriminator Model ---
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),  # Output: (64, 8, 8)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # Output: (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # Output: (256, 2, 2)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # Output: (512, 1, 1)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 1, 1, 0, bias=False),  # Output: (1, 1, 1)
        )

    def forward(self, image):
        output = self.main(image)
        return output.view(-1, 1).squeeze(1)

# --- Initialize Models ---
vae = VAE().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

# --- Optimizers ---
optimizer_vae = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

# --- Loss Functions ---
reconstruction_loss = nn.MSELoss(reduction="sum")

# --- Training Functions ---
scaler = torch.amp.GradScaler('cuda')

def train_vae(epoch, real_images):
    vae.train()
    train_loss = 0
    with torch.amp.autocast('cuda'):
        recon_images, mu, logvar = vae(real_images)
        recon_loss = reconstruction_loss(recon_images, real_images)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss
    optimizer_vae.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer_vae)
    scaler.update()
    train_loss += loss.item()
    return train_loss

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=DEVICE)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(real_samples.size(0), device=DEVICE)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_discriminator(epoch, real_images, recon_images):
    discriminator.train()
    optimizer_discriminator.zero_grad()
    with torch.amp.autocast('cuda'):
        output_real = discriminator(real_images)
        output_fake = discriminator(recon_images.detach())
        loss_D = -(torch.mean(output_real) - torch.mean(output_fake))
        gradient_penalty = compute_gradient_penalty(discriminator, real_images, recon_images.detach())
        loss_D += 10 * gradient_penalty
    scaler.scale(loss_D).backward()
    scaler.step(optimizer_discriminator)
    scaler.update()
    return loss_D.item()

# --- Training Loop ---
if __name__ == '__main__':
    freeze_support()
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        for batch_idx, real_images in enumerate(dataloader):
            real_images = real_images.to(DEVICE)
            recon_images, _, _ = vae(real_images)
            train_vae(epoch, real_images)
            if batch_idx % 2 == 0:
                train_discriminator(epoch, real_images, recon_images)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_elapsed_time = epoch_end_time - start_time
        estimated_total_time = total_elapsed_time / epoch * EPOCHS
        estimated_remaining_time = estimated_total_time - total_elapsed_time

        print(f"Epoch {epoch}/{EPOCHS} completed in {epoch_duration:.2f} seconds.")
        print(f"Estimated total time: {estimated_total_time:.2f} seconds.")
        print(f"Estimated remaining time: {estimated_remaining_time:.2f} seconds.")
        print(f"Current learning rate: {optimizer_vae.param_groups[0]['lr']}")

        if epoch % 5 == 0:
            with torch.no_grad():
                z = torch.randn(64, LATENT_DIM).to(DEVICE)
                decoded_imgs = vae.decoder_input(z)
                sample = vae.decoder(decoded_imgs).cpu()
                save_image(
                    sample.view(64, 3, IMAGE_SIZE, IMAGE_SIZE),
                    os.path.join(OUTPUT_DIR, f"sample_epoch_{epoch}.png"),
                )

    # --- Image Generation ---
    def generate_images(num_images, output_dir):
        vae.eval()
        with torch.no_grad():
            for i in range(num_images):
                z = torch.randn(1, LATENT_DIM).to(DEVICE)
                decoded_imgs = vae.decoder_input(z)
                generated_image = vae.decoder(decoded_imgs).cpu()
                save_image(
                    generated_image.view(3, IMAGE_SIZE, IMAGE_SIZE),
                    os.path.join(output_dir, f"generated_{i}.png"),
                )
        print("Images generated and saved")

    # Save the models after training is complete
    torch.save(vae.state_dict(), os.path.join(OUTPUT_DIR, "vae_model.pth"))
    torch.save(discriminator.state_dict(), os.path.join(OUTPUT_DIR, "discriminator_model.pth"))

    # Generate 10 unique images
    generate_images(10, OUTPUT_DIR)