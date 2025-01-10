import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from datetime import datetime
import logging

# --- Configuration ---
IMAGE_SIZE = 32  # Increased to 32 for better dimension handling
BATCH_SIZE = 8
LATENT_DIM = 128
EPOCHS = 200
LEARNING_RATE = 1e-4
BETA1 = 0.5
LAMBDA_KLD = 0.01
LAMBDA_REC = 1.0
GRADIENT_ACCUMULATION_STEPS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

def print_gpu_memory():
     if torch.cuda.is_available():
         logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
         logger.info(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def setup_directories(base_dir="outputs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    model_dir = os.path.join(run_dir, "models")
    image_dir = os.path.join(run_dir, "images")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    return run_dir, model_dir, image_dir

# --- Dataset ---
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname) 
            for fname in os.listdir(root_dir) 
            if fname.lower().endswith(('png', 'jpg', 'jpeg'))
        ]
        logger.info(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))

# --- VAE Model ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.fc_mu = nn.Linear(128 * 4 * 4, LATENT_DIM)
        self.fc_logvar = nn.Linear(128 * 4 * 4, LATENT_DIM)
        
        # Decoder
        self.decoder_input = nn.Linear(LATENT_DIM, 128 * 4 * 4)
        
        self.decoder = nn.Sequential(
            # Input: 128 x 4 x 4
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),  # 32x32
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        # Get latent representation
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x = self.decoder_input(z)
        x = x.view(-1, 128, 4, 4)
        x = self.decoder(x)
        return x, mu, logvar

# --- Discriminator Model ---
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 1, 3, stride=1, padding=0),  # 2x2
            nn.AdaptiveAvgPool2d(1)  # 1x1
        )

    def forward(self, x):
        return self.main(x).view(-1)  # No Sigmoid here


# --- Training Function ---
def train_model(vae, discriminator, train_loader, num_epochs, run_dir, model_dir, image_dir):
    vae_optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(num_epochs):
        vae.train()
        discriminator.train()
        
        total_vae_loss = 0
        total_disc_loss = 0
        
        for batch_idx, real_images in enumerate(train_loader):
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)
            
            # Train with accumulated gradients
            with torch.amp.autocast('cuda'):
                # Train Discriminator
                disc_optimizer.zero_grad()
                
                recon_images, mu, logvar = vae(real_images)
                real_labels = torch.ones(batch_size, device=DEVICE)
                fake_labels = torch.zeros(batch_size, device=DEVICE)
                
                real_output = discriminator(real_images)
                criterion = nn.BCEWithLogitsLoss()
                d_loss_real = criterion(real_output, real_labels)
                
                fake_output = discriminator(recon_images.detach())
                d_loss_fake = criterion(fake_output, fake_labels)
                
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.scale(d_loss).backward()
                    scaler.step(disc_optimizer)
                    scaler.update()
                
                # Train VAE
                vae_optimizer.zero_grad()
                
                recon_loss = nn.MSELoss()(recon_images, real_images)
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
                
                fake_output = discriminator(recon_images)
                g_loss = nn.BCEWithLogitsLoss()(fake_output, real_labels)
                
                vae_loss = LAMBDA_REC * recon_loss + LAMBDA_KLD * kld_loss + g_loss
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.scale(vae_loss).backward()
                    scaler.step(vae_optimizer)
                    scaler.update()
            
            total_vae_loss += vae_loss.item()
            total_disc_loss += d_loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                          f'VAE Loss: {vae_loss.item():.4f} Disc Loss: {d_loss.item():.4f}')
        
        # Save sample images
        if epoch % 5 == 0:
            vae.eval()
            with torch.no_grad():
                # Generate samples
                z = torch.randn(16, LATENT_DIM).to(DEVICE)
                samples = vae.decoder_input(z)
                samples = samples.view(-1, 128, 4, 4)
                samples = vae.decoder(samples).cpu()
                save_image(samples, os.path.join(image_dir, f'samples_epoch_{epoch}.png'),
                          normalize=True, nrow=4)
                
                # Save reconstructions
                real_batch = next(iter(train_loader))[:8].to(DEVICE)
                recon_batch, _, _ = vae(real_batch)
                comparison = torch.cat([real_batch, recon_batch])
                save_image(comparison, os.path.join(image_dir, f'reconstruction_epoch_{epoch}.png'),
                          normalize=True, nrow=8)
                
        # Save final model
        if epoch == num_epochs:
            torch.save(vae.state_dict(), os.path.join(model_dir, f"vae_epoch_{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(model_dir, f"discriminator_epoch_{epoch}.pth"))
            logger.info(f"Models saved at: {model_dir}")

def main():
    # Setup directories
    run_dir, model_dir, image_dir = setup_directories()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Dataset and DataLoader
    dataset = ImageDataset(
        root_dir=r"C:\Users\harvi\Codebases\DL\synth_img\img_data\street\train",
        transform=transform
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    # Initialize models
    vae = VAE().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    # Print initial memory usage
    print_gpu_memory()
    
    # Train
    train_model(vae, discriminator, train_loader, EPOCHS, run_dir, model_dir, image_dir)

if __name__ == "__main__":
    main()