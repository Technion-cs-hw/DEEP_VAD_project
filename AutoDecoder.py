import torch
import torch.nn as nn

class AutoDecoder(nn.Module):
    def __init__(self, x_dim, z_dim, latent_vectors, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.latent_vectors=nn.parameter.Parameter(latent_vectors, True)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 128, kernel_size=7, stride=1, padding=0),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        self.mlp = nn.Linear(x_dim,x_dim,bias=True)
        
    
    def decode(self, z):
        z = z.view(-1, self.z_dim, 1, 1)
        reconstructed_images = 255 * self.decoder(z).view(-1,28*28) # Output in range [0, 255]
        reconstructed_images = self.mlp(reconstructed_images)
        reconstructed_images = reconstructed_images.view(z.shape[0], 28, 28)
        return reconstructed_images
    
    def forward(self, z):
        z = z.view(-1, self.z_dim, 1, 1)
        reconstructed_images = 255 * self.decoder(z).view(-1,28*28) # Output in range [0, 255]
        reconstructed_images = self.mlp(reconstructed_images)
        reconstructed_images = reconstructed_images.view(z.shape[0], 28, 28)
        return reconstructed_images 
