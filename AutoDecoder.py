import torch
import torch.nn as nn
import utils
import training

class AutoDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, train_ds_len, device = "cpu"):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, out_channels, kernel_size=4, stride=2, padding=1),

            nn.Tanh()
        )

        self.latent_vectors = torch.randn(train_ds_len, in_channels, requires_grad=True, device = device)
        #self.optimizer = torch.optim.Adam(self.parameters() , lr=1e-4)
        #self.loss = nn.MSELoss()

    def forward(self, indices):
        # Select the latent vectors corresponding to the input indices
        latent_vectors = self.latent_vectors[indices]
    
        # Reshape the latent vectors to add the spatial dimensions (1, 1) required by ConvTranspose2d
        latent_vectors = latent_vectors.view(len(indices), -1, 1, 1)
    
        # Decode each latent vector into an image
        reconstructed_images = self.decoder(latent_vectors)
    
        return reconstructed_images
