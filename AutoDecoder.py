import torch
import torch.nn as nn
import utils
import training

class AutoDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, train_ds_len):
        super().__init__()

        #channels = [in_channels, 512, 256, 128, out_channels]
        conv_params = dict(kernel_size=4, stride=2, padding=1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, *conv_params),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.ConvTranspose2d(512, 256, *conv_params),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, out_channels, *conv_params),

            nn.Tanh()
        )

        self.latent_vectors = torch.randn(train_ds_len, latent_dim, requires_grad=True, device='cuda')
        #self.optimizer = torch.optim.Adam(self.parameters() , lr=1e-4)
        #self.loss = nn.MSELoss()

    def forward(self, x):
        return self.decoder(x)


    


    """
    # Training loop
    #TODO: FIX TRAINING LOOP
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, (indices, images) in enumerate(train_dl):
            images = images.view(-1, 28 * 28).cuda()  # Flatten the images

            # Get the corresponding latent vectors for the batch
            z = model.latent_vectors[indices]

            # Forward pass (reconstruction)
            reconstructed = model(z)

            # Calculate the loss (reconstruction loss)
            loss = model.loss(reconstructed, images)

            # Backward pass and optimization
            model.optimizer.zero_grad()
            model.loss.backward()
            model.optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dl)}")
    """
    




