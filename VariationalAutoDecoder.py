import torch
import torch.nn as nn

class VariationalAutoDecoder(nn.Module):
    def __init__(self, x_dim, z_dim, mu, sigma, distr = "normal", device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.distr = distr
        #self.mu = nn.parameter.Parameter(mu,True) 
        #self.sigma = nn.parameter.Parameter(sigma,True) 
        self.dist_params=nn.parameter.Parameter(torch.cat((mu,sigma),dim=1),True)

        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 128, kernel_size=7, stride=1, padding=0),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in range [0, 1]
        )
        
        self.mlp =  nn.Linear(28*28,28*28,bias=True)
        
    def sample_vectors(self, dist_params):
        mu = dist_params[:,:self.z_dim]
        sigma = dist_params[:,self.z_dim:]
        Z = torch.randn_like(mu, requires_grad=True).to(self.device)
        X = sigma*Z + mu
        return X
        
    def forward(self, dist_params):        
        z = self.sample_vectors(dist_params)
        z = z.view(-1, self.z_dim, 1, 1)
        reconstructed_images = 255 * self.decoder(z).view(-1,28*28) # Output in range [0, 255]
        reconstructed_images = self.mlp(reconstructed_images)
        reconstructed_images = reconstructed_images.view(dist_params.shape[0], 28, 28)
        return reconstructed_images

"""        
    def reparameterize(self, mu, logvar, x):
        std = torch.exp(0.5 * logvar)
        if self.distr == "normal":
            # Reparameterization for Normal distribution
            mu = self.fc1(x)
            logvar = self.fc2(x)
            eps = torch.randn_like(std, requires_grad=True).to(self.device)
            z = mu + eps * std
        elif self.distr == "uniform":
            # Reparameterization for Uniform distribution in the range [a, b]
            a = self.fc1(x)
            b = self.fc2(x)
            if b < a:
                b =  a + nn.functional.softplus(b)
            z = torch.rand_like(std, requires_grad=True).to(self.device) * (b-a) + a
        elif self.distr == "lognormal":
            # Reparameterization for Log-Normal distribution
            mu = self.fc1(x)
            logvar = self.fc2(x)
            eps = torch.randn_like(std, requires_grad=True).to(self.device)
            z = torch.exp(mu + eps * std)  # Exponentiate to get the log-normal sample
        else:
            raise ValueError(f"Distribution {self.distr} not recognized.")

        return z
"""       
