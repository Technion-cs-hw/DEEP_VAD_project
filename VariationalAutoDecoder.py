import torch
import torch.nn as nn

class VariationalAutoDecoder(nn.Module):
    def __init__(self, x_dim, z_dim, mu, sigma, distr = "normal", device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.distr = distr
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
        
        self.mlp = nn.Linear(x_dim,x_dim,bias=True)

    def sample_vectors(self, dist_params):
        if self.distr == "normal":
            # Reparameterization for Normal distribution
            mu = dist_params[:,:self.z_dim]
            sigma = dist_params[:,self.z_dim:]
            eps = torch.randn_like(mu, requires_grad=True).to(self.device)
            z = sigma*eps + mu
        elif self.distr == "uniform":
            # Reparameterization for Uniform distribution in the range [a, b]
            a = dist_params[:,:self.z_dim]
            b = dist_params[:,self.z_dim:] + a
            z = torch.rand_like(a, requires_grad=True).to(self.device) * (b-a) + a
        elif self.distr == "laplace":
            mu = dist_params[:,:self.z_dim]
            sigma = dist_params[:,self.z_dim:]
            #z = torch.distributions.laplace.Laplace(mu,sigma).rsample().to(self.device) 
            eps = torch.rand_like(mu, requires_grad=True).to(self.device)
            eps1 = 1e-5 + eps
            eps2 = eps - 1e-5
            v1 = sigma*((2*eps1).log())+mu
            v2 = mu - sigma*((2*(1-eps2)).log())
            z = torch.where(eps<0.5,v1,v2)
        else:
            raise ValueError(f"Distribution {self.distr} not recognized.")
        return z
    
    def decode(self, z):
        z = z.view(-1, self.z_dim, 1, 1)
        reconstructed_images = 255 * self.decoder(z).view(-1,28*28) # Output in range [0, 255]
        reconstructed_images = self.mlp(reconstructed_images)
        reconstructed_images = reconstructed_images.view(z.shape[0], 28, 28)
        return reconstructed_images
    
    def forward(self, dist_params):
        z = self.sample_vectors(dist_params)
        z = z.view(-1, self.z_dim, 1, 1)
        reconstructed_images = 255 * self.decoder(z).view(-1,28*28) # Output in range [0, 255]
        reconstructed_images = self.mlp(reconstructed_images)
        reconstructed_images = reconstructed_images.view(z.shape[0], 28, 28)
        return reconstructed_images    