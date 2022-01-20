import torch
from torch import nn
from torch.nn import functional as F
import random
from typing import List
from torch import Tensor
import numpy as np


class VAE(nn.Module):
    """
    The main VAE class used to fit the VAE model, class parameters include:
    
    beta: The beta value used in the beta VAE, constant across epochs
    latent_dim: The number of latent dimensions (z)
    in_channels: The dimension of input parameters (x)
    norm_mean: normalized inputs mean, used in decoding only during evaluation mode
    norm_std: normalized inputs standard deviation, used in decoding only during evaluation mode
    """

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 beta: float = 1.0,
                 hidden_dims: List = None,
                 norm_mean = None,
                 norm_std = None,
                ) -> None:
        super().__init__()
        
        self.beta = beta
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        hidden_dims.reverse()
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0])

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(hidden_dims[-1], self.in_channels)
        
    def encode(self, input) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent values
        """
        self.encoder = self.encoder.float()
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z) -> Tensor:
        """
        Maps the given latent values onto the parameter space, if in eval mode renormalize
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
            
        # Un-normalize if decoding in eval mode
        if not self.training:
            if self.norm_mean is None or self.norm_std is None:
                raise Exception('Did not set norm constants before eval')
            else:
                result = self.norm_std * result + self.norm_mean
            
        return result

    def reparameterize(self, mu, logvar) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      recons,
                      input,
                      mu,
                      log_var,
                      ) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        recons_loss = F.mse_loss(recons, input)*self.in_channels

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        
        loss = recons_loss + self.beta * kld_loss
        
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self, num_samples) -> Tensor:
        """
        Samples from the latent space and return the corresponding parameter map
        """
        z = torch.randn(num_samples, self.latent_dim)

        samples = self.decode(z)
        return samples

    def generate(self, z) -> Tensor:
        """
        Given an latent code z, returns the reconstructed parameter set
        """

        return self.decode(z)
    
    
def fit_VAE(full_data, epochs, latent_dim, hidden_dims, batch_size, lr, beta, weight_decay):
    """
    Function used to fit a VAE based on provided parameters:
    
    full_data: historical data used for training
    epochs: number of epochs to run
    latent_dim: number of latent dimensions
    hidden_dims: list of hidden dimensions of encoder/decoder networks
    batch_size: size of random batch to be sampled from historical data
    lr: learning rate
    beta: fixed beta vlue
    weight_decay: Decay factor used in optimizer (AdamW)
    """
    
    running_losses = []
    losses = []
    
    vae = VAE(full_data.size(1), latent_dim, beta, hidden_dims)
    
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay = weight_decay)
    
    for epoch in range(epochs):
        batch = torch.stack(random.sample(list(full_data), min(batch_size, full_data.size(0))), axis = 0)
        
        optimizer.zero_grad()
        recon, input, mu, log_var = vae(batch)
        
        loss = vae.loss_function(recon, input, mu, log_var)
        loss['loss'].backward()
        optimizer.step()
        running_losses.append(loss['loss'].item())
        losses.append({'loss': loss['loss'].item(), 'Reconstruction_Loss':loss['Reconstruction_Loss'].item(), 'KLD':loss['KLD'].item()})
        
    return losses, vae
        