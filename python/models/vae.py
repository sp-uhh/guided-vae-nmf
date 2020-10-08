import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, latent_dim , num_hidden_layers):
        super(VAE, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        self.in_out_dim = in_out_dim
        
        # encoder  
        self.enc_in = nn.Linear(in_out_dim, hidden_dim)
        self.enc_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        
        # decoder
        self.dec_in = nn.Linear(latent_dim, hidden_dim)
        self.dec_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.dec_out = nn.Linear(hidden_dim, in_out_dim)


    def encoder(self, x):
        h = torch.tanh(self.enc_in(x))
        for i in range(self.num_hidden_layers):
            h = torch.tanh(self.enc_hidden[i](h))  
        mu = self.mu(h)
        logvar = self.logvar(h) 
        z = self.reparameterize(mu, logvar)   
        return mu, logvar, z
    
    
    def decoder(self, z):
        h = torch.tanh(self.dec_in(z))
        for i in range(self.num_hidden_layers):
            h = torch.tanh(self.dec_hidden[i](h))
        return torch.exp(self.dec_out(h))


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def forward(self, x):
        mu, logvar, z = self.encode(x)
        return self.decode(z.T).T, mu, logvar, z
