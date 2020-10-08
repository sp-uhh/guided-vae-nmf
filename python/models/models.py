import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from python.models.distributions import log_gaussian, log_standard_gaussian

class Stochastic(nn.Module):
    def reparametrize(self, mu, log_var):
        epsilon = torch.randn(mu.size(), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.to(mu.get_device(), non_blocking=True)

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul(std, epsilon)

        return z

class GaussianSample(Stochastic):
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        #log_var = F.softplus(self.log_var(x))
        log_var = self.log_var(x)

        return self.reparametrize(mu, log_var), mu, log_var


class Classifier(nn.Module):
    def __init__(self, dims):
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.input_layer = nn.Linear(x_dim, h_dim)
        self.output_layer = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        #TODO: maybe modify activation functions?
        h = F.relu(self.input_layer(x))
        y = torch.sigmoid(self.output_layer(h))
        return y


class Encoder(nn.Module):
    def __init__(self, dims, sample_layer=GaussianSample):
        super(Encoder, self).__init__()

        [x_dim, h_dim, z_dim] = dims
        neurons = [x_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)
        self.sample = sample_layer(h_dim[-1], z_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.tanh(layer(x))
        return self.sample(x)


class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()

        [z_dim, h_dim, x_dim] = dims

        neurons = [z_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)
        self.reconstruction = nn.Linear(h_dim[-1], x_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.tanh(layer(x))
        return torch.exp(self.reconstruction(x))


class VariationalAutoencoder(nn.Module):
    def __init__(self, dims):
        super(VariationalAutoencoder, self).__init__()

        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim
        self.flow = None

        self.encoder = Encoder([x_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim, list(reversed(h_dim)), x_dim])
        self.kl_divergence = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, q_param, p_param=None):
        (mu, log_var) = q_param

        if self.flow is not None:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        else:
            #TODO: bug here
            qz = log_gaussian(z, mu, log_var)

        if p_param is None:
            #TODO: bug here
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)

        kl = qz - pz

        return kl

    def _kld_v2(self, z, q_param):
        (mu, log_var) = q_param
        return -0.5 * torch.sum(log_var - mu.pow(2) - log_var.exp(), axis=-1)

    def add_flow(self, flow):
        self.flow = flow

    def forward(self, x, y=None):
        z, z_mu, z_log_var = self.encoder(x)

        self.kl_divergence = self._kld_v2(z, (z_mu, z_log_var))

        x_mu = self.decoder(z)

        return x_mu, z_mu, z_log_var

    def sample(self, z):
        return self.decoder(z)


class DeepGenerativeModel(VariationalAutoencoder):
    def __init__(self, dims, classifier):
        [x_dim, self.y_dim, z_dim, h_dim] = dims
        super(DeepGenerativeModel, self).__init__([x_dim, z_dim, h_dim])

        self.encoder = Encoder([x_dim + self.y_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim])
        self.classifier = classifier 

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=1))
        x_mu = self.decoder(torch.cat([z, y], dim=1))
        return x_mu, z_mu, z_log_var

    def test(self, x):
        y = classify(x)
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=1))
        x_mu = self.decoder(torch.cat([z, y], dim=1))
        return x_mu, z_mu, z_log_var

    def classify(self, x):
        y = self.classifier(x)
        return y

    def sample(self, z, y):
        y = y.float()
        x = self.decoder(torch.cat([z, y], dim=1))
        return x

