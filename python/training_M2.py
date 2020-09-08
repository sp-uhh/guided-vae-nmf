import torch
from torch import nn
from itertools import cycle

from models import DeepGenerativeModel
from variational import SVI, ImportanceWeightedSampler


# Settings

cuda = torch.cuda.is_available()

x_dim = 513
y_dim = 1
z_dim = 128
h_dim = [256, 128]


# Create model

model = DeepGenerativeModel([x_dim, y_dim, z_dim, h_dim])
if cuda: model = model.cuda()


# Load data

# training_data = DataLoader()
# validation_data = DataLoader()

alpha = 0.1


# Optimizer settings

def binary_cross_entropy(r, x):
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

# We can use importance weighted samples [Burda, 2015] to get a better estimate
# on the log-likelihood.
sampler = ImportanceWeightedSampler(mc=1, iw=1)

elbo = SVI(model, likelihood=binary_cross_entropy, sampler=sampler)


# Training

for epoch in range(10):
    model.train()
    total_loss, accuracy = (0, 0)

    for (x, y) in training_data:

        if cuda:
            # They need to be on the same device and be synchronized.
            x, y = x.cuda(device=0), y.cuda(device=0)

        L = -elbo(x, y)
        # U = -elbo(u)

        # Add auxiliary classification loss q(y|x)
        y_hat = model.classify(x)
        
        # Regular cross entropy
        classication_loss = torch.sum(y * torch.log(y_hat + 1e-8), dim=1).mean()

        J_alpha = L - alpha * classication_loss  # + U

        J_alpha.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += J_alpha.data[0]
        accuracy += torch.mean((torch.max(y_hat, 1)[1].data == torch.max(y, 1)[1].data).float())
        
    if epoch % 1 == 0:
        model.eval()
        m = len(unlabelled)
        print("Epoch: {}".format(epoch))
        print("[Train]\t\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))

        total_loss, accuracy = (0, 0)
        for x, y in validation_data:

            if cuda:
                x, y = x.cuda(device=0), y.cuda(device=0)

            L = -elbo(x, y)
            U = -elbo(x)

            y_hat = model.classify(x)
            classication_loss = -torch.sum(y * torch.log(y_hat + 1e-8), dim=1).mean()

            J_alpha = L + alpha * classication_loss + U

            total_loss += J_alpha.data[0]

            _, pred_idx = torch.max(y_hat, 1)
            _, lab_idx = torch.max(y, 1)
            accuracy += torch.mean((torch.max(y_hat, 1)[1].data == torch.max(y, 1)[1].data).float())

        m = len(validation)
        print("[Validation]\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
