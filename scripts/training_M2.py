import torch
from torch import nn
from torch.utils.data import DataLoader
from itertools import cycle
import pickle
import numpy as np

from python.models.models import DeepGenerativeModel
from python.models.variational import SVI, ImportanceWeightedSampler
from python.models.utils import binary_cross_entropy, ikatura_saito_divergence

from python.data import SpectrogramLabeledFrames

# Settings

cuda = torch.cuda.is_available()

## Deep Generative Model
x_dim = 513 # frequency bins (spectrogram)
#y_dim = 1 # frequency bins (binary mask)
y_dim = 513 # frequency bins (binary mask)
z_dim = 128
h_dim = [256, 128]

## Loss
alpha = 0.1

## Training
batch_size = 16
learning_rate = 1e-3
log_interval = 1
start_epoch = 1
end_epoch = 10


def main():
    # Create model
    model = DeepGenerativeModel([x_dim, y_dim, z_dim, h_dim])
    if cuda: model = model.cuda()


    # Load data
    print('Load data')
    train_data = pickle.load(open('data/subset/pickle/si_tr_s_frames.p', 'rb'))
    valid_data = pickle.load(open('data/subset/pickle/si_dt_05_frames.p', 'rb'))

    train_labels = pickle.load(open('data/subset/pickle/si_tr_s_labels.p', 'rb'))
    valid_labels = pickle.load(open('data/subset/pickle/si_dt_05_labels.p', 'rb'))

    # Dataset class
    train_dataset = SpectrogramLabeledFrames(train_data, train_labels)
    valid_dataset = SpectrogramLabeledFrames(valid_data, valid_labels)

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=None, 
                            batch_sampler=None, num_workers=0, pin_memory=False, 
                            drop_last=False, timeout=0, worker_init_fn=None)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=None, 
    batch_sampler=None, num_workers=0, pin_memory=False, 
                            drop_last=False, timeout=0, worker_init_fn=None)

    print('- Number of training samples: {}'.format(len(train_dataset)))
    print('- Number of validation samples: {}'.format(len(valid_dataset)))


    # Optimizer settings
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # We can use importance weighted samples [Burda, 2015] to get a better estimate
    # on the log-likelihood.
    sampler = ImportanceWeightedSampler(mc=1, iw=1)

    #elbo = SVI(model, likelihood=binary_cross_entropy, sampler=sampler)
    elbo = SVI(model, likelihood=ikatura_saito_divergence, sampler=sampler)


    # Training
    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_loss, accuracy = (0, 0)

        for batch_idx, (x, y) in enumerate(train_loader):

            if cuda:
                # They need to be on the same device and be synchronized.
                x, y = x.cuda(device=0), y.cuda(device=0)

            L = -elbo(x, y)
            # U = -elbo(u)

            # Add auxiliary classification loss q(y|x)
            y_hat = model.classify(x)
            
            # Regular cross entropy
            #TODO: problem because it does not take the non speech class into account (cf PyTorch)
            classication_loss = torch.sum(y * torch.log(y_hat + 1e-8), dim=1).mean()

            J_alpha = L - alpha * classication_loss  # + U

            J_alpha.backward()
            optimizer.step()
            optimizer.zero_grad()

            # J_alpha is a scalar, so J_alpha.data[0] does not work
            total_loss += J_alpha.item()
            accuracy += torch.mean((torch.max(y_hat, 1)[1].data == torch.max(y, 1)[1].data).float())

        if epoch % 1 == 0:
            model.eval()
            
            m = valid_dataset.data.shape[1]

            print("Epoch: {}".format(epoch))
            print("[Train]\t\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))

            total_loss, accuracy = (0, 0)
            for batch_idx, (x, y) in enumerate(valid_loader):

                if cuda:
                    x, y = x.cuda(device=0), y.cuda(device=0)
                
                #TODO: 1st classify, then encode
                L = -elbo(x, y)
                #U = -elbo(x)

                y_hat = model.classify(x)
                classication_loss = -torch.sum(y * torch.log(y_hat + 1e-8), dim=1).mean()

                J_alpha = L + alpha * classication_loss #+ U

                # J_alpha is a scalar, so J_alpha.data[0] does not work
                total_loss += J_alpha.item()

                _, pred_idx = torch.max(y_hat, 1)
                _, lab_idx = torch.max(y, 1)
                accuracy += torch.mean((torch.max(y_hat, 1)[1].data == torch.max(y, 1)[1].data).float())

            m = valid_dataset.data.shape[1]
            print("[Validation]\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))

    # Save model
    torch.save(model.state_dict(), 'models/dummy_M2_10_epoch_{:03d}_vloss_{:.2f}.pt'.format(
        end_epoch, total_loss / m))

if __name__ == '__main__':
    main()
