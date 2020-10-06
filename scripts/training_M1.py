import sys
sys.path.append('.')

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from itertools import cycle
import pickle
import numpy as np
from tqdm import tqdm

from python.models.models import VariationalAutoencoder
from python.models.variational import SVI_M1, ImportanceWeightedSampler
from python.models.utils import ikatura_saito_divergence, kl_divergence

from python.data import SpectrogramFrames

# Settings
## Dataset
#dataset_size = 'subset'
dataset_size = 'complete'

# eps to fix (not necessarily 1e-8)
eps = 1e-8

cuda = torch.cuda.is_available()
num_workers = 8
device = torch.device("cuda:1" if cuda else "cpu")
pin_memory = True
non_blocking = True

## Deep Generative Model
x_dim = 513 # frequency bins (spectrogram)
z_dim = 32
h_dim = [128]

## Training
batch_size = 128
learning_rate = 1e-3
log_interval = 1
start_epoch = 1
end_epoch = 250

# Load data
print('Load data')
train_data = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_tr_s_frames.p'), 'rb'))
valid_data = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_dt_05_frames.p'), 'rb'))

# Dataset class
train_dataset = SpectrogramFrames(train_data)
valid_dataset = SpectrogramFrames(valid_data)

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=None, 
                        batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory, 
                        drop_last=False, timeout=0, worker_init_fn=None)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=None, 
batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory, 
                        drop_last=False, timeout=0, worker_init_fn=None)

print('- Number of training samples: {}'.format(len(train_dataset)))
print('- Number of validation samples: {}'.format(len(valid_dataset)))

def main():
    # Create model
    model = VariationalAutoencoder([x_dim, z_dim, h_dim])
    if cuda: model = model.to(device, non_blocking=non_blocking)

    # Create model folder
    model_dir = os.path.join('models', 'M1_KLv3_eps1e-8_h{:03d}_z{:03d}_end_epoch_{:03d}'.format(h_dim[0], z_dim, end_epoch))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Start log file
    file = open(model_dir + '/' +'output_batch.log','w') 
    file = open(model_dir + '/' +'output_epoch.log','w') 

    # Optimizer settings
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # We can use importance weighted samples [Burda, 2015] to get a better estimate
    # on the log-likelihood.
    sampler = ImportanceWeightedSampler(mc=1, iw=1)

    #elbo = SVI_M1(model=model, likelihood=ikatura_saito_divergence, sampler=sampler, eps=eps)
    elbo = SVI_M1(model=model, likelihood=kl_divergence, sampler=sampler, eps=eps)


    t = len(train_loader)
    m = len(valid_loader)

    # Training
    for epoch in range(start_epoch, end_epoch):
        
        model.train()
        
        total_elbo, total_likelihood, total_kl = (0, 0, 0)

        for batch_idx, x in tqdm(enumerate(train_loader)):

            if cuda:
                # They need to be on the same device and be synchronized.
                x = x.to(device, non_blocking=non_blocking)

            [L, likelihood, kl] = elbo(x) # sign minus is inside elbo(x) now

            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            # # J_alpha is a scalar, so J_alpha.data[0] does not work
            total_elbo += L.item()
            total_likelihood += likelihood.item()
            total_kl += kl.item()

            # Save to log
            if batch_idx % log_interval == 0:
                print(('Train Epoch: {:2d}   [{:4d}/{:4d} ({:2d}%)]    '\
                    'ELBO: {:.3f}    Recon.: {:.3f}    KL: {:.3f}    '\
                    + '').format(epoch, batch_idx*len(x), len(train_loader.dataset), int(100.*batch_idx/len(train_loader)),\
                            L.item(), likelihood.item(), kl.item()), 
                    file=open(model_dir + '/' + 'output_batch.log','a'))

        if epoch % 1 == 0:
            model.eval()
            
            print("Epoch: {}".format(epoch))
            print("[Train]\t\t ELBO: {:.2f}, Recon.: {:.2f}, KL: {:.2f}"\
                "".format(total_elbo / t, total_likelihood / t, total_kl / t))

            print(("Epoch: {}".format(epoch)), file=open(model_dir + '/' + 'output_epoch.log','a'))
            print(("[Train]\t\t ELBO: {:.2f}, Recon.: {:.2f}, KL: {:.2f}"\
                "".format(total_elbo / t, total_likelihood / t, total_kl / t)),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            total_elbo, total_likelihood, total_kl = (0, 0, 0)

            for batch_idx, x in tqdm(enumerate(valid_loader)):

                if cuda:
                    x = x.cuda(device=device, non_blocking=non_blocking)
                
                #L = -elbo(x, y)
                [L, likelihood, kl] = elbo(x) # sign minus is inside elbo(x,y) now
                #U = -elbo(x)

                # # J_alpha is a scalar, so J_alpha.data[0] does not work
                total_elbo += L.item()
                total_likelihood += likelihood.item()
                total_kl += kl.item()

                # _, pred_idx = torch.max(y_hat, 1)
                # _, lab_idx = torch.max(y, 1)

                # accuracy += torch.mean((torch.max(y_hat, 1)[1].data == torch.max(y, 1)[1].data).float())
    
            print("[Validation]\t ELBO: {:.2f}, Recon.: {:.2f}, KL: {:.2f}"\
                "".format(total_elbo / m, total_likelihood / m, total_kl / m))

            print(("[Validation]\t ELBO: {:.2f}, Recon.: {:.2f}, KL: {:.2f}"\
                "".format(total_elbo / m, total_likelihood / m, total_kl / m)),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            # Save model
            torch.save(model.state_dict(), model_dir + '/' + 'M1_epoch_{:03d}_vloss_{:.2f}.pt'.format(
                epoch,
                total_elbo / m))

if __name__ == '__main__':
    main()