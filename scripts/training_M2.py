import sys
sys.path.append('.')

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from itertools import cycle
import pickle
import numpy as np
#from sklearn.metrics import f1_score
from tqdm import tqdm

from python.models.models import DeepGenerativeModel
from python.models.variational import SVI, ImportanceWeightedSampler
from python.models.utils import binary_cross_entropy, ikatura_saito_divergence, f1_loss
#from python.models.distributions import prior_categorical

from python.data import SpectrogramLabeledFrames

# Settings
## Dataset
dataset_size = 'subset'
#dataset_size = 'complete'

# eps to fix (not necessarily 1e-8)
eps = 1e-8

cuda = torch.cuda.is_available()
num_workers = 0
device = torch.device("cuda:1" if cuda else "cpu")
pin_memory = True
non_blocking = True

## Deep Generative Model
x_dim = 513 # frequency bins (spectrogram)
#y_dim = 1 # frequency bins (binary mask)
y_dim = 513 # frequency bins (binary mask)
z_dim = 128
h_dim = [256, 128]

## Loss
alphas = [1.]

## Training
batch_size = 16
learning_rate = 1e-3
log_interval = 1
start_epoch = 1
end_epoch = 50

# Load data
print('Load data')
train_data = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_tr_s_frames.p'), 'rb'))
valid_data = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_dt_05_frames.p'), 'rb'))

train_labels = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_tr_s_labels.p'), 'rb'))
valid_labels = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_dt_05_labels.p'), 'rb'))

# Dataset class
train_dataset = SpectrogramLabeledFrames(train_data, train_labels)
valid_dataset = SpectrogramLabeledFrames(valid_data, valid_labels)

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=None, 
                        batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory, 
                        drop_last=False, timeout=0, worker_init_fn=None)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=None, 
batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory, 
                        drop_last=False, timeout=0, worker_init_fn=None)

print('- Number of training samples: {}'.format(len(train_dataset)))
print('- Number of validation samples: {}'.format(len(valid_dataset)))

def main(alpha):
    # Create model
    model = DeepGenerativeModel([x_dim, y_dim, z_dim, h_dim])
    if cuda: model = model.to(device, non_blocking=non_blocking)

    # Create model folder
    model_dir = os.path.join('models', 'M2_alpha_{:.1f}_end_epoch_{:03d}'.format(alpha, end_epoch))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Start log file
    file = open(model_dir + '/' +'output.log','w') 

    # Optimizer settings
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # We can use importance weighted samples [Burda, 2015] to get a better estimate
    # on the log-likelihood.
    sampler = ImportanceWeightedSampler(mc=1, iw=1)

    # # Uniform prior over y
    # prior = prior_categorical(batch_size=batch_size, y_dim=y_dim, device=device)

    #elbo = SVI(model, likelihood=binary_cross_entropy, sampler=sampler)
    elbo = SVI(model=model, likelihood=ikatura_saito_divergence, sampler=sampler, eps=eps)

    #BCE = nn.BCELoss()

    t = train_dataset.data.shape[1]
    m = valid_dataset.data.shape[1]

    # Training
    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_loss, total_elbo, total_likelihood, total_prior, total_kl, total_classif, total_f1_score = (0, 0, 0, 0, 0, 0, 0)

        for batch_idx, (x, y) in tqdm(enumerate(train_loader)):

            if cuda:
                # They need to be on the same device and be synchronized.
                x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)

            #L = -elbo(x, y)
            [L, likelihood, prior, kl] = elbo(x, y) # sign minus is inside elbo(x,y) now
            #U = -elbo(x)

            # Add auxiliary classification loss q(y|x)
            y_hat = model.classify(x)

            
            # Regular cross entropy
            classification_loss = -torch.sum(y*torch.log(y_hat + eps) + \
                                       (1.0-y)*torch.log(1.0 - y_hat + eps), dim=1).mean()
            # classification_loss = BCE(y_hat, y)

            J_alpha = L + alpha * classification_loss  # + U

            J_alpha.backward()
            optimizer.step()
            optimizer.zero_grad()

            # # J_alpha is a scalar, so J_alpha.data[0] does not work
            total_loss += J_alpha.item()
            total_elbo += L.item()
            total_likelihood += likelihood.item()
            total_prior += prior.item()
            total_kl += kl.item()
            total_classif += alpha * classification_loss.item()

            y_seg = (y_hat > 0.5).int()
            #accuracy += F1_score(y, y_seg)
            #total_f1_score += f1_score(y.cpu().numpy().flatten(), y_seg.cpu().numpy().flatten(), average="binary")
            f1_score = f1_loss(torch.flatten(y_seg), torch.flatten(y))
            total_f1_score += f1_score.item()

            # 
            if batch_idx % log_interval == 0:
                print(('Train Epoch: {:2d}   [{:4d}/{:4d} ({:2d}%)]    '\
                    'Loss: {:.3f}    ELBO: {:.3f}    Recon.: {:.3f}    prior: {:.3f}    KL: {:.3f}    classif.: {:.3f}    '\
                    +'F1-score: {:.3f}').format(epoch, batch_idx*len(x), len(train_loader.dataset), int(100.*batch_idx/len(train_loader)),\
                            J_alpha.item(), L.item(), likelihood.item(), prior.item(), kl.item(), alpha * classification_loss.item(), f1_score.item()), 
                    file=open(model_dir + '/' + 'output.log','a'))

            #accuracy += torch.mean((torch.max(y_hat, 1)[1].data == torch.max(y, 1)[1].data).float())

        if epoch % 1 == 0:
            model.eval()
            
            print("Epoch: {}".format(epoch))
            print("[Train]\t\t Loss: {:.2f}, ELBO: {:.2f}, Recon.: {:.2f}, prior: {:.2f}, KL: {:.2f} classif..: {:.2f}, "\
                "F1-score: {:.3f}".format(total_loss / t, total_elbo/t, total_likelihood/t, total_prior/t, total_kl/t, total_classif/t, total_f1_score/t))

            total_loss, total_elbo, total_likelihood, total_prior, total_kl, total_classif, total_f1_score = (0, 0, 0, 0, 0, 0, 0)
            for batch_idx, (x, y) in tqdm(enumerate(valid_loader)):

                if cuda:
                    x, y = x.cuda(device=device, non_blocking=non_blocking), y.cuda(device=device, non_blocking=non_blocking)
                
                #TODO: 1st classify, then encode
                #L = -elbo(x, y)
                [L, likelihood, prior, kl] = elbo(x, y) # sign minus is inside elbo(x,y) now
                #U = -elbo(x)

                # Add auxiliary classification loss q(y|x)
                y_hat = model.classify(x)

                classification_loss = -torch.sum(y*torch.log(y_hat + eps) + \
                                       (1.0-y)*torch.log(1.0 - y_hat + eps), dim=1).mean()
                # classification_loss = BCE(y_hat, y)

                J_alpha = L + alpha * classification_loss #+ U

                # # J_alpha is a scalar, so J_alpha.data[0] does not work
                total_loss += J_alpha.item()
                total_elbo += L.item()
                total_likelihood += likelihood.item()
                total_prior += prior.item()
                total_kl += kl.item()
                total_classif += alpha * classification_loss.item()

                y_seg = (y_hat > 0.5).int()
                #accuracy += F1_score(y, y_seg)
                #total_f1_score += f1_score(y.cpu().numpy().flatten(), y_seg.cpu().numpy().flatten(), average="binary")
                f1_score = f1_loss(torch.flatten(y_seg), torch.flatten(y))
                total_f1_score += f1_score.item()

                # _, pred_idx = torch.max(y_hat, 1)
                # _, lab_idx = torch.max(y, 1)

                # accuracy += torch.mean((torch.max(y_hat, 1)[1].data == torch.max(y, 1)[1].data).float())

            print("[Validation]\t Loss: {:.2f}, ELBO: {:.2f}, Recon.: {:.2f}, prior: {:.2f}, KL: {:.2f} classif..: {:.2f}, "\
                "F1-score: {:.3f}".format(total_loss / m, total_elbo/m, total_likelihood/m, total_prior/m, total_kl/m, total_classif/m, total_f1_score/m))

            # Save model
            torch.save(model.state_dict(), model_dir + '/' + 'M2_alpha_{:.1f}_epoch_{:03d}_vloss_{:.2f}.pt'.format(
                alpha,
                epoch,
                total_loss / m))

if __name__ == '__main__':
    for alpha in alphas:
        main(alpha)