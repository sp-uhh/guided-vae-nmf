import os
import sys
import torch
import pickle
from tqdm import tqdm

sys.path.append('.')

from torch.utils.data import DataLoader
from python.utils import count_parameters
from python.data import SpectrogramLabeledFrames
from python.models.models import DeepGenerativeModel
from python.models.utils import elbo

##################################### SETTINGS #####################################################

# Dataset
# dataset_size = 'subset'
dataset_size = 'complete'

# System 
cuda = torch.cuda.is_available()
cuda_device = "cuda:2"
device = torch.device(cuda_device if cuda else "cpu")
num_workers = 8
pin_memory = True
non_blocking = True
eps = 1e-8

# Deep Generative Model
x_dim = 513 
y_dim = 513
z_dim = 16
h_dim = [128, 128]

# Classifier
classifier = None

# Training
batch_size = 128
learning_rate = 1e-3
log_interval = 250
start_epoch = 1
end_epoch = 200

model_name = 'ntcd_M2_hdim_{:03d}_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], z_dim, end_epoch)
# model_name = 'M2_hdim_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], z_dim, end_epoch)

#####################################################################################################

print('Load data')
train_data = pickle.load(open(os.path.join('data', dataset_size, 'pickle/ntcd_timit/train_frames.p'), 'rb'))
valid_data = pickle.load(open(os.path.join('data', dataset_size, 'pickle/ntcd_timit/dev_frames.p'), 'rb'))

train_labels = pickle.load(open(os.path.join('data', dataset_size, 'pickle/ntcd_timit/train_labels.p'), 'rb'))
valid_labels = pickle.load(open(os.path.join('data', dataset_size, 'pickle/ntcd_timit/dev_labels.p'), 'rb'))

train_dataset = SpectrogramLabeledFrames(train_data, train_labels)
valid_dataset = SpectrogramLabeledFrames(valid_data, valid_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=None, 
                        batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory, 
                        drop_last=False, timeout=0, worker_init_fn=None)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=None, 
                        batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory, 
                        drop_last=False, timeout=0, worker_init_fn=None)

print('- Number of training samples: {}'.format(len(train_dataset)))
print('- Number of validation samples: {}'.format(len(valid_dataset)))

def main():
    print('Create model')
    model = DeepGenerativeModel([x_dim, y_dim, z_dim, h_dim], classifier)
    if cuda: model = model.to(device, non_blocking=non_blocking)

    # Create model folder
    model_dir = os.path.join('models', model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)      

    # Start log file
    file = open(model_dir + '/' +'output_batch.log','w') 
    file = open(model_dir + '/' +'output_epoch.log','w') 

    # Optimizer settings
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    t = len(train_loader)
    m = len(valid_loader)
    print('- Number of learnable parameters: {}'.format(count_parameters(model)))

    print('Start training')
    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_elbo, total_likelihood, total_kl = (0, 0, 0)
        for batch_idx, (x, y) in enumerate(train_loader):
            if cuda:
                x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)

            r, mu, logvar = model(x, y)
            loss, recon_loss, KL = elbo(x, r, mu, logvar, eps)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_elbo += loss.item()
            total_likelihood += recon_loss.item()
            total_kl += KL.item()

            # Save to log
            if batch_idx % log_interval == 0:
                print(('Train Epoch: {:2d}   [{:7d}/{:7d} ({:2d}%)]    '\
                    'ELBO: {:.3f}    Recon.: {:.3f}    KL: {:.3f}    '\
                    + '').format(epoch, batch_idx*len(x), len(train_loader.dataset), int(100.*batch_idx/len(train_loader)),\
                            loss.item(), recon_loss.item(), KL.item()), 
                    file=open(model_dir + '/' + 'output_batch.log','a'))

        if epoch % 1 == 0:
            model.eval()

            print("Epoch: {}".format(epoch))
            print("[Train]\t\t ELBO: {:.2f}, Recon.: {:.2f}, KL: {:.2f}"\
                "".format(total_elbo / t, total_likelihood / t, total_kl / t))

            # Save to log
            print(("Epoch: {}".format(epoch)), file=open(model_dir + '/' + 'output_epoch.log','a'))
            print(("[Train]\t\t ELBO: {:.2f}, Recon.: {:.2f}, KL: {:.2f}"\
                "".format(total_elbo / t, total_likelihood / t, total_kl / t)),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            total_elbo, total_likelihood, total_kl = (0, 0, 0)

            for batch_idx, (x, y) in enumerate(valid_loader):

                if cuda:
                    x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)

                r, mu, logvar = model(x, y)
                loss, recon_loss, KL = elbo(x, r, mu, logvar, eps)

                total_elbo += loss.item()
                total_likelihood += recon_loss.item()
                total_kl += KL.item()
  
            print("[Validation]\t ELBO: {:.2f}, Recon.: {:.2f}, KL: {:.2f}"\
                "".format(total_elbo / m, total_likelihood / m, total_kl / m))

            print(("[Validation]\t ELBO: {:.2f}, Recon.: {:.2f}, KL: {:.2f}"\
                "".format(total_elbo / m, total_likelihood / m, total_kl / m)),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            # Save model
            torch.save(model.state_dict(), model_dir + '/' + 'M2_epoch_{:03d}_vloss_{:.2f}.pt'.format(
                epoch, total_elbo / m))
            
if __name__ == '__main__':
    main()