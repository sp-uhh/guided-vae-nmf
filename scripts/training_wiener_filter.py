import os
import sys
import torch
import pickle
import numpy as np
from tqdm import tqdm

sys.path.append('.')

from torch.utils.data import DataLoader
from python.utils import count_parameters
from python.data import SpectrogramLabeledFrames
from python.models.models import Classifier
from python.models.utils import mean_square_error_signal, mean_square_error_mask

##################################### SETTINGS #####################################################

# Dataset
# dataset_size = 'subset'
dataset_size = 'complete'

# System 
cuda = torch.cuda.is_available()
cuda_device = "cuda:0"
device = torch.device(cuda_device if cuda else "cpu")
num_workers = 8
pin_memory = True
non_blocking = True
eps = 1e-8

# Deep Generative Model
x_dim = 513 
y_dim = 513
h_dim = [128, 128]
batch_norm = False
std_norm =True

# Training
batch_size = 128
learning_rate = 1e-3
log_interval = 250
start_epoch = 1
end_epoch = 200

model_name = 'wiener_avgfreq_maskloss_normdataset_input_amplitude_hdim_{:03d}_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], end_epoch)

#####################################################################################################

print('Load data')
train_data = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_tr_s_noisy_frames.p'), 'rb'))
valid_data = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_dt_05_noisy_frames.p'), 'rb'))

if std_norm:
    # Normalize train_data, valid_data
    mean = np.mean(train_data, axis=1)[:, None]
    std = np.std(train_data, axis=1, ddof=1)[:, None]

train_labels = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_tr_s_noisy_wiener_labels.p'), 'rb'))
valid_labels = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_dt_05_noisy_wiener_labels.p'), 'rb'))

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
    model = Classifier([x_dim, h_dim, y_dim], batch_norm=batch_norm)
    if cuda: model = model.to(device, non_blocking=non_blocking)

    # Create model folder
    model_dir = os.path.join('models', model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if std_norm:
        # Save mean and std
        global mean, std
        np.save(model_dir + '/' + 'trainset_mean.npy', mean)
        np.save(model_dir + '/' + 'trainset_std.npy', std)

    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)

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
        total_loss = (0)
        for batch_idx, (x, y) in tqdm(enumerate(train_loader)):
            if cuda:
                x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)

            # Normalize power spectrogram
            if std_norm:
                x_wiener = x - mean.T
                x_wiener /= (std + eps).T

                y_hat_soft = model(x_wiener) 
            else:
                y_hat_soft = model(x)  

            # loss = mean_square_error_signal(x=torch.sqrt(x), y=y, y_hat=y_hat_soft)
            loss = mean_square_error_mask(y=y, y_hat=y_hat_soft)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            # Save to log
            if batch_idx % log_interval == 0:
                print(('Train Epoch: {:2d}   [{:7d}/{:7d} ({:2d}%)]    Loss: {:.6f}    '\
                    + '').format(epoch, batch_idx*len(x), len(train_loader.dataset), int(100.*batch_idx/len(train_loader)),\
                            loss.item()), 
                    file=open(model_dir + '/' + 'output_batch.log','a'))

        if epoch % 1 == 0:
            model.eval()

            print("Epoch: {}".format(epoch))
            print("[Train]       Loss: {:.6f}    ".format(total_loss / t))

            # Save to log
            print(("Epoch: {}".format(epoch)), file=open(model_dir + '/' + 'output_epoch.log','a'))
            print("[Train]       Loss: {:.6f}    ".format(total_loss / t),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            total_loss = (0)

            for batch_idx, (x, y) in enumerate(valid_loader):

                if cuda:
                    x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)

                # Normalize power spectrogram
                if std_norm:
                    x_wiener = x - mean.T
                    x_wiener /= (std + eps).T

                    y_hat_soft = model(x_wiener) 
                else:
                    y_hat_soft = model(x)  

                # loss = mean_square_error_signal(x=torch.sqrt(x), y=y, y_hat=y_hat_soft)
                loss = mean_square_error_mask(y=y, y_hat=y_hat_soft)

                total_loss += loss.item()

            print("[Validation]  Loss: {:.6f}    ".format(total_loss / m))

            # Save to log
            print("[Validation] Loss: {:.6f}    ".format(total_loss / m),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            # Save model
            torch.save(model.state_dict(), model_dir + '/' + 'Classifier_epoch_{:03d}_vloss_{:.6f}.pt'.format(
                epoch, total_loss / m))

if __name__ == '__main__':
    main()