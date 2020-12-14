import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import h5py as h5

sys.path.append('.')

from torch.utils.data import DataLoader
from python.utils import count_parameters
from python.data import HDF5SpectrogramLabeledFrames
from python.models.models import Classifier
from python.models.utils import binary_cross_entropy, f1_loss

##################################### SETTINGS #####################################################

# Dataset
# dataset_size = 'subset'
dataset_size = 'complete'

dataset_name = 'CSR-1-WSJ-0'
data_dir = 'export'
labels = 'noisy_labels'
# labels = 'noisy_vad_labels'

# System 
cuda = torch.cuda.is_available()
cuda_device = "cuda:0"
device = torch.device(cuda_device if cuda else "cpu")
num_workers = 16
pin_memory = True
non_blocking = True
rdcc_nbytes = 1024**2*400  # The number of bytes to use for the chunk cache
                           # Default is 1 Mb
                           # Here we are using 400Mb of chunk_cache_mem here
rdcc_nslots = 1e5 # The number of slots in the cache's hash table
                  # Default is 521
                  # ideally 100 x number of chunks that can be fit in rdcc_nbytes
                  # (see https://docs.h5py.org/en/stable/high/file.html?highlight=rdcc#chunk-cache)
eps = 1e-8

# Deep Generative Model
x_dim = 513 
if labels == 'noisy_labels':
    y_dim = 513
if labels == 'noisy_vad_labels':
    y_dim = 1
h_dim = [128, 128]
batch_norm=False
std_norm =True

# Training
batch_size = 128
learning_rate = 1e-3
log_interval = 250
start_epoch = 1
end_epoch = 100

if labels == 'noisy_labels':
    # model_name = 'wsj0_snr-15_5_classif_normdataset_batchnorm_before_hdim_{:03d}_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], end_epoch)
    model_name = 'dummy_wsj0_snr-15_5_classif_normdataset_batchnorm_before_hdim_{:03d}_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], end_epoch)

if labels == 'noisy_vad_labels':
    # model_name = 'classif_VAD_normdataset_batchnorm_before_hdim_{:03d}_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], end_epoch)
    #model_name = 'classif_VAD_batchnorm_before_hdim_{:03d}_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], end_epoch)
    model_name = 'dummy_classif_VAD_qf0.999_normdataset_hdim_{:03d}_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], end_epoch)

#####################################################################################################

print('Load data')
output_h5_dir = os.path.join('data', dataset_size, data_dir, dataset_name + '_' + labels + '.h5')

train_dataset = HDF5SpectrogramLabeledFrames(output_h5_dir=output_h5_dir, dataset_type='train', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots)
valid_dataset = HDF5SpectrogramLabeledFrames(output_h5_dir=output_h5_dir, dataset_type='validation', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots)

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
        print('Load mean and std')
        # Normalize train_data, valid_data
        # mean = np.mean(np.power(abs(train_data), 2), axis=1)[:, None]
        # std = np.std(np.power(abs(train_data), 2), axis=1, ddof=1)[:, None]
        with h5.File(output_h5_dir, 'r') as file:
            mean = file['X_train_mean'][:]
            std = file['X_train_std'][:]

        # Save mean and std
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
        total_loss, total_tp, total_tn, total_fp, total_fn = (0, 0, 0, 0, 0)
        for batch_idx, (x, y) in tqdm(enumerate(train_loader)):
            if cuda:
                x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)

            # Normalize power spectrogram
            if std_norm:
                x_norm = x - mean.T
                x_norm /= (std + eps).T

                y_hat_soft = model(x_norm) 
            else:
                y_hat_soft = model(x)
            
            loss = binary_cross_entropy(y_hat_soft, y, eps)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            y_hat_hard = (y_hat_soft > 0.5).int()

            f1_score, tp, tn, fp, fn = f1_loss(y_hat_hard=torch.flatten(y_hat_hard), y=torch.flatten(y), epsilon=eps)
            total_tp += tp.item()
            total_tn += tn.item()
            total_fp += fp.item()
            total_fn += fn.item()

            # Save to log
            if batch_idx % log_interval == 0:
                print(('Train Epoch: {:2d}   [{:7d}/{:7d} ({:2d}%)]    Loss: {:.2f}    F1-score.: {:.2f}'\
                    + '').format(epoch, batch_idx*len(x), len(train_loader.dataset), int(100.*batch_idx/len(train_loader)),\
                            loss.item(), f1_score.item()), 
                    file=open(model_dir + '/' + 'output_batch.log','a'))

        if epoch % 1 == 0:
            model.eval()

            total_precision = total_tp / (total_tp + total_fp + eps)
            total_recall = total_tp / (total_tp + total_fn + eps) 
            total_f1_score = 2 * (total_precision * total_recall) / (total_precision + total_recall + eps)

            print("Epoch: {}".format(epoch))
            print("[Train]       Loss: {:.2f}    F1_score: {:.2f}".format(total_loss / t, total_f1_score))

            # Save to log
            print(("Epoch: {}".format(epoch)), file=open(model_dir + '/' + 'output_epoch.log','a'))
            print("[Train]       Loss: {:.2f}    F1_score: {:.2f}".format(total_loss / t, total_f1_score),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            total_loss, total_tp, total_tn, total_fp, total_fn = (0, 0, 0, 0, 0)

            for batch_idx, (x, y) in enumerate(valid_loader):

                if cuda:
                    x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
                
                # Normalize power spectrogram
                if std_norm:
                    x_norm = x - mean.T
                    x_norm /= (std + eps).T

                    y_hat_soft = model(x_norm) 
                else:
                    y_hat_soft = model(x)
                
                loss = binary_cross_entropy(y_hat_soft, y, eps)

                total_loss += loss.item()
                y_hat_hard = (y_hat_soft > 0.5).int()
                f1_score, tp, tn, fp, fn = f1_loss(y_hat_hard=torch.flatten(y_hat_hard), y=torch.flatten(y), epsilon=eps)
                total_tp += tp.item()
                total_tn += tn.item()
                total_fp += fp.item()
                total_fn += fn.item()

            total_precision = total_tp / (total_tp + total_fp + eps)
            total_recall = total_tp / (total_tp + total_fn + eps) 
            total_f1_score = 2 * (total_precision * total_recall) / (total_precision + total_recall + eps)

            print("[Validation]  Loss: {:.2f}    F1_score: {:.2f}".format(total_loss / m, total_f1_score))

            # Save to log
            print("[Validation] Loss: {:.2f}    F1_score: {:.2f}".format(total_loss / m, total_f1_score),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            # Save model
            torch.save(model.state_dict(), model_dir + '/' + 'Classifier_epoch_{:03d}_vloss_{:.3f}.pt'.format(
                epoch, total_loss / m))

if __name__ == '__main__':
    main()