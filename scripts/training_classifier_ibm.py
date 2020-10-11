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
from python.models.utils import binary_cross_entropy, f1_loss

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
h_dim = [128, 128]
batch_norm=True

# Training
batch_size = 128
learning_rate = 1e-3
log_interval = 250
start_epoch = 1
end_epoch = 100

model_name = 'classif_normdataset_batchnorm_before_hdim_{:03d}_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], end_epoch)

#####################################################################################################

print('Load data')
train_data = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_tr_s_noisy_frames.p'), 'rb'))
valid_data = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_dt_05_noisy_frames.p'), 'rb'))

# Normalize train_data, valid_data
mean = np.mean(train_data, axis=1)[:, None]
std = np.std(train_data, axis=1, ddof=1)[:, None]

train_data -= mean
valid_data -= mean

train_data /= (std + eps)
valid_data /= (std + eps)

train_labels = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_tr_s_noisy_labels.p'), 'rb'))
valid_labels = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_dt_05_noisy_labels.p'), 'rb'))

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

    # Save mean and variance
    np.save(model_dir + '/' + 'trainset_mean.npy', mean)
    np.save(model_dir + '/' + 'trainset_std.npy', std)

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
            torch.save(model.state_dict(), model_dir + '/' + 'Classifier_epoch_{:03d}_vloss_{:.2f}.pt'.format(
                epoch, total_loss / m))

if __name__ == '__main__':
    main()