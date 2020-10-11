import os
import sys
import torch
import pickle
from tqdm import tqdm

sys.path.append('.')

from torch.utils.data import DataLoader
from python.utils import count_parameters
from python.data import SpectrogramFrames
from python.models.models import DeepGenerativeModel, Classifier
from python.models.utils import L_loss, U_loss

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
y_dim = 1
z_dim = 16
h_dim = [128, 128]

# Classifier
h_dim_cl = [128, 128]

# Training
batch_size = 128
learning_rate = 1e-3
log_interval = 250
start_epoch = 1
end_epoch = 100

model_name = 'M2_VAD_unsupervised_hdim_{:03d}_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], z_dim, end_epoch)

#####################################################################################################

print('Load data')
train_data = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_tr_s_frames.p'), 'rb'))
valid_data = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_dt_05_frames.p'), 'rb'))

train_dataset = SpectrogramFrames(train_data)
valid_dataset = SpectrogramFrames(valid_data)

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
    classifier = Classifier([x_dim, h_dim_cl, y_dim])
    if cuda: classifier = classifier.to(device, non_blocking=non_blocking)

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
        total_U, total_L, total_likelihood, total_kl = (0, 0, 0, 0)
        for batch_idx, x in tqdm(enumerate(train_loader)):
            if cuda:
                x = x.to(device, non_blocking=non_blocking)

            # Enumerate choices of label
            y0 = torch.zeros((x.size(0), y_dim)).to(device)
            y1 = torch.ones((x.size(0), y_dim)).to(device)
            y = torch.cat([y0, y1], dim=0)
            x = x.repeat(len([y0,y1]), 1)

            r, mu, logvar = model(x, y)
            y_hat_soft = model.classify(x)

            loss, L, recon_loss, KL = U_loss(x, r, mu, logvar, y_hat_soft, eps)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_U += loss.item()
            total_L += L.item()
            total_likelihood += recon_loss.item()
            total_kl += KL.item()

            # Save to log
            if batch_idx % log_interval == 0:
                print(('Train Epoch: {:2d}   [{:7d}/{:7d} ({:2d}%)]    '\
                    'U: {:.3f}    L: {:.3f}    Recon.: {:.3f}    KL: {:.3f}    '\
                    + '').format(epoch, batch_idx*len(x), len(train_loader.dataset), int(100.*batch_idx/len(train_loader)),\
                            loss.item(), L.item(), recon_loss.item(), KL.item()), 
                    file=open(model_dir + '/' + 'output_batch.log','a'))

        if epoch % 1 == 0:
            model.eval()

            print("Epoch: {}".format(epoch))
            print("[Train]\t\t U: {:.2f}, L: {:.2f}, Recon.: {:.2f}, KL: {:.2f}"\
                "".format(total_U / t, total_L / t, total_likelihood / t, total_kl / t))

            # Save to log
            print(("Epoch: {}".format(epoch)), file=open(model_dir + '/' + 'output_epoch.log','a'))
            print(("[Train]\t\t U: {:.2f}, L: {:.2f}, Recon.: {:.2f}, KL: {:.2f}"\
                "".format(total_U / t, total_L / t, total_likelihood / t, total_kl / t)),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            total_U, total_L, total_likelihood, total_kl = (0, 0, 0, 0)

            for batch_idx, x in enumerate(valid_loader):

                if cuda:
                    x = x.to(device, non_blocking=non_blocking)

                # Enumerate choices of label
                y0 = torch.zeros((x.size(0), y_dim)).to(device)
                y1 = torch.ones((x.size(0), y_dim)).to(device)
                y = torch.cat([y0, y1], dim=0)
                x = x.repeat(len([y0,y1]), 1)

                r, mu, logvar = model(x, y)
                y_hat_soft = model.classify(x)

                loss, L, recon_loss, KL = U_loss(x, r, mu, logvar, y_hat_soft, eps)

                total_U += loss.item()
                total_L += L.item()
                total_likelihood += recon_loss.item()
                total_kl += KL.item()
  
            print("[Validation]\t U: {:.2f}, L: {:.2f}, Recon.: {:.2f}, KL: {:.2f}"\
                "".format(total_U / m, total_L, total_likelihood / m, total_kl / m))

            print(("[Validation]\t U: {:.2f}, L: {:.2f}, Recon.: {:.2f}, KL: {:.2f}"\
                "".format(total_U / m, total_L / m, total_likelihood / m, total_kl / m)),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            # Save model
            torch.save(model.state_dict(), model_dir + '/' + 'M2_epoch_{:03d}_vloss_{:.2f}.pt'.format(
                epoch, total_U / m))
            
if __name__ == '__main__':
    main()