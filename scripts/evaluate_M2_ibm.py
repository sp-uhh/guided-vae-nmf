import sys
sys.path.append('.')

import os
import numpy as np
import torch
import time
import soundfile as sf
import torch.multiprocessing as multiprocessing

from python.processing.stft import stft, istft
from python.processing.target import clean_speech_IBM
from python.utils import count_parameters
from python.models.mcem import MCEM_M2
from python.models.models import DeepGenerativeModel, Classifier
from python.models.spp_estimation import timo_mask_estimation

##################################### SETTINGS #####################################################

# Dataset
dataset_name = 'CSR-1-WSJ-0'
if dataset_name == 'CSR-1-WSJ-0':
    from python.dataset.csr1_wjs0_dataset import speech_list

dataset_type = 'test'

dataset_size = 'subset'
# dataset_size = 'complete'

# System 
cuda = torch.cuda.is_available()

# STFT parameters
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'

## Ideal binary mask
quantile_fraction = 0.999
quantile_weight = 0.999

# Hyperparameters 
# M2
# model_name = 'M2_hdim_128_128_zdim_032_end_epoch_100/M2_epoch_085_vloss_417.69'
model_name = 'M2_hdim_128_128_zdim_032_end_epoch_100/M2_epoch_098_vloss_414.57'
x_dim = 513 
y_dim = 513
z_dim = 32
h_dim = [128, 128]
eps = 1e-8

## Classifier
classif_type = 'dnn'
# classif_type = 'oracle'
# classif_type = 'timo'

if classif_type == 'dnn':
    # classif_name = 'classif_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_096_vloss_57.53'
    classif_name = 'classif_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_073_vloss_56.43'
    h_dim_cl = [128, 128]
    std_norm = True
    
if classif_type == 'oracle':
    classif_name = 'oracle_classif'

if classif_type == 'timo':
    classif_name = 'timo_classif'        

# NMF
nmf_rank = 10

### MCEM settings
niter = 100 # results reported in the paper were obtained with 500 iterations 
nsamples_E_step = 10
burnin_E_step = 30
nsamples_WF = 25
burnin_WF = 75
var_RW = 0.01

# GPU Multiprocessing
nb_devices = torch.cuda.device_count()
nb_process_per_device = 1

# Data directories
model_path = os.path.join('models_wsj0', model_name + '.pt')
classif_path = os.path.join('models_wsj0', classif_name + '.pt')
input_speech_dir = os.path.join('data', dataset_size,'raw/')
output_data_dir = os.path.join('data', dataset_size, 'models', model_name, classif_name + '/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')

#####################################################################################################

def process_utt(mcem, model, classifier, mean, std, file_path, device):
    
    # Input
    x_t, fs_x = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_x.wav') # mixture
    T_orig = len(x_t)
    x_tf = stft(x_t,
                fs=fs,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent,
                dtype=dtype) # (frames, freq_bins)
    
    # Transpose to match PyTorch
    x_tf = x_tf.T # (frames, freq_bins)

    x = torch.tensor(np.power(np.abs(x_tf), 2), device=device)

    # Target
    s_t, fs_s = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_s.wav') # clean speech
    s_tf = stft(s_t,
                fs=fs,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent,
                dtype=dtype) # (freq_bins, frames)

    if classif_type == 'dnn':    
        # Normalize power spectrogram
        if std_norm:
            x_norm = x - mean.T
            x_norm /= (std + eps).T

            y_hat_soft = classifier(x_norm) 
        else:
            y_hat_soft = classifier(x)   
        y_hat_hard = (y_hat_soft > 0.5).float()

    if classif_type == 'oracle':
        y_hat_soft = clean_speech_IBM(s_tf, quantile_fraction=quantile_fraction, quantile_weight=quantile_weight)
        y_hat_hard = torch.from_numpy(y_hat_soft.T).to(device)

    if classif_type == 'timo':
        x_numpy = np.power(np.abs(x_tf), 2)
        y_hat_soft = timo_mask_estimation(x_numpy.T)
        y_hat_hard = (y_hat_soft > 0.5).astype(int)
        y_hat_hard = y_hat_hard.T # (frames, freq_bins)
        y_hat_hard = torch.tensor(y_hat_hard).to(device)
    
    # Init MCEM
    mcem.init_parameters(X=x_tf,
                         y=y_hat_hard,
                        vae=model,
                        nmf_rank=nmf_rank,
                        eps=eps,
                        device=device)

    cost = mcem.run()

    S_hat = mcem.S_hat #+ np.finfo(np.float32).eps
    N_hat = mcem.N_hat #+ np.finfo(np.float32).eps

    s_hat = istft(S_hat, fs=fs, wlen_sec=wlen_sec, win=win, hop_percent=hop_percent, max_len=T_orig)
    n_hat = istft(N_hat, fs=fs, wlen_sec=wlen_sec, win=win, hop_percent=hop_percent, max_len=T_orig)

    # Save .wav files
    output_path = output_data_dir + file_path
    output_path = os.path.splitext(output_path)[0]

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    sf.write(output_path + '_s_est.wav', s_hat, fs)
    sf.write(output_path + '_n_est.wav', n_hat, fs)
    
    # Save binary mask
    torch.save(y_hat_soft, output_path + ' _ibm_soft_est.pt')
    torch.save(y_hat_hard, output_path + '_ibm_hard_est.pt')
    
def process_sublist(device, sublist, mcem, model, classifier):

    if cuda: model = model.to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if classif_type == 'dnn':
        if cuda: classifier = classifier.to(device)
        
        classifier.eval()
        for param in classifier.parameters():
            param.requires_grad = False

        if std_norm:
            # Load mean and variance
            mean = np.load(os.path.dirname(classif_path) + '/' + 'trainset_mean.npy')
            std = np.load(os.path.dirname(classif_path) + '/' + 'trainset_std.npy')

            mean = torch.tensor(mean).to(device)
            std = torch.tensor(std).to(device)
    
    for file_path in sublist:
        process_utt(mcem, model, classifier, mean, std, file_path, device)

def main():
    file = open('output.log','w') 

    print('Torch version: {}'.format(torch.__version__))

    # Start context for GPU multiprocessing
    ctx = multiprocessing.get_context('spawn')

    print('Load models')
    if classif_type == 'dnn':
        classifier = Classifier([x_dim, h_dim_cl, y_dim])
        classifier.load_state_dict(torch.load(classif_path, map_location="cpu"))
    
    if classif_type in ['oracle', 'timo']:
        classifier = None

    model = DeepGenerativeModel([x_dim, y_dim, z_dim, h_dim], None)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    print('- Number of learnable parameters: {}'.format(count_parameters(model)))
    
    mcem = MCEM_M2(niter=niter,
                   nsamples_E_step=nsamples_E_step,
                   burnin_E_step=burnin_E_step, nsamples_WF=nsamples_WF, 
                   burnin_WF=burnin_WF, var_RW=var_RW)

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir, dataset_type=dataset_type)
    print('- Number of test samples: {}'.format(len(file_paths)))

    # Split list in nb_devices * nb_processes_per_device
    b = np.array_split(file_paths, nb_devices*nb_process_per_device)
    
    # Assign each list to a process
    b = [(i%nb_devices, sublist, mcem, model, classifier) for i, sublist in enumerate(b)]

    print('Start evaluation')
    # start = time.time()
    t1 = time.perf_counter()
    
    with ctx.Pool(processes=nb_process_per_device*nb_devices) as multi_pool:
        multi_pool.starmap(process_sublist, b)
    
    # # Test script on 1 sublist
    # process_sublist(*b[0])

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

if __name__ == '__main__':
    main()