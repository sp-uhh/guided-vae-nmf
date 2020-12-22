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
from python.models.mcem import MCEM_M1
from python.models.models import VariationalAutoencoder
#from utils import count_parameters

# Dataset
dataset_name = 'CSR-1-WSJ-0'
if dataset_name == 'CSR-1-WSJ-0':
    from python.dataset.csr1_wjs0_dataset import speech_list

dataset_type = 'test'

dataset_size = 'subset'
# dataset_size = 'complete'

cuda = torch.cuda.is_available()
eps = 1e-8


# Parameters
## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'


## Deep Generative Model
# model_name = 'M1_end_epoch_050/M1_epoch_036_vloss_465.28'
# model_name = 'M1_end_epoch_050/M1_epoch_025_vloss_476.34'
# model_name = 'M1_end_epoch_050/M1_epoch_005_vloss_527.95'
# x_dim = 513 # frequency bins (spectrogram)
# z_dim = 128
# h_dim = [256, 128]

# model_name = 'M1_h128_end_epoch_050/M1_epoch_044_vloss_486.29'
# x_dim = 513 # frequency bins (spectrogram)
# z_dim = 128
# h_dim = [128]

# model_name = 'M1_h128_z16_end_epoch_250/M1_epoch_010_vloss_526.58'
# x_dim = 513 # frequency bins (spectrogram)
# z_dim = 16
# h_dim = [128]

# model_name = 'M1_h128_z032_end_epoch_250/M1_epoch_200_vloss_467.83'
# x_dim = 513 # frequency bins (spectrogram)
# z_dim = 32
# h_dim = [128]

# model_name = 'M1_ISv2_eps1e-5_h128_z032_end_epoch_250/M1_epoch_200_vloss_434.08'
# x_dim = 513 # frequency bins (spectrogram)
# z_dim = 32
# h_dim = [128]
# eps = 1e-5

# model_name = 'M1_ISv2_h128_z032_end_epoch_250/M1_epoch_200_vloss_467.95'
# x_dim = 513 # frequency bins (spectrogram)
# z_dim = 32
# h_dim = [128]
# eps = 1e-8

# model_name = 'M1_hdim_128_128_zdim_032_end_epoch_200/M1_epoch_085_vloss_479.69'
model_name = 'M1_hdim_128_128_zdim_032_end_epoch_200/M1_epoch_124_vloss_475.95'
x_dim = 513 # frequency bins (spectrogram)
z_dim = 32
h_dim = [128, 128]
eps = 1e-8

# model_name = 'M1_end_epoch_200/M1_epoch_085_vloss_475.56'
# x_dim = 513 # frequency bins (spectrogram)
# z_dim = 16
# h_dim = [128, 128]
# eps = 1e-8

### NMF parameters (noise model)
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
input_speech_dir = os.path.join('data',dataset_size,'raw/')
output_data_dir = os.path.join('data', dataset_size, 'models', model_name + '/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')

#####################################################################################################

def process_utt(mcem, model, file_path, device):

    # Input
    x_t, fs_x = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_x.wav') # mixture
    T_orig = len(x_t)

    # TF reprepsentation
    # Input should be (frames, freq_bibs)
    x_tf = stft(x_t,
                fs=fs,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent,
                dtype=dtype)
                    
    # Transpose to match PyTorch
    x_tf = x_tf.T # (frames, freq_bins)
    
    # Init MCEM
    mcem.init_parameters(X=x_tf,
                        vae=model,
                        nmf_rank=nmf_rank,
                        eps=eps,
                        device=device)
    
    #%% Run speech enhancement algorithm
    cost = mcem.run()

    # Estimated sources
    S_hat = mcem.S_hat #+ np.finfo(np.float32).eps
    N_hat = mcem.N_hat #+ np.finfo(np.float32).eps

    # iSTFT
    s_hat = istft(S_hat,
                fs=fs,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent,
                max_len=T_orig)

    n_hat = istft(N_hat,
        fs=fs,
        wlen_sec=wlen_sec,
        win=win,
        hop_percent=hop_percent,
        max_len=T_orig)

    # Save .wav files
    output_path = output_data_dir + file_path
    output_path = os.path.splitext(output_path)[0]

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    sf.write(output_path + '_s_est.wav', s_hat, fs)
    sf.write(output_path + '_n_est.wav', n_hat, fs)

    # end_file = time.time()
    # elapsed.append(end_file - start_file)
    # etc = (len(file_paths)-i-1)*np.mean(elapsed)

    # end = time.time()
    # print('- File {}/{}   '.format(len(file_paths), len(file_paths)))
    # print('                     total time: {:6.1f} s'.format(end-start))

def process_sublist(device, sublist, mcem, model):

    if cuda: model = model.to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    for file_path in sublist:
        process_utt(mcem, model, file_path, device)

def main():
    file = open('output.log','w') 

    print('Torch version: {}'.format(torch.__version__))
    # print('Device: %s' % (device))
    # if torch.cuda.device_count() >= 1: print("Number GPUs: ", torch.cuda.device_count())
    
    ctx = multiprocessing.get_context('spawn')

    # Load model
    model = VariationalAutoencoder([x_dim, z_dim, h_dim])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    # Create MCEM object
    mcem = MCEM_M1(niter=niter,
        nsamples_E_step=nsamples_E_step,
        burnin_E_step=burnin_E_step, nsamples_WF=nsamples_WF, 
        burnin_WF=burnin_WF, var_RW=var_RW)

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                             dataset_type=dataset_type)
   
    # Split list in nb_devices * nb_processes_per_device
    b = np.array_split(file_paths, nb_devices*nb_process_per_device)
    
    # Assign each list to a process
    b = [(i%nb_devices, sublist, mcem, model) for i, sublist in enumerate(b)]
    
    # print('Start evaluation')
    # start = time.time()
    # elapsed = []
    print('Start evaluation')

    t1 = time.perf_counter()
    
    with ctx.Pool(processes=nb_process_per_device*nb_devices) as multi_pool:
        multi_pool.starmap(process_sublist, b)

    # Test script on 1 sublist
    # process_sublist(*b[0])

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

if __name__ == '__main__':
    main()