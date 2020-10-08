import sys
sys.path.append('.')

import os
import pickle
import numpy as np
import torch
from torch import nn
import time
import soundfile as sf
from tqdm import tqdm

from python.dataset.csr1_wjs0_dataset import speech_list
from python.processing.stft import stft, istft
from python.processing.target import clean_speech_IBM
from python.models import mcem_simon, mcem_julius
from python.models.models import VariationalAutoencoder
#from utils import count_parameters


# Settings
dataset_type = 'test'

dataset_size = 'subset'
#dataset_size = 'complete'

input_speech_dir = os.path.join('data',dataset_size,'raw/')
#processed_data_dir = os.path.joint('data',dataset_size,'processed/')

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

model_name = 'M1_KLv3_eps1e-8_h128_z032_end_epoch_250/M1_epoch_200_vloss_439.46'
x_dim = 513 # frequency bins (spectrogram)
z_dim = 32
h_dim = [128]
eps = 1e-8

## Monte-Carlo EM
use_mcem_julius = False
use_mcem_simon = True

### NMF parameters (noise model)
nmf_rank = 10

### MCEM settings
niter = 100 # results reported in the paper were obtained with 500 iterations 
nsamples_E_step = 10
burnin_E_step = 30
nsamples_WF = 25
burnin_WF = 75
var_RW = 0.01

# Output_data_dir
output_data_dir = os.path.join('data', dataset_size, 'models', model_name + '/')


def main():

    device = torch.device("cuda" if cuda else "cpu")
    file = open('output.log','w') 

    print('Torch version: {}'.format(torch.__version__))
    print('Device: %s' % (device))
    if torch.cuda.device_count() >= 1: print("Number GPUs: ", torch.cuda.device_count())

    #TODO: modify and just read stored .wav files
    test_data = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_et_05_mixture-505.p'), 'rb'))

    model = VariationalAutoencoder([x_dim, z_dim, h_dim])
    model.load_state_dict(torch.load(os.path.join('models', model_name + '.pt'), map_location="cuda:0"))
    if cuda: model = model.cuda()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                             dataset_type=dataset_type)

    # s_hat_list = []
    # n_hat_list = []

    for i, (x_t, file_path) in tqdm(enumerate(zip(test_data, file_paths))):
        
        print('File {}/{}'.format(i+1,len(test_data)), file=open('output.log','a'))
        # x = x/np.max(x)
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
        
        # Power spectrogram (transpose)
        x = torch.tensor(np.power(np.abs(x_tf), 2)).to(device)

        # Encode
        _, Z_init, _ = model.encoder(x)

        # MCEM
        if use_mcem_julius and not use_mcem_simon:

            # NMF parameters are initialized inside MCEM
            mcem = mcem_julius.MCEM_M1(X=x_tf.T,
                                    Z=Z_init.T,
                                    model=model,
                                    device=device,
                                    niter_MCEM=niter,
                                    niter_MH=nsamples_E_step+burnin_E_step,
                                    burnin=burnin_E_step,
                                    var_MH=var_RW,
                                    NMF_rank=nmf_rank,
                                    eps=eps)
            
            t0 = time.time()

            mcem.run()
            mcem.separate(niter_MH=nsamples_WF+burnin_WF, burnin=burnin_WF)

            elapsed = time.time() - t0
            print("elapsed time: %.4f s" % (elapsed))

        elif not use_mcem_julius and use_mcem_simon:

            # NMF parameters are initialized outside MCEM
            N, F = x_tf.shape
            W_init = np.maximum(np.random.rand(F,nmf_rank), eps, dtype='float32')
            H_init = np.maximum(np.random.rand(nmf_rank, N), eps, dtype='float32')
            g_init = torch.ones(N).to(device) # float32 by default

            mcem = mcem_simon.MCEM_M1(X=x_tf,
                                W=W_init,
                                H=H_init,
                                g=g_init,
                                Z=Z_init,
                                vae=model, device=device, niter=niter,
                                nsamples_E_step=nsamples_E_step,
                                burnin_E_step=burnin_E_step, nsamples_WF=nsamples_WF, 
                                burnin_WF=burnin_WF, var_RW=var_RW)
            
            #%% Run speech enhancement algorithm
            cost = mcem.run()
        else:
            ValueError('You must set use_mcem_julius OR use_mcem_simon to True.')

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

    # pickle.dump(s_hat, open('../data/pickle/s_hat_vae', 'wb'), protocol=4)
    # pickle.dump(n_hat, open('../data/pickle/n_hat_vae', 'wb'), protocol=4)
        
if __name__ == '__main__':
    main()