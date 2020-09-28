import os
import pickle
import numpy as np
import torch
from torch import nn
import time
import soundfile as sf

from python.dataset.csr1_wjs0_dataset import speech_list
from python.processing.stft import stft, istft
from python.processing.target import clean_speech_IBM
from python.models import mcem_julius, mcem_simon
from python.models.models import DeepGenerativeModel
#from stcn import STCN, evaluate, model_parameters
#from utils import count_parameters


# Settings
dataset_type = 'test'
input_speech_dir = 'data/subset/raw/'
processed_data_dir = 'data/subset/processed/'
#TODO: input from processed data

cuda = torch.cuda.is_available()
eps = np.finfo(float).eps # machine epsilon


# Parameters
## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'



## Deep Generative Model
model_name = 'dummy_M2_10_epoch_010_vloss_108.79.pt'
x_dim = 513 # frequency bins (spectrogram)
y_dim = 513 # frequency bins (binary mask)
z_dim = 128
h_dim = [256, 128]

## Monte-Carlo EM
use_mcem_julius = False
use_mcem_simon = True

### NMF parameters (noise model)
nmf_rank = 8

### MCEM settings
niter = 100 # results reported in the paper were obtained with 500 iterations 
nsamples_E_step = 10
burnin_E_step = 30
nsamples_WF = 25
burnin_WF = 75
var_RW = 0.01

# Output_data_dir
output_data_dir = 'data/subset/models/' + model_name + '/'


def main():

    device = torch.device("cuda" if cuda else "cpu")
    file = open('output.log','w') 

    print('Torch version: {}'.format(torch.__version__))
    print('Device: %s' % (device))
    if torch.cuda.device_count() >= 1: print("Number GPUs: ", torch.cuda.device_count())

    #TODO: modify and just read stored .wav files
    if dataset_type:
        test_data = pickle.load(open('data/subset/pickle/si_et_05_mixture-505.p', 'rb'))

    model = DeepGenerativeModel([x_dim, y_dim, z_dim, h_dim])
    model.load_state_dict(torch.load('models/' + model_name))
    if cuda: model = model.cuda()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                             dataset_type=dataset_type)

    # s_hat_list = []
    # n_hat_list = []

    for i, (x_t, file_path) in enumerate(zip(test_data, file_paths)):
        
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

        # Classify
        y_hat = model.classify(x) # (frames, freq_bins)
        
        # # Target
        # s_t, fs_s = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_s.wav') # clean speech
        # s_tf = stft(s_t,
        #          fs=fs,
        #          wlen_sec=wlen_sec,
        #          win=win,
        #          hop_percent=hop_percent,
        #          dtype=dtype) # shape = (freq_bins, frames)
        # y_hat = clean_speech_IBM(s_tf,
        #                      quantile_fraction=0.98,
        #                      quantile_weight=0.999)
        # y_hat = torch.from_numpy(y_hat.T).to(device)

        # Encode
        Z_init, _, _ = model.encoder(torch.cat([x, y_hat], dim=1))

        # MCEM
        if use_mcem_julius and not use_mcem_simon:

            # NMF parameters are initialized inside MCEM
            mcem = mcem_julius.MCEM(X=x_tf.T,
                                    Z=Z_init.T,
                                    y=y_hat.T,
                                    model=model,
                                    device=device,
                                    niter_MCEM=niter,
                                    niter_MH=nsamples_E_step+burnin_E_step,
                                    burnin=burnin_E_step,
                                    var_MH=var_RW,
                                    NMF_rank=nmf_rank)
            
            t0 = time.time()

            mcem.run()
            mcem.separate(niter_MH=nsamples_WF+burnin_WF, burnin=burnin_WF)

            elapsed = time.time() - t0
            print("elapsed time: %.4f s" % (elapsed))

        elif not use_mcem_julius and use_mcem_simon:

            # NMF parameters are initialized outside MCEM
            N, F = x_tf.shape
            W_init = np.maximum(np.random.rand(F,nmf_rank), eps)
            H_init = np.maximum(np.random.rand(nmf_rank, N), eps)
            g_init = torch.ones(N).to(device)

            mcem = mcem_simon.MCEM(X=x_tf,
                                W=W_init,
                                H=H_init,
                                g=g_init,
                                Z=Z_init,
                                y=y_hat,
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
        
        # Save binary mask
        torch.save(y_hat, output_path + '_ibm_est.pt')

    # pickle.dump(s_hat, open('../data/pickle/s_hat_vae', 'wb'), protocol=4)
    # pickle.dump(n_hat, open('../data/pickle/n_hat_vae', 'wb'), protocol=4)
        
if __name__ == '__main__':
    main()