import os
import sys
import pickle
import numpy as np
import torch
import time
import soundfile as sf

sys.path.append('.')

from python.dataset.csr1_wjs0_dataset import speech_list
from python.processing.stft import stft, istft
from python.processing.target import clean_speech_IBM
from python.utils import count_parameters
from python.models.mcem import MCEM_M2
from python.models.models import DeepGenerativeModel, Classifier
from python.models.spp_estimation import timo_mask_estimation

##################################### SETTINGS #####################################################

# Dataset
dataset_type = 'test'

# dataset_size = 'subset'
dataset_size = 'complete'

# System 
cuda = torch.cuda.is_available()
cuda_device = "cuda:2"
device = torch.device(cuda_device if cuda else "cpu")

# STFT parameters
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window

# Hyperparameters 
## Wiener
# model_name = 'wiener_maskloss_normdataset_hdim_128_128_128_128_128_end_epoch_200/Classifier_epoch_096_vloss_46.936924'
model_name = 'wiener_maskloss_normdataset_hdim_128_128_128_128_128_end_epoch_200/Classifier_epoch_052_vloss_47.230169'
# model_name = 'wiener_signalloss_normdataset_hdim_128_128_128_128_128_end_epoch_200/Classifier_epoch_022_vloss_82.984183'
x_dim = 513 
y_dim = 513
h_dim = [128, 128, 128, 128, 128]
eps = 1e-8
std_norm = True

model_dir = os.path.join('models', model_name + '.pt')

if std_norm:
    # Load mean and variance
    mean = np.load(os.path.dirname(model_dir) + '/' + 'trainset_mean.npy')
    std = np.load(os.path.dirname(model_dir) + '/' + 'trainset_std.npy')

    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)

# NMF
nmf_rank = 10

### MCEM settings
niter = 100 # results reported in the paper were obtained with 500 iterations 
nsamples_E_step = 10
burnin_E_step = 30
nsamples_WF = 25
burnin_WF = 75
var_RW = 0.01

# Data directories
input_speech_dir = os.path.join('data', dataset_size,'raw/')
output_data_dir = os.path.join('data', dataset_size, 'models', model_name + '/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')

#####################################################################################################

def main():
    file = open('output.log','w') 

    print('Load models')
    model = Classifier([x_dim, h_dim, y_dim])
    model.load_state_dict(torch.load(model_dir, map_location=cuda_device))
    if cuda: model = model.to(device)

    print('- Number of learnable parameters: {}'.format(count_parameters(model)))

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir, dataset_type=dataset_type)
    print('- Number of test samples: {}'.format(len(file_paths)))

    print('Start evaluation')
    start = time.time()
    elapsed = []
    for i, file_path in enumerate(file_paths):
        start_file = time.time()
        print('- File {}/{}'.format(i+1,len(file_paths)), end='\r')

        x_t, fs_x = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_x.wav') # mixture

        T_orig = len(x_t)
        x_tf = stft(x_t, fs, wlen_sec, win, hop_percent).T # (frames, freq_bins)
        x = torch.tensor(np.power(np.abs(x_tf), 2)).to(device)
        
        
        # Normalize power spectrogram
        if std_norm:
            x_classif = x - mean.T
            x_classif /= (std + eps).T

            y_hat_soft = model(x_classif) 
        else:
            y_hat_soft = model(x)   

        # Apply estimated filter
        S_hat = y_hat_soft.cpu().numpy() * x_tf
        S_hat = S_hat.T

        s_hat = istft(S_hat, fs=fs, wlen_sec=wlen_sec, win=win, hop_percent=hop_percent, max_len=T_orig)

        # Save .wav files
        output_path = output_data_dir + file_path
        output_path = os.path.splitext(output_path)[0]

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        
        sf.write(output_path + '_s_est.wav', s_hat, fs)
        
        # Save binary mask
        torch.save(y_hat_soft, output_path + ' _ibm_soft_est.pt')
        
        end_file = time.time()
        elapsed.append(end_file - start_file)
        etc = (len(file_paths)-i-1)*np.mean(elapsed)

        print("                   average time per file: {:4.1f} s      ETC: {:d} h, {:2d} min, {:2d} s"\
            "".format(np.mean(elapsed), int(etc/(60*60)), int((etc/60) % 60), int(etc % 60)), end='\r')

    end = time.time()
    print('- File {}/{}   '.format(len(file_paths), len(file_paths)))
    print('                     total time: {:6.1f} s'.format(end-start))
        
if __name__ == '__main__':
    main()