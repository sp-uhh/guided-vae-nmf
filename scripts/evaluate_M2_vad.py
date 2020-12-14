import os
import sys
import pickle
import numpy as np
import torch
import time
import soundfile as sf

sys.path.append('.')

from python.processing.stft import stft, istft
from python.processing.target import clean_speech_VAD
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
cuda_device = "cuda:0"
device = torch.device(cuda_device if cuda else "cpu")

# STFT parameters
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window

## Ideal binary mask
quantile_fraction = 0.999
quantile_weight = 0.999

# Hyperparameters 
# M2
# model_name = 'M2_VAD_hdim_128_128_zdim_032_end_epoch_100/M2_epoch_085_vloss_465.98'
model_name = 'M2_VAD_quantile_0.999_hdim_128_128_zdim_032_end_epoch_200/M2_epoch_087_vloss_482.93'
x_dim = 513 
y_dim = 1
z_dim = 32
h_dim = [128, 128]
eps = 1e-8

model_dir = os.path.join('models_wsj0', model_name + '.pt')

## Classifier
# classif_type = 'dnn'
classif_type = 'oracle'

if classif_type == 'dnn':
    # classif_name = 'classif_VAD_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_096_vloss_0.21'
    classif_name = 'classif_VAD_qf0.999_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_090_vloss_0.23'
    h_dim_cl = [128, 128]
    std_norm = True

if classif_type == 'oracle':
    classif_name = 'oracle_classif'
    # classif_name = 'ones_classif'
    # classif_name = 'zeros_classif'

classif_dir = os.path.join('models_wsj0', classif_name + '.pt')

if classif_type == 'dnn' and std_norm:
    # Load mean and variance
    mean = np.load(os.path.dirname(classif_dir) + '/' + 'trainset_mean.npy')
    std = np.load(os.path.dirname(classif_dir) + '/' + 'trainset_std.npy')

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
output_data_dir = os.path.join('data', dataset_size, 'models', model_name, classif_name + '/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')

#####################################################################################################

def main():
    file = open('output.log','w') 

    print('Load models')
    if classif_type == 'dnn':
        classifier = Classifier([x_dim, h_dim_cl, y_dim])
        classifier.load_state_dict(torch.load(classif_dir, map_location=cuda_device))
        if cuda: classifier = classifier.cuda()

    if classif_type == 'oracle':
        classifier = None

    model = DeepGenerativeModel([x_dim, y_dim, z_dim, h_dim], None)
    model.load_state_dict(torch.load(model_dir, map_location=cuda_device))
    if cuda: model = model.cuda()

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

        # Input
        x_t, fs_x = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_x.wav') # mixture
        T_orig = len(x_t)
        x_tf = stft(x_t, fs, wlen_sec, win, hop_percent).T # (frames, freq_bins)
        
        x = torch.tensor(np.power(np.abs(x_tf), 2)).to(device)

        # Target
        s_t, fs_s = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_s.wav') # clean speech
        s_tf = stft(s_t, fs, wlen_sec, win, hop_percent)

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
            y_hat_soft = clean_speech_VAD(s_tf, quantile_fraction=quantile_fraction, quantile_weight=quantile_weight)
            #y_hat_soft = np.ones(s_tf.shape[1], dtype='float32')[None]
            # y_hat_soft = np.zeros(s_tf.shape[1], dtype='float32')[None]
            y_hat_hard = torch.from_numpy(y_hat_soft.T).to(device)

        # Encode
        _, Z_init, _ = model.encoder(torch.cat([x, y_hat_hard], dim=1))

        # NMF parameters are initialized outside MCEM
        N, F = x_tf.shape
        W_init = np.maximum(np.random.rand(F,nmf_rank), eps)
        H_init = np.maximum(np.random.rand(nmf_rank, N), eps)
        g_init = torch.ones(N).to(device)

        mcem = MCEM_M2(X=x_tf, W=W_init, H=H_init, g=g_init, Z=Z_init, y=y_hat_hard,
                            vae=model, device=device, niter=niter,
                            nsamples_E_step=nsamples_E_step,
                            burnin_E_step=burnin_E_step, nsamples_WF=nsamples_WF, 
                            burnin_WF=burnin_WF, var_RW=var_RW)
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