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

dataset_size = 'subset'
#dataset_size = 'complete'

# System 
cuda = torch.cuda.is_available()
cuda_device = "cuda:0"
device = torch.device(cuda_device if cuda else "cpu")

# STFT parameters
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window

# Hyperparameters 
# M2
model_name = 'M2_hdim_128_128_zdim_032_end_epoch_100/M2_epoch_085_vloss_417.69'
x_dim = 513 
y_dim = 513
z_dim = 32
h_dim = [128, 128]
eps = 1e-8

# Classifier
classifier_name = 'CLASSIFIER_NAME'
h_dim_cl = [128, 128]

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

    print('Load data')

    #TODO: modify and just read stored .wav files
    test_data = pickle.load(open(os.path.join('data', dataset_size, 'pickle/si_et_05_mixture-505.p'), 'rb'))

    print('- Number of test samples: {}'.format(len(test_data)))

    print('Load models')
    classifier = Classifier([x_dim, h_dim_cl, y_dim])
    classifier.load_state_dict(torch.load(os.path.join('models', classifier_name + '.pt'), map_location=cuda_device))
    if cuda: classifier = classifier.cuda()

    model = DeepGenerativeModel([x_dim, y_dim, z_dim, h_dim], classifier)
    model.load_state_dict(torch.load(os.path.join('models', model_name + '.pt'), map_location=cuda_device))
    if cuda: model = model.cuda()

    print('- Number of learnable parameters: {}'.format(count_parameters(model)))

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir, dataset_type=dataset_type)

    print('Start evaluation')
    start = time.time()
    elapsed = []
    for i, (x_t, file_path) in enumerate(zip(test_data, file_paths)):
        start_file = time.time()
        print('- File {}/{}'.format(i+1,len(test_data)), end='\r')

        T_orig = len(x_t)
        x_tf = stft(x_t, fs, wlen_sec, win, hop_percent).T # (frames, freq_bins)
        x = torch.tensor(np.power(np.abs(x_tf), 2)).to(device)

        y_hat_soft = model.classify(x) 
        y_hat_hard = (y_hat_soft > 0.5).float()

        # Encode
        x = torch.tensor(x).to(device)
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
        
        sf.write(output_path + '_' + classifier_name + '_s_est.wav', s_hat, fs)
        sf.write(output_path + '_' + classifier_name + '_n_est.wav', n_hat, fs)
        
        # Save binary mask
        torch.save(y_hat_soft, output_path + '_' + classifier_name + ' _ibm_soft_est.pt')
        torch.save(y_hat_hard, output_path + '_' + classifier_name + '_ibm_hard_est.pt')
        
        end_file = time.time()
        elapsed.append(end_file - start_file)
        etc = (len(test_data)-i-1)*np.mean(elapsed)

        print("                   average time per file: {:4.1f} s      ETC: {:d} h, {:2d} min, {:2d} s"\
            "".format(np.mean(elapsed), int(etc/(60*60)), int((etc/60) % 60), int(etc % 60)), end='\r')

    end = time.time()
    print('- File {}/{}   '.format(len(test_data), len(test_data)))
    print('                     total time: {:6.1f} s'.format(end-start))
        
if __name__ == '__main__':
    main()