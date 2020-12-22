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
# import torch.multiprocessing as mp
# from torch.nn.parallel import DataParallel as DP
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.multiprocessing import Pool, Process
import torch.multiprocessing as multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

from python.dataset.csr1_wjs0_dataset import speech_list
from python.processing.stft import stft, istft
from python.processing.target import clean_speech_IBM
from python.models import mcem_julius
from python.models.mcem import MCEM_M1
from python.models.models import VariationalAutoencoder
#from utils import count_parameters


# Settings
dataset_type = 'test'

dataset_size = 'subset'
# dataset_size = 'complete'

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

# Data directories
output_data_dir = os.path.join('data', dataset_size, 'models', model_name + '/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')


# class MyDataParallel(nn.DataParallel):
#     def __getattr__(self, name):
#         return getattr(self.module, name)

# def process_utt(args):
def process_utt(mcem, model, file_path, device):
    # mcem = args[0]
    # file_path = args[1]

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
    
    # Power spectrogram (transpose)
    # x = torch.tensor(np.power(np.abs(x_tf), 2)).to(device)

    # Encode
    # _, Z_init, _ = model.encoder(x)
    # _, Z_init, _ = model.module.encoder(x)

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
        # W_init = np.maximum(np.random.rand(F,nmf_rank), eps, dtype='float32')
        W_init = torch.max(torch.rand(F,nmf_rank, device=device), eps * torch.ones(F, nmf_rank, device=device))
        # H_init = np.maximum(np.random.rand(nmf_rank, N), eps, dtype='float32')
        H_init = torch.max(torch.rand(nmf_rank, N, device=device), eps * torch.ones(nmf_rank, N, device=device))
        g_init = torch.ones(N, device=device) # float32 by default
        # g_init = np.ones(N, dtype='float32')

        # mcem = MCEM_M1(X=x_tf,
        #                 W=W_init,
        #                 H=H_init,
        #                 g=g_init,
        #                 Z=Z_init,
        #                 vae=model, device=device, niter=niter,
        #                 nsamples_E_step=nsamples_E_step,
        #                 burnin_E_step=burnin_E_step, nsamples_WF=nsamples_WF, 
        #                 burnin_WF=burnin_WF, var_RW=var_RW)
        
        mcem.weight_reset(vae=model,
                          X=x_tf,
                          W=W_init,
                          H=H_init,
                          g=g_init,
                          device=device)
        
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

    # # Save .wav files
    # output_path = output_data_dir + file_path
    # output_path = os.path.splitext(output_path)[0]

    # if not os.path.exists(os.path.dirname(output_path)):
    #     os.makedirs(os.path.dirname(output_path))
    
    # sf.write(output_path + '_s_est.wav', s_hat, fs)
    # sf.write(output_path + '_n_est.wav', n_hat, fs)

    # end_file = time.time()
    # elapsed.append(end_file - start_file)
    # etc = (len(file_paths)-i-1)*np.mean(elapsed)

    # end = time.time()
    # print('- File {}/{}   '.format(len(file_paths), len(file_paths)))
    # print('                     total time: {:6.1f} s'.format(end-start))

class MyDataParallel:
    def __init__(self, model, mcem, device):
        self.model = model
        self.mcem = mcem
        self.device = device
    
    def process_utt(self, file_path):
        return process_utt(file_path, self.model, self.mcem, self.device)


def process_sublist(device, sublist, mcem, model):
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model)
    #     mcem = torch.nn.DataParallel(mcem)
    if cuda: model = model.to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    for file_path in sublist:
        process_utt(mcem, model, file_path, device)

def main():
    #TODO: insert Pool Process here
    #TODO: count the number of GPUs, then use torch.multiprocessing.Process or Pool
    # n_gpus = torch.cuda.device_count()
    # with torch.multiprocessing.Pool(processes=torch.cuda.device_count()) as pool:

    # cuda_device = "cuda:0"
    # cuda_device = "cuda:3"
    # device = torch.device(cuda_device if cuda else "cpu")
    file = open('output.log','w') 

    print('Torch version: {}'.format(torch.__version__))
    # print('Device: %s' % (device))
    # if torch.cuda.device_count() >= 1: print("Number GPUs: ", torch.cuda.device_count())
    
    nb_devices = torch.cuda.device_count()
    nb_process_per_device = 2

    ctx = multiprocessing.get_context('spawn')

    # mcems = []
    # for device in range(nb_devices):
    #     cuda_device = "cuda:%g"%device
    #     model = VariationalAutoencoder([x_dim, z_dim, h_dim])
    #     model.load_state_dict(torch.load(os.path.join('models_wsj0', model_name + '.pt'), map_location=cuda_device))

    #     # if torch.cuda.device_count() > 1:
    #     #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     #     model = torch.nn.DataParallel(model)
    #     #     mcem = torch.nn.DataParallel(mcem)
    #     if cuda: model = model.to(device)

    #     model.eval()
    #     for param in model.parameters():
    #         param.requires_grad = False
        
    #     mcem = MCEM_M1(vae=model, device=device, niter=niter,
    #         nsamples_E_step=nsamples_E_step,
    #         burnin_E_step=burnin_E_step, nsamples_WF=nsamples_WF, 
    #         burnin_WF=burnin_WF, var_RW=var_RW)
        
    #     # models.append(model)
    #     mcems.append(mcem)

    model = VariationalAutoencoder([x_dim, z_dim, h_dim])
    model.load_state_dict(torch.load(os.path.join('models_wsj0', model_name + '.pt'), map_location="cpu"))
    
    mcem = MCEM_M1(niter=niter,
        nsamples_E_step=nsamples_E_step,
        burnin_E_step=burnin_E_step, nsamples_WF=nsamples_WF, 
        burnin_WF=burnin_WF, var_RW=var_RW)

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                             dataset_type=dataset_type)
    
    file_paths = file_paths * 20
   
    # Split list in nb_devices * nb_processes_per_device
    # b = [(mcems[i%nb_devices], file_path) for i, file_path in enumerate(file_paths)]
    # b = [[mcems[i%nb_devices], file_path] for i, file_path in enumerate(file_paths)]
    b = np.array_split(file_paths, nb_devices*nb_process_per_device)
    
    # Assign each list to a process
    b = [(i%nb_devices, sublist, mcem, model) for i, sublist in enumerate(b)]
    
    #TODO: repartir les fichiers sur les 4 GPUs
    # print('Start evaluation')
    # start = time.time()
    # elapsed = []

    t1 = time.perf_counter()
    
    # num_processes = 2
    # do_something = MyDataParallel(model, mcem, device)
    # multi_pool = Pool(processes=nb_devices)

    with ctx.Pool(processes=nb_process_per_device*nb_devices) as multi_pool:
        # predictions = multi_pool.map(process_utt, b)
        predictions = multi_pool.starmap(process_sublist, b)
        # predictions = multi_pool.apply_async(process_utt, b)
    
    # multi_pool.close() 
    # multi_pool.join()

    # processes = [ctx.Process(target=process_utt, args=(b_item,)) for b_item in range(nb_devices)]
    # for process in processes:
    #     process.start()
    # for process in processes:
    #     process.join() 

    # print('start')
    # with ctx.Pool(processes=nb_devices) as p:
    #     results = p.starmap(process_utt, b)

    # #TODO: split in batches?
    # for i, file_path in tqdm(enumerate(file_paths)):   
    #     start_file = time.time()
    #     print('- File {}/{}'.format(i+1,len(file_paths)), end='\r')
    #     process_utt(mcem, file_path)

    # process_sublist(*b[0])

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

def use_gpu(ind, arr):  #A):
    # A = (2 * arr).to(ind)
    # arr = A[ind].to(ind)
    return (arr.std() + arr.mean()/(1+ arr.abs())).sum()


def mysenddata(mydata):
    return [(ii%4, mydata[ii%4].cuda(ii%4)) for ii in range(8)]

def mp_worker(gpu):
    print(torch.cuda.get_device_properties(gpu))

def foo(worker,tl):
    # tl[worker] += (worker+1) * 1000
    print((tl[worker].std() + tl[worker].mean()/(1+ tl[worker].abs())).sum())

if __name__ == '__main__':
    main()

    # print('create big tensor')
    # aa = 10*torch.randn(4,10000,10000).double()
    # print('send data')
    # b = mysenddata(aa)

    # ctx = multiprocessing.get_context('spawn')

    # # for ii in range(10):
    # # pool = Pool(processes=4)
    # a = time.time()
    # print('start')
    # with ctx.Pool(processes=4) as p:
    # #result = pool.starmap(use_gpu, b,)
    #     results = p.starmap(use_gpu, b,)
    # print('end')
    # print("cost time :", time.time() - a)
    
    # for ii, (rr, bb) in enumerate(zip(results, b)):
    #     print('idx:{}, inshape:{}, indevice:{}, intype:{}, outshape:{}, outdevice:{}, outtype:{}'.format(ii, bb[1].shape, bb[1].get_device(), bb[1].type(), rr.shape, rr.get_device(), rr.type()))

    # gpus = list(range(torch.cuda.device_count()))
    # gpus = 2 * gpus

    # ctx = multiprocessing.get_context('spawn')

    # processes = [ctx.Process(target=mp_worker, args=(gpui,)) for gpui in gpus]
    # for process in processes:
    #     process.start()
    # for process in processes:
    #     process.join()   

    # print('start')
    # with ctx.Pool(processes=len(gpus)) as p:
    #     results = p.starmap(mp_worker, gpus)

    # tl = [10*torch.randn(10000,10000).double().to(0),
    #       10*torch.randn(10000,10000).double().to(1),
    #       10*torch.randn(10000,10000).double().to(2),
    #       10*torch.randn(10000,10000).double().to(3)]

    # for t in tl:
    #     t.share_memory_()

    # print("before mp: tl=")
    # print(tl)

    # a = time.time()
    # print('start')
    # processes = []
    # for i in range(20):
    #     p = multiprocessing.Process(target=foo, args=(i%4, tl))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
    # print('end')
    # print("cost time :", time.time() - a)


    # print("after mp: tl=")
    # print(tl)