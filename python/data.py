import numpy as np
import torch
from torch.utils.data import Dataset
from librosa import stft
from glob import glob
# import tables #needed for blosc compression
import h5py as h5

# STFT parameters
fs = int(16e3) 
wlen_sec = 64e-3 
wlen = np.int(np.power(2, np.ceil(np.log2(wlen_sec*fs))))
win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi)
hop_percent = 0.25 
hop = np.int(hop_percent*wlen) 
nfft = wlen

def collate_fn(data):    
    max_length = max(sample.shape[1] for sample in data)      
    batch = []
    for sample in data:
        batch.append(np.pad(sample, ((0, 0), (0, max_length-sample.shape[1])), 'minimum'))
    return torch.tensor(batch)

class Spectogram(Dataset):
    def __init__(self, data):
        self.data = data
        self.index = np.arange(len(self.data))
 
    def __getitem__(self, i):
        return np.power(np.abs(stft(self.data[i], n_fft=nfft, hop_length=hop, 
                                            win_length=wlen, window=win)), 2)
    
    def __len__(self):
        return len(self.data)

class SpectrogramFrames(Dataset):
    def __init__(self, data):
        self.data = data
        self.index = np.arange(len(self.data))

    def __getitem__(self, i):
        return self.data[:,i]

    def __len__(self):
        return len(self.data[0,:])


class SpectrogramLabeledFrames(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.index = np.arange(len(self.data))

    def __getitem__(self, i):
        return self.data[:,i], self.labels[:,i]

    def __len__(self):
        return len(self.data[0,:])


class HDF5SpectrogramLabeledFrames(Dataset):
    def __init__(self, output_h5_dir, dataset_type, rdcc_nbytes, rdcc_nslots):
        # Do not load hdf5 in __init__ if num_workers > 0
        self.output_h5_dir = output_h5_dir
        self.dataset_type = dataset_type
        self.rdcc_nbytes = rdcc_nbytes
        self.rdcc_nslots = rdcc_nslots
        with h5.File(self.output_h5_dir, 'r') as file:
            self.dataset_len = file["X_" + dataset_type].shape[1]

    def open_hdf5(self):
        #We are using 400Mb of chunk_cache_mem here ("rdcc_nbytes" and "rdcc_nslots")
        self.f = h5.File(self.output_h5_dir, 'r', rdcc_nbytes=self.rdcc_nbytes, rdcc_nslots=self.rdcc_nslots)
        
        # Faster to open datasets once, rather than at every call of __getitem__
        self.data = self.f['X_' + self.dataset_type]
        self.labels = self.f['Y_' + self.dataset_type]

    def __getitem__(self, i):
        # Open hdf5 here if num_workers > 0
        if not hasattr(self, 'f'):
            self.open_hdf5()
        return self.data[:,i], self.labels[:,i]

    def __len__(self):
        return self.dataset_len

    def __del__(self): 
        if hasattr(self, 'f'):
            self.f.close()

#TODO: include STFT analysis in dataloader, in order to avoid preprocess in advance
# (ask Julius or JM)
class SpectrogramFramesRawAudio(Dataset):
    #TODO: don't forget to close the file
    #TODO: read the attributes of the h5py (key, value)
    #TODO: specify chunk size 
    #TODO: compression
    def __init__(self, data_dir, dataset_name):
            
        self.data_files = glob(os.path.join(data_dir, dataset_name, '**/*.wav',recursive=True))
        self.data_files = sorted(self.data_files)

    def __getindex__(self, idx):
        return load_file(self.data_files[idx])

    def __len__(self):
        return len(self.data_files)