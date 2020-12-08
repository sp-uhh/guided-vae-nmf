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


class SpectrogramLabeledFramesH5(Dataset):
    #TODO: read the attributes of the h5py (key, value)
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

def hdf5_loader_generator(output_h5_dir, dataset_type, rdcc_nbytes, rdcc_nslots, batch_size, shuffle):
    """Given an h5 path to a file that holds the arrays, returns a generator
    that can get certain data at a time."""
    f = h5.File(output_h5_dir, 'r', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots)
    n_samples = f["X_" + dataset_type].shape[1]

    # Faster to open datasets once, rather than at every call of __getitem__
    data = f['X_' + dataset_type]
    labels = f['Y_' + dataset_type]
    
    batch_index = 0
    total_batches_seen = 0
    
    while 1:
        if batch_index == 0:
            # For MLP
            index_array = np.arange(n_samples)

            if shuffle:
                np.random.shuffle(index_array)

        current_index = (batch_index * batch_size) % n_samples
        if n_samples > current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        # If last batch of epoch, reduce batch_size
        else:
            current_batch_size = n_samples - current_index
            batch_index = 0
        total_batches_seen += 1

        # sort indices before putting in HDF5 dataset
        # yield (index_array[current_index: current_index + current_batch_size],
        #        current_index, current_batch_size)
        index_values =  index_array[current_index: current_index + current_batch_size]
        index_values = sorted(index_values)

        X = data[:,index_values]
        Y = labels[:,index_values]
            
        yield (torch.from_numpy(X).T, torch.from_numpy(Y).T)

def hdf5_data_iterator(output_h5_dir, dataset_type, rdcc_nbytes, rdcc_nslots, batch_size, shuffle):
    return iter(hdf5_loader_generator(output_h5_dir, dataset_type, rdcc_nbytes, rdcc_nslots, batch_size, shuffle))

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