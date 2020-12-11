import sys
sys.path.append('.')

import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
import h5py as h5

from python.dataset.csr1_wjs0_dataset import speech_list, write_dataset
from python.dataset.demand_database import noise_list, preprocess_noise, noise_segment
from python.processing.stft import stft
from python.processing.target import clean_speech_IBM, clean_speech_VAD, ideal_wiener_mask
from python.utils import open_file


# Parameters
## Dataset
dataset_types = ['train', 'validation']

# dataset_size = 'subset'
dataset_size = 'complete'

# data_dir = 'h5'
data_dir = 'export'

speech_dataset_name = 'CSR-1-WSJ-0'
noise_dataset_name = 'Demand'

labels = 'noisy_labels'
# labels = 'noisy_vad_labels'
# labels = 'noisy_wiener_labels'

input_speech_dir = os.path.join('data', dataset_size, 'raw/')

input_noise_dir = os.path.join('data/complete/raw/', noise_dataset_name + '/') # change the name of the subfolder in your computer
output_noise_dir = os.path.join('data/complete/processed/', noise_dataset_name + '/') # change the name of the subfolder in your computer

output_wav_dir = os.path.join('data', dataset_size, 'processed/')
output_dataset_file = os.path.join('data', dataset_size, data_dir, speech_dataset_name + '_' + labels + '.h5')

## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'

## Ideal binary mask
quantile_fraction = 0.999
quantile_weight = 0.999

# Ideal wiener mask
eps = 1e-8

# HDF5 parameters
rdcc_nbytes = 1024**2*400 # The number of bytes to use for the chunk cache
                          # Default is 1 Mb
                          # Here we are using 400Mb of chunk_cache_mem here
rdcc_nslots = 1e5 # The number of slots in the cache's hash table
                  # Default is 521
                  # ideally 100 x number of chunks that can be fit in rdcc_nbytes
                  # (see https://docs.h5py.org/en/stable/high/file.html?highlight=rdcc#chunk-cache)
                  # for compression 'zlf' --> 1e4 - 1e7
                  # for compression 32001 --> 1e4
X_shape = (513, 0)
X_maxshape = (513, None)
X_chunks = (513, 1)

if labels in ['noisy_labels', 'noisy_wiener_labels']:
    Y_shape = (513, 0)
    Y_maxshape = (513, None)
    Y_chunks = (513, 1)
    
if labels == 'noisy_vad_labels':
    Y_shape = (1, 0)
    Y_maxshape = (1, None)    
    Y_chunks = (1,1)    
    
compression = 'lzf'
shuffle = False

def process_noise():

    for dataset_type in dataset_types:

        if dataset_type == 'train':
            noise_types = ['domestic', 'nature', 'office', 'transportation']
        if dataset_type == 'validation':
            noise_types = ['nature', 'office', 'public', 'transportation']

        # Create noise audios
        noise_paths = noise_list(input_noise_dir=input_noise_dir,
                                dataset_type=dataset_type)
        noise_audios = {}

        for noise_type, samples in noise_paths.items():

            if dataset_type == 'train':
                output_noise_path = output_noise_dir + 'si_tr_s' + '/' + noise_type + '.wav'
            if dataset_type == 'validation':
                output_noise_path = output_noise_dir + 'si_dt_05' + '/' + noise_type + '.wav'

            #if noise already preprocessed, read files directly
            if os.path.exists(output_noise_path):
                
                noise_audio, fs_noise = sf.read(output_noise_path)
                
                if fs != fs_noise:
                    raise ValueError('Unexpected sampling rate. Did you preprocess the 16kHz version of the DEMAND database?')
            
            # else preprocess the noise audio files, i.e. concatenate all audio samples in 1 big audio (for each noise type)
            else:
                noise_audio = []

                for sample_id, noise_path in samples.items():
                
                    noise_audio_sample, fs_noise = sf.read(input_noise_dir + noise_path)

                    noise_audio.append(noise_audio_sample)
                
                noise_audio = np.concatenate(noise_audio,axis=0)

                # Preprocess noise   
                noise_audio = preprocess_noise(noise_audio, fs_noise, fs)
                
                # Save the big file
                if not os.path.exists(os.path.dirname(output_noise_path)):
                    os.makedirs(os.path.dirname(output_noise_path))
                sf.write(output_noise_path, noise_audio, fs)

def main():

    if not os.path.exists(os.path.dirname(output_dataset_file)):
        os.makedirs(os.path.dirname(output_dataset_file))

    with h5.File(output_dataset_file, 'a', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots) as f:    
            
        # STFT attributes
        f.attrs['fs'] = fs
        f.attrs['wlen_sec'] = wlen_sec
        f.attrs['hop_percent'] = hop_percent
        f.attrs['win'] = win
        f.attrs['dtype'] = dtype

        # label attributes
        f.attrs['quantile_fraction'] = quantile_fraction
        f.attrs['quantile_weight'] = quantile_weight

        # HDF5 attributes
        f.attrs['X_chunks'] = X_chunks
        f.attrs['Y_chunks'] = Y_chunks
        f.attrs['compression'] = compression

        for dataset_type in dataset_types:

            # Create file list
            file_paths = speech_list(input_speech_dir=input_speech_dir,
                                    dataset_type=dataset_type)
            
            if dataset_type == 'train':
                noise_types = ['domestic', 'nature', 'office', 'transportation']
            if dataset_type == 'validation':
                noise_types = ['nature', 'office', 'public', 'transportation']
                        
            # Create SNR list
            np.random.seed(0)
            noise_index = np.random.randint(len(noise_types), size=len(file_paths))
            snrs = [-5, -2.5, 0, 2.5, 5.0]
            snrs_index = np.random.randint(len(snrs), size=len(file_paths))
            
            # Create noise audios
            noise_paths = noise_list(input_noise_dir=input_noise_dir,
                                    dataset_type=dataset_type)
            noise_audios = {}

            # Load the noise files
            for noise_type, samples in noise_paths.items():

                if dataset_type == 'train':
                    output_noise_path = output_noise_dir + 'si_tr_s' + '/' + noise_type + '.wav'
                if dataset_type == 'validation':
                    output_noise_path = output_noise_dir + 'si_dt_05' + '/' + noise_type + '.wav'

                #if noise already preprocessed, read files directly
                if os.path.exists(output_noise_path):
                    
                    noise_audio, fs_noise = sf.read(output_noise_path)
                    
                    if fs != fs_noise:
                        raise ValueError('Unexpected sampling rate. Did you preprocess the 16kHz version of the DEMAND database?')

                    noise_audios[noise_type] = noise_audio
            
            # Init list of SNR
            all_snr_dB = []

            # Delete datasets if already exists
            if 'X_' + dataset_type in f:
                del f['X_' + dataset_type]
                del f['Y_' + dataset_type]
            
            # Exact shape of dataset is unknown in advance unfortunately
            # Faster writing if you know the shape in advance
            # Size of chunks corresponds to one spectrogram frame
            f.create_dataset('X_' + dataset_type, shape=X_shape, dtype='float32', maxshape=X_maxshape, chunks=X_chunks, compression=compression, shuffle=shuffle)
            f.create_dataset('Y_' + dataset_type, shape=Y_shape, dtype='float32', maxshape=Y_maxshape, chunks=Y_chunks, compression=compression, shuffle=shuffle)

            # Store dataset in variables for faster I/O
            fx = f['X_' + dataset_type]
            fy = f['Y_' + dataset_type]

            # Compute mean, std
            if dataset_type == 'train':
                # VAR = E[X**2] - E[X]**2
                channels_sum, channels_squared_sum = 0., 0.

            # Loop over the speech files
            for i, file_path in tqdm(enumerate(file_paths)):

                speech, fs_speech = sf.read(input_speech_dir + file_path, samplerate=None)

                # Cut burst at begining of file
                speech = speech[int(0.1*fs):]

                # Normalize audio
                speech = speech/(np.max(np.abs(speech)))

                if fs != fs_speech:
                    raise ValueError('Unexpected sampling rate')

                # Select noise_type            
                noise_type = noise_types[noise_index[i]]
        
                # Extract noise segment
                noise = noise_segment(noise_audios, noise_type, speech)

                # Select SNR
                snr_dB = snrs[snrs_index[i]]
                all_snr_dB.append(snr_dB)

                # Compute noise gain
                speech_power = np.sum(np.power(speech, 2))
                noise_power = np.sum(np.power(noise, 2))
                noise_power_target = speech_power*np.power(10,-snr_dB/10)
                k = noise_power_target / noise_power
                noise = noise * np.sqrt(k)

                mixture = speech + noise

                # # Normalize by max of speech, noise, speech+noise
                # norm = np.max(abs(np.concatenate([speech, noise, speech+noise])))
                # mixture = (speech+noise) / norm
                # speech /= norm
                # noise /= norm

                if dataset_size == 'subset':
                    # Save .wav files, just to check if it working
                    output_path = output_wav_dir + file_path
                    output_path = os.path.splitext(output_path)[0]

                    if not os.path.exists(os.path.dirname(output_path)):
                        os.makedirs(os.path.dirname(output_path))
                    
                    sf.write(output_path + '_s.wav', speech, fs)
                    sf.write(output_path + '_n.wav', noise, fs)
                    sf.write(output_path + '_x.wav', mixture, fs)

                # TF reprepsentation
                mixture_tf = stft(mixture, fs=fs, wlen_sec=wlen_sec, win=win, 
                    hop_percent=hop_percent, dtype=dtype)
                
                noisy_spectrogram = np.power(abs(mixture_tf), 2)    

                # TF reprepsentation
                speech_tf = stft(speech, fs=fs, wlen_sec=wlen_sec, win=win, 
                    hop_percent=hop_percent, dtype=dtype)
                
                if labels == 'noisy_wiener_labels':
                    # TF reprepsentation
                    noise_tf = stft(noise, fs=fs, wlen_sec=wlen_sec, win=win, 
                        hop_percent=hop_percent, dtype=dtype)
                    
                    # wiener mask
                    speech_wiener_mask = ideal_wiener_mask(speech_tf,
                                             noise_tf,
                                             eps)
                    label = speech_wiener_mask

                if labels == 'noisy_labels':
                    # binary mask
                    speech_ibm = clean_speech_IBM(speech_tf,
                                            quantile_fraction=quantile_fraction,
                                            quantile_weight=quantile_weight)
                    label = speech_ibm
                    
                if labels == 'noisy_vad_labels':
                    # binary mask
                    speech_vad = clean_speech_VAD(speech_tf,
                                            quantile_fraction=quantile_fraction,
                                            quantile_weight=quantile_weight)
                    label = speech_vad

                # Compute mean, std
                if dataset_type == 'train':
                    # VAR = E[X**2] - E[X]**2
                    channels_sum += np.sum(noisy_spectrogram, axis=-1)
                    channels_squared_sum += np.sum(noisy_spectrogram**2, axis=-1)
        
                # Store spectrogram in dataset
                fx.resize((fx.shape[1] + noisy_spectrogram.shape[1]), axis = 1)
                fx[:,-noisy_spectrogram.shape[1]:] = noisy_spectrogram

                # Store spectrogram in label
                fy.resize((fy.shape[1] + label.shape[1]), axis = 1)
                fy[:,-label.shape[1]:] = label
            
            # Compute and save mean, std
            if dataset_type == 'train':
                print('Compute mean and std')
                #NB: compute the empirical std (!= regular std)
                n_samples = fx.shape[1]
                mean = channels_sum / n_samples
                std = np.sqrt((1/(n_samples*(n_samples - 1)))*(channels_squared_sum - n_samples * mean**2))
                
                # Delete datasets if already exists
                if 'X_' + dataset_type + '_mean' in f:
                    del f['X_' + dataset_type + '_mean']
                    del f['X_' + dataset_type + '_std']

                f.create_dataset('X_' + dataset_type + '_mean', shape=X_chunks, dtype='float32', maxshape=X_chunks, chunks=None, compression=compression, shuffle=shuffle)
                f.create_dataset('X_' + dataset_type + '_std', shape=X_chunks, dtype='float32', maxshape=X_chunks, chunks=None, compression=compression, shuffle=shuffle)
                
                f['X_' + dataset_type + '_mean'][:] = mean[..., None] # Add axis to fit chunks shape
                f['X_' + dataset_type + '_std'][:] = std[..., None] # Add axis to fit chunks shape
                print('Mean and std saved in HDF5.')

        # TODO: save SNR, level_s, level_n in 1 big csv
        write_dataset(all_snr_dB, output_wav_dir, dataset_type, 'snr_db')
        # TODO: save histogram of SNR

        # open_file(output_pickle_dir)

if __name__ == '__main__':
    #process_noise()
    main()