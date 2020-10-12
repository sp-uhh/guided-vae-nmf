import sys
sys.path.append('.')

import numpy as np
import soundfile as sf
import os
from tqdm import tqdm

from python.dataset.csr1_wjs0_dataset import speech_list, write_dataset
from python.dataset.demand_database import noise_list, preprocess_noise, noise_segment
from python.processing.stft import stft
from python.processing.target import clean_speech_IBM, clean_speech_VAD, ideal_wiener_mask
from python.utils import open_file


# Parameters
## Dataset
dataset_types = ['train', 'validation']

dataset_size = 'subset'
# dataset_size = 'complete'

input_speech_dir = os.path.join('data', dataset_size, 'raw/')

input_noise_dir = 'data/complete/raw/Demand/' # change the name of the subfolder in your computer
output_noise_dir = 'data/complete/processed/Demand/' # change the name of the subfolder in your computer

output_wav_dir = os.path.join('data', dataset_size, 'processed/')
output_pickle_dir = os.path.join('data', dataset_size, 'pickle/')

## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'

## Ideal binary mask
quantile_fraction = 0.98
quantile_weight = 0.999

# Ideal wiener mask
eps = 1e-8


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

    for dataset_type in dataset_types:

        # Create file list
        file_paths = speech_list(input_speech_dir=input_speech_dir,
                                dataset_type=dataset_type)
        
        # Create SNR list
        np.random.seed(0)
        
        if dataset_type == 'train':
            noise_types = ['domestic', 'nature', 'office', 'transportation']
        if dataset_type == 'validation':
            noise_types = ['nature', 'office', 'public', 'transportation']
        
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

        # Create mixture
        noisy_spectrograms = []
        noisy_labels = []
        all_snr_dB = []

        # Do 2 iterations to save separately noisy_spectro and noisy_labels (RAM issues)
        # for iteration in range(2):
        for iteration in range(1):

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
                # speech /= norm
                # noise /= norm
                # mixture = (speech+noise) / norm

                if dataset_size == 'subset':
                    # Save .wav files, just to check if it working
                    output_path = output_wav_dir + file_path
                    output_path = os.path.splitext(output_path)[0]

                    if not os.path.exists(os.path.dirname(output_path)):
                        os.makedirs(os.path.dirname(output_path))
                    
                    sf.write(output_path + '_s.wav', speech, fs)
                    sf.write(output_path + '_n.wav', noise, fs)
                    sf.write(output_path + '_x.wav', speech+noise, fs)

                if iteration == 0:                
                    # TF reprepsentation
                    speech_tf = stft(speech, fs=fs, wlen_sec=wlen_sec, win=win, 
                        hop_percent=hop_percent, dtype=dtype)
                    
                    # TF reprepsentation
                    noise_tf = stft(noise, fs=fs, wlen_sec=wlen_sec, win=win, 
                        hop_percent=hop_percent, dtype=dtype)
                    
                    # wiener mask
                    speech_wiener_mask = ideal_wiener_mask(speech_tf,
                                             noise_tf,
                                             eps)
                    noisy_labels.append(speech_wiener_mask)

                    # # binary mask
                    # speech_ibm = clean_speech_IBM(speech_tf,
                    #                         quantile_fraction=quantile_fraction,
                    #                         quantile_weight=quantile_weight)
                    # noisy_labels.append(speech_ibm)
                    
                    # # binary mask
                    # speech_vad = clean_speech_VAD(speech_tf,
                    #                         quantile_fraction=quantile_fraction,
                    #                         quantile_weight=quantile_weight)
                    # noisy_labels.append(speech_vad)

                # if iteration == 1:
                #     # TF reprepsentation
                #     mixture_tf = stft(mixture, fs=fs, wlen_sec=wlen_sec, win=win, 
                #         hop_percent=hop_percent, dtype=dtype)
                    
                                
                #     noisy_spectrograms.append(np.power(abs(mixture_tf), 2))

            if iteration == 0:
                noisy_labels = np.concatenate(noisy_labels, axis=1)

                # write spectrograms
                write_dataset(noisy_labels,
                            output_data_dir=output_pickle_dir,
                            dataset_type=dataset_type,
                            suffix='noisy_wiener_labels')

                # # write spectrograms
                # write_dataset(noisy_labels,
                #             output_data_dir=output_pickle_dir,
                #             dataset_type=dataset_type,
                #             suffix='noisy_labels')

                # # write spectrograms
                # write_dataset(noisy_labels,
                #             output_data_dir=output_pickle_dir,
                #             dataset_type=dataset_type,
                #             suffix='noisy_vad_labels')

                del noisy_labels

            # if iteration == 1:
            #     noisy_spectrograms = np.concatenate(noisy_spectrograms, axis=1)

            #     # write spectrograms
            #     write_dataset(noisy_spectrograms,
            #                 output_data_dir=output_pickle_dir,
            #                 dataset_type=dataset_type,
            #                 suffix='noisy_frames')
                
            #     del noisy_spectrograms


        # TODO: save SNR, level_s, level_n in 1 big csv
        write_dataset(all_snr_dB, output_wav_dir, dataset_type, 'snr_db')
        # TODO: save histogram of SNR

        # open_file(output_pickle_dir)

if __name__ == '__main__':
    #process_noise()
    main()