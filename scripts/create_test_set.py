import sys
sys.path.append('.')

import numpy as np
import soundfile as sf
import os
from tqdm import tqdm

from python.utils import open_file


# Parameters
## Dataset
speech_dataset_name = 'CSR-1-WSJ-0'
noise_dataset_name = 'qutnoise_databases'

if speech_dataset_name == 'CSR-1-WSJ-0':
    from python.dataset.csr1_wjs0_dataset import speech_list, write_dataset
if noise_dataset_name == 'qutnoise_databases':
    from python.dataset.qut_database import noise_list, preprocess_noise, noise_segment

dataset_type = 'test'

dataset_size = 'subset'
# dataset_size = 'complete'

output_data_folder = 'processed'

input_speech_dir = os.path.join('data', dataset_size, 'raw/')
input_noise_dir = os.path.join('data/complete/raw', noise_dataset_name + '/') # change the name of the subfolder in your computer
output_noise_dir = os.path.join('data/complete', output_data_folder, noise_dataset_name + '/') # change the name of the subfolder in your computer
output_wav_dir = os.path.join('data', dataset_size, output_data_folder + '/')

## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'

def process_noise():

    noise_types = ['cafe', 'home', 'street', 'car']

    # Create noise audios
    noise_paths = noise_list(input_noise_dir=input_noise_dir,
                             dataset_type=dataset_type)
    noise_audios = {}

    for noise_type, noise_path in noise_paths.items():
        if dataset_type == 'test':
            output_noise_path = output_noise_dir + 'si_et_05' + '/' + noise_type + '.wav'
        
        #if noise already preprocessed, read files directly
        if os.path.exists(output_noise_path):
            
            noise_audio, fs_noise = sf.read(output_noise_path)
            
            if fs != fs_noise:
                raise ValueError('Unexpected sampling rate')
        else:
            noise_audio, fs_noise = sf.read(input_noise_dir + noise_path)
            
            # Preprocess noise   
            noise_audio = preprocess_noise(noise_audio, noise_type, fs_noise, fs)

            if not os.path.exists(os.path.dirname(output_noise_path)):
                os.makedirs(os.path.dirname(output_noise_path))
            sf.write(output_noise_path, noise_audio, fs)
        
        noise_audios[noise_type] = noise_audio

def main():

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                             dataset_type=dataset_type)
    
    # Create SNR list
    np.random.seed(0)
    noise_types = ['cafe', 'home', 'street', 'car']
    #TODO: more noise index
    noise_index = np.random.randint(len(noise_types), size=len(file_paths)) 
    snrs = [-5.0, 0.0, 5.0]
    snrs_index = np.random.randint(len(snrs), size=len(file_paths))

    # Create noise audios
    noise_paths = noise_list(input_noise_dir=input_noise_dir,
                             dataset_type=dataset_type)
    noise_audios = {}

    # Load the noise files
    for noise_type, noise_path in noise_paths.items():
        if dataset_type == 'test':
            output_noise_path = output_noise_dir + 'si_et_05' + '/' + noise_type + '.wav'
        
        #if noise already preprocessed, read files directly
        if os.path.exists(output_noise_path):
            
            noise_audio, fs_noise = sf.read(output_noise_path)
            
            if fs != fs_noise:
                raise ValueError('Unexpected sampling rate')
            
            noise_audios[noise_type] = noise_audio


    # Save all SNRs
    all_snr_dB = []

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

        # Normalize by max of speech, noise, speech+noise
        norm = np.max(abs(np.concatenate([speech, noise, speech+noise])))
        mixture = (speech+noise) / norm
        speech /= norm
        noise /= norm

        # Save .wav files
        output_path = output_wav_dir + file_path
        output_path = os.path.splitext(output_path)[0]

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        
        sf.write(output_path + '_s.wav', speech, fs)
        sf.write(output_path + '_n.wav', noise, fs)
        sf.write(output_path + '_x.wav', mixture, fs)

        # TODO: save SNR, level_s, level_n in a figure
    
    # TODO: save SNR, level_s, level_n in 1 big csv
    write_dataset(all_snr_dB, output_wav_dir, dataset_type, 'snr_db')
    # TODO: save histogram of SNR

    #open_file(output_pickle_dir)

if __name__ == '__main__':
    #process_noise()
    main()