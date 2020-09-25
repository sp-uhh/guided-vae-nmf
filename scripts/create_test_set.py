import numpy as np
import soundfile as sf
import os

from python.dataset.csr1_wjs0_dataset import speech_list, noise_list, preprocess_noise, noise_segment, write_dataset, read_dataset
# from python.processing.stft import stft
# from python.processing.target import clean_speech_IBM

from python.utils import open_file


# Parameters
## Dataset
dataset_type = 'test'

input_speech_dir = 'data/subset/raw/'

input_noise_dir = 'data/complete/raw/qutnoise_databases/' # change the name of the subfolder in your computer
output_noise_dir = 'data/complete/processed/qutnoise_databases/' # change the name of the subfolder in your computer

output_wav_dir = 'data/subset/processed/'
output_pickle_dir = 'data/subset/pickle/'

## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'

def main():

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                             dataset_type=dataset_type)
    
    # Create SNR list
    np.random.seed(0)
    
    noise_types = ['cafe', 'home', 'street', 'car']
    noise_index = np.random.randint(4, size=len(file_paths))
    
    snrs = [-5.0, 0.0, 5.0]
    snrs_index = np.random.randint(3, size=len(file_paths))

    # Create noise audios
    noise_paths = noise_list(input_noise_dir=input_noise_dir,
                             dataset_type=dataset_type)
    noise_audios = {}

    for key, noise_path in noise_paths.items():
        output_noise_path = output_noise_dir + dataset_type + '/' + key + '.wav'
        
        #if noise already preprocessed, read files directly
        if os.path.exists(output_noise_path):
            
            noise_audio, fs_noise = sf.read(output_noise_path)
            
            if fs != fs_noise:
                raise ValueError('Unexpected sampling rate')
        else:
            noise_audio, fs_noise = sf.read(input_noise_dir + noise_path)
            
            # Preprocess noise   
            noise_audio = preprocess_noise(noise_audio, key, fs_noise, fs)

            if not os.path.exists(os.path.dirname(output_noise_path)):
                os.makedirs(os.path.dirname(output_noise_path))
            sf.write(output_noise_path, noise_audio, fs)
        
        noise_audios[key] = noise_audio

    # Create mixture
    speeches = []
    mixtures = []
    noises = []
    all_snr_dB = []

    for i, file_path in enumerate(file_paths):

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
        
        # Append processed audios
        speeches.append(speech) 
        noises.append(noise)
        #mixtures.append((speech+noise)/np.max(speech+noise))
        mixtures.append(speech+noise)

        # Save .wav files
        output_path = output_wav_dir + file_path
        output_path = os.path.splitext(output_path)[0]

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        
        sf.write(output_path + '_s.wav', speech, fs)
        sf.write(output_path + '_n.wav', noise, fs)
        sf.write(output_path + '_x.wav', speech+noise, fs)

        # TODO: save SNR, level_s, level_n in a figure
    
    #pickle.dump(audio_files, open(output_data_dir + '../data/pickle/clean_speech.p', 'wb'), protocol=4)
    write_dataset(speeches, output_pickle_dir, dataset_type, 'speech-505')
    write_dataset(noises, output_pickle_dir, dataset_type, 'noise-505')
    write_dataset(mixtures, output_pickle_dir, dataset_type, 'mixture-505')

    # TODO: save SNR, level_s, level_n in 1 big csv
    write_dataset(all_snr_dB, output_wav_dir, dataset_type, 'snr_db')
    # TODO: save histogram of SNR

    open_file(output_pickle_dir)

if __name__ == '__main__':
    main()