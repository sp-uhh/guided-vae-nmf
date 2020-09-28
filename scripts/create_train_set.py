#TODO: finish

# Parameters
## Dataset
input_data_dir = 'data/complete/raw/'
output_data_dir = 'data/complete/processed/'
fs = int(16e3) # Sampling rate

## STFT
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'

def main():


    # Create file list
    file_paths = speech_list(input_data_dir=input_data_dir,
                             dataset_type=dataset_type)

    spectrograms = []

    for path in file_paths:

        x, fs_x = sf.read(input_data_dir + path, samplerate=None)

        # Cut burst at begining of file
        x = x[int(0.1*fs):]

        # Normalize audio
        x = x/(np.max(np.abs(x)))

        if not os.path.exists(os.path.dirname(output_data_dir + path)):
            os.makedirs(os.path.dirname(output_data_dir + path))
        sf.write(output_data_dir + path, x, fs_x)
        
        if fs != fs_x:
            raise ValueError('Unexpected sampling rate')

        # TF reprepsentation
        x_tf = stft(x,
                    fs=fs,
                    wlen_sec=wlen_sec,
                    win=win,
                    hop_percent=hop_percent,
                    dtype=dtype)

        spectrograms.append(np.power(abs(x_tf), 2))

    spectrograms = np.concatenate(spectrograms, axis=1)
    #spectrograms = spectrograms[1]

    # write spectrograms
    write_dataset(spectrograms,
                  output_data_dir=output_data_dir,
                  dataset_type=dataset_type,
                  suffix='frames')
    
 
    ## Ideal binary mask
    quantile_fraction = 0.98
    quantile_weight = 0.999

    # Create file list
    file_paths = speech_list(input_data_dir=input_data_dir,
                             dataset_type=dataset_type)

    labels = []

    for path in file_paths:

        x, fs_x = sf.read(input_data_dir + path, samplerate=None)

        # Cut burst at begining of file
        x[:int(0.1*fs)] = x[int(0.1*fs):int(0.2*fs)]

        # Normalize audio
        x = x/(np.max(np.abs(x)))
        #x = x/(np.max(np.abs(x)) + 2)
        #x = x/np.linalg.norm(x)

        if fs != fs_x:
            raise ValueError('Unexpected sampling rate')

        # TF reprepsentation
        x_tf = stft(x,
                    fs=fs,
                    wlen_sec=wlen_sec,
                    win=win,
                    hop_percent=hop_percent,
                    dtype=dtype)

         # binary mask
        x_ibm = clean_speech_IBM(x_tf,
                                 quantile_fraction=quantile_fraction,
                                 quantile_weight=quantile_weight)

        labels.append(x_ibm)

    labels = np.concatenate(labels, axis=1)
    #labels = labels[1]

    # write spectrograms
    write_dataset(labels,
                  output_data_dir=output_data_dir,
                  dataset_type=dataset_type,
                  suffix='labels')
    
    # Read pickle
    pickle_labels = read_dataset(data_dir=output_data_dir,
                 dataset_type=dataset_type,
                 suffix='labels') 
    
    # Assert stored data is same as spectrograms
    assert_array_equal(labels, pickle_labels)

if __name__ == '__main__':
    main()