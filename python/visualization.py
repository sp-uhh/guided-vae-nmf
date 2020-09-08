import numpy as np
import librosa.display
import matplotlib.pyplot as plt

def display_waveplot(x,
                     fs=16e3,
                     xticks_sec=1.0,
                     fontsize=50):
    """
    Display spectrogram using Librosa

    Args:
        complex_spec ([type]): [description]
        vmin (int, optional): [description]. Defaults to -60.
        vmax (int, optional): [description]. Defaults to 10.
        fs ([type], optional): [description]. Defaults to 16e3.
        wlen_sec ([type], optional): [description]. Defaults to 50e-3.
        hop_percent (float, optional): [description]. Defaults to 0.5.
        xticks_sec (float, optional): [description]. Defaults to 1.0.
    """

    # librosa.display.specshow params
    sr=int(fs/1e3) # to make yticks concise
    time_sec = len(x) / fs # time length of the signal in seconds

    # plot params
    plt.rcParams.update({'font.size': fontsize})

    img = librosa.display.waveplot(x, sr=fs)

    plt.ylabel('Amplitude', fontsize=fontsize+10) #, fontweight="bold")
    plt.xlabel('Time (s)', fontsize=fontsize+10) #, fontweight="bold")
    plt.xticks(np.arange(0, time_sec, step=xticks_sec), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    return img

def display_spectrogram(complex_spec,
                        convert_to_db=False,
                        vmin=-60,
                        vmax=10,
                        fs=16e3,
                        wlen_sec=50e-3,
                        hop_percent=0.5,
                        xticks_sec=1.0,
                        cmap='magma',
                        fontsize=50):
    """
    Display spectrogram using Librosa

    Args:
        complex_spec ([type]): [description]
        vmin (int, optional): [description]. Defaults to -60.
        vmax (int, optional): [description]. Defaults to 10.
        fs ([type], optional): [description]. Defaults to 16e3.
        wlen_sec ([type], optional): [description]. Defaults to 50e-3.
        hop_percent (float, optional): [description]. Defaults to 0.5.
        xticks_sec (float, optional): [description]. Defaults to 1.0.
    """

    # Transform to amplitude_db
    amplitude_spec = abs(complex_spec)
    if convert_to_db:
        amplitude_spec = librosa.core.amplitude_to_db(amplitude_spec)

    # librosa.display.specshow params
    sr=int(fs/1e3) # to make yticks concise
    nfft = int(wlen_sec * fs) # STFT window length in samples
    hop_length = int(hop_percent * nfft) # hop size in samples
    hop_sec = hop_length / fs # hop size in seconds
    time_sec = amplitude_spec.shape[0] * hop_sec # time length of the signal in seconds

    # plot params
    plt.rcParams.update({'font.size': fontsize})

    img = librosa.display.specshow(amplitude_spec.T,
                            x_axis='time',
                            y_axis='linear',
                            vmin=vmin,
                            vmax=vmax,
                            sr=sr,
                            hop_length=hop_length,
                            x_coords=np.arange(0, time_sec + hop_sec,(time_sec + hop_sec) / amplitude_spec.shape[0]),
                            cmap=cmap)

    plt.ylabel('Frequency (kHz)', fontsize=fontsize+10) #, fontweight="bold")
    plt.xlabel('Time (s)', fontsize=fontsize+10) #, fontweight="bold")
    plt.xticks(np.arange(0, time_sec + hop_sec, step=xticks_sec), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    return img