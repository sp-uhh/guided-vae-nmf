import numpy as np
import librosa.display
import matplotlib
matplotlib.use('pdf') # disable interactive backend (required when remote ssh)
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd

def display_waveplot(x,
                     fs=16e3,
                     ymax=1.,
                     ymin=-1.,
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
    plt.ylim(ymin=ymin, ymax=ymax)

    return img

def display_spectrogram(complex_spec,
                        convert_to_db=False,
                        fs=16e3,
                        vmin=-60,
                        vmax=10,
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

    # Trick to plot VAD
    if amplitude_spec.shape[0] == 1:
        freq_bins = 513
        amplitude_spec = np.repeat(amplitude_spec, freq_bins, axis=0)

    # librosa.display.specshow params
    freq_bins, frames = amplitude_spec.shape

    
    sr=int(fs/1e3) # to make yticks concise
    nfft = int(wlen_sec * fs) # STFT window length in samples
    hop_length = int(hop_percent * nfft) # hop size in samples
    hop_sec = hop_length / fs # hop size in seconds
    time_sec = frames * hop_sec # time length of the signal in seconds

    # plot params
    plt.rcParams.update({'font.size': fontsize})

    img = librosa.display.specshow(amplitude_spec,
                            x_axis='time',
                            y_axis='linear',
                            vmin=vmin,
                            vmax=vmax,
                            sr=sr,
                            hop_length=hop_length,
                            x_coords=np.arange(0, time_sec + hop_sec,(time_sec + hop_sec) / frames),
                            cmap=cmap)

    plt.ylabel('Frequency (kHz)', fontsize=fontsize+10) #, fontweight="bold")
    plt.xlabel('Time (s)', fontsize=fontsize+10) #, fontweight="bold")
    plt.xticks(np.arange(0, time_sec + hop_sec, step=xticks_sec), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    return img

def display_power_spectro(psd,
                fs=16e3,
                vmin=-60,
                vmax=10,
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

    # librosa.display.specshow params
    freq_bins, frames = psd.shape
    sr=int(fs/1e3) # to make yticks concise
    nfft = int(wlen_sec * fs) # STFT window length in samples
    hop_length = int(hop_percent * nfft) # hop size in samples
    hop_sec = hop_length / fs # hop size in seconds
    time_sec = frames * hop_sec # time length of the signal in seconds

    # plot params
    plt.rcParams.update({'font.size': fontsize})

    img = librosa.display.specshow(psd,
                            x_axis='time',
                            y_axis='linear',
                            vmin=vmin,
                            vmax=vmax,
                            sr=sr,
                            hop_length=hop_length,
                            x_coords=np.arange(0, time_sec + hop_sec,(time_sec + hop_sec) / frames),
                            cmap=cmap)

    plt.ylabel('Frequency (kHz)', fontsize=fontsize+10) #, fontweight="bold")
    plt.xlabel('Time (s)', fontsize=fontsize+10) #, fontweight="bold")
    plt.xticks(np.arange(0, time_sec + hop_sec, step=xticks_sec), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    return img

def display_wav_spectro_mask(x,
                             x_tf,
                             x_ibm,
                             fs=16e3,
                             vmin=-60,
                             vmax=10,
                             wlen_sec=50e-3,
                             hop_percent=0.5,
                             xticks_sec=1.0,
                             fontsize=50):
        
    # Plot waveplot + spectrogram + binary mask
    fig = plt.figure(figsize=(20,25))

    # create a 2 X 2 grid
    gs = grd.GridSpec(3, 2,
                    height_ratios=[5,10,10],
                    width_ratios=[10,0.5],
                    wspace=0.1,
                    hspace=0.3,
                    left=0.08)

    # line plot
    ax = plt.subplot(gs[0])
    display_waveplot(x=x, fs=fs, xticks_sec=xticks_sec, fontsize=fontsize)

    # image plot
    ax = plt.subplot(gs[2])
    display_spectrogram(x_tf, True, fs, vmin, vmax, wlen_sec, hop_percent, xticks_sec, 'magma', fontsize)

    # color bar in it's own axis
    colorAx = plt.subplot(gs[3])
    cbar = plt.colorbar(cax=colorAx, format='%+2.0f dB')

    # image plot
    ax = plt.subplot(gs[4])
    display_spectrogram(x_ibm, False, fs, 0, 1, wlen_sec, hop_percent, xticks_sec, 'Greys_r', fontsize)

    # color bar in it's own axis
    colorAx = plt.subplot(gs[5])
    plt.colorbar(cax=colorAx, format='%0.1f')

    return fig

def display_multiple_signals(signal_list,
                             fs=16e3,
                             vmin=-60,
                             vmax=10,
                             wlen_sec=50e-3,
                             hop_percent=0.5,
                             xticks_sec=1.0,
                             fontsize=50):
    """Generate waveplot + spectrogram + mask of multiple signals

    Args:
        signal_list ([type]): list of signals as [[waveform_1, tf_signal_1, mask_1], [waveform_2, ...], [...]]
        fs ([type], optional): [description]. Defaults to 16e3.
        vmin (int, optional): [description]. Defaults to -60.
        vmax (int, optional): [description]. Defaults to 10.
        wlen_sec ([type], optional): [description]. Defaults to 50e-3.
        hop_percent (float, optional): [description]. Defaults to 0.5.
        xticks_sec (float, optional): [description]. Defaults to 1.0.
        fontsize (int, optional): [description]. Defaults to 50.

    Returns:
        [type]: [description]
    """
    # Number of different signals
    nb_signals = len(signal_list)
    
    # Plot waveplot + spectrogram + binary mask
    fig = plt.figure(figsize=(25*nb_signals,25))
    
    # create a 2 X 2 grid
    gs = grd.GridSpec(3, 3*nb_signals,
                    height_ratios=[5,10,10],
                    width_ratios=[10,0.5,2.0]*nb_signals,
                    wspace=0.1,
                    hspace=0.3,
                    left=0.08)

    for i, [x, x_tf, x_ibm] in enumerate(signal_list):

        if not (x is None):
            # line plot
            ax = plt.subplot(gs[3*i])
            display_waveplot(x=x, fs=fs, xticks_sec=xticks_sec, fontsize=fontsize)

        if not (x_tf is None):
            # image plot
            #ax = plt.subplot(gs[(i+2)])
            ax = plt.subplot(gs[3*(i+3)])
            display_spectrogram(x_tf, True, fs, vmin, vmax, wlen_sec, hop_percent, xticks_sec, 'magma', fontsize)

            # color bar in it's own axis
            #colorAx = plt.subplot(gs[(i+2)*nb_signals + 1])
            colorAx = plt.subplot(gs[3*(i+3) + 1])
            cbar = plt.colorbar(cax=colorAx, format='%+2.0f dB')

        if not (x_ibm is None):
            # image plot
            #ax = plt.subplot(gs[(i+4)*nb_signals])
            ax = plt.subplot(gs[3*(i+6)])
            display_spectrogram(x_ibm, False, fs, 0, 1, wlen_sec, hop_percent, xticks_sec, 'Greys_r', fontsize)

            # color bar in it's own axis
            #colorAx = plt.subplot(gs[(i+4)*nb_signals+1])
            colorAx = plt.subplot(gs[3*(i+6)+1])
            plt.colorbar(cax=colorAx, format='%0.1f')
    
    #gs.tight_layout(fig)

    return fig

def display_multiple_spectro(signal_list,
                             fs=16e3,
                             vmin=-60,
                             vmax=10,
                             wlen_sec=50e-3,
                             hop_percent=0.5,
                             xticks_sec=1.0,
                             fontsize=50):
    """Generate waveplot + spectrogram + mask of multiple signals

    Args:
        signal_list ([type]): list of signals as [[waveform_1, tf_signal_1, mask_1], [waveform_2, ...], [...]]
        fs ([type], optional): [description]. Defaults to 16e3.
        vmin (int, optional): [description]. Defaults to -60.
        vmax (int, optional): [description]. Defaults to 10.
        wlen_sec ([type], optional): [description]. Defaults to 50e-3.
        hop_percent (float, optional): [description]. Defaults to 0.5.
        xticks_sec (float, optional): [description]. Defaults to 1.0.
        fontsize (int, optional): [description]. Defaults to 50.

    Returns:
        [type]: [description]
    """
    # Number of different signals
    nb_signals = len(signal_list)
    
    # Plot waveplot + spectrogram + binary mask
    fig = plt.figure(figsize=(25*nb_signals,16))
    
    # create a 2 X 2 grid
    gs = grd.GridSpec(2, 3*nb_signals,
                    height_ratios=[5,10],
                    width_ratios=[10,0.5,2.0]*nb_signals,
                    wspace=0.1,
                    hspace=0.3,
                    left=0.08)

    for i, [x, x_psd] in enumerate(signal_list):

        if not (x is None):
            # line plot
            ax = plt.subplot(gs[3*i])
            display_waveplot(x=x, fs=fs, xticks_sec=xticks_sec, fontsize=fontsize)

        # image plot
        #ax = plt.subplot(gs[(i+2)])
        ax = plt.subplot(gs[3*(i+3)])
        display_power_spectro(x_psd, fs, vmin, vmax, wlen_sec, hop_percent, xticks_sec, 'magma', fontsize)

        # color bar in it's own axis
        #colorAx = plt.subplot(gs[(i+2)*nb_signals + 1])
        colorAx = plt.subplot(gs[3*(i+3) + 1])
        cbar = plt.colorbar(cax=colorAx, format='%+2.0f dB')
    
    #gs.tight_layout(fig)

    return fig