import numpy as np
import librosa
#from torchaudio.transforms import Spectrogram
#import librosa.display
import math


"""
TF representation
"""


#TODO: write the spectrogram transform in pytorch (torchaudio)


def stft(x,
         fs=16e3,
         wlen_sec=50e-3,
         window='hann',
         hop_percent=0.25,
         center=True,
         pad_mode='reflect',
         dtype='complex64',
         *args,
         **kwargs):
    """
    Arguments
        x: input (as time series)
            # librosa 0.6.1: x must be float
        fs: frequence sampling (in Hz)
        framesz: framesize (in seconds)
        hop: 1 - overlap (in [0,1])

    return
        f, t, Sxx (default: complex64 i.e. float32 for amplitude and phase)
        WARNING: null frequency is included in the spectrogram bins
    """
    nfft = int(wlen_sec * fs) # STFT window length in samples
    hopsamp = int(hop_percent * nfft) # hop size in samples

    # # Convert x to float (librosa 0.6.1)
    # if not np.issubdtype(x.dtype, np.floating):
    #     x = x.astype('float64')

    # Sometimes stft / istft shortens the ouput due to window size
    # so you need to pad the end with hopsamp zeros
    utt_len = len(x) / fs
    if math.ceil(utt_len / wlen_sec / hop_percent) != int(utt_len / wlen_sec / hop_percent):
        x_ = np.pad(x, (0,hopsamp), mode='constant')
    else:
        x_ = x

    Sxx = librosa.core.stft(y=x_,
                            n_fft=nfft,
                            hop_length=hopsamp,
                            win_length=None,
                            window=window,
                            center=center,
                            pad_mode=pad_mode,
                            dtype=dtype)
    return Sxx.T


def istft(Sxx,
          fs=16000,
          framesz=32 / 1000.,
          window='hann',
          hop=0.25,
          center=True,
          dtype='float32',
          max_len=None):
    """
    Inverse STFT
    Sxx: input (as spectrogram)
    fs: frequence sampling (in Hz)
    framesz: framesize (in seconds)
    hop: 1 - overlap (in [0,1])

    return
    t, x (default: float64)
    """

    Thop = hop * framesz
    # Tnoverlap = (1. - hop) * framesz

    hopsamp = int(Thop * fs)
    framesamp = int(framesz * fs)
    # noverlapsamp = int(Tnoverlap * fs)

    x = librosa.core.istft(stft_matrix=Sxx.T,
                           hop_length=hopsamp,
                           win_length=framesamp,
                           window='hann',
                           center=True,
                           dtype=dtype,
                           length=None)
    # t, x = signal.istft(Zxx=Sxx,
    #                    fs=fs,
    #                    window='hann',
    #                    nperseg=framesamp,
    #                    noverlap=noverlapsamp,
    #                    nfft=nfft*2,
    #                    input_onesided=input_onesided,
    #                    boundary=boundary,
    #                    time_axis=0,
    #                    freq_axis=1)
    if max_len:
        x = x[:int(max_len*fs)]
    return x