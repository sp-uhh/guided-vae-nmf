"""
    Implements conventional noise PSD estimators. Currently the following
    methods are implemented:
        * SPP-based noise PSD estimator
"""

import numpy as np

# default values for SPP-based noise PSD estimator
SPP_FIX_SMOOTH = 0.8
SPP_PROB_SMOOTH = 0.9
SPP_PRIOR = 0.5
SPP_SNR_OPT_DB = 15
SPP_NUM_FRAMES_INIT = 10
#SPP_NUM_FRAMES_INIT = 0

class SPPNoiseEstimator:

    """
        Implements the speech presence probability (SPP) based noise PSD
        estimator proposed in [1], [2].

        NOTE: This algorithm is designed for 32 ms with 16 ms shift. If you
        want to use other parameters for your STFT, you need to adjust the
        parameters accordingly. Even after adjusting these parameters, a lower
        performance can be expected.

        [1] T. Gerkmann and R. C. Hendriks, “Noise power estimation based on
        the probability of speech presence,” in IEEE Workshop on Applications
        of Signal Processing to Audio and Acoustics (WASPAA), New Paltz, NY,
        USA, 2011, pp. 145–148.
        [2] T. Gerkmann and R. C. Hendriks, “Unbiased MMSE-based noise power
        estimation with low complexity and low tracking delay,” IEEE
        Transactions on Audio, Speech, and Language Processing, vol. 20, no. 4,
        pp. 1383–1393, May 2012.
    """

    def __init__(self, frame_length,
                 fixed_smooth=SPP_FIX_SMOOTH,
                 prob_smooth=SPP_PROB_SMOOTH,
                 prior=SPP_PRIOR,
                 snr_opt_db=SPP_SNR_OPT_DB,
                 num_frames_init=SPP_NUM_FRAMES_INIT):
        """ Generates a SPP noise PSD estimator object. For all optional
        parameters, the default values in [1] are used.

        :frame_length: frame length (number of samples)
        :fixed_smooth: (optional) fixed smoothing constant (see line 6 of
                       Algorithm 1 in [1]). Default value is 0.8.

        :prob_smooth: (optional) fixed smoothing constant for smoothing the
                      SPP values (see line 3 of Algorithm 1 in [1]). Default
                      value is 0.9.
        :prior: (optional) prior probability for speech (P(H1) in [1]). Default
                value is 0.5.
        :snr_opt_db: (optional) fixed speech SNR (dB) (xi_opt in [1]). Default
                value is 15 dB.
        :num_frames_init: (optional) number of frames to be used for
                          initialization. Default value is 10.
        """
        self._frame_length = frame_length

        # fixed smoothing constant for smoothing the noise periodogram
        self._fixed_smooth = fixed_smooth

        # fixed smoothing constant for smoothing the SPP
        self._prob_smooth = prob_smooth

        # prior probability for speech presence (P(H1) in [1])
        self._prior = prior

        # fixed SNR (\xi_opt in [1])
        self._snr_opt_lin = 10.**(snr_opt_db/10.)

        # number of frames used for initialization
        self._num_frames_init = num_frames_init

        # internal states
        self._v_old_psd = np.zeros(frame_length // 2 + 1)
        #self._v_old_psd = np.ones(frame_length // 2 + 1)
        self._v_smooth_prob = np.zeros(frame_length // 2 + 1)
        self._inv_glr_factor = (1 - prior)/prior*(1. + self._snr_opt_lin)
        self._inv_glr_exp_factor = self._snr_opt_lin/(1. + self._snr_opt_lin)
        self._num_frames_processed = 0

    def update(self, v_noisy_per, v_spp_in=None):
        """ Estimates noise PSD from the noisy input periodogram.

        The computations correspond to Algorithm 1 in [1]. All computations are
        performed for a single frame. Therefore, all input parameters have to
        be vectors with DFT length // 2 + 1 elements.

        :noisy_per: noisy periodogram (|Y|^2) (numpy array)
        :returns: noise PSD (sigma^2_n) (numpy array)

        """
        if v_spp_in is None:
            if self._num_frames_processed < self._num_frames_init:
                # average first frames to obtain first noise PSD estimate
                v_noise_psd = self._v_old_psd + v_noisy_per / self._num_frames_init

                self._v_old_psd = v_noise_psd

                # increment frame counter
                self._num_frames_processed += 1

                v_spp = np.zeros_like(self._v_old_psd) # SPP considered 0 at the beginning

                return v_noisy_per, v_spp
            else:
                # compute inverse GLR
                v_inv_glr = self._inv_glr_factor * \
                    np.exp(-v_noisy_per / (self._v_old_psd + 1e-8) * self._inv_glr_exp_factor)

                # compute SPP (corresponds to line 2 in Algorithm 1, [1])
                v_spp = 1. / (1. + v_inv_glr)

                # stuck protection (corresponds to line 3 and 4 in Algorithm 1,
                # [1])
                self._v_smooth_prob = (1 - self._prob_smooth) * v_spp + \
                    self._prob_smooth * self._v_smooth_prob
                v_mask = self._v_smooth_prob > 0.99
                v_spp[v_mask] = np.minimum(v_spp[v_mask], 0.99)

                # estimate noise periodogram (corresponds to line 5 in Algorithm
                # 1, [1])
                v_noise_per = (1. - v_spp) * v_noisy_per + \
                    v_spp * self._v_old_psd
                # corresponds to line 6 in Algorithm 1, [1]
                v_noise_psd = (1. - self._fixed_smooth) * v_noise_per + \
                    self._fixed_smooth * self._v_old_psd

            # update old noise PSD estimate
            self._v_old_psd = v_noise_psd

            return v_noise_psd, v_spp
        else:
            # estimate noise periodogram (corresponds to line 5 in Algorithm
            # 1, [1])
            #TODO: better use formula of Wang with alpha combined with mask
            v_noise_per = (1. - v_spp_in) * v_noisy_per + \
                v_spp_in * self._v_old_psd
            # corresponds to line 6 in Algorithm 1, [1]
            v_noise_psd = (1. - self._fixed_smooth) * v_noise_per + \
                self._fixed_smooth * self._v_old_psd
        return v_noise_psd

    def reset(self):
        """ Resets the internal states of the algorithm.

        """
        self._v_old_psd = np.zeros(self._frame_length // 2 + 1)
        self._v_smooth_prob = np.zeros(self._frame_length // 2 + 1)
        self._num_frames_processed = 0

    def from_stft(self, mat_per):
        """ Estimate the noise PSD from a matrix, i.e., a spectrogram, of noisy
        periodograms.

        :mat_per: noisy periodogram (|Y|^2, frames x coefficients)
        :returns: noise PSD estimate (\sigma^2_n, frames x coefficients)

        """
        # pre-allocate noise PSD
        mat_psd = np.zeros(mat_per.shape)

        for frame, per in enumerate(mat_per):
            mat_psd[frame] = self.update(per)

        self.reset()

        return mat_psd


def timo_mask_estimation(spectrogram):
    """
    Run SPPNoiseEstimator on the whole power spectrogram

    Args:
        spectrogram ([type]): power spectrogram of noisy speech (i.e. |Y|^2)
    """

    freq_bins, frames = spectrogram.shape
    frame_length = (freq_bins - 1) * 2 # STFT window size

    spp_estimator = SPPNoiseEstimator(frame_length=frame_length)

    mask = np.zeros_like(spectrogram)

    for i, frame in enumerate(spectrogram.T):
        v_noise_psd, v_spp = spp_estimator.update(frame)
        mask[:,i] = v_spp
    
    return mask

def timo_vad_estimation(spectrogram):
    """
    Run SPPNoiseEstimator on the whole power spectrogram

    Args:
        spectrogram ([type]): power spectrogram of noisy speech (i.e. |Y|^2)
    """
    
    spectrogram_sum = spectrogram.sum(axis=0)
    frame_length = 0 # sum all freq_bins

    spp_estimator = SPPNoiseEstimator(frame_length=frame_length)

    vad = np.zeros_like(spectrogram_sum)

    for i, frame in enumerate(spectrogram_sum):
        v_noise_psd, v_spp = spp_estimator.update(frame)
        vad[i] = v_spp
    
    return vad

def timo_noise_estimation(spectrogram, mask):
    """
    Run SPPNoiseEstimator on the whole power spectrogram

    Args:
        spectrogram ([type]): power spectrogram of noisy speech (i.e. |Y|^2)
    """

    freq_bins, frames = spectrogram.shape
    frame_length = (freq_bins - 1) * 2 # STFT window size

    spp_estimator = SPPNoiseEstimator(frame_length=frame_length)

    noise_psd = np.zeros_like(spectrogram)

    for i, (frame, v_ssp_in) in enumerate(zip(spectrogram.T, mask.T)):
        v_noise_psd = spp_estimator.update(frame, v_ssp_in)
        noise_psd[:,i] = v_noise_psd
    
    return noise_psd