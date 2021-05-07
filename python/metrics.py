import numpy as np
import scipy.stats
import json

def mean_confidence_interval(data, confidence=0.95, round=3):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return np.round(m,3), np.round(h,3)

def si_sdr_components(s_hat, s, n):
    """
    Compute the components of s_hat as

    s_hat = alpha_s s + alpha_n n + e_art

    Args:
        s_hat ([type]): [description]
        s ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    # s_target
    alpha_s = np.dot(s_hat, s) / np.linalg.norm(s)**2
    s_target = alpha_s * s

    # e_noise
    alpha_n = np.dot(s_hat, n) / np.linalg.norm(n)**2
    e_noise = alpha_n * n

    # e_art
    e_art = s_hat - s_target - e_noise
    
    return s_target, e_noise, e_art

def energy_ratios(s_hat, s, n):
    """
    Compute si_sdr, si_sir, si_sar

    si_sir = si_snr
    (I call it like this because there is only noise as interfering source)

    Args:
        s_hat ([type]): [description]
        s ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    s_target, e_noise, e_art = si_sdr_components(s_hat, s, n)

    si_sdr = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise + e_art)**2)
    si_sir = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise)**2)
    si_sar = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_art)**2)

    return si_sdr, si_sir, si_sar


#TODO: segmental SI-SDR (later, maybe in another work)
# 1. Detect speech activity on clean speech
          # ex: speech_boundaries = librosa.effects.split(s_t, top_db=top_db)
# 2. compute SI-SDR of the segments
# 3. weight SI-SDR by the length of the segments
# 4. average all resulting SI-SDR 

def compute_stats(metrics_keys, all_metrics, all_snr_db, model_data_dir, confidence):

    # Dictionary with all metrics
    metrics = {}
    for id, key in enumerate(metrics_keys):
        metrics[key] = [j[id] for j in all_metrics]

    # Confidence interval
    stats = {}

    # Print the names of the columns. 
    print ("{:<10} {:<10} {:<10}".format('METRIC', 'AVERAGE', 'CONF. INT.')) 
    for key, metric in metrics.items():
        m, h = mean_confidence_interval(metric, confidence=confidence)
        stats[key] = {'avg': m, '+/-': h}
        print ("{:<10} {:<10} {:<10}".format(key, m, h))
    print('\n')

    # # Save stats (si_sdr, si_sar, etc. )
    # with open(model_data_dir + 'stats.json', 'w') as f:
    #     json.dump(stats, f)

    # Metrics by input SNR
    for snr_db in np.unique(all_snr_db):
        stats = {}

        print('Input SNR = {:.2f}'.format(snr_db))
        # Print the names of the columns. 
        print ("{:<10} {:<10} {:<10}".format('METRIC', 'AVERAGE', 'CONF. INT.')) 
        for key, metric in metrics.items():
            subset_metric = np.array(metric)[np.where(all_snr_db == snr_db)]
            m, h = mean_confidence_interval(subset_metric, confidence=confidence)
            stats[key] = {'avg': m, '+/-': h}
            print ("{:<10} {:<10} {:<10}".format(key, m, h))
        print('\n')

        # # Save stats (si_sdr, si_sar, etc. )
        # with open(model_data_dir + 'stats_{:g}.json'.format(snr_db), 'w') as f:
        #     json.dump(stats, f)

def compute_stats_noisnr(metrics_keys, all_metrics, model_data_dir, confidence):

    # Dictionary with all metrics
    metrics = {}
    for id, key in enumerate(metrics_keys):
        metrics[key] = [j[id] for j in all_metrics]

    # Confidence interval
    stats = {}

    # Print the names of the columns. 
    print ("{:<10} {:<10} {:<10}".format('METRIC', 'AVERAGE', 'CONF. INT.')) 
    for key, metric in metrics.items():
        m, h = mean_confidence_interval(metric, confidence=confidence)
        stats[key] = {'avg': m, '+/-': h}
        print ("{:<10} {:<10} {:<10}".format(key, m, h))
    print('\n')

    # Save stats (si_sdr, si_sar, etc. )
    with open(model_data_dir + 'polqa_stats.json', 'w') as f:
        json.dump(stats, f)