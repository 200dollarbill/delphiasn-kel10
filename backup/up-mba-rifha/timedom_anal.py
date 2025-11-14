import numpy as np
def calculate_rr_intervals_from_peaks(peaks, fs):
    if len(peaks) < 2:
        return np.array([])
    rr_samples = np.diff(peaks)
    rr_intervals = (rr_samples / fs) * 1000.0  
    return rr_intervals


def rr_tachogram(rr_intervals):
    if len(rr_intervals) == 0:
        return np.array([]), np.array([]), 0
    rr_mean_ms = np.mean(rr_intervals)
    fs_rr = 1000.0 / rr_mean_ms if rr_mean_ms > 0 else 0
    time_axis = np.cumsum(rr_intervals) / 1000.0
    return time_axis, rr_intervals, fs_rr


def sdnn(rr_intervals):
    if len(rr_intervals) < 2:
        return np.nan
    rr_intervals = np.array(rr_intervals)
    rr_mean = np.mean(rr_intervals)
    squared_diffs = (rr_intervals - rr_mean) ** 2
    variance = np.sum(squared_diffs) / (len(rr_intervals) - 1)
    return np.sqrt(variance)


def sdann(rr_intervals, segment_duration=300000):
    rr = np.array(rr_intervals)
    if len(rr) < 2:
        return np.nan
    cumsum_time = np.cumsum(rr)
    num_segments = int(cumsum_time[-1] / segment_duration)
    if num_segments < 2:
        return np.nan
    
    segment_means = []
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        mask = (cumsum_time >= start_time) & (cumsum_time < end_time)
        segment_rr = rr[mask]
        if len(segment_rr) > 0:
            segment_means.append(np.mean(segment_rr))
    
    if len(segment_means) < 2:
        return np.nan
    return np.std(segment_means, ddof=1)


def sdnn_index(rr_intervals, segment_duration=300000):
    rr = np.array(rr_intervals)
    if len(rr) < 2:
        return np.nan
    cumsum_time = np.cumsum(rr)
    num_segments = int(cumsum_time[-1] / segment_duration)
    if num_segments < 1:
        return np.nan
    
    segment_sdnns = []
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        mask = (cumsum_time >= start_time) & (cumsum_time < end_time)
        segment_rr = rr[mask]
        if len(segment_rr) > 1:
            segment_sdnns.append(np.std(segment_rr, ddof=1))
    
    return np.mean(segment_sdnns) if len(segment_sdnns) > 0 else np.nan


def rmssd(rr_intervals):
    if len(rr_intervals) < 2:
        return np.nan
    successive_diff = np.diff(rr_intervals)
    squared_diffs = successive_diff ** 2
    mean_squared = np.mean(squared_diffs)
    return np.sqrt(mean_squared)


def sdsd(rr_intervals):
    if len(rr_intervals) < 3:
        return np.nan
    successive_diff = rr_intervals[1:] - rr_intervals[:-1]
    mean_diff = np.mean(successive_diff)
    centered_diffs = successive_diff - mean_diff
    squared_diffs = centered_diffs ** 2
    variance = np.sum(squared_diffs) / (len(rr_intervals) - 2)
    return np.sqrt(variance)


def nn50(rr_intervals):
    if len(rr_intervals) < 2:
        return 0
    successive_diff = np.abs(np.diff(rr_intervals))
    return int(np.sum(successive_diff > 50))


def pnn50(rr_intervals):
    if len(rr_intervals) < 2:
        return 0
    nn50_count = nn50(rr_intervals)
    total_pairs = len(rr_intervals) - 1
    return (nn50_count / total_pairs) * 100 if total_pairs > 0 else 0


def mean_hr(rr_intervals):
    if len(rr_intervals) < 1:
        return np.nan
    mean_rr_sec = np.mean(rr_intervals) / 1000  
    return 60.0 / mean_rr_sec if mean_rr_sec > 0 else 0


def hrv_triangular_index(rr_intervals, bin_width=8):
    if len(rr_intervals) < 2:
        return np.nan
    hist, bin_edges = np.histogram(
        rr_intervals, 
        bins=np.arange(
            np.min(rr_intervals), 
            np.max(rr_intervals) + bin_width, 
            bin_width
        )
    )
    
    N = len(rr_intervals)  
    h_max = np.max(hist)   
    
    return N / h_max if h_max > 0 else np.nan


def tinn(rr_intervals, bin_width=8):
    if len(rr_intervals) < 2:
        return np.nan
    
    hist, bin_edges = np.histogram(
        rr_intervals, 
        bins=np.arange(
            np.min(rr_intervals), 
            np.max(rr_intervals) + bin_width, 
            bin_width
        )
    )
    
    non_zero_indices = np.where(hist > 0)[0]
    if len(non_zero_indices) < 2:
        return np.nan
    
    left_idx = non_zero_indices[0]
    right_idx = non_zero_indices[-1]
    tinn_value = (right_idx - left_idx) * bin_width
    
    return tinn_value


def cvnn(rr_intervals):
    if len(rr_intervals) < 2:
        return np.nan
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals, ddof=1)
    return (std_rr / mean_rr) * 100 if mean_rr > 0 else np.nan


def cvsd(rr_intervals):
    if len(rr_intervals) < 3:
        return np.nan
    
    successive_diff = np.diff(rr_intervals)
    mean_diff = np.mean(np.abs(successive_diff))
    sdsd_value = sdsd(rr_intervals)
    
    if np.isnan(sdsd_value) or mean_diff == 0:
        return np.nan
    
    return (sdsd_value / mean_diff) * 100


def skewness_nn(rr_intervals):
    if len(rr_intervals) < 3:
        return np.nan
    
    rr = np.array(rr_intervals)
    N = len(rr)
    
    mean_rr = np.mean(rr)
    
    std_rr = np.std(rr, ddof=1)  
    
    if std_rr == 0:
        return np.nan
  
    numerator = N * np.sum((rr - mean_rr) ** 3)
    
    denominator = (N - 1) * (N - 2) * (std_rr ** 3)
    
    if denominator == 0:
        return np.nan
    
    skewness = numerator / denominator
    
    return skewness


def calculate_time_domain_features(rr_intervals):
  
    features = {
        'SDNN': {'value': sdnn(rr_intervals), 'unit': 'ms'},
        'SDANN': {'value': sdann(rr_intervals), 'unit': 'ms'},
        'SDNN Index': {'value': sdnn_index(rr_intervals), 'unit': 'ms'},
        'RMSSD': {'value': rmssd(rr_intervals), 'unit': 'ms'},
        'SDSD': {'value': sdsd(rr_intervals), 'unit': 'ms'},
        'NN50': {'value': nn50(rr_intervals), 'unit': 'count'},
        'pNN50': {'value': pnn50(rr_intervals), 'unit': '%'},
        'Mean HR': {'value': mean_hr(rr_intervals), 'unit': 'bpm'},
        'HRV Triangular Index': {'value': hrv_triangular_index(rr_intervals), 'unit': ''},
        'TINN': {'value': tinn(rr_intervals), 'unit': 'ms'},
        'CVNN': {'value': cvnn(rr_intervals), 'unit': '%'},
        'CVSD': {'value': cvsd(rr_intervals), 'unit': '%'},
        'Skewness': {'value': skewness_nn(rr_intervals), 'unit': ''}
    }
    
    return features