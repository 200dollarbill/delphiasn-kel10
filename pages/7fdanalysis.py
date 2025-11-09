import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from deps import handler
from scipy.fft import fft

# -----------------------------------------------------------------------------
# Page Configuration and Title
# -----------------------------------------------------------------------------
st.set_page_config(page_title="FFT Analysis", layout="wide")
st.title("Frequency Domain Analysis (FFT)")
st.markdown("This page performs Power Spectral Density (PSD) estimation on the raw PPG signal to analyze its frequency components.")

# -----------------------------------------------------------------------------
# Data Loading and Preparation
# -----------------------------------------------------------------------------
try:
    # Load raw data for FFT
    ppg = handler.load("../rawdata")
    fs = 50.0  # Sampling Frequency is 50 Hz
    time_vector = np.array(ppg.time) / fs
    signal_values = np.array(ppg.value)

    # Load interval data for SD1 calculation
    tacho_data = handler.load("savedTacho")
    rr_intervals_sec = np.array(tacho_data.time)

    st.subheader("Raw PPG Signal Preview")
    fig_raw = go.Figure(data=go.Scatter(x=time_vector, y=signal_values, mode='lines'))
    fig_raw.update_layout(title="Loaded Raw PPG Data", xaxis_title="Time (s)", yaxis_title="Amplitude")
    st.plotly_chart(fig_raw, use_container_width=True)

except Exception as e:
    st.error(f"Failed to load necessary data. Ensure 'rawdata' and 'savedTacho' are available. Error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Calculation Functions
# -----------------------------------------------------------------------------
@st.cache_data
def run_full_fft_analysis(signal, fs, window_size, overlap, n_fft):
    """Orchestrates the FFT analysis to calculate PSD and frequency-domain features."""
    
    def hamming(N):
        return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))

    def calculate_psd(full_signal, fs, window_size, overlap, pad_to):
        step = window_size - overlap
        segments = []
        for start in range(0, len(full_signal) - window_size + 1, step):
            segment = np.array(full_signal[start:start + window_size]) - np.mean(full_signal[start:start + window_size])
            windowed = segment * hamming(window_size)
            padded = np.pad(windowed, (0, pad_to - window_size), 'constant')
            fft_res = fft(padded)
            psd = (2 * (np.abs(fft_res) ** 2)) / (fs * np.sum(hamming(window_size)**2))
            segments.append(psd[:pad_to // 2 + 1])
        psd_avg = np.mean(segments, axis=0)
        freqs = np.linspace(0, fs / 2, len(psd_avg))
        return freqs, psd_avg

    def band_power(freqs, psd, f_low, f_high):
        indices = np.where((freqs >= f_low) & (freqs < f_high))
        return np.trapz(psd[indices], freqs[indices])

    freqs, psd = calculate_psd(signal, fs, window_size, overlap, pad_to=n_fft)
    
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    lf_power = band_power(freqs, psd, *lf_band)
    hf_power = band_power(freqs, psd, *hf_band)
    total_power = lf_power + hf_power

    # Find peak frequencies
    lf_indices = np.where((freqs >= lf_band[0]) & (freqs < lf_band[1]))
    hf_indices = np.where((freqs >= hf_band[0]) & (freqs < hf_band[1]))
    
    peak_lf_freq = freqs[lf_indices][np.argmax(psd[lf_indices])] if len(lf_indices[0]) > 0 else 0
    peak_hf_freq = freqs[hf_indices][np.argmax(psd[hf_indices])] if len(hf_indices[0]) > 0 else 0

    features = {
        'TP': total_power,
        'TP_of_LF': lf_power,
        'TP_of_HF': hf_power,
        'LF_HF_Ratio': lf_power / hf_power if hf_power > 0 else np.inf,
        'LF_nu': (lf_power / total_power) * 100 if total_power > 0 else 0,
        'HF_nu': (hf_power / total_power) * 100 if total_power > 0 else 0,
        'Peak_Freq_LF': peak_lf_freq,
        'Peak_Freq_HF': peak_hf_freq,
    }
    return freqs, psd, features

@st.cache_data
def calculate_sd1(rr_intervals):
    """Calculates the SD1 metric from RR intervals."""
    rr_ms = rr_intervals * 1000
    diffs = np.diff(rr_ms)
    return np.std(diffs) / np.sqrt(2)

# -----------------------------------------------------------------------------
# Sidebar for User Controls
# -----------------------------------------------------------------------------
st.sidebar.header("FFT Analysis Controls")
win_size = st.sidebar.slider("Window Size", 50, 500, 256, 1)
overlap_percent = st.sidebar.slider("Overlap Percentage", 0, 90, 50, 5)
overlap_val = int(win_size * (overlap_percent / 100))
st.sidebar.info(f"Current Overlap: {overlap_val} samples")
n_fft = st.sidebar.selectbox(
    "FFT Resolution (N-points)",
    options=[256, 512, 1024, 2048, 4096],
    index=3,
    help="Higher values increase frequency resolution (smoother plot)."
)

# -----------------------------------------------------------------------------
# Perform Analysis and Display Results
# -----------------------------------------------------------------------------
freqs, psd, features = run_full_fft_analysis(signal_values, fs, win_size, overlap_val, n_fft)
sd1_metric = calculate_sd1(rr_intervals_sec)

st.markdown("---")
st.subheader("Summary of Calculated Features")

# Create a dictionary for the summary table
summary_data = {
    "TP (ms²/Hz)": features['TP'],
    "TP of LF (ms²/Hz)": features['TP_of_LF'],
    "TP of HF (ms²/Hz)": features['TP_of_HF'],
    "LF/HF Ratio": features['LF_HF_Ratio'],
    "LF (n.u.)": features['LF_nu'],
    "HF (n.u.)": features['HF_nu'],
    "Peak Frequency of LF (Hz)": features['Peak_Freq_LF'],
    "Peak Frequency of HF (Hz)": features['Peak_Freq_HF'],
    "SD1 (ms)": sd1_metric
}

# Convert to a DataFrame for clean display
summary_df = pd.DataFrame.from_dict(summary_data, orient='index', columns=['Value'])
summary_df.index.name = "Metric"
st.dataframe(summary_df.style.format("{:.4f}")) # Format values to 4 decimal places

# -----------------------------------------------------------------------------
# Plotting with Plotly
# -----------------------------------------------------------------------------
st.markdown("---")
plot_col1, plot_col2 = st.columns([2, 1])

with plot_col1:
    st.subheader("Power Spectral Density (PSD)")
    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(x=freqs[0:(round(0.014648438*n_fft))+2], y=psd, mode='lines', name='PSD', line=dict(color='blue')))
    
    fig_psd.add_vrect(x0=0.04, x1=0.15, fillcolor="yellow", opacity=0.2, layer="below", line_width=0, name='LF')
    fig_psd.add_vrect(x0=0.15, x1=0.4, fillcolor="red", opacity=0.2, layer="below", line_width=0, name='HF')

    fig_psd.update_layout(
        title='PSD with Frequency Bands',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Power Spectral Density',
        xaxis=dict(range=[0, 0.7], autorange=False)
    )
    st.plotly_chart(fig_psd, use_container_width=True)

with plot_col2:
    st.subheader("Normalized Power Distribution")
    fig_bar = go.Figure(data=[go.Bar(
        x=['LF (n.u.)', 'HF (n.u.)'],
        y=[features['LF_nu'], features['HF_nu']],
        text=[f"{features['LF_nu']:.1f}%", f"{features['HF_nu']:.1f}%"],
        textposition='auto',
        marker_color=['blue', 'red']
    )])
    
    fig_bar.update_layout(
        title=f"LF/HF Ratio: {features['LF_HF_Ratio']:.2f}",
        yaxis_title="Normalized Units (%)",
        yaxis_range=[0, 100]
    )
    st.plotly_chart(fig_bar, use_container_width=True)