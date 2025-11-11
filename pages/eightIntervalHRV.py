import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from scipy.fft import fft
import os
import pickle
from deps import handler, updater

st.set_page_config(page_title="Advanced HRV Analysis", layout="wide")
st.title("Advanced Interval Variability Analysis")

st.subheader("1. Load Data Files")
col1, col2 = st.columns(2)
tacho_filename = col1.text_input("Enter Tachogram Data Filename:", "savedTachoData")
raw_filename = col2.text_input("Enter Raw Data Filename:", "rawdata")

if st.button("Generate Advanced Analysis Summary"):
    try:
        ppg = handler.load(f"data/{raw_filename}")
        tacho_data = handler.load(f"data/{tacho_filename}")
        if ppg is None or tacho_data is None:
            st.stop()
        
        fs = 50.0
        signal_values = np.array(ppg.value)
        rr_intervals_sec = np.array(tacho_data.time)
        #st.write(rr_intervals_sec)

    except Exception as e:
        st.error(f"Failed to load necessary data. Error: {e}")
        st.stop()

    @st.cache_data
    def run_fft_for_features(signal, fs):
        window_size, overlap, n_fft = 256, 128, 2048
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
        vlf_power = band_power(freqs, psd, 0.003, 0.04)
        lf_power = band_power(freqs, psd, 0.04, 0.15)
        hf_power = band_power(freqs, psd, 0.15, 0.4)
        total_power = vlf_power + lf_power + hf_power
        return {
            'LF_percent': (lf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0,
            'HF_percent': (hf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
        }
    

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


    def cvsd(rr_intervals,sdsd):
        if len(rr_intervals) < 3:
            return np.nan
        
        successive_diff = np.diff(rr_intervals)
        mean_diff = np.mean(np.abs(successive_diff))
        sdsd_value = sdsd
        
        if np.isnan(sdsd_value) or mean_diff == 0:
            return np.nan
        
        return (sdsd_value / mean_diff) 


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

    @st.cache_data
    def calculate_interval_metrics(rr_intervals):
        rr_ms = rr_intervals * 1000
        diffs = np.diff(rr_ms)
        sdnn = np.std(rr_ms, ddof=1) 
        rmssd = np.sqrt(np.mean(diffs**2)) 
        nn50 = np.sum(np.abs(diffs) > 50)
        pnn50 = (nn50 / len(diffs)) * 100 if len(diffs) > 0 else 0
        sdsd = np.std(diffs, ddof=1) 
        sd1 = np.std(diffs) / np.sqrt(2)
        sd2 = np.sqrt(2 * np.std(rr_ms)**2 - 0.5 * np.std(diffs)**2)

        sd_ratio = sd1 / sd2 if sd2 > 0 else np.nan
        return {
            "sdnn": sdnn, "rmssd": rmssd, "nn50": nn50, "pnn50": pnn50, "sdsd": sdsd,
            "sd1": sd1, "sd2": sd2, "sd_ratio": sd_ratio,
            "hrv_ti": hrv_triangular_index(rr_ms),
            "tinn": tinn(rr_ms),
            "cvnn": cvnn(rr_ms),
            "cvsd": cvsd(rr_ms,sdsd),
            "skewness": skewness_nn(rr_ms)
        }

    def plot_autonomic_balance_diagram(lf_percent, hf_percent):
        fig = go.Figure()
        colors = [['#f4aaaa', '#f4c07a', '#f4aaaa'], ['#f4f47a', '#b4e197', '#f4f47a'], ['#f4f47a', '#b4e197', '#b4e197']]
        zone_number = 1
        for i in range(3):
            for j in range(3):
                x0, y0 = j * 33.33, i * 33.33
                x1, y1 = (j + 1) * 33.33, (i + 1) * 33.33
                fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="Black"), fillcolor=colors[i][j], layer='below')
                fig.add_annotation(x=x0 + 16.66, y=y0 + 16.66, text=str(zone_number), showarrow=False, font=dict(size=20, color="black", family="Arial, bold"))
                zone_number += 1
        fig.add_trace(go.Scatter(
            x=[lf_percent], y=[hf_percent], mode='markers+text',
            marker=dict(color='red', size=15, symbol='circle'),
            text=['Area'], textposition="top right",
            textfont=dict(size=14, color='red', family="Arial, bold"), name='Your Position'
        ))
        fig.update_layout(
            title=dict(text='<b>Autonomic Balance Diagram</b>', x=0.5),
            xaxis_title='Sympathetic NS - LF (n.u.)', yaxis_title='Parasympathetic NS - HF (n.u.)',
            xaxis=dict(range=[0, 100], showgrid=False), yaxis=dict(range=[0, 100], showgrid=False),
            width=600, height=600, showlegend=False
        )
        return fig

    fft_features = run_fft_for_features(signal_values, fs)
    interval_metrics = calculate_interval_metrics(rr_intervals_sec)

    st.markdown("---")
    disp_col1, disp_col2 = st.columns([1, 1])
    with disp_col1:
        st.subheader("Autonomic Balance")
        fig = plot_autonomic_balance_diagram(fft_features['LF_percent'], fft_features['HF_percent'])
        st.plotly_chart(fig, use_container_width=True)
    with disp_col2:
        st.subheader("Interval Variability Metrics")
        st.info("**Time-Domain Metrics**")
        sub_col1, sub_col2 = st.columns(2)
        sub_col1.metric("SDNN", f"{interval_metrics['sdnn']:.2f} ms", help="Standard deviation of all intervals. Reflects overall variability.")
        sub_col2.metric("RMSSD", f"{interval_metrics['rmssd']:.2f} ms", help="Root mean square of successive differences. Reflects short-term, parasympathetic activity.")
        updater.save(interval_metrics['sdnn'], "SDNN")
        updater.save(interval_metrics['rmssd'], "RMSSD")
        with st.expander("Show more time-domain metrics..."):
            sub_col3, sub_col4, sub_col5 = st.columns(3)
            sub_col3.metric("NN50", f"{interval_metrics['nn50']}", help="Number of successive interval differences > 50 ms.")
            sub_col4.metric("pNN50", f"{interval_metrics['pnn50']:.2f} %", help="Percentage of NN50.")
            sub_col5.metric("SDSD", f"{interval_metrics['sdsd']:.2f} ms", help="Standard deviation of successive differences.")
            updater.save(interval_metrics['nn50'], "nn50")
            updater.save(interval_metrics['pnn50'], "pnn50")
            updater.save(interval_metrics['sdsd'], "SDSD")
            st.write("---")
            st.markdown("###### Geometric & Statistical Metrics")
            n_col1, n_col2, n_col3 = st.columns(3)
            n_col1.metric("HRV Triangular Index", f"{interval_metrics['hrv_ti']:.2f}", help="Total number of RR intervals divided by the peak height of the histogram.")
            n_col2.metric("TINN", f"{interval_metrics['tinn']:.2f} ms", help="Baseline width of the RR interval histogram.")
            n_col3.metric("CVNN", f"{interval_metrics['cvnn']:.2f} %", help="Coefficient of variation of NN intervals (SDNN/MeanNN).")
            
            n_col4, n_col5, _ = st.columns(3)
            n_col4.metric("CVSD", f"{interval_metrics['cvsd']:.2f} %", help="Coefficient of variation of successive differences (SDSD/MeanDiff).")
            n_col5.metric("Skewness", f"{interval_metrics['skewness']:.3f}", help="Skewness of the RR interval distribution.")
            updater.save(interval_metrics['hrv_ti'], "hrv_ti")
            updater.save(interval_metrics['tinn'], "tinn")
            updater.save(interval_metrics['cvnn'], "cvnn")
            updater.save(interval_metrics['cvsd'], "cvsd")
            updater.save(interval_metrics['skewness'], "skewness")

        st.info("**Non-Linear (Poincaré) Metrics**")
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric("SD1", f"{interval_metrics['sd1']:.2f} ms", help="Represents short-term variability (width of Poincaré ellipse).")
        p_col2.metric("SD2", f"{interval_metrics['sd2']:.2f} ms", help="Represents long-term variability (length of Poincaré ellipse).")
        p_col3.metric("SD1/SD2 Ratio", f"{interval_metrics['sd_ratio']:.3f}")

        #updater.save(interval_metrics['sd1'], "sd1")
        #updater.save(interval_metrics['sd2'], "sd2")
        updater.save(interval_metrics['sd_ratio'], "sdratio")
