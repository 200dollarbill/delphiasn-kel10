import streamlit as st
from dwt_coeff import DWTCoeff
from deps import handler
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.header("Page 4")
st.write("Threshholding")

# const
var = handler.load("dwt6")
fs = 125
dwt6 = var.value
timedwt6 = var.time
total = len(dwt6)



def moving_average_filter(signal, M):
    """
    Applies a two-pass moving average filter to the signal.
    """
    # First pass (causal)
    y1 = np.zeros_like(signal, dtype=float)
    for i in range(len(signal)):
        for j in range(M):
            if i - j >= 0:
                y1[i] += signal[i - j]
        y1[i] /= M

    # Second pass (non-causal, smoothing)
    y2 = np.zeros_like(y1, dtype=float)
    for i in range(len(y1)):
        for j in range(M):
            if i + j < len(y1):
                y2[i] += y1[i + j]
        y2[i] /= M

    return y2


# --- NEW: Peak Detection and RR Interval Functions ---
def deteksi_puncak(sinyal, persen):
    maks = max(abs(x) for x in sinyal)
    batas = persen * maks
    indeks = []
    nilai = []

    for i in range(1, len(sinyal)-1):
        if sinyal[i] > sinyal[i-1] and sinyal[i] > sinyal[i+1]:
            if sinyal[i] > batas:
                indeks.append(i)
                nilai.append(sinyal[i])
    return indeks, nilai

def buat_tabel_rr_rrinterval(waktu, indeks):
    if len(indeks) < 2:
        return pd.DataFrame(), [], []
        
    interval_detik = [float(waktu[indeks[i+1]] - waktu[indeks[i]]) for i in range(len(indeks)-1)]
    bpm_per_beat = [60.0 / i for i in interval_detik if i > 0]
    beat_labels = [f"{waktu[indeks[i]]:.2f} s → {waktu[indeks[i+1]]:.2f} s" for i in range(len(indeks)-1)]

    df = pd.DataFrame({
        "Beat Interval (waktu)": beat_labels,
        "RR Interval (s)": interval_detik,
        "Respiratory Rate (BPM)": bpm_per_beat
    })

    df["RR Interval (s)"] = df["RR Interval (s)"].round(3)
    df["Respiratory Rate (BPM)"] = df["Respiratory Rate (BPM)"].round(2)
    return df, interval_detik, bpm_per_beat


# --- Implementation Section 1: MAV Filter ---
st.markdown("### 1. Signal Filtering")
M = 71 
st.write(f"Filter window size **M = {M}**")
filtered_dwt6 = moving_average_filter(dwt6, M)

fig_filter = go.Figure()
fig_filter.add_trace(go.Scatter(x=timedwt6, y=dwt6, mode='lines', name='Original dwt6 Signal', line=dict(color='rgba(23, 190, 207, 0.6)')))
fig_filter.add_trace(go.Scatter(x=timedwt6, y=filtered_dwt6, mode='lines', name=f'Filtered Signal (M={M})', line=dict(color='rgba(214, 39, 40, 1.0)', width=2)))
fig_filter.update_layout(title_text='Comparison of Original and Moving Average Filtered Data', xaxis_title='Time / Index', yaxis_title='Amplitude', hovermode='x unified')
st.plotly_chart(fig_filter, use_container_width=True)


# --- Implementation Section 2: Peak Detection & RR Analysis ---
st.markdown("### 2. Peak Detection and Respiratory Rate Analysis")
st.write("This section uses the **filtered signal** to detect peaks and calculate the respiratory rate.")

# Interactive threshold for peak detection
peak_threshold_percent = st.slider("Peak Detection Threshold (% of max amplitude)", 0.0, 1.0, 0.3, 0.05)

# Detect peaks on the filtered signal
idx_peaks, val_peaks = deteksi_puncak(filtered_dwt6, peak_threshold_percent)


fig_peaks = go.Figure()
fig_peaks.add_trace(go.Scatter(x=timedwt6, y=filtered_dwt6, mode='lines', name='Filtered Signal', line=dict(color='blue')))
fig_peaks.add_trace(go.Scatter(x=[timedwt6[i] for i in idx_peaks], y=val_peaks, mode='markers', name='Detected Peaks', marker=dict(color='red', symbol='x', size=8)))
fig_peaks.update_layout(title='Detected Peaks on Filtered Signal', xaxis_title='Time (s)', yaxis_title='Amplitude', hovermode='x unified')
st.plotly_chart(fig_peaks, use_container_width=True)

st.write(f"**Total peaks detected:** {len(idx_peaks)}")

df_rr, rr_intervals, bpm_values = buat_tabel_rr_rrinterval(timedwt6, idx_peaks)