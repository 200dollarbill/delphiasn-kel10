import streamlit as st
from scipy.signal import find_peaks # Import the required function
from deps import handler
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Page Configuration and Header
# -----------------------------------------------------------------------------
st.header("Page 4")
st.write("Peak Detection and Interval Analysis")

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
try:
    var = handler.load("rawdata1")
    fs = 125  # Sampling Frequency
    dwt6 = var.value
    timedwt6 = var.time
except Exception as e:
    st.error(f"Failed to load data from 'rawdata1'. Error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

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

def buat_tabel_rr_rrinterval(waktu_indices, peak_indices, fs):
    """
    Creates a DataFrame of RR intervals and BPM from peak indices.
    Uses the correct sampling frequency (fs) for time conversion.
    """
    # Convert index-based time to seconds using the correct sampling frequency
    waktu_sec = waktu_indices / fs

    if len(peak_indices) < 2:
        return pd.DataFrame(), [], []
        
    # Calculate intervals between consecutive peaks in seconds
    interval_detik = [float(waktu_sec[peak_indices[i+1]] - waktu_sec[peak_indices[i]]) for i in range(len(peak_indices)-1)]
    bpm_per_beat = [60 / i for i in interval_detik if i > 0]
    beat_labels = [f"{waktu_sec[peak_indices[i]]:.2f} s to {waktu_sec[peak_indices[i+1]]:.2f} s" for i in range(len(peak_indices)-1)]

    df = pd.DataFrame({
        "Beat Interval (waktu)": beat_labels,
        "RR Interval (s)": interval_detik,
        "Respiratory Rate (BPM)": bpm_per_beat
    })

    df["RR Interval (s)"] = df["RR Interval (s)"].round(3)
    df["Respiratory Rate (BPM)"] = df["Respiratory Rate (BPM)"].round(2)
    return df, interval_detik, bpm_per_beat

# -----------------------------------------------------------------------------
# Section 1: Signal Filtering
# -----------------------------------------------------------------------------
st.markdown("### 1. Signal Filtering")
M = st.slider("Filter Window Size (M)", min_value=1, max_value=50, value=8)
filtered_dwt6 = moving_average_filter(dwt6, M)

fig_filter = go.Figure()
fig_filter.add_trace(go.Scatter(x=timedwt6, y=dwt6, mode='lines', name='Original Signal', line=dict(color='rgba(23, 190, 207, 0.6)')))
fig_filter.add_trace(go.Scatter(x=timedwt6, y=filtered_dwt6, mode='lines', name=f'Filtered Signal (M={M})', line=dict(color='rgba(214, 39, 40, 1.0)', width=2)))
fig_filter.update_layout(title_text='Comparison of Original and Filtered Data', xaxis_title='Time / Index', yaxis_title='Amplitude', hovermode='x unified')
st.plotly_chart(fig_filter, use_container_width=True)

# -----------------------------------------------------------------------------
# Section 2: Peak Detection with SciPy
# -----------------------------------------------------------------------------
st.markdown("### 2. Peak Detection and Respiratory Rate Analysis")
st.write("This section uses the **filtered signal** and `scipy.signal.find_peaks` for robust peak detection.")

# --- Interactive controls for SciPy's find_peaks ---
col1, col2 = st.columns(2)
prominence_val = col1.slider(
    "Peak Prominence", 
    min_value=0.0, 
    max_value=max(filtered_dwt6) if len(filtered_dwt6) > 0 else 1.0, 
    value=0.5, 
    step=0.1,
    help="Filters peaks based on how much they stand out vertically from their surroundings."
)
distance_val = col2.slider(
    "Minimum Distance Between Peaks (samples)", 
    min_value=1, 
    max_value=250, # fs * 2 seconds
    value=int(fs * 0.4), # Default to 40% of a second
    help="Minimum number of samples between consecutive peaks."
)

# Use scipy.signal.find_peaks for robust detection
idx_peaks, _ = find_peaks(
    filtered_dwt6, 
    prominence=prominence_val, 
    distance=distance_val
)
val_peaks = filtered_dwt6[idx_peaks]

# --- Plotting the results ---
fig_peaks = go.Figure()
fig_peaks.add_trace(go.Scatter(x=timedwt6, y=filtered_dwt6, mode='lines', name='Filtered Signal', line=dict(color='blue')))
fig_peaks.add_trace(go.Scatter(x=timedwt6[idx_peaks], y=val_peaks, mode='markers', name='Detected Peaks', marker=dict(color='red', symbol='x', size=8)))
fig_peaks.update_layout(title='Detected Peaks on Filtered Signal (SciPy)', xaxis_title='Time / Index', yaxis_title='Amplitude', hovermode='x unified')
st.plotly_chart(fig_peaks, use_container_width=True)

st.write(f"**Total peaks detected:** {len(idx_peaks)}")

# --- Analysis and Data Saving ---
df_rr, rr_intervals, bpm_values = buat_tabel_rr_rrinterval(timedwt6, idx_peaks, fs)

st.write("### Calculated Intervals and Rates")
st.dataframe(df_rr)

if st.button("Save Tachogram Data", key="SAVEDATAKEY"):
    if not df_rr.empty:
        # Correctly save the calculated columns without extra division
        handler.save(df_rr['RR Interval (s)'], df_rr['Respiratory Rate (BPM)'], "savedTachoData")
        st.success("Tachogram data saved successfully!")
    else:
        st.warning("No intervals were calculated, so no data was saved.")