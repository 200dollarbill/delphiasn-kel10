import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from deps import handler # Your data loading handler

# -----------------------------------------------------------------------------
# Page Configuration and Title
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Breath Rate Analysis", layout="wide")
st.title("Breath Rate Variability ")


try:
    tacho_data = handler.load("savedDWT")
    rdata = handler.load("rawdata")

    timedata = rdata.time
    rr_intervals = np.array(tacho_data.value)/50
    
    for i in range(len(rr_intervals)):
        if rr_intervals[i] > 5:
            rr_intervals[i] = np.mean(rr_intervals)
    
    hr_bpm = np.array(tacho_data.time)/50

    for i in range(len(hr_bpm)):
        if hr_bpm[i] < 0.004:
            hr_bpm[i] = np.mean(hr_bpm)

    rr_time = np.cumsum(rr_intervals)*(3/16)
    rr_time = np.insert(rr_time, 0, 0)[:-1]
    

except Exception as e:
    st.error(f"Could not load or process 'savedTacho' data. Please ensure the file exists and is in the correct format. Error: {e}")
    st.stop()

if len(rr_intervals) > 1:
    mean_interval = np.mean(rr_intervals) * 1000  
    mean_br = (len(hr_bpm)/300)*60
    sdnn = np.std(rr_intervals) * 1000 
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr**2)) * 1000 
else:
    mean_interval, mean_br, sdnn, rmssd = [np.nan] * 4

st.subheader("Key HRV Statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean Breath Rate", f"{mean_br:.2f} BPM")
col2.metric("Mean Breath Interval", f"{mean_interval/5:.2f} ms")


st.markdown("---")

col_plots, col_poincare = st.columns([1, 1])

with col_plots:
    st.subheader("Tachogram Plots")

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(rr_time, rr_intervals/5, marker='o', linestyle='-', color='green')
    ax1.set_title('RR Tachogram')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('RR Interval (s)')
    ax1.grid(True)
    plt.tight_layout()
    st.pyplot(fig1)

    # --- HRV (BPM) Tachogram ---
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    # Ensure hr_bpm and rr_time have the same length for plotting
    plot_len = min(len(rr_time), len(hr_bpm))
    ax2.plot(rr_time, hr_bpm[:plot_len]*50*60, marker='o', linestyle='-', color='darkred')
    ax2.set_title('HRV Tachogram (Beat-to-Beat HR)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Heart Rate (BPM)')
    ax2.grid(True)
    plt.tight_layout()
    st.pyplot(fig2)

with col_poincare:
    st.subheader("Poincaré Plot of RR Intervals")
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    
    rr_i = rr_intervals[:-1]
    rr_i_plus_1 = rr_intervals[1:]
    sd1 = np.std(np.subtract(rr_i, rr_i_plus_1) / np.sqrt(2))
    sd2 = np.std(np.add(rr_i, rr_i_plus_1) / np.sqrt(2))
    ax3.scatter(rr_i, rr_i_plus_1, alpha=0.5, color='blue', label=f'RR intervals')
    
    center_x = np.mean(rr_i)
    center_y = np.mean(rr_i_plus_1)
    ellipse = Ellipse((center_x, center_y), width=2*sd2, height=2*sd1, angle=45,
                      edgecolor='red', facecolor='none', lw=2, label='SD1/SD2 Ellipse')
    ax3.add_patch(ellipse)

    # Plot the line of identity (y=x)
    min_max = [min(rr_intervals), max(rr_intervals)+3]
    ax3.plot(min_max, min_max, color='black', linestyle='--', label='Line of Identity')

    ax3.set_title('Poincaré Plot')
    ax3.set_xlabel('RR$_i$ (s)')
    ax3.set_ylabel('RR$_{i+1}$ (s)')
    ax3.set_aspect('equal', 'box')
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)

   