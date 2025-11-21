import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import plotly.graph_objects as go
import os
import pickle
from deps import handler, updater
st.set_page_config(page_title="Tachogram Analysis", layout="wide")
st.title("Heart Rate Variability (HRV) and Tachogram")
st.markdown("This page visualizes the R-R interval from the PPG signal.")

st.subheader("1. Load Data Files")
col1, col2 = st.columns(2)
tacho_filename = col1.text_input("Enter Tachogram Data Filename:", "savedTachoData")
raw_filename = col2.text_input("Enter Raw Data Filename:", "rawdata")

if st.button("Load and Analyze Data"):
    try:
        tacho_data = handler.load(f"data/{tacho_filename}")
        rdata = handler.load(f"data/{raw_filename}")

        if tacho_data is None or rdata is None:
            st.stop()

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
        st.error(f"Could not load or process data. Please ensure the file names are correct. Error: {e}")
        st.stop()

    if len(rr_intervals) > 1:
        mean_rr = np.mean(rr_intervals) * 1000 
        mean_hr = (len(hr_bpm)/300)*60
        sdnn = np.std(rr_intervals) * 1000 

        diff_rr = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(diff_rr**2)) * 1000 
    else:
        mean_rr, mean_hr, sdnn, rmssd = [np.nan] * 4

    st.subheader("Key HRV Statistics")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_col1.metric("Mean Heart Rate", f"{mean_hr:.2f} BPM")
    stat_col2.metric("Mean RR Interval", f"{mean_rr/5:.2f} ms")
    stat_col3.metric("SDNN (Overall Variability)", f"{sdnn:.2f} ms")
    stat_col4.metric("RMSSD (Short-term Variability)", f"{rmssd:.2f} ms")

    st.markdown("---")

    updater.save(mean_hr, "mean_hr")
    col_plots, col_poincare = st.columns([1, 1])

    with col_plots:
        st.subheader("Tachogram Plots (Plotly)")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=rr_time, y=rr_intervals/5, mode='lines+markers', name='RR Interval', line=dict(color='green')))
        fig1.update_layout(title='RR Tachogram', xaxis_title='Time (s)', yaxis_title='RR Interval (s)')
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = go.Figure()
        plot_len = min(len(rr_time), len(hr_bpm))
        fig2.add_trace(go.Scatter(x=rr_time[:plot_len], y=hr_bpm[:plot_len]*50*60, mode='lines+markers', name='HRV', line=dict(color='darkred')))
        fig2.update_layout(title='HRV Tachogram (Beat-to-Beat HR)', xaxis_title='Time (s)', yaxis_title='Heart Rate (BPM)')
        st.plotly_chart(fig2, use_container_width=True)

    with col_poincare:
        st.subheader("Poincaré Plot of RR Intervals")

        fig3, ax3 = plt.subplots(figsize=(8, 8))
        
        rr_i = rr_intervals[:-1]
        rr_i_plus_1 = rr_intervals[1:]

        sd1 = np.std(np.subtract(rr_i, rr_i_plus_1) / np.sqrt(2))
        sd2 = np.std(np.add(rr_i, rr_i_plus_1) / np.sqrt(2))
        
        updater.save(sd1, "sd1")
        updater.save(sd2, "sd2")
        ax3.scatter(rr_i, rr_i_plus_1, alpha=0.5, color='blue', label=f'RR intervals')
        
        center_x = np.mean(rr_i)
        center_y = np.mean(rr_i_plus_1)
        ellipse = Ellipse((center_x, center_y), width=2*sd2, height=2*sd1, angle=45,
                          edgecolor='red', facecolor='none', lw=2, label='SD1/SD2 Ellipse')
        ax3.add_patch(ellipse)

        min_max = [min(rr_intervals), max(rr_intervals)+3]
        ax3.plot(min_max, min_max, color='black', linestyle='--', label='Line of Identity')

        ax3.set_title('Poincaré Plot')
        ax3.set_xlabel('RR$_i$ (s)')
        ax3.set_ylabel('RR$_{i+1}$ (s)')
        ax3.set_aspect('equal', 'box')
        ax3.grid(True)
        ax3.legend()
        st.pyplot(fig3)

        st.markdown(f"""
        The Poincaré plot visualizes the correlation between consecutive RR intervals.
        - **SD1 (Ellipse Width):** {sd1*1000:.2f} ms. Represents short-term HRV, primarily influenced by parasympathetic activity.
        - **SD2 (Ellipse Length):** {sd2*1000:.2f} ms. Represents long-term HRV, influenced by both sympathetic and parasympathetic activity.
        """)