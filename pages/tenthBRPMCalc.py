import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deps import handler, updater
import plotly.graph_objects as go
st.set_page_config(page_title="Breath Rate Analysis", layout="wide")
st.title("Results:")


tacho_data = handler.load("data/savedDWT")
brdata = handler.load("data/rawdata1")
timedata = brdata.time
td2 = brdata.time
# st.write(timedata)
breath_intervals = np.array(tacho_data.value)/50
# st.write(tacho_data.time)

for i in range(len(breath_intervals)):
    if breath_intervals[i] > 5:
        breath_intervals[i] = np.mean(breath_intervals)

breath_bpm = np.array(tacho_data.time)/50

for i in range(len(breath_bpm)):
    if breath_bpm[i] < 0.004:
        breath_bpm[i] = np.mean(breath_bpm)
br_time2 = np.cumsum(breath_intervals)
br_time = np.cumsum(breath_intervals)*(3/16)
# rr_time = np.insert(rr_time, 0, 0)[:-1]
    


if len(breath_intervals) > 1:
    mean_interval = np.mean(breath_intervals) * 1000  
    mean_br = (len(breath_bpm)/300)*60
    diff_breath = np.diff(breath_intervals)

cvnn = updater.load("cvnn")
cvnn = cvnn.value

st.subheader("Breath Rate")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean Breath Rate", f"{mean_br:.2f} BPM")
col2.metric("Mean Breath Interval", f"{mean_interval:.2f} ms")
st.markdown("---") 

metric_map = {
    # time domain
    "cvsd": "CVSD",
    "hrv_ti": "HRV Triangular Index",
    "nn50": "NN50 Count",
    "pnn50": "pNN50",
    "RMSSD": "RMSSD",
    "sd1": "SD1",
    "sd2": "SD2",
    "SDNN": "SDNN",
    "sdratio": "SD Ratio (SD2/SD1)",
    "SDSD": "SDSD",
    "tinn": "TINN",
    "cvnn": "CVNN",

    # f domain
    "hfNU": "HF (Normalized Units)",
    "hfTP": "HF (Total Power)",
    "lfhfratio": "LF/HF Ratio",
    "lfNU": "LF (Normalized Units)",
    "lfTP": "LF (Total Power)",
    "peakHF": "Peak HF",
    "peakLF": "Peak LF",
    "TP": "Total Power",
    "skewness": "Skewness",
}

st.subheader("Time-Domain HRV Metrics")
col_t1, col_t2, col_t3, col_t4 = st.columns(4)
time_domain_cols = [col_t1, col_t2, col_t3, col_t4]
time_domain_keys = ["cvsd", "hrv_ti", "nn50", "pnn50", "RMSSD", "sd1", "sd2", "SDNN", "sdratio", "SDSD", "tinn", "cvnn"]

for i, key in enumerate(time_domain_keys):
    col = time_domain_cols[i % 4]
    value_obj = updater.load(key)
    value = value_obj.value
    display_name = metric_map.get(key, key.upper()) 
    unit = "ms" if key in ["cvsd", "RMSSD", "sd1", "sd2", "SDNN", "SDSD", "tinn"] else ""
    unit = " Count" if key == "nn50" else unit
    unit = "%" if key == "pnn50" else unit
    
    col.metric(display_name, f"{value:.2f}{unit}")
        
st.markdown("---") 

st.subheader("Frequency-Domain HRV Metrics")
col_f1, col_f2, col_f3, col_f4 = st.columns(4)
freq_domain_cols = [col_f1, col_f2, col_f3, col_f4]
freq_domain_keys = ["hfNU", "lfNU", "lfhfratio", "hfTP", "lfTP", "TP", "peakLF", "peakHF"]

for i, key in enumerate(freq_domain_keys):
    col = freq_domain_cols[i % 4]
    value_obj = updater.load(key)
    value = value_obj.value
    display_name = metric_map.get(key, key.upper())
    unit = "ms²" if key in ["hfTP", "lfTP", "TP"] else ""
    unit = " n.u." if key in ["hfNU", "lfNU"] else ""
    
    col.metric(display_name, f"{value:.4f}{unit}")

st.markdown("---") 
st.subheader("Other Metrics")
col_o1, col_o2 = st.columns(2)
other_keys = ["skewness"]

for i, key in enumerate(other_keys):
    col = col_o1 if i == 0 else col_o2
    value_obj = updater.load(key)
    value = value_obj.value
    display_name = metric_map.get(key, key.upper())
    
    col.metric("display_name", f"{value:.2f}")

st.markdown("---")
st.subheader("Tachogram Plots")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=td2*3,
    y=breath_intervals*3,
    mode='lines+markers',  
    name='Breath Rate Interval',
    line=dict(color='green'),
    marker=dict(color='green')
))

fig1.update_layout(
    title='Breath Rate Tachogram',
    xaxis_title='Time (s)',
    yaxis_title='Interval (s)'
)

st.plotly_chart(fig1, use_container_width=True)