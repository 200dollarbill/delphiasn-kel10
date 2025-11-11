import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deps import handler, updater
import plotly.graph_objects as go
st.set_page_config(page_title="Breath Rate Analysis", layout="wide")
st.title("Breath Rate Variability ")


try:
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
    

except Exception as e:
    st.error(f"Could not load or process 'savedTacho' data. Please ensure the file exists and is in the correct format. Error: {e}")
    st.stop()

if len(breath_intervals) > 1:
    mean_interval = np.mean(breath_intervals) * 1000  
    mean_br = (len(breath_bpm)/300)*60
    diff_breath = np.diff(breath_intervals)

st.subheader("Key Breath Rate Statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean Breath Rate", f"{mean_br:.2f} BPM")
col2.metric("Mean Breath Interval", f"{mean_interval:.2f} ms")


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