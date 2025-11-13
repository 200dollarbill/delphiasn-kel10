import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from dwt_coeff import DWTCoeff
from deps import handler


st.header("Section 3: Breath Rate Analysis")
st.write("Select raw data to apply Discrete Wavelet Transform (DWT).")

st.subheader("1. Select Data File")

try:
    available_files = [f for f in os.listdir('./data/') if f.endswith('.dat')]
except FileNotFoundError:
    available_files = []

tim = [0]
v = [0]
dwt_results_df = pd.DataFrame({'time': tim, 'value': v})

selected_file = st.selectbox(
    "Select the data session to analyze:",
    options=available_files
)

def tes():
    global valu, timu, session_name
    session_name = "data/"+os.path.splitext(selected_file)[0]
    var = handler.load(session_name)

    if var is None:
        st.stop()
    valu = pd.Series(var.value)
    timu = pd.Series(var.time)

tes()
coeff = DWTCoeff()
ppgdata = valu.to_numpy()
time = timu.to_numpy()
total = len(ppgdata)
w2fb = np.zeros((9, total))
scalecount = 8

for j in range(1, scalecount + 1):
    res = coeff.get_filter(scale=j)
    T = round(2**(j-1)) - 1
    start = len(res)
    for n in range(start, total):
        signalNEW = ppgdata[n - len(res):n]
        w2fb[j, n - T] = np.sum(signalNEW * res[::-1])

st.divider()
st.subheader("2. DWT Results by Scale")

for j in range(1, scalecount + 1):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time / 50, y=w2fb[j], mode='lines', line=dict(color='orange'), name=f'DWT Skala {j}'))
    fig.add_trace(go.Scatter(x=time / 50, y=ppgdata, mode='lines', line=dict(color='blue'), name='PPG Baseline', opacity=0.6))
    fig.update_layout(title=f"Hasil DWT Skala {j}", xaxis_title='Time (s)', yaxis_title='Amplitude')
    st.write(f"**Skala {j}**")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("3. Save DWT Results")
save_filename = st.text_input("Enter filename for the DWT scale 8 results:", value=f"dwt8_{session_name}")

dwt_results_df = pd.DataFrame({'time': time, 'value': w2fb[8]})

if st.button("Save DWT Scale 6 Data For Breath Analysis"):
    tes()
    dwt_results_df = pd.DataFrame({'time': time, 'value': w2fb[6]})
    handler.save(dwt_results_df['time'], dwt_results_df['value'], f"data/{save_filename}")
    st.success(f"Data saving process initiated for `{save_filename}.dat`.")
if st.button("Save DWT Scale 8 Data For Vasomotor Analysis"):
    tes()
    dwt_results_df = pd.DataFrame({'time': time, 'value': w2fb[8]})
    handler.save(dwt_results_df['time'], dwt_results_df['value'], f"data/{save_filename}")
    st.success(f"Data saving process initiated for `{save_filename}.dat`.")
