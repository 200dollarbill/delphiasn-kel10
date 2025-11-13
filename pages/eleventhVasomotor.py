import streamlit as st
from dwt_coeff import DWTCoeff
from deps import handler
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
st.header("Page 4")
st.write("Threshholding")

try:
    available_files = [f for f in os.listdir('./data/') if f.endswith('.dat')]
except FileNotFoundError:
    available_files = []

selected_file = st.selectbox(
    "Select the data session to analyze:",
    options=available_files
)
selected_file = selected_file.replace('.dat', '')

var = handler.load(f"data/{selected_file}")
# const
fs = 125
dwt6 = var.value
timedwt6 = var.time / 50
total = len(dwt6)


fig_filter = go.Figure()
fig_filter.add_trace(go.Scatter(x=timedwt6, y=dwt6, mode='lines', name='Vasomotor Signal', line=dict(color='rgba(23, 190, 207, 0.6)')))
#fig_filter.add_trace(go.Scatter(x=timedwt6, y=filtered_dwt6, mode='lines', name=f'Filtered Signal (M={M})', line=dict(color='rgba(214, 39, 40, 1.0)', width=2)))
fig_filter.update_layout(title_text='Vasomotor Signal', xaxis_title='Time / Index', yaxis_title='Amplitude', hovermode='x unified')
st.plotly_chart(fig_filter, use_container_width=True)
