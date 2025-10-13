import streamlit as st
from dwt_coeff import DWTCoeff
from deps import handler
import numpy as np
import pandas as pd
import plotly.graph_objects as go


st.header("Page 2")
st.write("mohon ditunggu soalnya lama")

# const
coeff = DWTCoeff()
var = handler.load("rawdata")
fs = 125
ppgdata = var.value.to_numpy()
time = var.time.to_numpy()
total = len(ppgdata)

#print(var.time)
#print(var.value)
w2fb = np.zeros((9, total))
scalecount = 8

# a trous algo
for j in range(1,scalecount+1):
    res = coeff.get_filter(scale=j)
    a = -(round(2**j) + round(2**(j-1)) - 2)
    b = -(1 - round(2**(j-1))) + 1
    T = round(2**(j-1)) - 1

    start = len(res)
    for n in range(start, total):
        signalNEW = ppgdata[n-len(res):n]
        w2fb[j,n-T] = np.sum(signalNEW*res[::-1])

out = {}

for j in range(1,scalecount):
    out[j] = w2fb[j,:]

out_df = pd.DataFrame(out, index=var.time)

print(out_df.head)

for j in range(1, scalecount+1):
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=w2fb[j],
        mode='lines',
        line=dict(color='orange'),
        name=f'DWT Skala {j}'
    ))
    fig.add_trace(go.Scatter(
        x=time,
        y=ppgdata,
        mode='lines',
        line=dict(color='blue'),
        name='PPG Baseline',
        opacity=0.6
    ))
    fig.update_layout(
        title=f"Hasil DWT Skala {j}",
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        width=900,
        height=400,
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=50, b=40)
    )
    st.write("Skala", j)
    st.plotly_chart(fig)

handler.save(time,w2fb[6],"dwt6")

# var=coeff.get_filter(scale=2)
# print(var)
