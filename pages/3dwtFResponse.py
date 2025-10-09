
import matplotlib
# matplotlib.use('module://matplotlib_kitty') # 
import matplotlib.pyplot as plt
import streamlit as st 
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from deps import handler
from dwt_coeff import DWTCoeff



NUM_SCALES_TO_ANALYZE = 8
FS = 125
print("\n[1/3] Generating Delay Table...")
delay_data = {
    "Skala (j)": [],
    "Delay T (sampel)": [],
    "Delay dalam detik (T / fs)": []
}

for j in range(1, NUM_SCALES_TO_ANALYZE + 1):
    T = round(2**(j-1)) - 1
    delay_data["Skala (j)"].append(j)
    delay_data["Delay T (sampel)"].append(T)
    delay_data["Delay dalam detik (T / fs)"].append(T / FS)

df_delay = pd.DataFrame(delay_data)
print("Tabel Delay Kompensasi (T):")
st.write(df_delay.to_string(index=False))

N_FFT = 2048 
coeff_generator = DWTCoeff()
g = coeff_generator.get_filter(scale=1) 

h = np.array([1, 3, 3, 1]) / 8.0 

Gw = np.abs(np.fft.rfft(g, N_FFT))
Hw = np.abs(np.fft.rfft(h, N_FFT))
freq_axis = np.fft.rfftfreq(N_FFT, 1/FS) 
Q = np.zeros((NUM_SCALES_TO_ANALYZE + 1, len(freq_axis)))

for j in range(1, NUM_SCALES_TO_ANALYZE + 1):
    temp_Q = np.interp(freq_axis, freq_axis / (2**(j-1)), Gw)
    for k in range(j - 1):
        temp_Hw = np.interp(freq_axis, freq_axis / (2**k), Hw)
        temp_Q *= temp_Hw
        
    Q[j] = temp_Q

# plt.style.use('seaborn-v0_8-whitegrid')
# fig, ax = plt.subplots(figsize=(12, 6))
# fig.patch.set_alpha(0.0)
# ax.patch.set_alpha(0.0)

# for j in range(1, NUM_SCALES_TO_ANALYZE + 1):
#     ax.plot(freq_axis, Q[j], label=f"Skala {j}")

# ax.set_title('Frequency Response', color='white')
# ax.set_xlabel('Frekuensi (Hz)',color='white')
# ax.set_ylabel('Magnitude',color='white')
# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')
# ax.spines['left'].set_color('white')
# ax.spines['bottom'].set_color('white')
# ax.set_xlim(0, FS / 2) 
# ax.legend(labelcolor='white')
# st.pyplot(plt)



fig = go.Figure()

for j in range(1, NUM_SCALES_TO_ANALYZE + 1):
    fig.add_trace(go.Scatter(
        x=freq_axis, 
        y=Q[j], 
        mode='lines',
        name=f"Skala {j}" # 'name' is used for the legend label
    ))

fig.update_layout(
    title='Frequency Response',
    xaxis_title='Frekuensi (Hz)',
    yaxis_title='Magnitude',
    xaxis_range=[0, FS / 2],       
    template='plotly_dark',       
    paper_bgcolor='rgba(0,0,0,0)',  
    plot_bgcolor='rgba(0,0,0,0)',   
    legend_title_text='Scales'     
)

st.plotly_chart(fig, use_container_width=True)

for j in range(1, NUM_SCALES_TO_ANALYZE + 1):
    fig.add_trace(go.Scatter(
        x=freq_axis, 
        y=Q[j], 
        mode='lines',
        name=f"Skala {j}" # 'name' is used for the legend label
    ))

range_data = []
for j in range(1, NUM_SCALES_TO_ANALYZE + 1):
    f_min = FS / (2**(j + 1))
    f_max = FS / (2**j)
    bandwidth = f_max - f_min
    range_data.append({
        "Skala": j,
        "Frekuensi Minimum (Hz)": f_min,
        "Frekuensi Maksimum (Hz)": f_max,
        "Bandwidth (Hz)": bandwidth
    })

df_range = pd.DataFrame(range_data)

def highlight_skala_8(row):
    return ['background-color: lightblue' if row.Skala == 8 else '' for _ in row]

styled_df = df_range.style.apply(highlight_skala_8, axis=1).format({
    "Frekuensi Minimum (Hz)": "{:.2f}",
    "Frekuensi Maksimum (Hz)": "{:.2f}",
    "Bandwidth (Hz)": "{:.2f}"
})

st.dataframe(df_range)



print("Rentang Frekuensi dan Bandwidth Teoritis Tiap Skala DWT:")
print(df_range.to_string(index=False, float_format="%.2f"))
