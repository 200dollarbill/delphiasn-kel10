import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dwt_functions import compute_dwt_scale_features, compute_frequency_response

st.set_page_config(page_title="PPG Signal Processing", layout="wide")

def highpass_filter_iir(x, lowcut, dt):
    N = len(x)
    RC_hp = 1.0 / (2.0 * np.pi * lowcut)
    alpha_hp = RC_hp / (RC_hp + dt)
    
    baseline_removed = np.zeros(N)
    for i in range(1, N):
        baseline_removed[i] = alpha_hp * (baseline_removed[i-1] + x[i] - x[i-1])
    
    return baseline_removed

def bandpass_filter(signal, fs, lowcut=0.1, highcut=0.5):
    
    N = len(signal)
    dt = 1.0 / fs
    # hpf
    RC_hp = 1.0 / (2.0 * np.pi * lowcut)
    alpha_hp = RC_hp / (RC_hp + dt)
    
    hp_output = np.zeros(N)
    for i in range(1, N):
        hp_output[i] = alpha_hp * (hp_output[i-1] + signal[i] - signal[i-1])
    
    # lpf
    RC_lp = 1.0 / (2.0 * np.pi * highcut)
    alpha_lp = dt / (RC_lp + dt)
    
    bp_output = np.zeros(N)
    bp_output[0] = hp_output[0]
    for i in range(1, N):
        bp_output[i] = alpha_lp * hp_output[i] + (1 - alpha_lp) * bp_output[i-1]
    
    return bp_output

st.title("PPG Signal Processing")

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'w2fb' not in st.session_state:
    st.session_state.w2fb = None
if 'downsampled_t' not in st.session_state:
    st.session_state.downsampled_t = None
if 'downsampled_signal' not in st.session_state:
    st.session_state.downsampled_signal = None
st.sidebar.header("Upload File")
uploaded_file = st.sidebar.file_uploader("Upload PPG File (.csv)", type=["csv"])

st.sidebar.header("Signal Parameters")
lowcut = st.sidebar.number_input("High-pass cutoff (Hz)", value=0.7, step=0.05)
downsample_factor = st.sidebar.slider("Downsample factor", 1, 20, 2)
max_scale = st.sidebar.slider("Max DWT Scale", 1, 8, 8)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Input Data", "DWT Analysis", "Frequency Response", "Peak Detection", "Peak Detection (BPF)"])

#input data (sekalian preprocessing)
with tab1:
    st.header("Input Data Processing")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = ['timestamp', 'red']
        
        df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / 1000000.0

        t = df['timestamp'].values
        x = df['red'].values
        N = len(x)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", f"{N:,}")
        with col2:
            dt = np.mean(np.diff(t))
            fs = 1.0 / dt
            st.metric("Sampling Rate", f"{fs:.2f} Hz")
        with col3:
            st.metric("Duration", f"{(t[-1]-t[0]):.2f} s")
        
        st.divider()
        
        baseline_removed = highpass_filter_iir(x, lowcut, dt)
        ds_t = t[::downsample_factor]
        ds_orig = x[::downsample_factor]
        ds_filt = baseline_removed[::downsample_factor]
        
        st.session_state.downsampled_t = ds_t
        st.session_state.downsampled_signal = ds_filt
        st.session_state.downsampled_fs = fs / downsample_factor
        st.session_state.data_loaded = True
        
        st.subheader("Original PPG Signal")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=ds_t, y=ds_orig, mode='lines', 
                                  line=dict(color='red', width=1.5), name='Original'))
        fig1.update_layout(height=400, xaxis_title="Time (s)", yaxis_title="Amplitude",
                          template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Baseline-Removed Signal")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ds_t, y=ds_filt, mode='lines', 
                                  line=dict(color='green', width=1.5), name='Filtered'))
        fig2.update_layout(height=400, xaxis_title="Time (s)", yaxis_title="Amplitude",
                          template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
        
    else:
        st.info("Please upload a PPG file (.csv) from the sidebar")
#dwt
with tab2:
    st.header("Discrete Wavelet Transform (DWT) Analysis")
    
    if st.session_state.data_loaded:
        if st.button("Compute DWT", type="primary"):
            with st.spinner("Computing DWT scale features..."):
                w2fb, qj = compute_dwt_scale_features(
                    st.session_state.downsampled_signal, 
                    st.session_state.downsampled_fs, 
                    max_scale=max_scale
                )
                st.session_state.w2fb = w2fb
                st.session_state.qj = qj
                st.success("DWT computation completed!")
        
        if st.session_state.w2fb is not None:
            st.subheader("DWT Decomposition")
            
            num_scales = max(st.session_state.w2fb.keys())
            rows = (num_scales + 1) // 2
            
            fig = make_subplots(rows=rows, cols=2, 
                               subplot_titles=[f"Scale {i}" for i in range(1, num_scales+1)])
            
            for i, scale in enumerate(range(1, num_scales + 1)):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                if scale in st.session_state.w2fb and len(st.session_state.w2fb[scale]) > 0:
                    valid_length = min(len(st.session_state.downsampled_t), 
                                      len(st.session_state.w2fb[scale]))
                    fig.add_trace(
                        go.Scatter(x=st.session_state.downsampled_t[:valid_length],
                                  y=st.session_state.w2fb[scale][:valid_length],
                                  mode='lines', name=f'Scale {scale}',
                                  line=dict(width=1.5)),
                        row=row, col=col
                    )
            
            fig.update_xaxes(title_text="Time (s)")
            fig.update_yaxes(title_text="Amplitude")
            fig.update_layout(height=rows*300, showlegend=False, template="plotly_white",
                             title_text="DWT Feature Extraction")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("DWT Delay Table")
            delay_data = []
            for j in range(1, max_scale + 1):
                T = (2**(j - 1)) - 1
                delay_data.append({
                    "Scale (j)": j,
                    "Delay T (samples)": T,
                    "Delay (seconds)": f"{T / st.session_state.downsampled_fs:.4f}"
                })
            df_delay = pd.DataFrame(delay_data)
            st.dataframe(df_delay, use_container_width=True)
            
        else:
            st.info("start analysis")
    else:
        st.warning(" Please upload data in the 'Input Data' tab first")

#freq response
with tab3:
    st.header("Frequency Response Analysis")
    
    if st.session_state.data_loaded:
        fs = st.session_state.downsampled_fs
        
        Q, i_list = compute_frequency_response(fs, max_scale)
  
        st.subheader("DWT Frequency Response")
        fig_freq = go.Figure()
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for level in range(1, max_scale+1):
            fig_freq.add_trace(go.Scatter(x=i_list, y=Q[level], mode='lines',
                                         name=f'Q{level}', line=dict(width=2, color=colors[level-1])))
        
        fig_freq.update_layout(height=500, xaxis_title="Frequency (Hz)", 
                              yaxis_title="Magnitude", template="plotly_white",
                              title="Frequency Response of DWT Filters")
        st.plotly_chart(fig_freq, use_container_width=True)
        
        st.subheader("Frequency Range & Bandwidth Table")
        freq_data = []
        for j in range(1, max_scale + 1):
            f_min = fs / (2**(j + 1))
            f_max = fs / (2**j)
            bandwidth = f_max - f_min
            freq_data.append({
                "Scale": j,
                "Min Freq (Hz)": round(f_min, 3),
                "Max Freq (Hz)": round(f_max, 3),
                "Bandwidth (Hz)": round(bandwidth, 3)
            })
        
        df_freq = pd.DataFrame(freq_data)
        st.dataframe(df_freq, use_container_width=True)
        
    else:
        st.warning(" Please upload data in the 'Input Data' tab first")

#p2p selected scale
with tab4:
    st.header("MAV & Peak-to-Peak Detection (DWT)")
    
    if st.session_state.data_loaded and st.session_state.w2fb is not None:
        
        st.sidebar.header(" DWT Peak Detection Settings")
        scale_for_detection = st.sidebar.selectbox("Select Scale for Detection", 
                                                    list(range(1, max_scale+1)), 
                                                    index=6)  # Default Scale 7
        mav_window_dwt = st.sidebar.slider("MAV Window Size (DWT)", 5, 50, 20, key="mav_dwt")
        
        if scale_for_detection in st.session_state.w2fb and len(st.session_state.w2fb[scale_for_detection]) > 0:
            coeffs = st.session_state.w2fb[scale_for_detection][:len(st.session_state.downsampled_t)]
            N = len(coeffs)
            m = mav_window_dwt
            
            # Forward 
            y_forward = np.zeros(N)
            for i in range(N):
                temp = 0
                for j in range(m):
                    if i - j >= 0:
                        temp += coeffs[i - j]
                y_forward[i] = temp / m
            
            # Backward
            y_backward = np.zeros(N)
            for i in reversed(range(N)):
                temp = 0
                for j in range(m):
                    if i + j < N:
                        temp += y_forward[i + j]
                y_backward[i] = temp / m
            
            st.subheader(f"Scale {scale_for_detection} with MAV Smoothing")
            fig_mav = go.Figure()
            fig_mav.add_trace(go.Scatter(x=st.session_state.downsampled_t, y=coeffs,
                                        mode='lines', name=f'Scale {scale_for_detection}',
                                        line=dict(color='lightblue', width=1), opacity=0.6))
            fig_mav.add_trace(go.Scatter(x=st.session_state.downsampled_t, y=y_backward,
                                        mode='lines', name='MAV Smoothed',
                                        line=dict(color='red', width=2)))
            fig_mav.update_layout(height=400, xaxis_title="Time (s)", yaxis_title="Amplitude",
                                 template="plotly_white")
            st.plotly_chart(fig_mav, use_container_width=True)
            
            st.subheader("Peak-to-Peak Detection")
            signal_dwt = y_backward
            peaks_dwt, valleys_dwt = [], []
            
            for i in range(1, N - 1):
                if signal_dwt[i] > signal_dwt[i - 1] and signal_dwt[i] > signal_dwt[i + 1]:
                    peaks_dwt.append(i)
                elif signal_dwt[i] < signal_dwt[i - 1] and signal_dwt[i] < signal_dwt[i + 1]:
                    valleys_dwt.append(i)
        
            p2p_values_dwt = []
            p2p_intervals_dwt = []
            pi, vi = 0, 0
            while pi < len(peaks_dwt) and vi < len(valleys_dwt):
                if valleys_dwt[vi] > peaks_dwt[pi]:
                    p2p = signal_dwt[peaks_dwt[pi]] - signal_dwt[valleys_dwt[vi]]
                    p2p_values_dwt.append(p2p)
                    p2p_intervals_dwt.append((peaks_dwt[pi], valleys_dwt[vi]))
                    pi += 1
                    vi += 1
                else:
                    vi += 1
            
            fig_peaks = go.Figure()
            fig_peaks.add_trace(go.Scatter(x=st.session_state.downsampled_t, y=signal_dwt,
                                          mode='lines', name='MAV Signal',
                                          line=dict(color='blue', width=2)))
            fig_peaks.add_trace(go.Scatter(x=st.session_state.downsampled_t[peaks_dwt], 
                                          y=signal_dwt[peaks_dwt],
                                          mode='markers', name='Peaks',
                                          marker=dict(color='red', size=8, symbol='circle')))
            
            fig_peaks.update_layout(height=500, xaxis_title="Time (s)", yaxis_title="Amplitude",
                                   template="plotly_white", title="Peak Detection (DWT)")
            st.plotly_chart(fig_peaks, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Peaks", len(peaks_dwt))
            with col2:
                st.metric("P2P Intervals", len(p2p_values_dwt))
            
            if len(p2p_values_dwt) > 0:
                st.subheader("Peak-to-Peak Values (DWT)")
                p2p_df_dwt = pd.DataFrame({
                    "P2P #": range(1, len(p2p_values_dwt) + 1),
                    "Amplitude": [f"{val:.6f}" for val in p2p_values_dwt],
                    "Peak Index": [interval[0] for interval in p2p_intervals_dwt],
                    "Time (s)": [f"{st.session_state.downsampled_t[interval[0]]:.2f}" for interval in p2p_intervals_dwt]
                })
                st.dataframe(p2p_df_dwt, use_container_width=True)
                
                #respiratory rate
                if len(peaks_dwt) > 1:
                    peak_times = st.session_state.downsampled_t[peaks_dwt]
                    intervals = np.diff(peak_times)
                    avg_interval = np.mean(intervals)
                    breaths_per_minute_dwt = 60 / avg_interval
                else:
                    breaths_per_minute_dwt = 0
                
                st.subheader("Statistics (DWT)")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Mean P2P", f"{np.mean(p2p_values_dwt):.6f}")
                with col2:
                    st.metric("Std Dev", f"{np.std(p2p_values_dwt):.6f}")
                with col3:
                    st.metric("Min P2P", f"{np.min(p2p_values_dwt):.6f}")
                with col4:
                    st.metric("Max P2P", f"{np.max(p2p_values_dwt):.6f}")
                with col5:
                    st.metric("BRPM", f"{breaths_per_minute_dwt:.2f}")
            else:
                st.warning("No peak-to-peak intervals detected. Try adjusting the MAV window size.")
        else:
            st.error(f"Scale {scale_for_detection} is not available. Please compute DWT first.")
            
    elif st.session_state.data_loaded and st.session_state.w2fb is None:
        st.warning(" Please compute DWT in the 'DWT Analysis' tab first")
    else:
        st.warning(" Please upload data in the 'Input Data' tab first")
 
#p2p bpf        
with tab5:
    st.header("BPF Signal Peaks")
    
    if st.session_state.data_loaded and st.session_state.w2fb:
        
        st.sidebar.header("BPF Peak Detection Settings")
        bpf_lowcut = st.sidebar.number_input("Low Cutoff (Hz)", value=0.1, step=0.05, min_value=0.05, key="bpf_low")
        bpf_highcut = st.sidebar.number_input("High Cutoff (Hz)", value=0.5, step=0.05, min_value=0.1, key="bpf_high")
        st.subheader(f"Band-Pass Filtered Signal ({bpf_lowcut}-{bpf_highcut} Hz)")
        bpf_signal = bandpass_filter(
            st.session_state.downsampled_signal,
            st.session_state.downsampled_fs,
            lowcut=bpf_lowcut,
            highcut=bpf_highcut
        )
        fig_bpf = go.Figure()
        fig_bpf.add_trace(go.Scatter(x=st.session_state.downsampled_t, y=st.session_state.downsampled_signal,
                                     mode='lines', name='Original',
                                     line=dict(color='lightgray', width=1), opacity=0.5))
        fig_bpf.add_trace(go.Scatter(x=st.session_state.downsampled_t, y=bpf_signal,
                                     mode='lines', name='BPF Signal',
                                     line=dict(color='purple', width=2)))
        fig_bpf.update_layout(height=400, xaxis_title="Time (s)", yaxis_title="Amplitude",
                             template="plotly_white")
        st.plotly_chart(fig_bpf, use_container_width=True)
        

        st.subheader("Peak-to-Peak Detection")
        peaks_bpf, valleys_bpf = [], []
        
        for i in range(1, N - 1):
            if bpf_signal[i] > bpf_signal[i - 1] and bpf_signal[i] > bpf_signal[i + 1]:
                peaks_bpf.append(i)
            elif bpf_signal[i] < bpf_signal[i - 1] and bpf_signal[i] < bpf_signal[i + 1]:
                valleys_bpf.append(i)
        
        p2p_values_bpf = []
        p2p_intervals_bpf = []
        pi, vi = 0, 0
        while pi < len(peaks_bpf) and vi < len(valleys_bpf):
            if valleys_bpf[vi] > peaks_bpf[pi]:
                p2p = bpf_signal[peaks_bpf[pi]] - bpf_signal[valleys_bpf[vi]]
                p2p_values_bpf.append(p2p)
                p2p_intervals_bpf.append((peaks_bpf[pi], valleys_bpf[vi]))
                pi += 1
                vi += 1
            else:
                vi += 1
         
        fig_peaks_bpf = go.Figure()
        fig_peaks_bpf.add_trace(go.Scatter(x=st.session_state.downsampled_t, y=bpf_signal,
                                          mode='lines', name='BPF Signal',
                                          line=dict(color='blue', width=2)))
        fig_peaks_bpf.add_trace(go.Scatter(x=st.session_state.downsampled_t[peaks_bpf], 
                                          y=bpf_signal[peaks_bpf],
                                          mode='markers', name='Peaks',
                                          marker=dict(color='red', size=8, symbol='circle')))
        
        fig_peaks_bpf.update_layout(height=500, xaxis_title="Time (s)", yaxis_title="Amplitude",
                                   template="plotly_white", title="Peak Detection (BPF)")
        st.plotly_chart(fig_peaks_bpf, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Peaks", len(peaks_bpf))
        with col2:
            st.metric("P2P Intervals", len(p2p_values_bpf))
        
        if len(p2p_values_bpf) > 0:
            st.subheader("Peak-to-Peak Values (BPF)")
            p2p_df_bpf = pd.DataFrame({
                "P2P #": range(1, len(p2p_values_bpf) + 1),
                "Amplitude": [f"{val:.6f}" for val in p2p_values_bpf],
                "Peak Index": [interval[0] for interval in p2p_intervals_bpf],
                "Time (s)": [f"{st.session_state.downsampled_t[interval[0]]:.2f}" for interval in p2p_intervals_bpf]
            })
            st.dataframe(p2p_df_bpf, use_container_width=True)
            
            # Respiratory rate
            if len(peaks_bpf) > 1:
                peak_times = st.session_state.downsampled_t[peaks_bpf]
                intervals = np.diff(peak_times)
                avg_interval = np.mean(intervals)
                breaths_per_minute_bpf = 60 / avg_interval
            else:
                breaths_per_minute_bpf = 0
            
    else:
        st.warning("Please Compute DWT")

st.divider()