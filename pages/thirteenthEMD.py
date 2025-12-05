import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from deps import handler, updater
st.set_page_config(page_title="EMD Analysis", layout="wide")
from scipy import interpolate
from scipy.fft import fft, fftfreq

def find_extrema(signal):
    maxima = []
    minima = []
    N = len(signal)
    for i in range(1, N - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            maxima.append(i)
        elif signal[i] < signal[i-1] and signal[i] < signal[i+1]:
            minima.append(i)
    return np.array(maxima), np.array(minima)

def get_envelopes(signal, maxima, minima):
    t = np.arange(len(signal))
    if len(maxima) < 2 or len(minima) < 2:
        return None, None
    
    max_prepend = 2 * maxima[0] - maxima[1] if maxima[0] > 0 else 0
    max_append = 2 * maxima[-1] - maxima[-2] if maxima[-1] < len(signal)-1 else len(signal)-1
    min_prepend = 2 * minima[0] - minima[1] if minima[0] > 0 else 0
    min_append = 2 * minima[-1] - minima[-2] if minima[-1] < len(signal)-1 else len(signal)-1
    
    ext_max_idx = np.concatenate([[max_prepend], maxima, [max_append]])
    ext_max_val = np.concatenate([[signal[maxima[0]]], signal[maxima], [signal[maxima[-1]]]])
    ext_min_idx = np.concatenate([[min_prepend], minima, [min_append]])
    ext_min_val = np.concatenate([[signal[minima[0]]], signal[minima], [signal[minima[-1]]]])
    
    try:
        upper_envelope_func = interpolate.CubicSpline(ext_max_idx, ext_max_val, bc_type='natural')
        lower_envelope_func = interpolate.CubicSpline(ext_min_idx, ext_min_val, bc_type='natural')
        return upper_envelope_func(t), lower_envelope_func(t)
    except:
        return None, None

def sift(signal, max_iter=10, sd_threshold=0.3):
    h = signal.copy()
    for iteration in range(max_iter):
        h_prev = h.copy()
        maxima, minima = find_extrema(h)
        if len(maxima) < 2 or len(minima) < 2:
            break
        upper_env, lower_env = get_envelopes(h, maxima, minima)
        if upper_env is None or lower_env is None:
            break
        mean_env = (upper_env + lower_env) / 2
        h = h - mean_env
        if iteration > 0:
            sd = np.sum(((h_prev - h) ** 2) / (h_prev ** 2 + 1e-10))
            if sd < sd_threshold:
                break
    return h

def emd(signal, max_imfs=10, max_iter=10, sd_threshold=0.3):
    imfs = []
    residual = signal.copy()
    for i in range(max_imfs):
        maxima, minima = find_extrema(residual)
        if len(maxima) < 2 or len(minima) < 2:
            break
        imf = sift(residual, max_iter=max_iter, sd_threshold=sd_threshold)
        imfs.append(imf)
        residual = residual - imf
        if np.std(residual) < 0.001 * np.std(signal):
            break
    return imfs, residual


def compute_dominant_frequency(imf, fs):
    N = len(imf)
    yf = fft(imf)
    xf = fftfreq(N, 1/fs)
    positive_freq_idx = xf > 0
    xf_positive = xf[positive_freq_idx]
    yf_positive = np.abs(yf[positive_freq_idx])
    dominant_idx = np.argmax(yf_positive)
    dominant_freq = xf_positive[dominant_idx]
    return dominant_freq, xf_positive, yf_positive

def analyze_imf(imf, fs):
    maxima, minima = find_extrema(imf)
    upper_env, lower_env = get_envelopes(imf, maxima, minima)
    mean_env = None
    if upper_env is not None and lower_env is not None:
        mean_env = (upper_env + lower_env) / 2
    
    dominant_freq, freq_spectrum, magnitude_spectrum = compute_dominant_frequency(imf, fs)
    
    energy = np.sum(imf ** 2)
    std_dev = np.std(imf)
    mean_val = np.mean(imf)
    max_amp = np.max(np.abs(imf))
    zero_crossings = np.sum(np.diff(np.sign(imf)) != 0)
    
    return {
        'maxima': maxima, 'minima': minima,
        'upper_envelope': upper_env, 'lower_envelope': lower_env, 'mean_envelope': mean_env,
        'dominant_freq': dominant_freq, 'freq_spectrum': freq_spectrum, 'magnitude_spectrum': magnitude_spectrum,
         'std_dev': std_dev, 'mean': mean_val, 'max_amplitude': max_amp,
        'zero_crossings': zero_crossings, 'num_maxima': len(maxima), 'num_minima': len(minima)
    }

def select_respiratory_imf(imfs, fs, expected_range=(0.1, 0.5)):
    best_score = 0
    best_idx = 0
    best_rr = 0
    
    for idx, imf in enumerate(imfs):
        dominant_freq, _, magnitude_spectrum = compute_dominant_frequency(imf, fs)
        if expected_range[0] <= dominant_freq <= expected_range[1]:
            score = np.max(magnitude_spectrum)
        else:
            score = 0
        
        if score > best_score:
            best_score = score
            best_idx = idx
            best_rr = dominant_freq * 60
    
    return best_idx, best_rr

st.title("Empirical Mode Decomposition (EMD) Analysis")
st.markdown("---")

st.sidebar.header("EMD Settings")

session_name = st.sidebar.text_input(
    "Session Name",
    value="raw_data_session",
    help="Enter the session name to load processed data"
)

max_imfs_emd = st.sidebar.slider("Max number of IMFs", 3, 10, 8, key="max_imfs_emd")
max_iter_emd = st.sidebar.slider("Max iterations per sifting", 5, 50, 10, key="max_iter_emd")
sd_threshold_emd = st.sidebar.slider("SD threshold", 0.1, 0.5, 0.3, step=0.05, key="sd_threshold_emd")

st.sidebar.subheader("Respiratory Frequency Range")
min_resp_freq = st.sidebar.number_input("Min Respiratory Freq (Hz)", 0.05, 0.5, 0.15, step=0.05, key="min_resp_freq")
max_resp_freq = st.sidebar.number_input("Max Respiratory Freq (Hz)", 0.1, 1.0, 0.40, step=0.05, key="max_resp_freq")

if 'emd_computed' not in st.session_state:
    st.session_state.emd_computed = False
if 'imfs_emd' not in st.session_state:
    st.session_state.imfs_emd = None

if st.button("Load Data & Compute EMD", type="primary", use_container_width=True):
    try:
        with st.spinner("Loading data..."):
            data_obj = handler.load(f"data/{session_name}")
            time = data_obj.time
            signal = data_obj.value
            
            dt = np.mean(np.diff(time))
            fs = 50
            
            st.session_state.time = time
            st.session_state.signal = signal
            st.session_state.fs = fs
            
        st.success(f"Length: {len(signal)}, Fs: {fs:.2f} Hz")
        
        # Compute EMD
        with st.spinner("Running EMD decomposition..."):
            imfs, residual = emd(signal, max_imfs=max_imfs_emd, max_iter=max_iter_emd, sd_threshold=sd_threshold_emd)
            
            imf_analyses = []
            for imf in imfs:
                analysis = analyze_imf(imf, fs)
                imf_analyses.append(analysis)
            
            st.session_state.imfs_emd = imfs
            st.session_state.residual_emd = residual
            st.session_state.imf_analyses = imf_analyses
            st.session_state.emd_computed = True
            
        st.success(f"EMD completed! Extracted {len(imfs)} IMFs")
        
    except FileNotFoundError:
        st.error(f"Session '{session_name}' not found! Please load data first in the Data Loading page.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

if st.session_state.emd_computed:
    imfs = st.session_state.imfs_emd
    residual = st.session_state.residual_emd
    imf_analyses = st.session_state.imf_analyses
    time = st.session_state.time
    fs = st.session_state.fs
    
    # Metrics
    # total_energy = sum([a['energy'] for a in imf_analyses]) + np.sum(residual**2)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total IMFs", len(imfs))
    with col2:
        st.metric("Signal Length", len(time))
    with col3:
        st.metric("Sampling Rate", f"{fs:.2f} Hz")
    with col4:
        # st.metric("Total Energy", f"{total_energy:.2f}")
        pass

    st.markdown("---")
    
    st.subheader("IMF Analysis")
    
    for i, (imf, analysis) in enumerate(zip(imfs, imf_analyses)):
        with st.expander(f"**IMF {i+1}** - Dominant Frequency: **{analysis['dominant_freq']:.4f} Hz**", expanded=(i==0)):
            
            # Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Dominant Freq", f"{analysis['dominant_freq']:.4f} Hz")
            with col2:
                pass
                # st.metric("Energy", f"{analysis['energy']:.2f}")
            with col3:
                st.metric("Std Dev", f"{analysis['std_dev']:.6f}")
            with col4:
                st.metric("Max Amplitude", f"{analysis['max_amplitude']:.6f}")
            with col5:
                pass
                # energy_pct = (analysis['energy'] / total_energy) * 100
                # st.metric("Energy %", f"{energy_pct:.2f}%")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Maxima", analysis['num_maxima'])
            with col2:
                st.metric("Minima", analysis['num_minima'])
            with col3:
                st.metric("Zero Crossings", analysis['zero_crossings'])
            
            st.markdown("---")
            
            # Plot IMF with Envelopes
            st.markdown(f"#### IMF {i+1} ")
            fig_env = go.Figure()
            
            fig_env.add_trace(go.Scatter(x=time, y=imf, mode='lines', name=f'IMF {i+1}',
                                        line=dict(color='blue', width=2)))
            
            if analysis['upper_envelope'] is not None:
                fig_env.add_trace(go.Scatter(x=time, y=analysis['upper_envelope'], mode='lines',
                                            name='Upper Envelope', line=dict(color='red', width=1, dash='dash')))
            if analysis['lower_envelope'] is not None:
                fig_env.add_trace(go.Scatter(x=time, y=analysis['lower_envelope'], mode='lines',
                                            name='Lower Envelope', line=dict(color='green', width=1, dash='dash')))
            if analysis['mean_envelope'] is not None:
                fig_env.add_trace(go.Scatter(x=time, y=analysis['mean_envelope'], mode='lines',
                                            name='Mean Envelope', line=dict(color='orange', width=1, dash='dot')))
            
            if len(analysis['maxima']) > 0:
                fig_env.add_trace(go.Scatter(x=time[analysis['maxima']], y=imf[analysis['maxima']],
                                            mode='markers', name='Maxima',
                                            marker=dict(color='red', size=6, symbol='triangle-up')))
            if len(analysis['minima']) > 0:
                fig_env.add_trace(go.Scatter(x=time[analysis['minima']], y=imf[analysis['minima']],
                                            mode='markers', name='Minima',
                                            marker=dict(color='green', size=6, symbol='triangle-down')))
            
            fig_env.update_layout(height=400, xaxis_title="Time (s)", yaxis_title="Amplitude",
                                 template="plotly_white", hovermode='x unified')
            st.plotly_chart(fig_env, use_container_width=True)
            
            st.markdown("---")
            
            st.markdown(f"#### FFT")
            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(x=analysis['freq_spectrum'], y=analysis['magnitude_spectrum'],
                                        mode='lines', name='Magnitude', line=dict(color='darkblue', width=2),
                                        fill='tozeroy'))
            
            fig_fft.add_vline(x=analysis['dominant_freq'], line_dash="dash", line_color="red",
                             annotation_text=f"Dominant: {analysis['dominant_freq']:.4f} Hz", annotation_position="top")
            
            fig_fft.add_vrect(x0=min_resp_freq, x1=max_resp_freq, fillcolor="lightgreen", opacity=0.2,
                             layer="below", line_width=0, annotation_text="Respiratory", annotation_position="top left")
            
            fig_fft.update_layout(height=350, xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
                                 template="plotly_white", showlegend=False)
            st.plotly_chart(fig_fft, use_container_width=True)
            
            dominant_freq = analysis['dominant_freq']
            if min_resp_freq <= dominant_freq <= max_resp_freq:
                rr_bpm = dominant_freq * 60
                st.success(f"**Respiratory Rate**: {rr_bpm:.2f} BPM (f_peak = {dominant_freq:.3f} Hz)")
            else:
                st.warning(f"Dominant frequency ({dominant_freq:.3f} Hz) is outside respiratory range")
    
    st.markdown("---")
    
    st.subheader("Residual")
    with st.expander("**Residual Component**", expanded=False):
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=time, y=residual, mode='lines', name='Residual',
                                     line=dict(color='darkred', width=2)))
        fig_res.update_layout(height=300, xaxis_title="Time (s)", yaxis_title="Amplitude", template="plotly_white")
        st.plotly_chart(fig_res, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"{np.mean(residual):.6f}")
        with col2:
            st.metric("Std Dev", f"{np.std(residual):.6f}")
        with col3:
            residual_energy = np.sum(residual**2)
            # residual_pct = (residual_energy / total_energy) * 100
            # st.metric("Energy %", f"{residual_pct:.2f}%")
    
    st.markdown("---")
    
    st.subheader(" Respiratory Rate Detection")
    
    best_imf_idx, auto_rr = select_respiratory_imf(imfs, fs, expected_range=(min_resp_freq, max_resp_freq))
    best_fpeak = imf_analyses[best_imf_idx]['dominant_freq']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best IMF", f"IMF {best_imf_idx + 1}")
    with col2:
        st.metric("f_peak (Hz)", f"{best_fpeak:.3f}")
    with col3:
        st.metric("RR (BPM)", f"{auto_rr:.2f}")
    with col4:
        period = 1.0 / best_fpeak if best_fpeak > 0 else 0
        st.metric("Period (s)", f"{period:.2f}")
    
    # Save RR to temp
    if st.button("Save Respiratory Rate", key="save_rr"):
        updater.save(auto_rr, "emd_respiratory_rate")
        st.success(f"Respiratory Rate ({auto_rr:.2f} BPM) saved to temp!")

else:
    st.info("Click 'Load Data & Compute EMD' to start analysis")
