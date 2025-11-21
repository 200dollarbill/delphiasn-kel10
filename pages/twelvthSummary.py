import streamlit as st
import numpy as np
import pandas as pd
import math
import glob
import plotly.graph_objects as go
from scipy.fft import fft
import os
import pickle
from deps import handler, updater

DATA_DIR = './temp/'
FILE_EXTENSION = '.dat'

def delete_dat_files():
    search_pattern = os.path.join(DATA_DIR, f'*{FILE_EXTENSION}')
    files_to_delete = glob.glob(search_pattern)
    
    deleted_count = 0
    errors = []

    if not files_to_delete:
        st.info(f"No files ending with '{FILE_EXTENSION}' found to delete.")
        return
    with st.spinner(f"Deleting {len(files_to_delete)} files..."):
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                deleted_count += 1
            except OSError as e:
                errors.append(f"Failed to delete {file_path}: {e}")

    if deleted_count > 0:
        st.success(f"Successfully deleted {deleted_count} files!")
    if errors:
        st.error("Errors encountered during deletion:")
        for err in errors:
            st.code(err)

st.set_page_config(page_title="Final Results", layout="wide")
st.header("Section 4: Results")

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
    "maxVasoFreq":"Peak Vasometric Frequency",
    "maxVasoMag" : "Peak Vasometric Magnitude",
    "mean_br":"Mean Breath Rate",
    "mean_hr":"Mean Heart Rate"
}

st.subheader("Time-Domain HRV Metrics")
col_t1, col_t2, col_t3, col_t4 = st.columns(4)
time_domain_cols = [col_t1, col_t2, col_t3, col_t4]
time_domain_keys = ["mean_hr", "mean_br","cvsd", "hrv_ti", "nn50", "pnn50", "RMSSD", "sd1", "sd2", "SDNN", "sdratio", "SDSD", "tinn", "cvnn"]

for i, key in enumerate(time_domain_keys):
    col = time_domain_cols[i % 4]
    value_obj = updater.load(key)
    value = value_obj.value
    display_name = metric_map.get(key, key.upper()) 
    unit = "ms" if key in ["cvsd", "RMSSD", "sd1", "sd2", "SDNN", "SDSD", "tinn"] else ""
    unit = " Count" if key == "nn50" else unit
    unit = "%" if key == "pnn50" else unit
    unit = "BPM" if key in ["mean_hr", "mean_br"] else unit
    
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
    if key in ["hfTP", "lfTP", "TP"] : unit = "ms²" 
    if key in ["hfNU", "lfNU"]:unit = " n.u." 
    if key in ["peakLF", "peakHF", "lfhfratio"]:unit = "" 
    col.metric(display_name, f"{value:.4f}{unit}")

st.markdown("---") 
st.subheader("Other Metrics")
col_o1, col_o2, col_o3 = st.columns(3)
other_cols = [col_o1,col_o2,col_o3]
other_keys = ["skewness","maxVasoMag","maxVasoFreq"]

for i, key in enumerate(other_keys):
    col = other_cols[i%3]
    value_obj = updater.load(key)
    value = value_obj.value
    display_name = metric_map.get(key, key.upper())
    if key == "maxVasoFreq" : unit = " Hz" 
    if key == "maxVasoMag" : unit = " dB" 
    col.metric(display_name, f"{value:.2f}{unit}")


if st.button("Clean Readings"):
    delete_dat_files()
    st.rerun()
