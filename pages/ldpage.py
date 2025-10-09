import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from deps import handler
#ini buat load data aja

st.header("Load Data Page")
st.write("halo")


# button loading data bidmc
if st.button("Load Data", key="LOADDATAKEY"):
    df = pd.read_csv('ASN-Kel10/data/bidmc_03_Signals.csv')
    fig, ax = plt.subplots()
    ax.plot(df['Time [s]'], df[' PLETH'])
    ax.set_title('Data from CSV File')
    ax.set_xlabel('Time')
    ax.set_ylabel('PLETH')
    ax.grid(True)
    st.pyplot(fig)
    handler.save(df['Time [s]'], df[' PLETH'], "rawdata")





