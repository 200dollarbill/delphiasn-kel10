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
    df = pd.read_csv('data/ddddfffa2.csv')
    fig, ax = plt.subplots()
    ax.plot(df['Index'], df['Amplitude (0-4096)'])
    ax.set_title('Data from CSV File')
    ax.set_xlabel('Time')
    ax.set_ylabel('Data')
    ax.grid(True)
    st.pyplot(fig)
    handler.save(df['Index'], df['Amplitude (0-4096)'], "rawdata1")





