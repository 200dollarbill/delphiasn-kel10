import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
#ini buat load data aja

st.header("Load Data Page")
st.write("halo")

class loadedData():
    def __init__(self, data):
        self.data = data

    @staticmethod
    def parse(data):
        return data



if st.button("Load Data", key="LOADDATAKEY"):
    df = pd.read_csv('kuliah/data/bidmc_03_Signals.csv')
    fig, ax = plt.subplots()
    ax.plot(df['Time [s]'], df[' PLETH'])
    ax.set_title('Data from CSV File')
    ax.set_xlabel('Time')
    ax.set_ylabel('PLETH')
    ax.grid(True)
    st.pyplot(fig)


    arr = [20,30,40]
    st.text(loadedData.parse(arr))





