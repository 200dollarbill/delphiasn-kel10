import streamlit as st
import pandas as pd
import plotly.express as px
import os
import pickle
from deps import handler


st.set_page_config(layout="wide")
# st.title("Kelompok 10")
st.header("Data Loading and Initial Visualization")
# st.write("Accpets CSV")

session_name = st.text_input(
    "Enter a session name for the data",
    value="raw_data_session",
    help="This name will be used to save the processed data as a `.dat` file."
)

uploaded_file = st.file_uploader(
    "CSV file",
    type="csv",
    help="Upload a CSV file with 'Index' and 'Amplitude (0-4096)' columns."
)

if st.button("Load and Process Data", key="LOAD_PROCESS_KEY"):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['Index', 'Amplitude (0-4096)']
            if not all(col in df.columns for col in required_columns):
                st.error(f"CSV must contain the following columns: {', '.join(required_columns)}")
            else:
                st.info(f"Successfully loaded `{uploaded_file.name}` with {df.shape[0]} rows.")

                st.write("### Data Visualization")
                fig = px.line(
                    df,
                    x='Index',
                    y='Amplitude (0-4096)',
                    title='ECG Data Visualization',
                    labels={'Index': 'Time (Index)', 'Amplitude (0-4096)': 'Amplitude'}
                )
                fig.update_traces(line_color='#00A9FF')
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Data"
                )
                st.plotly_chart(fig, use_container_width=True)

                handler.save(df['Index'], df['Amplitude (0-4096)'], f"data/{session_name}")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.warning("Please upload a CSV file first.")