import streamlit as st
import os
from deps import handler

st.header("Page1 ")
# st.write("udah ke load")
rawdata = handler.load("rawdata")


st.write(rawdata)