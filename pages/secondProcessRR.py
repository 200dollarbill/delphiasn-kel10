import streamlit as st
import os
from deps import handler

st.header("Page1 ")
# st.write("udah ke load")

# load pcg
pcgraw = handler.load("rawdata")
# load ecg
ecgraw = handler.load("ecgraw")

st.write(pcgraw)