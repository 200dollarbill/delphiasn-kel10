import streamlit as st
from ldpage import loadedData

#ini buat plot data awal + baseline correction

st.header("Page1 ")
st.write("haloaadadad")
var=[1,2,3]
st.text(loadedData.parse(var))