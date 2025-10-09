import streamlit as st

page_dict = {}
st.title('Kelompok 10 ')

# if "page" not in st.session_state:
#     st.session_state.page = None

# active sessions
#setup sessions by buttons

# st.button("button1")
# if st.button("button1"):
#     st.session_state.page = pages[0]
#     st.rerun()

# st.button("button2")


# if st.button("button2"):
#     st.session_state.page = pages[1]
#     st.rerun()

# activepage = st.session_state.page 


# setup page

ldPage = st.Page("pages/ldpage.py", title="Data Loading")
preProcessPage = st.Page("pages/2preProcess.py", title="Pre Processing")
DWTResultsPage = st.Page("pages/4dwtResults.py", title="DWT Results")
dwtPage = st.Page("pages/3dwtFResponse.py", title="DWT Filter Response")
thresholdingPage = st.Page("pages/5thresholding.py", title="Thresholding & Respiratory rate")
peakDPage = st.Page("pages/6peakdet.py", title="Peak Detection")


dataLoadingPage = [ldPage, preProcessPage]
processingPages = [DWTResultsPage, dwtPage, thresholdingPage, peakDPage]

# nav page dictionary
mainNavigation = st.navigation({"Load Data": dataLoadingPage, "Processing:": processingPages})
# pg1 = st.navigation({"Processing:": processingPages})
# pg1.run()
mainNavigation.run()


