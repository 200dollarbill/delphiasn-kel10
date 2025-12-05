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

firstpage = st.Page("pages/firstLoadPage.py", title="Data Loading")
secondpage = st.Page("pages/secondProcessRR.py", title="Pre Processing")
# br pages
thirdpage = st.Page("pages/thirdDWTResultsRR.py", title="DWT Results")
fourthpage = st.Page("pages/fourthDWTFResponseRR.py", title="DWT Filter Response")
fifthpage = st.Page("pages/fifthThresholdingBR.py", title="Thresholding")
ninthpage = st.Page("pages/ninthPeakDetBR.py", title="HR Peak Detection")
tenthpage = st.Page("pages/tenthBRPMCalc.py", title="Breath Rate Calculation")
# HR Analysis Pages
sixthpage = st.Page("pages/sixthTachogramRR.py", title="RR Tachogram")
seventhpage = st.Page("pages/seventFDAnalysisHRV.py", title="Frequency Domain Analysis")
eightpage = st.Page("pages/eightIntervalHRV.py", title="Interval Variability")
eleventhpage = st.Page("pages/eleventhVasomotor.py", title="Vasomotor Frequency Analysis")
summarypage = st.Page("pages/twelvthSummary.py", title = "Conclusion")
emdpage = st.Page("pages/thirteenthEMD.py")


dataLoadingPage = [firstpage, secondpage]
processingPages = [ninthpage, sixthpage, seventhpage, eightpage]
brPages = [thirdpage, fourthpage, fifthpage,tenthpage]
vasoPages = [eleventhpage, emdpage]
finalPages = [summarypage]
# nav page dictionary
mainNavigation = st.navigation({"Load Data": dataLoadingPage, "HR Analysis:": processingPages, "BR Analysis" : brPages, "Vasomotor Analysis" : vasoPages, "Summary" : finalPages})
# pg1 = st.navigation({"Processing:": processingPages})
# pg1.run()
mainNavigation.run()


