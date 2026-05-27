"""
app_svm.py
----------
Streamlit entry point exclusively for the high-accuracy SVM Landmark model.

Run with:
    python -m streamlit run app_svm.py
"""

import streamlit as st

st.set_page_config(
    page_title="MudraNet — SVM Native",
    page_icon="🪷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🪷 MudraNet AI")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Live Practice (SVM)", "Progress Dashboard", "Recommendations"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Classifier:** Pure SVM (RBF)\n\n"
    "Optimized for maximum speed and zero ensemble dilution.\n\n"
    "Features: MediaPipe 21 hand landmarks (63-d)"
)

# ---- Pages ----

if page == "Home":
    st.title("Welcome to MudraNet AI 🪷 (SVM Edition)")
    st.markdown("""
    ### Real-Time Bharatanatyam Mudra Identification System

    This platform identifies your hand gestures strictly using a **Support Vector Machine (SVM)**
    trained on **MediaPipe hand landmarks**.

    By removing the ensemble, this version achieves the highest possible accuracy by maintaining crisp, 
    strict mathematical boundaries without being diluted by less confident models.

    ---

    > **First time?** Train models by running:
    > ```
    > python -m training.train_landmark_ensemble
    > ```
    > Then come back and open **Live Practice (SVM)**.

    👈 Select a page from the sidebar to begin!
    """)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Bharatanatyam_Hasta_Mudras_1.jpg/800px-Bharatanatyam_Hasta_Mudras_1.jpg",
        width=600
    )

elif page == "Live Practice (SVM)":
    from ui.live_practice_landmark_svm import render_live_practice_svm
    render_live_practice_svm()

elif page == "Progress Dashboard":
    from ui.progress_dashboard import render_dashboard
    render_dashboard()

elif page == "Recommendations":
    from ui.recommendations import render_recommendations
    render_recommendations()
