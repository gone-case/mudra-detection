"""
app_landmark.py
---------------
Streamlit entry point for the Landmark Ensemble version of MudraNet.

Run with:
    python -m streamlit run app_landmark.py
"""

import streamlit as st

st.set_page_config(
    page_title="MudraNet — Landmark Ensemble",
    page_icon="🪷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🪷 MudraNet AI")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Live Practice", "Progress Dashboard", "Recommendations"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Classifier:** 4-model ensemble\n\n"
    "SVM · Random Forest · KNN · Gradient Boosting\n\n"
    "Features: MediaPipe 21 hand landmarks (63-d)"
)

# ---- Pages ----

if page == "Home":
    st.title("Welcome to MudraNet AI 🪷")
    st.markdown("""
    ### Real-Time Bharatanatyam Mudra Identification System

    This platform identifies your hand gestures using a **4-model ensemble**
    trained on **MediaPipe hand landmarks** — no image cropping required,
    just pure geometric features from 21 keypoints.

    | Model | Why included |
    |---|---|
    | **SVM (RBF)** | Excellent on high-dimensional, well-separated feature spaces |
    | **Random Forest** | Handles feature interaction, robust to noise |
    | **K-Nearest Neighbours** | Instance-based, great when classes are compact |
    | **Gradient Boosting** | Captures subtle non-linear decision boundaries |

    All 4 predictions are **soft-voted** (average probabilities) for the final result.

    ---

    > **First time?** Train models by running:
    > ```
    > python -m training.train_landmark_ensemble
    > ```
    > Then come back and open **Live Practice**.

    👈 Select a page from the sidebar to begin!
    """)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Bharatanatyam_Hasta_Mudras_1.jpg/800px-Bharatanatyam_Hasta_Mudras_1.jpg",
        width=600
    )

elif page == "Live Practice":
    from ui.live_practice_landmark import render_live_practice_landmark
    render_live_practice_landmark()

elif page == "Progress Dashboard":
    from ui.progress_dashboard import render_dashboard
    render_dashboard()

elif page == "Recommendations":
    from ui.recommendations import render_recommendations
    render_recommendations()
