import streamlit as st

st.set_page_config(
    page_title="MudraNet - Bharatanatyam AI",
    page_icon="🪷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Navigation
st.sidebar.title("🪷 MudraNet AI")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Live Practice", "Progress Dashboard", "Recommendations"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "MudraNet tracks your mudras in real-time, "
    "calculates accuracy, and provides AI-tailored feedback."
)

# Route to pages
if page == "Home":
    st.title("Welcome to MudraNet AI 🪷")
    st.markdown("""
    ### Real-Time Bharatanatyam Mudra Identification System
    This platform helps you perfect your hand gestures (Mudras) through:
    
    1. **Live Practice:** Practice in front of your webcam and get instant, real-time AI feedback powered by an ensemble of deep learning models.
    2. **Progress Board:** View your accuracy graphs over time and see which mudras need the most attention.
    3. **Actionable Suggestions:** Receive tailored AI recommendations based on your unique mistakes and session history.
    
    👈 Select a page from the sidebar to begin!
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Bharatanatyam_Hasta_Mudras_1.jpg/800px-Bharatanatyam_Hasta_Mudras_1.jpg", width=600)

elif page == "Live Practice":
    from ui.live_practice import render_live_practice
    render_live_practice()

elif page == "Progress Dashboard":
    from ui.progress_dashboard import render_dashboard
    render_dashboard()

elif page == "Recommendations":
    from ui.recommendations import render_recommendations
    render_recommendations()
