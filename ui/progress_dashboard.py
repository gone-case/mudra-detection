import streamlit as st
import pandas as pd
from database.schema import SessionLocal, UserProgress, Mudra, PracticeSession
import matplotlib.pyplot as plt

def render_dashboard():
    st.header("📊 Progress Dashboard")
    st.markdown("Track your learning curve and practice sessions.")
    
    db = SessionLocal()
    try:
        user_id = 1 # hardcoded single user for now
        
        # 1. High-level Stats
        sessions = db.query(PracticeSession).filter(PracticeSession.user_id == user_id).all()
        total_sessions = len(sessions)
        avg_acc = sum([s.overall_accuracy for s in sessions]) / max(1, total_sessions)
        total_practice_time = sum([s.duration_minutes for s in sessions])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sessions", total_sessions)
        col2.metric("Overall Accuracy", f"{avg_acc:.1f}%")
        col3.metric("Practice Time (min)", f"{total_practice_time:.1f}")
        
        st.markdown("---")
        
        # 2. Mudra Mastery Metrics
        st.subheader("Mudra Mastery")
        progress = db.query(UserProgress).filter(UserProgress.user_id == user_id).all()
        
        if not progress:
            st.info("No practice data yet. Head over to the Live Practice tab to start learning!")
            return
            
        data = []
        for p in progress:
            mudra = db.query(Mudra).filter(Mudra.id == p.mudra_id).first()
            if mudra:
                acc = (p.correct_attempts / max(1, p.total_attempts)) * 100
                data.append({
                    "Mudra": mudra.name,
                    "Total Attempts": p.total_attempts,
                    "Accuracy (%)": round(acc, 2),
                    "Avg Confidence": round(p.avg_confidence, 2)
                })
                
        df = pd.DataFrame(data).sort_values("Accuracy (%)", ascending=False)
        st.dataframe(df, use_container_width=True)
        
        # 3. Bar Chart
        st.subheader("Accuracy by Mudra")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(df["Mudra"][:10], df["Accuracy (%)"][:10], color='skyblue')
        plt.xticks(rotation=45)
        plt.ylabel("Accuracy (%)")
        st.pyplot(fig)
        
    finally:
        db.close()
