import streamlit as st
from progress_tracker.improvement_engine import ImprovementEngine

def render_recommendations():
    st.header("💡 Personalized Recommendations")
    st.markdown("AI-generated tips based on your recent practice sessions.")
    
    engine = ImprovementEngine(user_id=1)
    
    st.subheader("Your Weakest Mudras")
    weak_mudras = engine.get_weak_mudras(limit=3)
    
    if not weak_mudras:
        st.success("No data available yet. Please complete a practice session!")
        return
        
    for idx, wm in enumerate(weak_mudras):
        with st.expander(f"🔹 {wm['mudra_name']} (Accuracy: {wm['accuracy']:.1f}%)", expanded=(idx==0)):
            st.warning(f"**Common Mistake:** {wm['common_mistakes']}")
            st.info(f"**How to Fix:** {wm['improvement_tips']}")
            
    st.markdown("---")
    st.subheader("Action Plan")
    suggestions = engine.generate_recommendations()
    
    for sug in suggestions:
        st.markdown(sug)
        
    st.markdown("---")
    st.button("Refresh Recommendations")
