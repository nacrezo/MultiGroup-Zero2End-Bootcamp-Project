"""
Streamlit web application for Gaming User Segmentation.
Provides a user-friendly interface for model inference.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.inference import predict_user_segment, predict_batch, get_segment_profile, UserSegmentationPipeline
from src.config import MODELS_DIR, BUSINESS_RULES

# Page configuration
st.set_page_config(
    page_title="Gaming User Segmentation",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .segment-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    try:
        st.session_state.pipeline = UserSegmentationPipeline()
        st.session_state.pipeline.load()
        st.session_state.model_loaded = True
    except Exception as e:
        st.session_state.model_loaded = False
        st.session_state.model_error = str(e)


def main():
    """Main application function."""
    # Header
    st.markdown('<div class="main-header">🎮 Gaming User Segmentation Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Single Prediction", "Batch Prediction", "Segment Analysis"]
        )
    
    if not st.session_state.model_loaded:
        st.error(f"❌ Model not loaded: {st.session_state.get('model_error', 'Unknown error')}")
        st.info("Please train the model first by running: python src/pipeline.py")
        return
    
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    elif page == "Segment Analysis":
        segment_analysis_page()


def single_prediction_page():
    """Single user prediction page."""
    st.header("Single User Segmentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Information")
        age = st.number_input("Age", min_value=13, max_value=100, value=25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        country = st.selectbox("Country", ["USA", "UK", "Germany", "France", "Turkey", "Japan", "South Korea"])
        device_type = st.selectbox("Device Type", ["Mobile", "PC", "Console"])
    
    with col2:
        st.subheader("Gaming Behavior")
        total_sessions = st.number_input("Total Sessions", min_value=0, value=50)
        total_playtime_hours = st.number_input("Total Playtime (Hours)", min_value=0.0, value=120.0)
        total_spent_usd = st.number_input("Total Spent (USD)", min_value=0.0, value=50.0)
        login_frequency_per_week = st.number_input("Login Frequency per Week", min_value=0.0, value=5.0)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Game Progress")
        max_level_reached = st.number_input("Max Level Reached", min_value=0, value=30)
        levels_completed = st.number_input("Levels Completed", min_value=0, value=25)
        quests_completed = st.number_input("Quests Completed", min_value=0, value=40)
        achievements_unlocked = st.number_input("Achievements Unlocked", min_value=0, value=10)
    
    with col4:
        st.subheader("Social & Engagement")
        days_since_last_login = st.number_input("Days Since Last Login", min_value=0, value=2)
        days_since_registration = st.number_input("Days Since Registration", min_value=0, value=60)
        friend_count = st.number_input("Friend Count", min_value=0, value=5)
        guild_member = st.selectbox("Guild Member", [0, 1])
        premium_subscription = st.selectbox("Premium Subscription", [0, 1])
    
    # Additional features (collapsed by default)
    with st.expander("Additional Features (Optional)"):
        col5, col6 = st.columns(2)
        with col5:
            avg_session_duration_minutes = st.number_input("Avg Session Duration (Minutes)", min_value=0.0, value=30.0)
            avg_purchase_value = st.number_input("Avg Purchase Value (USD)", min_value=0.0, value=16.67)
            last_purchase_days_ago = st.number_input("Last Purchase Days Ago", min_value=0, value=10)
            pvp_matches_played = st.number_input("PvP Matches Played", min_value=0, value=15)
        with col6:
            pve_missions_completed = st.number_input("PvE Missions Completed", min_value=0, value=40)
            chat_messages_sent = st.number_input("Chat Messages Sent", min_value=0, value=50)
            reviews_written = st.number_input("Reviews Written", min_value=0, value=2)
            events_participated = st.number_input("Events Participated", min_value=0, value=5)
    
    if st.button("Predict Segment", type="primary"):
        # Calculate derived features
        playtime_per_session = total_playtime_hours / (total_sessions + 1) if total_sessions > 0 else 0
        spending_per_session = total_spent_usd / (total_sessions + 1) if total_sessions > 0 else 0
        level_completion_rate = levels_completed / (max_level_reached + 1) if max_level_reached > 0 else 0
        engagement_score = (total_sessions * 0.3 + total_playtime_hours * 0.3 + 
                           max_level_reached * 0.2 + quests_completed * 0.2)
        
        user_data = {
            'age': age,
            'gender': gender,
            'country': country,
            'device_type': device_type,
            'total_sessions': total_sessions,
            'total_playtime_hours': total_playtime_hours,
            'avg_session_duration_minutes': avg_session_duration_minutes,
            'max_level_reached': max_level_reached,
            'levels_completed': levels_completed,
            'quests_completed': quests_completed,
            'achievements_unlocked': achievements_unlocked,
            'days_since_last_login': days_since_last_login,
            'days_since_registration': days_since_registration,
            'login_frequency_per_week': login_frequency_per_week,
            'friend_count': friend_count,
            'guild_member': guild_member,
            'total_spent_usd': total_spent_usd,
            'purchase_count': 3,
            'avg_purchase_value': avg_purchase_value,
            'last_purchase_days_ago': last_purchase_days_ago,
            'premium_subscription': premium_subscription,
            'win_rate': 0.6,
            'avg_score': 1500,
            'pvp_matches_played': pvp_matches_played,
            'pve_missions_completed': pve_missions_completed,
            'chat_messages_sent': chat_messages_sent,
            'reviews_written': reviews_written,
            'events_participated': events_participated,
            # Derived features
            'playtime_per_session': playtime_per_session,
            'spending_per_session': spending_per_session,
            'level_completion_rate': level_completion_rate,
            'engagement_score': engagement_score,
        }
        
        segment = predict_user_segment(user_data, st.session_state.pipeline)
        
        segment_names = {
            0: "Casual Players",
            1: "Regular Players",
            2: "Engaged Players",
            3: "Whales (High Spenders)"
        }
        
        st.success(f"✅ Predicted Segment: **{segment}** - {segment_names.get(segment, 'Unknown')}")
        
        # Show segment profile
        try:
            profile = get_segment_profile(segment)
            st.subheader("Segment Profile")
            st.json(profile)
        except:
            pass


def batch_prediction_page():
    """Batch prediction page."""
    st.header("Batch User Segmentation")
    
    st.info("Upload a CSV file with user data or enter data manually")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        if st.button("Predict Segments", type="primary"):
            segments = predict_batch(df, st.session_state.pipeline)
            df['segment'] = segments
            
            st.success(f"✅ Segmented {len(df)} users")
            st.dataframe(df)
            
            # Segment distribution
            segment_counts = df['segment'].value_counts().sort_index()
            fig = px.bar(
                x=segment_counts.index,
                y=segment_counts.values,
                labels={'x': 'Segment', 'y': 'User Count'},
                title="Segment Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)


def segment_analysis_page():
    """Segment analysis page."""
    st.header("Segment Analysis")
    
    try:
        processed_path = Path("data/processed/segmented_users.csv")
        if processed_path.exists():
            df = pd.read_csv(processed_path)
            
            st.subheader("Segment Overview")
            segment_counts = df['cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=segment_counts.values,
                    names=[f"Segment {i}" for i in segment_counts.index],
                    title="Segment Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=segment_counts.index,
                    y=segment_counts.values,
                    labels={'x': 'Segment', 'y': 'User Count'},
                    title="Segment Sizes"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Segment profiles
            st.subheader("Segment Profiles")
            for segment_id in sorted(df['cluster'].unique()):
                with st.expander(f"Segment {segment_id}"):
                    profile = get_segment_profile(segment_id, df)
                    st.json(profile)
        else:
            st.warning("Processed data not found. Please train the model first.")
    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()

