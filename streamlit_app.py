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
    page_icon=None,
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
    st.markdown('<div class="main-header">Gaming User Segmentation Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Single Prediction", "Batch Prediction", "Segment Analysis"]
        )
    
    if not st.session_state.model_loaded:
        st.error(f"Model not loaded: {st.session_state.get('model_error', 'Unknown error')}")
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
    st.info("Predict segment for Gaming Behavior Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics & Info")
        # PlayerID is not used in model, just for display/tracking. 
        # User requested to hide it from UI. Auto-generating in background.
        import random
        random_id = random.randint(1000, 9999)
        player_id = random_id # Hidden from UI
        age = st.number_input("Age", min_value=13, max_value=99, value=25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        location = st.selectbox("Location", ["USA", "Europe", "Asia", "Other"])
        
    with col2:
        st.subheader("Gaming Habits")
        game_genre = st.selectbox("Game Genre", ["Action", "RPG", "Simulation", "Strategy", "Sports"])
        game_difficulty = st.selectbox("Game Difficulty", ["Easy", "Medium", "Hard"])
        player_level = st.number_input("Player Level", min_value=1, value=10)
        
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Activity Metrics")
        play_time_hours = st.number_input("Play Time (Hours)", min_value=0.0, value=35.5)
        sessions_per_week = st.number_input("Sessions Per Week", min_value=0, value=5)
        avg_session_duration = st.number_input("Avg Session Duration (Min)", min_value=1.0, value=45.0)

    with col4:
        st.subheader("Engagement & Achievements")
        in_game_purchases = st.selectbox("In-Game Purchases", [0, 1], help="0: No, 1: Yes")
        achievements_unlocked = st.number_input("Achievements Unlocked", min_value=0, value=15)
        
    if st.button("Predict Segment", type="primary"):
        # Prepare user data
        user_data = {
            'PlayerID': player_id,
            'Age': age,
            'Gender': gender,
            'Location': location,
            'GameGenre': game_genre,
            'PlayTimeHours': play_time_hours,
            'InGamePurchases': in_game_purchases,
            'GameDifficulty': game_difficulty,
            'SessionsPerWeek': sessions_per_week,
            'AvgSessionDurationMinutes': avg_session_duration,
            'PlayerLevel': player_level,
            'AchievementUnlocked': achievements_unlocked,
        }
        
        try:
            segment = predict_user_segment(user_data, st.session_state.pipeline)
        
            # Segment names derived from cluster analysis (mainly Genre-based differentiation)
            # Updated based on actual segment analysis:
            # Segment 0: Action dominant (33.8%), also Sports (33.2%) and Strategy (33.0%)
            # Segment 1: Sports dominant (33.7%), also Strategy (33.6%) and Action (32.7%)
            # Segment 2: RPG dominant (100%)
            # Segment 3: Simulation dominant (100%)
            segment_names = {
                0: "Action & Sports Fans",  # Action dominant, mixed with Sports/Strategy
                1: "Sports & Strategy Players",  # Sports dominant, mixed with Strategy/Action
                2: "RPG Adventurers",  # 100% RPG
                3: "Simulation Enthusiasts"  # 100% Simulation
            }
            
            segment_name = segment_names.get(segment, f"Segment {segment}")
            
            st.success(f"Predicted Segment: **{segment} - {segment_name}**")
            
            # Show segment profile
            try:
                profile = get_segment_profile(segment)
                st.subheader("Segment Profile")
                st.json(profile)
            except:
                pass
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.warning("Make sure the model is trained with the new dataset: Run '4) Train Model' in run.sh")
        



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
            
            st.success(f"Segmented {len(df)} users")
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
                # Define colors for each segment
                colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']  # Blue, Red, Green, Orange
                
                fig = px.pie(
                    values=segment_counts.values,
                    names=[f"Segment {i}" for i in segment_counts.index],
                    title="Segment Distribution",
                    color_discrete_sequence=colors
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Define colors for each segment
                colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']  # Blue, Red, Green, Orange
                
                # Create bar chart with custom colors
                fig = go.Figure(data=[
                    go.Bar(
                        x=segment_counts.index,
                        y=segment_counts.values,
                        marker_color=[colors[i % len(colors)] for i in segment_counts.index],
                        text=segment_counts.values,
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    title="Segment Sizes",
                    xaxis_title="Segment",
                    yaxis_title="User Count",
                    showlegend=False
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

