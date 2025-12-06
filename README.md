# Gaming User Segmentation - ML Project

🎮 **End-to-End Machine Learning Project for Gaming User Segmentation**

This project is an end-to-end machine learning solution for user segmentation in the gaming industry. It segments users into meaningful groups based on behavioral and demographic features, enabling personalized strategies for each segment.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Deployment](#deployment)
- [Results](#results)
- [Contact](#contact)

## 🎯 Project Overview

This project uses **unsupervised learning** (K-Means clustering) to segment gaming users based on behavioral metrics. Each segment can be used for different marketing strategies and in-game experiences.

### Business Impact

- ✅ **Personalized Marketing**: Custom campaigns for each segment
- ✅ **User Retention**: Segment-based retention strategies
- ✅ **Revenue Optimization**: Identify high-value users
- ✅ **Product Development**: Feature optimization based on segment needs

## 🔍 Problem Statement

Gaming companies need to segment their users to understand them better and provide the most suitable experience. This project:

1. Segments users based on behavioral features
2. Profiles each segment
3. Recommends segment-based strategies

### Segmentation Approach

- **Unsupervised Learning**: K-Means Clustering
- **Feature Engineering**: Behavioral metrics and engagement scores
- **Segment Profiling**: Analyzing characteristics of each segment

## 📊 Dataset

### Dataset Characteristics

- **Format**: CSV (Tabular)
- **Number of Rows**: 20,000+ users
- **Number of Features**: 34+ features
- **Source**: Kaggle or sample dataset

### Features

- **Demographic**: age, gender, country, device_type
- **Gaming Behavior**: sessions, playtime, levels, quests
- **Engagement**: login frequency, days since last login
- **Monetization**: total spent, purchase count, premium subscription
- **Social**: friend count, guild membership, chat messages
- **Performance**: win rate, average score, PvP/PvE stats

## 📁 Project Structure

```
user-segmentation-ml-project/
├── .gitignore
├── README.md
├── requirements.txt
├── run.sh                    # Single command runner
├── app.py                    # FastAPI REST API
├── streamlit_app.py          # Streamlit frontend
├── data/
│   ├── raw/                  # Raw data
│   └── processed/            # Processed data
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis
│   ├── 02_Baseline.ipynb     # Baseline model
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Optimization.ipynb
│   ├── 05_Model_Evaluation.ipynb
│   └── 06_Final_Pipeline.ipynb
├── src/
│   ├── config.py             # Configuration
│   ├── data_loader.py        # Data loading
│   ├── pipeline.py           # ML pipeline
│   ├── inference.py          # Prediction functions
│   └── download_dataset.py   # Kaggle dataset download
├── models/                   # Trained models
├── outputs/                  # Outputs
├── logs/                     # Log files
└── docs/                     # Documentation
```

## 🛠️ Technologies Used

### Machine Learning
- **Scikit-learn**: K-Means clustering, preprocessing
- **Pandas & NumPy**: Data processing

### Visualization
- **Matplotlib & Seaborn**: Visualization
- **Plotly**: Interactive charts

### Deployment
- **FastAPI**: REST API
- **Streamlit**: Web application
- **Uvicorn**: ASGI server

### Utilities
- **Kaggle API**: Dataset download
- **Joblib**: Model save/load

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd user-segmentation-ml-project
```

### 2. Initial Setup

**Option 1: Using the run script (Recommended)**

```bash
chmod +x run.sh
./run.sh
# Select option 7 to install all dependencies
# Select option 5 to create/download dataset
# Select option 4 to train the model
```

**Option 2: Manual Setup**

```bash
# Virtual Environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Dataset
python src/data_loader.py

# Model Training
python src/pipeline.py
```

## 💻 Usage

### 🚀 Single Command Runner

Run all operations with a single script:

```bash
./run.sh
```

Select from the menu:
1. 📊 Start Jupyter Notebooks
2. 🚀 Start FastAPI (http://localhost:8000)
3. 🎨 Start Streamlit App (http://localhost:8501)
4. 🤖 Train Model
5. 📥 Download/Create Dataset
6. 🧪 Test Inference
7. 📦 Install All Dependencies
8. ❌ Exit

### REST API Usage

```bash
./run.sh
# Select option 2
# Open in browser: http://localhost:8000/docs
```

**Example Request:**

```python
import requests

user_data = {
    "age": 25,
    "gender": "Male",
    "country": "USA",
    "device_type": "Mobile",
    "total_sessions": 50,
    "total_playtime_hours": 120,
    "total_spent_usd": 50,
    "login_frequency_per_week": 5,
    "max_level_reached": 30,
    "levels_completed": 25,
    "quests_completed": 40,
    "achievements_unlocked": 10,
    "days_since_last_login": 2,
    "days_since_registration": 60,
    "friend_count": 5,
    "guild_member": 1,
    "purchase_count": 3,
    "premium_subscription": 0,
    "win_rate": 0.6,
    "avg_score": 1500,
    "avg_session_duration_minutes": 2.4,
    "avg_purchase_value": 16.6,
    "last_purchase_days_ago": 10,
    "pvp_matches_played": 5,
    "pve_missions_completed": 15,
    "chat_messages_sent": 20,
    "reviews_written": 1,
    "events_participated": 3
}

response = requests.post("http://localhost:8000/predict", json=user_data)
print(response.json())
```

### Streamlit Application

```bash
./run.sh
# Select option 3
# Automatically opens in browser: http://localhost:8501
```

### Running Notebooks

```bash
./run.sh
# Select option 1
# Open notebooks in browser and run sequentially:
# 1. 01_EDA.ipynb - Data exploration
# 2. 02_Baseline.ipynb - Baseline model
# 3. 03_Feature_Engineering.ipynb - Feature engineering
# 4. 04_Model_Optimization.ipynb - Model optimization
# 5. 05_Model_Evaluation.ipynb - Model evaluation
# 6. 06_Final_Pipeline.ipynb - Final pipeline
```

## 📓 Notebooks

### 1. EDA (Exploratory Data Analysis)
- Problem definition
- Data structure analysis
- Variable distributions
- Correlation analysis
- Visualizations

### 2. Baseline Model
- Minimal feature set (4 features)
- K-Means clustering
- Optimal cluster count using Elbow method
- Baseline metrics

### 3. Feature Engineering
- Ratio features
- Interaction features
- Categorical encoding
- Temporal features
- Aggregate features
- Feature selection

### 4. Model Optimization
- Different clustering algorithms (K-Means, DBSCAN, Hierarchical)
- Hyperparameter optimization
- Grid search
- Cross-validation

### 5. Model Evaluation
- Silhouette score
- Davies-Bouldin index
- Calinski-Harabasz index
- Feature importance
- Segment profiling

### 6. Final Pipeline
- Final feature set selection
- Final model training
- Model saving
- Production pipeline

## 🚢 Deployment

### Local Deployment

```bash
# FastAPI
uvicorn app:app --host 0.0.0.0 --port 8000

# Streamlit
streamlit run streamlit_app.py
```

### Streamlit Cloud Deployment (Recommended)

Streamlit Cloud is the easiest way to deploy your Streamlit app for free.

#### Prerequisites

1. **GitHub Repository**: Push your code to GitHub
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Model File**: Ensure `models/user_segmentation_pipeline.pkl` is committed to GitHub
   - The `.gitignore` is configured to include the model file
   - Verify: `git ls-files models/user_segmentation_pipeline.pkl`

#### Deployment Steps

1. **Sign up for Streamlit Cloud**
   - Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account

2. **Deploy Your App**
   - Click "New app"
   - Select your GitHub repository
   - Choose the branch (usually `main`)
   - Set the main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Your App is Live!**
   - Streamlit Cloud will automatically build and deploy your app
   - You'll get a URL like: `https://your-app-name.streamlit.app`
   - The app will automatically redeploy on every push to the main branch

#### Streamlit Cloud Configuration

The project includes `.streamlit/config.toml` for optimal Streamlit Cloud settings.

#### Troubleshooting

- **Model not found**: Ensure `models/user_segmentation_pipeline.pkl` is committed to GitHub
- **Import errors**: Check that all dependencies are in `requirements.txt`
- **Path issues**: The app uses relative paths that work on Streamlit Cloud

### Alternative Cloud Deployment Options

- **HuggingFace Spaces**: [https://huggingface.co/spaces](https://huggingface.co/spaces)
  - Create a new Space
  - Select Streamlit as SDK
  - Upload your files
- **Render**: [https://render.com](https://render.com)
  - Create a new Web Service
  - Build command: `pip install -r requirements.txt`
  - Start command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
- **Heroku**: Use the provided `Procfile` for deployment

## 📈 Results

### Baseline Model
- **Silhouette Score**: ~0.30-0.35
- **Davies-Bouldin Index**: ~1.5-2.0
- **Features**: 4 basic features

### Final Model
- **Silhouette Score**: Improved from baseline
- **Davies-Bouldin Index**: Reduced from baseline
- **Features**: 15-30 features

### Segments
1. **Casual Players**: Low engagement, low spending
2. **Regular Players**: Medium engagement
3. **Engaged Players**: High engagement, medium spending
4. **Whales (High Spenders)**: High engagement, high spending

## 📝 Validation Schema

- **Train/Test Split**: 80% train, 20% test
- **Clustering Metrics**: Silhouette, Davies-Bouldin, Calinski-Harabasz
- **Business Validation**: Segment profiles aligned with business logic

## 🔄 Model Production

### Monitoring Metrics

- **Segment Distribution**: User percentage per segment
- **Segment Stability**: Segment changes over time
- **Model Drift**: Model performance with new data

### Retraining Strategy

- Monthly retraining
- Retraining when new features are added
- Retraining when segment distribution changes

## 📚 Additional Resources

- [Example Project](https://github.com/enesmanan/credit-risk-model)
- [Made with ML](https://madewithml.com/)
- [ML Engineering Book](https://soclibrary.futa.edu.ng/books/Machine%20Learning%20Engineering%20(Andriy%20Burkov)%20(Z-Library).pdf)

## 👤 Contact

- **Project**: Gaming User Segmentation
- **Industry**: Gaming
- **Problem**: User Segmentation
- **Pipeline**: Unsupervised Learning (K-Means)
- **Metrics**: Silhouette Score, Davies-Bouldin Index

## 📄 License

This project is for educational purposes.

---

**Note**: This project was developed as part of the ML Bootcamp Final Project.
