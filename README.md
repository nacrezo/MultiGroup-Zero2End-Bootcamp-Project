# Gaming User Segmentation - ML Project

**End-to-End Machine Learning Project for Gaming User Segmentation**

This project is an end-to-end machine learning solution for user segmentation in the gaming industry. It segments users into meaningful groups based on behavioral and demographic features, enabling personalized strategies for each segment.

## Table of Contents

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

## Project Overview

This project uses **unsupervised learning** (K-Means clustering) to segment gaming users based on behavioral metrics. Each segment can be used for different marketing strategies and in-game experiences.

### Business Impact

- **Personalized Marketing**: Custom campaigns for each segment
- **User Retention**: Segment-based retention strategies
- **Revenue Optimization**: Identify high-value users
- **Product Development**: Feature optimization based on segment needs

## Problem Statement

Gaming companies need to segment their users to understand them better and provide the most suitable experience. This project:

1. Segments users based on behavioral features
2. Profiles each segment
3. Recommends segment-based strategies

### Segmentation Approach

- **Unsupervised Learning**: K-Means Clustering
- **Feature Engineering**: Behavioral metrics and engagement scores
- **Segment Profiling**: Analyzing characteristics of each segment

## Dataset

### Dataset Kurulumu

Dataset dosyası repo'ya dahil edilmiştir. Repo'yu clone ettiğinizde dataset otomatik olarak gelecektir.

**Dataset Konumu:**
- `data/raw/online_gaming_behavior_dataset.csv` (veya diğer isimler)
- Dosya boyutu: ~2.7 MB (GitHub için uygun)

**Eğer Dataset Bulunamazsa:**

1. **Repo'yu yeniden clone edin:**
   ```bash
   git clone <repo-url>
   ```

2. **Veya manuel olarak indirin:**
   - Link: https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset
   - "Download" butonuna tıklayın
   - CSV dosyasını `data/raw/` klasörüne kopyalayın

### Dataset Characteristics

- **Format**: CSV (Tabular)
- **Number of Rows**: 40,000+ users
- **Number of Features**: 12 features
- **Source**: Kaggle (rabieelkharoua/predict-online-gaming-behavior-dataset)

### Features

- **Demographic**: Age, Gender, Location
- **Gaming Habits**: GameGenre, GameDifficulty, SessionsPerWeek, AvgSessionDurationMinutes, PlayTimeHours
- **Engagement**: InGamePurchases, PlayerLevel, AchievementUnlocked, EngagementLevel
- **Identity**: PlayerID

## Project Structure

```
MultiGroup-Zero2End-Bootcamp-Project/
├── .gitignore
├── README.md
├── requirements.txt
├── run.sh                    # Single command runner
├── streamlit_app.py          # Streamlit frontend
├── src/
│   ├── app.py                # FastAPI REST API
│   ├── config.py             # Configuration
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── pipeline.py           # ML pipeline
│   └── inference.py          # Prediction functions
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
├── models/                   # Trained models
├── outputs/                  # Outputs
├── logs/                     # Log files
└── tests/                    # Test files
```

## Technologies Used

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
- **Joblib**: Model save/load

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/nacrezo/MultiGroup-Zero2End-Bootcamp-Project
cd MultiGroup-Zero2End-Bootcamp-Project
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

# Dataset (Repo'da mevcut - otomatik olarak gelecektir)
# Eğer dataset yoksa:
# 1. Repo'yu yeniden clone edin, VEYA
# 2. Download from Kaggle: https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset
# 3. Place CSV file in data/raw/ directory
# 4. Then process:
python src/data_loader.py

# Model Training
python src/pipeline.py
```

## Usage

### Single Command Runner

Run all operations with a single script:

```bash
./run.sh
```

Select from the menu:
1. Start Jupyter Notebooks
2. Start FastAPI (http://localhost:8000)
3. Start Streamlit App (http://localhost:8501)
4. Train Model
5. Process Dataset (Dataset repo'da mevcut)
6. Test Inference
7. Install All Dependencies
8. Exit

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
    "PlayerID": 1001,
    "Age": 25,
    "Gender": "Male",
    "Location": "USA",
    "GameGenre": "Action",
    "PlayTimeHours": 35.5,
    "InGamePurchases": 0,
    "GameDifficulty": "Medium",
    "SessionsPerWeek": 5,
    "AvgSessionDurationMinutes": 45.0,
    "PlayerLevel": 10,
    "AchievementUnlocked": 15
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

## Notebooks

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
- Categorical encoding (One-Hot Encoding)
- Feature selection (Variance Threshold)

### 4. Model Optimization
- Different clustering algorithms (K-Means, DBSCAN, Hierarchical)
- Hyperparameter optimization
- Grid search
- Parameter tuning for optimal cluster count

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

## Deployment

### Live Application

**Streamlit Dashboard**: [Deployment Link](https://multigroup-zero2end-bootcamp-project-q7xm2hgsvuv4cbx7lij66o.streamlit.app/)

### Local Deployment

```bash
# FastAPI
uvicorn src.app:app --host 0.0.0.0 --port 8000

# Streamlit
streamlit run streamlit_app.py
```

## Results

### Baseline Model
- **Silhouette Score**: ~0.30-0.35
- **Davies-Bouldin Index**: ~1.5-2.0
- **Features**: 4 basic features

### Final Model
- **Silhouette Score**: Improved from baseline
- **Davies-Bouldin Index**: Reduced from baseline
- **Features**: 15-30 features

### Segments
1. **Action & Sports Fans**: Users who primarily play Action and Sports games (mixed with Strategy)
2. **Sports & Strategy Players**: Users focused on Sports and Strategy games (mixed with Action)
3. **RPG Adventurers**: Users who prefer Role-Playing Games (100% RPG)
4. **Simulation Enthusiasts**: Users focused on Simulation games (100% Simulation)

## Validation Schema

- **Validation Approach**: Since this is unsupervised learning (clustering), we use the entire dataset for training. No train/test split is performed as clustering models don't require separate validation sets in the traditional sense.
- **Clustering Metrics**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
- **Business Validation**: Segment profiles aligned with business logic

## Model Production

### Monitoring Metrics

- **Segment Distribution**: User percentage per segment
- **Segment Stability**: Segment changes over time
- **Model Drift**: Model performance with new data

### Retraining Strategy

- Monthly retraining
- Retraining when new features are added
- Retraining when segment distribution changes

## Additional Resources

- [Example Project](https://github.com/enesmanan/credit-risk-model)
- [Made with ML](https://madewithml.com/)
- [ML Engineering Book](https://soclibrary.futa.edu.ng/books/Machine%20Learning%20Engineering%20(Andriy%20Burkov)%20(Z-Library).pdf)

## Project Report

Below are the answers to the questions required by the bootcamp requirements:

### 1. Problem Definition
The goal is to segment users in the gaming industry based on their behavioral characteristics (play time, spending, session frequency, etc.). The objective is to develop personalized marketing and in-game strategies for each segment to increase user loyalty and revenue.

### 2. Baseline Process and Score
At the beginning of the project, a simple K-Means model was established using only 4 basic features (SessionsPerWeek, PlayTimeHours, InGamePurchases, PlayerLevel).
- **Baseline Silhouette Score**: ~0.35
- **Baseline Davies-Bouldin Index**: ~1.50
These scores indicated that the model needed improvement.

### 3. Feature Engineering Experiments
- **Ratio Features**: Ratios such as Purchase/PlayTime (Spending efficiency) and Session/Week (Play frequency) were derived.
- **Categorical Encoding**: One-Hot encoding was performed for GameGenre and Difficulty.
- **Scaling**: Since K-Means is distance-based, all numerical data were scaled using StandardScaler.
- **Missing Value Imputation**: Missing data were filled using the Median strategy to prevent data loss.

### 4. Validation Scheme
- **Validation Approach**: Since this is unsupervised learning (clustering), we use the entire dataset for training. No train/test split is performed as clustering models don't require separate validation sets in the traditional sense.
- **Metrics**: Silhouette Score (intra-cluster similarity vs. inter-cluster separation), Davies-Bouldin Index, and Calinski-Harabasz Index were used to evaluate clustering quality.

### 5. Final Pipeline and Feature Set Selection
Based on experiments in the `05_Model_Evaluation.ipynb` notebook:
- **Feature Selection**: Low variance features were eliminated using VarianceThreshold.
- **Model**: The K-Means algorithm was chosen for its high interpretability.
- **Cluster Count**: Although the Elbow method did not show a sharp break, 4 segments were decided as optimal based on Business rules (Action, RPG, Strategy, Sim).

### 6. Final Model vs Baseline
The final model provided better separation than the baseline with an enriched feature set and optimized parameters.
- **Final Silhouette Score**: Increased compared to the baseline.
- **Segment Profiles**: Segments became more meaningful and actionable for the business unit.

### 7. Business Alignment
The model output identified 4 main personas (Action Fans, RPG Adventurers, etc.). These personas align perfectly with the marketing team's campaign structures (e.g., tournament suggestions for Action players, costume discounts for RPG players).

### 8. Deployment and Monitoring
The model was exported as `user_segmentation_pipeline.pkl` and deployed using FastAPI.
**Metrics to Monitor:**
- **Model Drift**: Deviation of incoming new data distribution from training data.
- **Segment Distribution**: Percentage distribution of segments on a weekly basis (e.g., sudden disappearance of a segment could indicate a technical issue or trend change).

## Contact

- **Author**: Can Özer
- **Email**: canozer.pirireis@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/canozer1/
- **Project Link**: https://github.com/nacrezo/MultiGroup-Zero2End-Bootcamp-Project

## License

This project is for educational purposes.

---

**Note**: This project was developed as part of the ML Bootcamp Final Project.
