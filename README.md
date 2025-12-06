# Gaming User Segmentation - ML Project

🎮 **End-to-End Machine Learning Project for Gaming User Segmentation**

Bu proje, oyun sektöründe kullanıcı segmentasyonu için uçtan uca bir makine öğrenmesi çözümüdür. Kullanıcıları davranışsal ve demografik özelliklerine göre anlamlı segmentlere ayırarak, her segment için özelleştirilmiş stratejiler geliştirmeyi amaçlar.

## 📋 İçindekiler

- [Proje Özeti](#proje-özeti)
- [Problem Tanımı](#problem-tanımı)
- [Dataset](#dataset)
- [Proje Yapısı](#proje-yapısı)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Notebook'lar](#notebooklar)
- [Deployment](#deployment)
- [Sonuçlar](#sonuçlar)
- [İletişim](#iletişim)

## 🎯 Proje Özeti

Bu proje, oyun kullanıcılarını davranışsal metriklerine göre segmentlere ayırmak için **unsupervised learning** (K-Means clustering) kullanır. Her segment, farklı pazarlama stratejileri ve oyun içi deneyimler için kullanılabilir.

### Business Impact

- ✅ **Kişiselleştirilmiş Pazarlama**: Her segment için özel kampanyalar
- ✅ **Kullanıcı Tutma**: Segment bazlı retention stratejileri
- ✅ **Gelir Optimizasyonu**: Yüksek değerli kullanıcıları belirleme
- ✅ **Ürün Geliştirme**: Segment ihtiyaçlarına göre özellik optimizasyonu

## 🔍 Problem Tanımı

Oyun şirketleri, kullanıcılarını anlamak ve onlara en uygun deneyimi sunmak için segmentasyon yapmalıdır. Bu proje:

1. Kullanıcıları davranışsal özelliklerine göre segmentlere ayırır
2. Her segmentin profilini çıkarır
3. Segment bazlı stratejiler önerir

### Segmentasyon Yaklaşımı

- **Unsupervised Learning**: K-Means Clustering
- **Feature Engineering**: Davranış metrikleri ve engagement skorları
- **Segment Profilleme**: Her segmentin özelliklerini analiz etme

## 📊 Dataset

### Dataset Özellikleri

- **Format**: CSV (Tabular)
- **Satır Sayısı**: 20,000+ kullanıcı
- **Özellik Sayısı**: 34+ feature
- **Kaynak**: Kaggle veya sample dataset

### Özellikler

- **Demografik**: age, gender, country, device_type
- **Oyun Davranışı**: sessions, playtime, levels, quests
- **Engagement**: login frequency, days since last login
- **Monetization**: total spent, purchase count, premium subscription
- **Sosyal**: friend count, guild membership, chat messages
- **Performans**: win rate, average score, PvP/PvE stats

## 📁 Proje Yapısı

```
user-segmentation-ml-project/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                    # FastAPI REST API
├── streamlit_app.py          # Streamlit frontend
├── data/
│   ├── raw/                  # Ham veri
│   └── processed/            # İşlenmiş veri
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis
│   ├── 02_Baseline.ipynb     # Baseline model
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Optimization.ipynb
│   ├── 05_Model_Evaluation.ipynb
│   └── 06_Final_Pipeline.ipynb
├── src/
│   ├── config.py             # Konfigürasyon
│   ├── data_loader.py        # Veri yükleme
│   ├── pipeline.py           # ML pipeline
│   ├── inference.py          # Tahmin fonksiyonları
│   └── download_dataset.py   # Kaggle dataset indirme
├── models/                   # Eğitilmiş modeller
├── outputs/                  # Çıktılar
├── logs/                     # Log dosyaları
└── docs/                     # Dokümantasyon
```

## 🛠️ Kullanılan Teknolojiler

### Machine Learning
- **Scikit-learn**: K-Means clustering, preprocessing
- **Pandas & NumPy**: Veri işleme

### Visualization
- **Matplotlib & Seaborn**: Görselleştirme
- **Plotly**: İnteraktif grafikler

### Deployment
- **FastAPI**: REST API
- **Streamlit**: Web uygulaması
- **Uvicorn**: ASGI server

### Utilities
- **Kaggle API**: Dataset indirme
- **Joblib**: Model kaydetme/yükleme

## 🚀 Kurulum

### 1. Repository'yi Klonlayın

```bash
git clone <repository-url>
cd user-segmentation-ml-project
```

### 2. İlk Kurulum

```bash
# Virtual Environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıklar
pip install --upgrade pip
pip install -r requirements.txt

# Dataset
python src/data_loader.py

# Model Eğitimi
python src/pipeline.py
```

## 💻 Kullanım

### 🚀 Tek Komutla Çalıştırma

Tüm işlemleri tek bir script ile yapabilirsiniz:

```bash
./run.sh
```

Menüden istediğiniz seçeneği seçin:
1. Jupyter Notebook'ları Başlat
2. FastAPI'yi Başlat (http://localhost:8000)
3. Streamlit Uygulamasını Başlat (http://localhost:8501)
4. Modeli Eğit
5. Dataset İndir/Oluştur
6. Inference Testi
7. Tüm Bağımlılıkları Yükle

### REST API Kullanımı

```bash
./run.sh
# Menüden 2'yi seçin
# Tarayıcıda: http://localhost:8000/docs
```

**Örnek Request:**

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
    "login_frequency_per_week": 5
}

response = requests.post("http://localhost:8000/predict", json=user_data)
print(response.json())
```

### Streamlit Uygulaması

```bash
./run.sh
# Menüden 3'ü seçin
# Otomatik olarak tarayıcıda açılacak: http://localhost:8501
```

### Notebook'ları Çalıştırma

```bash
./run.sh
# Menüden 1'i seçin
# Tarayıcıda notebook'ları açın ve sırayla çalıştırın:
# 1. 01_EDA.ipynb - Veri keşfi
# 2. 02_Baseline.ipynb - Baseline model
# 3. 03_Feature_Engineering.ipynb - Feature engineering
# 4. 04_Model_Optimization.ipynb - Model optimizasyonu
# 5. 05_Model_Evaluation.ipynb - Model değerlendirme
# 6. 06_Final_Pipeline.ipynb - Final pipeline
```

## 📓 Notebook'lar

### 1. EDA (Exploratory Data Analysis)
- Problem tanımı
- Veri yapısı analizi
- Değişken dağılımları
- Korelasyon analizi
- Görselleştirmeler

### 2. Baseline Model
- En basit feature set (4 özellik)
- K-Means clustering
- Elbow method ile optimal cluster sayısı
- Baseline metrikleri

### 3. Feature Engineering
- Ratio features
- Interaction features
- Categorical encoding
- Temporal features
- Aggregate features
- Feature selection

### 4. Model Optimization
- Farklı clustering algoritmaları (K-Means, DBSCAN, Hierarchical)
- Hiperparametre optimizasyonu
- Grid search
- Cross-validation

### 5. Model Evaluation
- Silhouette score
- Davies-Bouldin index
- Calinski-Harabasz index
- Feature importance
- Segment profilleme

### 6. Final Pipeline
- Final feature set seçimi
- Final model eğitimi
- Model kaydetme
- Production pipeline

## 🚢 Deployment

### Local Deployment

```bash
# FastAPI
uvicorn app:app --host 0.0.0.0 --port 8000

# Streamlit
streamlit run streamlit_app.py
```

### Cloud Deployment

Proje şu platformlara deploy edilebilir:
- **Render**: FastAPI ve Streamlit desteği
- **Heroku**: Web uygulamaları için
- **AWS/GCP/Azure**: Cloud platformlar
- **HuggingFace Spaces**: Streamlit için

## 📈 Sonuçlar

### Baseline Model
- **Silhouette Score**: ~0.30-0.35
- **Davies-Bouldin Index**: ~1.5-2.0
- **Features**: 4 temel özellik

### Final Model
- **Silhouette Score**: Baseline'dan %X artış
- **Davies-Bouldin Index**: Baseline'dan %X azalma
- **Features**: 15-30 özellik

### Segmentler
1. **Casual Players**: Düşük engagement, düşük spending
2. **Regular Players**: Orta seviye engagement
3. **Engaged Players**: Yüksek engagement, orta spending
4. **Whales (High Spenders)**: Yüksek engagement, yüksek spending

## 📝 Validasyon Şeması

- **Train/Test Split**: %80 train, %20 test
- **Clustering Metrikleri**: Silhouette, Davies-Bouldin, Calinski-Harabasz
- **Business Validation**: Segment profillerinin iş mantığına uygunluğu

## 🔄 Model Canlıya Çıkışı

### Monitoring Metrikleri

- **Segment Dağılımı**: Her segmentin kullanıcı yüzdesi
- **Segment Kararlılığı**: Segmentlerin zaman içindeki değişimi
- **Model Drift**: Yeni verilerle model performansı

### Retraining Stratejisi

- Aylık retraining
- Yeni feature'lar eklendiğinde retraining
- Segment dağılımı değiştiğinde retraining

## 📚 Ek Kaynaklar

- [Örnek Proje](https://github.com/enesmanan/credit-risk-model)
- [Made with ML](https://madewithml.com/)
- [ML Engineering Book](https://soclibrary.futa.edu.ng/books/Machine%20Learning%20Engineering%20(Andriy%20Burkov)%20(Z-Library).pdf)

## 👤 İletişim

- **Proje**: Gaming User Segmentation
- **Sektör**: Gaming
- **Problem**: User Segmentation
- **Pipeline**: Unsupervised Learning (K-Means)
- **Metrik**: Silhouette Score, Davies-Bouldin Index

## 📄 Lisans

Bu proje eğitim amaçlıdır.

---

**Not**: Bu proje ML Bootcamp Final Projesi kapsamında geliştirilmiştir.

