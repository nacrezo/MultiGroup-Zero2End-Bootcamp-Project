#!/bin/bash
# Tek komutla tüm işlemleri yapan master script

cd "$(dirname "$0")"

# Renkler
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Python komutunu belirle
# Windows'ta "python" komutu bazen Microsoft Store stub'ına işaret edebilir ve çalışmaz.
# Bu yüzden tek tek kontrol ediyoruz.
PYTHON_CMD=""
for cmd in python3 python py; do
    if command -v $cmd &>/dev/null; then
        # Versiyon kontrolü yaparak gerçekten çalışıp çalışmadığını test et
        if $cmd --version &>/dev/null; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${YELLOW}Çalışan bir Python yorumlayıcısı bulunamadı!${NC}"
    echo -e "Lütfen Python'un kurulu olduğundan ve PATH'e eklendiğinden emin olun."
    echo -e "Öneri: Python'u https://www.python.org/downloads/ adresinden indirip kurun."
    exit 1
fi

echo -e "${GREEN}Python bulundu: $($PYTHON_CMD --version)${NC}"

# Virtual environment kontrolü
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment oluşturuluyor ($PYTHON_CMD)...${NC}"
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Virtual environment oluşturulamadı.${NC}"
        echo -e "Hata detayları için yukarıdaki çıktıya bakın."
        exit 1
    fi
fi

# Virtual environment'ı aktifleştir
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo -e "${YELLOW}Virtual environment aktivasyon dosyası bulunamadı!${NC}"
    exit 1
fi

export PYTHONPATH=$(pwd):$PYTHONPATH

# Menü
show_menu() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}  Gaming User Segmentation - ML Project${NC}"
    echo -e "${BLUE}========================================${NC}\n"
    echo "1) 📊 Jupyter Notebook'ları Başlat"
    echo "2) 🚀 FastAPI'yi Başlat (http://localhost:8000)"
    echo "3) 🎨 Streamlit Uygulamasını Başlat (http://localhost:8501)"
    echo "4) 🤖 Modeli Eğit"
    echo "5) 📥 Dataset İndir/Oluştur"
    echo "6) 🧪 Inference Testi"
    echo "7) 📦 Tüm Bağımlılıkları Yükle"
    echo "8) ❌ Çıkış"
    echo -e "\n"
}

# Bağımlılıkları yükle
install_dependencies() {
    echo -e "${YELLOW}Bağımlılıklar yükleniyor...${NC}"
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✅ Bağımlılıklar yüklendi${NC}"
}

# Dataset oluştur
create_dataset() {
    echo -e "${YELLOW}Dataset oluşturuluyor...${NC}"
    python src/data_loader.py
    echo -e "${GREEN}✅ Dataset hazır${NC}"
}

# Model eğit
train_model() {
    echo -e "${YELLOW}Model eğitiliyor...${NC}"
    python src/pipeline.py
    echo -e "${GREEN}✅ Model eğitildi${NC}"
}

# Inference test
test_inference() {
    echo -e "${YELLOW}Inference testi yapılıyor...${NC}"
    python src/inference.py
}

# Ana döngü
while true; do
    show_menu
    read -p "Seçiminiz (1-8): " choice
    
    case $choice in
        1)
            echo -e "${GREEN}Jupyter Notebook başlatılıyor...${NC}"
            jupyter notebook
            ;;
        2)
            echo -e "${GREEN}FastAPI başlatılıyor...${NC}"
            echo -e "${BLUE}Tarayıcıda: http://localhost:8000/docs${NC}"
            uvicorn app:app --reload --host 0.0.0.0 --port 8000
            ;;
        3)
            echo -e "${GREEN}Streamlit başlatılıyor...${NC}"
            echo -e "${BLUE}Tarayıcıda: http://localhost:8501${NC}"
            streamlit run streamlit_app.py --server.port 8501
            ;;
        4)
            train_model
            ;;
        5)
            create_dataset
            ;;
        6)
            test_inference
            ;;
        7)
            install_dependencies
            ;;
        8)
            echo -e "${GREEN}Çıkılıyor...${NC}"
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Geçersiz seçim. Lütfen 1-8 arası bir sayı girin.${NC}"
            ;;
    esac
    
    echo ""
    read -p "Devam etmek için Enter'a basın..."
done

