import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

# Page Configuration
st.set_page_config(
    page_title="Ev Fiyat Tahmin Uygulaması",
    page_icon="🏠",
    layout="wide"
)

# ==============================================================================
# 1. YARDIMCI FONKSİYONLAR & MODEL EĞİTİMİ (ÖNBELLEKLİ)
# ==============================================================================
def create_features(df):
    """Girdilerden türetilmiş yeni özellikleri hesaplar."""
    df = df.copy()
    df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 1e-5)
    df['bath_per_bed_ratio'] = df['bathrooms'] / (df['bedrooms'] + 1e-5)
    df['area_per_story'] = df['area'] / (df['stories'] + 1e-5)
    return df

@st.cache_resource
def train_model_pipeline():
    """Veriyi üretir, pipeline'ı kurar ve modeli eğitir."""
    np.random.seed(42)
    n_samples = 600

    data = pd.DataFrame({
        'area': np.random.randint(1500, 10000, size=n_samples),
        'bedrooms': np.random.randint(1, 6, size=n_samples),
        'bathrooms': np.random.randint(1, 4, size=n_samples),
        'stories': np.random.randint(1, 5, size=n_samples),
        'mainroad': np.random.choice(['yes', 'no'], size=n_samples),
        'prefarea': np.random.choice(['yes', 'no'], size=n_samples),
        'furnishingstatus': np.random.choice(['furnished', 'semi-furnished', 'unfurnished'], size=n_samples),
    })

    # Hedef Değişken (Fiyat)
    data['price'] = (
        data['area'] * 450 + 
        data['bedrooms'] * 150000 + 
        data['bathrooms'] * 350000 + 
        (data['prefarea'] == 'yes') * 500000 + 
        np.random.normal(0, 200000, size=n_samples)
    )

    X = data.drop(columns=['price'])
    y = data['price']

    # Feature Engineering
    X_fe = create_features(X)

    num_cols = X_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_fe.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X_fe, y, test_size=0.2, random_state=42)
    full_pipeline.fit(X_train, y_train)

    y_pred = full_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return full_pipeline, mae, r2

# Modeli Eğit / Yükle
pipeline, model_mae, model_r2 = train_model_pipeline()

# ==============================================================================
# 2. STREAMLIT ARAYÜZÜ
# ==============================================================================
st.title("🏠 Yapay Zekâ Destekli Ev Fiyat Tahmin Paneli")
st.markdown("Evin temel özelliklerini seçerek tahmini piyasa değerini hesaplayabilirsiniz.")

st.divider()

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📋 Ev Özelliklerini Giriniz")
    
    area = st.number_input("Metrekare / Alan (sqft)", min_value=500, max_value=20000, value=3500, step=100)
    
    col_a, col_b = st.columns(2)
    with col_a:
        bedrooms = st.selectbox("Yatak Odası Sayısı", options=[1, 2, 3, 4, 5, 6], index=2)
        stories = st.selectbox("Kat Sayısı", options=[1, 2, 3, 4], index=1)
    with col_b:
        bathrooms = st.selectbox("Banyo Sayısı", options=[1, 2, 3, 4], index=1)
        furnishingstatus = st.selectbox("Eşya Durumu", options=['furnished', 'semi-furnished', 'unfurnished'], index=1)

    col_c, col_d = st.columns(2)
    with col_c:
        mainroad = st.radio("Ana Yola Cepheli mi?", options=['yes', 'no'], index=0, horizontal=True)
    with col_d:
        prefarea = st.radio("Presti̇jli Bölgede mi?", options=['yes', 'no'], index=1, horizontal=True)

    predict_btn = st.button("🔮 Fiyatı Tahmin Et", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 Tahmin Sonucu")
    
    if predict_btn:
        # Ham veri çerçevesi oluşturma
        raw_input = pd.DataFrame([{
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'mainroad': mainroad,
            'prefarea': prefarea,
            'furnishingstatus': furnishingstatus
        }])
        
        # Otomatik Feature Engineering
        processed_input = create_features(raw_input)
        
        # Tahmin
        pred_price = pipeline.predict(processed_input)[0]
        
        st.success("Tahmin Başarıyla Hesaplandı!")
        st.metric(
            label="Tahmini Ev Fiyatı",
            value=f"{pred_price:,.2f} TL"
        )
        
        st.info(
            f"**Model Performans Bi̇lgi̇si̇:**\n"
            f"- **Ortalama Hata (MAE):** ±{model_mae:,.2f} TL\n"
            f"- **Açıklayıcılık Oranı (R²):** %{model_r2 * 100:.1f}"
        )
    else:
        st.info("Tahmin sonucunu görmek için sol taraftaki değerleri ayarlayıp **'Fiyatı Tahmin Et'** butonuna basınız.")
