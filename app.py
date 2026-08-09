import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from lightgbm import LGBMRegressor

# Page Configuration
st.set_page_config(
    page_title="Ev Fiyat Tahmin Uygulaması",
    page_icon="🏠",
    layout="wide"
)

# ==============================================================================
# 1. YARDIMCI FONKSİYONLAR & LOG-TRANSFORM DESTEKLİ MODEL EĞİTİMİ
# ==============================================================================
def create_features(df):
    """Girdilerden türetilmiş yeni özellikleri hesaplar."""
    df = df.copy()
    df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 1e-5)
    df['bath_per_bed_ratio'] = df['bathrooms'] / (df['bedrooms'] + 1e-5)
    df['area_per_story'] = df['area'] / (df['stories'] + 1e-5)
    return df

@st.cache_resource
def train_and_optimize_model():
    """
    1. Sentetik veri kümesi üretir.
    2. Hedef değişkene Log Transformation (np.log1p) uygular.
    3. Ön işleme pipeline'ını kurar.
    4. RandomizedSearchCV (5-Fold CV) ile en iyi hiperparametreleri seçer.
    5. Tahminleri ters dönüşümle (np.expm1) orijinal TL ölçeğine çevirip değerlendirir.
    """
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

    # LOG TRANSFORMATION: Fiyat değişkeni logaritmik ölçeğe aktarılır
    y_log = np.log1p(y)

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

    # Ana Pipeline
    base_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LGBMRegressor(random_state=42, verbose=-1))
    ])

    # Hiperparametre Arama Dağılımı
    param_distributions = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7, -1],
        'model__num_leaves': [15, 31, 63],
        'model__subsample': [0.7, 0.8, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 1.0],
        'model__reg_alpha': [0.0, 0.1, 1.0],
        'model__reg_lambda': [0.0, 0.1, 1.0]
    }

    # Train / Test Ayrımı (Orijinal y ve Log y)
    X_train, X_test, y_train_log, y_test_log, y_train_orig, y_test_orig = train_test_split(
        X_fe, y_log, y, test_size=0.2, random_state=42
    )
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Hiperparametre Optimizasyonu (Log Hedef Üzerinde)
    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=15,
        cv=kf,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X_train, y_train_log)

    best_pipeline = search.best_estimator_

    # Test Kümesi Tahminleri & GERİ DÖNÜŞÜM (np.expm1)
    y_pred_log = best_pipeline.predict(X_test)
    y_pred_orig = np.expm1(y_pred_log)  # Logaritmik tahmini TL birimine geri çevirir

    # Metriklerin Orijinal TL Üzerinden Hesaplanması
    test_mae = mean_absolute_error(y_test_orig, y_pred_orig)
    test_mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig) * 100
    test_r2 = r2_score(y_test_orig, y_pred_orig)

    return best_pipeline, test_mae, test_mape, test_r2

# Modeli ve metrikleri yükle
best_model, test_mae, test_mape, test_r2 = train_and_optimize_model()

# ==============================================================================
# 2. STREAMLIT ARAYÜZÜ (LOG-TRANSFORMED PREDICTION)
# ==============================================================================
st.title("🏠 Yapay Zekâ Destekli Ev Fiyat Tahmin Paneli")
st.caption("Log Transformation (np.log1p) ve LightGBM ile Yüksek Fiyat Hassasiyetli Model")

st.divider()

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📋 Ev Özelliklerini Giriniz")
    
    area = st.number_input("Metrekare / Alan (sqft)", min_value=1.0, value=3500.0, step=50.0)
    
    col_a, col_b = st.columns(2)
    with col_a:
        bedrooms = st.number_input("Yatak Odası Sayısı", min_value=0, value=3, step=1)
        stories = st.number_input("Kat Sayısı", min_value=0, value=2, step=1)
    with col_b:
        bathrooms = st.number_input("Banyo Sayısı", min_value=0, value=2, step=1)
        furnishingstatus = st.selectbox("Eşya Durumu", options=['furnished', 'semi-furnished', 'unfurnished'], index=1)

    col_c, col_d = st.columns(2)
    with col_c:
        mainroad = st.radio("Ana Yola Cepheli mi?", options=['yes', 'no'], index=0, horizontal=True)
    with col_d:
        prefarea = st.radio("Prestijli Bölgede mi?", options=['yes', 'no'], index=1, horizontal=True)

    predict_btn = st.button("🔮 Fiyatı Tahmin Et", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 Tahmin Sonucu ve Model Hata Analizi")
    
    if predict_btn:
        raw_input = pd.DataFrame([{
            'area': float(area),
            'bedrooms': int(bedrooms),
            'bathrooms': int(bathrooms),
            'stories': int(stories),
            'mainroad': mainroad,
            'prefarea': prefarea,
            'furnishingstatus': furnishingstatus
        }])
        
        processed_input = create_features(raw_input)
        
        # 1. Log Ölçeğinde Tahmin Üret
        pred_log = best_model.predict(processed_input)[0]
        
        # 2. Tahmini Orijinal TL Ölçeğine Çevir (np.expm1)
        pred_price = np.expm1(pred_log)
        
        # Olası Aralık
        lower_bound = max(0, pred_price - test_mae)
        upper_bound = pred_price + test_mae

        st.success("Tahmin Başarıyla Üretildi!")
        
        # Nokta Tahmin
        st.metric(
            label="Tahmini Piyasa Değeri (Log-Transformed Model)",
            value=f"{pred_price:,.2f} TL"
        )

        # Hata Bildirimi
        st.warning(
            f"📢 **Model Güvenilirlik & Hata Bildirimi:**\n\n"
            f"- Model **Log Transformation** ile eğitildiği için yüksek fiyatlı ve uç değerdeki mülklerde daha kararlı sonuç verir.\n"
            f"- Test kümesi genelinde ortalama **%{test_mape:.2f} hata payı** ile çalışmaktadır.\n"
            f"- Yapılan tahminlerde ortalama sapma miktarı **±{test_mae:,.2f} TL**'dir.\n"
            f"- **Olası Fiyat Aralığı:** `{lower_bound:,.0f} TL` — `{upper_bound:,.0f} TL`"
        )

        st.divider()

        # Detaylı Metrikler
        m1, m2 = st.columns(2)
        m1.metric("Model Yüzdesel Hatalılık (MAPE)", f"%{test_mape:.2f}")
        m2.metric("Model Açıklayıcılık Oranı (R²)", f"%{test_r2 * 100:.1f}")

    else:
        st.info(
            f"ℹ️ **Sistem Bilgisi:** Model hedef değişkene `np.log1p` uygulanarak eğitilmiştir ve test kümesinde ortalama **%{test_mape:.2f}** hata payına sahiptir.\n\n"
            "Tahmin almak için soldaki form alanlarını doldurup **'Fiyatı Tahmin Et'** butonuna basınız."
        )
