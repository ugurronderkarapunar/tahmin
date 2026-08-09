import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
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
# 1. YARDIMCI FONKSİYONLAR & HİPERPARAMETRE OPTİMİZASYONLU MODEL EĞİTİMİ
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
    1. Sentetik veriyi üretir.
    2. Ön işleme pipeline'ını kurar.
    3. RandomizedSearchCV (5-Fold CV) ile MAE skorunu en minimize eden hiperparametreleri bulur.
    4. En iyi modeli (best_estimator_) döndürür.
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

    # Hiperparametre Dağılımı (MAE Odaklı)
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

    X_train, X_test, y_train, y_test = train_test_split(X_fe, y, test_size=0.2, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # MAE Skoruna Göre Hiperparametre Arama
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

    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    best_cv_mae = -search.best_score_
    best_params = search.best_params_

    # Test Kümesi Performansı
    y_pred_test = best_pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    return best_pipeline, best_cv_mae, test_mae, test_r2, best_params

# En iyi modeli oluştur ve önbelleğe al
best_model, best_cv_mae, test_mae, test_r2, best_params = train_and_optimize_model()

# ==============================================================================
# 2. STREAMLIT ARAYÜZÜ (SERBEST SAYISAL GİRDİ ALANLARI)
# ==============================================================================
st.title("🏠 Yapay Zekâ Destekli Ev Fiyat Tahmin Paneli")
st.caption("5-Fold CV & MAE Hiperparametre Optimizasyonu Yapılmış En İyi Model")

st.divider()

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📋 Ev Özelliklerini Giriniz")
    
    # Herhangi bir sayısal değer girilebilen serbest input alanları
    area = st.number_input(
        "Metrekare / Alan (sqft)", 
        min_value=1.0, 
        value=3500.0, 
        step=50.0,
        help="İstediğiniz herhangi bir metrekare değerini elle yazabilirsiniz."
    )
    
    col_a, col_b = st.columns(2)
    with col_a:
        bedrooms = st.number_input(
            "Yatak Odası Sayısı", 
            min_value=0, 
            value=3, 
            step=1,
            help="İstediğiniz oda sayısını giriniz."
        )
        stories = st.number_input(
            "Kat Sayısı", 
            min_value=0, 
            value=2, 
            step=1,
            help="İstediğiniz kat sayısını giriniz."
        )
    with col_b:
        bathrooms = st.number_input(
            "Banyo Sayısı", 
            min_value=0, 
            value=2, 
            step=1,
            help="İstediğiniz banyo sayısını giriniz."
        )
        furnishingstatus = st.selectbox(
            "Eşya Durumu", 
            options=['furnished', 'semi-furnished', 'unfurnished'], 
            index=1
        )

    col_c, col_d = st.columns(2)
    with col_c:
        mainroad = st.radio("Ana Yola Cepheli mi?", options=['yes', 'no'], index=0, horizontal=True)
    with col_d:
        prefarea = st.radio("Prestijli Bölgede mi?", options=['yes', 'no'], index=1, horizontal=True)

    predict_btn = st.button("🔮 Fiyatı Tahmin Et", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 Tahmin ve Model Performansı")
    
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
        
        # Otomatik Feature Engineering
        processed_input = create_features(raw_input)
        
        # En iyi model ile tahmin alma
        pred_price = best_model.predict(processed_input)[0]
        
        st.success("Tahmin Başarıyla Hesaplandı!")
        st.metric(
            label="En İyi Modelin Tahmin Ettiği Ev Fiyatı",
            value=f"{pred_price:,.2f} TL"
        )
        
        st.divider()
        st.markdown("### 🎯 Optimizasyon Metrikleri")
        m1, m2, m3 = st.columns(3)
        m1.metric("En İyi CV MAE", f"±{best_cv_mae:,.0f} TL")
        m2.metric("Test MAE", f"±{test_mae:,.0f} TL")
        m3.metric("Test R²", f"%{test_r2 * 100:.1f}")

        with st.expander("🛠️ Seçilen En İyi Hiperparametreler (Best Parameters)"):
            clean_params = {k.replace('model__', ''): v for k, v in best_params.items()}
            st.json(clean_params)
    else:
        st.info("İstediğiniz sayısal değerleri girip **'Fiyatı Tahmin Et'** butonuna basarak anlık tahmin alabilirsiniz.")
