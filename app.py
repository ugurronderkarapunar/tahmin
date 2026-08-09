import io
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
# 1. YARDIMCI FONKSİYONLAR & TAM MODEL EĞİTİMİ (CACHED)
# ==============================================================================
def create_features(df):
    """Girdilerden türetilmiş yeni özellikleri hesaplar."""
    df = df.copy()
    if 'area' in df.columns and 'bedrooms' in df.columns:
        df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 1e-5)
    if 'bathrooms' in df.columns and 'bedrooms' in df.columns:
        df['bath_per_bed_ratio'] = df['bathrooms'] / (df['bedrooms'] + 1e-5)
    if 'area' in df.columns and 'stories' in df.columns:
        df['area_per_story'] = df['area'] / (df['stories'] + 1e-5)
    return df

def generate_default_data():
    """CSV yüklenmediğinde kullanılacak varsayılan veri kümesi."""
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'area': np.random.randint(1000, 15000, size=n_samples),
        'bedrooms': np.random.randint(1, 8, size=n_samples),
        'bathrooms': np.random.randint(1, 6, size=n_samples),
        'stories': np.random.randint(1, 5, size=n_samples),
        'mainroad': np.random.choice(['yes', 'no'], size=n_samples),
        'guestroom': np.random.choice(['yes', 'no'], size=n_samples),
        'basement': np.random.choice(['yes', 'no'], size=n_samples),
        'hotwaterheating': np.random.choice(['yes', 'no'], size=n_samples),
        'airconditioning': np.random.choice(['yes', 'no'], size=n_samples),
        'parking': np.random.randint(0, 4, size=n_samples),
        'prefarea': np.random.choice(['yes', 'no'], size=n_samples),
        'furnishingstatus': np.random.choice(['furnished', 'semi-furnished', 'unfurnished'], size=n_samples),
    })
    base_price = (
        data['area'] * 850 + 
        (data['area'] ** 1.15) * 50 + 
        data['bedrooms'] * 400000 + 
        data['bathrooms'] * 800000 + 
        (data['prefarea'] == 'yes') * 2500000 + 
        (data['mainroad'] == 'yes') * 1000000
    )
    noise = np.random.normal(0, base_price * 0.08, size=n_samples)
    data['price'] = np.maximum(500000, base_price + noise)
    return data

@st.cache_resource(show_spinner="Model tüm değişkenler ile eğitiliyor ve hiperparametreler optimize ediliyor...")
def train_full_model(file_bytes, file_name):
    """Veri setindeki TÜM değişkenleri kullanarak modeli eğitir."""
    if file_bytes is not None:
        raw_data = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raw_data = generate_default_data()

    target_col = 'price'
    if target_col not in raw_data.columns:
        return None, None, None, None, raw_data

    X = raw_data.drop(columns=[target_col])
    y = raw_data[target_col]
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

    base_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LGBMRegressor(random_state=42, verbose=-1))
    ])

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

    X_train, X_test, y_train_log, y_test_log, y_train_orig, y_test_orig = train_test_split(
        X_fe, y_log, y, test_size=0.2, random_state=42
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

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

    y_pred_log = best_pipeline.predict(X_test)
    y_pred_orig = np.expm1(y_pred_log)

    test_mae = mean_absolute_error(y_test_orig, y_pred_orig)
    test_mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig) * 100
    test_r2 = r2_score(y_test_orig, y_pred_orig)

    return best_pipeline, test_mae, test_mape, test_r2, raw_data

# ==============================================================================
# 2. STREAMLIT ARAYÜZÜ (TÜM GİRDİLER DİNAMİK OLARAK ALINIR)
# ==============================================================================
st.title("🏠 Yapay Zekâ Destekli Ev Fiyat Tahmin Paneli")
st.caption("Veri Setindeki Tüm Değişkenlerin Eksiksiz Kullanıldığı Tam Model")

# YAN MENÜ: CSV Yükleme
with st.sidebar:
    st.header("📁 Veri Seti Ayarları")
    uploaded_file = st.file_uploader("Kendi CSV Dosyanızı Yükleyin", type=["csv"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        st.success(f"✅ `{file_name}` aktif.")
    else:
        file_bytes = None
        file_name = "default_data.csv"
        st.info("💡 Varsayılan veri seti kullanılmaktadır.")

best_model, test_mae, test_mape, test_r2, raw_df = train_full_model(file_bytes, file_name)

st.divider()

if best_model is not None:
    col_input, col_result = st.columns([1.2, 0.8], gap="large")

    with col_input:
        st.subheader("📋 Ev Özellikleri (Tüm Değişkenler)")
        st.caption("Tahmin başarısını artırmak için lütfen tüm özellikleri eksiksiz doldurunuz.")

        # 'price' dışındaki tüm sütunları al
        x_cols = [col for col in raw_df.columns if col != 'price']

        user_inputs = {}
        
        # Giriş formunu düzgün düzenlemek için 2 alt sütuna bölüyoruz
        sub_col1, sub_col2 = st.columns(2)

        for idx, col in enumerate(x_cols):
            target_sub = sub_col1 if idx % 2 == 0 else sub_col2
            
            if raw_df[col].dtype in ['int64', 'float64']:
                min_val = float(raw_df[col].min())
                max_val = float(raw_df[col].max())
                median_val = float(raw_df[col].median())
                
                # Tam sayı veya kesirli sayı ayrımı
                if raw_df[col].dtype == 'int64':
                    user_inputs[col] = target_sub.number_input(
                        f"{col.upper()}", 
                        min_value=int(min_val), 
                        max_value=int(max_val),
                        value=int(median_val),
                        step=1
                    )
                else:
                    user_inputs[col] = target_sub.number_input(
                        f"{col.upper()}", 
                        min_value=0.0, 
                        value=median_val
                    )
            else:
                unique_options = list(raw_df[col].dropna().unique())
                user_inputs[col] = target_sub.selectbox(
                    f"{col.upper()}", 
                    options=unique_options
                )

        st.write("")
        predict_btn = st.button("🔮 Fiyatı Tahmin Et", type="primary", use_container_width=True)

    with col_result:
        st.subheader("📊 Tahmin Sonucu ve Model Analizi")
        
        if predict_btn:
            # Tüm girdileri içeren tek satırlık DataFrame oluştur
            raw_input = pd.DataFrame([user_inputs])
            
            # Türetilmiş özellikleri ekle
            processed_input = create_features(raw_input)
            
            # Model Tahmini
            pred_log = best_model.predict(processed_input)[0]
            pred_price = np.expm1(pred_log)

            lower_bound = max(0, pred_price - test_mae)
            upper_bound = pred_price + test_mae

            st.success("Tahmin Başarıyla Üretildi!")
            
            st.metric(
                label="Tahmini Piyasa Değeri",
                value=f"{pred_price:,.2f} TL"
            )

            st.warning(
                f"📢 **Tam Model Güvenilirlik Bilgisi:**\n\n"
                f"- Model tüm konumsal ve donanımsal değişkenleri hesaba katmaktadır.\n"
                f"- Veri seti geneli ortalama sapma (MAE): **±{test_mae:,.2f} TL**\n"
                f"- **Olası Fiyat Aralığı:** `{lower_bound:,.0f} TL` — `{upper_bound:,.0f} TL`"
            )

            st.divider()

            m1, m2 = st.columns(2)
            m1.metric("Model Yüzdesel Hatalılık (MAPE)", f"%{test_mape:.2f}")
            m2.metric("Model Açıklayıcılık Oranı (R²)", f"%{test_r2 * 100:.1f}")

        else:
            st.info("Sol taraftaki tüm ev özelliklerini seçip **'Fiyatı Tahmin Et'** butonuna basınız.")
else:
    st.error("Veri setinizde 'price' sütunu bulunamadı. Lütfen 'price' sütununu içeren geçerli bir CSV yükleyin.")
