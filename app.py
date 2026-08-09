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
# 1. YARDIMCI FONKSİYONLAR & 2 AŞAMALI MODEL EĞİTİMİ (CACHED)
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

def build_pipeline(num_cols, cat_cols):
    """Pipeline mimarisini oluşturur."""
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

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LGBMRegressor(random_state=42, verbose=-1))
    ])

@st.cache_resource(show_spinner="1/2: Tüm veri ile Feature Importance hesaplanıyor...")
def get_top_4_features(data):
    """1. AŞAMA: Tüm değişkenlerle ön modeli eğitip en önemli 4 değişkeni bulur."""
    X = data.drop(columns=['price'])
    y = np.log1p(data['price'])

    X_fe = create_features(X)

    num_cols = X_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_fe.select_dtypes(include=['object', 'category']).columns.tolist()

    pipeline = build_pipeline(num_cols, cat_cols)
    pipeline.fit(X_fe, y)

    # Feature Importance Okuma
    importances = pipeline.named_steps['model'].feature_importances_
    ohe_cat_cols = list(pipeline.named_steps['preprocessor']
                        .named_transformers_['cat']
                        .named_steps['onehot']
                        .get_feature_names_out(cat_cols))
    all_transformed_cols = num_cols + ohe_cat_cols

    col_importance_map = {}
    for col, imp in zip(all_transformed_cols, importances):
        orig_col = col.split('_')[0] if '_' in col and col not in X.columns else col
        if orig_col in X.columns:
            col_importance_map[orig_col] = col_importance_map.get(orig_col, 0) + imp

    # En yüksek öneme sahip ilk 4 orijinal değişken
    sorted_features = sorted(col_importance_map.items(), key=lambda x: x[1], reverse=True)
    top_4 = [item[0] for item in sorted_features[:4]]
    return top_4

@st.cache_resource(show_spinner="2/2: Sadece en önemli 4 değişken ile yeni model eğitiliyor...")
def train_model_with_top_4(file_bytes, file_name):
    """2. AŞAMA: Diğer tüm değişkenleri veri setinden çıkartıp sadece 4 değişkenle modeli kurar."""
    if file_bytes is not None:
        raw_data = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raw_data = generate_default_data()

    if 'price' not in raw_data.columns:
        return None, None, None, None, None, raw_data

    # En önemli 4 değişkeni tespit et
    top_4_features = get_top_4_features(raw_data)

    # Diğer tüm değişkenleri veri setinden ÇIKART
    selected_cols = top_4_features + ['price']
    filtered_df = raw_data[selected_cols].copy()

    X = filtered_df.drop(columns=['price'])
    y = filtered_df['price']
    y_log = np.log1p(y)

    # Türetilmiş özellikleri sadece bu 4 değişken üzerinden yap
    X_fe = create_features(X)

    num_cols = X_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_fe.select_dtypes(include=['object', 'category']).columns.tolist()

    base_pipeline = build_pipeline(num_cols, cat_cols)

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
    final_pipeline = search.best_estimator_

    y_pred_log = final_pipeline.predict(X_test)
    y_pred_orig = np.expm1(y_pred_log)

    test_mae = mean_absolute_error(y_test_orig, y_pred_orig)
    test_mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig) * 100
    test_r2 = r2_score(y_test_orig, y_pred_orig)

    return final_pipeline, test_mae, test_mape, test_r2, top_4_features, filtered_df

# ==============================================================================
# 2. STREAMLIT ARAYÜZÜ
# ==============================================================================
st.title("🏠 Yapay Zekâ Destekli Ev Fiyat Tahmin Paneli")
st.caption("Gereksiz Sütunlar Çıkarıldı: Yalnızca En Önemli 4 Değişken ile Yeniden Eğitilen Model")

# YAN MENÜ: CSV Yükleme
with st.sidebar:
    st.header("📁 Veri Seti Ayarları")
    uploaded_file = st.file_uploader("Kendi CSV Dosyanızı Yükleyin", type=["csv"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        st.success(f"✅ `{file_name}` yüklendi.")
    else:
        file_bytes = None
        file_name = "default_data.csv"
        st.info("💡 Varsayılan veri seti kullanılmaktadır.")

# SADECE 4 DEĞİŞKENLİ MODELİ EĞİT
best_model, test_mae, test_mape, test_r2, top_4_features, filtered_df = train_model_with_top_4(file_bytes, file_name)

st.divider()

if best_model is not None:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("📋 Modelin Eğitildiği En Önemli 4 Değişken")
        st.success(f"Geriye kalan tüm sütunlar çıkarıldı. Model sadece şu 4 sütun ile kuruldu: **{', '.join(top_4_features)}**")

        user_inputs = {}
        for col in top_4_features:
            if filtered_df[col].dtype in ['int64', 'float64']:
                min_v = float(filtered_df[col].min())
                max_v = float(filtered_df[col].max())
                median_v = float(filtered_df[col].median())
                user_inputs[col] = st.number_input(
                    f"{col.upper()} (Min: {min_v:,.0f} - Max: {max_v:,.0f})", 
                    min_value=0.0, 
                    value=median_v
                )
            else:
                unique_vals = list(filtered_df[col].dropna().unique())
                user_inputs[col] = st.selectbox(f"{col.upper()}", options=unique_vals)

        predict_btn = st.button("🔮 Fiyatı Tahmin Et", type="primary", use_container_width=True)

    with col_result:
        st.subheader("📊 Tahmin Sonucu ve 4 Değişkenli Model Performansı")
        
        if predict_btn:
            # Doğrudan sadece bu 4 değişkeni içeren DataFrame oluşturulur (Eksik sütun hatası vermez)
            raw_input = pd.DataFrame([user_inputs])
            processed_input = create_features(raw_input)
            
            pred_log = best_model.predict(processed_input)[0]
            pred_price = np.expm1(pred_log)

            st.success("Tahmin Başarıyla Üretildi!")
            st.metric(label="Tahmini Piyasa Değeri", value=f"{pred_price:,.2f} TL")

            st.warning(
                f"📢 **4 Değişkenli Model Bilgileri:**\n\n"
                f"- Model sadece 4 ana değişkene dayalı eğitildiği için ortalama hata payı: **%{test_mape:.2f}**\n"
                f"- Ortalama Sapma Miktarı (MAE): **±{test_mae:,.2f} TL**"
            )

            st.divider()

            m1, m2 = st.columns(2)
            m1.metric("Model Yüzdesel Hatalılık (MAPE)", f"%{test_mape:.2f}")
            m2.metric("Model Açıklayıcılık Oranı (R²)", f"%{test_r2 * 100:.1f}")

        else:
            st.info("Sol taraftan 4 özelliği girip **'Fiyatı Tahmin Et'** butonuna basınız.")
else:
    st.error("Veri setinizde 'price' sütunu bulunamadı. Lütfen 'price' sütununu içeren geçerli bir CSV yükleyin.")
