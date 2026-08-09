import os
import joblib
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
import streamlit as st

from model_utils import FURNISHING_ORDER, build_prep_pipeline

st.set_page_config(page_title="Ev Fiyat Tahmini", layout="wide")


@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if os.path.exists("Housing.csv"):
        return pd.read_csv("Housing.csv")
    return None


@st.cache_resource
def load_saved_model():
    if os.path.exists("model.pkl"):
        try:
            return joblib.load("model.pkl")
        except Exception as e:
            st.error(f"Model yüklenirken sürüm hatası oluştu: {e}")
            return None
    return None


@st.cache_resource
def train_all_models(X_train, y_train_model, X_test, y_test_model):
    models = {
        "Ridge (Baseline)": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.1, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1
        ),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fitted_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([("prep", build_prep_pipeline()), ("model", model)])
        cv_res = cross_validate(
            pipe,
            X_train,
            y_train_model,
            cv=kf,
            scoring="neg_mean_absolute_error",
            return_train_score=True,
            n_jobs=-1,
        )
        pipe.fit(X_train, y_train_model)
        y_pred_test = pipe.predict(X_test)
        test_mae = mean_absolute_error(y_test_model, y_pred_test)

        results.append(
            {
                "Model": name,
                "CV Val MAE (log)": round(-cv_res["test_score"].mean(), 4),
                "CV Train MAE (log)": round(-cv_res["train_score"].mean(), 4),
                "Test MAE (log)": round(test_mae, 4),
            }
        )
        fitted_pipelines[name] = pipe

    results_df = (
        pd.DataFrame(results)
        .sort_values("CV Val MAE (log)")
        .reset_index(drop=True)
    )
    best_name = results_df.iloc[0]["Model"]
    return results_df, fitted_pipelines, best_name


# SIDEBAR
st.sidebar.title("Ev Fiyat Tahmini")
uploaded = st.sidebar.file_uploader(
    "Housing.csv yükle (opsiyonel — Analiz ve Model Karşılaştırma için)",
    type="csv",
)

df = load_data(uploaded)
data_available = df is not None

X_train = X_test = y_train = y_test = y_train_model = y_test_model = None
if data_available:
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_train_model, y_test_model = np.log1p(y_train), np.log1p(y_test)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Keşifçi Analiz", "🤖 Model Karşılaştırma", "📈 Hata Analizi", "🔮 Tahmin"]
)

# TAB 1 - Keşifçi Analiz
with tab1:
    if not data_available:
        st.info("Keşifçi analiz grafiklerini görmek için sol menüden `Housing.csv` yükleyebilirsiniz.")
    else:
        st.subheader("Hedef Değişken (Price) Dağılımı")
        col1, col2 = st.columns(2)
        skewness = y_train.skew()
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(y_train, kde=True, ax=ax, color="skyblue")
            ax.set_title(f"Orijinal Price (Skew: {skewness:.2f})")
            st.pyplot(fig)

        with col2:
            y_log = np.log1p(y_train)
            fig, ax = plt.subplots()
            sns.histplot(y_log, kde=True, ax=ax, color="teal")
            ax.set_title(f"Log1p(Price) (Skew: {y_log.skew():.2f})")
            st.pyplot(fig)

        st.subheader("Sayısal Değişken - Fiyat Korelasyonu")
        num_cols_raw = X_train.select_dtypes(include=[np.number]).columns.tolist()
        corr = (
            pd.concat([X_train[num_cols_raw], y_train], axis=1)
            .corr()["price"]
            .drop("price")
        )
        corr_sorted = corr.sort_values(key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            x=corr_sorted.values,
            y=corr_sorted.index,
            ax=ax,
            palette="coolwarm",
        )
        st.pyplot(fig)

        st.subheader("Kategorik Değişkenler")
        cat_cols_raw = X_train.select_dtypes(include=["object"]).columns.tolist()
        selected_cat = st.selectbox("İncelenecek kategorik değişken", cat_cols_raw)

        temp = pd.DataFrame({"feature": X_train[selected_cat], "target": y_train})
        groups = [g["target"].values for _, g in temp.groupby("feature")]
        f_stat, p_val = stats.f_oneway(*groups)

        fig, ax = plt.subplots(figsize=(6, 4))
        order = (
            temp.groupby("feature")["target"]
            .median()
            .sort_values(ascending=False)
            .index
        )
        sns.boxplot(
            data=temp,
            x="feature",
            y="target",
            order=order,
            ax=ax,
            palette="crest",
        )
        ax.set_title(f"{selected_cat} - Price (ANOVA p={p_val:.4f})")
        st.pyplot(fig)

# TAB 2 - Model Karşılaştırma
with tab2:
    if not data_available:
        st.info("Modelleri karşılaştırmak için sol menüden `Housing.csv` yükleyebilirsiniz.")
    else:
        st.subheader("Model Karşılaştırma (5-Fold CV, log-hedef üzerinde MAE)")
        with st.spinner("Modeller eğitiliyor..."):
            results_df, fitted_pipelines, best_name = train_all_models(
                X_train, y_train_model, X_test, y_test_model
            )
        st.dataframe(results_df, use_container_width=True)
        st.success(f"En iyi model (en düşük CV Val MAE): **{best_name}**")

        best_pipeline = fitted_pipelines[best_name]
        if hasattr(best_pipeline.named_steps["model"], "feature_importances_"):
            st.subheader("Feature Importance")
            feature_names = (
                best_pipeline.named_steps["prep"]
                .named_steps["preprocessor"]
                .get_feature_names_out()
            )
            importances = best_pipeline.named_steps["model"].feature_importances_
            imp_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": importances})
                .sort_values("Importance", ascending=False)
                .head(10)
            )

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(
                x="Importance",
                y="Feature",
                data=imp_df,
                ax=ax,
                palette="viridis",
            )
            st.pyplot(fig)

# TAB 3 - Hata Analizi
with tab3:
    if not data_available:
        st.info("Hata analizi yapmak için sol menüden `Housing.csv` yükleyebilirsiniz.")
    else:
        st.subheader("Test Kümesi Final Değerlendirme")
        with st.spinner("Hesaplanıyor..."):
            _, fitted_pipelines, best_name = train_all_models(
                X_train, y_train_model, X_test, y_test_model
            )
        best_pipeline = fitted_pipelines[best_name]

        y_pred_log = best_pipeline.predict(X_test)
        y_pred = np.expm1(y_pred_log)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:,.0f}")
        c2.metric("RMSE", f"{rmse:,.0f}")
        c3.metric("R²", f"{r2:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax, color="indigo")
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
            ax.set_xlabel("Gerçek")
            ax.set_ylabel("Tahmin")
            ax.set_title("Gerçek vs Tahmin")
            st.pyplot(fig)

        with col2:
            residuals = y_test - y_pred
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=residuals, alpha=0.6, ax=ax, color="darkred")
            ax.axhline(0, color="black", linestyle="--")
            ax.set_xlabel("Gerçek Fiyat")
            ax.set_ylabel("Hata")
            ax.set_title("Fiyata Göre Hata")
            st.pyplot(fig)

# TAB 4 - SADECE MODEL.PKL İLE ÇALIŞAN TAHMİN SEKMESİ
with tab4:
    st.subheader("Yeni Bir Ev İçin Fiyat Tahmini")

    active_pipeline = load_saved_model()

    if active_pipeline is None and data_available:
        with st.spinner("model.pkl bulunamadı, yüklenen veriden model eğitiliyor..."):
            _, fitted_pipelines, best_name = train_all_models(
                X_train, y_train_model, X_test, y_test_model
            )
            active_pipeline = fitted_pipelines[best_name]

    if active_pipeline is None:
        st.error(
            "Tahmin yapabilmek için `model.pkl` dosyasının dizinde bulunması gerekir. "
            "Lütfen `python train_and_save.py` çalıştırarak modeli oluşturun."
        )
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            area = st.number_input(
                "Alan (sqft)", min_value=500, max_value=20000, value=5000
            )
            bedrooms = st.number_input(
                "Yatak Odası", min_value=1, max_value=10, value=3
            )
            bathrooms = st.number_input("Banyo", min_value=1, max_value=5, value=2)
        with c2:
            stories = st.number_input(
                "Kat Sayısı", min_value=1, max_value=5, value=2
            )
            parking = st.number_input("Otopark", min_value=0, max_value=5, value=1)
            furnishingstatus = st.selectbox("Eşya Durumu", FURNISHING_ORDER)
        with c3:
            mainroad = st.selectbox("Ana Yola Yakın", ["yes", "no"])
            prefarea = st.selectbox("Tercih Edilen Bölge", ["yes", "no"])
            airconditioning = st.selectbox("Klima", ["yes", "no"])

        guestroom = st.selectbox("Misafir Odası", ["yes", "no"])
        basement = st.selectbox("Bodrum", ["yes", "no"])
        hotwaterheating = st.selectbox("Sıcak Su Isıtma", ["yes", "no"])

        if st.button("Tahmin Et", type="primary"):
            row = pd.DataFrame(
                [
                    {
                        "area": area,
                        "bedrooms": bedrooms,
                        "bathrooms": bathrooms,
                        "stories": stories,
                        "parking": parking,
                        "mainroad": mainroad,
                        "guestroom": guestroom,
                        "basement": basement,
                        "hotwaterheating": hotwaterheating,
                        "airconditioning": airconditioning,
                        "prefarea": prefarea,
                        "furnishingstatus": furnishingstatus,
                    }
                ]
            )
            pred_log = active_pipeline.predict(row)[0]
            pred = np.expm1(pred_log)
            st.success(f"Tahmini Fiyat: **{pred:,.0f} TL**")
