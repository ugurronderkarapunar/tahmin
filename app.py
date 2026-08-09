"""
Ev Fiyat Tahmini - Streamlit Uygulaması
==========================================
house_price_pipeline.py'deki metodolojinin (leakage-safe pipeline, log-hedef,
CV içinde preprocessing) interaktif arayüze taşınmış hali.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor


st.set_page_config(page_title="Ev Fiyat Tahmini", layout="wide")

# ==============================================================================
# SABİT TANIMLAR (Housing.csv şemasına göre)
# ==============================================================================
NUM_BASE = ["area", "bedrooms", "bathrooms", "stories", "parking"]
ORDINAL_COLS = ["furnishingstatus"]
FURNISHING_ORDER = ["unfurnished", "semi-furnished", "furnished"]
BINARY_COLS = ["mainroad", "guestroom", "basement", "hotwaterheating",
               "airconditioning", "prefarea"]
NUM_ENGINEERED = ["total_rooms", "area_per_bedroom", "area_per_room",
                   "bath_per_bed_ratio", "area_per_story"]
ALL_NUM_COLS = NUM_BASE + NUM_ENGINEERED


def create_features(df):
    X = df.copy()
    eps = 1e-6
    X["total_rooms"] = X["bedrooms"] + X["bathrooms"]
    X["area_per_bedroom"] = X["area"] / (X["bedrooms"] + eps)
    X["area_per_room"] = X["area"] / (X["total_rooms"] + eps)
    X["bath_per_bed_ratio"] = X["bathrooms"] / (X["bedrooms"] + eps)
    X["area_per_story"] = X["area"] / (X["stories"] + eps)
    return X


def build_prep_pipeline():
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal_enc", OrdinalEncoder(categories=[FURNISHING_ORDER]))
    ])
    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot_enc", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, ALL_NUM_COLS),
        ("ord", ordinal_pipeline, ORDINAL_COLS),
        ("bin", binary_pipeline, BINARY_COLS)
    ], remainder="drop")

    prep = Pipeline([
        ("feature_engineering", FunctionTransformer(create_features)),
        ("preprocessor", preprocessor)
    ])
    prep.set_output(transform="pandas")
    return prep


@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv("Housing.csv")


@st.cache_resource
def train_all_models(X_train, y_train_model, X_test, y_test_model):
    """Tüm modelleri CV ile karşılaştırır ve en iyi LightGBM'i (varsayılan model)
    tam X_train üzerinde eğitip döndürür. Streamlit her etkileşimde script'i
    yeniden çalıştırdığı için bu fonksiyon cache'lenmezse her tıklamada
    yeniden eğitim yapılır — bu yüzden cache_resource şart."""
    models = {
        "Ridge (Baseline)": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.1, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fitted_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([("prep", build_prep_pipeline()), ("model", model)])
        cv_res = cross_validate(
            pipe, X_train, y_train_model, cv=kf,
            scoring="neg_mean_absolute_error", return_train_score=True, n_jobs=-1
        )
        pipe.fit(X_train, y_train_model)
        y_pred_test = pipe.predict(X_test)
        test_mae = mean_absolute_error(y_test_model, y_pred_test)

        results.append({
            "Model": name,
            "CV Val MAE (log)": round(-cv_res["test_score"].mean(), 4),
            "CV Train MAE (log)": round(-cv_res["train_score"].mean(), 4),
            "Test MAE (log)": round(test_mae, 4),
        })
        fitted_pipelines[name] = pipe

    results_df = pd.DataFrame(results).sort_values("CV Val MAE (log)").reset_index(drop=True)
    best_name = results_df.iloc[0]["Model"]
    return results_df, fitted_pipelines, best_name


# ==============================================================================
# SIDEBAR — veri yükleme
# ==============================================================================
st.sidebar.title("Ev Fiyat Tahmini")
uploaded = st.sidebar.file_uploader("Housing.csv yükle (boş bırakılırsa repo'daki dosya kullanılır)", type="csv")

df = load_data(uploaded)
X = df.drop(columns=["price"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_model, y_test_model = np.log1p(y_train), np.log1p(y_test)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Keşifçi Analiz", "🤖 Model Karşılaştırma", "📈 Hata Analizi", "🔮 Tahmin"])

# ==============================================================================
# TAB 1 — EDA
# ==============================================================================
with tab1:
    st.subheader("Hedef Değişken (Price) Dağılımı")
    col1, col2 = st.columns(2)

    skewness = y_train.skew()
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(y_train, kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Orijinal Price (Skew: {skewness:.2f})")
        st.pyplot(fig)
        st.caption(
            "Skewness > 1 ise dağılım şiddetli sağa çarpıktır (birkaç pahalı ev "
            "ortalamayı yukarı çeker) — bu durumda log dönüşümü gerekir."
            if skewness > 1 else
            "Dağılım normale yakın veya hafif çarpık."
        )

    with col2:
        y_log = np.log1p(y_train)
        fig, ax = plt.subplots()
        sns.histplot(y_log, kde=True, ax=ax, color="teal")
        ax.set_title(f"Log1p(Price) (Skew: {y_log.skew():.2f})")
        st.pyplot(fig)
        st.caption("Log dönüşümü sonrası çarpıklık genelde belirgin şekilde azalır.")

    st.subheader("Sayısal Değişken - Fiyat Korelasyonu")
    num_cols_raw = X_train.select_dtypes(include=[np.number]).columns.tolist()
    corr = pd.concat([X_train[num_cols_raw], y_train], axis=1).corr()["price"].drop("price")
    corr_sorted = corr.sort_values(key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=corr_sorted.values, y=corr_sorted.index, ax=ax, palette="coolwarm")
    ax.set_title("Price ile Korelasyon (Pearson r)")
    st.pyplot(fig)
    st.caption(
        "0'a yakın değerler zayıf ilişkiyi, +1/-1'e yakın değerler güçlü doğrusal "
        "ilişkiyi gösterir. Bu grafikteki en yüksek |r| değerine sahip değişken, "
        "modelin muhtemelen en çok güveneceği öznitelik olacaktır."
    )

    st.subheader("Kategorik Değişkenler")
    cat_cols_raw = X_train.select_dtypes(include=["object"]).columns.tolist()
    selected_cat = st.selectbox("İncelenecek kategorik değişken", cat_cols_raw)

    temp = pd.DataFrame({"feature": X_train[selected_cat], "target": y_train})
    groups = [g["target"].values for _, g in temp.groupby("feature")]
    f_stat, p_val = stats.f_oneway(*groups)

    fig, ax = plt.subplots(figsize=(6, 4))
    order = temp.groupby("feature")["target"].median().sort_values(ascending=False).index
    sns.boxplot(data=temp, x="feature", y="target", order=order, ax=ax, palette="crest")
    ax.set_title(f"{selected_cat} - Price (ANOVA p={p_val:.4f})")
    st.pyplot(fig)
    st.caption(
        f"p-değeri {p_val:.4f} — "
        + ("0.05'in altında, yani bu değişkenin kategorileri arasında fiyat "
           "açısından istatistiksel olarak anlamlı bir fark var."
           if p_val < 0.05 else
           "0.05'in üzerinde, yani bu değişkenin fiyat üzerinde güçlü bir "
           "ayırt edici etkisi görünmüyor.")
    )

    top_ratio = X_train[selected_cat].value_counts(normalize=True).iloc[0]
    if top_ratio > 0.85:
        st.warning(f"'{selected_cat}' değişkeninde en baskın sınıf oranı %{top_ratio*100:.1f} — "
                   "modele az sinyal katıyor olabilir.")


# ==============================================================================
# TAB 2 — Model karşılaştırma
# ==============================================================================
with tab2:
    st.subheader("Model Karşılaştırma (5-Fold CV, log-hedef üzerinde MAE)")
    with st.spinner("Modeller eğitiliyor..."):
        results_df, fitted_pipelines, best_name = train_all_models(
            X_train, y_train_model, X_test, y_test_model
        )
    st.dataframe(results_df, use_container_width=True)
    st.caption(
        "CV Val MAE ile Test MAE arasındaki fark küçükse model tutarlı genelliyor demektir. "
        "CV Train MAE, CV Val MAE'den belirgin şekilde düşükse (Overfit Gap büyükse) model "
        "ezberlemeye yatkındır."
    )
    st.success(f"En iyi model (en düşük CV Val MAE): **{best_name}**")

    best_pipeline = fitted_pipelines[best_name]
    if hasattr(best_pipeline.named_steps["model"], "feature_importances_"):
        st.subheader("Feature Importance")
        feature_names = best_pipeline.named_steps["prep"].named_steps["preprocessor"].get_feature_names_out()
        importances = best_pipeline.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}) \
                   .sort_values("Importance", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Importance", y="Feature", data=imp_df, ax=ax, palette="viridis")
        st.pyplot(fig)
        st.caption(
            "En üstteki değişkenler modelin tahmin yaparken en çok dayandığı özniteliklerdir. "
            "Bu sıralamanın iş mantığıyla (örn. alanın en önemli faktör olması) örtüşmesi, "
            "modelin makul öğrendiğinin bir işaretidir."
        )


# ==============================================================================
# TAB 3 — Hata analizi
# ==============================================================================
with tab3:
    st.subheader("Test Kümesi Final Değerlendirme")
    with st.spinner("Hesaplanıyor..."):
        _, fitted_pipelines, best_name = train_all_models(X_train, y_train_model, X_test, y_test_model)
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
        ax.set_xlabel("Gerçek"); ax.set_ylabel("Tahmin")
        ax.set_title("Gerçek vs Tahmin")
        st.pyplot(fig)
        st.caption("Noktalar kırmızı çizgiye ne kadar yakınsa tahminler o kadar isabetli demektir.")

    with col2:
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=residuals, alpha=0.6, ax=ax, color="darkred")
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xlabel("Gerçek Fiyat"); ax.set_ylabel("Hata")
        ax.set_title("Fiyata Göre Hata (Heteroscedasticity Kontrolü)")
        st.pyplot(fig)
        st.caption(
            "Noktalar fiyat arttıkça huni gibi genişliyorsa (dağılıyorsa), model yüksek "
            "fiyatlı evlerde daha az güvenilir demektir."
        )


# ==============================================================================
# TAB 4 — Tekil tahmin
# ==============================================================================
with tab4:
    st.subheader("Yeni Bir Ev İçin Fiyat Tahmini")
    with st.spinner("Model hazırlanıyor..."):
        _, fitted_pipelines, best_name = train_all_models(X_train, y_train_model, X_test, y_test_model)
    best_pipeline = fitted_pipelines[best_name]
    st.caption(f"Kullanılan model: {best_name}")

    c1, c2, c3 = st.columns(3)
    with c1:
        area = st.number_input("Alan (sqft)", min_value=500, max_value=20000, value=5000)
        bedrooms = st.number_input("Yatak Odası", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Banyo", min_value=1, max_value=5, value=2)
    with c2:
        stories = st.number_input("Kat Sayısı", min_value=1, max_value=5, value=2)
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
        row = pd.DataFrame([{
            "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "stories": stories, "parking": parking, "mainroad": mainroad,
            "guestroom": guestroom, "basement": basement,
            "hotwaterheating": hotwaterheating, "airconditioning": airconditioning,
            "prefarea": prefarea, "furnishingstatus": furnishingstatus
        }])
        pred_log = best_pipeline.predict(row)[0]
        pred = np.expm1(pred_log)
        st.success(f"Tahmini Fiyat: **{pred:,.0f}**")
