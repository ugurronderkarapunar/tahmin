"""
model_utils.py
================
Pipeline tanımları (feature engineering, preprocessing) burada, TEK bir yerde
tutulur. train.py modeli eğitirken bunu kullanır, app.py da tahmin yaparken
aynı yerden import eder.

NEDEN AYRI DOSYA GEREKLİ:
joblib/pickle, bir Pipeline içindeki FunctionTransformer'ı kaydederken fonksiyonun
KENDİSİNİ değil, "hangi modülde tanımlandığını" kaydeder. create_features
train.py'nin __main__'inde tanımlıysa, model.pkl'i başka bir script (app.py)
içinde yüklemeye çalıştığında Python bu fonksiyonu bulamaz ve
"Can't get attribute 'create_features' on <module '__main__'>" hatası verir.
Fonksiyonu ortak, import edilebilir bir modülde tanımlamak bu sorunu çözer.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


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
