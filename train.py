"""
train.py
==========
Modeli SADECE burada, Housing.csv'nin bulunduğu ortamda (kendi bilgisayarında/
Colab'da) eğitip model.pkl olarak kaydet. Bu dosyayı model_utils.py ile birlikte
repo'ya ekle. Streamlit Cloud'a deploy edilen app.py, Housing.csv'ye hiç ihtiyaç
duymadan bu pkl'i okuyup tahmin yapar.

Kullanım:
    python train.py
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

from model_utils import build_prep_pipeline


def main():
    df = pd.read_csv("Housing.csv")
    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_train_model = np.log1p(y_train)
    y_test_model = np.log1p(y_test)

    pipeline = Pipeline([
        ("prep", build_prep_pipeline()),
        ("model", LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1))
    ])
    pipeline.fit(X_train, y_train_model)

    y_pred = np.expm1(pipeline.predict(X_test))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test MAE (gerçek fiyat ölçeğinde): {mae:,.2f}")

    joblib.dump(pipeline, "model.pkl")
    print("model.pkl kaydedildi. model_utils.py ile BİRLİKTE repo'ya ekle.")


if __name__ == "__main__":
    main()
