from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "patients_dakar.csv"
MODELS_DIR = BASE_DIR / "models"


def main() -> None:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    le_sexe = LabelEncoder()
    le_region = LabelEncoder()

    df["sexe_encoded"] = le_sexe.fit_transform(df["sexe"])
    df["region_encoded"] = le_region.fit_transform(df["region"])

    feature_cols = [
        "age",
        "sexe_encoded",
        "temperature",
        "tension_sys",
        "toux",
        "fatigue",
        "maux_tete",
        "frissons",
        "nausee",
        "region_encoded",
    ]

    X = df[feature_cols]
    y = df["diagnostic"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.pkl")
    joblib.dump(le_sexe, MODELS_DIR / "encoder_sexe.pkl")
    joblib.dump(le_region, MODELS_DIR / "encoder_region.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")

    print("Modele et encodeurs generes dans models/")


if __name__ == "__main__":
    main()
