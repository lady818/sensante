from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "patients_dakar.csv"
MODEL_PATH = BASE_DIR / "models" / "sensante_model.joblib"

TARGET_COLUMN = "diagnostic"
NUMERIC_COLUMNS = [
    "age",
    "temperature",
    "tension_sys",
    "toux",
    "fatigue",
    "maux_tete",
    "frissons",
    "nausee",
]
CATEGORICAL_COLUMNS = ["sexe", "region"]


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    df = load_dataset()

    X = df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    joblib.dump(pipeline, MODEL_PATH)

    print("=" * 60)
    print("SENSANTE - ETAPE 2 : TRAIN + SERIALIZE")
    print("=" * 60)
    print(f"\nDataset charge : {len(df)} lignes")
    print(f"Train set : {len(X_train)} lignes")
    print(f"Test set  : {len(X_test)} lignes")
    print(f"Classes cibles : {sorted(y.unique())}")

    print(f"\nAccuracy : {accuracy:.4f}")

    print("\n--- Matrice de confusion ---")
    print(matrix)

    print("\n--- Classification report ---")
    print(report)

    print(f"Modele sauvegarde dans : {MODEL_PATH}")


if __name__ == "__main__":
    main()
