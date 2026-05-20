"""
SenSante - Lab 2
Entraînement et sérialisation du modèle.
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "patients_dakar.csv"


def main() -> None:
   
    # CHARGEMENT DES DONNEES
 
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    print("=" * 60)
    print("SENSANTE - LAB 2 : ENTRAINEMENT MODELE")
    print("=" * 60)

    print(f"\nDataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
    print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

   
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

    print(f"\nFeatures : {X.shape}")
    print(f"Cible : {y.shape}")

  
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"\nTrain : {X_train.shape[0]} patients")
    print(f"Test : {X_test.shape[0]} patients")

    # MODELE
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    print("\nModele entraine !")

    importances = model.feature_importances_
    for name , imp in sorted ( zip ( feature_cols , importances ),
        key = lambda x: x[1] , reverse = True ):
        print (f" { name:20s} : { imp :.3f}")

    
    # PREDICTION ET EVALUATION
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy : {accuracy:.2%}")

    
    # CREATION DOSSIER MODELS
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)


    # SAUVEGARDE MODELE
    model_path = models_dir / "model.pkl"
    joblib.dump(model, model_path)

    print(f"\nModele sauvegarde : {model_path}")

    
    # SAUVEGARDE ENCODEURS + FEATURES
    joblib.dump(le_sexe, models_dir / "encoder_sexe.pkl")
    joblib.dump(le_region, models_dir / "encoder_region.pkl")
    joblib.dump(feature_cols, models_dir / "feature_cols.pkl")

    print("Encodeurs et metadata sauvegardes.")

    # EXERCICE 2 : TESTER AVEC 3 PATIENTS FICTIFS
    loaded_model = joblib.load(model_path)
    loaded_sexe_encoder = joblib.load(models_dir / "encoder_sexe.pkl")
    loaded_region_encoder = joblib.load(models_dir / "encoder_region.pkl")
    loaded_feature_cols = joblib.load(models_dir / "feature_cols.pkl")

    patients_fictifs = pd.DataFrame(
        [
            {
                "profil": "Jeune sans symptomes",
                "age": 19,
                "sexe": "F",
                "temperature": 36.8,
                "tension_sys": 12,
                "toux": 0,
                "fatigue": 0,
                "maux_tete": 0,
                "frissons": 0,
                "nausee": 0,
                "region": "Dakar",
            },
            {
                "profil": "Adulte avec forte fievre",
                "age": 40,
                "sexe": "M",
                "temperature": 40.2,
                "tension_sys": 9,
                "toux": 0,
                "fatigue": 1,
                "maux_tete": 1,
                "frissons": 1,
                "nausee": 1,
                "region": "Dakar",
            },
            {
                "profil": "Patient age avec toux",
                "age": 70,
                "sexe": "M",
                "temperature": 38.1,
                "tension_sys": 11,
                "toux": 1,
                "fatigue": 1,
                "maux_tete": 0,
                "frissons": 0,
                "nausee": 0,
                "region": "Thiès",
            },
        ]
    )

    patients_encoded = patients_fictifs.copy()
    patients_encoded["sexe_encoded"] = loaded_sexe_encoder.transform(
        patients_encoded["sexe"]
    )
    patients_encoded["region_encoded"] = loaded_region_encoder.transform(
        patients_encoded["region"]
    )

    X_new = patients_encoded[loaded_feature_cols]
    predictions = loaded_model.predict(X_new)

    resultats = patients_fictifs[["profil", "age", "sexe", "temperature", "region"]].copy()
    resultats["diagnostic_predit"] = predictions

    print("\n--- Exercice 2 : predictions pour 3 patients fictifs ---")
    print(resultats.to_string(index=False))



if __name__ == "__main__":
    main()
