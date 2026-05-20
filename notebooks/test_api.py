# Simuler ce que fera l'API en Lab 3

import joblib
import pandas as pd

model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

print(f"Modele recharge : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")


# NOUVEAU PATIENT
nouveau_patient = {
    "age": 28,
    "sexe": "F",
    "temperature": 39.5,
    "tension_sys": 110,
    "toux": True,
    "fatigue": True,
    "maux_tete": True,
    "frissons": True,
    "nausee": False,
    "region": "Dakar"
}

sexe_enc = le_sexe_loaded.transform([nouveau_patient["sexe"]])[0]
region_enc = le_region_loaded.transform([nouveau_patient["region"]])[0]

features = [
    nouveau_patient["age"],
    sexe_enc,
    nouveau_patient["temperature"],
    nouveau_patient["tension_sys"],
    int(nouveau_patient["toux"]),
    int(nouveau_patient["fatigue"]),
    int(nouveau_patient["maux_tete"]),
    int(nouveau_patient["frissons"]),
    int(nouveau_patient["nausee"]),
    region_enc
]

# PREDICTION
diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print("\n--- Résultat du pré-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilité : {proba_max:.1%}")

print("\nProbabilités par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = "#" * int(proba * 30)
    print(f"{classe:10s} : {proba:.1%} {bar}")