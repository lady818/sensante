from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age en annees")
    sexe: str = Field(..., description="Sexe : M ou F")
    temperature: float = Field(..., ge=35.0, le=42.0, description="Temperature en Celsius")
    tension_sys: int = Field(..., ge=60, le=250, description="Tension systolique")
    toux: bool = Field(..., description="Presence de toux")
    fatigue: bool = Field(..., description="Presence de fatigue")
    maux_tete: bool = Field(..., description="Presence de maux de tete")
    frissons: bool = Field(..., description="Presence de frissons")
    nausee: bool = Field(..., description="Presence de nausee")
    region: str = Field(..., description="Region du Senegal")


class DiagnosticOutput(BaseModel):
    diagnostic: str = Field(..., description="Diagnostic predit")
    probabilite: float = Field(..., description="Probabilite du diagnostic")
    confiance: str = Field(..., description="Niveau de confiance")
    message: str = Field(..., description="Recommandation")


app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.2.0",
)


print("Chargement du modele...")
model = joblib.load(MODELS_DIR / "model.pkl")
le_sexe = joblib.load(MODELS_DIR / "encoder_sexe.pkl")
le_region = joblib.load(MODELS_DIR / "encoder_region.pkl")
feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
print(f"Modele charge : {type(model).__name__}")
print(f"Classes : {list(model.classes_)}")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "SenSante API is running",
    }


@app.get("/model-info")
def model_info():
    return {
        "type": type(model).__name__,
        "nombre_arbres": getattr(model, "n_estimators", None),
        "classes_possibles": list(model.classes_),
        "nombre_features": getattr(model, "n_features_in_", None),
    }


@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}. Utiliser M ou F.",
        )

    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Region inconnue : {patient.region}",
        )

    features = np.array(
        [[
            patient.age,
            sexe_enc,
            patient.temperature,
            patient.tension_sys,
            int(patient.toux),
            int(patient.fatigue),
            int(patient.maux_tete),
            int(patient.frissons),
            int(patient.nausee),
            region_enc,
        ]]
    )

    diagnostic = model.predict(features)[0]
    probas = model.predict_proba(features)[0]
    proba_max = float(probas.max())

    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    messages = {
        "paludisme": "Suspicion de paludisme. Consultez un medecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandes.",
        "typhoide": "Suspicion de typhoide. Consultation medicale necessaire.",
        "sain": "Pas de pathologie detectee. Continuez a surveiller.",
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin."),
    )
