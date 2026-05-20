---
title: Sensante
emoji: "📊"
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
---

# SenSante

Assistant de pre-diagnostic medical pour le Senegal.

## Demo en ligne

https://mamesadio-sensante.hf.space

## Stack

- scikit-learn (modele ML)
- FastAPI (API REST)
- Tailwind CSS (frontend responsive)
- Groq / Llama 3 (explication LLM)
- Docker (conteneurisation)

## Description

SenSante utilise le machine learning pour aider au pre-diagnostic des
maladies courantes (paludisme, grippe, typhoide) a partir des symptomes
du patient.

## Structure du projet

- `data/` : donnees patients au format CSV
- `models/` : modele ML serialise
- `api/` : API FastAPI
- `frontend/` : interface web
- `notebooks/` : scripts d'exploration
- `scripts/` : scripts utilitaires Windows

## Prerequis

- Windows PowerShell
- Python 3.11+ installe avec l'option `Add Python to PATH`

## Installation rapide

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

Ce script :

- verifie qu'une commande Python est disponible
- recree `venv/` proprement
- installe les dependances de `requirements.txt`

## Lancer l'exploration

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_exploration.ps1
```

## Attention au fichier de donnees
Le script attend un vrai fichier CSV dans :

`data/patients_dakar.csv`

Si ce fichier est en realite un PDF renomme en `.csv`, l'exploration
s'arretera avec un message explicite. Dans ce cas, remplace ce fichier
par le vrai export CSV avant de relancer.

## Auteur
Mame Sadio Guisse - L2 GLSI B - ESP/UCAD

## Cours
Integration de Modeles IA - Dr. El Hadji Bassirou TOURE
