$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot "venv\Scripts\python.exe"
$scriptFile = Join-Path $projectRoot "notebooks\exploration.py"
$dataDir = Join-Path $projectRoot "data"

if (-not (Test-Path $venvPython)) {
    throw "Environnement virtuel introuvable. Lance d'abord .\scripts\setup_windows.ps1"
}

if (-not (Test-Path $dataDir)) {
    throw "Dossier de donnees introuvable : $dataDir"
}

$dataFile = Get-ChildItem $dataDir -Filter "patients_dakar*.csv" | Sort-Object Name | Select-Object -First 1

if ($null -eq $dataFile) {
    throw "Aucun fichier de donnees compatible trouve dans $dataDir"
}

$headerBytes = [System.IO.File]::ReadAllBytes($dataFile.FullName)[0..3]
$headerText = [System.Text.Encoding]::ASCII.GetString($headerBytes)

if ($headerText -eq "%PDF") {
    throw "Le fichier $($dataFile.FullName) est un PDF renomme. Remplace-le par un vrai CSV."
}

& $venvPython $scriptFile
