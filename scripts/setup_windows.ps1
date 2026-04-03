$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot "venv"

function Get-PythonCommand {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py -3"
    }

    throw "Python n'est pas disponible dans le PATH. Reinstalle Python en cochant 'Add Python to PATH'."
}

$pythonCmd = Get-PythonCommand

if (Test-Path $venvPath) {
    Write-Host "Suppression de l'ancien environnement virtuel..."
    Remove-Item -LiteralPath $venvPath -Recurse -Force
}

Write-Host "Creation du nouvel environnement virtuel..."
Invoke-Expression "$pythonCmd -m venv `"$venvPath`""

$venvPython = Join-Path $venvPath "Scripts\python.exe"

Write-Host "Mise a jour de pip..."
& $venvPython -m pip install --upgrade pip

Write-Host "Installation des dependances..."
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")

Write-Host ""
Write-Host "Environnement pret."
Write-Host "Commande suivante :"
Write-Host "powershell -ExecutionPolicy Bypass -File .\scripts\run_exploration.ps1"
