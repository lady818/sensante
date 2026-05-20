"""
SenSante - Exploration du dataset patients_dakar.csv
Lab 1 : Git, Python et structure de projet
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def resolve_data_path() -> Path:
    preferred = DATA_DIR / "patients_dakar.csv"
    if preferred.exists():
        return preferred

    matches = sorted(DATA_DIR.glob("patients_dakar*.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Aucun fichier CSV trouve dans {DATA_DIR}. "
        "Attendu: patients_dakar.csv"
    )


def load_dataset() -> pd.DataFrame:
    data_path = resolve_data_path()

    # Evite un message pandas trompeur si le fichier est en fait un PDF renomme.
    with data_path.open("rb") as file:
        header = file.read(4)
    if header == b"%PDF":
        raise ValueError(
            f"{data_path} n'est pas un CSV valide. Le fichier commence par %PDF."
        )

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def main() -> None:
    df = load_dataset()

    print("=" * 50)
    print("SENSANTE - Exploration du dataset")
    print("=" * 50)

    print(f"\nNombre de patients : {len(df)}")
    print(f"Nombre de colonnes : {df.shape[1]}")
    print(f"Colonnes : {list(df.columns)}")

    print("\n--- 5 premiers patients ---")
    print(df.head())

    print("\n--- Statistiques descriptives ---")
    print(df.describe(include="all").round(2))

    if "diagnostic" in df.columns:
        print("\n--- Repartition des diagnostics ---")
        diag_counts = df["diagnostic"].value_counts()
        for diag, count in diag_counts.items():
            pct = count / len(df) * 100
            print(f"{diag:12s} : {count:3d} patients ({pct:.1f}%)")

    if "region" in df.columns:
        print("\n--- Repartition par region (top 5) ---")
        region_counts = df["region"].value_counts().head(5)
        for region, count in region_counts.items():
            print(f"{region:15s} : {count:3d} patients")

    if {"diagnostic", "temperature"}.issubset(df.columns):
        print("\n--- Temperature moyenne par diagnostic ---")
        temp_by_diag = df.groupby("diagnostic")["temperature"].mean()
        for diag, temp in temp_by_diag.items():
            print(f"{diag:12s} : {temp:.1f} C")

    if {"sexe", "diagnostic"}.issubset(df.columns):
        print("\n--- Nombre de patients par sexe et diagnostic ---")
        counts_by_sex_diag = df.groupby(["sexe", "diagnostic"]).size()
        for (sexe, diagnostic), count in counts_by_sex_diag.items():
            print(f"{sexe} - {diagnostic:12s} : {count:3d} patients")

    print(f"\n{'= ' * 25}")
    print("Exploration terminee !")
    print("Prochain lab : entrainer un modele ML")
    print(f"{'= ' * 25}")


if __name__ == "__main__":
    main()
