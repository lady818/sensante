import json
from urllib import request


API_URL = "http://localhost:8000/predict"

patients = [
    {
        "label": "Enfant de 5 ans, sans symptomes",
        "payload": {
            "age": 5,
            "sexe": "F",
            "temperature": 36.8,
            "tension_sys": 100,
            "toux": False,
            "fatigue": False,
            "maux_tete": False,
            "frissons": False,
            "nausee": False,
            "region": "Dakar",
        },
        "attendu": "sain",
    },
    {
        "label": "Adulte de 40 ans, forte fievre, toux, fatigue, maux de tete",
        "payload": {
            "age": 40,
            "sexe": "M",
            "temperature": 40.0,
            "tension_sys": 100,
            "toux": True,
            "fatigue": True,
            "maux_tete": True,
            "frissons": True,
            "nausee": False,
            "region": "Dakar",
        },
        "attendu": "paludisme probable",
    },
    {
        "label": "Jeune de 20 ans, fievre 38.5C, fatigue seule",
        "payload": {
            "age": 20,
            "sexe": "M",
            "temperature": 38.5,
            "tension_sys": 110,
            "toux": False,
            "fatigue": True,
            "maux_tete": False,
            "frissons": False,
            "nausee": False,
            "region": "Dakar",
        },
        "attendu": "grippe ou typhoide possible",
    },
]


def call_api(payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    for patient in patients:
        result = call_api(patient["payload"])
        print("=" * 70)
        print(patient["label"])
        print(f"Attendu : {patient['attendu']}")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
