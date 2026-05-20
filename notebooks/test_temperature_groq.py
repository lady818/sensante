import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERREUR : GROQ_API_KEY non trouvee dans .env")
    raise SystemExit(1)

client = Groq(api_key=api_key)

system_prompt = """Tu es un assistant medical senegalais.
Tu recois un diagnostic et des donnees patient.
Explique le resultat en francais simple,
comme un medecin parlerait a son patient.
Sois rassurant mais recommande toujours
une consultation medicale.
Maximum 3 phrases.
Ne fais jamais de diagnostic toi-meme.
Tu expliques uniquement le diagnostic fourni."""

user_prompt = """Patient : F, 28 ans, region Dakar
Temperature : 39.5 C
Diagnostic du modele : paludisme (probabilite 72%)
Explique ce resultat au patient."""

for temp in [0.0, 0.5, 1.0]:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=200,
        temperature=temp,
    )

    print("=" * 60)
    print(f"Temperature = {temp}")
    print("=" * 60)
    print(response.choices[0].message.content)
    print()
