# Reprendre le projet sur un autre ordinateur

## Installation

Extraire le ZIP entier dans un dossier local. Python 3.12 est la version utilisee
pour nos essais. Ouvrir PowerShell dans le dossier extrait :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --timeout 300 --retries 10 -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -v
.\.venv\Scripts\python.exe verify_project.py --root outputs/city_k8 --image city.png
```

Les tests doivent afficher OK, puis la verification doit afficher Audit PASS.
Les chemins relatifs du RAG permettent de retrouver les sources apres extraction.
Conserver l organisation interne des fichiers. Une installation de paquets
necessite une connexion Internet; le ZIP ne contient pas l environnement .venv.

## Comprendre puis ameliorer

Lire GUIDE_PROJET_DARIJA.md pour les explications et commandes de chaque etape,
et AUDIT_TECHNIQUE.md pour les controles et limites. Le PDF du sujet est inclus.
Les deux PDF de cours de l enseignant ne sont pas inclus : les partager separement.

Etat : K-means, regions connexes, RAG, fusion experimentale, cinq variantes et
verification implementes. Le critere de Beveridge reste a implementer; la fusion
actuelle ne doit pas etre presentee comme Beveridge. Le rapport et la presentation
finale restent a produire.

Priorites conseillees :

1. Documenter puis implementer Beveridge avec variance, perimetres et une regle
   d arret explicite. References universitaires dans le guide.
2. Tester une image urbaine sans noms/icones et plusieurs valeurs de K.
3. Comparer aux resultats de reference sans les remplacer.
4. Executer les tests et verifier les sorties apres modification.
5. Consolider une version commune avant de rediger rapport et presentation.

Pour une nouvelle experience, utiliser --output-root avec un nouveau dossier;
les commandes completes sont dans le guide. Ne pas partager .venv ou __pycache__.

## Travailler a deux

Pour le premier transfert, le ZIP suffit. Pour echanger regulierement des
modifications, utiliser un depot Git prive commun et une branche par modification.
Conserver sources, tests, documentation et requirements dans Git. Garder les images
et gros resultats dans un dossier partage si necessaire, avec les memes chemins
relatifs; ne pas versionner .venv, tmp ni les archives ZIP.

Avant de fusionner une amelioration, noter le changement, ses parametres, les tests
passes et les resultats obtenus. La copie initiale du ZIP sert de reference.
