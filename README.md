# Segmentation urbaine : fusion de regions (Projet 5)

Pipeline : image -> K-means (couleurs) -> composantes connexes -> graphe d'adjacence (RAG, NetworkX) -> fusion de Beveridge.

Le programme ne nomme pas les objets (route, batiment) : il fusionne des regions d'apparence proche.

## Installation (Python 3.12 avec Tkinter)
```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Lancer l'interface
Double-clic sur `LANCER.bat`, ou :
```
.\.venv\Scripts\python.exe interface.py
```
Utiliser les boutons dans l'ordre : 1. K-means, 2. Regions + RAG, 3. Fusion Beveridge.

## Tests
```
.\.venv\Scripts\python.exe -m unittest discover -s moteur -p "test_*.py" -v
```

## Balayage pour le rapport
```
.\.venv\Scripts\python.exe experiments\sweep.py images\MON_IMAGE.jpg --k 8 --t 0.1 0.15 0.2 0.3
```
Resultats dans `experiments/output/` (CSV + images).

## Dossiers
- `moteur/` : calculs (K-means, regions, RAG, Beveridge) et tests
- `images/` : images d'essai
- `experiments/` : script de balayage
- `images/dronedeploy/` : extraits DroneDeploy pour les essais
- `images/mapillary/` : photos et labels Mapillary (choisir une photo, pas un label)

## Version retenue
Base Claude revue et testee. Les 7 tests du moteur passent. Aucun entrainement supervise : K-means est recalcule pour chaque image. Le balayage compare les parametres de Beveridge, pas plusieurs criteres de fusion. La comparaison des criteres et l'analyse des resultats restent a preparer pour le rapport.

Les labels ne sont pas utilises automatiquement par cette interface. Les resultats sauvegardes depuis la fenetre vont dans resultats/.
