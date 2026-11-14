# Fusion experimentale des regions

Executer depuis le dossier du projet :

```powershell
.\.venv\Scripts\python.exe merge_regions.py
```

Cette implementation est une base experimentale. Elle ne pretend pas implementer
le critere de Beveridge, dont la formule reste a obtenir dans une source de cours.

## Regle utilisee

Deux regions ne peuvent fusionner que si elles partagent un cote de pixel
(voisinage 4) et si les deux conditions suivantes sont satisfaites :

- Distance euclidienne entre leurs moyennes RGB <= 20.
- Contraste moyen sur leur frontiere commune <= 40.

Le contraste est la moyenne des distances RGB entre les pixels qui se font face
de part et d'autre de la frontiere. Ce n'est pas une magnitude de Sobel.
Les valeurs 20 et 40 sont des parametres experimentaux, pas des constantes du cours.

Les paires ayant au moins une region de moins de 10 pixels ont la priorite.
Elles restent soumises aux deux conditions precedentes. Dans chaque groupe de
priorite, on choisit le plus petit score : distance/20 + contraste/40.
La taille intervient dans l'ordre des fusions, pas comme une obligation de fusionner.

Apres chaque fusion, le programme met a jour la surface, la somme et la moyenne
des couleurs, le centroide, les voisins et les longueurs de frontiere.
Les contrastes des frontieres combinees sont ponderes par leur longueur.
Les decisions devenues anciennes sont abandonnees et recalculees.
L'algorithme s'arrete lorsqu'aucune paire adjacente ne satisfait les deux seuils.
L'ordre des fusions peut influencer la partition finale.

## Resultats et controles

Les sorties sont dans un sous-dossier `fusion_c20_b40_s10` du dossier du RAG.
Les fichiers d'entree sont conserves.

- `region_labels.npy` : partition fusionnee, numeros consecutifs a partir de 0.
- `initial_to_merged.npy` : correspondance de chaque ancienne region vers la nouvelle.
- `segmentation_after.png` : chaque region affichee avec sa moyenne RGB.
- `comparison.png` : original, regions avant fusion et regions apres fusion.
  Les vues avant/apres utilisent toutes deux la moyenne RGB originale de chaque
  region, afin de comparer le meme mode de rendu.
- `rag_after.graphml.gz` : graphe complet compresse, lisible avec NetworkX.
- `rag_after.png`, `rag_detail_after.png` : vues du graphe.
- `summary.json` : parametres, nombres de regions et temps de fusion.

Les surfaces et longueurs sont mesurees dans la grille de l'image redimensionnee
(plus grand cote : 960 pixels), et non en metres.

Le programme reconstruit independamment le RAG a partir des nouvelles etiquettes
et compare ses surfaces, moyennes, aretes, longueurs et contrastes au graphe
obtenu par contraction. Les tests se lancent avec :

```powershell
.\.venv\Scripts\python.exe -m unittest test_merge_regions -v
```

Une reduction du nombre de regions mesure la simplification, pas la precision
semantique. Les routes, batiments et pietons ne sont pas automatiquement nommes.
La comparaison experimentale des criteres est implementee dans `compare_fusion.py`.
Le choix definitif des seuils demande encore une evaluation sur plusieurs images.

## Comparaison des variantes

`compare_fusion.py` compare cinq variantes en repartant du meme graphe initial :
couleur seule, couleur et frontiere sans priorite de taille, reference avec
priorite, seuils stricts et seuils plus permissifs. Les resultats sont dans
`outputs/city_k8/connected_regions/rag/comparison_experiments/`.
Le tableau `comparison.md` definit les mesures et les limites de leur interpretation.

L'option `--criterion color` de `merge_regions.py` desactive le critere de frontiere
dans l'admissibilite et dans le score. `--small-area 0` desactive la priorite de taille.
