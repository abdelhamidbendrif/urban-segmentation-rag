# Segmentation urbaine et fusion de regions par RAG

Projet Python : K-means sur les couleurs, composantes 4-connexes, graphe
d adjacence NetworkX, fusion experimentale et comparaison de cinq variantes.

**Etat : prototype teste. Le critere de Beveridge et le rapport final restent a
realiser. La fusion actuelle ne constitue pas Beveridge.**

## Installation

Python utilise pour les essais : 3.12.3. Dans le dossier du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --timeout 300 --retries 10 -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -v
```

## Executer la chaine complete

Les resultats sont generes localement dans `outputs/`, qui est exclu de Git.
La commande de verification demande donc d avoir execute les etapes precedentes.

```powershell
.\.venv\Scripts\python.exe kmeans_segmentation.py city.png --k 8
.\.venv\Scripts\python.exe connected_regions.py outputs/city_k8/cluster_labels.npy
.\.venv\Scripts\python.exe build_rag.py --image city.png --labels outputs/city_k8/connected_regions/region_labels.npy
.\.venv\Scripts\python.exe merge_regions.py --graph outputs/city_k8/connected_regions/rag/rag_before.graphml.gz
.\.venv\Scripts\python.exe compare_fusion.py --graph outputs/city_k8/connected_regions/rag/rag_before.graphml.gz
.\.venv\Scripts\python.exe verify_project.py --root outputs/city_k8 --image city.png
```

Les fusions peuvent prendre plusieurs minutes. La capture `city.png` provient
de Google Maps et contient textes et icones; elle sert a une experience de
prototype, pas a mesurer la precision semantique sur des objets annotes.

Resultat de reference observe : 91822 regions et 206684 aretes avant fusion,
56347 regions et 127536 aretes apres fusion avec seuils RGB 20/40 et priorite
aux regions de moins de 10 pixels. Ces nombres ne mesurent pas une accuracy.

## Documentation

- [Guide detaille en darija](GUIDE_PROJET_DARIJA.md) : concepts, calculs, commandes, presentation et rapport.
- [Audit technique](AUDIT_TECHNIQUE.md) : verifications, corrections et limites.
- [Fusion experimentale](FUSION.md) : criteres et parametres.
- [Reprise sur un autre ordinateur](PARTAGE_COLLEGUE.md).

Le ZIP de transfert contient des resultats deja generes; le depot Git conserve
les sources et permet de les regenerer. Les deux PDF de cours de l enseignant
ne sont pas inclus dans ce dossier; les partager separement entre collegues.

## Travail a deux

Conserver `main` pour les changements testes. Chaque amelioration se fait sur
une branche, puis une pull request permet a l autre de relire le changement.

```powershell
git pull
git switch -c amelioration-beveridge
# Modifier, puis executer les tests et les experiences utiles.
git add .
git commit -m "Implementer et tester le critere de Beveridge"
git push -u origin amelioration-beveridge
```

Le message de commit doit decrire le travail effectivement realise. Pour les
experiences, employer un nouveau `--output-root` et documenter les parametres.
Ne pas versionner `.venv`, les fichiers temporaires, les archives ou les gros
resultats generes. Echanger ces resultats a part si necessaire.

## Prochaines contributions

1. Documenter et implementer Beveridge, avec tests et condition d arret explicite.
2. Comparer plusieurs images sans surimpressions et plusieurs valeurs de K.
3. Consolider une version finale verifiee.
4. Rediger le rapport et preparer la presentation a partir des resultats mesures.
