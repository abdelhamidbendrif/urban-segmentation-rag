# Audit du prototype de segmentation urbaine

Cet audit distingue la correction numerique du prototype, la qualite de la
segmentation et la conformite au sujet. Il ne constitue pas le rapport academique final.

## Conclusion

Les artefacts actuels de `city.png` sont coherents avec les algorithmes annonces.
Aucune erreur centrale d etiquetage, d adjacence, de moyenne ponderee ou de
contraction n a ete detectee dans les cas testes. Cela ne garantit pas l absence
de toute erreur sur toute entree possible.

Le critere de Beveridge et le rapport final restent a realiser. Les seuils RGB
et la priorite de taille sont une methode experimentale distincte de Beveridge.

## Verifications realisees

- Dependances : `pip check` sans conflit detecte.
- Nouvelle execution de K-means dans `tmp/audit_reproduction` : etiquettes identiques
  a celles conservees dans `outputs/city_k8`.
- Composantes connexes recalculees : identiques aux etiquettes sauvegardees.
- Image originale : 1447 x 812; grille de calcul : 960 x 539, soit 517440 pixels.
- Graphe initial : 91822 noeuds et 206684 aretes, reconstruits depuis les pixels.
- Graphe initial et graphe apres fusion relus depuis les fichiers GraphML et compares
  aux reconstructions : aires, centroides, sommes/moyennes RGB, adjacences,
  longueurs de frontiere et sommes/moyennes des contrastes.
- Les cinq variantes et la premiere fusion de reference ont ete verifiees :
  les correspondances d etiquettes sont correctes, les regions sont connexes,
  la surface est conservee, les rendus et mesures sont coherents, et aucune
  paire admissible ne reste apres arret selon le critere de la variante.
- 15 tests automatises passes : erreurs de voisinage diagonal, longueur de frontiere,
  moyenne ponderee, mise a jour du score, contraste fort, cas sans frontiere forte,
  garde des fichiers d entree, seuils non finis et contractions aleatoires.
- Revue independante supplementaire : 150 petits cas aleatoires compares a une
  reconstruction independante. La suite enregistree comprend aussi 12 essais
  aleatoires reproductibles dans un test de regression.
- Les cinq commandes de traitement ont ete executees sur une petite image
  synthetique avec K=3 dans un dossier separe, puis sur une image a faible
  contraste avec K=2 pour verifier le rapport N/A.

Preuve numerique conservee : `outputs/city_k8/audit/verification.json`.

## Problemes detectes et corriges

| Probleme avant audit | Correction | Effet sur city.png existant |
|---|---|---|
| Comparaison fixee a city_k8 | Option `compare_fusion.py --graph` | Aucun changement numerique |
| Confusion possible entre anciennes et nouvelles entrees | Empreintes SHA-256 et metadonnees entre etapes | Entrees actuelles auditees et empreintes ajoutees |
| Chemins absolus peu portables | Chemins relatifs en plus des chemins absolus | Graphe transportable avec ses sources |
| Comparaison obligatoire a une ancienne reference potentiellement perimee | Controle seulement si les empreintes des entrees correspondent | La reference actuelle a ete verifiee separement |
| NaN/inf acceptes comme seuils | Seuils obligatoirement finis et positifs | Les seuils ordinaires 20/40 restent identiques |
| Rapport plante si aucune frontiere forte n existe | Valeur indefinie affichee N/A | Aucun changement, car city.png a des frontieres fortes |

Les nouveaux resultats issus des scripts possedent des metadonnees. Les anciens
fichiers sans metadonnees restent lisibles pour compatibilite; leur provenance
ne peut pas etre garantie automatiquement. Les artefacts actuels ont ete audites
avant ajout de leurs empreintes. Une empreinte detecte un fichier modifie; elle
ne verifie pas la pertinence scientifique de la photographie.

## Limites scientifiques importantes

- Une image Google Maps avec noms et icones est un prototype d entree urbaine,
  et ne constitue pas un jeu de donnees annote DroneDeploy/Mapillary.
- La distance RGB et les seuils 20/40 sont nos choix experimentaux. La taille
  <10 pixels controle la priorite, pas une suppression obligatoire des petits objets.
- La fusion de moyennes proches n impose aucune borne explicite a la variance
  interne d une region fusionnee.
- Les images K-means montrent les centres globaux des classes. Les rendus avant
  et apres fusion montrent les moyennes originales des regions : plus de huit
  couleurs dans ces rendus est donc normal.
- Les lignes du dessin RAG sont des liens de voisinage entre centroides, pas
  des contours et pas des routes. Le zoom du RAG masque les noeuds dont le
  centroide est hors cadre : c est une vue partielle du graphe complet.
- Un RMSE faible indique une fidelite des couleurs. Un pourcentage de frontieres
  conservees indique la conservation de contrasts definis experimentalement.
  Aucune de ces mesures n est une precision de reconnaissance des objets.
- La methode est gloutonne; l ordre influence les resultats. Les temps mesures
  une fois ne sont pas une comparaison statistique de performance.
- Pas de detection semantique de routes, batiments ou pietons; pas de suivi video.
  Le sujet donne des images comme donnees, donc une video n est pas un livrable
  explicitement requis par le seul titre.

## Refaire les verifications

Dans le dossier du projet :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe verify_project.py --root outputs/city_k8 --image city.png
```

La derniere commande verifie les artefacts actuels sans refaire les fusions;
elle ecrit uniquement le fichier de synthese d audit. L execution peut prendre
quelques minutes car elle relit et reconstruit plusieurs graphes.

Pour verifier aussi la reproductibilite de K-means sans remplacer les resultats :

```powershell
.\.venv\Scripts\python.exe kmeans_segmentation.py city.png --k 8 --output-root tmp/audit_reproduction
.\.venv\Scripts\python.exe verify_project.py --fresh-labels tmp/audit_reproduction/city_k8/cluster_labels.npy
```

Le controle exact des etiquettes teste la reproductibilite dans l environnement
actuel. Des versions ou plateformes differentes peuvent produire des partitions
ou des numeros differents sans que cela signifie automatiquement une erreur.

`requirements.txt` enregistre les versions directes utilisees pour cet audit;
Python utilise : 3.12.3. Le fichier ne pretend pas figer tous les details de la
plateforme ou tous les paquets transitifs.

## Beveridge : reference trouvee, implementation restante

Deux cours universitaires donnent une formulation combinant similarite normalisee
par les ecarts-types, surface et rapport entre perimetres et frontiere commune :

- Anne Vialard, LaBRI, page 48 : https://dept-info.labri.fr/~vialard/Image3D/cours/cours-segmentation.pdf
- Luc Brun, ENSICAEN, pages 72-74 : https://lucbrun.ensicaen.fr/ENSEIGNEMENT/COURS/segmentation.pdf

Il faut encore definir explicitement le traitement scalaire ou couleur, les
parametres et la condition d arret, ajouter variance et perimetre, puis tester
l implementation de cette formulation. La variante attendue par l enseignant
reste a confirmer si elle n est pas precisee dans son support.
