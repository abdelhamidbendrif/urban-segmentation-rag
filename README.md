# Urban Image Segmentation and Region Merging

Student project: **Urban Video Surveillance — Region Merging with a Region Adjacency Graph (RAG)**.

This application simplifies urban images by grouping similar pixel colours, extracting connected regions, building a graph of neighbouring regions, and merging them using a Beveridge criterion.

The current application processes **still images**. It does not detect or name buildings, roads, pedestrians, or vehicles, and it does not process video streams.

## Workflow

```text
Input image
    -> Resize and convert to RGB
    -> K-means colour clustering
    -> 4-connected regions
    -> Region Adjacency Graph (NetworkX)
    -> Beveridge region merging
    -> Display and save results
```

1. **K-means:** assigns pixels to K colour clusters. Each displayed pixel receives its cluster's average colour.
2. **Connected regions:** splits each cluster into spatially connected areas. Pixels are connected through their top, bottom, left, and right neighbours.
3. **RAG:** represents each region as a node and each shared boundary as an edge.
4. **Beveridge merging:** combines adjacent regions using similarity, size, and shared-boundary factors. The current workflow evaluates the criterion in grayscale and displays results using mean RGB colours.

No supervised model is trained. K-means is fitted separately to each input image.

## Requirements

Use **Python 3.12 with Tkinter**, preferably on Windows. Install the pinned dependencies from `requirements.txt`: OpenCV, NumPy, Matplotlib, NetworkX, and Pillow. Tkinter comes with the standard Windows Python installation when Tcl/Tk support is enabled.

## Installation

Extract the project ZIP, open PowerShell inside the extracted project folder, and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

A virtual environment must be created on each computer; copying someone else's `.venv` is unnecessary.

## Start the application

Double-click **LANCER.bat**, or run:

```powershell
.\.venv\Scripts\python.exe interface.py
```

The interface is currently in French:

1. **Choisir une image:** select an input photograph.
2. **1. K-means:** calculate colour clusters.
3. **2. Regions + RAG:** extract connected regions and build their graph.
4. **3. Fusion Beveridge:** merge eligible neighbouring regions.
5. **Sauvegarder le resultat:** export the completed experiment.

The tabs show images, a local before/after RAG zoom, and full-image region/edge counts. Yellow points represent regions; blue lines represent adjacency. The zoom is only a small central window, not the complete graph.

Calculations run in a background thread to keep the interface responsive.

## Parameters

| Parameter | Meaning | Expected effect |
|---|---|---|
| K | Number of colour clusters | Lower values simplify colours more; higher values preserve more colour variations and often create more regions. |
| Taille max | Maximum image side in pixels | Smaller images are faster; larger images retain more spatial detail. Aspect ratio is preserved and small images are not enlarged. |
| Seuil T | Threshold for accepting a Beveridge merge | Higher values allow more merging, potentially losing useful boundaries. |

Start with **K=8, maximum side=480, T=0.2** as a trial setting, not a universally optimal configuration.

When changing only T, rerun step 3: each fusion starts from a copy of the same initial RAG. When changing K or image size, rerun steps 1, 2, and 3.

## Understanding the output

K colour clusters do **not** mean K connected regions or K object classes. A grey road and a grey roof may share a cluster, while one building can be divided by shadows and windows.

The connected-region step may look similar to K-means because assigning region identifiers does not require visibly different colours. Merging can substantially reduce region counts while making only subtle visual changes.

A smaller number of regions indicates stronger simplification, **not higher accuracy**. Check whether important boundaries remain visible and whether small details are lost.

Dataset reference masks are semantic annotations. Our region labels are numerical region identifiers, so the result is not expected to reproduce the semantic mask or its colour palette. The application does not automatically load masks or compute semantic accuracy, IoU, or class predictions.

## Saved results

Each GUI export creates a timestamped folder under `resultats/` containing:

| File | Content |
|---|---|
| original.png | Input image after preprocessing/resizing |
| kmeans.png | Colour-clustering result |
| fusion.png | Image after region merging |
| regions_initiales.npy | Initial connected-region IDs |
| regions_fusionnees.npy | Final region IDs |
| mesures.json | Parameters and full-image region/edge counts |
| rag_zoom.png | Local before/after graph illustration |

NPY files store numerical arrays; PNG files are visual previews. `original.png` is not necessarily the original full-resolution source file.

## Automated parameter experiments

Compare thresholds on a supplied Mapillary photograph:

```powershell
.\.venv\Scripts\python.exe experiments\sweep.py images\mapillary_asia_crossing.jpg --k 8 --t 0.1 0.2 0.3 --sizes 480
```

Compare K values while keeping T fixed:

```powershell
.\.venv\Scripts\python.exe experiments\sweep.py images\mapillary_asia_crossing.jpg --k 4 8 12 --t 0.2 --sizes 480
```

Outputs go to `experiments/output/`: one merged PNG per configuration and a CSV containing region counts, edge counts, and elapsed fusion-method time. This time includes the method's verification work, but excludes the preceding K-means and RAG stages.

Repeated runs with the same image stem overwrite the corresponding CSV and matching PNG filenames. Copy previous outputs elsewhere if you want to retain separate runs.

These experiments compare **parameters of the current workflow**, not different merging criteria.

## Project structure

```text
interface.py                 Tkinter interface, background worker, plots and controls
LANCER.bat                   Windows launcher using the local virtual environment
requirements.txt             Pinned Python dependencies
README.md                    Setup, workflow and limitations
moteur/
    experience.py            Coordinates processing, measurements and saving
    pipeline_io.py           Image loading and file-provenance utilities
    connected_regions.py     Extracts 4-connected regions from cluster labels
    build_rag.py             Builds region adjacency and region/boundary properties
    beveridge.py             Merge scoring, graph contraction and verification
    test_moteur.py           Automated workflow and calculation tests
experiments/
    sweep.py                 Runs combinations of K, T and image sizes
    compare_criteria.py      Compares Beveridge and mean difference
images/
    dronedeploy/              Supplied aerial crops and source notes
    mapillary/photos/         Supplied street-level photographs
    mapillary/labels/         Reference annotation masks
    mapillary_*.jpg           Convenient top-level example photographs
resultats/                   GUI exports, created when saving
```

`Experience` exposes `kmeans()`, `rag()`, `fusion()`, `summary()`, and `save()`. The separate `window_graph()` function prepares the local graph illustration.

The GUI and sweep script are the recommended entry points. Some standalone helper scripts retain older command-line defaults that may not match this folder layout.

## Input data and provenance

- **DroneDeploy:** four JPEG crops supplied by the project author, declared to originate from the Kaggle Drone Deploy Medium dataset. Their correspondence to the original full TIFF scenes has not been independently verified. No reference masks accompany these crops.
- **Mapillary:** supplied sample photographs and indexed PNG masks. The sample archive did not include its exact dataset version, class configuration, or licence documentation. Obtain the matching configuration before attempting class-based evaluation, and check the source terms before further redistribution.
- See the `SOURCE.txt` files inside the dataset folders for the available provenance notes. Open photographs, not annotation masks, as program inputs.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s moteur -p "test_*.py" -v
```

The reviewed version passes seven tests covering stage order, clustering/regions, repeated fusion from the initial partition, resetting results after K changes, export consistency, graph-window positions, and numerical merge calculations.

Passing these tests checks specific implementation behaviours; it does not establish segmentation quality on every image.

## Compare merging criteria

Run this command from the project folder (replace the photograph path as needed):

```powershell
.\.venv\Scripts\python.exe experiments\compare_criteria.py images\mapillary_asia_crossing.jpg
```

For the aerial intersection used in the latest experiment:

```powershell
.\.venv\Scripts\python.exe experiments\compare_criteria.py images\dronedeploy\12fa5e614f_53197F206FOPENPIPELINE-ortho.jpg
```

The script computes K-means and the initial RAG once, then applies Beveridge and mean difference to independent copies. Defaults: K=8, maximum side=480; Beveridge thresholds 0.1, 0.2, 0.3; mean-difference thresholds 2, 5, 10, 20 in grayscale intensity units.

Results go to `experiments/output/criteria/<image>_s480_k8/`: `comparison.csv`, merged images, three-panel comparisons, and interpretation notes. Thresholds can be adjusted with `--beveridge 0.1 0.2` and `--mean-diff 5 10`. They have different scales: equal threshold numbers are not equivalent. Compare approximately similar final region counts, then inspect preserved boundaries and lost details. No criterion is declared best automatically.

`merge_seconds` measures the merge algorithm only, excluding graph copying, preprocessing, relabeling and verification. Results are independently checked by rebuilding the graph and checking the stopping condition. Repeated configurations overwrite matching files.

## Current status and limitations

The GUI remains a simple Beveridge teaching workflow. The separate comparison script adds the requested two-criterion experiments without complicating the interface. Parameter sweeps and criterion comparisons answer different questions. The criteria comparison has been run on the Mapillary street scene and DroneDeploy aerial intersection. The updated four-page French report includes the measured results and visual comparisons: [Report PDF](documents/Rapport_Projet5_4pages.pdf). Timings are single-run observations, not general performance rankings.

The assignment also requires a presentation and notebook; these are not supplied by the comparison script. Document the exact Beveridge formula and its reference in the final report. Segmentation here is unsupervised region merging, not semantic classification.

## Troubleshooting

- Missing environment: follow Installation above.
- Slow processing: use maximum side 480 first.
- Cannot save: finish all three GUI stages.
- Outputs differ from dataset masks: region IDs do not represent semantic classes.

Git ignores `.venv/`, Python caches, GUI exports, experiment outputs, and ZIP archives. Share exports separately when needed.
