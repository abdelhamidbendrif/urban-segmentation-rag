"""Controlled experiments on the same initial partition; no ground-truth claims."""
import json
import argparse
import time
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_rag import build_rag
from merge_regions import fuse
from pipeline_io import graph_sources, read_rgb, sha256


EXPERIMENTS = [
    ('color_only', 'Color only', 'color', 20, 40, 10),
    ('combined_no_priority', 'Color + boundary; no size priority', 'combined', 20, 40, 0),
    ('combined_reference', 'Color + boundary; small first', 'combined', 20, 40, 10),
    ('combined_strict', 'Strict thresholds', 'combined', 12, 25, 10),
    ('combined_relaxed', 'Relaxed thresholds', 'combined', 30, 60, 10),
]


def discussion(records):
    r = {item['experiment']: item for item in records}
    ref, no_size, color, relaxed = (r[k] for k in
        ['combined_reference', 'combined_no_priority', 'color_only', 'combined_relaxed'])
    return [
        '', '## Lecture des resultats observes', '',
        f"- L ajout des frontieres change peu le nombre final de regions ici : {color['regions']} avec la couleur seule contre {ref['regions']} pour la reference. Cette comparaison change aussi le score de classement des paires; elle n isole donc pas uniquement le filtre d admissibilite.",
        f"- A seuils identiques, la priorite aux petites regions donne {no_size['regions']-ref['regions']} regions de moins que la variante sans priorite. L effet de l ordre des fusions est mesurable.",
        '- La variante stricte conserve presque la partition de depart : elle simplifie peu, tout en gardant une meilleure fidelite des couleurs.',
        f"- La variante permissive reduit le nombre de regions de {relaxed['reduction_percent']:.2f} %, mais son RMSE RGB atteint {relaxed['rgb_rmse']:.2f}, contre {ref['rgb_rmse']:.2f} pour la reference. Le zoom montre des aplats plus grands et une perte de details.",
        '- La reference constitue un point de depart intermediaire pour cette image, sans preuve qu elle soit optimale. Beaucoup de minuscules regions subsistent. Il faut une image sans surimpressions et des annotations pour evaluer la segmentation des objets.',
        '- Le critere de Beveridge reste a documenter et implementer avant de declarer satisfait cet objectif du sujet.',
    ]


def measure(labels, rgb, initial):
    areas = np.bincount(labels.ravel())
    n = len(areas)
    sums = np.column_stack([np.bincount(labels.ravel(), weights=rgb[:,:,c].ravel()) for c in range(3)])
    means = sums / areas[:, None]
    reconstruction = means[labels]
    rmse = float(np.sqrt(np.mean((rgb.astype(float) - reconstruction)**2)))
    retained = strong_total = strong_retained = 0
    image = rgb.astype(float)
    for a, b, ia, ib, ca, cb in [
        (labels[:,:-1], labels[:,1:], initial[:,:-1], initial[:,1:], image[:,:-1], image[:,1:]),
        (labels[:-1,:], labels[1:,:], initial[:-1,:], initial[1:,:], image[:-1,:], image[1:,:]),
    ]:
        boundary = a != b
        strong = (ia != ib) & (np.linalg.norm(ca-cb, axis=2) >= 60)
        retained += int(boundary.sum())
        strong_total += int(strong.sum())
        strong_retained += int((strong & boundary).sum())
    return dict(regions=n, tiny_regions_under_10=int((areas < 10).sum()),
                mean_area_pixels=float(areas.mean()), rgb_rmse=rmse,
                retained_pixel_sides=retained, strong_sides_total=strong_total,
                strong_sides_retained=strong_retained,
                strong_boundary_retention_percent=100*strong_retained/strong_total if strong_total else None), means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', default='outputs/city_k8/connected_regions/rag/rag_before.graphml.gz')
    args = parser.parse_args()
    graph_path = Path(args.graph)
    root = graph_path.parent
    out = root / 'comparison_experiments'
    out.mkdir(exist_ok=True)
    print('Loading identical initial RAG for all experiments...', flush=True)
    base = nx.read_graphml(graph_path, node_type=int)
    image_path, labels_path = graph_sources(base, graph_path)
    initial = np.load(labels_path, allow_pickle=False)
    rgb = read_rgb(image_path)
    input_labels_hash, input_image_hash = sha256(labels_path), sha256(image_path)
    if rgb.shape[:2] != initial.shape:
        raise ValueError('Image and label dimensions differ.')
    original_metrics, original_means = measure(initial, rgb, initial)
    records = []
    displays = [('Initial regions', original_means[initial], len(base))]
    for key, title, criterion, color, boundary, small in EXPERIMENTS:
        print(f'Running {key}...', flush=True)
        graph = base.copy()
        start = time.perf_counter()
        parent = fuse(graph, color, boundary, small, criterion)
        elapsed = time.perf_counter() - start
        compact = {n: i for i,n in enumerate(sorted(graph))}
        mapping = np.array([compact[parent[n]] for n in range(len(base))], dtype=np.int32)
        labels = mapping[initial]
        # Independently rebuild the result to check graph contractions.
        rebuilt, _ = build_rag(labels, rgb)
        contracted = nx.relabel_nodes(graph, compact, copy=True)
        assert {tuple(sorted(e)) for e in rebuilt.edges} == {tuple(sorted(e)) for e in contracted.edges}
        for n,d in rebuilt.nodes(data=True):
            assert d['area'] == contracted.nodes[n]['area']
            assert all(np.isclose(d[f'mean_{c}'], contracted.nodes[n][f'mean_{c}']) for c in 'rgb')
        for a,b,d in rebuilt.edges(data=True):
            assert d['boundary_length'] == contracted[a][b]['boundary_length']
            assert np.isclose(d['boundary_contrast_sum'], contracted[a][b]['boundary_contrast_sum'])
            assert d['color_distance'] > color or (criterion == 'combined' and d['boundary_contrast'] > boundary)
        assert sum(d['area'] for _,d in rebuilt.nodes(data=True)) == initial.size
        assert nx.is_connected(rebuilt)
        metrics, means = measure(labels, rgb, initial)
        assert metrics['retained_pixel_sides'] == sum(d['boundary_length'] for _,_,d in rebuilt.edges(data=True))
        if key == 'combined_reference':
            previous = root / 'fusion_c20_b40_s10' / 'region_labels.npy'
            summary_path = previous.parent / 'summary.json'
            summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
            same_inputs = (summary.get('initial_labels_sha256') == input_labels_hash and
                           summary.get('source_image_sha256') == input_image_hash)
            if previous.exists() and same_inputs:
                assert np.array_equal(labels, np.load(previous, allow_pickle=False))
            elif previous.exists():
                print('Skipping comparison to earlier baseline: provenance missing or inputs differ.', flush=True)
        folder = out / key
        folder.mkdir(exist_ok=True)
        np.save(folder / 'region_labels.npy', labels)
        np.save(folder / 'initial_to_merged.npy', mapping)
        image = np.clip(means[labels], 0, 255).astype(np.uint8)
        cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tofile(folder / 'segmentation.png')
        record = dict(experiment=key, criterion=criterion, color_threshold=color,
                      initial_labels_sha256=input_labels_hash, source_image_sha256=input_image_hash,
                      boundary_threshold=boundary if criterion=='combined' else None,
                      small_area_priority=small, edges=rebuilt.number_of_edges(),
                      fusion_seconds=round(elapsed,3), reduction_percent=100*(1-len(rebuilt)/len(base)),
                      verified=True, **metrics)
        records.append(record)
        (folder/'metrics.json').write_text(json.dumps(record,indent=2), encoding='utf-8')
        displays.append((title, image, len(rebuilt)))
        print(f"{key}: {len(rebuilt):,} regions; {elapsed:.1f}s", flush=True)
        del graph, rebuilt, contracted
    (out/'results.json').write_text(json.dumps(dict(initial=original_metrics, experiments=records),indent=2),encoding='utf-8')
    for name, crop in [('overview',False),('detail',True)]:
        fig, axes = plt.subplots(2,3,figsize=(15,9), constrained_layout=True)
        h,w = initial.shape
        for ax,(title,image,n) in zip(axes.flat,displays):
            ax.imshow(np.clip(image,0,255).astype(np.uint8))
            if crop:
                ax.set_xlim(max(0,w//2-70),min(w,w//2+70))
                ax.set_ylim(min(h,h//2+70),max(0,h//2-70))
            ax.set_title(f'{title}\n{n:,} regions', fontsize=10)
            ax.axis('off')
        fig.savefig(out/f'{name}.png',dpi=170)
        plt.close(fig)
    lines = [
        '# Comparaison experimentale des criteres de fusion', '',
        'Une seule image, une seule partition initiale et le meme RAG sont utilises pour tous les essais.',
        'Ces variantes sont experimentales et ne constituent pas le critere de Beveridge.', '',
        '| Essai | Seuil couleur | Seuil frontiere | Priorite taille | Regions | Reduction % | Petites <10 px | RMSE RGB | Frontieres fortes conservees % | Temps fusion s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in records:
        retention = r['strong_boundary_retention_percent']
        retention_text = f'{retention:.2f}' if retention is not None else 'N/A'
        lines.append(f"| {r['experiment']} | {r['color_threshold']} | {r['boundary_threshold'] if r['boundary_threshold'] is not None else '-'} | {r['small_area_priority']} | {r['regions']} | {r['reduction_percent']:.2f} | {r['tiny_regions_under_10']} | {r['rgb_rmse']:.2f} | {retention_text} | {r['fusion_seconds']:.2f} |")
    lines += ['', '## Interpretation des mesures', '',
        '- RMSE RGB : ecart entre les pixels originaux et la reconstruction par moyenne de chaque region; plus faible indique une meilleure fidelite des couleurs, pas une meilleure reconnaissance des objets.',
        '- Frontieres fortes : cotes de pixels separes dans la partition initiale avec une distance RGB originale >= 60. Le pourcentage est la fraction encore separee apres fusion. Les textes et icones peuvent aussi contribuer a cette mesure.',
        '- Le seuil fort 60 est fixe pour toutes les experiences et distinct du seuil moyen utilise pour accepter une fusion.',
        '- N/A signifie qu aucune frontiere forte initiale n existe; le pourcentage est alors indefini.',
        '- Taille 0 : aucune priorite de taille. Taille 10 : les paires contenant une region de moins de 10 pixels passent en priorite, sans obligation de fusion.',
        '- Les temps mesurent une execution de la fusion, hors chargement, copie du graphe, verification et figures. Ils ne constituent pas un benchmark statistique.',
        '- La fusion est gloutonne : changer le score, les seuils ou la priorite peut changer l ordre des fusions et la partition finale.',
        '- Les graphes ont ete reconstruits independamment pour verifier surfaces, couleurs, adjacences, longueurs et contrastes. La comparaison avec une ancienne reference est executee seulement si les empreintes des entrees correspondent.',
        '', '## Limites', '',
        'Aucune annotation de reference : pas de mesure IoU, precision ou rappel. Une reduction plus forte du nombre de regions ne suffit pas a designer un meilleur resultat. Les captures Google Maps contiennent des textes et symboles qui fragmentent les regions. Ces essais portent uniquement sur cette image; les resultats ne se generalisent pas encore a d autres villes.',
        '', '## Reproduction', '', '```powershell', f'.\\.venv\\Scripts\\python.exe compare_fusion.py --graph "{graph_path.as_posix()}"', '```', '',
        'Voir overview.png, detail.png et results.json dans ce dossier. Chaque essai conserve ses etiquettes et sa reconstruction.',
    ]
    lines += discussion(records)
    (out/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Comparison saved to: {out.resolve()}',flush=True)


if __name__ == '__main__':
    main()
