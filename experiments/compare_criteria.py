"""Compare two merge criteria from exactly the same initial regions."""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'moteur'))
from experience import Experience
import beveridge as bev


def merge(experiment, threshold, combine):
    """Copy the initial graph, merge, then independently verify the result."""
    graph = experiment.base.copy()
    topt = bev.default_topt(experiment.regions.shape)
    start = time.perf_counter()
    parent, _ = bev.fuse_beveridge(graph, threshold, topt,
                                  space='gray', combine=combine)
    seconds = time.perf_counter() - start
    labels, _, compact = bev.relabel(parent, experiment.regions, len(experiment.base))
    final, means = bev.verify_contraction(graph, labels, experiment.rgb, compact)
    if bev.pairs_below_threshold(final, threshold, topt, space='gray', combine=combine):
        raise AssertionError('An eligible pair remains after merging.')
    image = means[labels].clip(0, 255).astype(np.uint8)
    return image, len(final), final.number_of_edges(), seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--size', type=int, default=480)
    parser.add_argument('--beveridge', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    parser.add_argument('--mean-diff', type=float, nargs='+', default=[2, 5, 10, 20])
    parser.add_argument('--out', type=Path, default=ROOT / 'experiments' / 'output' / 'criteria')
    args = parser.parse_args()
    if args.size < 1:
        parser.error('--size must be positive')
    folder = args.out / f'{args.image.stem}_s{args.size}_k{args.k}'
    folder.mkdir(parents=True, exist_ok=True)
    experiment = Experience(args.image, args.size)
    experiment.kmeans(args.k)
    experiment.rag()
    rows = []
    images = {}
    for criterion, combine, thresholds in [('beveridge', 'sqrt', args.beveridge),
                                          ('mean_difference', 'mean_diff', args.mean_diff)]:
        for threshold in thresholds:
            image, regions, edges, seconds = merge(experiment, threshold, combine)
            row = dict(image=args.image.name, width=experiment.rgb.shape[1],
                       height=experiment.rgb.shape[0], k=args.k, criterion=criterion,
                       threshold=threshold, regions_before=len(experiment.base),
                       regions_after=regions, edges_before=experiment.base.number_of_edges(),
                       edges_after=edges, merge_seconds=round(seconds, 4))
            rows.append(row)
            images[criterion, threshold] = image
            stem = f'{criterion}_T{threshold:g}'
            plt.imsave(folder / f'{stem}.png', image)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
            for ax, img, title in zip(axes, [experiment.rgb, experiment.kmeans_image, image],
                                     ['Input', f'K-means K={args.k}', f'{criterion}, T={threshold:g}\n{regions} regions']):
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(title)
            fig.savefig(folder / f'{stem}_comparison.png', dpi=140)
            plt.close(fig)
            print(row, flush=True)
    # Match by region count only; this does not identify the best-quality output.
    pairs = [(a, b) for a in rows for b in rows
             if a['criterion'] == 'beveridge' and b['criterion'] == 'mean_difference']
    a, b = min(pairs, key=lambda pair: abs(math.log(
        pair[0]['regions_after'] / pair[1]['regions_after'])))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    axes[0].imshow(experiment.rgb)
    axes[0].set_title('Input')
    for ax, row in zip(axes[1:], [a, b]):
        ax.imshow(images[row['criterion'], row['threshold']])
        ax.set_title(f"{row['criterion']}, T={row['threshold']:g}\n{row['regions_after']} regions")
    for ax in axes:
        ax.axis('off')
    fig.suptitle('Closest region counts among tested thresholds (not a quality ranking)')
    fig.savefig(folder / 'criteria_side_by_side.png', dpi=140)
    plt.close(fig)
    with (folder / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / 'READ_ME.txt').write_text(
        'Both criteria start from the same K-means and 4-connected partition.\n'
        'Mean difference thresholds are grayscale intensity differences (0-255 scale); '
        'Beveridge uses a different score. Equal thresholds are not equivalent.\n'
        'Compare similar final region counts and inspect the same boundaries. '
        'This is not semantic accuracy. No best criterion is selected automatically.\n'
        'merge_seconds excludes preprocessing, copying, relabeling and verification.\n'
        'Repeating the same configuration overwrites matching output files.\n', encoding='utf-8')
    print('Saved:', folder)


if __name__ == '__main__':
    main()
