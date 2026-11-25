"""Build and export a 4-neighbor Region Adjacency Graph using NetworkX.

Areas and boundary lengths are measured on the resized image grid.
RGB distances describe color similarity, not semantic object identity.
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import networkx as nx
import numpy as np
from pipeline_io import read_rgb, verify_stage_metadata, set_graph_sources

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def build_rag(regions, rgb):
    if regions.ndim != 2 or rgb.shape != (*regions.shape, 3):
        raise ValueError('Labels and RGB image must have matching dimensions.')
    ids = np.unique(regions)
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError('Expected contiguous region IDs starting at zero.')
    flat = regions.ravel()
    count = len(ids)
    areas = np.bincount(flat, minlength=count)
    yy, xx = np.indices(regions.shape)
    cx = np.bincount(flat, weights=xx.ravel()) / areas
    cy = np.bincount(flat, weights=yy.ravel()) / areas
    sums = np.column_stack([
        np.bincount(flat, weights=rgb[:, :, c].ravel()) for c in range(3)
    ])
    means = sums / areas[:, None]
    graph = nx.Graph(connectivity=4, color_space='RGB',
                     boundary_length_unit='pixel-side',
                     boundary_contrast_definition='mean RGB distance across shared pixel sides')
    for i in range(count):
        graph.add_node(i, area=int(areas[i]), x=float(cx[i]), y=float(cy[i]),
                       **{f'mean_{c}': float(means[i, j]) for j, c in enumerate('rgb')},
                       **{f'sum_{c}': float(sums[i, j]) for j, c in enumerate('rgb')})
    # Only horizontal and vertical contacts: diagonal touches are not edges.
    contacts, contrasts = [], []
    image = rgb.astype(np.float64)
    for a, b, ca, cb in [
        (regions[:, :-1], regions[:, 1:], image[:, :-1], image[:, 1:]),
        (regions[:-1, :], regions[1:, :], image[:-1, :], image[1:, :]),
    ]:
        mask = a != b
        contacts.append(np.sort(np.column_stack([a[mask], b[mask]]), axis=1))
        contrasts.append(np.linalg.norm(ca[mask] - cb[mask], axis=1))
    pairs = np.concatenate(contacts)
    unique, inverse, lengths = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    contrast_sums = np.bincount(inverse, weights=np.concatenate(contrasts), minlength=len(unique))
    for (a, b), length, contrast_sum in zip(unique, lengths, contrast_sums):
        graph.add_edge(int(a), int(b), boundary_length=int(length),
                       color_distance=float(np.linalg.norm(means[a] - means[b])),
                       boundary_contrast_sum=float(contrast_sum),
                       boundary_contrast=float(contrast_sum / length))
    return graph, means


def plot_graph(graph, rgb, means, destination, crop=None, stage='before'):
    height, width = rgb.shape[:2]
    if crop is None:
        nodes = list(graph)
        title = f'RAG {stage} merging: {len(graph):,} regions'
    else:
        x0, y0, x1, y1 = crop
        nodes = [n for n, d in graph.nodes(data=True)
                 if x0 <= d['x'] < x1 and y0 <= d['y'] < y1]
        title = f'Central detail: {len(nodes)} nodes (centroids inside crop)'
    subgraph = graph.subgraph(nodes)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(rgb, alpha=0.65)
    positions = {n: (graph.nodes[n]['x'], graph.nodes[n]['y']) for n in nodes}
    lines = [(positions[a], positions[b]) for a, b in subgraph.edges]
    ax.add_collection(LineCollection(lines, colors='#126fbd', linewidths=0.35, alpha=0.65))
    points = np.array([positions[n] for n in nodes])
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], c=means[nodes] / 255,
                   s=10 if crop else 0.8, edgecolors='black' if crop else 'none', linewidths=0.25)
    if crop:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    else:
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='city.png')
    parser.add_argument('--labels', default='outputs/city_k8/connected_regions/region_labels.npy')
    args = parser.parse_args()
    image_path, labels_path = Path(args.image), Path(args.labels)
    if not image_path.is_file() or not labels_path.is_file():
        parser.error('Image or region labels file not found.')
    verify_stage_metadata(labels_path, image_path)
    regions = np.load(labels_path, allow_pickle=False)
    if regions.ndim != 2 or not np.issubdtype(regions.dtype, np.integer):
        parser.error('Expected a 2D integer label array.')
    rgb = read_rgb(image_path)
    if rgb.shape[:2] != regions.shape:
        parser.error('Image and labels dimensions differ. Use the original image from K-means.')
    print('Building graph...', flush=True)
    graph, means = build_rag(regions, rgb)
    out = labels_path.parent / 'rag'
    out.mkdir(exist_ok=True)
    set_graph_sources(graph, out / 'rag_before.graphml.gz', image_path, labels_path)
    nx.write_graphml(graph, out / 'rag_before.graphml.gz')
    summary = {
        'nodes': graph.number_of_nodes(), 'edges': graph.number_of_edges(),
        'connectivity': 4, 'connected_components': nx.number_connected_components(graph),
        'total_region_area': sum(d['area'] for _, d in graph.nodes(data=True)),
        'shared_pixel_sides': sum(d['boundary_length'] for _, _, d in graph.edges(data=True)),
        'image_width': rgb.shape[1], 'image_height': rgb.shape[0],
    }
    assert summary['total_region_area'] == regions.size
    assert summary['connected_components'] == 1
    assert nx.number_of_selfloops(graph) == 0
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plot_graph(graph, rgb, means, out / 'rag_before.png')
    h, w = regions.shape
    plot_graph(graph, rgb, means, out / 'rag_detail.png',
               (max(0, w // 2 - 50), max(0, h // 2 - 50), min(w, w // 2 + 50), min(h, h // 2 + 50)))
    print(json.dumps(summary, indent=2))
    print(f'Saved to: {out.resolve()}')


if __name__ == '__main__':
    main()
