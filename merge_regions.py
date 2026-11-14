"""Experimental RAG fusion: RGB similarity AND weak shared boundaries.

This baseline is not the Beveridge criterion. Small regions get priority;
they still must pass both thresholds. All statistics update after each merge.
"""
import argparse
import heapq
import math
import json
import time
from pathlib import Path

import cv2
import matplotlib
import networkx as nx
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_rag import build_rag, plot_graph
from pipeline_io import graph_sources, read_rgb, set_graph_sources, sha256


def fuse(graph, color_threshold=20.0, boundary_threshold=40.0, small_area=10, criterion='combined'):
    """Mutate graph; return original-node -> surviving-node mapping."""
    if not all(math.isfinite(t) and t > 0 for t in (color_threshold, boundary_threshold)) or small_area < 0:
        raise ValueError('Thresholds must be finite and positive and small_area >= 0.')
    if criterion not in ('combined', 'color'):
        raise ValueError('Criterion must be combined or color.')
    parent = {n: n for n in graph}
    version = {n: 0 for n in graph}
    heap = []

    def queue(a, b):
        a, b = sorted((a, b))
        edge = graph[a][b]
        da, db = graph.nodes[a], graph.nodes[b]
        distance = sum((da[f'mean_{c}'] - db[f'mean_{c}']) ** 2 for c in 'rgb') ** 0.5
        edge['color_distance'] = distance
        contrast = edge['boundary_contrast_sum'] / edge['boundary_length']
        edge['boundary_contrast'] = contrast
        if distance <= color_threshold and (criterion == 'color' or contrast <= boundary_threshold):
            large = int(min(da['area'], db['area']) >= small_area)
            score = distance / color_threshold
            if criterion == 'combined':
                score += contrast / boundary_threshold
            heapq.heappush(heap, (large, score, a, b, version[a], version[b]))

    for a, b in graph.edges:
        queue(a, b)
    merges = 0
    while heap:
        _, _, a, b, va, vb = heapq.heappop(heap)
        # Discard decisions calculated before either region changed.
        if a not in graph or b not in graph or version[a] != va or version[b] != vb:
            continue
        if graph.nodes[a]['area'] < graph.nodes[b]['area']:
            a, b = b, a
        da, db = graph.nodes[a], graph.nodes[b]
        area = da['area'] + db['area']
        for coord in ('x', 'y'):
            da[coord] = (da[coord] * da['area'] + db[coord] * db['area']) / area
        for c in 'rgb':
            da[f'sum_{c}'] += db[f'sum_{c}']
            da[f'mean_{c}'] = da[f'sum_{c}'] / area
        da['area'] = area
        for neighbor, edge in list(graph[b].items()):
            if neighbor == a:
                continue
            if graph.has_edge(a, neighbor):
                graph[a][neighbor]['boundary_length'] += edge['boundary_length']
                graph[a][neighbor]['boundary_contrast_sum'] += edge['boundary_contrast_sum']
            else:
                graph.add_edge(a, neighbor, **edge)
        graph.remove_node(b)
        parent[b] = a
        version[a] += 1
        for neighbor in graph[a]:
            queue(a, neighbor)
        merges += 1
        if merges % 20000 == 0:
            print(f'{merges:,} merges; {len(graph):,} regions remain', flush=True)
    # Path compression maps every initial region to its final survivor.
    for node in parent:
        root = node
        while parent[root] != root:
            root = parent[root]
        current = node
        while parent[current] != current:
            following = parent[current]
            parent[current] = root
            current = following
    return parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', default='outputs/city_k8/connected_regions/rag/rag_before.graphml.gz')
    parser.add_argument('--color-threshold', type=float, default=20)
    parser.add_argument('--boundary-threshold', type=float, default=40)
    parser.add_argument('--small-area', type=int, default=10)
    parser.add_argument('--criterion', choices=['combined', 'color'], default='combined')
    args = parser.parse_args()
    if not all(math.isfinite(t) and t > 0 for t in (args.color_threshold, args.boundary_threshold)) or args.small_area < 0:
        parser.error('Thresholds must be finite and positive; small-area must be >= 0 (disabled).')
    graph_path = Path(args.graph)
    print('Loading RAG...', flush=True)
    graph = nx.read_graphml(graph_path, node_type=int)
    image_path, labels_path = graph_sources(graph, graph_path)
    labels = np.load(labels_path, allow_pickle=False)
    rgb = read_rgb(image_path)
    if rgb.shape[:2] != labels.shape:
        parser.error('Image and labels do not match.')
    if set(graph) != set(np.unique(labels)):
        parser.error('Graph nodes and input labels do not match.')
    before_nodes, before_edges = len(graph), graph.number_of_edges()
    before_means = np.array([[graph.nodes[n][f'mean_{c}'] for c in 'rgb'] for n in range(before_nodes)])
    before_image = np.clip(before_means[labels], 0, 255).astype(np.uint8)
    start = time.perf_counter()
    parent = fuse(graph, args.color_threshold, args.boundary_threshold, args.small_area, args.criterion)
    seconds = time.perf_counter() - start
    survivors = sorted(graph)
    compact = {node: i for i, node in enumerate(survivors)}
    mapping = np.array([compact[parent[n]] for n in range(before_nodes)], dtype=np.int32)
    merged_labels = mapping[labels]
    # Independent reconstruction verifies adjacency and shared boundaries after contraction.
    rebuilt, means = build_rag(merged_labels, rgb)
    contracted = nx.relabel_nodes(graph, compact, copy=True)
    assert set(rebuilt.edges) == set(contracted.edges) or {
        tuple(sorted(e)) for e in rebuilt.edges
    } == {tuple(sorted(e)) for e in contracted.edges}
    for n, d in rebuilt.nodes(data=True):
        assert d['area'] == contracted.nodes[n]['area']
        assert all(np.isclose(d[f'mean_{c}'], contracted.nodes[n][f'mean_{c}']) for c in 'rgb')
    for a, b, d in rebuilt.edges(data=True):
        edge = contracted[a][b]
        assert d['boundary_length'] == edge['boundary_length']
        assert np.isclose(d['boundary_contrast_sum'], edge['boundary_contrast_sum'])
    assert sum(d['area'] for _, d in rebuilt.nodes(data=True)) == labels.size
    assert nx.is_connected(rebuilt)
    # Confirm no eligible pair remains at this stopping point.
    assert all(d['color_distance'] > args.color_threshold or
               (args.criterion == 'combined' and d['boundary_contrast'] > args.boundary_threshold)
               for _, _, d in rebuilt.edges(data=True))
    out = graph_path.parent / f'fusion_c{args.color_threshold:g}_b{args.boundary_threshold:g}_s{args.small_area}'
    if args.criterion != 'combined':
        out = out.with_name(out.name + '_' + args.criterion)
    out.mkdir(exist_ok=True)
    np.save(out / 'region_labels.npy', merged_labels)
    np.save(out / 'initial_to_merged.npy', mapping)
    rebuilt.graph.update(baseline='experimental, not Beveridge', criterion=args.criterion,
                         color_threshold=args.color_threshold, boundary_threshold=args.boundary_threshold,
                         small_area=args.small_area)
    set_graph_sources(rebuilt, out / 'rag_after.graphml.gz', image_path, out / 'region_labels.npy')
    nx.write_graphml(rebuilt, out / 'rag_after.graphml.gz')
    after_image = np.clip(means[merged_labels], 0, 255).astype(np.uint8)
    cv2.imencode('.png', cv2.cvtColor(after_image, cv2.COLOR_RGB2BGR))[1].tofile(out / 'segmentation_after.png')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, im, title in zip(axes, [rgb, before_image, after_image],
                            ['Original', f'Before fusion: {before_nodes:,} regions', f'After fusion: {len(rebuilt):,} regions']):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=160)
    plt.close(fig)
    plot_graph(rebuilt, rgb, means, out / 'rag_after.png', stage='after')
    h, w = labels.shape
    plot_graph(rebuilt, rgb, means, out / 'rag_detail_after.png',
               (max(0, w//2-50), max(0, h//2-50), min(w, w//2+50), min(h, h//2+50)), stage='after')
    summary = dict(criterion='experimental baseline, not Beveridge',
                   mode=args.criterion,
                   color_threshold=args.color_threshold, boundary_threshold=args.boundary_threshold,
                   initial_labels_sha256=sha256(labels_path), source_image_sha256=sha256(image_path),
                   small_area_priority=args.small_area, nodes_before=before_nodes, nodes_after=len(rebuilt),
                   edges_before=before_edges, edges_after=rebuilt.number_of_edges(),
                   merges=before_nodes-len(rebuilt), reduction_percent=100*(1-len(rebuilt)/before_nodes),
                   fusion_seconds=round(seconds, 3), total_area=int(labels.size),
                   verification='Rebuilt graph agrees with contraction; areas conserved; no eligible edge remains')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Saved to: {out.resolve()}')


if __name__ == '__main__':
    main()
