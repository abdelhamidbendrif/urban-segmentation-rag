"""Split each K-means class into 4-connected regions, without merging."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pipeline_io import verify_stage_metadata, write_stage_metadata


def split_regions(clusters):
    regions = np.full(clusters.shape, -1, dtype=np.int32)
    classes = []
    offset = 0
    for color_class in np.unique(clusters):
        count, components = cv2.connectedComponents(
            (clusters == color_class).astype(np.uint8), connectivity=4
        )
        mask = components > 0
        regions[mask] = components[mask] - 1 + offset
        classes.extend([int(color_class)] * (count - 1))
        offset += count - 1
    return regions, classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labels', nargs='?', default='outputs/city_k8/cluster_labels.npy')
    args = parser.parse_args()
    path = Path(args.labels)
    if not path.is_file():
        parser.error(f'File not found: {path}')
    metadata = verify_stage_metadata(path)
    if metadata:
        source_image = (path.parent / metadata['source_image_relative']).resolve() if 'source_image_relative' in metadata else Path(metadata['source_image'])
        verify_stage_metadata(path, source_image)
    clusters = np.load(path, allow_pickle=False)
    if clusters.ndim != 2 or not np.issubdtype(clusters.dtype, np.integer):
        parser.error('Expected a 2D array of integer cluster labels.')
    regions, classes = split_regions(clusters)
    areas = np.bincount(regions.ravel())
    assert regions.min() == 0 and areas.sum() == clusters.size and np.all(areas > 0)
    # Each region contains exactly one original K-means class.
    assert np.array_equal(np.asarray(classes)[regions], clusters)
    out = path.parent / 'connected_regions'
    out.mkdir(exist_ok=True)
    np.save(out / 'region_labels.npy', regions)
    if metadata:
        write_stage_metadata(out / 'region_labels.npy', source_image, connectivity=4,
                             source_clusters_sha256=metadata['labels_sha256'])
    rng = np.random.default_rng(42)
    palette = rng.integers(30, 256, size=(len(classes), 3), dtype=np.uint8)
    cv2.imencode('.png', palette[regions])[1].tofile(out / 'regions.png')
    summary = {
        'connectivity': 4,
        'color_classes': len(np.unique(clusters)),
        'regions': len(classes),
        'regions_under_10_pixels': int((areas < 10).sum()),
        'smallest_area': int(areas.min()),
        'largest_area': int(areas.max()),
        'pixels': int(clusters.size),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Saved to: {out.resolve()}')


if __name__ == '__main__':
    main()
