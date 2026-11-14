"""Shared image preprocessing and provenance checks between pipeline stages."""
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(path):
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Cannot decode image: {path}')
    h, w = bgr.shape[:2]
    scale = min(1.0, 960 / max(h, w))
    if scale < 1:
        bgr = cv2.resize(bgr, (max(1, round(w*scale)), max(1, round(h*scale))),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def verify_stage_metadata(labels_path, image_path=None):
    """Legacy arrays without metadata remain readable; new arrays are checked."""
    labels_path = Path(labels_path)
    meta_path = labels_path.with_name(labels_path.stem + '.metadata.json')
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if sha256(labels_path) != meta['labels_sha256']:
        raise ValueError('Labels changed since this stage was generated. Rerun the upstream stage.')
    if image_path is not None and sha256(image_path) != meta['source_image_sha256']:
        raise ValueError('This image differs from the one used for these labels. Rerun K-means and connected regions.')
    return meta


def write_stage_metadata(labels_path, image_path, **extra):
    labels_path = Path(labels_path)
    data = dict(source_image=str(Path(image_path).resolve()),
                source_image_relative=os.path.relpath(Path(image_path).resolve(), labels_path.resolve().parent),
                source_image_sha256=sha256(image_path), labels_sha256=sha256(labels_path),
                max_image_side=960, **extra)
    labels_path.with_name(labels_path.stem + '.metadata.json').write_text(
        json.dumps(data, indent=2), encoding='utf-8')


def set_graph_sources(graph, graph_path, image_path, labels_path):
    directory = Path(graph_path).resolve().parent
    for key, path in [('source_image', image_path), ('source_labels', labels_path)]:
        path = Path(path).resolve()
        graph.graph[key] = str(path)
        graph.graph[key + '_relative'] = os.path.relpath(path, directory)
        graph.graph[key + '_sha256'] = sha256(path)


def graph_sources(graph, graph_path):
    paths = []
    directory = Path(graph_path).resolve().parent
    for key in ('source_image', 'source_labels'):
        relative = graph.graph.get(key + '_relative')
        path = (directory / relative).resolve() if relative else Path(graph.graph[key])
        if not path.is_file():
            raise ValueError(f'Missing graph source: {path}. Regenerate the RAG.')
        expected = graph.graph.get(key + '_sha256')
        if expected and sha256(path) != expected:
            raise ValueError(f'{key} changed since this RAG was built. Regenerate downstream results.')
        paths.append(path)
    return tuple(paths)
