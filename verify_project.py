"""Read-only numerical audit of saved pipeline artifacts (writes only audit JSON)."""
import argparse
import json
from pathlib import Path

import cv2
import networkx as nx
import numpy as np

from build_rag import build_rag
from connected_regions import split_regions
from compare_fusion import measure
from pipeline_io import read_rgb, sha256


def same_graph(actual, expected):
    assert set(actual)==set(expected), 'Node IDs differ'
    assert {tuple(sorted(e)) for e in actual.edges}=={tuple(sorted(e)) for e in expected.edges}, 'Adjacency differs'
    for n,data in expected.nodes(data=True):
        for key,value in data.items():
            assert np.isclose(actual.nodes[n][key],value), (n,key)
    for a,b,data in expected.edges(data=True):
        for key,value in data.items():
            assert np.isclose(actual[a][b][key],value), (a,b,key)


def check_connected_coarsening(base, mapping):
    """Each final region must induce one connected component in the initial RAG."""
    parent=list(range(len(mapping)))
    def find(n):
        while parent[n]!=n:
            parent[n]=parent[parent[n]]
            n=parent[n]
        return n
    for a,b in base.edges:
        if mapping[a]==mapping[b]:
            ra,rb=find(a),find(b)
            parent[ra]=rb
    roots=np.array([find(i) for i in range(len(mapping))])
    assert len(np.unique(np.column_stack([mapping,roots]),axis=0))==len(np.unique(mapping)), 'Disconnected merged region'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='outputs/city_k8')
    parser.add_argument('--image',default='city.png')
    parser.add_argument('--fresh-labels')
    args=parser.parse_args()
    root=Path(args.root)
    rgb=read_rgb(args.image)
    clusters=np.load(root/'cluster_labels.npy',allow_pickle=False)
    regions=np.load(root/'connected_regions/region_labels.npy',allow_pickle=False)
    assert rgb.shape[:2]==clusters.shape==regions.shape
    recreated,classes=split_regions(clusters)
    assert np.array_equal(regions,recreated), 'Connected regions do not reproduce'
    assert np.array_equal(np.asarray(classes)[regions],clusters)
    if args.fresh_labels:
        assert np.array_equal(clusters,np.load(args.fresh_labels,allow_pickle=False)), 'Fresh K-means labels differ'
    png=cv2.cvtColor(cv2.imdecode(np.fromfile(root/'segmentation.png',dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    assert png.shape==rgb.shape
    for color in np.unique(clusters):
        mask=clusters==color
        displayed=np.unique(png[mask],axis=0)
        assert len(displayed)==1
        assert np.max(np.abs(displayed[0].astype(float)-rgb[mask].mean(axis=0)))<1.1
    rag_path=root/'connected_regions/rag'
    base,_=build_rag(regions,rgb)
    stored=nx.read_graphml(rag_path/'rag_before.graphml.gz',node_type=int)
    same_graph(stored,base)
    del stored
    print(f'Initial stages OK: {len(base):,} regions; {base.number_of_edges():,} edges',flush=True)
    results_path=rag_path/'comparison_experiments/results.json'
    records=json.loads(results_path.read_text(encoding='utf-8'))['experiments']
    verified=[]
    folders=[(rag_path/'fusion_c20_b40_s10',None)]+[(rag_path/'comparison_experiments'/r['experiment'],r) for r in records]
    for folder,record in folders:
        labels=np.load(folder/'region_labels.npy',allow_pickle=False)
        mapping=np.load(folder/'initial_to_merged.npy',allow_pickle=False)
        assert mapping.shape==(len(base),)
        assert np.array_equal(mapping[regions],labels)
        check_connected_coarsening(base,mapping)
        rebuilt,means=build_rag(labels,rgb)
        assert nx.is_connected(rebuilt)
        assert sum(d['area'] for _,d in rebuilt.nodes(data=True))==regions.size
        if record is None:
            saved=nx.read_graphml(folder/'rag_after.graphml.gz',node_type=int)
            same_graph(saved,rebuilt)
            del saved
            summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            assert summary['nodes_after']==len(rebuilt)
            assert summary['edges_after']==rebuilt.number_of_edges()
            color,boundary,mode=20,40,'combined'
        else:
            metrics,_=measure(labels,rgb,regions)
            for key,value in metrics.items():
                assert (record[key] is None if value is None else np.isclose(record[key],value)), (folder.name,key)
            assert record['edges']==rebuilt.number_of_edges()
            assert np.isclose(record['reduction_percent'],100*(1-len(rebuilt)/len(base)))
            color,boundary,mode=record['color_threshold'],record['boundary_threshold'],record['criterion']
        assert all(d['color_distance']>color or (mode=='combined' and d['boundary_contrast']>boundary)
                   for _,_,d in rebuilt.edges(data=True)), 'Eligible pair remains'
        png_path=folder/('segmentation_after.png' if record is None else 'segmentation.png')
        displayed=cv2.cvtColor(cv2.imdecode(np.fromfile(png_path,dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        assert np.array_equal(displayed,np.clip(means[labels],0,255).astype(np.uint8))
        verified.append(dict(name=folder.name,regions=len(rebuilt),edges=rebuilt.number_of_edges(),
                             labels_sha256=sha256(folder/'region_labels.npy')))
        print(f'Verified {folder.name}: {len(rebuilt):,} connected regions',flush=True)
        del rebuilt
    result=dict(status='PASS',source_image_sha256=sha256(args.image),image_shape=list(rgb.shape),
                classes=len(np.unique(clusters)),initial_regions=len(base),initial_edges=base.number_of_edges(),
                total_pixels=int(regions.size),fresh_kmeans_compared=bool(args.fresh_labels),variants=verified,
                checks=['K-means palette and original color means','connected component reproduction',
                        'node/edge attributes from independent reconstruction','mapping and region connectivity',
                        'area conservation','saved metrics and rendered pixel colors','merge stopping conditions'])
    output=root/'audit'
    output.mkdir(exist_ok=True)
    (output/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'Audit PASS: {output.resolve()}',flush=True)


if __name__=='__main__':
    main()
