"""Small teaching workflow using the reviewed Beveridge implementation."""
import json
from pathlib import Path

import cv2
import numpy as np
import beveridge as bev
from connected_regions import split_regions
from pipeline_io import read_rgb


class Experience:
    def __init__(self, path, max_side=480):
        self.path = Path(path)
        self.rgb = read_rgb(path, max_side)
        self.clear()

    def clear(self):
        self.clusters = self.regions = self.base = self.merged = self.final = None
        self.k = self.threshold = None

    def kmeans(self, k):
        if not 2 <= k <= len(np.unique(self.rgb.reshape(-1, 3), axis=0)):
            raise ValueError('Choisir K entre 2 et le nombre de couleurs distinctes.')
        self.clear()
        pixels = self.rgb.reshape(-1, 3).astype(np.float32)
        cv2.setRNGSeed(42)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        self.clusters = labels.reshape(self.rgb.shape[:2])
        self.k = k
        self.kmeans_image = centers[labels.ravel()].clip(0,255).astype(np.uint8).reshape(self.rgb.shape)
        return self.kmeans_image

    def rag(self):
        if self.clusters is None:
            raise ValueError('Executer K-means avant de construire les regions.')
        self.regions, _ = split_regions(self.clusters)
        self.base, means = bev.build_augmented_rag(self.regions, self.rgb)
        self.merged = self.final = None
        return means[self.regions].clip(0,255).astype(np.uint8)

    def fusion(self, threshold):
        if self.base is None:
            raise ValueError('Construire les regions et le RAG avant la fusion.')
        work = self.base.copy()  # Each new T starts from the same initial regions.
        topt = bev.default_topt(self.regions.shape)
        parent, history = bev.fuse_beveridge(work, threshold, topt, space='gray')
        labels, mapping, compact = bev.relabel(parent, self.regions, len(self.base))
        final, means = bev.verify_contraction(work, labels, self.rgb, compact)
        if bev.pairs_below_threshold(final, threshold, topt, space='gray'):
            raise AssertionError('Une paire eligible subsiste apres fusion.')
        self.threshold, self.merged, self.final = threshold, labels, final
        self.mapping, self.history = mapping, history
        self.final_image = means[labels].clip(0,255).astype(np.uint8)
        return self.final_image

    def summary(self):
        return dict(image=self.path.name, width=self.rgb.shape[1], height=self.rgb.shape[0],
                    k=self.k, threshold=self.threshold, space='gray',
                    topt=bev.default_topt(self.rgb.shape[:2]),
                    regions_before=len(self.base) if self.base is not None else None,
                    edges_before=self.base.number_of_edges() if self.base is not None else None,
                    regions_after=len(self.final) if self.final is not None else None,
                    edges_after=self.final.number_of_edges() if self.final is not None else None)

    def save(self, folder):
        if self.final is None:
            raise ValueError('Terminer la fusion avant de sauvegarder.')
        folder = Path(folder)
        folder.mkdir(parents=True,exist_ok=True)
        for name, img in [('original',self.rgb),('kmeans',self.kmeans_image),('fusion',self.final_image)]:
            cv2.imencode('.png',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))[1].tofile(folder/f'{name}.png')
        np.save(folder/'regions_initiales.npy',self.regions)
        np.save(folder/'regions_fusionnees.npy',self.merged)
        (folder/'mesures.json').write_text(json.dumps(self.summary(),indent=2),encoding='utf-8')


def window_graph(labels, rgb):
    """Spatial window using global region IDs, not full-image graph counts."""
    h,w=labels.shape
    x0=max(0,w//2-32);y0=max(0,h//2-20)
    labels=labels[y0:min(h,y0+40),x0:min(w,x0+64)]
    rgb=rgb[y0:y0+labels.shape[0],x0:x0+labels.shape[1]]
    ids,inverse=np.unique(labels,return_inverse=True)
    lab=inverse.reshape(labels.shape)
    yy,xx=np.indices(lab.shape)
    areas=np.bincount(lab.ravel())
    cx=np.bincount(lab.ravel(),weights=xx.ravel())/areas
    cy=np.bincount(lab.ravel(),weights=yy.ravel())/areas
    distances=(xx-cx[lab])**2+(yy-cy[lab])**2
    order=np.lexsort((distances.ravel(),lab.ravel()))
    first=np.searchsorted(lab.ravel()[order],np.arange(len(ids)))
    points=np.column_stack([xx.ravel()[order][first],yy.ravel()[order][first]])
    edges=set()
    for a,b in [(lab[:,:-1],lab[:,1:]),(lab[:-1,:],lab[1:,:])]:
        mask=a!=b
        edges.update(map(tuple,np.sort(np.column_stack([a[mask],b[mask]]),axis=1)))
    return rgb,points,sorted(edges)
