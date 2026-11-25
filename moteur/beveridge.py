"""Critere de fusion de Beveridge pour un Region Adjacency Graph (RAG).

Source de la formule : cours de segmentation d'Anne Vialard (LaBRI, Universite
de Bordeaux), diapositives "Image Segmentation", section "Beveridge" :

    f(Ri, Rj) = f_sim(Ri, Rj) * sqrt( f_size(Ri, Rj) * f_cont(Ri, Rj) )

    f_sim  = |mu_i - mu_j| / max(1, sigma_i + sigma_j)
    f_size = min(2, min(|Ri|, |Rj|) / T_opt)           T_opt fixe selon la taille de l'image
    f_cont = clip(C, 1/2, 2)   avec   C = min(|dRi|, |dRj|) / (4 |dRi inter dRj|)

|Ri| est l'aire, dRi le contour de Ri (perimetre) et |dRi inter dRj| la longueur
de la frontiere commune. Un score FAIBLE signifie des regions a fusionner : on
fusionne de facon gloutonne la paire de score minimal tant que ce score reste
<= au seuil T. Apres chaque fusion on recalcule aire, moyenne, ecart-type,
perimetre et frontieres des voisins.

Points d'interpretation, a verifier contre les supports du professeur :
  * le cours est ecrit en niveaux de gris ; `space='gray'` (luminance) est la
    version fidele, `space='rgb'` en est une extension (distance euclidienne
    des moyennes RGB, sigma = racine de la somme des variances par canal) ;
  * la racine carree provient de la diapositive telle qu'extraite ; l'option
    `combine='product'` fournit la variante sans racine ;
  * `combine='mean_diff'` n'est PAS Beveridge : c'est le critere simple des supports ENSAJ
    (deux regions adjacentes fusionnent si |mu_A - mu_B| <= T), garde pour comparaison ;
    avec `max_sigma` il ajoute le predicat d'homogeneite sigma <= seuil du cours ;
  * le perimetre compte tous les cotes de pixel qui touchent une autre region
    ou le bord de l'image ; la frontiere commune est comptee dans la meme unite
    (cotes de pixel). Le facteur 4 est celui de la formule du cours.
"""
import heapq
import math

import networkx as nx
import numpy as np

from build_rag import build_rag

LUMA = np.array([0.299, 0.587, 0.114])
SPACES = ('gray', 'rgb')
COMBINES = ('sqrt', 'product', 'mean_diff')


# ----------------------------------------------------------------- statistiques
def augment_graph(graph, labels, rgb):
    """Ajoute a chaque noeud : perimetre, somme et somme des carres (RGB et luminance)."""
    n = graph.number_of_nodes()
    flat = labels.ravel()
    img = rgb.astype(np.float64)
    luma = img @ LUMA
    sq = [np.bincount(flat, weights=(img[:, :, c] ** 2).ravel(), minlength=n) for c in range(3)]
    sum_l = np.bincount(flat, weights=luma.ravel(), minlength=n)
    sumsq_l = np.bincount(flat, weights=(luma ** 2).ravel(), minlength=n)
    border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    border_sides = np.bincount(border, minlength=n)
    for node, data in graph.nodes(data=True):
        contact = sum(e['boundary_length'] for e in graph[node].values())
        data['perimeter'] = int(contact + border_sides[node])
        for c, name in enumerate('rgb'):
            data[f'sumsq_{name}'] = float(sq[c][node])
        data['sum_l'] = float(sum_l[node])
        data['sumsq_l'] = float(sumsq_l[node])
    return graph


def build_augmented_rag(labels, rgb):
    graph, means = build_rag(labels, rgb)
    augment_graph(graph, labels, rgb)
    return graph, means


def mean_sigma(d, space):
    """(vecteur moyenne, ecart-type) d'une region dans l'espace choisi."""
    n = d['area']
    if space == 'gray':
        mu = d['sum_l'] / n
        var = d['sumsq_l'] / n - mu * mu
        return np.array([mu]), math.sqrt(max(var, 0.0))
    mus = np.array([d[f'sum_{c}'] / n for c in 'rgb'])
    var = sum(d[f'sumsq_{c}'] / n - mus[i] ** 2 for i, c in enumerate('rgb'))
    return mus, math.sqrt(max(var, 0.0))


# ------------------------------------------------------------------- le critere
def beveridge_factors(da, db, shared, topt, space='gray'):
    """Retourne (f_sim, f_size, f_cont) pour deux regions et leur frontiere commune."""
    mu_a, s_a = mean_sigma(da, space)
    mu_b, s_b = mean_sigma(db, space)
    f_sim = float(np.linalg.norm(mu_a - mu_b)) / max(1.0, s_a + s_b)
    f_size = min(2.0, min(da['area'], db['area']) / topt)
    c = min(da['perimeter'], db['perimeter']) / (4.0 * shared)
    f_cont = min(2.0, max(0.5, c))
    return f_sim, f_size, f_cont


def beveridge_score(da, db, shared, topt, space='gray', combine='sqrt',
                    use_size=True, use_contour=True):
    if combine == 'mean_diff':  # critere simple du cours ENSAJ : |mu_A - mu_B| <= T
        return float(np.linalg.norm(mean_sigma(da, space)[0] - mean_sigma(db, space)[0]))
    f_sim, f_size, f_cont = beveridge_factors(da, db, shared, topt, space)
    if not use_size:
        f_size = 1.0
    if not use_contour:
        f_cont = 1.0
    if combine == 'sqrt':
        return f_sim * math.sqrt(f_size * f_cont)
    return f_sim * f_size * f_cont


def default_topt(shape, fraction=0.001):
    """T_opt = fraction de la surface de l'image (au moins 1 pixel)."""
    return max(1.0, fraction * shape[0] * shape[1])


# ----------------------------------------------------------------------- fusion
def fuse_beveridge(graph, threshold, topt, space='gray', combine='sqrt', use_size=True,
                   use_contour=True, max_sigma=None, min_regions=1):
    """Fusion gloutonne de Beveridge ; modifie `graph` (contraction d'aretes).

    Conditions d'arret : (1) plus aucune paire adjacente de score <= threshold
    (et, si max_sigma est donne, d'ecart-type de l'union <= max_sigma) ;
    (2) nombre de regions == min_regions.
    Retourne (parent, history) ou history est un tableau (n_regions, score)
    de chaque fusion effectuee.
    """
    if not (math.isfinite(threshold) and threshold >= 0 and topt > 0):
        raise ValueError('threshold doit etre fini et >= 0, topt > 0.')
    if space not in SPACES or combine not in COMBINES:
        raise ValueError(f'space dans {SPACES}, combine dans {COMBINES}.')
    if min_regions < 1:
        raise ValueError('min_regions >= 1.')
    if any('perimeter' not in d for _, d in graph.nodes(data=True)):
        raise ValueError('Le graphe doit etre augmente (augment_graph).')
    parent = {n: n for n in graph}
    version = {n: 0 for n in graph}
    heap = []
    kw = dict(space=space, combine=combine, use_size=use_size, use_contour=use_contour)

    def union_sigma(da, db):
        n = da['area'] + db['area']
        if space == 'gray':
            s, q = da['sum_l'] + db['sum_l'], da['sumsq_l'] + db['sumsq_l']
            return math.sqrt(max(q / n - (s / n) ** 2, 0.0))
        var = 0.0
        for c in 'rgb':
            s, q = da[f'sum_{c}'] + db[f'sum_{c}'], da[f'sumsq_{c}'] + db[f'sumsq_{c}']
            var += q / n - (s / n) ** 2
        return math.sqrt(max(var, 0.0))

    def queue(a, b):
        a, b = sorted((a, b))
        da, db = graph.nodes[a], graph.nodes[b]
        shared = graph[a][b]['boundary_length']
        score = beveridge_score(da, db, shared, topt, **kw)
        graph[a][b]['beveridge_f'] = score
        if score <= threshold and (max_sigma is None or union_sigma(da, db) <= max_sigma):
            heapq.heappush(heap, (score, a, b, version[a], version[b]))

    for a, b in list(graph.edges):
        queue(a, b)
    scores, counts = [], []
    while heap and len(graph) > min_regions:
        score, a, b, va, vb = heapq.heappop(heap)
        if a not in graph or b not in graph or version[a] != va or version[b] != vb:
            continue  # decision calculee avant un changement de a ou b
        if graph.nodes[a]['area'] < graph.nodes[b]['area']:
            a, b = b, a
        da, db = graph.nodes[a], graph.nodes[b]
        shared_ab = graph[a][b]['boundary_length']
        area = da['area'] + db['area']
        for coord in ('x', 'y'):
            da[coord] = (da[coord] * da['area'] + db[coord] * db['area']) / area
        for key in [k for k in da if k.startswith(('sum_', 'sumsq_'))]:
            da[key] += db[key]
        da['area'] = area
        for c in 'rgb':
            da[f'mean_{c}'] = da[f'sum_{c}'] / area
        da['perimeter'] = da['perimeter'] + db['perimeter'] - 2 * shared_ab
        for neighbor, edge in list(graph[b].items()):
            if neighbor == a:
                continue
            if graph.has_edge(a, neighbor):
                graph[a][neighbor]['boundary_length'] += edge['boundary_length']
                graph[a][neighbor]['boundary_contrast_sum'] += edge['boundary_contrast_sum']
            else:
                graph.add_edge(a, neighbor, boundary_length=edge['boundary_length'],
                               boundary_contrast_sum=edge['boundary_contrast_sum'],
                               boundary_contrast=0.0, color_distance=0.0)
        graph.remove_node(b)
        parent[b] = a
        version[a] += 1
        for neighbor in graph[a]:
            queue(a, neighbor)
        scores.append(score)
        counts.append(len(graph))
    for node in parent:  # compression de chemin : chaque region initiale -> survivante
        root = node
        while parent[root] != root:
            root = parent[root]
        current = node
        while parent[current] != current:
            following = parent[current]
            parent[current] = root
            current = following
    history = np.column_stack([counts, scores]) if scores else np.zeros((0, 2))
    return parent, history


def relabel(parent, labels, n_initial):
    """Etiquettes fusionnees consecutives + correspondance initiale -> fusionnee."""
    survivors = sorted(set(parent.values()))
    compact = {node: i for i, node in enumerate(survivors)}
    mapping = np.array([compact[parent[n]] for n in range(n_initial)], dtype=np.int32)
    return mapping[labels], mapping, compact


# ----------------------------------------------------------------- verification
def pairs_below_threshold(graph, threshold, topt, **kw):
    """Paires adjacentes dont le score recalcule est <= threshold (doit etre vide a l'arret)."""
    keep = {k: kw[k] for k in ('space', 'combine', 'use_size', 'use_contour') if k in kw}
    return [(a, b) for a, b in graph.edges
            if beveridge_score(graph.nodes[a], graph.nodes[b], graph[a][b]['boundary_length'],
                               topt, **keep) <= threshold]


def verify_contraction(contracted, merged_labels, rgb, compact, full=True):
    """full=False : ne controle que aires, moyennes RGB et frontieres (fusion baseline)."""
    """Reconstruit le RAG depuis les etiquettes fusionnees et le compare a la contraction."""
    rebuilt, means = build_augmented_rag(merged_labels, rgb)
    contracted = nx.relabel_nodes(contracted, compact, copy=True)
    assert {tuple(sorted(e)) for e in rebuilt.edges} == {tuple(sorted(e)) for e in contracted.edges}
    for n, d in rebuilt.nodes(data=True):
        c = contracted.nodes[n]
        assert d['area'] == c['area']
        keys = ('mean_r', 'mean_g', 'mean_b')
        if full:
            assert d['perimeter'] == c['perimeter']
            keys += ('sumsq_r', 'sumsq_g', 'sumsq_b', 'sum_l', 'sumsq_l')
        for key in keys:
            assert np.isclose(d[key], c[key], rtol=1e-9, atol=1e-6), (n, key)
    for a, b, d in rebuilt.edges(data=True):
        assert d['boundary_length'] == contracted[a][b]['boundary_length']
        assert np.isclose(d['boundary_contrast_sum'], contracted[a][b]['boundary_contrast_sum'])
    assert sum(d['area'] for _, d in rebuilt.nodes(data=True)) == merged_labels.size
    assert nx.is_connected(rebuilt)
    return rebuilt, means
