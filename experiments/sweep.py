"""Balayage de parametres pour le rapport : K, T et taille d'image.

Exemples (depuis le dossier du projet) :
  python experiments/sweep.py images/city.png --k 8 --t 0.1 0.15 0.2 0.3
  python experiments/sweep.py images/city.png --k 4 8 12 --t 0.2 --sizes 480 960

Sortie : experiments/output/<image>_sweep.csv (+ une image fusionnee par ligne).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'moteur'))
from experience import Experience  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('image')
    p.add_argument('--k', type=int, nargs='+', default=[8])
    p.add_argument('--t', type=float, nargs='+', default=[0.1, 0.15, 0.2, 0.3])
    p.add_argument('--sizes', type=int, nargs='+', default=[480])
    args = p.parse_args()

    image = Path(args.image)
    out = ROOT / 'experiments' / 'output'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in args.sizes:
        for k in args.k:
            e = Experience(image, size)
            e.kmeans(k)
            e.rag()
            before, edges_before = len(e.base), e.base.number_of_edges()
            for t in args.t:
                start = time.time()
                e.fusion(t)
                seconds = round(time.time() - start, 1)
                row = dict(image=image.name, size=size, k=k, T=t,
                           regions_before=before, edges_before=edges_before,
                           regions_after=len(e.final), edges_after=e.final.number_of_edges(),
                           seconds=seconds)
                rows.append(row)
                print(row, flush=True)
                name = f'{image.stem}_s{size}_k{k}_T{t}.png'
                cv2.imencode('.png', cv2.cvtColor(e.final_image, cv2.COLOR_RGB2BGR))[1].tofile(out / name)
    csv_path = out / f'{image.stem}_sweep.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', csv_path)


if __name__ == '__main__':
    main()
