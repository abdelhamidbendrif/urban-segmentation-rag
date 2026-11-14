"""First experiment: segment an urban image by color with K-means."""

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from pipeline_io import read_rgb, write_stage_metadata

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default="city.jpg")
    parser.add_argument("--k", type=int, default=8, help="Number of color groups")
    parser.add_argument("--output-root", default="outputs", help="Parent folder for experiment outputs")
    args = parser.parse_args()
    path = Path(args.image)
    if not path.is_file():
        parser.error(f"Image not found: {path}. Put city.jpg in the project folder or supply an image path.")
    try:
        rgb = read_rgb(path)
    except ValueError as exc:
        parser.error(str(exc))
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    if not 2 <= args.k <= len(np.unique(pixels, axis=0)):
        parser.error("K must be at least 2 and no greater than the number of distinct colors.")
    cv2.setRNGSeed(42)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, args.k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    segmented = np.clip(centers[labels.ravel()], 0, 255).astype(np.uint8).reshape(rgb.shape)
    output = Path(args.output_root) / f"{path.stem}_k{args.k}"
    output.mkdir(parents=True, exist_ok=True)
    plt.imsave(output / "segmentation.png", segmented)
    np.save(output / "cluster_labels.npy", labels.reshape(rgb.shape[:2]))
    write_stage_metadata(output / 'cluster_labels.npy', path, k=args.k, seed=42,
                         centers_rgb=centers.tolist(), shape=list(rgb.shape[:2]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for axis, picture, title in zip(axes, [rgb, segmented], ["Original", f"K-means (K={args.k})"]):
        axis.imshow(picture)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    print(f"Results saved to: {output.resolve()}")
    print("Color groups are not yet connected regions or semantic object classes.")


if __name__ == "__main__":
    main()
