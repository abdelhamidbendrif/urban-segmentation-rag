"""Small cases that catch stale merge decisions and boundary mistakes."""
import unittest

import numpy as np

from build_rag import build_rag
from merge_regions import fuse


class FusionTests(unittest.TestCase):
    def test_rechecks_mean_after_merge(self):
        labels = np.array([[0, 1, 2]], dtype=np.int32)
        rgb = np.array([[[0, 0, 0], [10, 0, 0], [20, 0, 0]]], dtype=np.uint8)
        graph, _ = build_rag(labels, rgb)
        mapping = fuse(graph, color_threshold=11, boundary_threshold=50)
        self.assertEqual(len(graph), 2)
        self.assertEqual(mapping[0], mapping[1])
        self.assertNotEqual(mapping[1], mapping[2])
        self.assertEqual(graph.nodes[mapping[0]]['mean_r'], 5)

    def test_strong_boundary_blocks_equal_region_means(self):
        labels = np.array([[0, 1], [0, 1]], dtype=np.int32)
        rgb = np.array([[[0]*3, [100]*3], [[100]*3, [0]*3]], dtype=np.uint8)
        graph, _ = build_rag(labels, rgb)
        self.assertEqual(graph[0][1]['color_distance'], 0)
        fuse(graph, color_threshold=20, boundary_threshold=40)
        self.assertEqual(len(graph), 2)

    def test_full_merge_conserves_area_and_handles_single_region(self):
        labels = np.arange(9, dtype=np.int32).reshape(3, 3)
        rgb = np.full((3, 3, 3), 50, dtype=np.uint8)
        graph, _ = build_rag(labels, rgb)
        mapping = fuse(graph)
        self.assertEqual(len(graph), 1)
        self.assertEqual(len(set(mapping.values())), 1)
        self.assertEqual(next(iter(graph.nodes(data=True)))[1]['area'], 9)
        self.assertEqual(graph.number_of_edges(), 0)
        fuse(graph)  # An already merged graph must be supported too.

    def test_color_only_ignores_boundary(self):
        labels = np.array([[0, 1], [0, 1]], dtype=np.int32)
        rgb = np.array([[[0]*3, [100]*3], [[100]*3, [0]*3]], dtype=np.uint8)
        graph, _ = build_rag(labels, rgb)
        fuse(graph, criterion='color', small_area=0)
        self.assertEqual(len(graph), 1)

    def test_invalid_criterion(self):
        with self.assertRaises(ValueError):
            fuse({}, criterion='unknown')


if __name__ == '__main__':
    unittest.main()
