"""Regression checks for connectivity, weighted statistics and input identity."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import networkx as nx

from connected_regions import split_regions
from build_rag import build_rag
from compare_fusion import measure
from merge_regions import fuse
from pipeline_io import (write_stage_metadata, verify_stage_metadata,
                         set_graph_sources, graph_sources)


class PipelineTests(unittest.TestCase):
    def test_diagonal_same_class_is_two_regions(self):
        classes = np.array([[0,1],[1,0]], dtype=np.int32)
        regions, source = split_regions(classes)
        self.assertEqual(len(source), 4)
        self.assertNotEqual(regions[0,0], regions[1,1])
        np.testing.assert_array_equal(np.array(source)[regions], classes)

    def test_shared_boundaries_and_centroid(self):
        labels = np.array([[0,0,1],[0,2,1]], dtype=np.int32)
        rgb = np.array([[10,20,30],[40,50,60],[70,80,90]], dtype=np.uint8)[labels]
        graph, _ = build_rag(labels, rgb)
        self.assertEqual(graph.nodes[0]['area'], 3)
        self.assertAlmostEqual(graph.nodes[0]['x'], 1/3)
        self.assertAlmostEqual(graph.nodes[0]['y'], 1/3)
        self.assertEqual(graph[0][2]['boundary_length'], 2)
        self.assertEqual(graph[0][1]['boundary_length'], 1)
        self.assertAlmostEqual(graph[0][1]['color_distance'], np.sqrt(2700))
        self.assertAlmostEqual(graph[0][1]['boundary_contrast'], np.sqrt(2700))

    def test_no_diagonal_edges(self):
        labels = np.arange(4, dtype=np.int32).reshape(2,2)
        graph, _ = build_rag(labels, np.zeros((2,2,3), dtype=np.uint8))
        self.assertFalse(graph.has_edge(0,3))
        self.assertFalse(graph.has_edge(1,2))
        self.assertEqual(graph.number_of_edges(),4)

    def test_weighted_mean_not_average_of_means(self):
        labels = np.array([[0,0,0,1]], dtype=np.int32)
        rgb = np.array([[[0,0,0]]*3 + [[40,0,0]]], dtype=np.uint8)
        graph, _ = build_rag(labels,rgb)
        fuse(graph,50,50)
        self.assertEqual(len(graph),1)
        data = next(iter(graph.nodes(data=True)))[1]
        self.assertEqual(data['area'],4)
        self.assertEqual(data['mean_r'],10)

    def test_non_finite_thresholds_are_rejected(self):
        for value in (float('nan'),float('inf'),-1,0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fuse(nx.Graph(),color_threshold=value)
            with self.subTest(boundary=value), self.assertRaises(ValueError):
                fuse(nx.Graph(),boundary_threshold=value)

    def test_metric_is_color_error_and_boundary_retention(self):
        rgb=np.array([[[0,0,0],[100,0,0]]],dtype=np.uint8)
        original=np.array([[0,1]],dtype=np.int32)
        result,_=measure(np.zeros_like(original),rgb,original)
        self.assertAlmostEqual(result['rgb_rmse'],np.sqrt(2500/3))
        self.assertEqual(result['strong_boundary_retention_percent'],0)

    def test_no_strong_boundary_has_no_percentage(self):
        rgb=np.zeros((1,2,3),dtype=np.uint8)
        original=np.array([[0,1]],dtype=np.int32)
        result,_=measure(original,rgb,original)
        self.assertIsNone(result['strong_boundary_retention_percent'])

    def test_metadata_detects_changed_image_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            image=root/'image.png'
            labels=root/'labels.npy'
            image.write_bytes(b'original')
            np.save(labels,np.array([[0]],dtype=np.int32))
            write_stage_metadata(labels,image)
            verify_stage_metadata(labels,image)
            image.write_bytes(b'changed!')
            with self.assertRaises(ValueError):
                verify_stage_metadata(labels,image)
            image.write_bytes(b'original')
            np.save(labels,np.array([[1]],dtype=np.int32))
            with self.assertRaises(ValueError):
                verify_stage_metadata(labels,image)

    def test_graph_sources_are_portable_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            image=root/'image.png'; image.write_bytes(b'image')
            labels=root/'labels.npy'; labels.write_bytes(b'labels')
            graph_path=root/'rag.graphml.gz'
            graph=nx.Graph()
            set_graph_sources(graph,graph_path,image,labels)
            graph.graph['source_image']='X:/nonexistent/path.png'
            self.assertEqual(graph_sources(graph,graph_path)[0],image.resolve())
            labels.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                graph_sources(graph,graph_path)

    def test_randomized_contractions_match_reconstruction(self):
        rng=np.random.default_rng(17)
        for trial in range(12):
            labels=np.arange(20,dtype=np.int32).reshape(4,5)
            rgb=rng.integers(0,60,size=(4,5,3),dtype=np.uint8)
            mode='combined' if trial%2 else 'color'
            graph,_=build_rag(labels,rgb)
            parent=fuse(graph,40,50,10 if trial%3 else 0,mode)
            compact={n:i for i,n in enumerate(sorted(graph))}
            merged=np.array([compact[parent[n]] for n in range(20)])[labels]
            rebuilt,_=build_rag(merged,rgb)
            contracted=nx.relabel_nodes(graph,compact,copy=True)
            self.assertEqual({tuple(sorted(e)) for e in rebuilt.edges},
                             {tuple(sorted(e)) for e in contracted.edges})
            for node,data in rebuilt.nodes(data=True):
                for key,value in data.items():
                    self.assertAlmostEqual(value,contracted.nodes[node][key])
                count,_=__import__('cv2').connectedComponents((merged==node).astype(np.uint8),connectivity=4)
                self.assertEqual(count,2)
            for a,b,data in rebuilt.edges(data=True):
                for key,value in data.items():
                    self.assertAlmostEqual(value,contracted[a][b][key])
                self.assertTrue(data['color_distance']>40 or (mode=='combined' and data['boundary_contrast']>50))


if __name__=='__main__':
    unittest.main()
