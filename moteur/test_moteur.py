import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'moteur'))
from experience import Experience, window_graph
import beveridge as bev


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.image=Path(self.temp.name)/'test.png'
        rgb=np.zeros((12,20,3),dtype=np.uint8)
        rgb[:,:10]=[100,100,100];rgb[:,10:]=[110,110,110]
        cv2.imencode('.png',rgb)[1].tofile(self.image)
        self.engine=Experience(self.image)

    def tearDown(self):
        self.temp.cleanup()

    def test_stage_order(self):
        with self.assertRaises(ValueError):self.engine.rag()
        with self.assertRaises(ValueError):self.engine.fusion(.2)

    def test_kmeans_classes_and_connected_regions(self):
        image=self.engine.kmeans(2)
        self.assertEqual(len(np.unique(image.reshape(-1,3),axis=0)),2)
        self.engine.rag()
        self.assertEqual(len(self.engine.base),2)
        self.assertEqual(self.engine.base.number_of_edges(),1)

    def test_fusion_checks_and_repeats_from_initial_partition(self):
        self.engine.kmeans(2);self.engine.rag()
        before=len(self.engine.base)
        self.engine.fusion(100)
        self.assertEqual(len(self.engine.final),1)
        self.engine.fusion(0)
        self.assertEqual(len(self.engine.final),2)
        self.assertEqual(len(self.engine.base),before)

    def test_changing_k_clears_downstream_results(self):
        self.engine.kmeans(2);self.engine.rag();self.engine.fusion(.2)
        self.engine.kmeans(2)
        self.assertIsNone(self.engine.base)
        self.assertIsNone(self.engine.final)

    def test_export_labels_and_image_agree(self):
        self.engine.kmeans(2);self.engine.rag();self.engine.fusion(.2)
        folder=Path(self.temp.name)/'result'
        self.engine.save(folder)
        self.assertTrue((folder/'mesures.json').exists())
        np.testing.assert_array_equal(np.load(folder/'regions_fusionnees.npy'),self.engine.merged)

    def test_window_nodes_stay_inside_regions(self):
        self.engine.kmeans(2);self.engine.rag()
        rgb,points,edges=window_graph(self.engine.regions,self.engine.rgb)
        self.assertEqual(len(points),2)
        self.assertEqual(len(edges),1)
        for x,y in points:
            self.assertTrue(0<=x<rgb.shape[1] and 0<=y<rgb.shape[0])

    def test_formula_perimeter_and_weighted_statistics(self):
        labels=np.array([[0,0,1,1],[0,0,1,1]],dtype=np.int32)
        rgb=np.repeat(np.array([[100,100,110,110]]*2,dtype=np.uint8)[:,:,None],3,axis=2)
        graph,_=bev.build_augmented_rag(labels,rgb)
        self.assertEqual(graph.nodes[0]['perimeter'],8)
        self.assertAlmostEqual(bev.beveridge_score(graph.nodes[0],graph.nodes[1],2,8),10*np.sqrt(.5))


if __name__=='__main__':unittest.main()
