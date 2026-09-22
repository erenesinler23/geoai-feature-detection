import unittest
import tempfile
from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from rasterio.transform import from_origin
from src.validation.validator import DetectionValidator
from src.pipeline import Pipeline, FetchedScene, restrict_detections
from src.vectors.osm_fetcher import OSMBuildingFetcher
from src.detectors.naive_color import NaiveColorDetector
from src.detectors.hybrid import fuse_detections


def frame(geometries, crs=32630):
    return gpd.GeoDataFrame({'detection_id': list(range(1,len(geometries)+1)), 'geometry':geometries},crs=crs)


class EvaluationTests(unittest.TestCase):
    def test_duplicate_predictions(self):
        result=DetectionValidator().validate(frame([box(0,0,10,10)]*2),frame([box(0,0,10,10)]),'test',32630)
        self.assertEqual((result.n_matched,result.precision,result.recall),(1,50,100))

    def test_one_box_cannot_match_two_buildings(self):
        result=DetectionValidator().validate(frame([box(0,0,20,10)]),frame([box(0,0,10,10),box(10,0,20,10)]),'test',32630)
        self.assertEqual(result.n_matched,1)
        self.assertEqual(result.recall,50)

    def test_touching_is_not_matching(self):
        result=DetectionValidator().validate(frame([box(0,0,10,10)]),frame([box(10,0,20,10)]),'test',32630)
        self.assertEqual(result.n_matched,0)

    def test_empty_predictions(self):
        result=DetectionValidator().validate(frame([]),frame([box(0,0,10,10)]),'test',32630)
        self.assertIsNone(result.precision)
        self.assertIsNone(result.false_discovery_rate)
        self.assertEqual(result.recall,0)

    def test_empty_references(self):
        result=DetectionValidator().validate(frame([box(0,0,10,10)]),frame([]),'test',32630)
        self.assertIsNone(result.recall)
        self.assertEqual(result.precision,0)

    def test_metric_size(self):
        result=DetectionValidator(max_size_m=50).validate(frame([box(0,0,51,10)]),frame([]),'test',32630)
        self.assertEqual(result.n_oversized_excluded,1)

    def test_reprojected_inputs(self):
        b=frame([box(400000,5800000,400010,5800010)])
        result=DetectionValidator().validate(b.to_crs(4326),b,'test',32630)
        self.assertAlmostEqual(result.precision,100)

    def test_duplicate_ids_rejected(self):
        d=frame([box(0,0,10,10)]*2);d.detection_id=1
        with self.assertRaises(ValueError):DetectionValidator().validate(d,frame([]),'test',32630)

    def test_uint8_subtraction(self):
        r=np.array([[10]],dtype='uint8');g=np.array([[240]],dtype='uint8')
        d=NaiveColorDetector(0,20,20).detect(r,g,g,from_origin(400000,5800000,10,10),32630)
        self.assertTrue(d.empty)

    def test_crop_and_mask_reference_scope(self):
        class ImageSource:
            metadata={}
            valid_mask=np.ones((10,10),dtype=bool)
            def fetch(self,*args):return *(np.ones((10,10))*100 for _ in range(3)),from_origin(400000,5800000,10,10),32630,'test'
        class MapSource:
            metadata={}
            def fetch(self,bbox):
                self.bbox=bbox
                return frame([box(400010,5799920,400020,5799930),box(399990,5799920,400005,5799930)])
        source=MapSource();scene=Pipeline(ImageSource(),source,DetectionValidator()).fetch(-3,52)
        self.assertEqual(len(scene.buildings),1)
        self.assertEqual(scene.metadata['boundary_or_mask_buildings_excluded'],1)

    def test_relation_hole(self):
        def ring(coords):return [{'lon':x,'lat':y} for x,y in coords]
        data={'elements':[{'type':'relation','id':1,'tags':{'building':'yes'},'members':[
            {'type':'way','ref':2,'role':'outer','geometry':ring([(0,0),(10,0),(10,10),(0,10),(0,0)])},
            {'type':'way','ref':3,'role':'inner','geometry':ring([(2,2),(4,2),(4,4),(2,4),(2,2)])}]}]}
        result=OSMBuildingFetcher()._parse(data)
        self.assertEqual(result.geometry.iloc[0].area,96)

    def test_hybrid_large_box(self):
        primary=frame([box(400000,5800000,400010,5800010)]).to_crs(4326)
        confirming=frame([box(399000,5799000,402000,5802000)]).to_crs(4326)
        self.assertTrue(fuse_detections(primary,confirming).empty)

    def test_assignment_keeps_two_valid_pairs(self):
        predictions=frame([box(0,0,18,10),box(0,0,8,10)])
        references=frame([box(0,0,10,10),box(8,0,18,10)])
        result=DetectionValidator().validate(predictions,references,'test',32630)
        self.assertEqual(result.n_matched,2)

    def test_subqueries_deduplicate_buildings(self):
        from unittest.mock import patch
        data={'elements':[{'type':'way','id':1,'tags':{'building':'yes'},'geometry':[
            {'lon':0,'lat':0},{'lon':1,'lat':0},{'lon':1,'lat':1},{'lon':0,'lat':0}]}]}
        source=OSMBuildingFetcher()
        replies=[RuntimeError('busy')]+[(data,{'osm_base':'test'})]*4
        with patch.object(source,'_request',side_effect=replies):
            result=source.fetch((0,0,1,1))
        self.assertEqual(len(result),1)
        self.assertEqual(len(source.metadata['subqueries']),4)

    def test_partial_download_is_not_scored(self):
        from unittest.mock import patch
        source=OSMBuildingFetcher()
        with patch.object(source,'_request',side_effect=RuntimeError('busy')):
            with self.assertRaises(RuntimeError):source.fetch((0,0,1,1))

    def test_invalid_polygons_are_repaired(self):
        from shapely.geometry import Polygon
        from src.pipeline import clean_polygons
        result=clean_polygons(frame([Polygon([(0,0),(2,2),(2,0),(0,2),(0,0)])]))
        self.assertTrue(result.geometry.is_valid.all())
        self.assertFalse(result.empty)


if __name__=='__main__':unittest.main()
