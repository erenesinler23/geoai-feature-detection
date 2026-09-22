import unittest
from unittest.mock import patch
from app import app, RESULTS_CACHE
from src.validation.validator import DetectionValidator
from test_evaluation import frame
from shapely.geometry import box


class AppTests(unittest.TestCase):
    def setUp(self):
        RESULTS_CACHE.clear()
        self.client=app.test_client()

    def test_empty_metrics_render(self):
        result=DetectionValidator().validate(frame([]),frame([box(0,0,10,10)]),'naive',32630)
        output={'scene_id':'test','scene_image':'test.png','results':[{'name':'Naive','image':'test.png','result':result}]}
        with patch('app.run_full_pipeline',return_value=output):
            response=self.client.post('/run',data={'preset':'ankara'})
        self.assertEqual(response.status_code,200)
        self.assertIn(b'N/A',response.data)

    def test_recent_search_uses_saved_name(self):
        with patch('app.geocode',return_value={'display_name':'Prague','lon':14.4,'lat':50.1,'search_bbox':[14.3,50,14.5,50.2]}) as geocode:
            with patch('app.run_full_pipeline',return_value={'scene_id':'test','scene_image':'x.png','results':[]}):
                response=self.client.post('/run',data={'preset':'prague','force':'true'})
        geocode.assert_called_once_with('prague')
        self.assertEqual(response.status_code,200)


if __name__=='__main__':unittest.main()
