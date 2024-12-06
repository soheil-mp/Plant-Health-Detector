import unittest
import os
from plant_detector import PlantDiseaseDetector

class TestPlantDiseaseDetector(unittest.TestCase):
    def setUp(self):
        self.detector = PlantDiseaseDetector()
    
    def test_initialization(self):
        self.assertIsNone(self.detector.model)
        self.assertEqual(self.detector.image_size, (224, 224))
        self.assertEqual(self.detector.class_names, [])
    
    def test_invalid_model_path(self):
        with self.assertRaises(Exception):
            self.detector.load_model("nonexistent_model.h5")

if __name__ == '__main__':
    unittest.main() 