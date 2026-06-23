import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_article, predict_with_shap, load_all_models

class TestPredictionPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Ensure models are loaded
        cls.sbert, cls.clf, cls.meta = load_all_models(
            model_path="models/fake_news_model.pkl",
            metadata_path="models/model_metadata.pkl"
        )
        
    def test_end_to_end_prediction(self):
        sample_text = (
            "Breaking: Scientists at a top university have announced a breakthrough in "
            "fusion energy. The team successfully generated more energy than consumed, "
            "marking a major milestone for clean electricity generation globally. Officials "
            "confirm that the experiment was repeated multiple times with consistent results."
        )
        
        # Test base prediction
        res = predict_article(sample_text, self.sbert, self.clf, self.meta)
        
        # Validation checks
        self.assertIn("prediction", res)
        self.assertIn("label", res)
        self.assertIn("confidence", res)
        self.assertIn("stylometric_features", res)
        
        self.assertIn(res["label"], ["REAL", "FAKE"])
        self.assertTrue(0.5 <= res["confidence"] <= 1.0)
        self.assertIsInstance(res["prediction"], int)
        
        # Validate dtypes of features
        style_feats = res["stylometric_features"]
        for key, val in style_feats.items():
            self.assertIsInstance(val, float, f"Feature '{key}' is not float. Got type: {type(val)}")
            
        # Check feature vector structure
        feat_vector = res["feature_vector"]
        self.assertEqual(feat_vector.shape, (1, 392))
        self.assertEqual(feat_vector.dtype, np.float32)
        self.assertFalse(np.isnan(feat_vector).any(), "Feature vector contains NaN!")
        self.assertFalse(np.isinf(feat_vector).any(), "Feature vector contains Inf!")
        
    def test_shap_explanation(self):
        sample_text = (
            "This is another fake news headline about aliens landing in New York City! "
            "Unbelievable footage shows UFO flying near the Statue of Liberty. Share this "
            "immediately before it gets censored by the mainstream media!!!"
        )
        
        res = predict_with_shap(
            sample_text,
            top_n=5,
            model_path="models/fake_news_model.pkl",
            metadata_path="models/model_metadata.pkl"
        )
        
        self.assertIn("prediction", res)
        self.assertIn("shap_explanation", res)
        
        # Check SHAP structure if it succeeded
        shap_exp = res["shap_explanation"]
        if shap_exp is not None:
            self.assertIn("top_positive", shap_exp)
            self.assertIn("top_negative", shap_exp)
            
            # Verify shapes/sizes
            self.assertTrue(len(shap_exp["top_positive"]) <= 5)
            self.assertTrue(len(shap_exp["top_negative"]) <= 5)
            
            for item in shap_exp["top_positive"] + shap_exp["top_negative"]:
                self.assertIn("feature", item)
                self.assertIn("shap_value", item)
                self.assertIsInstance(item["shap_value"], float)

if __name__ == "__main__":
    unittest.main()
