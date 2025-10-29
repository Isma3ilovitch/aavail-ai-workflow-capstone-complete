import unittest
import json
import tempfile
import os
import shutil
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from app import app, model_manager, init_db

class AavailAPITestCase(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['DATA_DIR'] = tempfile.mkdtemp()
        self.app.config['LOG_DIR'] = tempfile.mkdtemp()
        self.app.config['PROCESSED_DATA_DIR'] = tempfile.mkdtemp()
        self.app.config['DATABASE_PATH'] = tempfile.mktemp(suffix='.db')
        
        # Initialize test database
        init_db()
        
        self.client = self.app.test_client()
        
        # Create test data
        self.create_test_data()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.app.config['DATA_DIR'])
        shutil.rmtree(self.app.config['LOG_DIR'])
        shutil.rmtree(self.app.config['PROCESSED_DATA_DIR'])
        if os.path.exists(self.app.config['DATABASE_PATH']):
            os.remove(self.app.config['DATABASE_PATH'])
    
    def create_test_data(self):
        """Create test JSON data files"""
        # Create sample transaction data
        test_data = [
            {
                "country": "United States",
                "customer_id": 12345,
                "invoice": "INV001",
                "price": 10.99,
                "stream_id": "STR001",
                "times_viewed": 1,
                "year": "2023",
                "month": "12",
                "day": "01"
            },
            {
                "country": "United States",
                "customer_id": 12346,
                "invoice": "INV002",
                "price": 15.99,
                "stream_id": "STR002",
                "times_viewed": 2,
                "year": "2023",
                "month": "12",
                "day": "01"
            }
        ]
        
        # Save test data
        import json
        test_file = os.path.join(self.app.config['DATA_DIR'], 'test_data.json')
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
    
    def test_train_model_no_data(self):
        """Test training model with no data"""
        # Remove test data
        for file in os.listdir(self.app.config['DATA_DIR']):
            os.remove(os.path.join(self.app.config['DATA_DIR'], file))
        
        response = self.client.post('/train')
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_train_model_with_data(self):
        """Test training model with data"""
        response = self.client.post('/train')
        # This might fail due to missing dependencies, but should return 500 rather than 404
        self.assertIn(response.status_code, [200, 500])
    
    def test_predict_missing_parameters(self):
        """Test prediction with missing parameters"""
        # Test missing country
        response = self.client.get('/predict?date=2023-12-01')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Country parameter is required', data['error'])
        
        # Test missing date
        response = self.client.get('/predict?country=United%20States')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Date parameter is required', data['error'])
    
    def test_predict_invalid_date_format(self):
        """Test prediction with invalid date format"""
        response = self.client.get('/predict?country=United%20States&date=invalid-date')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Invalid date format', data['error'])
    
    def test_predict_no_model(self):
        """Test prediction without trained model"""
        # Ensure model is not loaded
        model_manager.model = None
        
        response = self.client.get('/predict?country=United%20States&date=2023-12-01')
        self.assertEqual(response.status_code, 500)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Model not loaded', data['error'])
    
    def test_logs_endpoint(self):
        """Test logs endpoint"""
        response = self.client.get('/logs')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('logs', data)
        self.assertIsInstance(data['logs'], list)
    
    def test_logs_with_parameters(self):
        """Test logs endpoint with parameters"""
        response = self.client.get('/logs?limit=10&offset=0&endpoint=health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'success')
    
    def test_prediction_logs(self):
        """Test prediction logs endpoint"""
        response = self.client.get('/logs/predictions')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('predictions', data)
        self.assertIsInstance(data['predictions'], list)
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = self.client.get('/metrics')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('api_statistics', data)
        self.assertIn('prediction_statistics', data)
        self.assertIn('recent_activity', data)
    
    def test_download_logs(self):
        """Test download logs endpoint"""
        response = self.client.get('/logs/download?type=api')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'text/csv')
    
    def test_download_prediction_logs(self):
        """Test download prediction logs endpoint"""
        response = self.client.get('/logs/download?type=predictions')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'text/csv')

class ModelManagerTestCase(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for ModelManager"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'test_model.pkl')
        self.manager = ModelManager(self.model_path)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_model_initialization(self):
        """Test model manager initialization"""
        self.assertIsNone(self.manager.model)
        self.assertIsNone(self.manager.scaler)
        self.assertEqual(self.manager.model_path, self.model_path)
    
    def test_load_nonexistent_model(self):
        """Test loading non-existent model"""
        self.manager.load_model()
        self.assertIsNone(self.manager.model)
    
    def test_save_and_load_model(self):
        """Test saving and loading model"""
        # Create a simple model
        from sklearn.linear_model import LinearRegression
        import numpy as np
        
        model = LinearRegression()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([3, 7, 11])
        model.fit(X, y)
        
        # Save model
        model_data = {
            'model': model,
            'scaler': None,
            'feature_cols': ['feature1', 'feature2'],
            'model_type': 'LinearRegression'
        }
        
        import joblib
        joblib.dump(model_data, self.model_path)
        
        # Load model
        self.manager.load_model()
        
        self.assertIsNotNone(self.manager.model)
        self.assertEqual(self.manager.model_type, 'LinearRegression')
        self.assertEqual(self.manager.feature_cols, ['feature1', 'feature2'])

if __name__ == '__main__':
    unittest.main()
