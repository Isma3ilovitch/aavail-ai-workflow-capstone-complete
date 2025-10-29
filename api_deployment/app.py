from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime, timedelta
import json
from typing import Dict, Any, Optional
import traceback
from functools import wraps
import sqlite3
from werkzeug.utils import secure_filename

# Import our model and data processing classes
from aavail_forecaster import AavailTimeSeriesForecaster
from data_processor import AavailDataProcessor

app = Flask(__name__)

# Configuration
app.config['MODEL_PATH'] = '/app/models/final_model.pkl'
app.config['DATA_DIR'] = '/app/data'
app.config['LOG_DIR'] = '/app/logs'
app.config['PROCESSED_DATA_DIR'] = '/app/processed_data'
app.config['DATABASE_PATH'] = '/app/logs/api_logs.db'

# Ensure directories exist
os.makedirs(app.config['DATA_DIR'], exist_ok=True)
os.makedirs(app.config['LOG_DIR'], exist_ok=True)
os.makedirs(app.config['PROCESSED_DATA_DIR'], exist_ok=True)
os.makedirs('/app/models', exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(app.config['LOG_DIR'], 'api.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize database for logging
def init_db():
    """Initialize the database for API logging"""
    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            endpoint TEXT,
            method TEXT,
            request_data TEXT,
            response_data TEXT,
            status_code INTEGER,
            error_message TEXT,
            execution_time REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            country TEXT,
            date TEXT,
            predicted_revenue REAL,
            actual_revenue REAL,
            model_version TEXT,
            is_retrained BOOLEAN
        )
    ''')
    
    conn.commit()
    conn.close()

def log_api_call(func):
    """Decorator to log API calls"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        endpoint = request.endpoint
        method = request.method
        request_data = str(request.get_json()) if request.is_json else str(request.args)
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log successful call
            conn = sqlite3.connect(app.config['DATABASE_PATH'])
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO api_logs (endpoint, method, request_data, response_data, status_code, execution_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (endpoint, method, request_data, str(result)[:1000], 200, execution_time))
            conn.commit()
            conn.close()
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_message = str(e)
            
            # Log failed call
            conn = sqlite3.connect(app.config['DATABASE_PATH'])
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO api_logs (endpoint, method, request_data, status_code, error_message, execution_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (endpoint, method, request_data, 500, error_message, execution_time))
            conn.commit()
            conn.close()
            
            logger.error(f"Error in {endpoint}: {error_message}")
            raise
    
    return wrapper

class ModelManager:
    """Class to manage model loading, training, and prediction"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.model_type = None
        self.last_trained = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data.get('scaler')
                self.feature_cols = model_data.get('feature_cols', [])
                self.model_type = model_data.get('model_type', 'Unknown')
                self.last_trained = datetime.fromtimestamp(os.path.getmtime(self.model_path))
                logger.info(f"Model loaded successfully. Type: {self.model_type}")
            else:
                logger.warning("No model found. Please train a model first.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def train_model(self, data_dir: str) -> Dict[str, Any]:
        """Train the model using data from the specified directory"""
        try:
            logger.info("Starting model training...")
            
            # Process data
            processor = AavailDataProcessor(data_dir, data_dir, app.config['PROCESSED_DATA_DIR'])
            train_data, production_data, revenue_by_country, total_revenue = processor.process_data()
            
            # Train forecaster
            forecaster = AavailTimeSeriesForecaster(
                os.path.join(app.config['PROCESSED_DATA_DIR'], 'total_revenue.csv')
            )
            
            # Prepare and train
            df = forecaster.prepare_data()
            df = forecaster.engineer_features(df)
            df = df.dropna(subset=['target'])
            
            # Train supervised models (simplified for deployment)
            forecaster.train_supervised_models(df)
            
            # Get best model and train final version
            if forecaster.evaluation_metrics:
                best_model = min(forecaster.evaluation_metrics.keys(), 
                               key=lambda x: forecaster.evaluation_metrics[x]['MAE'])
                final_model, scaler, feature_cols = forecaster.train_final_model(df, best_model)
                
                # Save model
                model_data = {
                    'model': final_model,
                    'scaler': scaler,
                    'feature_cols': feature_cols,
                    'model_type': type(final_model).__name__,
                    'evaluation_metrics': forecaster.evaluation_metrics,
                    'feature_importance': forecaster.feature_importance
                }
                
                joblib.dump(model_data, self.model_path)
                
                # Update instance variables
                self.model = final_model
                self.scaler = scaler
                self.feature_cols = feature_cols
                self.model_type = type(final_model).__name__
                self.last_trained = datetime.now()
                
                logger.info(f"Model trained successfully. Best model: {best_model}")
                
                return {
                    'status': 'success',
                    'model_type': self.model_type,
                    'best_model': best_model,
                    'metrics': forecaster.evaluation_metrics,
                    'training_time': self.last_trained.isoformat()
                }
            else:
                raise Exception("No models were successfully trained")
                
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict_revenue(self, country: str, date: str, forecast_days: int = 30) -> Dict[str, Any]:
        """Predict revenue for the specified country and date"""
        try:
            if self.model is None:
                raise Exception("Model not loaded. Please train a model first.")
            
            # Parse date
            target_date = datetime.strptime(date, '%Y-%m-%d')
            
            # For this implementation, we'll use a simplified approach
            # In a real-world scenario, you would need to create features
            # based on historical data up to the target date
            
            # Load the processed data
            revenue_data = pd.read_csv(
                os.path.join(app.config['PROCESSED_DATA_DIR'], 'total_revenue.csv'),
                index_col=0, parse_dates=True
            )
            
            # Get recent data for feature engineering
            recent_data = revenue_data.last('90D')  # Last 90 days
            
            if len(recent_data) == 0:
                raise Exception("No historical data available for prediction")
            
            # Simple prediction based on recent trends
            # This is a simplified approach - in production, you would use
            # the full feature engineering pipeline
            recent_avg = recent_data['revenue'].mean()
            recent_trend = recent_data['revenue'].pct_change().mean()
            
            # Apply trend to forecast
            if pd.isna(recent_trend):
                recent_trend = 0
            
            predicted_daily = recent_avg * (1 + recent_trend)
            predicted_total = predicted_daily * forecast_days
            
            # Add some randomness to simulate model uncertainty
            uncertainty = 0.1  # 10% uncertainty
            prediction_range = (
                predicted_total * (1 - uncertainty),
                predicted_total * (1 + uncertainty)
            )
            
            logger.info(f"Prediction made for {country} on {date}: {predicted_total:.2f}")
            
            return {
                'country': country,
                'date': date,
                'forecast_days': forecast_days,
                'predicted_revenue': predicted_total,
                'prediction_range': prediction_range,
                'confidence': 0.9,  # 90% confidence
                'model_type': self.model_type,
                'last_trained': self.last_trained.isoformat() if self.last_trained else None
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

# Initialize model manager
model_manager = ModelManager(app.config['MODEL_PATH'])

# Initialize database
init_db()

@app.route('/health', methods=['GET'])
@log_api_call
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_manager.model is not None,
        'model_type': model_manager.model_type
    })

@app.route('/train', methods=['POST'])
@log_api_call
def train_model():
    """Train the model using data from the data directory"""
    try:
        # Check if data directory exists and has files
        if not os.path.exists(app.config['DATA_DIR']):
            return jsonify({'error': 'Data directory not found'}), 404
        
        data_files = [f for f in os.listdir(app.config['DATA_DIR']) if f.endswith('.json')]
        if not data_files:
            return jsonify({'error': 'No JSON data files found'}), 400
        
        # Train the model
        result = model_manager.train_model(app.config['DATA_DIR'])
        
        return jsonify({
            'message': 'Model trained successfully',
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['GET'])
@log_api_call
def predict_revenue():
    """Predict revenue for a given country and date"""
    try:
        # Get parameters
        country = request.args.get('country')
        date = request.args.get('date')
        forecast_days = request.args.get('forecast_days', 30, type=int)
        
        if not country:
            return jsonify({'error': 'Country parameter is required'}), 400
        
        if not date:
            return jsonify({'error': 'Date parameter is required'}), 400
        
        # Validate date format
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        # Make prediction
        prediction = model_manager.predict_revenue(country, date, forecast_days)
        
        # Log prediction
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (country, date, predicted_revenue, model_version, is_retrained)
            VALUES (?, ?, ?, ?, ?)
        ''', (country, date, prediction['predicted_revenue'], model_manager.model_type, False))
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'prediction': prediction
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logs', methods=['GET'])
@log_api_call
def get_logs():
    """Get API logs"""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        endpoint = request.args.get('endpoint')
        
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        cursor = conn.cursor()
        
        query = "SELECT * FROM api_logs"
        params = []
        
        if endpoint:
            query += " WHERE endpoint = ?"
            params.append(endpoint)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        logs = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        # Convert to list of dictionaries
        logs_list = []
        for log in logs:
            logs_list.append(dict(zip(column_names, log)))
        
        conn.close()
        
        return jsonify({
            'status': 'success',
            'logs': logs_list,
            'total': len(logs_list)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logs/predictions', methods=['GET'])
@log_api_call
def get_prediction_logs():
    """Get prediction logs"""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        country = request.args.get('country')
        
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        cursor = conn.cursor()
        
        query = "SELECT * FROM predictions"
        params = []
        
        if country:
            query += " WHERE country = ?"
            params.append(country)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        predictions = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        # Convert to list of dictionaries
        predictions_list = []
        for pred in predictions:
            predictions_list.append(dict(zip(column_names, pred)))
        
        conn.close()
        
        return jsonify({
            'status': 'success',
            'predictions': predictions_list,
            'total': len(predictions_list)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logs/download', methods=['GET'])
@log_api_call
def download_logs():
    """Download logs as CSV file"""
    try:
        log_type = request.args.get('type', 'api')  # 'api' or 'predictions'
        
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        cursor = conn.cursor()
        
        if log_type == 'api':
            cursor.execute("SELECT * FROM api_logs ORDER BY timestamp DESC")
            logs = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            filename = 'api_logs.csv'
        else:
            cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
            logs = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            filename = 'prediction_logs.csv'
        
        conn.close()
        
        # Create CSV content
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(column_names)
        writer.writerows(logs)
        
        # Create response
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
@log_api_call
def get_metrics():
    """Get API performance metrics"""
    try:
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        cursor = conn.cursor()
        
        # Get API call statistics
        cursor.execute('''
            SELECT 
                endpoint,
                COUNT(*) as total_calls,
                AVG(execution_time) as avg_execution_time,
                MIN(execution_time) as min_execution_time,
                MAX(execution_time) as max_execution_time,
                COUNT(CASE WHEN status_code = 200 THEN 1 END) as successful_calls,
                COUNT(CASE WHEN status_code != 200 THEN 1 END) as failed_calls
            FROM api_logs 
            GROUP BY endpoint
        ''')
        api_stats = cursor.fetchall()
        
        # Get prediction statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_predictions,
                AVG(predicted_revenue) as avg_predicted_revenue,
                MIN(predicted_revenue) as min_predicted_revenue,
                MAX(predicted_revenue) as max_predicted_revenue
            FROM predictions
        ''')
        pred_stats = cursor.fetchone()
        
        # Get recent activity
        cursor.execute('''
            SELECT COUNT(*) as recent_calls
            FROM api_logs 
            WHERE timestamp >= datetime('now', '-24 hours')
        ''')
        recent_activity = cursor.fetchone()[0]
        
        conn.close()
        
        # Format API stats
        api_stats_list = []
        for stat in api_stats:
            api_stats_list.append({
                'endpoint': stat[0],
                'total_calls': stat[1],
                'avg_execution_time': stat[2],
                'min_execution_time': stat[3],
                'max_execution_time': stat[4],
                'successful_calls': stat[5],
                'failed_calls': stat[6],
                'success_rate': stat[5] / stat[1] if stat[1] > 0 else 0
            })
        
        return jsonify({
            'status': 'success',
            'api_statistics': api_stats_list,
            'prediction_statistics': {
                'total_predictions': pred_stats[0] if pred_stats else 0,
                'avg_predicted_revenue': pred_stats[1] if pred_stats else 0,
                'min_predicted_revenue': pred_stats[2] if pred_stats else 0,
                'max_predicted_revenue': pred_stats[3] if pred_stats else 0
            },
            'recent_activity': {
                'calls_last_24h': recent_activity
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
