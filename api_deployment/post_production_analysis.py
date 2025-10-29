import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class PostProductionAnalyzer:
    """
    Class for analyzing model performance in production
    """
    
    def __init__(self, database_path: str, output_dir: str):
        """
        Initialize the analyzer
        
        Args:
            database_path: Path to the SQLite database
            output_dir: Directory to save analysis results
        """
        self.database_path = database_path
        self.output_dir = output_dir
        self.conn = sqlite3.connect(database_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load data from the database
        
        Returns:
            Dictionary containing DataFrames for different data types
        """
        data = {}
        
        # Load API logs
        data['api_logs'] = pd.read_sql_query("SELECT * FROM api_logs", self.conn)
        data['api_logs']['timestamp'] = pd.to_datetime(data['api_logs']['timestamp'])
        
        # Load prediction logs
        data['predictions'] = pd.read_sql_query("SELECT * FROM predictions", self.conn)
        data['predictions']['timestamp'] = pd.to_datetime(data['predictions']['timestamp'])
        data['predictions']['date'] = pd.to_datetime(data['predictions']['date'])
        
        return data
    
    def analyze_api_performance(self, api_logs: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze API performance metrics
        
        Args:
            api_logs: DataFrame containing API logs
            
        Returns:
            Dictionary containing performance analysis results
        """
        analysis = {}
        
        # Overall statistics
        analysis['total_calls'] = len(api_logs)
        analysis['successful_calls'] = len(api_logs[api_logs['status_code'] == 200])
        analysis['failed_calls'] = len(api_logs[api_logs['status_code'] != 200])
        analysis['success_rate'] = analysis['successful_calls'] / analysis['total_calls'] if analysis['total_calls'] > 0 else 0
        
        # Execution time analysis
        execution_times = api_logs[api_logs['status_code'] == 200]['execution_time']
        analysis['avg_execution_time'] = execution_times.mean()
        analysis['median_execution_time'] = execution_times.median()
        analysis['min_execution_time'] = execution_times.min()
        analysis['max_execution_time'] = execution_times.max()
        analysis['std_execution_time'] = execution_times.std()
        
        # Endpoint-wise analysis
        endpoint_stats = api_logs.groupby('endpoint').agg({
            'execution_time': ['count', 'mean', 'median', 'std'],
            'status_code': lambda x: (x == 200).sum()
        }).round(2)
        
        analysis['endpoint_statistics'] = endpoint_stats.to_dict()
        
        # Time-based analysis
        api_logs['hour'] = api_logs['timestamp'].dt.hour
        api_logs['day_of_week'] = api_logs['timestamp'].dt.dayofweek
        
        hourly_stats = api_logs.groupby('hour').size()
        daily_stats = api_logs.groupby('day_of_week').size()
        
        analysis['hourly_distribution'] = hourly_stats.to_dict()
        analysis['daily_distribution'] = daily_stats.to_dict()
        
        # Error analysis
        errors = api_logs[api_logs['status_code'] != 200]
        if not errors.empty:
            error_analysis = errors.groupby('endpoint').size().sort_values(ascending=False)
            analysis['error_analysis'] = error_analysis.to_dict()
        else:
            analysis['error_analysis'] = {}
        
        return analysis
    
    def analyze_prediction_performance(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze prediction performance
        
        Args:
            predictions: DataFrame containing prediction logs
            
        Returns:
            Dictionary containing prediction analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis['total_predictions'] = len(predictions)
        analysis['unique_countries'] = predictions['country'].nunique()
        analysis['date_range'] = {
            'start': predictions['date'].min().strftime('%Y-%m-%d'),
            'end': predictions['date'].max().strftime('%Y-%m-%d')
        }
        
        # Revenue statistics
        analysis['revenue_stats'] = {
            'mean': predictions['predicted_revenue'].mean(),
            'median': predictions['predicted_revenue'].median(),
            'std': predictions['predicted_revenue'].std(),
            'min': predictions['predicted_revenue'].min(),
            'max': predictions['predicted_revenue'].max()
        }
        
        # Country-wise analysis
        country_stats = predictions.groupby('country').agg({
            'predicted_revenue': ['count', 'mean', 'std'],
            'timestamp': ['min', 'max']
        }).round(2)
        
        analysis['country_statistics'] = country_stats.to_dict()
        
        # Time-based analysis
        predictions['month'] = predictions['date'].dt.to_period('M')
        monthly_stats = predictions.groupby('month')['predicted_revenue'].agg(['count', 'mean', 'std'])
        analysis['monthly_trends'] = monthly_stats.to_dict()
        
        # Model version analysis
        if 'model_version' in predictions.columns:
            version_stats = predictions.groupby('model_version').size()
            analysis['model_version_distribution'] = version_stats.to_dict()
        
        return analysis
    
    def analyze_model_drift(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze potential model drift
        
        Args:
            predictions: DataFrame containing prediction logs
            
        Returns:
            Dictionary containing drift analysis results
        """
        analysis = {}
        
        if len(predictions) < 2:
            analysis['insufficient_data'] = True
            return analysis
        
        # Sort by date
        predictions = predictions.sort_values('date')
        
        # Split data into time periods
        mid_point = len(predictions) // 2
        first_half = predictions.iloc[:mid_point]
        second_half = predictions.iloc[mid_point:]
        
        # Compare revenue distributions
        first_mean = first_half['predicted_revenue'].mean()
        second_mean = second_half['predicted_revenue'].mean()
        
        analysis['mean_comparison'] = {
            'first_half_mean': first_mean,
            'second_half_mean': second_mean,
            'difference': second_mean - first_mean,
            'percentage_change': ((second_mean - first_mean) / first_mean * 100) if first_mean != 0 else 0
        }
        
        # Statistical test for distribution difference
        from scipy import stats
        try:
            _, p_value = stats.ttest_ind(
                first_half['predicted_revenue'],
                second_half['predicted_revenue']
            )
            analysis['distribution_test'] = {
                'p_value': p_value,
                'significant_drift': p_value < 0.05
            }
        except:
            analysis['distribution_test'] = {
                'p_value': None,
                'significant_drift': False
            }
        
        # Trend analysis
        predictions['prediction_date'] = predictions['timestamp'].dt.date
        daily_predictions = predictions.groupby('prediction_date')['predicted_revenue'].mean()
        
        if len(daily_predictions) > 1:
            # Simple linear trend
            x = np.arange(len(daily_predictions))
            y = daily_predictions.values
            
            slope, intercept = np.polyfit(x, y, 1)
            analysis['trend_analysis'] = {
                'slope': slope,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                'trend_strength': abs(slope)
            }
        else:
            analysis['trend_analysis'] = {
                'slope': 0,
                'trend_direction': 'stable',
                'trend_strength': 0
            }
        
        return analysis
    
    def create_visualizations(self, data: Dict[str, pd.DataFrame]):
        """
        Create visualization plots
        
        Args:
            data: Dictionary containing DataFrames
        """
        api_logs = data['api_logs']
        predictions = data['predictions']
        
        # 1. API Performance Over Time
        plt.figure(figsize=(15, 10))
        
        # Success rate over time
        plt.subplot(2, 2, 1)
        daily_success = api_logs.set_index('timestamp').resample('D').apply(
            lambda x: (x['status_code'] == 200).sum() / len(x) * 100
        )
        daily_success.plot()
        plt.title('Daily Success Rate (%)')
        plt.xlabel('Date')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)
        
        # Execution time distribution
        plt.subplot(2, 2, 2)
        successful_calls = api_logs[api_logs['status_code'] == 200]
        plt.hist(successful_calls['execution_time'], bins=30, alpha=0.7)
        plt.title('Execution Time Distribution')
        plt.xlabel('Execution Time (seconds)')
        plt.ylabel('Frequency')
        
        # Hourly call distribution
        plt.subplot(2, 2, 3)
        hourly_calls = api_logs['timestamp'].dt.hour.value_counts().sort_index()
        hourly_calls.plot(kind='bar')
        plt.title('Hourly Call Distribution')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Calls')
        
        # Endpoint usage
        plt.subplot(2, 2, 4)
        endpoint_usage = api_logs['endpoint'].value_counts()
        plt.pie(endpoint_usage.values, labels=endpoint_usage.index, autopct='%1.1f%%')
        plt.title('Endpoint Usage Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'api_performance.png'))
        plt.close()
        
        # 2. Prediction Analysis
        if not predictions.empty:
            plt.figure(figsize=(15, 10))
            
            # Revenue distribution
            plt.subplot(2, 2, 1)
            plt.hist(predictions['predicted_revenue'], bins=30, alpha=0.7)
            plt.title('Predicted Revenue Distribution')
            plt.xlabel('Predicted Revenue')
            plt.ylabel('Frequency')
            
            # Country-wise predictions
            plt.subplot(2, 2, 2)
            country_revenue = predictions.groupby('country')['predicted_revenue'].sum().sort_values(ascending=False).head(10)
            country_revenue.plot(kind='bar')
            plt.title('Top 10 Countries by Predicted Revenue')
            plt.xlabel('Country')
            plt.ylabel('Total Predicted Revenue')
            plt.xticks(rotation=45)
            
            # Monthly trends
            plt.subplot(2, 2, 3)
            predictions['month'] = predictions['date'].dt.to_period('M')
            monthly_revenue = predictions.groupby('month')['predicted_revenue'].mean()
            monthly_revenue.plot(kind='line', marker='o')
            plt.title('Monthly Average Predicted Revenue')
            plt.xlabel('Month')
            plt.ylabel('Average Predicted Revenue')
            plt.xticks(rotation=45)
            
            # Prediction volume over time
            plt.subplot(2, 2, 4)
            daily_volume = predictions.groupby(predictions['timestamp'].dt.date).size()
            daily_volume.plot(kind='line')
            plt.title('Daily Prediction Volume')
            plt.xlabel('Date')
            plt.ylabel('Number of Predictions')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'prediction_analysis.png'))
            plt.close()
        
        # 3. Model Drift Analysis
        if not predictions.empty and len(predictions) > 10:
            plt.figure(figsize=(15, 8))
            
            # Predictions over time
            plt.subplot(1, 2, 1)
            predictions_sorted = predictions.sort_values('date')
            plt.scatter(predictions_sorted['date'], predictions_sorted['predicted_revenue'], alpha=0.6)
            
            # Add trend line
            x = np.arange(len(predictions_sorted))
            y = predictions_sorted['predicted_revenue'].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(predictions_sorted['date'], p(x), "r--", alpha=0.8)
            
            plt.title('Predicted Revenue Over Time')
            plt.xlabel('Date')
            plt.ylabel('Predicted Revenue')
            plt.xticks(rotation=45)
            
            # Rolling statistics
            plt.subplot(1, 2, 2)
            predictions_sorted = predictions.sort_values('timestamp')
            rolling_mean = predictions_sorted['predicted_revenue'].rolling(window=min(30, len(predictions_sorted)//3)).mean()
            rolling_std = predictions_sorted['predicted_revenue'].rolling(window=min(30, len(predictions_sorted)//3)).std()
            
            plt.plot(predictions_sorted['timestamp'], rolling_mean, label='Rolling Mean')
            plt.fill_between(predictions_sorted['timestamp'], 
                           rolling_mean - rolling_std, 
                           rolling_mean + rolling_std, 
                           alpha=0.3, label='±1 Std Dev')
            
            plt.title('Rolling Statistics')
            plt.xlabel('Timestamp')
            plt.ylabel('Predicted Revenue')
            plt.legend()
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'model_drift.png'))
            plt.close()
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive analysis report
        
        Args:
            analysis_results: Dictionary containing all analysis results
            
        Returns:
            Report as a string
        """
        report = []
        
        report.append("="*80)
        report.append("AAVAIL MODEL POST-PRODUCTION ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        api_perf = analysis_results.get('api_performance', {})
        pred_perf = analysis_results.get('prediction_performance', {})
        drift_analysis = analysis_results.get('model_drift', {})
        
        total_calls = api_perf.get('total_calls', 0)
        success_rate = api_perf.get('success_rate', 0) * 100
        total_predictions = pred_perf.get('total_predictions', 0)
        
        report.append(f"• Total API calls processed: {total_calls:,}")
        report.append(f"• API success rate: {success_rate:.1f}%")
        report.append(f"• Total predictions made: {total_predictions:,}")
        report.append(f"• Countries served: {pred_perf.get('unique_countries', 0)}")
        
        if drift_analysis.get('significant_drift', False):
            report.append("• ⚠️  WARNING: Significant model drift detected")
        else:
            report.append("• ✅ Model performance appears stable")
        
        report.append("")
        
        # API Performance Analysis
        report.append("API PERFORMANCE ANALYSIS")
        report.append("-" * 40)
        
        if api_perf:
            report.append(f"Total API Calls: {api_perf['total_calls']:,}")
            report.append(f"Successful Calls: {api_perf['successful_calls']:,}")
            report.append(f"Failed Calls: {api_perf['failed_calls']:,}")
            report.append(f"Success Rate: {api_perf['success_rate']*100:.2f}%")
            report.append("")
            report.append("Execution Time Statistics:")
            report.append(f"  • Average: {api_perf['avg_execution_time']:.3f} seconds")
            report.append(f"  • Median: {api_perf['median_execution_time']:.3f} seconds")
            report.append(f"  • Minimum: {api_perf['min_execution_time']:.3f} seconds")
            report.append(f"  • Maximum: {api_perf['max_execution_time']:.3f} seconds")
            report.append(f"  • Standard Deviation: {api_perf['std_execution_time']:.3f} seconds")
            
            if api_perf.get('error_analysis'):
                report.append("")
                report.append("Error Analysis:")
                for endpoint, error_count in api_perf['error_analysis'].items():
                    report.append(f"  • {endpoint}: {error_count} errors")
        else:
            report.append("No API performance data available.")
        
        report.append("")
        
        # Prediction Performance Analysis
        report.append("PREDICTION PERFORMANCE ANALYSIS")
        report.append("-" * 40)
        
        if pred_perf:
            report.append(f"Total Predictions: {pred_perf['total_predictions']:,}")
            report.append(f"Unique Countries: {pred_perf['unique_countries']}")
            report.append(f"Date Range: {pred_perf['date_range']['start']} to {pred_perf['date_range']['end']}")
            report.append("")
            report.append("Revenue Statistics:")
            rev_stats = pred_perf['revenue_stats']
            report.append(f"  • Mean: ${rev_stats['mean']:,.2f}")
            report.append(f"  • Median: ${rev_stats['median']:,.2f}")
            report.append(f"  • Standard Deviation: ${rev_stats['std']:,.2f}")
            report.append(f"  • Minimum: ${rev_stats['min']:,.2f}")
            report.append(f"  • Maximum: ${rev_stats['max']:,.2f}")
        else:
            report.append("No prediction performance data available.")
        
        report.append("")
        
        # Model Drift Analysis
        report.append("MODEL DRIFT ANALYSIS")
        report.append("-" * 40)
        
        if drift_analysis:
            if drift_analysis.get('insufficient_data'):
                report.append("Insufficient data for drift analysis.")
            else:
                mean_comp = drift_analysis.get('mean_comparison', {})
                report.append("Mean Comparison (First Half vs Second Half):")
                report.append(f"  • First Half Mean: ${mean_comp.get('first_half_mean', 0):,.2f}")
                report.append(f"  • Second Half Mean: ${mean_comp.get('second_half_mean', 0):,.2f}")
                report.append(f"  • Difference: ${mean_comp.get('difference', 0):,.2f}")
                report.append(f"  • Percentage Change: {mean_comp.get('percentage_change', 0):.2f}%")
                
                dist_test = drift_analysis.get('distribution_test', {})
                if dist_test.get('p_value') is not None:
                    report.append("")
                    report.append("Statistical Test Results:")
                    report.append(f"  • P-value: {dist_test['p_value']:.4f}")
                    report.append(f"  • Significant Drift: {'Yes' if dist_test['significant_drift'] else 'No'}")
                
                trend_analysis = drift_analysis.get('trend_analysis', {})
                report.append("")
                report.append("Trend Analysis:")
                report.append(f"  • Trend Direction: {trend_analysis.get('trend_direction', 'Unknown')}")
                report.append(f"  • Trend Strength: {trend_analysis.get('trend_strength', 0):.4f}")
        else:
            report.append("No drift analysis data available.")
        
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        # Generate recommendations based on analysis
        recommendations = []
        
        if api_perf.get('success_rate', 0) < 0.95:
            recommendations.append("• Investigate and address API failures to improve success rate above 95%")
        
        if api_perf.get('avg_execution_time', 0) > 1.0:
            recommendations.append("• Optimize API performance to reduce average execution time")
        
        if drift_analysis.get('significant_drift', False):
            recommendations.append("• Retrain model due to detected drift in predictions")
            recommendations.append("• Investigate external factors that may have caused the drift")
        
        if pred_perf.get('total_predictions', 0) < 100:
            recommendations.append("• Increase model usage to gather more performance data")
        
        if not recommendations:
            recommendations.append("• Continue monitoring model performance")
            recommendations.append("• Schedule regular retraining to maintain model accuracy")
        
        for rec in recommendations:
            report.append(rec)
        
        report.append("")
        
        # Next Steps
        report.append("NEXT STEPS")
        report.append("-" * 40)
        report.append("1. Implement automated retraining pipeline")
        report.append("2. Set up real-time monitoring dashboards")
        report.append("3. Establish alert thresholds for performance metrics")
        report.append("4. Conduct A/B testing for model improvements")
        report.append("5. Expand model to support additional countries")
        report.append("6. Implement feedback loop for continuous improvement")
        
        report.append("")
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        return '\n'.join(report)
    
    def run_analysis(self) -> Dict[str, Any]:
        """
        Run complete post-production analysis
        
        Returns:
            Dictionary containing all analysis results
        """
        print("Loading data...")
        data = self.load_data()
        
        print("Analyzing API performance...")
        api_performance = self.analyze_api_performance(data['api_logs'])
        
        print("Analyzing prediction performance...")
        prediction_performance = self.analyze_prediction_performance(data['predictions'])
        
        print("Analyzing model drift...")
        model_drift = self.analyze_model_drift(data['predictions'])
        
        print("Creating visualizations...")
        self.create_visualizations(data)
        
        # Compile results
        analysis_results = {
            'api_performance': api_performance,
            'prediction_performance': prediction_performance,
            'model_drift': model_drift,
            'data_summary': {
                'api_logs_count': len(data['api_logs']),
                'predictions_count': len(data['predictions']),
                'analysis_date': datetime.now().isoformat()
            }
        }
        
        # Generate report
        report = self.generate_report(analysis_results)
        
        # Save report
        report_path = os.path.join(self.output_dir, 'post_production_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save analysis results as JSON
        results_path = os.path.join(self.output_dir, 'analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"Analysis complete. Results saved to {self.output_dir}")
        print(f"Report saved to: {report_path}")
        print(f"Analysis results saved to: {results_path}")
        
        return analysis_results

def main():
    """Main function to run post-production analysis"""
    # Configuration
    database_path = '/app/logs/api_logs.db'
    output_dir = '/app/analysis_results'
    
    # Run analysis
    analyzer = PostProductionAnalyzer(database_path, output_dir)
    results = analyzer.run_analysis()
    
    return results

if __name__ == "__main__":
    main()
