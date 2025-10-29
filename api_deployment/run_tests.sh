#!/bin/bash


echo "Starting AAVAIL Capstone Project Tests..."

# Build and run Docker containers
echo "Building Docker containers..."
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Health check
echo "Performing health check..."
curl -f http://localhost:5000/health || exit 1

# Test model training
echo "Testing model training..."
curl -X POST http://localhost:5000/train

# Test prediction
echo "Testing prediction..."
curl -f "http://localhost:5000/predict?country=United%20States&date=2023-12-01"

# Test logs endpoint
echo "Testing logs endpoint..."
curl -f "http://localhost:5000/logs?limit=10"

# Test metrics endpoint
echo "Testing metrics endpoint..."
curl -f "http://localhost:5000/metrics"

# Run unit tests
echo "Running unit tests..."
python -m pytest test_app.py -v --cov=app --cov-report=html

# Run post-production analysis
echo "Running post-production analysis..."
python post_production_analysis.py

# Generate final report
echo "Generating final report..."
python final_report.py

echo "All tests completed successfully!"
echo "Check the following files for results:"
echo "- test_results.html (coverage report)"
echo "- analysis_results/ (post-production analysis)"
echo "- final_report.txt (comprehensive report)"
