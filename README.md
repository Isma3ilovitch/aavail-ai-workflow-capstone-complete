# aavail-ai-workflow-capstone-complete
IBM AI Enterprise Workflow Capstone

### Part 1: Data Investigation and Business Understanding
This part focuses on understanding the AAVAIL business scenario and conducting exploratory data analysis. We developed a comprehensive data ingestion pipeline to process JSON transaction data, clean and transform it, and aggregate revenue metrics by country and time periods. Through extensive visualization and analysis, we identified key patterns in the data, formulated testable hypotheses about revenue drivers, and established the foundation for predictive modeling. The deliverables include a Python data processing module, EDA visualizations, and a detailed report outlining the business opportunity and data insights.

### Part 2: Time-Series Modeling and Forecasting
In this part, we developed and compared multiple time-series forecasting approaches to predict AAVAIL's 30-day revenue. We implemented supervised learning models (Random Forest, Gradient Boosting, Linear Regression), traditional time-series models (ARIMA, SARIMA), and Facebook's Prophet. Through systematic feature engineering, hyperparameter tuning, and model evaluation, we identified the best-performing approach and trained a final model on the complete dataset. The deliverables include a comprehensive modeling framework, performance comparisons, visualizations of model accuracy, and a detailed report documenting the modeling process and findings.

### Part 3: Model Deployment and Production Analysis
The final part focuses on deploying the forecasting model as a production-ready API and implementing monitoring systems. We developed a Flask API with endpoints for model training, revenue prediction, and system monitoring, containerized the application using Docker, and implemented comprehensive unit tests. Additionally, we created a post-production analysis framework to monitor model performance, detect drift, and analyze business impact. The solution includes automated logging, performance metrics collection, and a reporting system that provides ongoing insights into model accuracy and business value.
